"""Online RAG-ASR retrieval service: projector frames + hotword list."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rag_asr.cache import (
    load_hotword_pool_file,
    load_text_emb_cache,
    save_text_emb_cache,
    text_emb_cache_lock,
    text_emb_cache_path,
)
from rag_asr.dataset import tokenise_words
from rag_asr.hotwords import hotword_dedupe_key, normalize_hotword, normalize_hotwords
from rag_asr.model_layout import (
    DEFAULT_HOTWORD_ADAPTER_FILENAME,
    DEFAULT_HOTWORD_ADAPTER_SUBDIR,
    resolve_hotword_adapter,
)

logger = logging.getLogger(__name__)
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
_ENROLLMENT_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


@dataclass
class ServeConfig:
    base_model_path: str
    hotword_pool_file: Optional[str] = None
    hotword_pool_dir: str = "var/hotwords"
    seed_pool_file: Optional[str] = None
    default_user: str = "default"
    adapter_ckpt: Optional[str] = None
    adapter_subdir: str = DEFAULT_HOTWORD_ADAPTER_SUBDIR
    adapter_filename: str = DEFAULT_HOTWORD_ADAPTER_FILENAME
    embed_dim: int = 512
    adapter_hidden_dim: Optional[int] = 512
    default_top_k: int = 50
    cache_dir: Optional[str] = "_retrieve_cache"
    device: str = "cuda"
    num_mel_bins: int = 128
    batch_text: int = 512
    enrollment_store_dir: str = "var/enrollments"
    enrollment_cache_max_entries: int = 256
    enrollment_cache_device: str = "auto"
    enrollment_metadata_touch_interval_sec: float = 300.0


@dataclass
class InferResult:
    word_list: list[str]
    projector_out: np.ndarray  # (T', D_proj) float32
    projector_len: int
    enrollment_id: Optional[str] = None
    enrollment_projector_out: Optional[np.ndarray] = None
    enrollment_projector_len: int = 0
    enrollment_cache_hit: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "word_list": self.word_list,
            "projector_out": self.projector_out,
            "projector_len": self.projector_len,
            "enrollment_id": self.enrollment_id,
            "enrollment_projector_out": self.enrollment_projector_out,
            "enrollment_projector_len": self.enrollment_projector_len,
            "enrollment_cache_hit": self.enrollment_cache_hit,
        }


@dataclass
class _UserPool:
    user_id: str
    pool_file: Path
    words: list[str]
    embs_gpu: torch.Tensor
    lock: threading.RLock


@dataclass
class _EnrollmentCacheEntry:
    user_id: str
    enrollment_id: str
    tensor: torch.Tensor
    meta: dict[str, object]
    cache_device: str


class RAGASRRetriever:
    """Load dual-tower model once; run per-request audio → words + projector output."""

    def __init__(self, cfg: ServeConfig):
        from rag_asr.model_loader import load_towers

        self.cfg = cfg
        self._device = torch.device(cfg.device)
        self.default_user = self._resolve_user(cfg.default_user)
        self._pools_lock = threading.RLock()
        self._pools: dict[str, _UserPool] = {}
        self._enrollment_lock = threading.RLock()
        self._enrollment_cache: OrderedDict[
            tuple[str, str], _EnrollmentCacheEntry
        ] = OrderedDict()
        self.adapter_ckpt = str(
            resolve_hotword_adapter(
                cfg.base_model_path,
                cfg.adapter_ckpt,
                adapter_subdir=cfg.adapter_subdir,
                adapter_filename=cfg.adapter_filename,
            )
        )

        _, self.audio_tower, self.text_tower, self.tokenizer = load_towers(
            cfg.base_model_path,
            self.adapter_ckpt,
            embed_dim=cfg.embed_dim,
            adapter_hidden_dim=cfg.adapter_hidden_dim,
            device=self._device,
        )
        self.projector_dim = int(self.audio_tower.projector_dim)
        default_pool = self._get_or_load_pool(None)

        from lhotse.features import WhisperFbank, WhisperFbankConfig

        self.fbank = WhisperFbank(WhisperFbankConfig(num_filters=cfg.num_mel_bins))
        logger.info(
            "RAGASRRetriever ready: default_user=%s pool=%d words, projector_dim=%d, device=%s",
            self.default_user,
            len(default_pool.words),
            self.projector_dim,
            self._device,
        )

    @staticmethod
    def _now() -> float:
        return time.time()

    def _resolve_user(self, user_id: Optional[str]) -> str:
        user = normalize_hotword(user_id or self.cfg.default_user)
        if not user:
            user = getattr(self, "default_user", self.cfg.default_user)
        if not _USER_ID_RE.fullmatch(user) or user in {".", ".."}:
            raise ValueError(
                "HOTWORD_POOL_ID must contain only letters, digits, dot, underscore, or hyphen"
            )
        return user

    def _resolve_enrollment_id(self, enrollment_id: Optional[str]) -> str:
        value = normalize_hotword(enrollment_id or "")
        if (
            not value
            or not _ENROLLMENT_ID_RE.fullmatch(value)
            or value in {".", ".."}
        ):
            raise ValueError(
                "ENROLLMENT_ID must contain only letters, digits, dot, underscore, or hyphen"
            )
        return value

    def _enrollment_paths(self, user_id: str, enrollment_id: str) -> tuple[Path, Path]:
        root = Path(self.cfg.enrollment_store_dir)
        return (
            root / user_id / f"{enrollment_id}.pt",
            root / user_id / f"{enrollment_id}.json",
        )

    @staticmethod
    def _audio_hash(wav: np.ndarray, sample_rate: int) -> str:
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        digest = hashlib.sha256()
        digest.update(str(int(sample_rate)).encode("ascii"))
        digest.update(audio.tobytes())
        return digest.hexdigest()

    def _enrollment_model_meta(self) -> dict[str, object]:
        return {
            "base_model_path": str(self.cfg.base_model_path),
            "adapter_ckpt": str(self.adapter_ckpt),
            "projector_dim": int(getattr(self, "projector_dim", self.cfg.embed_dim)),
            "embed_dim": int(self.cfg.embed_dim),
        }

    def _warn_on_enrollment_model_drift(
        self,
        *,
        user_id: str,
        enrollment_id: str,
        meta: dict[str, object],
    ) -> None:
        current = self._enrollment_model_meta()
        for key in ("base_model_path", "adapter_ckpt"):
            existing = meta.get(key)
            if existing and str(existing) != str(current[key]):
                logger.warning(
                    "enrollment %s/%s %s differs from current service: stored=%s current=%s",
                    user_id,
                    enrollment_id,
                    key,
                    existing,
                    current[key],
                )

    def _validate_enrollment_compatible(
        self,
        *,
        user_id: str,
        enrollment_id: str,
        tensor: torch.Tensor,
        meta: dict[str, object],
    ) -> None:
        current = self._enrollment_model_meta()
        for key in ("projector_dim", "embed_dim"):
            existing = meta.get(key)
            if existing is not None and int(existing) != int(current[key]):
                raise ValueError(
                    f"enrollment embedding incompatible: enrollment_scope_id={user_id} "
                    f"enrollment_id={enrollment_id} {key}={existing} "
                    f"current_{key}={current[key]}"
                )
        projector_dim = int(current["projector_dim"])
        if tensor.ndim != 2 or int(tensor.shape[1]) != projector_dim:
            raise ValueError(
                f"enrollment embedding incompatible: enrollment_scope_id={user_id} "
                f"enrollment_id={enrollment_id} shape={list(tensor.shape)} "
                f"current_projector_dim={projector_dim}"
            )
        self._warn_on_enrollment_model_drift(
            user_id=user_id,
            enrollment_id=enrollment_id,
            meta=meta,
        )

    def _should_touch_enrollment_meta(
        self,
        *,
        meta_path: Path,
        previous_last_used_at: object,
        now: float,
    ) -> bool:
        if not meta_path.exists():
            return True
        interval = max(float(self.cfg.enrollment_metadata_touch_interval_sec), 0.0)
        if interval == 0.0:
            return True
        try:
            previous = float(previous_last_used_at)
        except (TypeError, ValueError):
            return True
        return now - previous >= interval

    def _cache_device(self) -> torch.device:
        configured = (self.cfg.enrollment_cache_device or "auto").strip().lower()
        if configured == "auto":
            return self._device
        return torch.device(configured)

    def _cache_enrollment_locked(
        self,
        *,
        user_id: str,
        enrollment_id: str,
        tensor: torch.Tensor,
        meta: dict[str, object],
    ) -> _EnrollmentCacheEntry:
        target_device = self._cache_device()
        try:
            cached_tensor = tensor.to(target_device)
            cache_device = str(target_device)
        except RuntimeError:
            logger.warning(
                "failed to move enrollment %s/%s to %s; caching on CPU",
                user_id,
                enrollment_id,
                target_device,
            )
            cached_tensor = tensor.cpu()
            cache_device = "cpu"
        entry = _EnrollmentCacheEntry(
            user_id=user_id,
            enrollment_id=enrollment_id,
            tensor=cached_tensor,
            meta=dict(meta),
            cache_device=cache_device,
        )
        key = (user_id, enrollment_id)
        self._enrollment_cache.pop(key, None)
        self._enrollment_cache[key] = entry
        max_entries = max(int(self.cfg.enrollment_cache_max_entries), 0)
        while max_entries and len(self._enrollment_cache) > max_entries:
            self._enrollment_cache.popitem(last=False)
        if max_entries == 0:
            self._enrollment_cache.clear()
        return entry

    @staticmethod
    def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(path.parent),
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                json.dump(payload, tmp, ensure_ascii=False, sort_keys=True)
            os.replace(tmp_name, path)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _atomic_torch_save(path: Path, tensor: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=str(path.parent),
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                torch.save(tensor.detach().cpu(), tmp)
            os.replace(tmp_name, path)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass

    def _pool_file_for_user(self, user_id: str) -> Path:
        if user_id == self.default_user and self.cfg.hotword_pool_file:
            return Path(self.cfg.hotword_pool_file)
        return Path(self.cfg.hotword_pool_dir) / f"{user_id}.txt"

    def _load_words_for_user(self, user_id: str, pool_file: Path) -> list[str]:
        if pool_file.is_file():
            return load_hotword_pool_file(pool_file)
        if user_id == self.default_user and self.cfg.seed_pool_file:
            seed = Path(self.cfg.seed_pool_file)
            if seed.is_file():
                return load_hotword_pool_file(seed)
        return []

    def _get_or_load_pool(self, user_id: Optional[str]) -> _UserPool:
        resolved_user = self._resolve_user(user_id)
        with self._pools_lock:
            cached = self._pools.get(resolved_user)
            if cached is not None:
                return cached
            pool_file = self._pool_file_for_user(resolved_user)
            words = self._load_words_for_user(resolved_user, pool_file)
            embs = self._load_pool_embeddings(words, pool_file).to(self._device)
            pool = _UserPool(
                user_id=resolved_user,
                pool_file=pool_file,
                words=words,
                embs_gpu=embs,
                lock=threading.RLock(),
            )
            self._pools[resolved_user] = pool
        if not pool_file.exists():
            with pool.lock:
                self._persist_pool_locked(pool)
        return pool

    def _cache_path(self, pool_file: Path) -> Optional[Path]:
        cache_dir = self.cfg.cache_dir
        if cache_dir and cache_dir.lower() in {"none", "off"}:
            cache_dir = None
        return text_emb_cache_path(
            cache_dir,
            adapter_ckpt=self.adapter_ckpt,
            hotword_pool_file=str(pool_file),
        )

    def _encode_hotwords(self, words: list[str]) -> torch.Tensor:
        if not words:
            return torch.empty((0, self.cfg.embed_dim), dtype=torch.float32)
        logger.info("encoding hotword pool chunk (%d words)", len(words))
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(words), self.cfg.batch_text):
                chunk = words[i : i + self.cfg.batch_text]
                ids, mask = tokenise_words(chunk, self.tokenizer, device=self._device)
                parts.append(self.text_tower(ids, mask).cpu())
        return torch.cat(parts, dim=0)

    def _load_pool_embeddings(self, words: list[str], pool_file: Path) -> torch.Tensor:
        cache_path = self._cache_path(pool_file)

        if cache_path is not None and cache_path.exists():
            cached_words, embs = load_text_emb_cache(cache_path)
            if cached_words == words:
                return embs

        def _encode() -> torch.Tensor:
            logger.info("encoding hotword pool %s (%d words)…", pool_file, len(words))
            embs = self._encode_hotwords(words)
            if cache_path is not None:
                save_text_emb_cache(
                    cache_path,
                    words,
                    embs,
                    acquire_lock=False,
                    overwrite=True,
                )
            return embs

        if cache_path is None:
            return _encode()

        with text_emb_cache_lock(cache_path):
            if cache_path.exists():
                cached_words, embs = load_text_emb_cache(cache_path)
                if cached_words == words:
                    return embs
            embs = _encode()
        return embs

    def _persist_pool_locked(self, pool: _UserPool) -> None:
        path = pool.pool_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(path.parent),
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                for word in pool.words:
                    tmp.write(f"{word}\n")
            os.replace(tmp_name, path)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass

    def _save_pool_cache_locked(self, pool: _UserPool) -> None:
        cache_path = self._cache_path(pool.pool_file)
        if cache_path is None:
            return
        save_text_emb_cache(
            cache_path,
            pool.words,
            pool.embs_gpu.detach().cpu(),
            overwrite=True,
        )

    def _set_pool_locked(
        self,
        pool: _UserPool,
        words: list[str],
        embs: torch.Tensor,
        *,
        persist: bool = True,
    ) -> None:
        pool.words = words
        pool.embs_gpu = embs.to(self._device)
        if persist:
            self._persist_pool_locked(pool)
        self._save_pool_cache_locked(pool)

    def list_hotwords(
        self,
        *,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> dict[str, object]:
        """Return a page from the in-memory hotword pool."""

        pool = self._get_or_load_pool(user_id)
        query_norm = normalize_hotword(query).casefold() if query else ""
        offset = max(int(offset), 0)
        if limit is None:
            limit_value = None
        else:
            limit_value = max(int(limit), 0)

        with pool.lock:
            total_count = len(pool.words)
            if query_norm:
                matched = [
                    word
                    for word in pool.words
                    if query_norm in word.casefold()
                ]
            else:
                matched = list(pool.words)

        if limit_value is None:
            page = matched[offset:]
        else:
            page = matched[offset : offset + limit_value]
        return {
            "action": "list",
            "status": "ok",
            "message": f"returned {len(page)} hotwords",
            "hotword_pool_id": pool.user_id,
            "hotwords": page,
            "total_count": total_count,
            "matched_count": len(matched),
            "offset": offset,
            "limit": limit_value,
        }

    def add_hotwords(
        self,
        values: list[object],
        *,
        user_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Add hotwords, incrementally encode new entries, and persist the pool."""

        pool = self._get_or_load_pool(user_id)
        with pool.lock:
            existing_keys = {hotword_dedupe_key(word) for word in pool.words}
            batch = normalize_hotwords(values, existing_keys=existing_keys)
            if batch.words:
                new_embs = self._encode_hotwords(batch.words)
                current_embs = pool.embs_gpu.detach().cpu()
                combined_words = pool.words + batch.words
                combined_embs = torch.cat([current_embs, new_embs], dim=0)
                order = sorted(
                    range(len(combined_words)),
                    key=lambda i: hotword_dedupe_key(combined_words[i]),
                )
                words = [combined_words[i] for i in order]
                embs = combined_embs[order]
                self._set_pool_locked(pool, words, embs)
            total_count = len(pool.words)
        added = len(batch.words)
        return {
            "action": "add",
            "status": "ok",
            "message": f"added {added} hotwords",
            "hotword_pool_id": pool.user_id,
            "hotwords": batch.words,
            "total_count": total_count,
            "added": added,
            "skipped_duplicates": len(batch.duplicates),
            "invalid": batch.invalid,
            "duplicates": batch.duplicates,
        }

    def delete_hotwords(
        self,
        values: list[object],
        *,
        user_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Delete hotwords by canonical dedupe key and persist the pool."""

        pool = self._get_or_load_pool(user_id)
        batch = normalize_hotwords(values, sort=False)
        delete_keys = {hotword_dedupe_key(word) for word in batch.words}
        with pool.lock:
            existing = {
                hotword_dedupe_key(word): word
                for word in pool.words
            }
            missing = [
                word
                for word in batch.words
                if hotword_dedupe_key(word) not in existing
            ]
            keep_indices = [
                i
                for i, word in enumerate(pool.words)
                if hotword_dedupe_key(word) not in delete_keys
            ]
            deleted = len(pool.words) - len(keep_indices)
            if deleted:
                words = [pool.words[i] for i in keep_indices]
                if keep_indices:
                    index = torch.tensor(keep_indices, device=self._device)
                    embs = pool.embs_gpu.index_select(0, index).detach().cpu()
                else:
                    embs = torch.empty((0, self.cfg.embed_dim), dtype=torch.float32)
                self._set_pool_locked(pool, words, embs)
            total_count = len(pool.words)
        return {
            "action": "delete",
            "status": "ok",
            "message": f"deleted {deleted} hotwords",
            "hotword_pool_id": pool.user_id,
            "hotwords": batch.words,
            "total_count": total_count,
            "deleted": deleted,
            "missing": missing,
            "invalid": batch.invalid,
            "duplicates": batch.duplicates,
        }

    def clear_hotwords(self, *, user_id: Optional[str] = None) -> dict[str, object]:
        """Clear one hotword pool and persist the empty runtime pool."""

        pool = self._get_or_load_pool(user_id)
        with pool.lock:
            cleared = len(pool.words)
            embs = torch.empty((0, self.cfg.embed_dim), dtype=torch.float32)
            self._set_pool_locked(pool, [], embs)
        return {
            "action": "clear",
            "status": "ok",
            "message": f"cleared {cleared} hotwords",
            "hotword_pool_id": pool.user_id,
            "hotwords": [],
            "total_count": 0,
            "cleared": cleared,
        }

    def reload_hotwords(self, *, user_id: Optional[str] = None) -> dict[str, object]:
        """Reload the configured pool file, rebuild embeddings, and refresh cache."""

        pool = self._get_or_load_pool(user_id)
        with pool.lock:
            old_count = len(pool.words)
            words = self._load_words_for_user(pool.user_id, pool.pool_file)
            embs = self._load_pool_embeddings(words, pool.pool_file)
            self._set_pool_locked(pool, words, embs, persist=True)
            total_count = len(pool.words)
        return {
            "action": "reload",
            "status": "ok",
            "message": f"reloaded hotword pool: {old_count} -> {total_count}",
            "hotword_pool_id": pool.user_id,
            "hotwords": [],
            "total_count": total_count,
            "reloaded": total_count,
        }

    def _wav_to_features(
        self,
        wav: np.ndarray,
        sample_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        sr = int(sample_rate)
        if sr != 16000:
            import torchaudio

            audio_t = torch.from_numpy(wav).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
            wav = audio_t.squeeze(0).numpy()
            sr = 16000
        feat = self.fbank.extract(samples=wav, sampling_rate=sr)
        features = torch.from_numpy(feat).unsqueeze(0)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.long)
        return features, feat_lens

    def _wavs_to_features(
        self,
        wavs: list[np.ndarray],
        sample_rates: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(wavs) != len(sample_rates):
            raise ValueError(
                f"wavs/sample_rates length mismatch: {len(wavs)} vs {len(sample_rates)}"
            )
        if not wavs:
            raise ValueError("infer_many requires at least one waveform")

        feats_list: list[torch.Tensor] = []
        for wav, sample_rate in zip(wavs, sample_rates):
            features, _ = self._wav_to_features(wav, sample_rate)
            feats_list.append(features.squeeze(0))

        feat_lens = torch.tensor([feat.shape[0] for feat in feats_list], dtype=torch.long)
        max_t = int(feat_lens.max().item())
        n_feats = int(feats_list[0].shape[1])
        features = torch.zeros(len(feats_list), max_t, n_feats, dtype=torch.float32)
        for i, feat in enumerate(feats_list):
            features[i, : feat.shape[0], :] = feat
        return features, feat_lens

    def _project_audio(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        features, feat_lens = self._wav_to_features(wav, sample_rate)
        features = features.to(self._device)
        feat_lens = feat_lens.to(self._device)
        with torch.no_grad():
            _, proj, proj_lens = self.audio_tower.forward_with_projector(
                features,
                feat_lens,
            )
        plen = int(proj_lens[0].item())
        return proj[0, :plen, :].detach().cpu().numpy().astype(np.float32)

    def upsert_enrollment(
        self,
        wav: np.ndarray,
        *,
        sample_rate: int = 16000,
        enrollment_id: Optional[str],
        enrollment_scope_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Persist target-speaker projector frames; raw audio is never stored."""
        resolved_user = self._resolve_user(enrollment_scope_id)
        resolved_enrollment = self._resolve_enrollment_id(enrollment_id)
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        duration_sec = float(audio.shape[0]) / float(sample_rate or 16000)
        tensor_path, meta_path = self._enrollment_paths(
            resolved_user,
            resolved_enrollment,
        )
        now = self._now()
        audio_hash = self._audio_hash(audio, sample_rate)
        previous_meta: dict[str, object] = {}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    previous_meta = loaded
            except Exception:
                logger.warning("failed to read existing enrollment metadata: %s", meta_path)
        if (
            previous_meta.get("audio_hash") == audio_hash
            and int(previous_meta.get("sample_rate", 0)) == int(sample_rate)
            and tensor_path.exists()
        ):
            try:
                with self._enrollment_lock:
                    entry, hit = self._load_enrollment_locked(
                        user_id=resolved_user,
                        enrollment_id=resolved_enrollment,
                    )
            except ValueError as exc:
                logger.warning(
                    "existing enrollment %s/%s is incompatible; recomputing: %s",
                    resolved_user,
                    resolved_enrollment,
                    exc,
                )
            else:
                if entry is not None:
                    return {
                        "action": "upsert_enrollment",
                        "status": "ok",
                        "message": "enrollment embedding unchanged",
                        "enrollment_scope_id": resolved_user,
                        "enrollment_id": resolved_enrollment,
                        "audio_hash": audio_hash,
                        "duration_sec": previous_meta.get("duration_sec", duration_sec),
                        "shape": list(entry.tensor.shape),
                        "tensor_path": str(tensor_path),
                        "cache_device": entry.cache_device,
                        "cache_hit": hit,
                        "skipped": True,
                        "overwritten": False,
                    }

        frames = self._project_audio(audio, sample_rate)
        tensor = torch.from_numpy(frames.astype(np.float32, copy=False))
        now = self._now()
        overwritten = tensor_path.exists() or meta_path.exists()
        meta: dict[str, object] = {
            "enrollment_scope_id": resolved_user,
            "enrollment_id": resolved_enrollment,
            "audio_hash": audio_hash,
            "sample_rate": int(sample_rate),
            "duration_sec": duration_sec,
            "shape": list(tensor.shape),
            "dtype": "float32",
            "tensor_path": str(tensor_path),
            "created_at": now,
            "updated_at": now,
            "last_used_at": now,
            **self._enrollment_model_meta(),
        }
        if previous_meta.get("created_at"):
            meta["created_at"] = previous_meta["created_at"]
        self._atomic_torch_save(tensor_path, tensor)
        self._atomic_json_write(meta_path, meta)
        with self._enrollment_lock:
            entry = self._cache_enrollment_locked(
                user_id=resolved_user,
                enrollment_id=resolved_enrollment,
                tensor=tensor,
                meta=meta,
            )
        return {
            "action": "upsert_enrollment",
            "status": "ok",
            "message": "enrollment embedding stored",
            "enrollment_scope_id": resolved_user,
            "enrollment_id": resolved_enrollment,
            "audio_hash": meta["audio_hash"],
            "duration_sec": duration_sec,
            "shape": list(tensor.shape),
            "tensor_path": str(tensor_path),
            "cache_device": entry.cache_device,
            "skipped": False,
            "overwritten": overwritten,
        }

    def _load_enrollment_locked(
        self,
        *,
        user_id: str,
        enrollment_id: str,
    ) -> tuple[_EnrollmentCacheEntry | None, bool]:
        key = (user_id, enrollment_id)
        cached = self._enrollment_cache.get(key)
        now = self._now()
        if cached is not None:
            previous_last_used_at = cached.meta.get("last_used_at")
            cached.meta["last_used_at"] = now
            self._enrollment_cache.move_to_end(key)
            _, meta_path = self._enrollment_paths(user_id, enrollment_id)
            if self._should_touch_enrollment_meta(
                meta_path=meta_path,
                previous_last_used_at=previous_last_used_at,
                now=now,
            ):
                self._atomic_json_write(meta_path, cached.meta)
            return cached, True

        tensor_path, meta_path = self._enrollment_paths(user_id, enrollment_id)
        if not tensor_path.exists():
            return None, False
        tensor = torch.load(tensor_path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32)
        meta: dict[str, object] = {
            "enrollment_scope_id": user_id,
            "enrollment_id": enrollment_id,
            "shape": list(tensor.shape),
            "dtype": "float32",
            "tensor_path": str(tensor_path),
            "last_used_at": now,
        }
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta.update(loaded)
            except Exception:
                logger.warning("failed to read enrollment metadata: %s", meta_path)
        self._validate_enrollment_compatible(
            user_id=user_id,
            enrollment_id=enrollment_id,
            tensor=tensor,
            meta=meta,
        )
        previous_last_used_at = meta.get("last_used_at")
        meta["last_used_at"] = now
        if self._should_touch_enrollment_meta(
            meta_path=meta_path,
            previous_last_used_at=previous_last_used_at,
            now=now,
        ):
            self._atomic_json_write(meta_path, meta)
        entry = self._cache_enrollment_locked(
            user_id=user_id,
            enrollment_id=enrollment_id,
            tensor=tensor,
            meta=meta,
        )
        return entry, False

    def get_enrollment(
        self,
        *,
        enrollment_id: Optional[str],
        enrollment_scope_id: Optional[str] = None,
    ) -> dict[str, object]:
        resolved_user = self._resolve_user(enrollment_scope_id)
        resolved_enrollment = self._resolve_enrollment_id(enrollment_id)
        try:
            with self._enrollment_lock:
                entry, hit = self._load_enrollment_locked(
                    user_id=resolved_user,
                    enrollment_id=resolved_enrollment,
                )
        except ValueError as exc:
            return {
                "action": "get_enrollment",
                "status": "incompatible",
                "message": str(exc),
                "enrollment_scope_id": resolved_user,
                "enrollment_id": resolved_enrollment,
            }
        if entry is None:
            return {
                "action": "get_enrollment",
                "status": "not_found",
                "message": "enrollment embedding not found",
                "enrollment_scope_id": resolved_user,
                "enrollment_id": resolved_enrollment,
            }
        return {
            "action": "get_enrollment",
            "status": "ok",
            "message": "enrollment embedding found",
            "enrollment_scope_id": resolved_user,
            "enrollment_id": resolved_enrollment,
            "shape": list(entry.tensor.shape),
            "cache_hit": hit,
            "cache_device": entry.cache_device,
            **{
                k: v
                for k, v in entry.meta.items()
                if k
                in {
                    "audio_hash",
                    "sample_rate",
                    "duration_sec",
                    "tensor_path",
                    "created_at",
                    "updated_at",
                    "last_used_at",
                    "base_model_path",
                    "adapter_ckpt",
                    "projector_dim",
                    "embed_dim",
                }
            },
        }

    def delete_enrollment(
        self,
        *,
        enrollment_id: Optional[str],
        enrollment_scope_id: Optional[str] = None,
    ) -> dict[str, object]:
        resolved_user = self._resolve_user(enrollment_scope_id)
        resolved_enrollment = self._resolve_enrollment_id(enrollment_id)
        tensor_path, meta_path = self._enrollment_paths(
            resolved_user,
            resolved_enrollment,
        )
        with self._enrollment_lock:
            self._enrollment_cache.pop((resolved_user, resolved_enrollment), None)
        deleted = False
        for path in (tensor_path, meta_path):
            try:
                path.unlink()
                deleted = True
            except FileNotFoundError:
                pass
        return {
            "action": "delete_enrollment",
            "status": "ok",
            "message": "enrollment embedding deleted" if deleted else "enrollment embedding absent",
            "enrollment_scope_id": resolved_user,
            "enrollment_id": resolved_enrollment,
            "deleted": deleted,
        }

    @staticmethod
    def _normalise_per_item_ints(
        values: Optional[list[Optional[int]]],
        *,
        n_items: int,
        default: int,
        max_value: Optional[int] = None,
    ) -> list[int]:
        if values is None:
            out = [default] * n_items
        elif len(values) == 1 and n_items != 1:
            out = [default if values[0] is None else int(values[0])] * n_items
        elif len(values) == n_items:
            out = [default if value is None else int(value) for value in values]
        else:
            raise ValueError(f"expected 1 or {n_items} values, got {len(values)}")
        if max_value is not None:
            out = [min(value, max_value) for value in out]
        return out

    def _normalise_per_item_user_ids(
        self,
        values: Optional[list[Optional[str]]],
        *,
        n_items: int,
    ) -> list[str]:
        if values is None:
            out = [self.default_user] * n_items
        elif len(values) == 1 and n_items != 1:
            out = [self._resolve_user(values[0])] * n_items
        elif len(values) == n_items:
            out = [self._resolve_user(value) for value in values]
        else:
            raise ValueError(f"expected 1 or {n_items} HOTWORD_POOL_ID values, got {len(values)}")
        return out

    def _normalise_per_item_enrollment_ids(
        self,
        values: Optional[list[Optional[str]]],
        *,
        n_items: int,
    ) -> list[Optional[str]]:
        if values is None:
            return [None] * n_items
        if len(values) == 1 and n_items != 1:
            value = values[0]
            resolved = self._resolve_enrollment_id(value) if value else None
            return [resolved] * n_items
        if len(values) == n_items:
            return [
                self._resolve_enrollment_id(value) if value else None
                for value in values
            ]
        raise ValueError(f"expected 1 or {n_items} ENROLLMENT_ID values, got {len(values)}")

    @staticmethod
    def _normalise_per_item_bools(
        values: Optional[list[Optional[bool]]],
        *,
        n_items: int,
        default: bool,
    ) -> list[bool]:
        if values is None:
            return [default] * n_items
        if len(values) == 1 and n_items != 1:
            value = default if values[0] is None else bool(values[0])
            return [value] * n_items
        if len(values) == n_items:
            return [default if value is None else bool(value) for value in values]
        raise ValueError(f"expected 1 or {n_items} boolean values, got {len(values)}")

    def _results_from_features(
        self,
        features: torch.Tensor,
        feat_lens: torch.Tensor,
        top_k_values: list[int],
        user_ids: list[str],
        *,
        enable_hotword_recalls: Optional[list[bool]] = None,
        enrollment_ids: Optional[list[Optional[str]]] = None,
        enrollment_scope_ids: Optional[list[str]] = None,
        packed_audio: bool = False,
    ) -> list[InferResult]:
        features = features.to(self._device)
        feat_lens = feat_lens.to(self._device)
        if enable_hotword_recalls is None:
            enable_hotword_recalls = [True] * len(top_k_values)
        if enrollment_ids is None:
            enrollment_ids = [None] * len(top_k_values)
        if enrollment_scope_ids is None:
            enrollment_scope_ids = user_ids

        if packed_audio and hasattr(self.audio_tower, "forward_with_projector_packed"):
            pooled, proj, proj_lens = self.audio_tower.forward_with_projector_packed(
                features, feat_lens
            )
        else:
            pooled, proj, proj_lens = self.audio_tower.forward_with_projector(
                features, feat_lens
            )

        results: list[InferResult] = []
        for i, top_k in enumerate(top_k_values):
            pool = self._get_or_load_pool(user_ids[i])
            with pool.lock:
                pool_embs_gpu = pool.embs_gpu
                hotword_pool = list(pool.words)
            pool_size = len(hotword_pool)
            top_k = 0 if not enable_hotword_recalls[i] else min(max(int(top_k), 0), pool_size)
            if top_k:
                scores = pooled[i] @ pool_embs_gpu.T
                indices = scores.topk(top_k).indices.tolist()
                words = [hotword_pool[j] for j in indices]
            else:
                words = []
            plen = int(proj_lens[i].item())
            projector_out = proj[i, :plen, :].detach().cpu().numpy().astype(np.float32)
            enrollment_id = enrollment_ids[i]
            enrollment_projector_out = None
            enrollment_projector_len = 0
            enrollment_cache_hit = None
            if enrollment_id:
                with self._enrollment_lock:
                    entry, enrollment_cache_hit = self._load_enrollment_locked(
                        user_id=enrollment_scope_ids[i],
                        enrollment_id=enrollment_id,
                    )
                if entry is None:
                    raise ValueError(
                        f"enrollment embedding not found: enrollment_scope_id={enrollment_scope_ids[i]} "
                        f"enrollment_id={enrollment_id}"
                    )
                enrollment_projector_out = (
                    entry.tensor.detach().cpu().numpy().astype(np.float32)
                )
                enrollment_projector_len = int(enrollment_projector_out.shape[0])
            results.append(
                InferResult(
                    word_list=words,
                    projector_out=projector_out,
                    projector_len=plen,
                    enrollment_id=enrollment_id,
                    enrollment_projector_out=enrollment_projector_out,
                    enrollment_projector_len=enrollment_projector_len,
                    enrollment_cache_hit=enrollment_cache_hit,
                )
            )
        return results

    @torch.no_grad()
    def infer(
        self,
        wav: np.ndarray,
        sample_rate: int = 16000,
        top_k: Optional[int] = None,
        user_id: Optional[str] = None,
        enable_hotword_recall: bool = True,
        enrollment_id: Optional[str] = None,
        enrollment_scope_id: Optional[str] = None,
    ) -> InferResult:
        """Run retrieval on a single waveform."""
        return self.infer_many(
            [wav],
            sample_rates=[sample_rate],
            top_ks=[top_k],
            user_ids=[user_id],
            enable_hotword_recalls=[enable_hotword_recall],
            enrollment_ids=[enrollment_id],
            enrollment_scope_ids=[enrollment_scope_id],
        )[0]

    @torch.no_grad()
    def infer_many(
        self,
        wavs: list[np.ndarray],
        *,
        sample_rates: Optional[list[int]] = None,
        top_ks: Optional[list[Optional[int]]] = None,
        user_ids: Optional[list[Optional[str]]] = None,
        enable_hotword_recalls: Optional[list[Optional[bool]]] = None,
        enrollment_ids: Optional[list[Optional[str]]] = None,
        enrollment_scope_ids: Optional[list[Optional[str]]] = None,
    ) -> list[InferResult]:
        """Run retrieval for multiple waveforms in one model call.

        The returned list preserves input order.  Variable-length projector
        outputs are sliced back to per-request arrays using ``projector_len``.
        """
        if not wavs:
            return []
        if sample_rates is None:
            sample_rates = [16000] * len(wavs)
        if top_ks is None:
            top_ks = [None] * len(wavs)
        if len(sample_rates) != len(wavs):
            raise ValueError(
                f"sample_rates length mismatch: {len(sample_rates)} vs {len(wavs)}"
            )
        if len(top_ks) != len(wavs):
            raise ValueError(f"top_ks length mismatch: {len(top_ks)} vs {len(wavs)}")

        top_k_values = self._normalise_per_item_ints(
            top_ks,
            n_items=len(wavs),
            default=self.cfg.default_top_k,
        )
        user_id_values = self._normalise_per_item_user_ids(
            user_ids,
            n_items=len(wavs),
        )
        hotword_recall_values = self._normalise_per_item_bools(
            enable_hotword_recalls,
            n_items=len(wavs),
            default=True,
        )
        enrollment_id_values = self._normalise_per_item_enrollment_ids(
            enrollment_ids,
            n_items=len(wavs),
        )
        enrollment_scope_id_values = (
            list(user_id_values)
            if enrollment_scope_ids is None
            else self._normalise_per_item_user_ids(
                enrollment_scope_ids,
                n_items=len(wavs),
            )
        )

        features, feat_lens = self._wavs_to_features(wavs, sample_rates)
        return self._results_from_features(
            features,
            feat_lens,
            top_k_values,
            user_id_values,
            enable_hotword_recalls=hotword_recall_values,
            enrollment_ids=enrollment_id_values,
            enrollment_scope_ids=enrollment_scope_id_values,
        )

    @torch.no_grad()
    def infer_padded_batch(
        self,
        wav_batch: np.ndarray,
        wav_lens: np.ndarray,
        *,
        sample_rates: Optional[list[Optional[int]]] = None,
        top_ks: Optional[list[Optional[int]]] = None,
        user_ids: Optional[list[Optional[str]]] = None,
        enable_hotword_recalls: Optional[list[Optional[bool]]] = None,
        enrollment_ids: Optional[list[Optional[str]]] = None,
        enrollment_scope_ids: Optional[list[Optional[str]]] = None,
        packed_audio: bool = False,
    ) -> list[InferResult]:
        """Run retrieval for an explicit padded waveform batch.

        ``wav_batch`` is ``(B, T_max)`` and ``wav_lens`` is ``(B,)`` in samples.
        This is the v2 Triton contract.
        """
        wav_batch = np.asarray(wav_batch, dtype=np.float32)
        wav_lens = np.asarray(wav_lens, dtype=np.int64).reshape(-1)
        if wav_batch.ndim != 2:
            raise ValueError(f"wav_batch must be 2D (B, T_max), got {wav_batch.shape}")
        if wav_lens.shape[0] != wav_batch.shape[0]:
            raise ValueError(
                f"wav_lens length mismatch: {wav_lens.shape[0]} vs batch {wav_batch.shape[0]}"
            )
        n_items = int(wav_batch.shape[0])
        sample_rate_values = self._normalise_per_item_ints(
            sample_rates,
            n_items=n_items,
            default=16000,
        )
        top_k_values = self._normalise_per_item_ints(
            top_ks,
            n_items=n_items,
            default=self.cfg.default_top_k,
        )
        user_id_values = self._normalise_per_item_user_ids(
            user_ids,
            n_items=n_items,
        )
        hotword_recall_values = self._normalise_per_item_bools(
            enable_hotword_recalls,
            n_items=n_items,
            default=True,
        )
        enrollment_id_values = self._normalise_per_item_enrollment_ids(
            enrollment_ids,
            n_items=n_items,
        )
        enrollment_scope_id_values = (
            list(user_id_values)
            if enrollment_scope_ids is None
            else self._normalise_per_item_user_ids(
                enrollment_scope_ids,
                n_items=n_items,
            )
        )
        wavs = [
            wav_batch[i, : int(wav_lens[i])].copy()
            for i in range(n_items)
        ]
        features, feat_lens = self._wavs_to_features(wavs, sample_rate_values)
        return self._results_from_features(
            features,
            feat_lens,
            top_k_values,
            user_id_values,
            enable_hotword_recalls=hotword_recall_values,
            enrollment_ids=enrollment_id_values,
            enrollment_scope_ids=enrollment_scope_id_values,
            packed_audio=packed_audio,
        )

    @classmethod
    def from_parameters(cls, params: dict[str, str]) -> "RAGASRRetriever":
        """Build from Triton ``config.pbtxt`` string parameters."""
        return cls(
            ServeConfig(
                base_model_path=params["base_model_path"],
                hotword_pool_file=params.get("hotword_pool_file") or None,
                hotword_pool_dir=params.get("hotword_pool_dir", "var/hotwords"),
                seed_pool_file=params.get("seed_pool_file") or None,
                default_user=params.get("default_user", "default"),
                adapter_ckpt=params.get("adapter_ckpt") or None,
                adapter_subdir=params.get(
                    "adapter_subdir", DEFAULT_HOTWORD_ADAPTER_SUBDIR
                ),
                adapter_filename=params.get(
                    "adapter_filename", DEFAULT_HOTWORD_ADAPTER_FILENAME
                ),
                embed_dim=int(params.get("embed_dim", "512")),
                adapter_hidden_dim=int(params["adapter_hidden_dim"])
                if params.get("adapter_hidden_dim")
                else 512,
                default_top_k=int(params.get("default_top_k", "50")),
                cache_dir=params.get("cache_dir", "_retrieve_cache"),
                device=params.get("device", "cuda"),
                enrollment_store_dir=params.get(
                    "enrollment_store_dir", "var/enrollments"
                ),
                enrollment_cache_max_entries=int(
                    params.get("enrollment_cache_max_entries", "256")
                ),
                enrollment_cache_device=params.get(
                    "enrollment_cache_device", "auto"
                ),
                enrollment_metadata_touch_interval_sec=float(
                    params.get("enrollment_metadata_touch_interval_sec", "300")
                ),
            )
        )


def word_list_json(words: list[str]) -> str:
    return json.dumps(words, ensure_ascii=False)
