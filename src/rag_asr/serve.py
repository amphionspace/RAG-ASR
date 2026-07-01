"""Online RAG-ASR retrieval service: projector frames + hotword list."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
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


@dataclass
class InferResult:
    word_list: list[str]
    projector_out: np.ndarray  # (T', D_proj) float32
    projector_len: int

    def to_dict(self) -> dict:
        return {
            "word_list": self.word_list,
            "projector_out": self.projector_out,
            "projector_len": self.projector_len,
        }


@dataclass
class _UserPool:
    user_id: str
    pool_file: Path
    words: list[str]
    embs_gpu: torch.Tensor
    lock: threading.RLock


class RAGASRRetriever:
    """Load dual-tower model once; run per-request audio → words + projector output."""

    def __init__(self, cfg: ServeConfig):
        from rag_asr.model_loader import load_towers

        self.cfg = cfg
        self._device = torch.device(cfg.device)
        self.default_user = self._resolve_user(cfg.default_user)
        self._pools_lock = threading.RLock()
        self._pools: dict[str, _UserPool] = {}
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

    def _resolve_user(self, user_id: Optional[str]) -> str:
        user = normalize_hotword(user_id or self.cfg.default_user)
        if not user:
            user = getattr(self, "default_user", self.cfg.default_user)
        if not _USER_ID_RE.fullmatch(user) or user in {".", ".."}:
            raise ValueError(
                "USER_ID must contain only letters, digits, dot, underscore, or hyphen"
            )
        return user

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
            "user_id": pool.user_id,
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
            "user_id": pool.user_id,
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
            "user_id": pool.user_id,
            "hotwords": batch.words,
            "total_count": total_count,
            "deleted": deleted,
            "missing": missing,
            "invalid": batch.invalid,
            "duplicates": batch.duplicates,
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
            "user_id": pool.user_id,
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
            raise ValueError(f"expected 1 or {n_items} USER_ID values, got {len(values)}")
        return out

    def _results_from_features(
        self,
        features: torch.Tensor,
        feat_lens: torch.Tensor,
        top_k_values: list[int],
        user_ids: list[str],
        *,
        packed_audio: bool = False,
    ) -> list[InferResult]:
        features = features.to(self._device)
        feat_lens = feat_lens.to(self._device)

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
            top_k = min(max(int(top_k), 0), pool_size)
            if top_k:
                scores = pooled[i] @ pool_embs_gpu.T
                indices = scores.topk(top_k).indices.tolist()
                words = [hotword_pool[j] for j in indices]
            else:
                words = []
            plen = int(proj_lens[i].item())
            projector_out = proj[i, :plen, :].detach().cpu().numpy().astype(np.float32)
            results.append(
                InferResult(
                    word_list=words,
                    projector_out=projector_out,
                    projector_len=plen,
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
    ) -> InferResult:
        """Run retrieval on a single waveform."""
        return self.infer_many(
            [wav],
            sample_rates=[sample_rate],
            top_ks=[top_k],
            user_ids=[user_id],
        )[0]

    @torch.no_grad()
    def infer_many(
        self,
        wavs: list[np.ndarray],
        *,
        sample_rates: Optional[list[int]] = None,
        top_ks: Optional[list[Optional[int]]] = None,
        user_ids: Optional[list[Optional[str]]] = None,
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

        features, feat_lens = self._wavs_to_features(wavs, sample_rates)
        return self._results_from_features(
            features,
            feat_lens,
            top_k_values,
            user_id_values,
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
            )
        )


def word_list_json(words: list[str]) -> str:
    return json.dumps(words, ensure_ascii=False)
