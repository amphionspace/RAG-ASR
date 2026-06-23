"""Online RAG-ASR retrieval service: projector frames + hotword list."""

from __future__ import annotations

import json
import logging
import os
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
from rag_asr.model_loader import load_towers

logger = logging.getLogger(__name__)


@dataclass
class ServeConfig:
    base_model_path: str
    adapter_ckpt: str
    hotword_pool_file: str
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


class RAGASRRetriever:
    """Load dual-tower model once; run per-request audio → words + projector output."""

    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self._device = torch.device(cfg.device)
        self._pool_lock = threading.RLock()

        _, self.audio_tower, self.text_tower, self.tokenizer = load_towers(
            cfg.base_model_path,
            cfg.adapter_ckpt,
            embed_dim=cfg.embed_dim,
            adapter_hidden_dim=cfg.adapter_hidden_dim,
            device=self._device,
        )
        self.projector_dim = int(self.audio_tower.projector_dim)

        self.hotword_pool = load_hotword_pool_file(cfg.hotword_pool_file)
        self._pool_embs_gpu = self._load_pool_embeddings().to(self._device)

        from lhotse.features import WhisperFbank, WhisperFbankConfig

        self.fbank = WhisperFbank(WhisperFbankConfig(num_filters=cfg.num_mel_bins))
        logger.info(
            "RAGASRRetriever ready: pool=%d words, projector_dim=%d, device=%s",
            len(self.hotword_pool),
            self.projector_dim,
            self._device,
        )

    def _cache_path(self) -> Optional[Path]:
        cache_dir = self.cfg.cache_dir
        if cache_dir and cache_dir.lower() in {"none", "off"}:
            cache_dir = None
        return text_emb_cache_path(
            cache_dir,
            adapter_ckpt=self.cfg.adapter_ckpt,
            hotword_pool_file=self.cfg.hotword_pool_file,
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

    def _load_pool_embeddings(self) -> torch.Tensor:
        cache_path = self._cache_path()

        if cache_path is not None and cache_path.exists():
            cached_words, embs = load_text_emb_cache(cache_path)
            if cached_words == self.hotword_pool:
                return embs

        def _encode() -> torch.Tensor:
            logger.info("encoding hotword pool (%d words)…", len(self.hotword_pool))
            embs = self._encode_hotwords(self.hotword_pool)
            if cache_path is not None:
                save_text_emb_cache(
                    cache_path,
                    self.hotword_pool,
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
                if cached_words == self.hotword_pool:
                    return embs
            embs = _encode()
        return embs

    def _persist_pool_locked(self) -> None:
        path = Path(self.cfg.hotword_pool_file)
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
                for word in self.hotword_pool:
                    tmp.write(f"{word}\n")
            os.replace(tmp_name, path)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass

    def _save_pool_cache_locked(self) -> None:
        cache_path = self._cache_path()
        if cache_path is None:
            return
        save_text_emb_cache(
            cache_path,
            self.hotword_pool,
            self._pool_embs_gpu.detach().cpu(),
            overwrite=True,
        )

    def _set_pool_locked(
        self,
        words: list[str],
        embs: torch.Tensor,
        *,
        persist: bool = True,
    ) -> None:
        self.hotword_pool = words
        self._pool_embs_gpu = embs.to(self._device)
        if persist:
            self._persist_pool_locked()
        self._save_pool_cache_locked()

    def list_hotwords(
        self,
        *,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> dict[str, object]:
        """Return a page from the in-memory hotword pool."""

        query_norm = normalize_hotword(query).casefold() if query else ""
        offset = max(int(offset), 0)
        if limit is None:
            limit_value = None
        else:
            limit_value = max(int(limit), 0)

        with self._pool_lock:
            total_count = len(self.hotword_pool)
            if query_norm:
                matched = [
                    word
                    for word in self.hotword_pool
                    if query_norm in word.casefold()
                ]
            else:
                matched = list(self.hotword_pool)

        if limit_value is None:
            page = matched[offset:]
        else:
            page = matched[offset : offset + limit_value]
        return {
            "action": "list",
            "status": "ok",
            "message": f"returned {len(page)} hotwords",
            "hotwords": page,
            "total_count": total_count,
            "matched_count": len(matched),
            "offset": offset,
            "limit": limit_value,
        }

    def add_hotwords(self, values: list[object]) -> dict[str, object]:
        """Add hotwords, incrementally encode new entries, and persist the pool."""

        with self._pool_lock:
            existing_keys = {hotword_dedupe_key(word) for word in self.hotword_pool}
            batch = normalize_hotwords(values, existing_keys=existing_keys)
            if batch.words:
                new_embs = self._encode_hotwords(batch.words)
                current_embs = self._pool_embs_gpu.detach().cpu()
                combined_words = self.hotword_pool + batch.words
                combined_embs = torch.cat([current_embs, new_embs], dim=0)
                order = sorted(
                    range(len(combined_words)),
                    key=lambda i: hotword_dedupe_key(combined_words[i]),
                )
                words = [combined_words[i] for i in order]
                embs = combined_embs[order]
                self._set_pool_locked(words, embs)
            total_count = len(self.hotword_pool)
        added = len(batch.words)
        return {
            "action": "add",
            "status": "ok",
            "message": f"added {added} hotwords",
            "hotwords": batch.words,
            "total_count": total_count,
            "added": added,
            "skipped_duplicates": len(batch.duplicates),
            "invalid": batch.invalid,
            "duplicates": batch.duplicates,
        }

    def delete_hotwords(self, values: list[object]) -> dict[str, object]:
        """Delete hotwords by canonical dedupe key and persist the pool."""

        batch = normalize_hotwords(values, sort=False)
        delete_keys = {hotword_dedupe_key(word) for word in batch.words}
        with self._pool_lock:
            existing = {
                hotword_dedupe_key(word): word
                for word in self.hotword_pool
            }
            missing = [
                word
                for word in batch.words
                if hotword_dedupe_key(word) not in existing
            ]
            keep_indices = [
                i
                for i, word in enumerate(self.hotword_pool)
                if hotword_dedupe_key(word) not in delete_keys
            ]
            deleted = len(self.hotword_pool) - len(keep_indices)
            if deleted:
                words = [self.hotword_pool[i] for i in keep_indices]
                if keep_indices:
                    index = torch.tensor(keep_indices, device=self._device)
                    embs = self._pool_embs_gpu.index_select(0, index).detach().cpu()
                else:
                    embs = torch.empty((0, self.cfg.embed_dim), dtype=torch.float32)
                self._set_pool_locked(words, embs)
            total_count = len(self.hotword_pool)
        return {
            "action": "delete",
            "status": "ok",
            "message": f"deleted {deleted} hotwords",
            "hotwords": batch.words,
            "total_count": total_count,
            "deleted": deleted,
            "missing": missing,
            "invalid": batch.invalid,
            "duplicates": batch.duplicates,
        }

    def reload_hotwords(self) -> dict[str, object]:
        """Reload the configured pool file, rebuild embeddings, and refresh cache."""

        words = load_hotword_pool_file(self.cfg.hotword_pool_file)
        with self._pool_lock:
            old_count = len(self.hotword_pool)
            self.hotword_pool = words
            embs = self._load_pool_embeddings()
            self._set_pool_locked(words, embs, persist=True)
            total_count = len(self.hotword_pool)
        return {
            "action": "reload",
            "status": "ok",
            "message": f"reloaded hotword pool: {old_count} -> {total_count}",
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

    def _results_from_features(
        self,
        features: torch.Tensor,
        feat_lens: torch.Tensor,
        top_k_values: list[int],
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

        with self._pool_lock:
            pool_embs_gpu = self._pool_embs_gpu
            hotword_pool = list(self.hotword_pool)

        scores = pooled @ pool_embs_gpu.T
        pool_size = len(hotword_pool)
        results: list[InferResult] = []
        for i, top_k in enumerate(top_k_values):
            top_k = min(max(int(top_k), 0), pool_size)
            indices = scores[i].topk(top_k).indices.tolist()
            words = [hotword_pool[j] for j in indices]
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
    ) -> InferResult:
        """Run retrieval on a single waveform."""
        return self.infer_many([wav], sample_rates=[sample_rate], top_ks=[top_k])[0]

    @torch.no_grad()
    def infer_many(
        self,
        wavs: list[np.ndarray],
        *,
        sample_rates: Optional[list[int]] = None,
        top_ks: Optional[list[Optional[int]]] = None,
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

        features, feat_lens = self._wavs_to_features(wavs, sample_rates)
        return self._results_from_features(features, feat_lens, top_k_values)

    @torch.no_grad()
    def infer_padded_batch(
        self,
        wav_batch: np.ndarray,
        wav_lens: np.ndarray,
        *,
        sample_rates: Optional[list[Optional[int]]] = None,
        top_ks: Optional[list[Optional[int]]] = None,
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
        wavs = [
            wav_batch[i, : int(wav_lens[i])].copy()
            for i in range(n_items)
        ]
        features, feat_lens = self._wavs_to_features(wavs, sample_rate_values)
        return self._results_from_features(
            features,
            feat_lens,
            top_k_values,
            packed_audio=packed_audio,
        )

    @classmethod
    def from_parameters(cls, params: dict[str, str]) -> "RAGASRRetriever":
        """Build from Triton ``config.pbtxt`` string parameters."""
        return cls(
            ServeConfig(
                base_model_path=params["base_model_path"],
                adapter_ckpt=params["adapter_ckpt"],
                hotword_pool_file=params["hotword_pool_file"],
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
