"""Online RAG-ASR retrieval service: projector frames + hotword list."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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

    def _load_pool_embeddings(self) -> torch.Tensor:
        cache_dir = self.cfg.cache_dir
        if cache_dir and cache_dir.lower() in {"none", "off"}:
            cache_dir = None
        cache_path = text_emb_cache_path(
            cache_dir,
            adapter_ckpt=self.cfg.adapter_ckpt,
            hotword_pool_file=self.cfg.hotword_pool_file,
        )

        if cache_path is not None and cache_path.exists():
            cached_words, embs = load_text_emb_cache(cache_path)
            if cached_words == self.hotword_pool:
                return embs

        def _encode() -> torch.Tensor:
            logger.info("encoding hotword pool (%d words)…", len(self.hotword_pool))
            parts: list[torch.Tensor] = []
            with torch.no_grad():
                for i in range(0, len(self.hotword_pool), self.cfg.batch_text):
                    chunk = self.hotword_pool[i : i + self.cfg.batch_text]
                    ids, mask = tokenise_words(chunk, self.tokenizer, device=self._device)
                    parts.append(self.text_tower(ids, mask).cpu())
            embs = torch.cat(parts, dim=0)
            if cache_path is not None:
                save_text_emb_cache(cache_path, self.hotword_pool, embs, acquire_lock=False)
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

        scores = pooled @ self._pool_embs_gpu.T
        results: list[InferResult] = []
        for i, top_k in enumerate(top_k_values):
            indices = scores[i].topk(top_k).indices.tolist()
            words = [self.hotword_pool[j] for j in indices]
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
            max_value=len(self.hotword_pool),
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
            max_value=len(self.hotword_pool),
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
