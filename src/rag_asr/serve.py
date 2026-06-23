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

    @torch.no_grad()
    def infer(
        self,
        wav: np.ndarray,
        sample_rate: int = 16000,
        top_k: Optional[int] = None,
    ) -> InferResult:
        """Run retrieval on a single 16 kHz mono waveform."""
        top_k = self.cfg.default_top_k if top_k is None else int(top_k)
        top_k = min(top_k, len(self.hotword_pool))

        features, feat_lens = self._wav_to_features(wav, sample_rate)
        features = features.to(self._device)
        feat_lens = feat_lens.to(self._device)

        pooled, proj, proj_lens = self.audio_tower.forward_with_projector(
            features, feat_lens
        )
        scores = pooled[0] @ self._pool_embs_gpu.T
        indices = scores.topk(top_k).indices.tolist()
        words = [self.hotword_pool[i] for i in indices]

        plen = int(proj_lens[0].item())
        projector_out = proj[0, :plen, :].detach().cpu().numpy().astype(np.float32)

        return InferResult(
            word_list=words,
            projector_out=projector_out,
            projector_len=plen,
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
