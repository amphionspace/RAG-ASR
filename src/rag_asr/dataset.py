"""Dataset and data utilities for dual-tower retrieval training.

Data format
-----------
Each training sample is derived from a Lhotse *cut* whose supervision
carries ``custom.hotwords`` — a list of rare-word strings that are
considered the ground-truth hotwords for that utterance.

The dataset yields items in the form::

    {
        "id"      : str,
        "audio"   : np.ndarray  shape (num_samples,)  float32, 16 kHz
        "hotwords": list[str]   (may be empty, those items are skipped)
    }

The ``retrieval_collate_fn`` batches those items, extracts WhisperFbank
mel features on-the-fly, and pads everything to the same length.

Usage
-----
>>> from lhotse import CutSet
>>> from lhotse.features import WhisperFbank, WhisperFbankConfig
>>> fbank = WhisperFbank(WhisperFbankConfig(num_filters=128))
>>> dataset = HotwordsRetrievalDataset(cuts)
>>> loader  = DataLoader(dataset, batch_size=16, collate_fn=retrieval_collate_fn_factory(fbank))
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def infer_retrieval_lang(language: Optional[str]) -> str:
    """Map Lhotse supervision ``language`` to retrieval pool key ``en`` / ``zh``."""
    if not language:
        return "zh"
    lang = language.strip().lower()
    if lang in {"english", "en"}:
        return "en"
    if lang in {"chinese", "zh", "cmn", "mandarin"}:
        return "zh"
    # ISO 639-3 style prefixes
    if lang.startswith("en"):
        return "en"
    if lang.startswith("zh") or lang.startswith("cmn"):
        return "zh"
    return "zh"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HotwordsRetrievalDataset(Dataset):
    """Lhotse-backed dataset for dual-tower contrastive training.

    Only cuts that have at least one non-empty hotword are retained.

    Parameters
    ----------
    cuts
        A Lhotse ``CutSet`` (in-memory list or lazy).  Cuts must be
        pre-trimmed so that each cut spans exactly one supervision.
    min_hotwords : int
        Minimum number of hotwords required; cuts with fewer are discarded.
    max_duration_s : float or None
        Discard cuts longer than this (seconds).  Avoids OOM on very long
        utterances during training.
    """

    def __init__(
        self,
        cuts,
        min_hotwords: int = 1,
        max_duration_s: Optional[float] = 30.0,
    ):
        super().__init__()
        self._cuts: list = []
        n_skipped = 0
        for cut in cuts:
            sups = list(cut.supervisions)
            if not sups:
                n_skipped += 1
                continue
            custom = getattr(sups[0], "custom", None) or {}
            hotwords = [
                h.strip()
                for h in (custom.get("hotwords") or [])
                if isinstance(h, str) and h.strip()
            ]
            if len(hotwords) < min_hotwords:
                n_skipped += 1
                continue
            if max_duration_s is not None and cut.duration > max_duration_s:
                n_skipped += 1
                continue
            lang = infer_retrieval_lang(getattr(sups[0], "language", None))
            self._cuts.append((cut, hotwords, lang))
        logger.info(
            "HotwordsRetrievalDataset: loaded %d cuts, skipped %d "
            "(too short hotword list or over max_duration).",
            len(self._cuts), n_skipped,
        )

    def __len__(self) -> int:
        return len(self._cuts)

    def __getitem__(self, idx: int) -> dict:
        cut, hotwords, lang = self._cuts[idx]
        # Load mono audio; CutSet should already be resampled to 16 kHz upstream.
        audio = cut.load_audio()  # (C, T) float32 numpy
        if audio.ndim == 2:
            audio = audio[0]  # take first channel → (T,)
        return {
            "id": cut.id,
            "audio": audio.astype(np.float32),
            "sample_rate": 16000,
            "hotwords": hotwords,
            "lang": lang,
        }


# ---------------------------------------------------------------------------
# Global rare-word vocabulary helper
# ---------------------------------------------------------------------------

def build_rare_word_vocab(cuts) -> list[str]:
    """Collect the union of all ``custom.hotwords`` across a CutSet.

    Used to build the global negative-sampling pool for training.
    Duplicates and empty strings are removed; ordering is alphabetical
    for reproducibility.
    """
    vocab: set[str] = set()
    for cut in cuts:
        sups = list(cut.supervisions)
        if not sups:
            continue
        custom = getattr(sups[0], "custom", None) or {}
        for h in custom.get("hotwords") or []:
            if isinstance(h, str) and h.strip():
                vocab.add(h.strip())
    return sorted(vocab)


# ---------------------------------------------------------------------------
# Collate function factory
# ---------------------------------------------------------------------------

def retrieval_collate_fn_factory(fbank_extractor) -> Callable[[list[dict]], dict]:
    """Return a collate function that extracts WhisperFbank features.

    Parameters
    ----------
    fbank_extractor
        A ``lhotse.features.WhisperFbank`` instance (or any object with a
        ``.extract(samples, sampling_rate)`` method that returns an
        ``(T, F)`` numpy array).

    Returns
    -------
    A function suitable for passing as ``collate_fn`` to a DataLoader.
    """

    def collate(batch: list[dict]) -> dict:
        ids = [item["id"] for item in batch]
        hotwords_list = [item["hotwords"] for item in batch]
        langs = [item.get("lang", "zh") for item in batch]

        # Extract per-utterance mel features
        feats_list: list[torch.Tensor] = []
        for item in batch:
            feat = fbank_extractor.extract(
                samples=item["audio"],
                sampling_rate=item["sample_rate"],
            )  # (T, F) numpy
            feats_list.append(torch.from_numpy(feat))

        # Pad features to max length in batch
        feature_lens = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
        max_t = int(feature_lens.max().item())
        n_feats = feats_list[0].shape[1]
        features = torch.zeros(len(feats_list), max_t, n_feats, dtype=torch.float32)
        for i, f in enumerate(feats_list):
            features[i, : f.shape[0], :] = f

        return {
            "ids": ids,
            "features": features,          # (B, T_max, F)
            "feature_lens": feature_lens,  # (B,)
            "hotwords": hotwords_list,     # list[list[str]]
            "langs": langs,                # list[str]  'en' | 'zh'
        }

    return collate


# ---------------------------------------------------------------------------
# Tokenise a flat list of words for the text tower
# ---------------------------------------------------------------------------

def tokenise_words(
    words: list[str],
    tokenizer,
    max_length: int = 32,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise ``words`` for the text tower.

    Returns
    -------
    input_ids : ``(N, L)`` long
    attention_mask : ``(N, L)`` float
    """
    enc = tokenizer(
        words,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"].float()
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask
