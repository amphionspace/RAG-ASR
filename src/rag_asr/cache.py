"""Text-embedding cache helpers for neural hotword retrieval."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from rag_asr.hotwords import normalize_hotwords

logger = logging.getLogger(__name__)


@contextmanager
def text_emb_cache_lock(path: Path) -> Iterator[None]:
    """Cross-process exclusive lock; removes the lock file on release."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def text_emb_cache_path(
    cache_dir: Optional[str | Path],
    *,
    adapter_ckpt: str,
    biasing_tsv_file: Optional[str] = None,
    hotword_pool_file: Optional[str] = None,
) -> Optional[Path]:
    if not cache_dir:
        return None
    key_parts = [adapter_ckpt]
    if biasing_tsv_file:
        key_parts.append(f"tsv:{biasing_tsv_file}")
    elif hotword_pool_file:
        key_parts.append(f"pool:{hotword_pool_file}")
    h = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"text_embs__{h}.npz"


def load_text_emb_cache(path: Path) -> tuple[list[str], torch.Tensor]:
    data = np.load(str(path), allow_pickle=True)
    words: list[str] = data["words"].tolist()
    embs = torch.from_numpy(data["embs"].astype("float32"))
    logger.info(
        "text emb cache loaded: %s (%d words, dim=%d)",
        path, len(words), embs.shape[1],
    )
    return words, embs


def save_text_emb_cache(
    path: Path,
    words: list[str],
    embs: torch.Tensor,
    *,
    acquire_lock: bool = True,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        if path.exists() and not overwrite:
            logger.info("text emb cache already exists, skip save: %s", path)
            return
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=str(path.parent),
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                np.savez_compressed(
                    tmp,
                    words=np.array(words, dtype=object),
                    embs=embs.cpu().numpy().astype("float32"),
                )
            os.replace(tmp_name, path)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
        logger.info("text emb cache saved: %s (%d words)", path, len(words))

    if not acquire_lock:
        _write()
        return

    with text_emb_cache_lock(path):
        _write()


def load_biasing_tsv(tsv_file: str) -> dict[str, list[str]]:
    """Parse IS21 Deep Bias TSV → {utt_id: [candidate, ...]}."""
    per_item: dict[str, list[str]] = {}
    with open(tsv_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                logger.warning(
                    "biasing_tsv line %d has %d fields, skipping: %s",
                    lineno, len(parts), tsv_file,
                )
                continue
            utt_id = parts[0].strip()
            try:
                candidates = json.loads(parts[3])
            except Exception as e:
                logger.warning(
                    "biasing_tsv line %d JSON parse failed: %s", lineno, e,
                )
                continue
            if not isinstance(candidates, list):
                continue
            per_item[utt_id] = [str(w) for w in candidates if w]
    logger.info("biasing_tsv loaded %d utterances from %s", len(per_item), tsv_file)
    return per_item


def load_hotword_pool_file(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return normalize_hotwords(f).words
