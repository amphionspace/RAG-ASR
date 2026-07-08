"""Retrieval scoring mode helpers."""

from __future__ import annotations


POOLED_RETRIEVAL_MODE = "pooled"
FRAME_MAX_RETRIEVAL_MODE = "frame_max"
VALID_RETRIEVAL_MODES = (POOLED_RETRIEVAL_MODE, FRAME_MAX_RETRIEVAL_MODE)
_RETRIEVAL_MODE_ALIASES = {
    "pooled": POOLED_RETRIEVAL_MODE,
    "global": POOLED_RETRIEVAL_MODE,
    "frame_max": FRAME_MAX_RETRIEVAL_MODE,
    "local": FRAME_MAX_RETRIEVAL_MODE,
}


def normalize_retrieval_mode(value: str) -> str:
    """Return the canonical retrieval mode name.

    ``global`` and ``local`` are kept as compatibility aliases for the old
    names. New config should use ``pooled`` or ``frame_max``.
    """
    key = str(value).strip().lower().replace("-", "_")
    try:
        return _RETRIEVAL_MODE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(VALID_RETRIEVAL_MODES)
        aliases = ", ".join(sorted(_RETRIEVAL_MODE_ALIASES))
        raise ValueError(
            f"retrieval_mode must be one of {valid} "
            f"(legacy aliases: {aliases}), got {value!r}."
        ) from exc


def is_frame_max_retrieval(value: str) -> bool:
    return normalize_retrieval_mode(value) == FRAME_MAX_RETRIEVAL_MODE
