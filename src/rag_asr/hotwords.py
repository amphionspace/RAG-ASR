"""Hotword normalisation, validation, and deduplication helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class HotwordBatch:
    """Canonical hotword batch plus rejected entries for API summaries."""

    words: list[str]
    invalid: list[str]
    duplicates: list[str]


def normalize_hotword(value: str) -> str:
    """Return the canonical storage form for a hotword.

    The online service stores the exact canonical surface form that will be
    embedded.  Stronger match-only transforms such as accent stripping or
    pinyin conversion belong in a later rerank/matching layer, not in storage.
    """

    return _WHITESPACE_RE.sub(" ", value.strip())


def is_char_based_script(hotword: str) -> bool:
    """Return True for scripts that are normally counted by character."""

    char_based = 0
    for ch in hotword:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0xF900 <= cp <= 0xFAFF
            or 0x0E00 <= cp <= 0x0E7F
            or 0x3000 <= cp <= 0x303F
            or 0xFF00 <= cp <= 0xFFEF
        ):
            char_based += 1
    alpha = sum(1 for ch in hotword if ch.isalpha())
    return alpha > 0 and char_based > alpha / 2


def hotword_token_count(hotword: str) -> int:
    """Count CJK/Thai hotwords by character and Latin phrases by words."""

    if is_char_based_script(hotword):
        return sum(1 for ch in hotword if not ch.isspace())
    return len(hotword.split())


def is_valid_hotword(
    hotword: str,
    *,
    max_len: int = 32,
    min_len: int = 2,
    min_len_noncjk: int = 1,
) -> bool:
    """Validate hotword length using AmphionASR-style script-aware counts."""

    if not hotword:
        return False
    count = hotword_token_count(hotword)
    effective_min = min_len if is_char_based_script(hotword) else min_len_noncjk
    return effective_min <= count <= max_len


def hotword_dedupe_key(
    hotword: str,
    *,
    casefold_noncjk: bool = True,
) -> str:
    """Return the key used to dedupe canonical hotwords."""

    if casefold_noncjk and not is_char_based_script(hotword):
        return hotword.casefold()
    return hotword


def normalize_hotwords(
    values: Iterable[object],
    *,
    existing_keys: set[str] | None = None,
    sort: bool = True,
) -> HotwordBatch:
    """Normalise, validate, and dedupe a batch of hotword-like values."""

    seen = set(existing_keys or set())
    words: list[str] = []
    invalid: list[str] = []
    duplicates: list[str] = []

    for value in values:
        if not isinstance(value, str):
            invalid.append(str(value))
            continue
        word = normalize_hotword(value)
        if not is_valid_hotword(word):
            invalid.append(value)
            continue
        key = hotword_dedupe_key(word)
        if key in seen:
            duplicates.append(word)
            continue
        seen.add(key)
        words.append(word)

    if sort:
        words.sort(key=hotword_dedupe_key)
    return HotwordBatch(words=words, invalid=invalid, duplicates=duplicates)

