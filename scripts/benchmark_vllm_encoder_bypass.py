#!/usr/bin/env python3
"""Batch benchmark vLLM raw-audio encoder path against Triton audio_embeds bypass."""

from __future__ import annotations

import argparse
import json
import statistics
import string
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

from test_vllm_encoder_bypass import (
    TritonProjectorClient,
    _normalize_triton_url,
    _normalize_vllm_url,
    audio_embeds_block,
    build_messages,
    call_chat_completions,
    discover_vllm_model,
    input_audio_block,
    load_audio,
    stable_embed_uuid,
)


DEFAULT_OUTPUT = Path("var/benchmarks/vllm_encoder_bypass/base_v2_kespeech_gpu1.jsonl")
DEFAULT_SUMMARY = Path("var/benchmarks/vllm_encoder_bypass/base_v2_kespeech_gpu1.summary.json")
PUNCT_CATEGORIES = {"P", "S"}


def iter_metadata(metadata_path: Path) -> Iterable[dict[str, Any]]:
    with open(metadata_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            yield row


def resolve_audio_path(dataset_root: Path, audio_path: str) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path

    candidates = [
        dataset_root / path,
        dataset_root.parent / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def normalize_for_cer(text: str) -> str:
    """Normalize ASR text for character error rate without language-specific deps."""
    chars: list[str] = []
    for char in unicodedata.normalize("NFKC", str(text)).casefold():
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category and category[0] in PUNCT_CATEGORIES:
            continue
        if char in string.punctuation:
            continue
        chars.append(char)
    return "".join(chars)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def cer_stats(ref: str, hyp: str) -> dict[str, Any]:
    ref_norm = normalize_for_cer(ref)
    hyp_norm = normalize_for_cer(hyp)
    denom = max(len(ref_norm), 1)
    edits = edit_distance(ref_norm, hyp_norm)
    return {
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "edits": edits,
        "ref_chars": len(ref_norm),
        "cer": edits / denom,
        "exact": ref_norm == hyp_norm,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(np.floor(rank))
    high = int(np.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def round_floats(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: round_floats(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, ndigits) for v in value]
    return value


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    failed = [record for record in records if record.get("status") != "ok"]

    def lat(name: str) -> list[float]:
        return [float(record["latency_ms"][name]) for record in ok]

    baseline_edits = sum(int(record["accuracy"]["baseline"]["edits"]) for record in ok)
    bypass_edits = sum(int(record["accuracy"]["bypass"]["edits"]) for record in ok)
    ref_chars = sum(int(record["accuracy"]["baseline"]["ref_chars"]) for record in ok)
    pair_edits = sum(int(record["comparison"]["bypass_vs_baseline_edits"]) for record in ok)
    pair_chars = sum(int(record["comparison"]["baseline_chars"]) for record in ok)

    n_ok = len(ok)
    summary = {
        "count": {
            "total": len(records),
            "ok": n_ok,
            "failed": len(failed),
        },
        "accuracy": {
            "normalization": "NFKC + casefold + remove whitespace/punctuation/symbols",
            "baseline_cer": baseline_edits / max(ref_chars, 1),
            "bypass_cer": bypass_edits / max(ref_chars, 1),
            "bypass_minus_baseline_cer": (bypass_edits - baseline_edits) / max(ref_chars, 1),
            "bypass_vs_baseline_cer": pair_edits / max(pair_chars, 1),
            "baseline_exact_rate": (
                sum(1 for record in ok if record["accuracy"]["baseline"]["exact"]) / max(n_ok, 1)
            ),
            "bypass_exact_rate": (
                sum(1 for record in ok if record["accuracy"]["bypass"]["exact"]) / max(n_ok, 1)
            ),
            "same_output_rate": (
                sum(1 for record in ok if record["comparison"]["text_equal"]) / max(n_ok, 1)
            ),
            "baseline_edits": baseline_edits,
            "bypass_edits": bypass_edits,
            "ref_chars": ref_chars,
        },
        "latency_ms": {
            "baseline_vllm_encoder": latency_summary(lat("baseline_vllm_encoder")),
            "triton_projector": latency_summary(lat("triton_projector")),
            "bypass_vllm_decode": latency_summary(lat("bypass_vllm_decode")),
            "bypass_total": latency_summary(lat("bypass_total")),
            "bypass_decode_minus_baseline": latency_summary(lat("bypass_decode_minus_baseline")),
            "bypass_total_minus_baseline": latency_summary(lat("bypass_total_minus_baseline")),
        },
        "failures": [
            {
                "utt_id": record.get("utt_id"),
                "error": record.get("error"),
            }
            for record in failed[:20]
        ],
    }
    return round_floats(summary)


def call_audio_transcription(
    *,
    vllm_url: str,
    model: str,
    audio_path: Path,
) -> tuple[float, str, dict[str, Any]]:
    t0 = time.perf_counter()
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{vllm_url}/v1/audio/transcriptions",
            data={"model": model},
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=180,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if not resp.ok:
        raise RuntimeError(
            "vLLM transcription request failed: "
            f"status={resp.status_code} body={resp.text[:800]}"
        )
    data = resp.json()
    return latency_ms, str(data.get("text", "")), data.get("usage", {}) or {}


def benchmark_one(
    *,
    row: dict[str, Any],
    audio_path: Path,
    triton: TritonProjectorClient,
    vllm_url: str,
    model: str,
    prompt_style: str,
    language: str,
    max_tokens: int,
    triton_top_k: int,
    use_retrieved_hotwords: bool,
    baseline_endpoint: str,
) -> dict[str, Any]:
    audio, sample_rate, audio_b64 = load_audio(audio_path)

    t0 = time.perf_counter()
    embedding = triton.infer(audio, sample_rate, top_k=triton_top_k)
    triton_latency_ms = (time.perf_counter() - t0) * 1000.0

    hotwords = embedding.word_list if use_retrieved_hotwords else []

    if baseline_endpoint == "transcriptions":
        baseline_latency_ms, baseline_text, baseline_usage = call_audio_transcription(
            vllm_url=vllm_url,
            model=model,
            audio_path=audio_path,
        )
    else:
        baseline_messages = build_messages(
            input_audio_block(audio_b64),
            prompt_style=prompt_style,
            hotwords=hotwords,
            language=language,
        )
        baseline_latency_ms, baseline_text, baseline_usage = call_chat_completions(
            vllm_url=vllm_url,
            model=model,
            messages=baseline_messages,
            max_tokens=max_tokens,
        )

    bypass_messages = build_messages(
        audio_embeds_block(embedding.frames, uuid=stable_embed_uuid(audio_b64)),
        prompt_style=prompt_style,
        hotwords=hotwords,
        language=language,
    )
    bypass_latency_ms, bypass_text, bypass_usage = call_chat_completions(
        vllm_url=vllm_url,
        model=model,
        messages=bypass_messages,
        max_tokens=max_tokens,
    )

    ref_text = str(row.get("text", ""))
    baseline_acc = cer_stats(ref_text, baseline_text)
    bypass_acc = cer_stats(ref_text, bypass_text)
    baseline_norm = str(baseline_acc["hyp_norm"])
    bypass_norm = str(bypass_acc["hyp_norm"])
    pair_edits = edit_distance(baseline_norm, bypass_norm)
    pair_chars = max(len(baseline_norm), 1)

    return round_floats(
        {
            "status": "ok",
            "utt_id": row.get("utt_id"),
            "lang": row.get("lang"),
            "audio_path": str(audio_path),
            "audio_sec_est": row.get("audio_sec_est"),
            "reference": ref_text,
            "baseline_text": baseline_text,
            "bypass_text": bypass_text,
            "latency_ms": {
                "baseline_vllm_encoder": baseline_latency_ms,
                "triton_projector": triton_latency_ms,
                "bypass_vllm_decode": bypass_latency_ms,
                "bypass_total": triton_latency_ms + bypass_latency_ms,
                "bypass_decode_minus_baseline": bypass_latency_ms - baseline_latency_ms,
                "bypass_total_minus_baseline": triton_latency_ms + bypass_latency_ms - baseline_latency_ms,
            },
            "accuracy": {
                "baseline": baseline_acc,
                "bypass": bypass_acc,
            },
            "comparison": {
                "text_equal": baseline_text == bypass_text,
                "norm_text_equal": baseline_norm == bypass_norm,
                "bypass_vs_baseline_edits": pair_edits,
                "baseline_chars": len(baseline_norm),
                "bypass_vs_baseline_cer": pair_edits / pair_chars,
            },
            "projector": {
                "projector_len": embedding.projector_len,
                "frames_shape": list(embedding.frames.shape),
                "retrieved_words": embedding.word_list,
            },
            "usage": {
                "baseline": baseline_usage,
                "bypass": bypass_usage,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch compare pure vLLM encoder and Triton audio_embeds bypass."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--vllm-url", default="http://localhost:8009")
    parser.add_argument("--model", default="AmphionASR-1.7B")
    parser.add_argument("--triton-url", default="localhost:10001")
    parser.add_argument("--triton-model", default="rag_asr_retrieve")
    parser.add_argument("--triton-top-k", type=int, default=0)
    parser.add_argument("--prompt-style", choices=["qwen3_asr", "swift"], default="qwen3_asr")
    parser.add_argument(
        "--baseline-endpoint",
        choices=["transcriptions", "chat"],
        default="chat",
        help="Use chat input_audio for pure vLLM; transcriptions is available as a fallback.",
    )
    parser.add_argument("--language", default="zh-cn")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--use-retrieved-hotwords", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    metadata_path = args.metadata or (dataset_root / "metadata.jsonl")
    vllm_url = _normalize_vllm_url(args.vllm_url)
    triton_url = _normalize_triton_url(args.triton_url)
    model = discover_vllm_model(vllm_url, args.model)
    triton = TritonProjectorClient(triton_url, args.triton_model)

    rows = list(iter_metadata(metadata_path))
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with open(args.output, "w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            audio_path = resolve_audio_path(dataset_root, str(row["audio_path"]))
            try:
                record = benchmark_one(
                    row=row,
                    audio_path=audio_path,
                    triton=triton,
                    vllm_url=vllm_url,
                    model=model,
                    prompt_style=args.prompt_style,
                    language=args.language,
                    max_tokens=args.max_tokens,
                    triton_top_k=args.triton_top_k,
                    use_retrieved_hotwords=args.use_retrieved_hotwords,
                    baseline_endpoint=args.baseline_endpoint,
                )
            except Exception as exc:
                record = {
                    "status": "error",
                    "utt_id": row.get("utt_id"),
                    "audio_path": str(audio_path),
                    "error": repr(exc),
                }
                if args.fail_fast:
                    raise

            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{idx}/{len(rows)}] {record.get('utt_id')} {record['status']}",
                file=sys.stderr,
                flush=True,
            )

    summary = summarize(records)
    summary.update(
        {
            "dataset": str(dataset_root),
            "metadata": str(metadata_path),
            "vllm_url": vllm_url,
            "model": model,
            "triton_url": triton_url,
            "triton_model": args.triton_model,
            "output": str(args.output),
        }
    )
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
