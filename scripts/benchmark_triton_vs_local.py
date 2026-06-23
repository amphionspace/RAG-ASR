#!/usr/bin/env python3
"""Compare Triton retrieval against local Python and run a small load test."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import tritonclient.http as httpclient

ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((pct / 100.0) * (len(xs) - 1)))))
    return xs[idx]


def _latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    return {
        "count": float(len(latencies_ms)),
        "mean_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "p50_ms": _percentile(latencies_ms, 50),
        "p90_ms": _percentile(latencies_ms, 90),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms) if latencies_ms else 0.0,
        "max_ms": max(latencies_ms) if latencies_ms else 0.0,
    }


def _load_examples(examples_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (examples_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit is not None else rows


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return wav, int(sr)


def _infer_triton(
    client: httpclient.InferenceServerClient,
    wav: np.ndarray,
    sample_rate: int,
    top_k: int,
) -> dict[str, Any]:
    inputs = [
        httpclient.InferInput("WAV", wav.shape, "FP32"),
        httpclient.InferInput("SAMPLE_RATE", [1], "INT32"),
        httpclient.InferInput("TOP_K", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(wav)
    inputs[1].set_data_from_numpy(np.array([sample_rate], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([top_k], dtype=np.int32))
    outputs = [
        httpclient.InferRequestedOutput("PROJECTOR_OUT"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        httpclient.InferRequestedOutput("WORD_LIST"),
    ]
    result = client.infer("rag_asr_retrieve", inputs, outputs=outputs)
    return {
        "word_list": json.loads(result.as_numpy("WORD_LIST")[0].decode()),
        "projector_len": int(result.as_numpy("PROJECTOR_LEN")[0]),
        "projector_out": result.as_numpy("PROJECTOR_OUT").astype(np.float32),
    }


def _compare_result(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    lhs_proj = lhs["projector_out"]
    rhs_proj = rhs["projector_out"]
    same_shape = lhs_proj.shape == rhs_proj.shape
    if same_shape:
        diff = np.abs(lhs_proj - rhs_proj)
        max_abs = float(diff.max()) if diff.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0
    else:
        max_abs = None
        mean_abs = None
    return {
        "word_list_equal": lhs["word_list"] == rhs["word_list"],
        "projector_len_equal": lhs["projector_len"] == rhs["projector_len"],
        "projector_shape_equal": same_shape,
        "projector_shape_lhs": list(lhs_proj.shape),
        "projector_shape_rhs": list(rhs_proj.shape),
        "projector_max_abs_diff": max_abs,
        "projector_mean_abs_diff": mean_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--examples-dir", type=Path, default=ROOT / "examples")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pressure-concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--pressure-requests", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    from rag_asr.serve import RAGASRRetriever, ServeConfig

    examples_dir = args.examples_dir.resolve()
    rows = _load_examples(examples_dir, limit=args.limit)
    wavs: list[np.ndarray] = []
    sample_rates: list[int] = []
    for row in rows:
        wav, sr = _load_wav(examples_dir / row["wav"])
        wavs.append(wav)
        sample_rates.append(sr)

    cfg = ServeConfig(
        base_model_path=str(ROOT / "checkpoints/base/amphion_1.7b_merged"),
        adapter_ckpt=str(
            ROOT / "checkpoints/adapters/amphion-1.7b_retrieval_v1.2/best_adapter.pt"
        ),
        hotword_pool_file="/chenmingjie/lx/data/hotword/zh/zh-10k.txt",
        embed_dim=512,
        adapter_hidden_dim=512,
        default_top_k=args.top_k,
        cache_dir=str(ROOT / "_retrieve_cache"),
        device="cuda",
    )
    local = RAGASRRetriever(cfg)
    triton_client = httpclient.InferenceServerClient(url=args.url)
    if not triton_client.is_server_ready() or not triton_client.is_model_ready("rag_asr_retrieve"):
        raise RuntimeError(f"Triton rag_asr_retrieve is not ready at {args.url}")

    # Warm up both paths before measuring.
    for _ in range(args.warmup):
        local.infer(wavs[0], sample_rate=sample_rates[0], top_k=args.top_k)
        _infer_triton(triton_client, wavs[0], sample_rates[0], args.top_k)

    t0 = time.perf_counter()
    local_single_results = [
        local.infer(wav, sample_rate=sr, top_k=args.top_k)
        for wav, sr in zip(wavs, sample_rates)
    ]
    local_single_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    local_batch_results = local.infer_many(
        wavs,
        sample_rates=sample_rates,
        top_ks=[args.top_k] * len(wavs),
    )
    local_batch_ms = (time.perf_counter() - t0) * 1000.0

    local_comparisons = []
    for row, single, batch in zip(rows, local_single_results, local_batch_results):
        cmp = _compare_result(single.to_dict(), batch.to_dict())
        cmp["id"] = row["id"]
        local_comparisons.append(cmp)

    triton_latencies_ms = []
    triton_comparisons = []
    for row, wav, sr, local_result in zip(rows, wavs, sample_rates, local_single_results):
        t0 = time.perf_counter()
        triton_result = _infer_triton(triton_client, wav, sr, args.top_k)
        triton_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        cmp = _compare_result(local_result.to_dict(), triton_result)
        cmp["id"] = row["id"]
        triton_comparisons.append(cmp)

    pressure_rows = []
    pressure_wav = wavs[0]
    pressure_sr = sample_rates[0]

    def _one_request() -> float:
        client = httpclient.InferenceServerClient(url=args.url)
        t_start = time.perf_counter()
        _infer_triton(client, pressure_wav, pressure_sr, args.top_k)
        return (time.perf_counter() - t_start) * 1000.0

    for concurrency in args.pressure_concurrency:
        request_count = max(args.pressure_requests, concurrency)
        t_start = time.perf_counter()
        latencies: list[float] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_one_request) for _ in range(request_count)]
            for future in concurrent.futures.as_completed(futures):
                latencies.append(float(future.result()))
        wall_s = time.perf_counter() - t_start
        stats = _latency_stats(latencies)
        pressure_rows.append({
            "concurrency": concurrency,
            "requests": request_count,
            "wall_time_s": wall_s,
            "throughput_rps": request_count / wall_s if wall_s > 0 else 0.0,
            **stats,
        })

    report = {
        "examples": [row["id"] for row in rows],
        "top_k": args.top_k,
        "local_single_total_ms": local_single_ms,
        "local_batch_total_ms": local_batch_ms,
        "local_batch_speedup": local_single_ms / local_batch_ms if local_batch_ms > 0 else 0.0,
        "local_single_vs_batch": {
            "all_word_lists_equal": all(c["word_list_equal"] for c in local_comparisons),
            "all_projector_lens_equal": all(c["projector_len_equal"] for c in local_comparisons),
            "all_projector_shapes_equal": all(c["projector_shape_equal"] for c in local_comparisons),
            "max_projector_abs_diff": max(
                (c["projector_max_abs_diff"] or 0.0) for c in local_comparisons
            ),
            "details": local_comparisons,
        },
        "triton_vs_local": {
            "latency": _latency_stats(triton_latencies_ms),
            "all_word_lists_equal": all(c["word_list_equal"] for c in triton_comparisons),
            "all_projector_lens_equal": all(c["projector_len_equal"] for c in triton_comparisons),
            "all_projector_shapes_equal": all(c["projector_shape_equal"] for c in triton_comparisons),
            "max_projector_abs_diff": max(
                (c["projector_max_abs_diff"] or 0.0) for c in triton_comparisons
            ),
            "details": triton_comparisons,
        },
        "pressure": pressure_rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
