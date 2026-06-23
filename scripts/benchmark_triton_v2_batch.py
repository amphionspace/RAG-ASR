#!/usr/bin/env python3
"""Benchmark explicit-batch Triton RAG-ASR v2 against local Python."""

from __future__ import annotations

import argparse
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


def _latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": _percentile(values, 50),
        "p90_ms": _percentile(values, 90),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def _load_examples(examples_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (examples_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav.astype(np.float32), int(sr)


def _make_padded_batch(
    wavs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    wav_lens = np.array([wav.shape[0] for wav in wavs], dtype=np.int32)
    max_len = int(wav_lens.max())
    wav_batch = np.zeros((len(wavs), max_len), dtype=np.float32)
    for i, wav in enumerate(wavs):
        wav_batch[i, : wav.shape[0]] = wav
    return wav_batch, wav_lens


def _decode_words(words_raw: np.ndarray) -> list[list[str]]:
    out: list[list[str]] = []
    for item in words_raw.tolist():
        out.append(json.loads(item.decode() if isinstance(item, bytes) else item))
    return out


def _infer_triton_v2(
    client: httpclient.InferenceServerClient,
    wavs: list[np.ndarray],
    sample_rates: list[int],
    top_k: int,
) -> dict[str, Any]:
    wav_batch, wav_lens = _make_padded_batch(wavs)
    sample_rates_np = np.array(sample_rates, dtype=np.int32)
    top_k_np = np.array([top_k], dtype=np.int32)
    inputs = [
        httpclient.InferInput("WAV_BATCH", wav_batch.shape, "FP32"),
        httpclient.InferInput("WAV_LEN", wav_lens.shape, "INT32"),
        httpclient.InferInput("SAMPLE_RATE", sample_rates_np.shape, "INT32"),
        httpclient.InferInput("TOP_K", top_k_np.shape, "INT32"),
    ]
    inputs[0].set_data_from_numpy(wav_batch)
    inputs[1].set_data_from_numpy(wav_lens)
    inputs[2].set_data_from_numpy(sample_rates_np)
    inputs[3].set_data_from_numpy(top_k_np)
    outputs = [
        httpclient.InferRequestedOutput("PROJECTOR_OUT"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        httpclient.InferRequestedOutput("WORD_LIST"),
    ]
    result = client.infer("rag_asr_retrieve_v2", inputs, outputs=outputs)
    return {
        "word_list": _decode_words(result.as_numpy("WORD_LIST")),
        "projector_len": result.as_numpy("PROJECTOR_LEN").astype(np.int32),
        "projector_out": result.as_numpy("PROJECTOR_OUT").astype(np.float32),
    }


def _compare_many(local_results, batch_result: dict[str, Any]) -> dict[str, Any]:
    details = []
    for i, local in enumerate(local_results):
        plen = int(batch_result["projector_len"][i])
        batch_proj = batch_result["projector_out"][i, :plen, :]
        local_proj = local.projector_out
        same_shape = local_proj.shape == batch_proj.shape
        if same_shape:
            diff = np.abs(local_proj - batch_proj)
            max_abs = float(diff.max()) if diff.size else 0.0
            mean_abs = float(diff.mean()) if diff.size else 0.0
        else:
            max_abs = None
            mean_abs = None
        details.append({
            "index": i,
            "word_list_equal": local.word_list == batch_result["word_list"][i],
            "projector_len_equal": local.projector_len == plen,
            "projector_shape_equal": same_shape,
            "projector_shape_local": list(local_proj.shape),
            "projector_shape_batch": list(batch_proj.shape),
            "projector_max_abs_diff": max_abs,
            "projector_mean_abs_diff": mean_abs,
        })
    return {
        "all_word_lists_equal": all(item["word_list_equal"] for item in details),
        "all_projector_lens_equal": all(item["projector_len_equal"] for item in details),
        "all_projector_shapes_equal": all(item["projector_shape_equal"] for item in details),
        "max_projector_abs_diff": max(
            (item["projector_max_abs_diff"] or 0.0) for item in details
        ),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--examples-dir", type=Path, default=ROOT / "examples")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--packed-audio", action="store_true")
    args = parser.parse_args()

    from rag_asr.serve import RAGASRRetriever, ServeConfig

    rows = _load_examples(args.examples_dir)
    wav_pool = []
    sr_pool = []
    for row in rows:
        wav, sr = _load_wav(args.examples_dir / row["wav"])
        wav_pool.append(wav)
        sr_pool.append(sr)

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
    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready() or not client.is_model_ready("rag_asr_retrieve_v2"):
        raise RuntimeError(f"Triton rag_asr_retrieve_v2 is not ready at {args.url}")

    results = []
    for batch_size in args.batch_sizes:
        wavs = [wav_pool[i % len(wav_pool)] for i in range(batch_size)]
        sample_rates = [sr_pool[i % len(sr_pool)] for i in range(batch_size)]

        for _ in range(args.warmup):
            local.infer_padded_batch(
                *_make_padded_batch(wavs),
                sample_rates=sample_rates,
                top_ks=[args.top_k],
                packed_audio=args.packed_audio,
            )
            _infer_triton_v2(client, wavs, sample_rates, args.top_k)

        local_single = [
            local.infer(wav, sample_rate=sample_rate, top_k=args.top_k)
            for wav, sample_rate in zip(wavs, sample_rates)
        ]

        t0 = time.perf_counter()
        local_batch = local.infer_padded_batch(
            *_make_padded_batch(wavs),
            sample_rates=sample_rates,
            top_ks=[args.top_k],
            packed_audio=args.packed_audio,
        )
        local_batch_ms = (time.perf_counter() - t0) * 1000.0
        wav_batch, wav_lens = _make_padded_batch(wavs)
        local_batch_as_dict = {
            "word_list": [item.word_list for item in local_batch],
            "projector_len": np.array([item.projector_len for item in local_batch], dtype=np.int32),
            "projector_out": np.zeros(
                (
                    batch_size,
                    max(item.projector_len for item in local_batch),
                    local_batch[0].projector_out.shape[1],
                ),
                dtype=np.float32,
            ),
        }
        for i, item in enumerate(local_batch):
            local_batch_as_dict["projector_out"][i, : item.projector_len, :] = item.projector_out

        triton_latencies = []
        triton_result = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            triton_result = _infer_triton_v2(client, wavs, sample_rates, args.top_k)
            triton_latencies.append((time.perf_counter() - t0) * 1000.0)
        assert triton_result is not None

        results.append({
            "batch_size": batch_size,
            "wav_batch_shape": list(wav_batch.shape),
            "wav_lens": wav_lens.tolist(),
            "local_packed_batch_ms": local_batch_ms,
            "local_single_vs_packed": _compare_many(local_single, local_batch_as_dict),
            "triton_v2_latency": _latency_stats(triton_latencies),
            "local_single_vs_triton_v2": _compare_many(local_single, triton_result),
        })

    print(json.dumps({
        "model": "rag_asr_retrieve_v2",
        "top_k": args.top_k,
        "repeats": args.repeats,
        "local_packed_audio": args.packed_audio,
        "batch_results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
