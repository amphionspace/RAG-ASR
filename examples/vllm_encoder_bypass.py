#!/usr/bin/env python3
"""Validate vLLM audio-encoder bypass with Triton projector outputs.

The script runs two requests with the same prompt:

1. baseline: vLLM receives OpenAI ``input_audio`` and runs its own encoder.
2. bypass: Triton produces ``PROJECTOR_OUT[:PROJECTOR_LEN]`` and vLLM receives
   that tensor as an ``audio_embeds`` content block.

The bypass request requires the vLLM service to be started with
``--enable-mm-embeds`` and a model/plugin that accepts ``audio_embeds``.

Shared protocol helpers live in ``rag_asr.vllm_bypass``; this file is only the
single-sample command-line demo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_asr.vllm_bypass import (  # noqa: E402
    DEFAULT_TRITON_MODEL,
    TritonProjectorClient,
    audio_embeds_block,
    build_messages,
    call_chat_completions,
    discover_vllm_model,
    input_audio_block,
    load_audio,
    normalize_triton_url,
    normalize_vllm_url,
    stable_embed_uuid,
)

DEFAULT_WAV = ROOT / "examples" / "audio" / "cv_zh_33411896.wav"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vLLM raw-audio encoder path with Triton audio_embeds bypass."
    )
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument("--vllm-url", default="http://localhost:8009")
    parser.add_argument("--model", default=None, help="vLLM served model id; autodetect if omitted")
    parser.add_argument("--triton-url", default="localhost:8000")
    parser.add_argument("--triton-model", default=DEFAULT_TRITON_MODEL)
    parser.add_argument("--triton-top-k", type=int, default=0)
    parser.add_argument("--prompt-style", choices=["qwen3_asr", "swift"], default="qwen3_asr")
    parser.add_argument("--language", default="zh-cn", help="Used only by --prompt-style swift")
    parser.add_argument("--hotwords", nargs="*", default=[])
    parser.add_argument("--use-retrieved-hotwords", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    vllm_url = normalize_vllm_url(args.vllm_url)
    triton_url = normalize_triton_url(args.triton_url)
    model = discover_vllm_model(vllm_url, args.model)

    audio, sample_rate, audio_b64 = load_audio(args.wav)
    triton = TritonProjectorClient(triton_url, args.triton_model)

    t0 = time.perf_counter()
    embedding = triton.infer(audio, sample_rate, top_k=args.triton_top_k)
    triton_latency_ms = (time.perf_counter() - t0) * 1000.0

    hotwords = list(args.hotwords)
    if args.use_retrieved_hotwords:
        hotwords.extend(embedding.word_list)

    baseline = None
    if not args.skip_baseline:
        baseline_messages = build_messages(
            input_audio_block(audio_b64),
            prompt_style=args.prompt_style,
            hotwords=hotwords,
            language=args.language,
        )
        latency_ms, text, usage = call_chat_completions(
            vllm_url=vllm_url,
            model=model,
            messages=baseline_messages,
            max_tokens=args.max_tokens,
        )
        baseline = {"latency_ms": latency_ms, "text": text, "usage": usage}

    bypass_messages = build_messages(
        audio_embeds_block(embedding.frames, uuid=stable_embed_uuid(audio_b64)),
        prompt_style=args.prompt_style,
        hotwords=hotwords,
        language=args.language,
    )
    bypass_latency_ms, bypass_text, bypass_usage = call_chat_completions(
        vllm_url=vllm_url,
        model=model,
        messages=bypass_messages,
        max_tokens=args.max_tokens,
    )

    report = {
        "vllm_url": vllm_url,
        "model": model,
        "triton_url": triton_url,
        "triton_model": args.triton_model,
        "wav": str(args.wav),
        "sample_rate": sample_rate,
        "prompt_style": args.prompt_style,
        "projector": {
            "latency_ms": round(triton_latency_ms, 2),
            "projector_len": embedding.projector_len,
            "projector_out_shape": list(embedding.projector_out.shape),
            "frames_shape": list(embedding.frames.shape),
            "retrieved_words": embedding.word_list,
        },
        "baseline_vllm_encoder": None
        if baseline is None
        else {
            "latency_ms": round(baseline["latency_ms"], 2),
            "text": baseline["text"],
            "usage": baseline["usage"],
        },
        "bypass_triton_encoder": {
            "latency_ms": round(bypass_latency_ms, 2),
            "text": bypass_text,
            "usage": bypass_usage,
        },
    }
    if baseline is not None:
        report["comparison"] = {
            "text_equal": baseline["text"] == bypass_text,
            "bypass_minus_baseline_vllm_ms": round(
                bypass_latency_ms - float(baseline["latency_ms"]), 2
            ),
            "triton_plus_bypass_minus_baseline_ms": round(
                triton_latency_ms + bypass_latency_ms - float(baseline["latency_ms"]),
                2,
            ),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
