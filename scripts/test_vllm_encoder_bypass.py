#!/usr/bin/env python3
"""Validate vLLM audio-encoder bypass with Triton projector outputs.

The script runs two requests with the same prompt:

1. baseline: vLLM receives OpenAI ``input_audio`` and runs its own encoder.
2. bypass: Triton produces ``PROJECTOR_OUT[:PROJECTOR_LEN]`` and vLLM receives
   that tensor as an ``audio_embeds`` content block.

The bypass request requires the vLLM service to be started with
``--enable-mm-embeds`` and a model/plugin that accepts ``audio_embeds``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAV = ROOT / "examples" / "audio" / "cv_zh_33411896.wav"
DEFAULT_TRITON_MODEL = "rag_asr_retrieve"


@dataclass(frozen=True)
class TritonAudioEmbedding:
    projector_out: np.ndarray
    projector_len: int
    word_list: list[str]

    @property
    def frames(self) -> np.ndarray:
        return np.asarray(self.projector_out[: self.projector_len], dtype=np.float32)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_vllm_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value


def _normalize_triton_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value if "://" in value else f"http://{value}")
    return parsed.netloc or parsed.path


def load_audio(path: Path, *, target_sample_rate: int = 16000) -> tuple[np.ndarray, int, str]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != int(target_sample_rate):
        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=int(sr),
            target_sr=int(target_sample_rate),
        ).astype(np.float32, copy=False)
        sr = int(target_sample_rate)

    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, audio, int(sr), format="WAV", subtype="PCM_16")
    audio_b64 = base64.b64encode(wav_bytes.getvalue()).decode("utf-8")
    return audio, int(sr), audio_b64


def discover_vllm_model(vllm_url: str, requested: str | None) -> str:
    if requested:
        return requested
    resp = requests.get(f"{vllm_url}/v1/models", timeout=5)
    resp.raise_for_status()
    models = resp.json().get("data") or []
    if not models:
        raise RuntimeError(f"no models returned by {vllm_url}/v1/models")
    return str(models[0]["id"])


def build_messages(
    audio_block: dict[str, Any],
    *,
    prompt_style: str,
    hotwords: list[str],
    language: str,
) -> list[dict[str, Any]]:
    """Build prompt shapes aligned with AmphionASR's vLLM eval client."""

    hotwords_text = ",".join(word for word in hotwords if word)
    if prompt_style == "qwen3_asr":
        # Amphion-1.7B/Qwen3-ASR style used by audiollm-demo: optional text
        # metadata in system; user message contains only audio-like blocks.
        system_lines = []
        if hotwords_text:
            system_lines.append(f"Hotwords: {hotwords_text}")
        return [
            {"role": "system", "content": "\n".join(system_lines)},
            {"role": "user", "content": [audio_block]},
        ]

    if prompt_style == "swift":
        lines = ["Transcribe the following audio."]
        if language:
            lines.append(f"Language: {language}")
        if hotwords_text:
            lines.append(f"Hotwords: {hotwords_text}")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "\n".join(lines)},
                    audio_block,
                ],
            }
        ]

    raise ValueError(f"unknown prompt style: {prompt_style}")


def input_audio_block(audio_b64: str) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {"data": audio_b64, "format": "wav"},
    }


def audio_embeds_block(frames: np.ndarray, *, uuid: str) -> dict[str, Any]:
    try:
        import torch
        from vllm.utils.serial_utils import tensor2base64
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "bypass mode needs torch and vllm in the client environment so the "
            "projector frames can be serialized with vllm.utils.serial_utils."
        ) from exc

    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    return {
        "type": "audio_embeds",
        "audio_embeds": tensor2base64(tensor),
        "uuid": uuid,
    }


def stable_embed_uuid(audio_b64: str) -> str:
    digest = hashlib.sha1(audio_b64.encode("utf-8")).hexdigest()[:16]
    return f"triton-audio-{digest}"


class TritonProjectorClient:
    def __init__(self, url: str, model_name: str):
        import tritonclient.http as httpclient

        self._httpclient = httpclient
        self._client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name

    def _string_input(self, name: str, value: str):
        tensor = self._httpclient.InferInput(name, [1], "BYTES")
        tensor.set_data_from_numpy(np.array([value], dtype=object))
        return tensor

    def _int_input(self, name: str, value: int):
        tensor = self._httpclient.InferInput(name, [1], "INT32")
        tensor.set_data_from_numpy(np.array([int(value)], dtype=np.int32))
        return tensor

    def infer(self, audio: np.ndarray, sample_rate: int, *, top_k: int) -> TritonAudioEmbedding:
        wav = np.asarray(audio, dtype=np.float32)
        inputs = [
            self._string_input("ACTION", "infer"),
            self._httpclient.InferInput("WAV", wav.shape, "FP32"),
            self._int_input("SAMPLE_RATE", sample_rate),
            self._int_input("TOP_K", top_k),
        ]
        inputs[1].set_data_from_numpy(wav)
        outputs = [
            self._httpclient.InferRequestedOutput("PROJECTOR_OUT"),
            self._httpclient.InferRequestedOutput("PROJECTOR_LEN"),
            self._httpclient.InferRequestedOutput("WORD_LIST"),
        ]
        result = self._client.infer(self.model_name, inputs, outputs=outputs)
        return TritonAudioEmbedding(
            projector_out=result.as_numpy("PROJECTOR_OUT").astype(np.float32, copy=False),
            projector_len=int(result.as_numpy("PROJECTOR_LEN")[0]),
            word_list=json.loads(_decode(result.as_numpy("WORD_LIST")[0])),
        )


def call_chat_completions(
    *,
    vllm_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[float, str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=180)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if not resp.ok:
        hint = ""
        if "audio_embeds" in resp.text or "mm-embeds" in resp.text:
            hint = " Check that vLLM was started with --enable-mm-embeds."
        raise RuntimeError(
            f"vLLM request failed: status={resp.status_code} body={resp.text[:800]}{hint}"
        )
    data = resp.json()
    return (
        latency_ms,
        data["choices"][0]["message"]["content"],
        data.get("usage", {}) or {},
    )


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

    vllm_url = _normalize_vllm_url(args.vllm_url)
    triton_url = _normalize_triton_url(args.triton_url)
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
