"""Shared protocol helpers for the vLLM audio-encoder bypass.

Two callers build on the exact same wire protocol:

1. baseline: vLLM receives an OpenAI ``input_audio`` block and runs its own
   audio encoder.
2. bypass: Triton produces ``PROJECTOR_OUT[:PROJECTOR_LEN]`` and vLLM receives
   that tensor as an ``audio_embeds`` content block.

The single-sample example (``examples/vllm_encoder_bypass.py``) and the batch
evaluation (``evaluation/benchmark_vllm_encoder_bypass.py``) import the helpers
below so the protocol lives in one place.

The bypass request requires the vLLM service to be started with
``--enable-mm-embeds`` and a model/plugin that accepts ``audio_embeds``.

Heavy or optional dependencies (tritonclient, torch, vllm, librosa) are imported
lazily inside the functions that need them. Importing this module only requires
numpy, soundfile and requests, so it never forces the Triton/vLLM client stack
onto the core package.
"""

from __future__ import annotations

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


def normalize_vllm_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value


def normalize_triton_url(value: str) -> str:
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
    def __init__(self, url: str, model_name: str = DEFAULT_TRITON_MODEL):
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
