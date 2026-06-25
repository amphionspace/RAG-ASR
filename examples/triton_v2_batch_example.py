#!/usr/bin/env python3
"""HTTP smoke-test client for explicit-batch Triton RAG-ASR v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_wavs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavs: list[np.ndarray] = []
    sample_rates: list[int] = []
    for path in paths:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wavs.append(wav.astype(np.float32))
        sample_rates.append(int(sr))
    wav_lens = np.array([wav.shape[0] for wav in wavs], dtype=np.int32)
    max_len = int(wav_lens.max())
    wav_batch = np.zeros((len(wavs), max_len), dtype=np.float32)
    for i, wav in enumerate(wavs):
        wav_batch[i, : wav.shape[0]] = wav
    return wav_batch, wav_lens, np.array(sample_rates, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call rag_asr_retrieve_v2 via Triton HTTP")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--wav", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    import tritonclient.http as httpclient

    wav_paths = [Path(path) for path in args.wav]
    wav_batch, wav_lens, sample_rates = _load_wavs(wav_paths)
    top_k = np.array([args.top_k], dtype=np.int32)

    client = httpclient.InferenceServerClient(url=args.url)
    inputs = [
        httpclient.InferInput("WAV_BATCH", wav_batch.shape, "FP32"),
        httpclient.InferInput("WAV_LEN", wav_lens.shape, "INT32"),
        httpclient.InferInput("SAMPLE_RATE", sample_rates.shape, "INT32"),
        httpclient.InferInput("TOP_K", top_k.shape, "INT32"),
    ]
    inputs[0].set_data_from_numpy(wav_batch)
    inputs[1].set_data_from_numpy(wav_lens)
    inputs[2].set_data_from_numpy(sample_rates)
    inputs[3].set_data_from_numpy(top_k)
    outputs = [
        httpclient.InferRequestedOutput("PROJECTOR_OUT"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        httpclient.InferRequestedOutput("WORD_LIST"),
    ]
    result = client.infer("rag_asr_retrieve_v2", inputs, outputs=outputs)
    words_raw = result.as_numpy("WORD_LIST")
    words = [
        json.loads(item.decode() if isinstance(item, bytes) else item)
        for item in words_raw.tolist()
    ]
    projector_len = result.as_numpy("PROJECTOR_LEN")
    projector_out = result.as_numpy("PROJECTOR_OUT")
    print(json.dumps({
        "word_list": words,
        "projector_len": projector_len.tolist(),
        "projector_out_shape": list(projector_out.shape),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
