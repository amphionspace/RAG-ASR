#!/usr/bin/env python3
"""HTTP smoke-test client for the Triton RAG-ASR retrieval model."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import soundfile as sf


def main() -> None:
    p = argparse.ArgumentParser(description="Call rag_asr_retrieve via Triton HTTP")
    p.add_argument("--url", default="localhost:8000")
    p.add_argument("--wav", required=True)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    import tritonclient.http as httpclient

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    client = httpclient.InferenceServerClient(url=args.url)
    inputs = [
        httpclient.InferInput("WAV", wav.shape, "FP32"),
        httpclient.InferInput("TOP_K", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(wav)
    inputs[1].set_data_from_numpy(np.array([args.top_k], dtype=np.int32))

    outputs = [
        httpclient.InferRequestedOutput("PROJECTOR_OUT"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        httpclient.InferRequestedOutput("WORD_LIST"),
    ]
    result = client.infer("rag_asr_retrieve", inputs, outputs=outputs)
    words = json.loads(result.as_numpy("WORD_LIST")[0].decode())
    plen = int(result.as_numpy("PROJECTOR_LEN")[0])
    proj = result.as_numpy("PROJECTOR_OUT")
    print(json.dumps({
        "word_list": words,
        "projector_len": plen,
        "projector_out_shape": list(proj.shape),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
