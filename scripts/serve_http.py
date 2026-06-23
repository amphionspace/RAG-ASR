#!/usr/bin/env python3
"""Local HTTP service (no Triton) for debugging — uses conda ``triton`` env directly."""

from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from rag_asr.serve import RAGASRRetriever, ServeConfig

app = FastAPI(title="RAG-ASR Retrieve")
_retriever: Optional[RAGASRRetriever] = None


def _numpy_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


@app.on_event("startup")
def _load_model():
    global _retriever
    cfg = app.state.serve_cfg
    _retriever = RAGASRRetriever(cfg)


@app.post("/retrieve")
async def retrieve(
    file: UploadFile = File(...),
    top_k: Optional[int] = Form(None),
    sample_rate: int = Form(16000),
):
    import soundfile as sf

    data = await file.read()
    import io

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if sr != sample_rate:
        sample_rate = sr
    result = _retriever.infer(wav, sample_rate=sample_rate, top_k=top_k)
    return JSONResponse({
        "word_list": result.word_list,
        "projector_len": result.projector_len,
        "projector_out": result.projector_out.tolist(),
        "projector_dim": int(result.projector_out.shape[1]),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-ckpt", required=True)
    p.add_argument("--hotword-pool-file", required=True)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--cache-dir", default="_retrieve_cache")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    app.state.serve_cfg = ServeConfig(
        base_model_path=args.base_model_path,
        adapter_ckpt=args.adapter_ckpt,
        hotword_pool_file=args.hotword_pool_file,
        default_top_k=args.top_k,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
