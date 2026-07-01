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

from rag_asr.config import load_config
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
    user_id: str = Form("default"),
    top_k: Optional[int] = Form(None),
    sample_rate: int = Form(16000),
):
    import soundfile as sf

    data = await file.read()
    import io

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if sr != sample_rate:
        sample_rate = sr
    result = _retriever.infer(
        wav,
        sample_rate=sample_rate,
        top_k=top_k,
        user_id=user_id,
    )
    return JSONResponse({
        "user_id": user_id,
        "word_list": result.word_list,
        "projector_len": result.projector_len,
        "projector_out": result.projector_out.tolist(),
        "projector_dim": int(result.projector_out.shape[1]),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="RAG-ASR YAML config path")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--base-model-path", default=None)
    p.add_argument("--adapter-ckpt", default=None)
    p.add_argument("--hotword-pool-file", default=None)
    p.add_argument("--hotword-pool-dir", default=None)
    p.add_argument("--seed-pool-file", default=None)
    p.add_argument("--default-user", default=None)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.config or not args.base_model_path:
        kwargs = load_config(args.config).to_serve_kwargs()
        if args.base_model_path:
            kwargs["base_model_path"] = args.base_model_path
        if args.adapter_ckpt:
            kwargs["adapter_ckpt"] = args.adapter_ckpt
        if args.hotword_pool_file:
            kwargs["hotword_pool_file"] = args.hotword_pool_file
        if args.hotword_pool_dir:
            kwargs["hotword_pool_dir"] = args.hotword_pool_dir
        if args.seed_pool_file:
            kwargs["seed_pool_file"] = args.seed_pool_file
        if args.default_user:
            kwargs["default_user"] = args.default_user
        if args.cache_dir:
            kwargs["cache_dir"] = args.cache_dir
        if args.device:
            kwargs["device"] = args.device
        if args.top_k:
            kwargs["default_top_k"] = args.top_k
        app.state.serve_cfg = ServeConfig(**kwargs)
    else:
        app.state.serve_cfg = ServeConfig(
            base_model_path=args.base_model_path,
            adapter_ckpt=args.adapter_ckpt,
            hotword_pool_file=args.hotword_pool_file,
            hotword_pool_dir=args.hotword_pool_dir or "var/hotwords",
            seed_pool_file=args.seed_pool_file,
            default_user=args.default_user or "default",
            default_top_k=args.top_k,
            cache_dir=args.cache_dir,
            device=args.device or "cuda",
        )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
