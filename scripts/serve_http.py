#!/usr/bin/env python3
"""Local HTTP service (no Triton) for debugging — uses conda ``triton`` env directly."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from rag_asr.config import load_config
from rag_asr.serve import RAGASRRetriever, ServeConfig

app = FastAPI(title="RAG-ASR Retrieve")
_retriever: Optional[RAGASRRetriever] = None


def _numpy_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def _require_retriever() -> RAGASRRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="retriever is not ready")
    return _retriever


def _hotword_values(payload: dict[str, object]) -> list[object]:
    values = payload.get("hotwords", payload.get("words", []))
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return values
    raise HTTPException(status_code=400, detail="hotwords must be a string or list")


def _reject_legacy_enrollment_scope_fields(**values: object) -> None:
    legacy = sorted(name for name, value in values.items() if value is not None)
    if legacy:
        joined = ", ".join(legacy)
        raise HTTPException(
            status_code=400,
            detail=f"use enrollment_scope_id; unsupported field(s): {joined}",
        )


def _reject_legacy_hotword_pool_fields(**values: object) -> None:
    legacy = sorted(name for name, value in values.items() if value is not None)
    if legacy:
        joined = ", ".join(legacy)
        raise HTTPException(
            status_code=400,
            detail=f"use hotword_pool_id; unsupported field(s): {joined}",
        )


async def _read_float32_upload(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) % np.dtype(np.float32).itemsize != 0:
        raise HTTPException(status_code=400, detail="audio file must be float32 PCM")
    return np.frombuffer(data, dtype=np.float32).copy()


@app.on_event("startup")
def _load_model():
    global _retriever
    cfg = app.state.serve_cfg
    _retriever = RAGASRRetriever(cfg)


@app.get("/health")
async def health():
    return {"status": "ok", "ready": _retriever is not None}


@app.post("/retrieve")
async def retrieve(
    file: UploadFile = File(...),
    hotword_pool_id: str = Form("default"),
    user_id: Optional[str] = Form(None),
    top_k: Optional[int] = Form(None),
    sample_rate: int = Form(16000),
):
    _reject_legacy_hotword_pool_fields(user_id=user_id)
    import soundfile as sf

    data = await file.read()
    import io

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if sr != sample_rate:
        sample_rate = sr
    result = _require_retriever().infer(
        wav,
        sample_rate=sample_rate,
        top_k=top_k,
        user_id=hotword_pool_id,
    )
    return JSONResponse({
        "hotword_pool_id": hotword_pool_id,
        "word_list": result.word_list,
        "projector_len": result.projector_len,
        "projector_out": result.projector_out.tolist(),
        "projector_dim": int(result.projector_out.shape[1]),
    })


@app.get("/hotword-pool")
async def list_hotword_pool(
    hotword_pool_id: str = Query("default"),
    user_id: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    limit: Optional[int] = Query(None),
    offset: int = Query(0),
):
    _reject_legacy_hotword_pool_fields(user_id=user_id)
    return JSONResponse(
        _require_retriever().list_hotwords(
            user_id=hotword_pool_id,
            query=query,
            limit=limit,
            offset=offset,
        )
    )


@app.post("/hotword-pool")
async def add_hotword_pool(payload: dict[str, object] = Body(...)):
    _reject_legacy_hotword_pool_fields(
        user_id=payload.get("user_id") if "user_id" in payload else None,
    )
    return JSONResponse(
        _require_retriever().add_hotwords(
            _hotword_values(payload),
            user_id=str(payload.get("hotword_pool_id") or "default"),
        )
    )


@app.delete("/hotword-pool")
async def delete_hotword_pool(payload: dict[str, object] = Body(...)):
    _reject_legacy_hotword_pool_fields(
        user_id=payload.get("user_id") if "user_id" in payload else None,
    )
    return JSONResponse(
        _require_retriever().delete_hotwords(
            _hotword_values(payload),
            user_id=str(payload.get("hotword_pool_id") or "default"),
        )
    )


@app.post("/hotword-pool/clear")
async def clear_hotword_pool(payload: dict[str, object] = Body(default_factory=dict)):
    _reject_legacy_hotword_pool_fields(
        user_id=payload.get("user_id") if "user_id" in payload else None,
    )
    return JSONResponse(
        _require_retriever().clear_hotwords(
            user_id=str(payload.get("hotword_pool_id") or "default"),
        )
    )


@app.post("/hotword-pool/reload")
async def reload_hotword_pool(payload: dict[str, object] = Body(default_factory=dict)):
    _reject_legacy_hotword_pool_fields(
        user_id=payload.get("user_id") if "user_id" in payload else None,
    )
    return JSONResponse(
        _require_retriever().reload_hotwords(
            user_id=str(payload.get("hotword_pool_id") or "default"),
        )
    )


@app.post("/enrollments/{enrollment_id}")
async def upsert_enrollment(
    enrollment_id: str,
    file: UploadFile = File(...),
    enrollment_scope_id: str = Form("default"),
    enrollment_user_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    sample_rate: int = Form(16000),
):
    _reject_legacy_enrollment_scope_fields(
        enrollment_user_id=enrollment_user_id,
        user_id=user_id,
    )
    wav = await _read_float32_upload(file)
    return JSONResponse(
        _require_retriever().upsert_enrollment(
            wav,
            sample_rate=sample_rate,
            enrollment_id=enrollment_id,
            enrollment_scope_id=enrollment_scope_id,
        )
    )


@app.get("/enrollments/{enrollment_id}")
async def get_enrollment(
    enrollment_id: str,
    enrollment_scope_id: str = Query("default"),
    enrollment_user_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
):
    _reject_legacy_enrollment_scope_fields(
        enrollment_user_id=enrollment_user_id,
        user_id=user_id,
    )
    return JSONResponse(
        _require_retriever().get_enrollment(
            enrollment_id=enrollment_id,
            enrollment_scope_id=enrollment_scope_id,
        )
    )


@app.delete("/enrollments/{enrollment_id}")
async def delete_enrollment(
    enrollment_id: str,
    enrollment_scope_id: str = Query("default"),
    enrollment_user_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
):
    _reject_legacy_enrollment_scope_fields(
        enrollment_user_id=enrollment_user_id,
        user_id=user_id,
    )
    return JSONResponse(
        _require_retriever().delete_enrollment(
            enrollment_id=enrollment_id,
            enrollment_scope_id=enrollment_scope_id,
        )
    )


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
    p.add_argument("--enrollment-store-dir", default=None)
    p.add_argument("--enrollment-metadata-touch-interval-sec", type=float, default=None)
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
        if args.enrollment_store_dir:
            kwargs["enrollment_store_dir"] = args.enrollment_store_dir
        if args.enrollment_metadata_touch_interval_sec is not None:
            kwargs["enrollment_metadata_touch_interval_sec"] = (
                args.enrollment_metadata_touch_interval_sec
            )
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
            enrollment_store_dir=args.enrollment_store_dir or "var/enrollments",
            enrollment_metadata_touch_interval_sec=(
                args.enrollment_metadata_touch_interval_sec
                if args.enrollment_metadata_touch_interval_sec is not None
                else 300.0
            ),
            default_user=args.default_user or "default",
            default_top_k=args.top_k,
            cache_dir=args.cache_dir,
            device=args.device or "cuda",
        )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
