#!/usr/bin/env python3
"""Triton client for RAG-ASR retrieval and hotword management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np
import soundfile as sf


MODEL_NAME = "rag_asr_retrieve"


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _string_input(name: str, value: str):
    import tritonclient.http as httpclient

    tensor = httpclient.InferInput(name, [1], "BYTES")
    tensor.set_data_from_numpy(np.array([value], dtype=object))
    return tensor


def _int_input(name: str, value: int):
    import tritonclient.http as httpclient

    tensor = httpclient.InferInput(name, [1], "INT32")
    tensor.set_data_from_numpy(np.array([int(value)], dtype=np.int32))
    return tensor


def _load_hotword_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if path.suffix.lower() == ".json" or stripped[0] in "[{":
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("hotwords", [])
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON hotwords must be a list")
        return [str(item) for item in data]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _collect_words(words: Iterable[str], files: Iterable[Path] | None = None) -> list[str]:
    out = list(words)
    for path in files or []:
        out.extend(_load_hotword_file(path))
    return out


def _management_call(args, action: str, *, words: list[str] | None = None) -> dict:
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=args.url)
    inputs = [
        _string_input("ACTION", action),
        _string_input("USER_ID", args.user),
    ]
    if words is not None:
        inputs.append(_string_input("HOTWORDS", json.dumps(words, ensure_ascii=False)))
    if getattr(args, "query", None):
        inputs.append(_string_input("QUERY", args.query))
    if getattr(args, "limit", None) is not None:
        inputs.append(_int_input("LIMIT", args.limit))
    if getattr(args, "offset", None) is not None:
        inputs.append(_int_input("OFFSET", args.offset))

    outputs = [
        httpclient.InferRequestedOutput("STATUS"),
        httpclient.InferRequestedOutput("MESSAGE"),
        httpclient.InferRequestedOutput("HOTWORD_COUNT"),
        httpclient.InferRequestedOutput("HOTWORD_LIST"),
    ]
    result = client.infer(args.model, inputs, outputs=outputs)
    message = json.loads(_decode(result.as_numpy("MESSAGE")[0]))
    hotwords = json.loads(_decode(result.as_numpy("HOTWORD_LIST")[0]))
    message["status"] = _decode(result.as_numpy("STATUS")[0])
    message["hotword_count"] = int(result.as_numpy("HOTWORD_COUNT")[0])
    message["hotwords"] = hotwords
    return message


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - status diagnostics
        return {"error": str(exc)} if default is None else default


def _simplify_io(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    simplified = []
    for item in items or []:
        shape = item.get("shape")
        if shape is None:
            shape = item.get("dims")
        simplified.append({
            "name": item.get("name"),
            "datatype": item.get("datatype") or item.get("data_type"),
            "shape": shape,
        })
    return simplified


def _simplify_parameters(params: dict[str, Any] | None) -> dict[str, Any]:
    out = {}
    for key, value in (params or {}).items():
        if isinstance(value, dict) and "string_value" in value:
            out[key] = value["string_value"]
        else:
            out[key] = value
    return out


def _model_config_summary(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    return {
        "name": config.get("name"),
        "backend": config.get("backend"),
        "max_batch_size": config.get("max_batch_size"),
        "inputs": _simplify_io(config.get("input")),
        "outputs": _simplify_io(config.get("output")),
        "parameters": _simplify_parameters(config.get("parameters")),
    }


def _model_metadata_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        "name": metadata.get("name"),
        "versions": metadata.get("versions"),
        "platform": metadata.get("platform"),
        "inputs": _simplify_io(metadata.get("inputs")),
        "outputs": _simplify_io(metadata.get("outputs")),
    }


def _important_parameters(config: dict[str, Any] | None) -> dict[str, Any]:
    params = _simplify_parameters(config.get("parameters") if isinstance(config, dict) else {})
    keys = [
        "base_model_path",
        "adapter_subdir",
        "adapter_filename",
        "adapter_ckpt",
        "hotword_pool_file",
        "hotword_pool_dir",
        "seed_pool_file",
        "default_user",
        "cache_dir",
        "default_top_k",
        "device",
        "embed_dim",
        "adapter_hidden_dim",
    ]
    return {key: params[key] for key in keys if key in params and params[key] != ""}


def _cmd_status(args) -> dict:
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=args.url)
    server_live = _safe_call(client.is_server_live, default=False)
    server_ready = _safe_call(client.is_server_ready, default=False)
    model_ready = _safe_call(lambda: client.is_model_ready(args.model), default=False)
    metadata = _safe_call(lambda: client.get_model_metadata(args.model), default={})
    config = _safe_call(lambda: client.get_model_config(args.model), default={})
    hotwords = _management_call(args, "list")

    status = {
        "url": args.url,
        "model": args.model,
        "server_live": server_live,
        "server_ready": server_ready,
        "model_ready": model_ready,
        "hotword_pool": {
            "user": hotwords.get("user_id") or args.user,
            "total_count": hotwords.get("hotword_count"),
            "matched_count": hotwords.get("matched_count"),
            "sample_limit": args.limit,
            "query": args.query,
            "sample": hotwords.get("hotwords", []),
        },
        "parameters": _important_parameters(config),
    }
    if args.verbose:
        status["metadata"] = _model_metadata_summary(metadata)
        status["config"] = _model_config_summary(config)
    return status


def _format_bool(value: Any) -> str:
    if value is True:
        return "OK"
    if value is False:
        return "NO"
    return str(value)


def _print_status_text(status: dict[str, Any]) -> None:
    hotword_pool = status.get("hotword_pool", {}) or {}
    params = status.get("parameters", {}) or {}
    sample = hotword_pool.get("sample") or []

    print("RAG-ASR Triton Hotword Service")
    print("=" * 32)
    print(f"URL          : {status.get('url')}")
    print(f"Model        : {status.get('model')}")
    print(f"Server live  : {_format_bool(status.get('server_live'))}")
    print(f"Server ready : {_format_bool(status.get('server_ready'))}")
    print(f"Model ready  : {_format_bool(status.get('model_ready'))}")
    print()
    print("Hotword Pool")
    print("-" * 32)
    print(f"User         : {hotword_pool.get('user')}")
    print(f"Total count  : {hotword_pool.get('total_count')}")
    print(f"Matched count: {hotword_pool.get('matched_count')}")
    if hotword_pool.get("query"):
        print(f"Query        : {hotword_pool.get('query')}")
    print(f"Sample limit : {hotword_pool.get('sample_limit')}")
    if sample:
        print("Sample words :")
        for idx, word in enumerate(sample, start=1):
            print(f"  {idx:>2}. {word}")
    else:
        print("Sample words : <empty>")

    if params:
        print()
        print("Key Parameters")
        print("-" * 32)
        labels = {
            "base_model_path": "Base model",
            "adapter_subdir": "Adapter dir",
            "adapter_filename": "Adapter file",
            "adapter_ckpt": "Adapter ckpt",
            "hotword_pool_file": "Pool file",
            "hotword_pool_dir": "Pool dir",
            "seed_pool_file": "Seed pool",
            "default_user": "Default user",
            "cache_dir": "Cache dir",
            "default_top_k": "Default top_k",
            "device": "Device",
            "embed_dim": "Embed dim",
            "adapter_hidden_dim": "Adapter hidden",
        }
        width = max(len(label) for label in labels.values())
        for key, label in labels.items():
            if key in params:
                print(f"{label:<{width}} : {params[key]}")

    if "metadata" in status or "config" in status:
        print()
        print("Verbose Schema")
        print("-" * 32)
        print(json.dumps(
            {
                "metadata": status.get("metadata"),
                "config": status.get("config"),
            },
            ensure_ascii=False,
            indent=2,
        ))


def _cmd_infer(args) -> dict:
    import tritonclient.http as httpclient

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    client = httpclient.InferenceServerClient(url=args.url)
    inputs = [
        httpclient.InferInput("WAV", wav.shape, "FP32"),
        _string_input("USER_ID", args.user),
        _int_input("SAMPLE_RATE", int(sr)),
        _int_input("TOP_K", args.top_k),
    ]
    inputs[0].set_data_from_numpy(wav)
    outputs = [
        httpclient.InferRequestedOutput("PROJECTOR_OUT"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        httpclient.InferRequestedOutput("WORD_LIST"),
    ]
    result = client.infer(args.model, inputs, outputs=outputs)
    projector_out = result.as_numpy("PROJECTOR_OUT")
    return {
        "word_list": json.loads(_decode(result.as_numpy("WORD_LIST")[0])),
        "projector_len": int(result.as_numpy("PROJECTOR_LEN")[0]),
        "projector_out_shape": list(projector_out.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call rag_asr_retrieve via Triton HTTP"
    )
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--user", default="default", help="upstream user hotword pool id")
    sub = parser.add_subparsers(dest="command", required=True)

    status_p = sub.add_parser("status", help="show Triton readiness and hotword pool summary")
    status_p.add_argument("--query", default=None)
    status_p.add_argument("--limit", type=int, default=10)
    status_p.add_argument("--offset", type=int, default=0)
    status_p.add_argument("--verbose", action="store_true", help="include full model I/O schema")
    status_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="json",
        help="output format; bash wrapper defaults to text",
    )

    infer_p = sub.add_parser("infer", help="retrieve hotwords for one wav")
    infer_p.add_argument("--wav", required=True)
    infer_p.add_argument("--top-k", type=int, default=50)

    list_p = sub.add_parser("list", help="show existing hotword pool")
    list_p.add_argument("--query", default=None)
    list_p.add_argument("--limit", type=int, default=100)
    list_p.add_argument("--offset", type=int, default=0)

    add_p = sub.add_parser("add", help="add hotwords")
    add_p.add_argument("words", nargs="*")
    add_p.add_argument("--file", type=Path, action="append", default=[])

    import_p = sub.add_parser("import", help="bulk import hotwords from files")
    import_p.add_argument("files", type=Path, nargs="+")

    delete_p = sub.add_parser("delete", help="delete hotwords")
    delete_p.add_argument("words", nargs="*")
    delete_p.add_argument("--file", type=Path, action="append", default=[])

    sub.add_parser("reload", help="reload hotword pool from server-side file")

    args = parser.parse_args()
    if args.command == "status":
        output = _cmd_status(args)
    elif args.command == "infer":
        output = _cmd_infer(args)
    elif args.command == "list":
        output = _management_call(args, "list")
    elif args.command == "add":
        output = _management_call(
            args,
            "add",
            words=_collect_words(args.words, args.file),
        )
    elif args.command == "import":
        output = _management_call(
            args,
            "add",
            words=_collect_words([], args.files),
        )
    elif args.command == "delete":
        output = _management_call(
            args,
            "delete",
            words=_collect_words(args.words, args.file),
        )
    elif args.command == "reload":
        output = _management_call(args, "reload")
    else:
        raise AssertionError(args.command)
    if args.command == "status" and args.format == "text":
        _print_status_text(output)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

