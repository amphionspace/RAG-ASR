#!/usr/bin/env python3
"""Triton client for RAG-ASR retrieval and hotword management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    inputs = [_string_input("ACTION", action)]
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
    result = client.infer(MODEL_NAME, inputs, outputs=outputs)
    message = json.loads(_decode(result.as_numpy("MESSAGE")[0]))
    hotwords = json.loads(_decode(result.as_numpy("HOTWORD_LIST")[0]))
    message["status"] = _decode(result.as_numpy("STATUS")[0])
    message["hotword_count"] = int(result.as_numpy("HOTWORD_COUNT")[0])
    message["hotwords"] = hotwords
    return message


def _cmd_infer(args) -> dict:
    import tritonclient.http as httpclient

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    client = httpclient.InferenceServerClient(url=args.url)
    inputs = [
        httpclient.InferInput("WAV", wav.shape, "FP32"),
        _int_input("SAMPLE_RATE", int(sr)),
        _int_input("TOP_K", args.top_k),
    ]
    inputs[0].set_data_from_numpy(wav)
    outputs = [
        httpclient.InferRequestedOutput("PROJECTOR_OUT"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        httpclient.InferRequestedOutput("WORD_LIST"),
    ]
    result = client.infer(MODEL_NAME, inputs, outputs=outputs)
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
    sub = parser.add_subparsers(dest="command", required=True)

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
    if args.command == "infer":
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
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

