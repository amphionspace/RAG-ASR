#!/usr/bin/env python3
"""End-to-end smoke test: Triton ``rag_asr_retrieve`` on ``examples/`` clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_asr.infer import aggregate_recall_stats, compute_hotword_recall


def load_examples(examples_dir: Path) -> list[dict]:
    meta = examples_dir / "metadata.jsonl"
    if not meta.is_file():
        raise FileNotFoundError(f"missing {meta}")
    rows = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def infer_triton(
    *,
    url: str,
    wav_path: Path,
    sample_rate: int,
    top_k: int,
) -> dict:
    import tritonclient.http as httpclient

    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != sample_rate:
        raise ValueError(f"{wav_path}: expected {sample_rate} Hz, got {sr}")

    client = httpclient.InferenceServerClient(url=url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton not ready at {url}")
    if not client.is_model_ready("rag_asr_retrieve"):
        raise RuntimeError("model rag_asr_retrieve is not READY")

    inputs = [
        httpclient.InferInput("WAV", wav.shape, "FP32"),
        httpclient.InferInput("SAMPLE_RATE", [1], "INT32"),
        httpclient.InferInput("TOP_K", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(wav)
    inputs[1].set_data_from_numpy(np.array([sample_rate], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([top_k], dtype=np.int32))

    outputs = [
        httpclient.InferRequestedOutput("PROJECTOR_OUT"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        httpclient.InferRequestedOutput("WORD_LIST"),
    ]
    result = client.infer("rag_asr_retrieve", inputs, outputs=outputs)

    word_list = json.loads(result.as_numpy("WORD_LIST")[0].decode())
    projector_len = int(result.as_numpy("PROJECTOR_LEN")[0])
    projector_out = result.as_numpy("PROJECTOR_OUT")
    return {
        "word_list": word_list,
        "projector_len": projector_len,
        "projector_out_shape": list(projector_out.shape),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Test Triton rag_asr_retrieve on examples/")
    p.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint")
    p.add_argument("--examples-dir", type=Path, default=ROOT / "examples")
    p.add_argument("--wav", type=Path, default=None, help="Run a single wav instead of all examples")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = p.parse_args()

    examples_dir = args.examples_dir.resolve()
    if args.wav is not None:
        wav_path = args.wav.resolve()
        ex = {
            "id": wav_path.stem,
            "wav": str(wav_path.relative_to(examples_dir))
            if wav_path.is_relative_to(examples_dir)
            else str(wav_path),
            "text": "",
            "hotwords": [],
        }
        hotwords_path = examples_dir / "hotwords.tsv"
        if hotwords_path.is_file():
            for line in hotwords_path.read_text(encoding="utf-8").splitlines():
                uid, hw_json = line.split("\t", 1)
                if uid == ex["id"]:
                    ex["hotwords"] = json.loads(hw_json)
                    break
        transcripts_path = examples_dir / "transcripts.tsv"
        if transcripts_path.is_file():
            for line in transcripts_path.read_text(encoding="utf-8").splitlines():
                uid, text = line.split("\t", 1)
                if uid == ex["id"]:
                    ex["text"] = text
                    break
        rows = [ex]
        wav_paths = [wav_path]
    else:
        rows = load_examples(examples_dir)
        wav_paths = [examples_dir / row["wav"] for row in rows]

    hw_map: dict[str, dict] = {}
    details = []

    for row, wav_path in zip(rows, wav_paths):
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)

        out = infer_triton(
            url=args.url,
            wav_path=wav_path,
            sample_rate=args.sample_rate,
            top_k=args.top_k,
        )
        retrieved = out["word_list"]
        real = list(row.get("hotwords") or [])
        recall = compute_hotword_recall(real, retrieved)
        hit = sorted(set(real) & set(retrieved)) if real else []

        hw_map[row["id"]] = {"real": real, "retrieved": retrieved}
        item = {
            "id": row["id"],
            "text": row.get("text", ""),
            "real_hotwords": real,
            "retrieved_top": retrieved[:10],
            "hit": hit,
            "recall": recall,
            "projector_len": out["projector_len"],
            "projector_out_shape": out["projector_out_shape"],
        }
        details.append(item)

        if not args.json:
            recall_s = "n/a" if recall is None else f"{recall * 100:.1f}%"
            print(f"\n=== {row['id']} ===")
            print(f"text:      {row.get('text', '')}")
            print(f"real:      {real}")
            print(f"hit:       {hit}")
            print(f"recall@K:  {recall_s}")
            print(f"top-10:    {retrieved[:10]}")
            print(f"projector: len={out['projector_len']} shape={out['projector_out_shape']}")

    stats = aggregate_recall_stats(hw_map)
    summary = {
        "url": args.url,
        "top_k": args.top_k,
        "n_examples": len(details),
        "recall_at_k": stats["recall_at_k"],
        "prrr": stats["prrr"],
        "details": details,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("\n=== summary ===")
        print(f"examples:   {stats['n_total']}")
        print(f"recall@K:   {stats['recall_at_k']:.2f}%")
        print(f"PRRR:       {stats['prrr']:.2f}%")
        print(f"zero_hit:   {stats['n_zero_hit']}")


if __name__ == "__main__":
    main()
