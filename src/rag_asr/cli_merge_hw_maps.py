"""CLI entry point for merging retrieval shard JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_asr.infer import write_recall_summary_txt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge retrieve shard JSONL files",
    )
    parser.add_argument("shards", nargs="+", help="Paths like hw_map_zh.shard0.jsonl")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--top-k", type=int, default=None, help="Recorded in recall summary")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_written = 0
    merged: dict[str, dict] = {}

    with open(out, "w", encoding="utf-8") as fout:
        for path in sorted(args.shards):
            pth = Path(path)
            if not pth.exists():
                print(f"missing shard: {pth}", file=sys.stderr)
                sys.exit(1)
            n_shard = 0
            with open(pth, encoding="utf-8") as fin:
                for lineno, line in enumerate(fin, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"{pth}:{lineno}: invalid JSON: {e}", file=sys.stderr)
                        sys.exit(1)
                    uid = rec.get("id")
                    if not uid:
                        print(f"{pth}:{lineno}: missing 'id'", file=sys.stderr)
                        sys.exit(1)
                    if uid in seen:
                        print(f"warning: duplicate utt id {uid} in {pth}", file=sys.stderr)
                        continue
                    seen.add(uid)
                    fout.write(line + "\n")
                    merged[uid] = rec
                    n_shard += 1
                    n_written += 1
            print(f"  + {pth.name}: {n_shard} utts")

    print(f"merged {n_written} utterances -> {out}")

    summary_path = out.with_suffix(".recall.txt")
    write_recall_summary_txt(summary_path, merged, top_k=args.top_k)
    print(f"wrote recall summary -> {summary_path}")


if __name__ == "__main__":
    main()
