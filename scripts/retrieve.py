#!/usr/bin/env python3
"""CLI: run neural hotword retrieval on Lhotse manifests."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _configure_cpu_threads(num_shards: int) -> int:
    """Avoid N processes each defaulting to 64 BLAS threads (kills throughput)."""
    if num_shards <= 1:
        return 0
    # GPU does the heavy lifting; keep CPU threads low per process.
    n_threads = max(1, min(8, (os.cpu_count() or 8) // num_shards))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n_threads)
    return n_threads


def _load_items_from_lhotse(
    sup_paths: list[str],
    rec_paths: list[str],
    limit: int | None,
    *,
    shard_id: int = 0,
    num_shards: int = 1,
):
    from lhotse import CutSet
    from lhotse.audio import RecordingSet
    from lhotse.supervision import SupervisionSet

    sups = recs = None
    for s, r in zip(sup_paths, rec_paths):
        s_part = SupervisionSet.from_jsonl_lazy(s)
        r_part = RecordingSet.from_jsonl_lazy(r)
        sups = s_part if sups is None else sups + s_part
        recs = r_part if recs is None else recs + r_part

    cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)
    cuts = cuts.trim_to_supervisions(keep_overlapping=False)
    if limit:
        cuts = cuts.subset(first=limit)

    items: list[dict] = []
    for idx, cut in enumerate(cuts):
        if num_shards > 1 and (idx % num_shards) != shard_id:
            continue
        sup = cut.supervisions[0]
        custom = getattr(sup, "custom", None) or {}
        items.append({
            "id": cut.id,
            "mixed_audio": cut.recording.sources[0].source,
            "start": cut.start,
            "duration": cut.duration,
            "hotwords": list(custom.get("hotwords") or []),
        })
    return items


def main():
    p = argparse.ArgumentParser(description="Neural dual-tower hotword retrieval")
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-ckpt", required=True)
    p.add_argument("--hotword-pool-file", required=True)
    p.add_argument("--supervisions", nargs="+", required=True)
    p.add_argument("--recordings", nargs="+", required=True)
    p.add_argument("--output", required=True, help="Output JSONL path for hw_map")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--adapter-hidden-dim", type=int, default=512,
                   help="Must match training (default 512).")
    p.add_argument("--cache-dir", default="_retrieve_cache")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-id", type=int, default=0,
                   help="Shard index in [0, num_shards). Default 0.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total shard count (= number of GPUs for data parallel).")
    args = p.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(
            f"shard_id must be in [0, {args.num_shards}), got {args.shard_id}"
        )

    n_threads = _configure_cpu_threads(args.num_shards)
    if n_threads:
        import torch
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        logger.info(
            "shard %d/%d: CPU threads capped to %d per process",
            args.shard_id, args.num_shards, n_threads,
        )

    from rag_asr.infer import retrieve_neural, write_hw_map_jsonl, write_recall_summary_txt

    items = _load_items_from_lhotse(
        args.supervisions,
        args.recordings,
        args.limit,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    if args.num_shards > 1:
        logger.info("shard %d/%d: %d items", args.shard_id, args.num_shards, len(items))
    else:
        logger.info("loaded %d items", len(items))

    cache_dir = None if args.cache_dir.lower() in {"none", "off"} else args.cache_dir
    hw_map = retrieve_neural(
        items,
        [],
        args.top_k,
        adapter_ckpt=args.adapter_ckpt,
        base_model_path=args.base_model_path,
        embed_dim=args.embed_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        hotword_pool_file=args.hotword_pool_file,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    out = Path(args.output)
    write_hw_map_jsonl(out, hw_map)
    logger.info("wrote %s (%d utterances)", out, len(hw_map))

    if args.num_shards <= 1:
        summary_path = out.with_suffix(".recall.txt")
        write_recall_summary_txt(summary_path, hw_map, top_k=args.top_k)
        logger.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
