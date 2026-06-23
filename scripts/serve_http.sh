#!/usr/bin/env bash
# Quick local HTTP service (conda ``triton`` env) — same logic as Triton model.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /ai_sds_wuzz/MODELS/miniconda3/etc/profile.d/conda.sh
conda activate triton

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python "$ROOT/scripts/serve_http.py" \
  --base-model-path "$ROOT/checkpoints/base/amphion_1.7b_merged" \
  --adapter-ckpt "$ROOT/checkpoints/adapters/amphion-1.7b_retrieval_v1.2/best_adapter.pt" \
  --hotword-pool-file /chenmingjie/lx/data/hotword/zh/zh-10k.txt \
  --cache-dir "$ROOT/_retrieve_cache" \
  --port "${PORT:-8080}" \
  "$@"
