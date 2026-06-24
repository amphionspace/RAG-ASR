#!/usr/bin/env bash
# Quick local HTTP service (conda ``triton`` env) — same logic as Triton model.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${RAG_ASR_CONFIG:-$ROOT/configs/serve.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH"
  echo "Create configs/serve.yaml or set RAG_ASR_CONFIG."
  exit 1
fi

if [[ -n "${CONDA_SH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda is not available. Set CONDA_SH=/path/to/conda.sh."
  exit 1
fi
conda activate "${RAG_ASR_TRITON_CONDA_ENV:-triton}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

exec python "$ROOT/scripts/serve_http.py" \
  --config "$CONFIG_PATH" \
  --port "${PORT:-8080}" \
  "$@"
