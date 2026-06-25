#!/usr/bin/env bash
# Build all local environments needed by the Triton online service.
#
# This orchestrates two separate environments:
#   1. triton:      launches tritonserver and provides tritonclient
#   2. triton-exec: runs Triton Python backend model.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_step() {
  local title="$1"
  shift

  echo
  echo "==> $title"
  "$@"
}

if [[ "${RAG_ASR_SKIP_TRITON_SERVER_ENV:-0}" != "1" ]]; then
  run_step "Build Triton server env" bash "$SCRIPT_DIR/build_triton_server_env.sh"
else
  echo "Skipping Triton server env because RAG_ASR_SKIP_TRITON_SERVER_ENV=1"
fi

if [[ "${RAG_ASR_SKIP_TRITON_EXEC_ENV:-0}" != "1" ]]; then
  run_step "Build Triton Python backend execution env" bash "$SCRIPT_DIR/build_triton_exec_env.sh"
else
  echo "Skipping Triton execution env because RAG_ASR_SKIP_TRITON_EXEC_ENV=1"
fi

echo
echo "Done."
echo "  Start service: bash scripts/start_triton.sh"
