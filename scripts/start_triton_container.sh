#!/usr/bin/env bash
# Start NVIDIA Triton inside a container image that already includes
# /opt/tritonserver and the packed RAG-ASR Python backend execution env.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${RAG_ASR_CONFIG:-$ROOT/configs/serve.compose.yaml}"
PYTHON_BIN="${RAG_ASR_CONTAINER_PYTHON:-$ROOT/triton-exec/bin/python}"
TRITONSERVER_BIN="${TRITONSERVER_BIN:-/opt/tritonserver/bin/tritonserver}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Invalid container Python: $PYTHON_BIN"
  echo "Set RAG_ASR_CONTAINER_PYTHON to the Python executable with rag_asr installed."
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH"
  echo "Set RAG_ASR_CONFIG to a container-readable service config."
  exit 1
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

eval "$(
"$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

from rag_asr.config import PROJECT_ROOT, load_config
from rag_asr.model_layout import resolve_hotword_adapter

cfg = load_config(sys.argv[1])
params = cfg.to_triton_parameters()

rendered = Path(cfg.triton.rendered_model_repo)
if not rendered.is_absolute():
    rendered = PROJECT_ROOT / rendered

backend_dir = cfg.triton.backend_dir or "/opt/tritonserver/backends"

values = {
    "EXEC_ENV": params["EXECUTION_ENV_PATH"],
    "BASE_MODEL_PATH": params["base_model_path"],
    "HOTWORD_POOL_FILE": params["hotword_pool_file"],
    "HOTWORD_POOL_DIR": params["hotword_pool_dir"],
    "SEED_POOL_FILE": params["seed_pool_file"],
    "HOTWORD_ADAPTER_PATH": str(
        resolve_hotword_adapter(
            params["base_model_path"],
            params.get("adapter_ckpt") or None,
            adapter_subdir=params["adapter_subdir"],
            adapter_filename=params["adapter_filename"],
            must_exist=False,
        )
    ),
    "MODEL_REPO_RENDERED": str(rendered),
    "TRITON_BACKEND_DIR": backend_dir,
    "TRITON_HTTP_PORT": str(cfg.triton.http_port),
    "TRITON_GRPC_PORT": str(cfg.triton.grpc_port),
    "TRITON_METRICS_PORT": str(cfg.triton.metrics_port),
    "CONFIG_CUDA_VISIBLE_DEVICES": cfg.runtime.cuda_visible_devices,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

if [[ -z "$EXEC_ENV" ]]; then
  echo "Missing triton.exec_env in $CONFIG_PATH."
  exit 1
fi

if [[ -f "$EXEC_ENV" ]]; then
  case "$EXEC_ENV" in
    *.tar|*.tar.gz|*.tgz) ;;
    *)
      echo "Invalid triton.exec_env archive: $EXEC_ENV"
      echo "Expected a conda-pack archive ending in .tar, .tar.gz, or .tgz."
      exit 1
      ;;
  esac
elif [[ ! -x "$EXEC_ENV/bin/python3" ]]; then
  echo "Invalid triton.exec_env: $EXEC_ENV"
  echo "Expected a conda-pack archive or execution env directory with bin/python3."
  exit 1
fi

if [[ -z "$BASE_MODEL_PATH" || ! -d "$BASE_MODEL_PATH" ]]; then
  echo "Invalid model.base_model_path: $BASE_MODEL_PATH"
  exit 1
fi

if [[ -z "$HOTWORD_ADAPTER_PATH" || ! -f "$HOTWORD_ADAPTER_PATH" ]]; then
  echo "Invalid hotword adapter: $HOTWORD_ADAPTER_PATH"
  echo "Expected base_model_path/hotword_adapter/{pooled,frame_max}/best_adapter.pt,"
  echo "set model.adapter_ckpt, or configure model.adapter_ckpts for retrieval.retrieval_mode."
  exit 1
fi

if [[ -z "$HOTWORD_POOL_DIR" ]]; then
  echo "Missing retrieval.hotword_pool_dir in $CONFIG_PATH."
  exit 1
fi
mkdir -p "$HOTWORD_POOL_DIR"

if [[ -n "$SEED_POOL_FILE" && ! -f "$SEED_POOL_FILE" ]]; then
  echo "Invalid retrieval.seed_pool_file: $SEED_POOL_FILE"
  exit 1
fi

if [[ -n "$HOTWORD_POOL_FILE" && ! -f "$HOTWORD_POOL_FILE" ]]; then
  echo "Invalid retrieval.hotword_pool_file: $HOTWORD_POOL_FILE"
  echo "Omit retrieval.hotword_pool_file to use retrieval.hotword_pool_dir/default_user.txt."
  exit 1
fi

if [[ ! -d "$TRITON_BACKEND_DIR" ]]; then
  echo "Invalid Triton backend directory: $TRITON_BACKEND_DIR"
  exit 1
fi

if [[ ! -x "$TRITONSERVER_BIN" ]]; then
  echo "Invalid tritonserver binary: $TRITONSERVER_BIN"
  exit 1
fi

MODEL_REPO="$("$PYTHON_BIN" -m rag_asr.cli_triton_config render \
  --config "$CONFIG_PATH" \
  --output "$MODEL_REPO_RENDERED")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CONFIG_CUDA_VISIBLE_DEVICES}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1

echo "config: $CONFIG_PATH"
echo "model-repository: $MODEL_REPO"
echo "execution-env: $EXEC_ENV"
exec "$TRITONSERVER_BIN" \
  --model-repository="$MODEL_REPO" \
  --backend-directory="$TRITON_BACKEND_DIR" \
  --allow-gpu-metrics=false \
  --disable-auto-complete-config \
  --http-port="$TRITON_HTTP_PORT" \
  --grpc-port="$TRITON_GRPC_PORT" \
  --metrics-port="$TRITON_METRICS_PORT"
