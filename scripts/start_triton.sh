#!/usr/bin/env bash
# Start NVIDIA Triton with the RAG-ASR retrieval model.
# Requires: conda env ``triton`` (tritonserver on PATH) + ``triton-exec`` execution env

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
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

eval "$(
python - "$CONFIG_PATH" <<'PY'
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

backend_dir = cfg.triton.backend_dir or ""

values = {
    "EXEC_ENV": params["EXECUTION_ENV_PATH"],
    "BASE_MODEL_PATH": params["base_model_path"],
    "HOTWORD_POOL_FILE": params["hotword_pool_file"],
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
    "PYENV_LINK": cfg.triton.python_stub_link,
    "TRITON_BACKEND_DIR": backend_dir,
    "TRITON_HTTP_PORT": str(cfg.triton.http_port),
    "TRITON_GRPC_PORT": str(cfg.triton.grpc_port),
    "CONFIG_CUDA_VISIBLE_DEVICES": cfg.runtime.cuda_visible_devices,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

if [[ -z "$EXEC_ENV" ]]; then
  echo "Missing triton.exec_env in $CONFIG_PATH."
  echo "Set triton.exec_env in configs/serve.yaml."
  exit 1
fi

if [[ ! -x "$EXEC_ENV/bin/python3" ]]; then
  echo "Invalid triton.exec_env: $EXEC_ENV"
  echo "Expected an existing Triton Python backend execution env with bin/python3."
  echo "Create it with: bash scripts/build_triton_exec_env.sh"
  exit 1
fi

if [[ -z "$BASE_MODEL_PATH" ]]; then
  echo "Missing model.base_model_path in $CONFIG_PATH."
  exit 1
fi
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
  echo "Invalid model.base_model_path: $BASE_MODEL_PATH"
  exit 1
fi

if [[ -z "$HOTWORD_ADAPTER_PATH" ]]; then
  echo "Missing hotword adapter path in $CONFIG_PATH."
  exit 1
fi
if [[ ! -f "$HOTWORD_ADAPTER_PATH" ]]; then
  echo "Invalid hotword adapter: $HOTWORD_ADAPTER_PATH"
  echo "Expected base_model_path/hotword_adapter/best_adapter.pt or set model.adapter_ckpt."
  exit 1
fi

if [[ -z "$HOTWORD_POOL_FILE" ]]; then
  echo "Missing retrieval.hotword_pool_file in $CONFIG_PATH."
  echo "Set retrieval.hotword_pool_file directly or export RAG_ASR_HOTWORD_POOL."
  exit 1
fi
if [[ ! -f "$HOTWORD_POOL_FILE" ]]; then
  echo "Invalid retrieval.hotword_pool_file: $HOTWORD_POOL_FILE"
  exit 1
fi

check_port_available() {
  local config_key="$1"
  local port="$2"
  if [[ -z "$port" ]]; then
    echo "Missing triton.$config_key in $CONFIG_PATH."
    exit 1
  fi
  if ! python - "$config_key" "$port" "$CONFIG_PATH" <<'PY'
import socket
import sys

config_key, port_text, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    port = int(port_text)
except ValueError:
    print(f"Invalid triton.{config_key}: {port_text}")
    sys.exit(1)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        print(f"Port unavailable for triton.{config_key}: {port} ({exc})")
        print(f"Change triton.{config_key} in {config_path} or stop the process using this port.")
        sys.exit(1)
PY
  then
    exit 1
  fi
}

check_port_available "http_port" "$TRITON_HTTP_PORT"
check_port_available "grpc_port" "$TRITON_GRPC_PORT"

# Triton Python backend stub links libpython built with this prefix (must be Python 3.12).
if [[ ! -e "$PYENV_LINK" ]] || [[ "$(readlink -f "$PYENV_LINK" 2>/dev/null)" != "$(readlink -f "$EXEC_ENV")" ]]; then
  echo "Creating pyenv symlink for Triton Python stub: $PYENV_LINK -> $EXEC_ENV"
  mkdir -p /opt/pyenv_build/versions
  ln -sfn "$EXEC_ENV" "$PYENV_LINK"
fi

MODEL_REPO="$(python -m rag_asr.cli_triton_config render \
  --config "$CONFIG_PATH" \
  --output "$MODEL_REPO_RENDERED")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CONFIG_CUDA_VISIBLE_DEVICES}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1

if [[ -z "$TRITON_BACKEND_DIR" ]]; then
  TRITON_BACKEND_DIR="${TRITONSERVER_ROOT}/backends"
fi

echo "config: $CONFIG_PATH"
echo "model-repository: $MODEL_REPO"
echo "execution-env: $EXEC_ENV"
exec tritonserver \
  --model-repository="$MODEL_REPO" \
  --backend-directory="$TRITON_BACKEND_DIR" \
  --allow-gpu-metrics=false \
  --disable-auto-complete-config \
  --http-port="$TRITON_HTTP_PORT" \
  --grpc-port="$TRITON_GRPC_PORT"
