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
TRITON_SERVER_ENV="${RAG_ASR_TRITON_CONDA_ENV:-triton}"
if ! conda activate "$TRITON_SERVER_ENV"; then
  echo "Failed to activate Triton server conda env: $TRITON_SERVER_ENV"
  echo "Create an env with tritonserver/tritonclient installed, or set RAG_ASR_TRITON_CONDA_ENV."
  exit 1
fi
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
python_stub_link = cfg.triton.python_stub_link or ""
if python_stub_link.lower() in {"", "none", "off"}:
    python_stub_link = ""
else:
    python_stub_path = Path(python_stub_link).expanduser()
    if not python_stub_path.is_absolute():
        python_stub_path = PROJECT_ROOT / python_stub_path
    python_stub_link = str(python_stub_path)

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
    "PYENV_LINK": python_stub_link,
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
  echo "Set triton.exec_env in configs/serve.yaml."
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
  echo "Expected a conda-pack archive or an existing Triton Python backend execution env with bin/python3."
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
check_port_available "metrics_port" "$TRITON_METRICS_PORT"

# Some custom Triton Python backend stubs link libpython built with a fixed prefix.
if [[ -n "$PYENV_LINK" && "$PYENV_LINK" != "none" && "$PYENV_LINK" != "off" && -d "$EXEC_ENV" ]] && \
   { [[ ! -e "$PYENV_LINK" ]] || [[ "$(readlink -f "$PYENV_LINK" 2>/dev/null)" != "$(readlink -f "$EXEC_ENV")" ]]; }; then
  echo "Creating pyenv symlink for Triton Python stub: $PYENV_LINK -> $EXEC_ENV"
  mkdir -p "$(dirname "$PYENV_LINK")"
  ln -sfn "$EXEC_ENV" "$PYENV_LINK"
fi

MODEL_REPO="$(python -m rag_asr.cli_triton_config render \
  --config "$CONFIG_PATH" \
  --output "$MODEL_REPO_RENDERED")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CONFIG_CUDA_VISIBLE_DEVICES}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1

if ! command -v tritonserver >/dev/null 2>&1; then
  echo "tritonserver is not available in conda env: $TRITON_SERVER_ENV"
  echo "Install NVIDIA Triton server in that env, or set RAG_ASR_TRITON_CONDA_ENV to the correct server env."
  exit 1
fi

if [[ -z "$TRITON_BACKEND_DIR" ]]; then
  if [[ -n "${TRITONSERVER_ROOT:-}" ]]; then
    TRITON_BACKEND_DIR="${TRITONSERVER_ROOT}/backends"
  else
    TRITONSERVER_BIN="$(command -v tritonserver)"
    TRITONSERVER_PREFIX="$(cd "$(dirname "$TRITONSERVER_BIN")/.." && pwd)"
    TRITON_BACKEND_DIR="$TRITONSERVER_PREFIX/backends"
  fi
fi

if [[ ! -d "$TRITON_BACKEND_DIR" ]]; then
  echo "Invalid Triton backend directory: $TRITON_BACKEND_DIR"
  echo "Set triton.backend_dir in $CONFIG_PATH to the directory containing the python backend."
  exit 1
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
  --grpc-port="$TRITON_GRPC_PORT" \
  --metrics-port="$TRITON_METRICS_PORT"
