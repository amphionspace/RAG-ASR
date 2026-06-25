#!/usr/bin/env bash
# Build the conda environment used to launch NVIDIA Triton Inference Server.
# This env wraps an existing official Triton server installation.
# It is separate from triton-exec, which is the Python backend execution env.

set -euo pipefail

ENV_NAME="${RAG_ASR_TRITON_SERVER_ENV_NAME:-triton}"
PYTHON_VERSION="${RAG_ASR_TRITON_SERVER_PYTHON:-3.10}"
TRITONSERVER_ROOT_CANDIDATE="${RAG_ASR_TRITONSERVER_ROOT:-${TRITONSERVER_ROOT:-}}"

resolve_conda() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    echo "$CONDA_EXE"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    for candidate in \
      "$CONDA_PREFIX/bin/conda" \
      "$CONDA_PREFIX/../bin/conda" \
      "$CONDA_PREFIX/../../bin/conda"; do
      if [[ -x "$candidate" ]]; then
        echo "$candidate"
        return
      fi
    done
  fi

  echo "ERROR: conda is not available. Set CONDA=/path/to/conda." >&2
  exit 1
}

resolve_tritonserver_root() {
  local candidate="$1"
  if [[ -n "$candidate" && -x "$candidate/bin/tritonserver" ]]; then
    (cd "$candidate" && pwd)
    return
  fi

  if command -v tritonserver >/dev/null 2>&1; then
    local server_bin
    server_bin="$(command -v tritonserver)"
    (cd "$(dirname "$server_bin")/.." && pwd)
    return
  fi

  if [[ -x "/opt/tritonserver/bin/tritonserver" ]]; then
    echo "/opt/tritonserver"
    return
  fi

  return 1
}

if ! TRITONSERVER_ROOT_RESOLVED="$(resolve_tritonserver_root "$TRITONSERVER_ROOT_CANDIDATE")"; then
  echo "ERROR: official Triton server installation was not found." >&2
  echo "" >&2
  echo "This script creates the conda wrapper/client env, but it does not install Triton server itself." >&2
  echo "Install or mount an official Triton server distribution first, then rerun, for example:" >&2
  echo "  RAG_ASR_TRITONSERVER_ROOT=/opt/tritonserver bash scripts/build_triton_server_env.sh" >&2
  echo "" >&2
  echo "Docker/NGC Triton images are the recommended upstream installation path." >&2
  exit 1
fi

if [[ ! -d "$TRITONSERVER_ROOT_RESOLVED/backends/python" ]]; then
  echo "ERROR: Triton Python backend not found: $TRITONSERVER_ROOT_RESOLVED/backends/python" >&2
  echo "Use a Triton server build/image that includes the Python backend." >&2
  exit 1
fi

CONDA="${CONDA:-$(resolve_conda)}"
ENV_DIR="$("$CONDA" info --base)/envs/$ENV_NAME"

env_exists() {
  "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

if ! env_exists; then
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  "$CONDA" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

echo "Installing Triton client/helper packages into $ENV_NAME ..."
"$CONDA" run -n "$ENV_NAME" python -m pip install -q -U pip
"$CONDA" run -n "$ENV_NAME" python -m pip install \
  -q \
  "tritonclient[http]" \
  PyYAML

# Triton needs libpython from the active conda env to be discoverable, and the
# official server bin/lib directories must be visible after conda activation.
mkdir -p "$ENV_DIR/etc/conda/activate.d"
cat >"$ENV_DIR/etc/conda/activate.d/tritonserver.sh" <<EOF
export TRITONSERVER_ROOT="$TRITONSERVER_ROOT_RESOLVED"
export PATH="\${TRITONSERVER_ROOT}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${TRITONSERVER_ROOT}/lib:\${LD_LIBRARY_PATH:-}"
EOF

echo "Verifying tritonserver ..."
if ! "$CONDA" run -n "$ENV_NAME" bash -lc \
  'source "$CONDA_PREFIX/etc/conda/activate.d/tritonserver.sh"; command -v tritonserver >/dev/null'; then
  echo "ERROR: tritonserver is not on PATH after writing the activation hook." >&2
  exit 1
fi

TRITONSERVER_BIN="$TRITONSERVER_ROOT_RESOLVED/bin/tritonserver"
BACKEND_DIR="$TRITONSERVER_ROOT_RESOLVED/backends"

echo "Done."
echo "  Server env:       $ENV_DIR"
echo "  Triton root:      $TRITONSERVER_ROOT_RESOLVED"
echo "  tritonserver:     $TRITONSERVER_BIN"
echo "  Backend dir:      $BACKEND_DIR"
echo "  Start service:    bash scripts/start_triton.sh"
