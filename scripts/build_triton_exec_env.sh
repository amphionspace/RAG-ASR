#!/usr/bin/env bash
# Build a conda-packed execution environment for Triton Python backend.
# Python version must match the Triton Python backend stub.
# Override output with RAG_ASR_TRITON_EXEC_ENV_TAR when a shared archive is needed.
# Does NOT modify the existing ``triton`` (vllm-clone) or ``vllm`` environments.

set -euo pipefail

RAG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${RAG_ASR_TRITON_EXEC_ENV_NAME:-triton-exec}"
PYTHON_VERSION="${RAG_ASR_TRITON_EXEC_PYTHON:-3.10}"
OUT_TAR="${RAG_ASR_TRITON_EXEC_ENV_TAR:-$RAG_ROOT/var/triton-exec-env.tar.gz}"
ACTIVATE_SRC="$(cd "$(dirname "$0")" && pwd)/triton_exec_activate.sh"

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

CONDA="${CONDA:-$(resolve_conda)}"
ENV_DIR="$("$CONDA" info --base)/envs/$ENV_NAME"

env_exists() {
  "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

if ! env_exists; then
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  "$CONDA" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

PY_VER="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" != "$PYTHON_VERSION" ]]; then
  echo "ERROR: $ENV_NAME must be Python $PYTHON_VERSION (Triton stub), got $PY_VER"
  echo "Use another RAG_ASR_TRITON_EXEC_ENV_NAME, or recreate this env with the matching Python version."
  exit 1
fi

echo "Installing RAG-ASR dependencies into $ENV_NAME ..."
# Match host driver CUDA 12.8 (triton conda env uses torch 2.10+cu128).
"$CONDA" run -n "$ENV_NAME" python -m pip uninstall -y rag-asr 2>/dev/null || true
"$CONDA" run -n "$ENV_NAME" python -m pip install \
  torch==2.10.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$CONDA" run -n "$ENV_NAME" python -m pip install \
  "transformers==4.57.6" "huggingface_hub==0.36.2" \
  lhotse "numpy<2" soundfile librosa faiss-cpu fastapi uvicorn python-multipart
"$CONDA" run -n "$ENV_NAME" python -m pip install "$RAG_ROOT" --no-deps

# Triton directory execution env requires bin/activate.
install -m 0755 "$ACTIVATE_SRC" "$ENV_DIR/bin/activate"

echo "Installing conda-pack ..."
"$CONDA" install -n "$ENV_NAME" -y conda-pack

# pip may clobber conda-managed pip/setuptools; restore before packing.
echo "Restoring conda-managed pip/setuptools ..."
"$CONDA" install -n "$ENV_NAME" pip setuptools -y --force-reinstall

echo "Packing environment -> $OUT_TAR"
mkdir -p "$(dirname "$OUT_TAR")"
rm -f "$OUT_TAR"
"$CONDA" run -n "$ENV_NAME" conda-pack -o "$OUT_TAR"

echo "Done."
echo "  Execution env (live): $("$CONDA" info --base)/envs/$ENV_NAME"
echo "  Portable archive:     $OUT_TAR"
echo "  Start server:         bash scripts/start_triton.sh"
