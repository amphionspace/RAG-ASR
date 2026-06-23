#!/usr/bin/env bash
# Build a conda-packed execution environment for Triton Python backend (Python 3.12).
# Output: /ai_sds_wuzz/MODELS/triton-exec-env.tar.gz
# Does NOT modify the existing ``triton`` (vllm-clone) or ``vllm`` environments.

set -euo pipefail

ENV_NAME=triton-exec
ENV_DIR="$("$CONDA" info --base)/envs/$ENV_NAME"
OUT_TAR=/ai_sds_wuzz/MODELS/triton-exec-env.tar.gz
RAG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA=/ai_sds_wuzz/MODELS/miniconda3/bin/conda
ACTIVATE_SRC="$(cd "$(dirname "$0")" && pwd)/triton_exec_activate.sh"

env_exists() {
  "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

if ! env_exists; then
  echo "Creating conda env: $ENV_NAME (python=3.12)"
  "$CONDA" create -n "$ENV_NAME" python=3.12 -y
fi

echo "Installing RAG-ASR dependencies into $ENV_NAME ..."
# Match host driver CUDA 12.8 (triton conda env uses torch 2.10+cu128).
"$CONDA" run -n "$ENV_NAME" python -m pip uninstall -y rag-asr 2>/dev/null || true
"$CONDA" run -n "$ENV_NAME" python -m pip install \
  torch==2.10.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$CONDA" run -n "$ENV_NAME" python -m pip install \
  "transformers==4.57.6" "huggingface_hub==0.36.2" \
  lhotse numpy soundfile librosa faiss-cpu fastapi uvicorn python-multipart
"$CONDA" run -n "$ENV_NAME" python -m pip install "$RAG_ROOT" --no-deps

# Triton directory execution env requires bin/activate.
install -m 0755 "$ACTIVATE_SRC" "$ENV_DIR/bin/activate"

PY_VER="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" != "3.12" ]]; then
  echo "ERROR: $ENV_NAME must be Python 3.12 (Triton stub), got $PY_VER"
  exit 1
fi

echo "Installing conda-pack ..."
"$CONDA" install -n "$ENV_NAME" -y conda-pack

# pip may clobber conda-managed pip/setuptools; restore before packing.
echo "Restoring conda-managed pip/setuptools ..."
"$CONDA" install -n "$ENV_NAME" pip setuptools -y --force-reinstall

echo "Packing environment -> $OUT_TAR"
rm -f "$OUT_TAR"
"$CONDA" run -n "$ENV_NAME" conda-pack -o "$OUT_TAR"

echo "Done."
echo "  Execution env (live): $("$CONDA" info --base)/envs/$ENV_NAME"
echo "  Portable archive:     $OUT_TAR"
echo "  Start server:         bash scripts/start_triton.sh"
