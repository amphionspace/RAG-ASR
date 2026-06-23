#!/usr/bin/env bash
# Start NVIDIA Triton with the RAG-ASR retrieval model.
# Requires: conda env ``triton`` (tritonserver on PATH) + ``triton-exec`` execution env

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_REPO="$ROOT/triton"
EXEC_ENV=/ai_sds_wuzz/MODELS/miniconda3/envs/triton-exec
EXEC_TAR=/ai_sds_wuzz/MODELS/triton-exec-env.tar.gz
PYENV_LINK=/opt/pyenv_build/versions/3.12.3
ACTIVATE_SRC="$ROOT/scripts/triton_exec_activate.sh"

if [[ ! -f "$EXEC_ENV/bin/activate" ]]; then
  install -m 0755 "$ACTIVATE_SRC" "$EXEC_ENV/bin/activate"
fi

if [[ ! -x "$EXEC_ENV/bin/python3" ]]; then
  echo "Missing $EXEC_ENV; run: bash scripts/build_triton_exec_env.sh"
  exit 1
fi

# Triton Python backend stub links libpython built with this prefix (must be Python 3.12).
if [[ ! -e "$PYENV_LINK" ]] || [[ "$(readlink -f "$PYENV_LINK" 2>/dev/null)" != "$(readlink -f "$EXEC_ENV")" ]]; then
  echo "Creating pyenv symlink for Triton Python stub: $PYENV_LINK -> $EXEC_ENV"
  mkdir -p /opt/pyenv_build/versions
  ln -sfn "$EXEC_ENV" "$PYENV_LINK"
fi

# shellcheck disable=SC1091
source /ai_sds_wuzz/MODELS/miniconda3/etc/profile.d/conda.sh
conda activate triton

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1

echo "model-repository: $MODEL_REPO"
echo "execution-env: $EXEC_ENV"
exec tritonserver \
  --model-repository="$MODEL_REPO" \
  --backend-directory="${TRITONSERVER_ROOT}/backends" \
  --allow-gpu-metrics=false \
  --http-port=8000 \
  --grpc-port=8001
