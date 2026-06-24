#!/usr/bin/env bash
# 双塔检索推理 — bash scripts/infer.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${RAG_ASR_INFER_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
BASE="${RAG_ASR_BASE_MODEL:-$ROOT/checkpoints/base/amphion_1.7b_merged}"
ADAPTER="${RAG_ASR_ADAPTER:-$BASE/hotword_adapter/best_adapter.pt}"
OUT="${RAG_ASR_INFER_OUT:-$ROOT/exp/infer}"
TOP_K="${RAG_ASR_TOP_K:-50}"
EMBED_DIM="${RAG_ASR_EMBED_DIM:-512}"
ADAPTER_HIDDEN_DIM="${RAG_ASR_ADAPTER_HIDDEN_DIM:-512}"
DATA_ROOT="${RAG_ASR_DATA_ROOT:-}"
HOTWORD_ROOT="${RAG_ASR_HOTWORD_ROOT:-${DATA_ROOT:+$DATA_ROOT/hotword}}"
CV_ZH_DIR="${RAG_ASR_CV_ZH_INFER_DIR:-${RAG_ASR_CV_ZH_MANIFEST_DIR:-${DATA_ROOT:+$DATA_ROOT/LHOTSE/common_voice_zh/data/manifests}}}"
CV_EN_DIR="${RAG_ASR_CV_EN_INFER_DIR:-${RAG_ASR_CV_EN_MANIFEST_DIR:-${DATA_ROOT:+$DATA_ROOT/common_voice_en/lhotse/hotwords}}}"
ZH_HOTWORD_POOL="${RAG_ASR_ZH_HOTWORD_POOL:-${HOTWORD_ROOT:+$HOTWORD_ROOT/zh/zh-10k.txt}}"
EN_HOTWORD_POOL="${RAG_ASR_EN_HOTWORD_POOL:-${HOTWORD_ROOT:+$HOTWORD_ROOT/en/en-10k.txt}}"
ZH_SUPERVISIONS="${RAG_ASR_ZH_SUPERVISIONS:-${CV_ZH_DIR:+$CV_ZH_DIR/cv-zh-CN_supervisions_test_punc_hotwords.jsonl.gz}}"
ZH_RECORDINGS="${RAG_ASR_ZH_RECORDINGS:-${CV_ZH_DIR:+$CV_ZH_DIR/cv-zh-CN_recordings_test.jsonl.gz}}"
EN_SUPERVISIONS="${RAG_ASR_EN_SUPERVISIONS:-${CV_EN_DIR:+$CV_EN_DIR/cv-en_supervisions_test_orig_punc_hotwords.jsonl.gz}}"
EN_RECORDINGS="${RAG_ASR_EN_RECORDINGS:-${CV_EN_DIR:+$CV_EN_DIR/cv-en_recordings_test.jsonl.gz}}"
mkdir -p "$OUT"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}

require_file() {
  local label=$1 path=$2 hint=$3
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "Missing $label: ${path:-<unset>}" >&2
    echo "Set $hint, or set RAG_ASR_DATA_ROOT/RAG_ASR_HOTWORD_ROOT for the default layout." >&2
    exit 1
  fi
}

require_dir() {
  local label=$1 path=$2 hint=$3
  if [[ -z "$path" || ! -d "$path" ]]; then
    echo "Missing $label: ${path:-<unset>}" >&2
    echo "Set $hint, or set RAG_ASR_DATA_ROOT/RAG_ASR_HOTWORD_ROOT for the default layout." >&2
    exit 1
  fi
}

if [[ "$N" -lt 1 || -z "${GPU_ARR[0]:-}" ]]; then
  echo "Missing GPU list. Set RAG_ASR_INFER_GPUS or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

require_dir "base model" "$BASE" "RAG_ASR_BASE_MODEL"
require_file "adapter checkpoint" "$ADAPTER" "RAG_ASR_ADAPTER"
require_file "Chinese hotword pool" "$ZH_HOTWORD_POOL" "RAG_ASR_ZH_HOTWORD_POOL"
require_file "Chinese supervisions" "$ZH_SUPERVISIONS" "RAG_ASR_ZH_SUPERVISIONS or RAG_ASR_CV_ZH_INFER_DIR"
require_file "Chinese recordings" "$ZH_RECORDINGS" "RAG_ASR_ZH_RECORDINGS or RAG_ASR_CV_ZH_INFER_DIR"
require_file "English hotword pool" "$EN_HOTWORD_POOL" "RAG_ASR_EN_HOTWORD_POOL"
require_file "English supervisions" "$EN_SUPERVISIONS" "RAG_ASR_EN_SUPERVISIONS or RAG_ASR_CV_EN_INFER_DIR"
require_file "English recordings" "$EN_RECORDINGS" "RAG_ASR_EN_RECORDINGS or RAG_ASR_CV_EN_INFER_DIR"

_run() {
  local lang=$1 pool=$2 sup=$3 rec=$4
  local out="${OUT}/hw_map_${lang}.jsonl"

  if [[ $N -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python3 scripts/retrieve.py \
      --base-model-path "$BASE" --adapter-ckpt "$ADAPTER" \
      --embed-dim "$EMBED_DIM" --adapter-hidden-dim "$ADAPTER_HIDDEN_DIM" \
      --hotword-pool-file "$pool" --supervisions "$sup" --recordings "$rec" \
      --top-k "$TOP_K" --output "$out"
    return
  fi

  # Must be set before Python starts (OMP read at import time).
  _cpu=$(( $(nproc) / N ))
  (( _cpu < 1 )) && _cpu=1
  (( _cpu > 8 )) && _cpu=8
  export OMP_NUM_THREADS="$_cpu" MKL_NUM_THREADS="$_cpu" \
         OPENBLAS_NUM_THREADS="$_cpu" NUMEXPR_NUM_THREADS="$_cpu"

  shards=() pids=()
  _kill_shards() {
    if ((${#pids[@]})); then
      kill "${pids[@]}" 2>/dev/null || true
    fi
  }
  trap '_kill_shards; exit 130' INT TERM

  for ((i=0; i<N; i++)); do
    s="${OUT}/hw_map_${lang}.shard${i}.jsonl"
    shards+=("$s")
    CUDA_VISIBLE_DEVICES="${GPU_ARR[i]}" python3 scripts/retrieve.py \
      --base-model-path "$BASE" --adapter-ckpt "$ADAPTER" \
      --embed-dim "$EMBED_DIM" --adapter-hidden-dim "$ADAPTER_HIDDEN_DIM" \
      --hotword-pool-file "$pool" --supervisions "$sup" --recordings "$rec" \
      --shard-id "$i" --num-shards "$N" --top-k "$TOP_K" --output "$s" &
    pids+=($!)
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
  done
  trap - INT TERM
  if (( status != 0 )); then
    return "$status"
  fi
  python3 scripts/merge_hw_maps.py "${shards[@]}" -o "$out" --top-k "$TOP_K"
  rm -f "${shards[@]}"
}

_run zh "$ZH_HOTWORD_POOL" "$ZH_SUPERVISIONS" "$ZH_RECORDINGS"

_run en "$EN_HOTWORD_POOL" "$EN_SUPERVISIONS" "$EN_RECORDINGS"
