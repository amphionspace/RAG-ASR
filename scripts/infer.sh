#!/usr/bin/env bash
# 双塔检索推理 — bash scripts/infer.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}

GPUS=0,1,2,3
ROOT=$(pwd)
BASE="${ROOT}/checkpoints/base/amphion_1.7b_merged"
ADAPTER="${ROOT}/checkpoints/adapters/amphion-1.7b_retrieval_v1.2/best_adapter.pt"
OUT="${ROOT}/exp/infer"
mkdir -p "$OUT"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}

_run() {
  local lang=$1 pool=$2 sup=$3 rec=$4
  local out="${OUT}/hw_map_${lang}.jsonl"

  if [[ $N -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python3 scripts/retrieve.py \
      --base-model-path "$BASE" --adapter-ckpt "$ADAPTER" \
      --embed-dim 512 --adapter-hidden-dim 512 \
      --hotword-pool-file "$pool" --supervisions "$sup" --recordings "$rec" \
      --top-k 50 --output "$out"
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
      --embed-dim 512 --adapter-hidden-dim 512 \
      --hotword-pool-file "$pool" --supervisions "$sup" --recordings "$rec" \
      --shard-id "$i" --num-shards "$N" --top-k 50 --output "$s" &
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
  python3 scripts/merge_hw_maps.py "${shards[@]}" -o "$out" --top-k 50
  rm -f "${shards[@]}"
}

_run zh /chenmingjie/lx/data/hotword/zh/zh-10k.txt \
  /ai_sds_wuzz/DATA_ASR/LHOTSE/common_voice_zh/data/manifests/cv-zh-CN_supervisions_test_punc_hotwords.jsonl.gz \
  /ai_sds_wuzz/DATA_ASR/LHOTSE/common_voice_zh/data/manifests/cv-zh-CN_recordings_test.jsonl.gz

_run en /chenmingjie/lx/data/hotword/en/en-10k.txt \
  /ai_sds_wuzz/DATA_ASR/common_voice_en/lhotse/hotwords/cv-en_supervisions_test_orig_punc_hotwords.jsonl.gz \
  /ai_sds_wuzz/DATA_ASR/common_voice_en/lhotse/hotwords/cv-en_recordings_test.jsonl.gz
