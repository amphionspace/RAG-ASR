#!/usr/bin/env bash
# RAG-ASR 双塔检索训练脚本
#
# 用法:
#   bash scripts/train_retrieval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# -----------------------------------------------------------------------
# 路径配置（按环境修改）
# -----------------------------------------------------------------------
DATA_ROOT="${RAG_ASR_DATA_ROOT:-}"
BASE_MODEL="${RAG_ASR_BASE_MODEL:-$ROOT/checkpoints/base/amphion_1.7b_merged}"
HOTWORD_DIR="${RAG_ASR_HOTWORD_ROOT:-${DATA_ROOT:+$DATA_ROOT/hotword}}"
TRAIN_V1="${RAG_ASR_TRAIN_V1:-${HOTWORD_DIR:+$HOTWORD_DIR/train_v1}}"
CV_EN_DIR="${RAG_ASR_CV_EN_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/common_voice_en/lhotse/hotwords}}"
CV_ZH_DIR="${RAG_ASR_CV_ZH_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/common_voice_zh/lhotse/hotwords}}"
GIGASPEECH_DIR="${RAG_ASR_GIGASPEECH_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/GigaSpeech/lhotse/hotwords}}"
AISHELL_DIR="${RAG_ASR_AISHELL_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/data_aishell/lhotse/hotwords}}"
AISHELL2_DIR="${RAG_ASR_AISHELL2_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/data_aishell2/lhotse/hotwords}}"
AISHELL3_DIR="${RAG_ASR_AISHELL3_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/data_aishell3/lhotse/hotwords}}"
MAGICDATA_DIR="${RAG_ASR_MAGICDATA_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/MAGICDATA/lhotse/hotwords}}"
THCHS30_DIR="${RAG_ASR_THCHS30_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/data_thchs30/lhotse/hotwords}}"
ZHVOICE_DIR="${RAG_ASR_ZHVOICE_HOTWORD_DIR:-${DATA_ROOT:+$DATA_ROOT/zhvoice/lhotse/hotwords}}"
EXP_NAME="${RAG_ASR_EXP_NAME:-amphion-1.7b_retrieval_v1.2}"
OUTPUT_DIR="${RAG_ASR_OUTPUT_DIR:-exp/retrieval/${EXP_NAME}}"

require_dir() {
    local label=$1 path=$2 hint=$3
    if [[ -z "$path" || ! -d "$path" ]]; then
        echo "Missing $label: ${path:-<unset>}" >&2
        echo "Set $hint, or set RAG_ASR_DATA_ROOT/RAG_ASR_HOTWORD_ROOT for the default layout." >&2
        exit 1
    fi
}

require_file() {
    local label=$1 path=$2 hint=$3
    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "Missing $label: ${path:-<unset>}" >&2
        echo "Set $hint, or set RAG_ASR_DATA_ROOT/RAG_ASR_HOTWORD_ROOT for the default layout." >&2
        exit 1
    fi
}

require_dir "base model" "${BASE_MODEL}" "RAG_ASR_BASE_MODEL"
require_dir "training supervision directory" "${TRAIN_V1}" "RAG_ASR_TRAIN_V1"
require_dir "hotword vocabulary directory" "${HOTWORD_DIR}" "RAG_ASR_HOTWORD_ROOT"
require_dir "Common Voice English manifest directory" "${CV_EN_DIR}" "RAG_ASR_CV_EN_HOTWORD_DIR"
require_dir "Common Voice Chinese manifest directory" "${CV_ZH_DIR}" "RAG_ASR_CV_ZH_HOTWORD_DIR"
require_dir "GigaSpeech manifest directory" "${GIGASPEECH_DIR}" "RAG_ASR_GIGASPEECH_HOTWORD_DIR"
require_dir "AISHELL manifest directory" "${AISHELL_DIR}" "RAG_ASR_AISHELL_HOTWORD_DIR"
require_dir "AISHELL2 manifest directory" "${AISHELL2_DIR}" "RAG_ASR_AISHELL2_HOTWORD_DIR"
require_dir "AISHELL3 manifest directory" "${AISHELL3_DIR}" "RAG_ASR_AISHELL3_HOTWORD_DIR"
require_dir "MAGICDATA manifest directory" "${MAGICDATA_DIR}" "RAG_ASR_MAGICDATA_HOTWORD_DIR"
require_dir "THCHS30 manifest directory" "${THCHS30_DIR}" "RAG_ASR_THCHS30_HOTWORD_DIR"
require_dir "zhvoice manifest directory" "${ZHVOICE_DIR}" "RAG_ASR_ZHVOICE_HOTWORD_DIR"

TRAIN_SUPS=(
    "${TRAIN_V1}/gigaspeech_supervisions_XL_punc_hotwords_nonempty.jsonl.gz"
    "${TRAIN_V1}/cv-en_supervisions_train_orig_punc_hotwords_nonempty.jsonl.gz"
    "${TRAIN_V1}/aishell_supervisions_train_punc_hotwords_nonempty.jsonl.gz"
    "${TRAIN_V1}/aishell2_supervisions_train_punc_hotwords_nonempty.jsonl.gz"
    "${TRAIN_V1}/aishell3_supervisions_train_punc_hotwords_nonempty.jsonl.gz"
    "${TRAIN_V1}/magicdata_supervisions_train_punc_hotwords_nonempty.jsonl.gz"
    "${TRAIN_V1}/thchs_30_supervisions_train_punc_hotwords_nonempty.jsonl.gz"
    "${TRAIN_V1}/zhaidatatang_supervisions_all_cleaned_punc_hotwords_nonempty.jsonl.gz"
)
TRAIN_RECS=(
    "${GIGASPEECH_DIR}/gigaspeech_recordings_XL.jsonl.gz"
    "${CV_EN_DIR}/cv-en_recordings_train.jsonl.gz"
    "${AISHELL_DIR}/aishell_recordings_train.jsonl.gz"
    "${AISHELL2_DIR}/aishell2_recordings_train.jsonl.gz"
    "${AISHELL3_DIR}/aishell3_recordings_train.jsonl.gz"
    "${MAGICDATA_DIR}/magicdata_recordings_train.jsonl.gz"
    "${THCHS30_DIR}/thchs_30_recordings_train.jsonl.gz"
    "${ZHVOICE_DIR}/zhaidatatang_recordings_all.jsonl.gz"
)

VAL_SUPS=(
    "${CV_EN_DIR}/cv-en_supervisions_test_orig_punc_hotwords.jsonl.gz"
    "${CV_ZH_DIR}/cv-zh-CN_supervisions_test_punc.jsonl.gz"
)
VAL_RECS=(
    "${CV_EN_DIR}/cv-en_recordings_test.jsonl.gz"
    "${CV_ZH_DIR}/cv-zh-CN_recordings_test.jsonl.gz"
)

EMBED_DIM="${RAG_ASR_EMBED_DIM:-512}"
ADAPTER_HIDDEN_DIM="${RAG_ASR_ADAPTER_HIDDEN_DIM:-512}"
BATCH_SIZE="${RAG_ASR_BATCH_SIZE:-64}"
N="${RAG_ASR_NUM_NEGATIVES:-4096}"
LR="${RAG_ASR_LR:-3e-4}"
EPOCHS="${RAG_ASR_EPOCHS:-10}"
TEMPERATURE="${RAG_ASR_TEMPERATURE:-0.07}"
LEARNABLE_TEMPERATURE="${RAG_ASR_LEARNABLE_TEMPERATURE:-true}"
LOSS_W_A2T="${RAG_ASR_LOSS_W_A2T:-1.0}"
LOSS_W_T2A="${RAG_ASR_LOSS_W_T2A:-1.0}"
GRAD_CLIP="${RAG_ASR_GRAD_CLIP:-1.0}"
NUM_WORKERS="${RAG_ASR_NUM_WORKERS:-8}"
SEED="${RAG_ASR_SEED:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${RAG_ASR_CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${RAG_ASR_NUM_GPUS:-${#GPU_ARR[@]}}"
WARMUP_STEPS="${RAG_ASR_WARMUP_STEPS:-0}"
WARMUP_RATIO="${RAG_ASR_WARMUP_RATIO:-0.05}"

for path in "${TRAIN_SUPS[@]}" "${TRAIN_RECS[@]}" "${VAL_SUPS[@]}" "${VAL_RECS[@]}"; do
    require_file "manifest" "$path" "the corresponding RAG_ASR_*_HOTWORD_DIR"
done
require_file "English training vocabulary" "${TRAIN_V1}/retrieval_vocab_en.txt" "RAG_ASR_TRAIN_V1"
require_file "Chinese training vocabulary" "${TRAIN_V1}/retrieval_vocab_zh.txt" "RAG_ASR_TRAIN_V1"
require_file "English validation vocabulary" "${HOTWORD_DIR}/en/en-50k.txt" "RAG_ASR_HOTWORD_ROOT"
require_file "Chinese validation vocabulary" "${HOTWORD_DIR}/zh/zh-50k.txt" "RAG_ASR_HOTWORD_ROOT"

mkdir -p "${OUTPUT_DIR}"
STDERR_LOG="${OUTPUT_DIR}/stderr.log"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    src/rag_asr/train.py \
    --base-model-path "${BASE_MODEL}" \
    --supervisions   "${TRAIN_SUPS[@]}" \
    --recordings     "${TRAIN_RECS[@]}" \
    --val-supervisions "${VAL_SUPS[@]}" \
    --val-recordings   "${VAL_RECS[@]}" \
    --output-dir     "${OUTPUT_DIR}" \
    --embed-dim      "${EMBED_DIM}" \
    $( [ -n "${ADAPTER_HIDDEN_DIM:-}" ] && echo "--adapter-hidden-dim ${ADAPTER_HIDDEN_DIM}" ) \
    --batch-size     "${BATCH_SIZE}" \
    --num-negatives  "${N}" \
    --lr             "${LR}" \
    --epochs         "${EPOCHS}" \
    --temperature    "${TEMPERATURE}" \
    $( [ "${LEARNABLE_TEMPERATURE}" = "true" ] && echo "--learnable-temperature" ) \
    --loss-w-a2t     "${LOSS_W_A2T}" \
    --loss-w-t2a     "${LOSS_W_T2A}" \
    --grad-clip      "${GRAD_CLIP}" \
    --num-workers    "${NUM_WORKERS}" \
    --seed           "${SEED}" \
    --warmup-steps   "${WARMUP_STEPS}" \
    --warmup-ratio   "${WARMUP_RATIO}" \
    --fp16 \
    --train-vocab-en   "${TRAIN_V1}/retrieval_vocab_en.txt" \
    --train-vocab-zh   "${TRAIN_V1}/retrieval_vocab_zh.txt" \
    --val-vocab-en     "${HOTWORD_DIR}/en/en-50k.txt" \
    --val-vocab-zh     "${HOTWORD_DIR}/zh/zh-50k.txt" \
    --val-recall-k     1 5 10 15 20 30 40 50 \
    --val-every-steps  1000 \
    --val-every-epoch  0 \
    --log-every 10 \
    2>"${STDERR_LOG}"

echo "Training done. Outputs in: ${OUTPUT_DIR}"
