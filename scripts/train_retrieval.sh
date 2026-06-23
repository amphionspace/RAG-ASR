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
BASE_MODEL="${ROOT}/checkpoints/base/amphion_1.7b_merged"
TRAIN_V1=/chenmingjie/lx/data/hotword/train_v1
HOTWORD_DIR=/chenmingjie/lx/data/hotword
CV_EN_DIR=/ai_sds_wuzz/DATA_ASR/common_voice_en/lhotse/hotwords
CV_ZH_DIR=/ai_sds_wuzz/DATA_ASR/common_voice_zh/lhotse/hotwords
EXP_NAME=amphion-1.7b_retrieval_v1.2
OUTPUT_DIR=exp/retrieval/${EXP_NAME}

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
    /ai_sds_wuzz/DATA_ASR/GigaSpeech/lhotse/hotwords/gigaspeech_recordings_XL.jsonl.gz
    "${CV_EN_DIR}/cv-en_recordings_train.jsonl.gz"
    /ai_sds_wuzz/DATA_ASR/data_aishell/lhotse/hotwords/aishell_recordings_train.jsonl.gz
    /ai_sds_wuzz/DATA_ASR/data_aishell2/lhotse/hotwords/aishell2_recordings_train.jsonl.gz
    /ai_sds_wuzz/DATA_ASR/data_aishell3/lhotse/hotwords/aishell3_recordings_train.jsonl.gz
    /ai_sds_wuzz/DATA_ASR/MAGICDATA/lhotse/hotwords/magicdata_recordings_train.jsonl.gz
    /ai_sds_wuzz/DATA_ASR/data_thchs30/lhotse/hotwords/thchs_30_recordings_train.jsonl.gz
    /ai_sds_wuzz/DATA_ASR/zhvoice/lhotse/hotwords/zhaidatatang_recordings_all.jsonl.gz
)

VAL_SUPS=(
    "${CV_EN_DIR}/cv-en_supervisions_test_orig_punc_hotwords.jsonl.gz"
    "${CV_ZH_DIR}/cv-zh-CN_supervisions_test_punc.jsonl.gz"
)
VAL_RECS=(
    "${CV_EN_DIR}/cv-en_recordings_test.jsonl.gz"
    "${CV_ZH_DIR}/cv-zh-CN_recordings_test.jsonl.gz"
)

EMBED_DIM=512
ADAPTER_HIDDEN_DIM=512
BATCH_SIZE=64
N=4096
LR=3e-4
EPOCHS=10
TEMPERATURE=0.07
LEARNABLE_TEMPERATURE=true
LOSS_W_A2T=1.0
LOSS_W_T2A=1.0
GRAD_CLIP=1.0
NUM_WORKERS=8
SEED=42
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
WARMUP_STEPS=0
WARMUP_RATIO=0.05

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
