#!/usr/bin/env bash
# ==============================================================================
# 04_run_requirement4_main.sh
# 正式实验：较大样本量，完整评测四个消融变体
# 复现 requirement4_20260407_main 的实验结果
# 预计耗时：2~6 小时（取决于 GPU 配置）
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/requirement4_$(date +%Y%m%d)_main"
MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"

echo "========== 正式实验开始，输出目录: ${OUTPUT_DIR} =========="

python -m modality_reap.requirement4 \
    --output-dir              "${OUTPUT_DIR}" \
    --model-name              "${MODEL_PATH}" \
    --attn-implementation     "sdpa" \
    --seed                    42 \
    --compression-ratio       0.5 \
    --obs-audio-samples       20 \
    --obs-text-samples        9 \
    --eval-audio-per-dataset  3 \
    --eval-text-per-dataset   2 \
    --gen-audio-per-dataset   1 \
    --gen-text-per-dataset    1 \
    --max-seq-length          2048 \
    --max-new-tokens          96

echo "========== 正式实验完成，结果位于: ${OUTPUT_DIR}/summary.md =========="
cat "${OUTPUT_DIR}/summary.md"
