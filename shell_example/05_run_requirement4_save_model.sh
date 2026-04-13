#!/usr/bin/env bash
# ==============================================================================
# 05_run_requirement4_save_model.sh
# 正式实验 + 保存 full_modality_reap 压缩模型到磁盘
# 适合需要将压缩后模型用于部署或进一步微调的场景
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/requirement4_save_model_$(date +%Y%m%d)"
MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"

echo "========== 正式实验（含保存模型），输出目录: ${OUTPUT_DIR} =========="

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
    --max-new-tokens          96 \
    --save-full-model              # 开启后 full_modality_reap 变体的压缩模型会保存到 merged_model/

echo "========== 完成，压缩模型保存于: ${OUTPUT_DIR}/full_modality_reap/compression/merged_model/ =========="
