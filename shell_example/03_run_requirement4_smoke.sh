#!/usr/bin/env bash
# ==============================================================================
# 03_run_requirement4_smoke.sh
# 快速 smoke 验证：用极少样本（1-2 条）跑通四个消融变体，确保整条链路不报错
# 不做评测，不保存模型，仅验证流程可达
# 预计耗时：20~40 分钟（取决于 GPU 数量）
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/smoke_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"

echo "========== Smoke 验证开始，输出目录: ${OUTPUT_DIR} =========="

python -m modality_reap.requirement4 \
    --output-dir          "${OUTPUT_DIR}" \
    --model-name          "${MODEL_PATH}" \
    --attn-implementation "sdpa" \
    --seed                42 \
    --compression-ratio   0.5 \
    --obs-audio-samples   2 \
    --obs-text-samples    2 \
    --eval-audio-per-dataset 1 \
    --eval-text-per-dataset  1 \
    --gen-audio-per-dataset  1 \
    --gen-text-per-dataset   1 \
    --max-seq-length      512 \
    --max-new-tokens      32

echo "========== Smoke 验证通过，输出: ${OUTPUT_DIR} =========="
