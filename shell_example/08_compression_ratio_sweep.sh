#!/usr/bin/env bash
# ==============================================================================
# 08_compression_ratio_sweep.sh
# 压缩率扫描实验：对 full_modality_reap 配置，在不同压缩率下评估性能退化曲线
# 压缩率：0.3 / 0.4 / 0.5 / 0.6 / 0.7
# 适合绘制压缩率 vs 指标的 trade-off 图
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"
SWEEP_OUTPUT="${PROJECT_ROOT}/outputs/compression_ratio_sweep_$(date +%Y%m%d)"

for RATIO in 0.3 0.4 0.5 0.6 0.7; do
    RATIO_TAG=$(echo "${RATIO}" | tr '.' '_')
    echo "========== 压缩率 ${RATIO} =========="

    python -m modality_reap.requirement4 \
        --output-dir              "${SWEEP_OUTPUT}/ratio_${RATIO_TAG}" \
        --model-name              "${MODEL_PATH}" \
        --attn-implementation     "sdpa" \
        --seed                    42 \
        --compression-ratio       "${RATIO}" \
        --obs-audio-samples       20 \
        --obs-text-samples        9 \
        --eval-audio-per-dataset  3 \
        --eval-text-per-dataset   2 \
        --gen-audio-per-dataset   1 \
        --gen-text-per-dataset    1 \
        --max-seq-length          2048 \
        --max-new-tokens          96

    echo "===== 压缩率 ${RATIO} 完成 ====="
done

echo "========== 全部压缩率扫描完成，输出根目录: ${SWEEP_OUTPUT} =========="
