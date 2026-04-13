#!/usr/bin/env bash
# ==============================================================================
# 09_resume_from_observation_cache.sh
# 断点续跑：复用已有 observation cache，跳过耗时的 MoE 激活收集阶段
#
# 场景：第一次实验已采集了 observations.pt，现在想调整 merge 参数重跑，
#       无需重新跑模型收集激活统计。
#
# 前提：已有 observation_cache/observations.pt 在目标 output_dir 下。
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"

# 指定已有实验目录（内含 observation_cache/observations.pt）
EXISTING_OUTPUT="${PROJECT_ROOT}/outputs/requirement4_20260407_main"

# 新实验产物写到这里
NEW_OUTPUT="${PROJECT_ROOT}/outputs/resume_$(date +%Y%m%d_%H%M%S)"

echo "========== 断点续跑，复用 observation cache =========="
echo "已有实验目录: ${EXISTING_OUTPUT}"
echo "新实验目录:   ${NEW_OUTPUT}"

# requirement4.py 会自动检测 observation_cache/observations.pt 并跳过采集
python -m modality_reap.requirement4 \
    --output-dir              "${NEW_OUTPUT}" \
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

# 若 NEW_OUTPUT 下无 observation_cache，手动软链到已有缓存可节省磁盘：
# mkdir -p "${NEW_OUTPUT}"
# ln -s "${EXISTING_OUTPUT}/observation_cache" "${NEW_OUTPUT}/observation_cache"
# 然后再执行上面的 python 命令

echo "========== 续跑完成，新结果: ${NEW_OUTPUT}/summary.md =========="
