#!/usr/bin/env bash
# ==============================================================================
# 11_custom_merge_method.sh
# 自定义 merge 方法实验：对比不同 merge 策略对同一压缩计划的影响
# 固定 Hybrid + LayerAdaptive 策略，切换 merge 方法
#
# 支持的 merge 方法：
#   frequency_weighted_average  (默认)
#   ties
#   multislerp
#   sce
#   karcher
#   submoe
#   conflict_aware_subspace     (创新点 3)
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"
BASE_OUTPUT="${PROJECT_ROOT}/outputs/merge_method_compare_$(date +%Y%m%d)"

MERGE_METHODS=(
    "frequency_weighted_average"
    "conflict_aware_subspace"
    "ties"
    "karcher"
)

COMMON_ARGS=(
    --model_name                  "${MODEL_PATH}"
    --attn_implementation         "sdpa"
    --disable_talker              True
    --total_audio_samples         10
    --total_text_samples          5
    --max_seq_length              1024
    --audio_sample_rate           16000
    --seed                        42
    --do_save                     False
    --smoke_test                  False
    --compression_ratio           0.5
    --use_hybrid_strategy         True
    --use_layer_adaptive_schedule True
)

for METHOD in "${MERGE_METHODS[@]}"; do
    echo "========== Merge 方法: ${METHOD} =========="
    DOM_AS_BASE="False"
    if [[ "${METHOD}" == "conflict_aware_subspace" || "${METHOD}" == "ties" || "${METHOD}" == "multislerp" ]]; then
        DOM_AS_BASE="True"
    fi

    python -m modality_reap.main \
        "${COMMON_ARGS[@]}" \
        --output_dir      "${BASE_OUTPUT}/${METHOD}" \
        --merge_method    "${METHOD}" \
        --dom_as_base     "${DOM_AS_BASE}"

    echo "===== ${METHOD} 完成 ====="
done

echo "========== 所有 merge 方法实验完成，输出根: ${BASE_OUTPUT} =========="
