#!/usr/bin/env bash
# ==============================================================================
# 14_cluster_method_compare.sh
# 聚类方法对比：agglomerative vs dynamic_ttm，以及不同 linkage 方法
#
# ClusterArgs 相关参数：
#   cluster_method: agglomerative | dynamic_ttm
#   linkage_method: average | ward | complete | single
#   frequency_penalty: True | False
#   max_cluster_size: 限制单个 cluster 最大专家数（可选）
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"
BASE_OUTPUT="${PROJECT_ROOT}/outputs/cluster_compare_$(date +%Y%m%d)"

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
    --merge_method                "conflict_aware_subspace"
    --dom_as_base                 True
)

# agglomerative + average（默认）
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir      "${BASE_OUTPUT}/agg_average" \
    --cluster_method  "agglomerative" \
    --linkage_method  "average"

# agglomerative + ward
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir      "${BASE_OUTPUT}/agg_ward" \
    --cluster_method  "agglomerative" \
    --linkage_method  "ward"

# agglomerative + complete
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir      "${BASE_OUTPUT}/agg_complete" \
    --cluster_method  "agglomerative" \
    --linkage_method  "complete"

# dynamic_ttm（频率动态聚类）
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir      "${BASE_OUTPUT}/dynamic_ttm" \
    --cluster_method  "dynamic_ttm"

# agglomerative + 关闭频率惩罚
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir          "${BASE_OUTPUT}/agg_no_freq_penalty" \
    --cluster_method      "agglomerative" \
    --linkage_method      "average" \
    --frequency_penalty   False

echo "===== 聚类方法对比完成，输出根: ${BASE_OUTPUT} ====="
