#!/usr/bin/env bash
# ==============================================================================
# 13_sensitivity_weight_ablation.sh
# 超参消融：Layer-Adaptive Schedule 的敏感度权重组合对比
#
# 消融四种权重配置，观察 target_experts 分布和压缩后性能变化
# 核心参数（ClusterArgs）：
#   sensitivity_top1_weight         (top1 集中度权重)
#   sensitivity_cv_weight           (负载均衡变异系数权重)
#   sensitivity_modal_gap_weight    (模态冲突权重)
#   sensitivity_shared_hotspot_weight (共享热点权重)
#   sensitivity_active_expert_weight  (活跃专家负收益权重)
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"
BASE_OUTPUT="${PROJECT_ROOT}/outputs/sensitivity_ablation_$(date +%Y%m%d)"

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

# --- 配置 A：默认权重（论文设置）---
echo "===== 配置 A: default ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                        "${BASE_OUTPUT}/config_A_default" \
    --sensitivity_top1_weight           0.35 \
    --sensitivity_cv_weight             0.30 \
    --sensitivity_modal_gap_weight      0.25 \
    --sensitivity_shared_hotspot_weight 0.10 \
    --sensitivity_active_expert_weight  0.20

# --- 配置 B：强调 top1 集中度 ---
echo "===== 配置 B: top1_heavy ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                        "${BASE_OUTPUT}/config_B_top1_heavy" \
    --sensitivity_top1_weight           0.60 \
    --sensitivity_cv_weight             0.15 \
    --sensitivity_modal_gap_weight      0.15 \
    --sensitivity_shared_hotspot_weight 0.10 \
    --sensitivity_active_expert_weight  0.20

# --- 配置 C：强调模态冲突 ---
echo "===== 配置 C: modal_gap_heavy ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                        "${BASE_OUTPUT}/config_C_modal_gap_heavy" \
    --sensitivity_top1_weight           0.20 \
    --sensitivity_cv_weight             0.15 \
    --sensitivity_modal_gap_weight      0.55 \
    --sensitivity_shared_hotspot_weight 0.10 \
    --sensitivity_active_expert_weight  0.20

# --- 配置 D：均等权重 ---
echo "===== 配置 D: uniform ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                        "${BASE_OUTPUT}/config_D_uniform" \
    --sensitivity_top1_weight           0.25 \
    --sensitivity_cv_weight             0.25 \
    --sensitivity_modal_gap_weight      0.25 \
    --sensitivity_shared_hotspot_weight 0.25 \
    --sensitivity_active_expert_weight  0.20

echo "===== 敏感度权重消融完成，输出根: ${BASE_OUTPUT} ====="
