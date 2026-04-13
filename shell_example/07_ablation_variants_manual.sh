#!/usr/bin/env bash
# ==============================================================================
# 07_ablation_variants_manual.sh
# 手动逐个启动四个消融变体（不走 requirement4.py 统一驱动）
# 适合需要单独调试或重跑某个变体的场景
# 每个变体会独立保存到各自子目录
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"
BASE_OUTPUT="${PROJECT_ROOT}/outputs/ablation_manual_$(date +%Y%m%d)"

COMMON_ARGS=(
    --model_name              "${MODEL_PATH}"
    --attn_implementation     "sdpa"
    --disable_talker          True
    --total_audio_samples     20
    --total_text_samples      9
    --max_seq_length          2048
    --audio_sample_rate       16000
    --seed                    42
    --do_save                 False
    --smoke_test              False
    --compression_ratio       0.5
)

# ---------- 变体 1：baseline_reap ----------
echo "===== baseline_reap ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                  "${BASE_OUTPUT}/baseline_reap" \
    --use_hybrid_strategy         False \
    --use_layer_adaptive_schedule False \
    --merge_method                "frequency_weighted_average" \
    --dom_as_base                 False

# ---------- 变体 2：hybrid_only ----------
echo "===== hybrid_only ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                  "${BASE_OUTPUT}/hybrid_only" \
    --use_hybrid_strategy         True \
    --use_layer_adaptive_schedule False \
    --merge_method                "frequency_weighted_average" \
    --dom_as_base                 False

# ---------- 变体 3：hybrid_plus_schedule ----------
echo "===== hybrid_plus_schedule ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                  "${BASE_OUTPUT}/hybrid_plus_schedule" \
    --use_hybrid_strategy         True \
    --use_layer_adaptive_schedule True \
    --merge_method                "frequency_weighted_average" \
    --dom_as_base                 False

# ---------- 变体 4：full_modality_reap ----------
echo "===== full_modality_reap ====="
python -m modality_reap.main \
    "${COMMON_ARGS[@]}" \
    --output_dir                  "${BASE_OUTPUT}/full_modality_reap" \
    --use_hybrid_strategy         True \
    --use_layer_adaptive_schedule True \
    --merge_method                "conflict_aware_subspace" \
    --dom_as_base                 True

echo "===== 四个变体均已完成，输出根目录: ${BASE_OUTPUT} ====="
