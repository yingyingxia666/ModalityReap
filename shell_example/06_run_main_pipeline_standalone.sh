#!/usr/bin/env bash
# ==============================================================================
# 06_run_main_pipeline_standalone.sh
# 直接调用底层 main.py 的压缩 pipeline（单次运行，非消融对比）
# 适合对某一个具体配置做定制实验，或在其他脚本中集成
#
# 使用 HfArgumentParser，参数通过 --reap_* / --model_* / --data_* 等前缀传入
# 注意：main.py 中参数名与 dataclass 字段名一致，用下划线分隔
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/standalone_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct"

echo "========== 单次压缩 Pipeline 开始 =========="

python -m modality_reap.main \
    --seed                        42 \
    --do_save                     True \
    --smoke_test                  False \
    --model_name                  "${MODEL_PATH}" \
    --attn_implementation         "sdpa" \
    --disable_talker              True \
    --total_audio_samples         20 \
    --total_text_samples          9 \
    --max_seq_length              2048 \
    --audio_sample_rate           16000 \
    --output_dir                  "${OUTPUT_DIR}" \
    --compression_ratio           0.5 \
    --use_hybrid_strategy         True \
    --use_layer_adaptive_schedule True \
    --merge_method                "conflict_aware_subspace" \
    --dom_as_base                 True \
    --subspace_rank_ratio         0.35 \
    --conflict_anchor_strength    0.65

echo "========== 压缩完成，输出目录: ${OUTPUT_DIR} =========="
