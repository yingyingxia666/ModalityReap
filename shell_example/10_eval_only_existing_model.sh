#!/usr/bin/env bash
# ==============================================================================
# 10_eval_only_existing_model.sh
# 仅评测：对一个已保存的压缩模型（merged_model/ 目录）运行评测，不重新压缩
#
# 适合场景：
#   - 已有保存好的 merged_model，想用不同评测集重跑指标
#   - 对比不同评测超参（max_new_tokens 等）的影响
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 已保存的压缩模型路径（full_modality_reap 变体）
COMPRESSED_MODEL_PATH="${PROJECT_ROOT}/outputs/requirement4_20260407_main/full_modality_reap/compression/merged_model"
EVAL_OUTPUT="${PROJECT_ROOT}/outputs/eval_only_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${EVAL_OUTPUT}"

echo "========== 独立评测压缩模型: ${COMPRESSED_MODEL_PATH} =========="

python - <<'PYEOF'
import sys, torch
from pathlib import Path

sys.path.insert(0, "/data/szs/250010072/slai/MOE2Dense/ModalityReap/src")

from modality_reap.main import load_model_and_processors
from modality_reap.args import ModelArgs
from modality_reap.eval import (
    evaluate_generation,
    evaluate_teacher_forcing,
    load_eval_samples,
    save_evaluation_results,
    select_generation_subset,
)
from modality_reap.reporting import save_json

COMPRESSED_MODEL = "/data/szs/250010072/slai/MOE2Dense/ModalityReap/outputs/requirement4_20260407_main/full_modality_reap/compression/merged_model"
EVAL_OUTPUT = "/data/szs/250010072/slai/MOE2Dense/ModalityReap/outputs/eval_only"

model_args = ModelArgs(
    model_name=COMPRESSED_MODEL,
    attn_implementation="sdpa",
    disable_talker=True,
)
model, processor, tokenizer = load_model_and_processors(model_args)

eval_samples = load_eval_samples(
    audio_samples_per_dataset=3,
    text_samples_per_dataset=2,
    sample_seed=142,
    audio_sample_rate=16000,
)
gen_samples = select_generation_subset(
    eval_samples, audio_per_dataset=1, text_per_dataset=1
)

tf_results = evaluate_teacher_forcing(model, processor, eval_samples, max_seq_length=2048)
gen_results = evaluate_generation(model, processor, gen_samples, max_seq_length=2048, max_new_tokens=96)

save_evaluation_results(
    Path(EVAL_OUTPUT),
    teacher_forcing=tf_results,
    generation=gen_results,
    sample_manifest=eval_samples,
    generation_manifest=gen_samples,
)
save_json(Path(EVAL_OUTPUT) / "tf_aggregate.json", tf_results["aggregate"])
save_json(Path(EVAL_OUTPUT) / "gen_aggregate.json", gen_results["aggregate"])
print("评测完成，结果保存于:", EVAL_OUTPUT)
PYEOF

echo "========== 独立评测完成，输出: ${EVAL_OUTPUT} =========="
