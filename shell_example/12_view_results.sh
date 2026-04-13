#!/usr/bin/env bash
# ==============================================================================
# 12_view_results.sh
# 快速查看已有实验结果：打印 leaderboard、summary.md
# 不运行任何模型，仅读文件
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"

echo "========== 已有实验目录列表 =========="
ls -lt "${OUTPUTS_DIR}" | head -20

echo ""
echo "========== 最新正式实验结果（requirement4_20260407_main）=========="

MAIN_RESULT="${OUTPUTS_DIR}/requirement4_20260407_main"
if [ -f "${MAIN_RESULT}/summary.md" ]; then
    cat "${MAIN_RESULT}/summary.md"
else
    echo "[警告] summary.md 不存在于 ${MAIN_RESULT}"
fi

echo ""
echo "========== Leaderboard JSON =========="
if [ -f "${MAIN_RESULT}/leaderboard.json" ]; then
    python3 -c "
import json, sys
with open('${MAIN_RESULT}/leaderboard.json') as f:
    data = json.load(f)
rows = data.get('rows', [])
print(f'共 {len(rows)} 个变体:')
for row in rows:
    name = row['name']
    ratio = row.get('compression_ratio', 0)
    loss  = row.get('tf_loss', 0)
    rouge = row.get('gen_rouge_l_f1', 0)
    f1    = row.get('gen_token_f1', 0)
    print(f'  {name:<30} compression={ratio:.4f}  tf_loss={loss:.4f}  rouge_l={rouge:.4f}  token_f1={f1:.4f}')
"
else
    echo "[警告] leaderboard.json 不存在"
fi

echo ""
echo "========== 各变体 summary 摘要 =========="
for VARIANT in baseline_reap hybrid_only hybrid_plus_schedule full_modality_reap; do
    SUMMARY_PATH="${MAIN_RESULT}/${VARIANT}/summary.json"
    if [ -f "${SUMMARY_PATH}" ]; then
        echo "--- ${VARIANT} ---"
        python3 -c "
import json
with open('${SUMMARY_PATH}') as f:
    d = json.load(f)
tf  = d.get('teacher_forcing', {})
gen = d.get('generation', {})
comp = d.get('compression', {})
print(f\"  TF  loss={tf.get('loss',0):.4f}  ppl={tf.get('ppl',0):.2f}  token_acc={tf.get('token_accuracy',0):.4f}\")
print(f\"  Gen rouge_l={gen.get('best_rouge_l_f1',0):.4f}  token_f1={gen.get('best_token_f1',0):.4f}\")
if comp:
    print(f\"  Compression: {comp.get('achieved_compression_ratio',0):.4f}  experts: {comp.get('total_experts_before',0)} -> {comp.get('total_experts_after',0)}\")
"
    else
        echo "  [${VARIANT}] summary.json 不存在"
    fi
done
