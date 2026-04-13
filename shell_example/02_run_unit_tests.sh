#!/usr/bin/env bash
# ==============================================================================
# 02_run_unit_tests.sh
# 运行所有单元测试，验证各模块语法和基础逻辑
# ==============================================================================

set -euo pipefail

source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "========== 运行全量单元测试 =========="
PYTHONPATH=src pytest \
    tests/test_cluster.py \
    tests/test_strategy.py \
    tests/test_merge.py \
    tests/test_data.py \
    tests/test_eval.py \
    tests/test_scoring.py \
    -v \
    --tb=short

echo "========== 单元测试全部通过 =========="
