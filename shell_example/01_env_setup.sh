#!/usr/bin/env bash
# ==============================================================================
# 01_env_setup.sh
# 激活 Python 环境，并确认 ModalityReap 包可正确导入
# ==============================================================================

set -euo pipefail

# ---------- 1. 激活 conda 环境 ----------
source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap

# ---------- 2. 进入项目根目录 ----------
PROJECT_ROOT="/data/szs/250010072/slai/MOE2Dense/ModalityReap"
cd "${PROJECT_ROOT}"

# ---------- 3. 将 src 目录加入 PYTHONPATH ----------
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# ---------- 4. 验证导入 ----------
python -c "import modality_reap; print('modality_reap 导入成功，版本路径:', modality_reap.__file__)"

# ---------- 5. 快速运行单元测试（可选） ----------
# PYTHONPATH=src pytest tests/ -q --tb=short
