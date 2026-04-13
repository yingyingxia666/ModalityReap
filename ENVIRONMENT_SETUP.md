# ModalityReap 环境配置说明

## 系统要求

| 项目 | 版本要求 |
|------|---------|
| 操作系统 | Linux（Ubuntu 20.04 / 22.04 推荐） |
| Python | 3.12.x |
| CUDA | 12.6（对应 PyTorch 2.7.1+cu126） |
| cuDNN | 9.x |
| conda | 任意版本（Miniconda / Anaconda 均可） |

> **注意**：PyTorch 和 vLLM 对 CUDA 版本有严格依赖，低于 CUDA 12.4 的机器需要换对应的 wheel。

---

## 一、快速配置（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/yingyingxia666/ModalityReap.git
cd ModalityReap

# 2. 用完整 environment.yml 一键还原环境（网络较好时推荐）
conda env create -f environment.yml

# 3. 激活
conda activate reap

# 4. 安装本项目包（开发模式）
pip install -e .

# 5. 验证
python -c "import modality_reap; import torch; print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

---

## 二、分步配置（网络较差 / 手动控制版本时）

### 2.1 创建基础 Python 环境

```bash
conda create -n reap python=3.12.12 -y
conda activate reap
```

### 2.2 安装 PyTorch（CUDA 12.6）

```bash
# 官方安装命令（从 PyPI 下载，包含 CUDA 12.6 运行时）
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
```

> 如果是 CUDA 12.4 环境，把 `cu126` 改成 `cu124`，torch 版本也需对应调整。

### 2.3 安装核心依赖

```bash
pip install -r requirements-core.txt
```

### 2.4 安装 vLLM（可选，用于推理加速）

vLLM 版本和 torch/CUDA 绑定严格，**必须在 torch 安装之后再装**：

```bash
pip install vllm==0.10.0
```

### 2.5 安装本项目包

```bash
# 开发模式：代码改动无需重新安装
pip install -e .
```

---

## 三、完整 environment.yml 说明

`environment.yml` 是从当前运行环境直接导出的，包含所有 pip 包的精确版本，适合需要**完整复现**实验结果的场景。

```bash
conda env create -f environment.yml
conda activate reap
pip install -e .
```

> 该文件由 `conda env export --no-builds` 生成，pip 部分锁定了完整版本。
> 如遇某个小包安装失败，可跳过后手动补装，不影响主体功能。

---

## 四、关键包版本一览

以下是影响实验结果的核心包，版本需严格对齐：

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.7.1+cu126 | 主框架 |
| `torchaudio` | 2.7.1 | 音频加载 |
| `transformers` | 5.2.0 | 模型加载（需含 Qwen3OmniMoe 支持） |
| `accelerate` | 1.12.0 | 多卡 device_map |
| `vllm` | 0.10.0 | 推理加速（可选） |
| `xformers` | 0.0.31 | Flash-Attention 替代 |
| `triton` | 3.3.1 | GPU kernel 编译 |
| `datasets` | 3.6.0 | 数据集加载 |
| `scikit-learn` | 1.8.0 | 聚类（agglomerative） |
| `scipy` | 1.17.0 | 层次聚类距离计算 |
| `librosa` | 0.11.0 | 音频预处理 |
| `soundfile` | 0.13.1 | 音频文件读写 |
| `safetensors` | 0.7.0 | 模型权重保存 |
| `sentencepiece` | 0.2.1 | 分词器依赖 |
| `lm-eval` | 0.4.10 | 外部评测框架（可选） |
| `sacrebleu` | 2.6.0 | BLEU 指标 |
| `rouge-score` | 0.1.2 | ROUGE 指标 |
| `trl` | 0.22.2 | SFT / RLHF 训练工具（可选） |

---

## 五、常见问题

### Q1：`transformers` 版本不对，找不到 `Qwen3OmniMoeForConditionalGeneration`

该类在 `transformers >= 4.55.0` 中引入，5.x 已正式收录。需确保版本 `>= 4.55.0`，推荐直接用 `5.2.0`：

```bash
pip install transformers==5.2.0
```

### Q2：vLLM 安装报 CUDA 版本不匹配

vLLM 的 wheel 和 CUDA 版本强绑定。查看当前 CUDA 版本：

```bash
python -c "import torch; print(torch.version.cuda)"
```

然后从 vLLM GitHub Releases 页面找对应 wheel 手动安装：
```
https://github.com/vllm-project/vllm/releases
```

### Q3：`xformers` 安装失败

可以跳过，直接用 `sdpa`（PyTorch 原生实现）：

```python
# 在 ModelArgs 里设置
attn_implementation = "sdpa"   # 而不是 "flash_attention_2"
```

### Q4：在没有 GPU 的机器上能否运行单元测试？

可以。单元测试不加载真实模型，全部用 mock 数据：

```bash
PYTHONPATH=src pytest tests/ -q
```

### Q5：conda 不可用，只有 pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-core.txt
pip install -e .
```

---

## 六、模型路径说明

本项目默认使用的模型：

```
/data/szs/share/Qwen3-Omni-30B-A3B-Instruct
```

这是服务器本地路径。其他机器可从 HuggingFace 下载：

```bash
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --local-dir /your/local/path/Qwen3-Omni-30B-A3B-Instruct
```

下载后通过 `--model-name` 参数或修改 `ModelArgs.model_name` 默认值指定路径。

---

## 七、数据集路径说明

数据集默认根路径：`/data/szs/share/voice_model_project/datasets`

包含以下子目录（用于 observation 采集和评测）：

```
datasets/
├── Clotho/            # Clotho 音频描述数据集
├── AudioCaps/         # AudioCaps 音频描述数据集
├── ultrachat/         # UltraChat 文本对话数据集
├── OpenHermes-2___5/  # OpenHermes 2.5 指令数据集
└── tulu-3-sft-mixture/  # Tulu-3 SFT 混合数据集
```

其他环境可通过修改 `DataArgs.dataset_root` 或 `eval.py` 中的 `DatasetSpec` 路径适配本地数据集。
