# ModalityReap 项目功能汇总

> 项目路径：`/data/szs/250010072/slai/MOE2Dense/ModalityReap`
> 基于 Requirement 4 对话记录（2026-04-07）整理

---

## 一、项目定位

ModalityReap 是一个针对 **多模态 MoE 大语言模型**（当前支持 Qwen3-Omni-30B-A3B-Instruct）的
**专家压缩框架**，核心目标是：在尽量保留音频/文本多模态性能的前提下，大幅减少 MoE 层的激活专家数，
从而实现模型推理加速与显存缩减。

---

## 二、三大创新点

### 创新点 1：Hybrid Modality-REAP（混合模态专家策略）
- **位置**：`strategy.py` → `build_hybrid_compression_plan`
- **机制**：
  - 对每一层专家按 **音频核心分（audio_priority_score）**、**跨模态通用分（generalist_score）**、**模态冲突分（conflict_score）** 分别打分
  - 将专家分成三类：`keep_audio`（音频核心，不压缩）、`keep_shared`（跨模态通用，不压缩）、`merge`（可合并）、`prune`（直接剪除）
  - 通过分位数阈值（`audio_keep_quantile`、`shared_keep_quantile` 等）控制各类边界
  - 提供 `rescue` 机制：当 merge budget 未满时，从 prune 候选中救回优先级高的专家
- **对比基线**：传统做法对所有专家一视同仁地 uniform 压缩

### 创新点 2：Layer-Adaptive Compression Schedule（层自适应压缩调度）
- **位置**：`strategy.py` → `build_layer_adaptive_schedule`
- **机制**：
  - 对每一层计算 **敏感度分（sensitivity_score）**，由以下加权组成：
    - `top1_share`：最大激活专家占比（越高越敏感）
    - `load_balance_cv`：负载均衡变异系数（越不均衡越敏感）
    - `modal_gap`：模态冲突均值（越大越敏感）
    - `shared_hotspot_share`：共享热点比重
    - `-active_ratio`：活跃专家比例（越高越不敏感）
  - 中间层（layers `max/3` ～ `2*max/3`）额外乘 `middle_layer_multiplier`（默认 1.5）以保护关键特征层
  - 根据敏感度在 `[min_compression_ratio, max_compression_ratio]` 范围内线性映射每层的目标压缩率：**高敏感层压缩少，低敏感层压缩多**
- **对比基线**：所有层使用统一的全局压缩率

### 创新点 3：Conflict-Aware Subspace Merge（冲突感知子空间合并）
- **位置**：`merge.py` → `MoEExpertMerger._conflict_aware_subspace_merge`
- **机制**：
  - 对每个 cluster 计算其 **模态冲突分**（各成员专家冲突分均值）
  - 合并时先提取权重矩阵的低秩共享子空间（通过 SVD，rank 由 `subspace_rank_ratio` 控制，且冲突越高 rank 越小）
  - 残差部分加权平均后，**冲突越高越向 anchor（主导专家）靠拢**（由 `conflict_anchor_strength` 控制）
  - 最终输出 = 共享子空间 + (1 - anchor_mix) × 残差均值 + anchor_mix × anchor 残差
- **对比基线**：传统频率加权平均（frequency_weighted_average），忽略模态冲突

---

## 三、完整 Pipeline 阶段（8 个阶段）

| 阶段 | 描述 | 关键代码 |
|------|------|---------|
| 1/8 | 加载 Qwen3-Omni 模型与处理器 | `main.py::load_model_and_processors` |
| 2/8 | 准备音频/文本观测样本 | `main.py::prepare_modality_samples` |
| 3/8 | 收集各层 MoE 激活统计（observation） | `main.py::collect_observations` |
| 4/8 | 计算模态打分，生成压缩计划 | `main.py::build_compression_plan` |
| 5/8 | 逐层聚类（将 merge 候选分组） | `main.py::build_cluster_labels` |
| 6/8 | Merge experts（按选定策略合并同 cluster） | `main.py::merge_model` |
| 7/8 | Compact（物理删除冗余专家，更新 gate） | `main.py::compact_fused_experts` |
| 8/8 | 保存压缩模型（可选） | `main.py::save_merged_model` |

---

## 四、消融实验设计（4 个变体）

| 变体名 | 创新1 Hybrid | 创新2 Schedule | 创新3 CAS Merge | 说明 |
|--------|:-----------:|:--------------:|:---------------:|------|
| `baseline_reap` | ❌ | ❌ | ❌ | 传统统一压缩基线 |
| `hybrid_only` | ✅ | ❌ | ❌ | 仅混合模态策略 |
| `hybrid_plus_schedule` | ✅ | ✅ | ❌ | Hybrid + 层自适应调度 |
| `full_modality_reap` | ✅ | ✅ | ✅ | 三项创新全开 |

---

## 五、评测指标体系

### Teacher-Forcing（闭卷）
| 指标 | 说明 |
|------|------|
| `loss` | 交叉熵损失 |
| `ppl` | 困惑度（exp(loss)） |
| `token_accuracy` | 逐 token 预测准确率 |

### Generation（开放生成）
| 指标 | 说明 |
|------|------|
| `best_exact_match` | 精确匹配（多参考取最优） |
| `best_token_f1` | Token-F1（多参考取最优） |
| `best_rouge_l_f1` | ROUGE-L F1（多参考取最优） |

分组维度：`overall` / `modality:audio` / `modality:text` / `dataset:<name>`

---

## 六、正式实验结果（requirement4_20260407_main）

| Variant | Compression | TF Loss | TF PPL | TF Token Acc | Gen Rouge-L | Gen Token F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_model | 0.0000 | 3.9448 | 389.2985 | 0.4220 | 0.2973 | 0.3787 |
| baseline_reap | 0.5000 | 4.5869 | 404.7068 | 0.3231 | 0.1121 | 0.1684 |
| hybrid_only | 0.5000 | 4.1295 | 505.6828 | 0.3893 | 0.2943 | 0.3562 |
| hybrid_plus_schedule | 0.5104 | 4.3380 | 698.0451 | 0.3796 | 0.3055 | 0.3722 |
| full_modality_reap | 0.5104 | 4.3419 | 710.6158 | 0.3860 | 0.3186 | 0.3870 |

**关键结论：**
- `baseline_reap` 在 50% 压缩下退化最严重（Gen Rouge-L 仅 0.1121）
- `hybrid_only` 是压缩后 teacher-forcing 最稳的方案（TF Loss 仅 +0.18）
- `hybrid_plus_schedule` 在更高压缩率下，生成 Rouge-L 已超过原模型
- `full_modality_reap` 综合生成效果最佳：Gen Token-F1（0.3870）和 Rouge-L（0.3186）均略超原模型

---

## 七、关键模块说明

| 模块文件 | 功能 |
|---------|------|
| `main.py` | 主压缩 pipeline，8 阶段编排 |
| `args.py` | 所有参数 dataclass 定义（ReapArgs/ModelArgs/DataArgs/ObserverArgs/ClusterArgs/MergeArgs/ReportArgs） |
| `strategy.py` | 层自适应调度 + Hybrid 压缩计划构建 |
| `cluster.py` | 层内聚类（agglomerative / dynamic_ttm），支持频率惩罚 |
| `merge.py` | 多种 merge 方法（FWA / TIES / MultiSLERP / SCE / Karcher / SubMoE / ConflictAwareSubspace） |
| `observer.py` | MoE 激活统计收集（hook-based） |
| `scoring.py` | 模态打分：音频优先分 / 通用分 / 冲突分 |
| `data.py` | 数据集加载（JSONL + 音频），多模态样本构建 |
| `eval.py` | 评测模块：teacher-forcing + generation，指标计算 |
| `model_util.py` | 模型工具：获取 MoE 层 / 验证 merge / 模型属性注册 |
| `permute.py` | 专家权重置换（用于 merge 前对齐） |
| `reporting.py` | JSON 保存 / 告警汇总 |
| `requirement4.py` | Requirement 4 实验驱动：消融对比 + 评测 + 断点续跑 |
| `metrics.py` / `scoring.py` | 辅助指标计算 |

---

## 八、输出目录结构

```
outputs/
└── requirement4_<date>/
    ├── experiment_config.json          # 实验超参
    ├── evaluation_manifest.json        # 评测样本统计
    ├── leaderboard.json                # 各变体对比汇总
    ├── summary.md                      # Markdown 对比表
    ├── observation_cache/
    │   ├── observations.pt             # 缓存的 MoE 激活统计（可跨变体复用）
    │   └── manifest.json
    ├── original_model/
    │   ├── summary.json
    │   └── evaluation/                 # 原始模型评测结果
    ├── baseline_reap/
    │   ├── summary.json
    │   ├── compression/                # 压缩过程产物（plans/clusters/scores）
    │   └── evaluation/                 # 变体评测结果
    ├── hybrid_only/         ...
    ├── hybrid_plus_schedule/ ...
    └── full_modality_reap/  ...
```

---

## 九、数据集

- **音频数据集**（observation）：Clotho、AudioCaps、LibriSpeech、MELD、IEMOCAP 等
- **文本数据集**（observation）：UltraChat、OpenHermes-2.5、tulu-3-sft-mixture
- **评测集**：Clotho-validation、AudioCaps-test（音频）+ UltraChat、OpenHermes、tulu（文本）
- 数据集根路径：`/data/szs/share/voice_model_project/datasets`

---

## 十、Python 环境

```bash
source /data/szs/250010072/szs/anaconda3/bin/activate
conda activate reap
```

项目以 `pyproject.toml` 管理，源码在 `src/` 下，通过 `PYTHONPATH=src` 或安装后使用。
