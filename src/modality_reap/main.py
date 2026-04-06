from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoTokenizer, HfArgumentParser

from modality_reap.args import (
    ClusterArgs,
    DataArgs,
    MergeArgs,
    ModelArgs,
    ObserverArgs,
    ReapArgs,
    ReportArgs,
    ensure_output_dir,
)
from modality_reap.cluster import (
    apply_protected_expert_constraints,
    dynamic_frequency_penalized_clustering,
    get_penalty_vector,
    hierarchical_clustering,
    restricted_hierarchical_clustering,
)
from modality_reap.data import load_modality_samples, materialize_model_inputs
from modality_reap.merge import MergeMethod, MoEExpertMerger
from modality_reap.model_util import MODEL_ATTRS, assert_merge, get_moe
from modality_reap.observer import MoETransformerObserver, OBSERVER_CONFIG_REGISTRY
from modality_reap.reporting import save_json, summarize_warnings
from modality_reap.scoring import save_scores, score_experts, select_protected_experts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> tuple[ReapArgs, ModelArgs, DataArgs, ObserverArgs, ClusterArgs, MergeArgs, ReportArgs]:
    parser = HfArgumentParser((ReapArgs, ModelArgs, DataArgs, ObserverArgs, ClusterArgs, MergeArgs, ReportArgs))
    return parser.parse_args_into_dataclasses()


def load_model_and_processors(model_args: ModelArgs):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    logger.info("[阶段 1/8] 加载 Qwen3-Omni 模型与处理器")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_args.model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=model_args.attn_implementation,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=True)
    logger.info("模型加载完成: %s", model.__class__.__name__)
    return model, processor, tokenizer


def summarize_sample_sources(samples: list[dict[str, Any]]) -> str:
    counter = Counter(sample["dataset"] for sample in samples)
    return ", ".join(f"{name}={count}" for name, count in sorted(counter.items()))


def collect_observations(
    model,
    samples: list[dict[str, Any]],
    obs_args: ObserverArgs,
    stage_name: str,
) -> dict[int, dict[str, Any]]:
    logger.info("[阶段 3/8] 开始收集 %s observation，共 %d 条样本", stage_name, len(samples))
    norm_topk_prob = getattr(model.config.thinker_config.text_config, "norm_topk_prob", False)
    observer_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__](
        distance_measure="cosine",
        renormalize_router_weights=norm_topk_prob and obs_args.renormalize_router_weights,
        record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
    )
    observer = MoETransformerObserver(model=model, hook_config=observer_config)
    try:
        with torch.no_grad():
            for sample in tqdm(samples, desc=f"Observing {stage_name}", unit="sample"):
                model_inputs = materialize_model_inputs(model, sample["inputs"])
                model.thinker(**model_inputs)
        state = observer.report_state()
        logger.info("完成 %s observation，收集到 %d 层统计", stage_name, len(state))
        return state
    finally:
        observer.close_hooks()


def cluster_layer(distances: torch.Tensor, expert_prob: torch.Tensor, num_clusters: int, cluster_args: ClusterArgs) -> torch.Tensor:
    distance = distances.clone()
    if cluster_args.frequency_penalty:
        penalty = get_penalty_vector(expert_prob, cluster_args.softmax_temperature)
        penalty_matrix = penalty.unsqueeze(0) + penalty.unsqueeze(1)
        distance = distance * penalty_matrix
        distance[distance.isnan()] = float("inf")

    if cluster_args.cluster_method == "agglomerative":
        if cluster_args.max_cluster_size is None:
            labels = hierarchical_clustering(distance, cluster_args.linkage_method, num_clusters)
        else:
            labels = restricted_hierarchical_clustering(
                distance,
                cluster_args.linkage_method,
                num_clusters,
                cluster_args.max_cluster_size,
            )
        return torch.tensor(labels, dtype=torch.long)
    if cluster_args.cluster_method == "dynamic_ttm":
        return dynamic_frequency_penalized_clustering(
            distance,
            expert_prob,
            num_clusters,
            cluster_args.softmax_temperature,
        )
    raise NotImplementedError(f"Unsupported cluster method: {cluster_args.cluster_method}")


def build_cluster_labels(
    observation_data: dict[int, dict[str, Any]],
    cluster_args: ClusterArgs,
    protected_experts: dict[int, list[int]],
) -> dict[int, torch.Tensor]:
    logger.info("[阶段 5/8] 开始逐层聚类")
    cluster_labels: dict[int, torch.Tensor] = {}
    for layer_idx, layer_state in tqdm(sorted(observation_data.items()), desc="Clustering layers", unit="layer"):
        expert_prob = layer_state["expert_frequency"].to(torch.float32)
        total = expert_prob.sum()
        if total > 0:
            expert_prob = expert_prob / total
        distance = layer_state.get("router_logit_similiarity")
        if distance is None:
            distance = layer_state["online_characteristic_activation_dist"]
        distance = distance.to(torch.float32)
        num_experts = expert_prob.shape[0]
        num_protected = len(protected_experts.get(layer_idx, []))
        requested_clusters = cluster_args.num_clusters or int(num_experts * (1 - cluster_args.compression_ratio))
        requested_clusters = max(requested_clusters, num_protected, 1)
        requested_clusters = min(requested_clusters, num_experts)
        cluster_labels[layer_idx] = cluster_layer(distance, expert_prob, requested_clusters, cluster_args)

    cluster_labels = apply_protected_expert_constraints(cluster_labels, protected_experts)
    logger.info("聚类完成，共处理 %d 层", len(cluster_labels))
    return cluster_labels


def merge_model(
    model,
    cluster_labels: dict[int, torch.Tensor],
    observation_data: dict[int, dict[str, Any]],
    merge_args: MergeArgs,
    protected_experts: dict[int, list[int]],
):
    logger.info("[阶段 6/8] 开始 merge experts")
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    for layer_idx, cluster_label in tqdm(sorted(cluster_labels.items()), desc="Merging layers", unit="layer"):
        expert_proba = observation_data[layer_idx]["expert_frequency"].to(torch.float32)
        total = expert_proba.sum()
        if total > 0:
            expert_proba = expert_proba / total

        protected = set(protected_experts.get(layer_idx, []))
        if protected:
            adjusted = expert_proba.clone()
            for expert_id in protected:
                if expert_id < adjusted.shape[0]:
                    adjusted[expert_id] = adjusted[expert_id] + merge_args.conservative_anchor_weight
            expert_proba = adjusted / adjusted.sum()

        moe = get_moe(model, layer_idx)
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_label,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=MergeMethod(merge_args.merge_method),
            dom_as_base=merge_args.dom_as_base,
            select_top_k=merge_args.select_top_k,
            permute=merge_args.permute,
            tie_tensors=merge_args.save_as_tied_params,
        )
        merger.merge_experts()
        assert_merge(model, moe, cluster_label)
    logger.info("merge 完成")


def compact_fused_experts(model, cluster_labels: dict[int, torch.Tensor], observation_data: dict[int, dict[str, Any]]):
    logger.info("[阶段 7/8] 开始 compact experts")
    for layer_idx, cluster_label in tqdm(sorted(cluster_labels.items()), desc="Compacting layers", unit="layer"):
        moe = get_moe(model, layer_idx)
        expert_proba = observation_data[layer_idx]["expert_frequency"].to(torch.float32)
        retained: list[int] = []
        for cluster_id in torch.unique(cluster_label).tolist():
            members = torch.where(cluster_label == cluster_id)[0]
            best_idx = members[expert_proba[members].argmax()].item()
            retained.append(best_idx)
        retained = sorted(retained)

        moe.experts.gate_up_proj = torch.nn.Parameter(moe.experts.gate_up_proj.data[retained].clone())
        moe.experts.down_proj = torch.nn.Parameter(moe.experts.down_proj.data[retained].clone())
        moe.experts.num_experts = len(retained)
        moe.gate.weight = torch.nn.Parameter(moe.gate.weight.data[retained].clone())
        moe.gate.num_experts = len(retained)

    first_layer = next(iter(cluster_labels))
    new_count = len(torch.unique(cluster_labels[first_layer]))
    model.config.thinker_config.text_config.num_experts = new_count
    logger.info("compact 完成，第一层压缩后 expert 数: %d", new_count)


def save_run_artifacts(
    output_dir: Path,
    run_args: dict[str, Any],
    warnings: list[str],
    observation_sets: dict[str, dict[int, dict[str, Any]]],
    cluster_labels: dict[int, torch.Tensor],
    protected_experts: dict[int, list[int]],
):
    save_json(output_dir / "run_args.json", run_args)
    save_json(output_dir / "warnings.json", summarize_warnings(warnings))

    serializable_observations = {
        modality: {
            str(layer_idx): {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in layer_state.items()
            }
            for layer_idx, layer_state in layers.items()
        }
        for modality, layers in observation_sets.items()
    }
    save_json(output_dir / "observations" / "observations.json", serializable_observations)

    serializable_clusters = {
        str(layer_idx): {
            "labels": labels.tolist(),
            "protected_experts": protected_experts.get(layer_idx, []),
        }
        for layer_idx, labels in cluster_labels.items()
    }
    save_json(output_dir / "clusters" / "clusters.json", serializable_clusters)


def smoke_test(model, tokenizer):
    prompt = [{"role": "user", "content": "请简单介绍你自己。"}]
    inputs = tokenizer.apply_chat_template(
        prompt,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
    ).to(next(model.parameters()).device)
    _ = model.generate(inputs, max_new_tokens=16, do_sample=False)


def main():
    reap_args, model_args, data_args, obs_args, cluster_args, merge_args, report_args = parse_args()
    set_seed(reap_args.seed)
    output_dir = ensure_output_dir(data_args.output_dir)

    logger.info("实验输出目录: %s", output_dir)
    logger.info(
        "目标配置: audio=%d, text=%d, compression_ratio=%.2f",
        data_args.total_audio_samples,
        data_args.total_text_samples,
        cluster_args.compression_ratio,
    )

    model, processor, tokenizer = load_model_and_processors(model_args)

    logger.info("[阶段 2/8] 准备音频与文本样本")
    audio_samples, audio_warnings = load_modality_samples(
        processor=processor,
        modality="audio",
        max_datasets=data_args.max_audio_datasets,
        max_samples_per_dataset=data_args.max_samples_per_dataset,
        sample_seed=data_args.sample_seed,
        max_seq_length=data_args.max_seq_length,
        audio_sample_rate=data_args.audio_sample_rate,
        dataset_root=data_args.dataset_root,
        total_samples=data_args.total_audio_samples,
    )
    text_samples, text_warnings = load_modality_samples(
        processor=processor,
        modality="text",
        max_datasets=data_args.max_text_datasets,
        max_samples_per_dataset=data_args.max_samples_per_dataset,
        sample_seed=data_args.sample_seed,
        max_seq_length=data_args.max_seq_length,
        audio_sample_rate=data_args.audio_sample_rate,
        dataset_root=data_args.dataset_root,
        total_samples=data_args.total_text_samples,
    )
    warnings = audio_warnings + text_warnings

    logger.info(
        "音频样本准备完成: %d 条 (%s)",
        len(audio_samples),
        summarize_sample_sources(audio_samples) if audio_samples else "none",
    )
    logger.info(
        "文本样本准备完成: %d 条 (%s)",
        len(text_samples),
        summarize_sample_sources(text_samples) if text_samples else "none",
    )
    if warnings:
        logger.warning("样本准备告警: %s", warnings)

    observation_sets: dict[str, dict[int, dict[str, Any]]] = {}
    if audio_samples:
        observation_sets["audio"] = collect_observations(model, audio_samples, obs_args, stage_name="audio")
    if text_samples:
        observation_sets["text"] = collect_observations(model, text_samples, obs_args, stage_name="text")
    if not observation_sets:
        raise RuntimeError("No available datasets were loaded. Please verify dataset paths or provide local JSONL data.")

    logger.info("[阶段 4/8] 计算模态打分与保护专家")
    layer_scores = score_experts(observation_sets, use_router_analysis_prior=cluster_args.use_router_analysis_prior)
    protected_experts = select_protected_experts(
        layer_scores,
        protect_ratio=cluster_args.audio_protect_ratio,
        protect_middle_layers=cluster_args.protect_middle_layers,
        middle_layer_multiplier=cluster_args.middle_layer_multiplier,
    )
    save_scores(output_dir / "scores", layer_scores, protected_experts)
    logger.info("打分完成，共 %d 层；保护专家已保存到 %s", len(layer_scores), output_dir / "scores")

    clustering_observation = observation_sets.get("audio") or observation_sets.get("text")
    cluster_labels = build_cluster_labels(clustering_observation, cluster_args, protected_experts)
    merge_model(model, cluster_labels, clustering_observation, merge_args, protected_experts)
    compact_fused_experts(model, cluster_labels, clustering_observation)

    if reap_args.do_save:
        logger.info("[阶段 8/8] 保存压缩模型与实验产物")
        merged_model_dir = output_dir / "merged_model"
        merged_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_model_dir)
        logger.info("模型已保存到: %s", merged_model_dir)

    save_run_artifacts(
        output_dir=output_dir,
        run_args={
            "reap_args": dataclasses.asdict(reap_args),
            "model_args": dataclasses.asdict(model_args),
            "data_args": dataclasses.asdict(data_args),
            "obs_args": dataclasses.asdict(obs_args),
            "cluster_args": dataclasses.asdict(cluster_args),
            "merge_args": dataclasses.asdict(merge_args),
            "report_args": dataclasses.asdict(report_args),
        },
        warnings=warnings,
        observation_sets=observation_sets,
        cluster_labels=cluster_labels,
        protected_experts=protected_experts,
    )
    logger.info("实验产物已写入: %s", output_dir)

    if reap_args.smoke_test:
        logger.info("执行保存后 smoke test")
        smoke_test(model, tokenizer)
        logger.info("smoke test 完成")


if __name__ == "__main__":
    main()
