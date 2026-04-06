from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from modality_reap.data import AUDIO_CORE_EXPERTS, CROSS_MODAL_EXPERTS


@dataclass(frozen=True)
class LayerCompressionPlan:
    layer_idx: int
    num_experts: int
    target_experts: int
    target_compression_ratio: float
    sensitivity_score: float
    normalized_sensitivity: float
    merge_cluster_count: int
    keep_experts: list[int]
    audio_core_experts: list[int]
    shared_experts: list[int]
    merge_experts: list[int]
    pruned_experts: list[int]
    decision_labels: list[str]
    layer_stats: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quantile_threshold(values: torch.Tensor, quantile: float) -> float:
    if values.numel() == 0:
        return 0.0
    quantile = min(max(float(quantile), 0.0), 1.0)
    return torch.quantile(values.to(torch.float32), quantile).item()


def _sorted_mask_indices(mask: torch.Tensor) -> list[int]:
    return torch.where(mask)[0].tolist()


def _normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float32)
    total = values.sum()
    if total <= 0:
        return torch.zeros_like(values)
    return values / total


def build_layer_adaptive_schedule(
    layer_scores: dict[int, dict[str, Any]],
    cluster_args: Any,
) -> dict[int, dict[str, float | int]]:
    if not layer_scores:
        return {}

    layer_ids = sorted(layer_scores)
    max_layer = max(layer_ids)
    middle_start = max_layer // 3
    middle_end = (2 * max_layer) // 3

    raw_sensitivity: dict[int, float] = {}
    stats_by_layer: dict[int, dict[str, float | int]] = {}
    for layer_idx in layer_ids:
        score_dict = layer_scores[layer_idx]
        audio_score = score_dict["audio_activation_score"].to(torch.float32)
        text_score = score_dict["text_activation_score"].to(torch.float32)
        generalist_score = score_dict["generalist_score"].to(torch.float32)
        conflict_score = score_dict["conflict_score"].to(torch.float32)

        combined = _normalize_tensor(audio_score + text_score)
        num_experts = int(combined.shape[0])
        active_experts = int(torch.count_nonzero(combined > 0).item())
        active_ratio = active_experts / max(num_experts, 1)
        top1_share = combined.max().item() if combined.numel() else 0.0

        nonzero = combined[combined > 0]
        if nonzero.numel() <= 1:
            load_balance_cv = 0.0
        else:
            mean_value = nonzero.mean()
            load_balance_cv = (nonzero.std(unbiased=False) / mean_value.clamp(min=1e-8)).item()

        modal_gap = conflict_score.mean().item() if conflict_score.numel() else 0.0
        shared_hotspot_threshold = generalist_score.mean()
        shared_hotspot_share = generalist_score[generalist_score >= shared_hotspot_threshold].sum().item()

        sensitivity = (
            cluster_args.sensitivity_top1_weight * top1_share
            + cluster_args.sensitivity_cv_weight * load_balance_cv
            + cluster_args.sensitivity_modal_gap_weight * modal_gap
            + cluster_args.sensitivity_shared_hotspot_weight * shared_hotspot_share
            - cluster_args.sensitivity_active_expert_weight * active_ratio
        )
        if cluster_args.protect_middle_layers and middle_start <= layer_idx <= middle_end:
            sensitivity *= cluster_args.middle_layer_multiplier

        raw_sensitivity[layer_idx] = float(sensitivity)
        stats_by_layer[layer_idx] = {
            "num_experts": num_experts,
            "top1_share": top1_share,
            "load_balance_cv": load_balance_cv,
            "modal_gap": modal_gap,
            "shared_hotspot_share": shared_hotspot_share,
            "active_experts": active_experts,
            "active_ratio": active_ratio,
        }

    if not cluster_args.use_layer_adaptive_schedule or len(layer_ids) == 1:
        normalized = {layer_idx: 0.5 for layer_idx in layer_ids}
    else:
        min_sensitivity = min(raw_sensitivity.values())
        max_sensitivity = max(raw_sensitivity.values())
        if max_sensitivity - min_sensitivity < 1e-8:
            normalized = {layer_idx: 0.5 for layer_idx in layer_ids}
        else:
            normalized = {
                layer_idx: (raw_sensitivity[layer_idx] - min_sensitivity) / (max_sensitivity - min_sensitivity)
                for layer_idx in layer_ids
            }

    min_ratio = min(cluster_args.min_compression_ratio, cluster_args.max_compression_ratio)
    max_ratio = max(cluster_args.min_compression_ratio, cluster_args.max_compression_ratio)
    base_ratio = min(max(cluster_args.compression_ratio, min_ratio), max_ratio)

    schedule: dict[int, dict[str, float | int]] = {}
    for layer_idx in layer_ids:
        num_experts = int(stats_by_layer[layer_idx]["num_experts"])
        if cluster_args.use_layer_adaptive_schedule and len(layer_ids) > 1:
            compression_ratio = max_ratio - normalized[layer_idx] * (max_ratio - min_ratio)
        else:
            compression_ratio = base_ratio
        compression_ratio = min(max(compression_ratio, min_ratio), max_ratio)
        target_experts = max(1, min(num_experts, int(round(num_experts * (1 - compression_ratio)))))

        schedule[layer_idx] = {
            **stats_by_layer[layer_idx],
            "compression_ratio": float(compression_ratio),
            "target_experts": target_experts,
            "sensitivity_score": raw_sensitivity[layer_idx],
            "normalized_sensitivity": normalized[layer_idx],
        }
    return schedule


def build_hybrid_compression_plan(
    layer_scores: dict[int, dict[str, Any]],
    layer_schedule: dict[int, dict[str, float | int]],
    cluster_args: Any,
) -> dict[int, LayerCompressionPlan]:
    plans: dict[int, LayerCompressionPlan] = {}
    for layer_idx, score_dict in layer_scores.items():
        audio_priority = score_dict["audio_priority_score"].to(torch.float32)
        shared_score = score_dict["generalist_score"].to(torch.float32)
        conflict_score = score_dict["conflict_score"].to(torch.float32)
        num_experts = int(audio_priority.shape[0])

        audio_keep_threshold = _quantile_threshold(audio_priority, cluster_args.audio_keep_quantile)
        shared_keep_threshold = _quantile_threshold(shared_score, cluster_args.shared_keep_quantile)
        prune_audio_threshold = _quantile_threshold(audio_priority, cluster_args.prune_audio_quantile)
        prune_shared_threshold = _quantile_threshold(shared_score, cluster_args.prune_shared_quantile)
        prune_conflict_threshold = _quantile_threshold(conflict_score, cluster_args.prune_conflict_quantile)

        audio_core_mask = audio_priority >= audio_keep_threshold
        shared_mask = shared_score >= shared_keep_threshold
        if cluster_args.use_router_analysis_prior:
            for expert_id in AUDIO_CORE_EXPERTS:
                if expert_id < num_experts:
                    audio_core_mask[expert_id] = True
            for expert_id in CROSS_MODAL_EXPERTS:
                if expert_id < num_experts:
                    shared_mask[expert_id] = True

        keep_mask = audio_core_mask | shared_mask
        prune_mask = (
            (audio_priority <= prune_audio_threshold)
            & (shared_score <= prune_shared_threshold)
            & (conflict_score <= prune_conflict_threshold)
            & ~keep_mask
        )
        merge_mask = ~(keep_mask | prune_mask)

        schedule_entry = layer_schedule[layer_idx]
        target_experts = max(int(schedule_entry["target_experts"]), int(keep_mask.sum().item()), 1)
        target_experts = min(target_experts, num_experts)
        merge_budget = max(target_experts - int(keep_mask.sum().item()), 0)

        merge_candidates = _sorted_mask_indices(merge_mask)
        pruned_candidates = _sorted_mask_indices(prune_mask)

        if merge_budget > 0 and len(merge_candidates) < merge_budget:
            merge_preference = audio_priority + shared_score + 0.5 * conflict_score
            rescue_candidates = sorted(
                pruned_candidates,
                key=lambda expert_idx: (float(merge_preference[expert_idx].item()), -expert_idx),
                reverse=True,
            )
            rescued = rescue_candidates[: merge_budget - len(merge_candidates)]
            rescue_set = set(rescued)
            merge_candidates = sorted(merge_candidates + rescued)
            pruned_candidates = [expert_idx for expert_idx in pruned_candidates if expert_idx not in rescue_set]

        if merge_budget == 0:
            pruned_experts = sorted(
                expert_idx for expert_idx in range(num_experts) if expert_idx not in set(_sorted_mask_indices(keep_mask))
            )
            merge_candidates = []
        else:
            pruned_experts = sorted(
                expert_idx for expert_idx in range(num_experts)
                if expert_idx not in set(_sorted_mask_indices(keep_mask)) and expert_idx not in set(merge_candidates)
            )

        merge_cluster_count = min(len(merge_candidates), merge_budget)

        audio_core_experts = sorted(
            expert_idx
            for expert_idx in _sorted_mask_indices(keep_mask)
            if audio_core_mask[expert_idx].item()
        )
        shared_experts = sorted(
            expert_idx
            for expert_idx in _sorted_mask_indices(keep_mask)
            if shared_mask[expert_idx].item() and expert_idx not in set(audio_core_experts)
        )

        decision_labels = ["prune"] * num_experts
        for expert_idx in merge_candidates:
            decision_labels[expert_idx] = "merge"
        for expert_idx in shared_experts:
            decision_labels[expert_idx] = "keep_shared"
        for expert_idx in audio_core_experts:
            decision_labels[expert_idx] = "keep_audio"

        plans[layer_idx] = LayerCompressionPlan(
            layer_idx=layer_idx,
            num_experts=num_experts,
            target_experts=target_experts,
            target_compression_ratio=float(schedule_entry["compression_ratio"]),
            sensitivity_score=float(schedule_entry["sensitivity_score"]),
            normalized_sensitivity=float(schedule_entry["normalized_sensitivity"]),
            merge_cluster_count=merge_cluster_count,
            keep_experts=_sorted_mask_indices(keep_mask),
            audio_core_experts=audio_core_experts,
            shared_experts=shared_experts,
            merge_experts=merge_candidates,
            pruned_experts=pruned_experts,
            decision_labels=decision_labels,
            layer_stats={
                key: float(value) if isinstance(value, (int, float)) else float(value)
                for key, value in schedule_entry.items()
                if key not in {"target_experts"}
            },
        )
    return plans


def build_cluster_conflict_scores(
    layer_scores: dict[int, dict[str, Any]],
    cluster_labels: dict[int, torch.Tensor],
) -> dict[int, dict[int, float]]:
    conflict_scores: dict[int, dict[int, float]] = {}
    for layer_idx, labels in cluster_labels.items():
        layer_conflict = layer_scores[layer_idx]["conflict_score"].to(torch.float32)
        layer_cluster_conflict: dict[int, float] = {}
        for cluster_id in torch.unique(labels):
            cluster_value = int(cluster_id.item())
            if cluster_value < 0:
                continue
            members = torch.where(labels == cluster_id)[0]
            layer_cluster_conflict[cluster_value] = float(layer_conflict[members].mean().item())
        conflict_scores[layer_idx] = layer_cluster_conflict
    return conflict_scores
