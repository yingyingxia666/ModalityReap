from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from modality_reap.data import AUDIO_CORE_EXPERTS, AUDIO_SECONDARY_EXPERTS, CROSS_MODAL_EXPERTS


DEFAULT_AUDIO_PRIOR_WEIGHT = 0.15
DEFAULT_GENERALIST_WEIGHT = 0.20
DEFAULT_TEXT_PENALTY_WEIGHT = 0.10



def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(torch.float32)
    total = tensor.sum()
    if total <= 0:
        return torch.zeros_like(tensor, dtype=torch.float32)
    return tensor / total



def _safe_mean(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("values must not be empty")
    return torch.stack(values).mean(dim=0)



def compute_modality_reference_stats(observation_sets: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    results: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for modality, layers in observation_sets.items():
        modality_layers: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, layer_state in layers.items():
            expert_frequency = layer_state["expert_frequency"].to(torch.float32)
            weighted_frequency = layer_state.get("weighted_expert_frequency_sum", torch.zeros_like(expert_frequency)).to(torch.float32)
            modality_layers[layer_idx] = {
                "expert_frequency": expert_frequency,
                "expert_frequency_norm": _normalize_tensor(expert_frequency),
                "weighted_frequency": weighted_frequency,
                "weighted_frequency_norm": _normalize_tensor(weighted_frequency),
            }
        results[modality] = modality_layers
    return results



def score_experts(observation_sets: dict[str, dict[int, dict[str, Any]]], use_router_analysis_prior: bool = True) -> dict[int, dict[str, torch.Tensor | list[int]]]:
    modality_stats = compute_modality_reference_stats(observation_sets)
    audio_layers = modality_stats.get("audio", {})
    text_layers = modality_stats.get("text", {})
    layer_ids = sorted(set(audio_layers) | set(text_layers))

    layer_scores: dict[int, dict[str, torch.Tensor | list[int]]] = {}
    for layer_idx in layer_ids:
        audio_state = audio_layers.get(layer_idx)
        text_state = text_layers.get(layer_idx)
        if audio_state is None and text_state is None:
            continue

        if audio_state is None:
            num_experts = text_state["expert_frequency"].shape[0]
            zeros = torch.zeros(num_experts, dtype=torch.float32)
            audio_freq = zeros
            audio_weighted = zeros
        else:
            num_experts = audio_state["expert_frequency"].shape[0]
            audio_freq = audio_state["expert_frequency_norm"]
            audio_weighted = audio_state["weighted_frequency_norm"]

        if text_state is None:
            text_freq = torch.zeros(num_experts, dtype=torch.float32)
            text_weighted = torch.zeros(num_experts, dtype=torch.float32)
        else:
            text_freq = text_state["expert_frequency_norm"]
            text_weighted = text_state["weighted_frequency_norm"]

        generalist_score = torch.minimum(audio_freq, text_freq)
        modal_gap_score = torch.abs(audio_freq - text_freq)
        weighted_modal_gap_score = torch.abs(audio_weighted - text_weighted)
        conflict_score = 0.5 * modal_gap_score + 0.5 * weighted_modal_gap_score
        audio_priority_score = 0.5 * audio_freq + 0.5 * audio_weighted + DEFAULT_GENERALIST_WEIGHT * generalist_score
        audio_priority_score = audio_priority_score - DEFAULT_TEXT_PENALTY_WEIGHT * text_weighted

        if use_router_analysis_prior:
            prior_bonus = torch.zeros(num_experts, dtype=torch.float32)
            for expert_id in AUDIO_CORE_EXPERTS:
                if expert_id < num_experts:
                    prior_bonus[expert_id] += DEFAULT_AUDIO_PRIOR_WEIGHT
            for expert_id in AUDIO_SECONDARY_EXPERTS:
                if expert_id < num_experts:
                    prior_bonus[expert_id] += DEFAULT_AUDIO_PRIOR_WEIGHT / 2
            for expert_id in CROSS_MODAL_EXPERTS:
                if expert_id < num_experts:
                    prior_bonus[expert_id] += DEFAULT_AUDIO_PRIOR_WEIGHT / 2
            audio_priority_score = audio_priority_score + prior_bonus

        ranked_experts = torch.argsort(audio_priority_score, descending=True).tolist()
        layer_scores[layer_idx] = {
            "audio_activation_score": audio_freq,
            "text_activation_score": text_freq,
            "weighted_audio_score": audio_weighted,
            "weighted_text_score": text_weighted,
            "generalist_score": generalist_score,
            "modal_gap_score": modal_gap_score,
            "weighted_modal_gap_score": weighted_modal_gap_score,
            "conflict_score": conflict_score,
            "audio_priority_score": audio_priority_score,
            "ranked_experts": ranked_experts,
        }
    return layer_scores



def select_protected_experts(
    layer_scores: dict[int, dict[str, torch.Tensor | list[int]]],
    protect_ratio: float,
    protect_middle_layers: bool,
    middle_layer_multiplier: float,
) -> dict[int, list[int]]:
    protected: dict[int, list[int]] = {}
    if not layer_scores:
        return protected
    max_layer = max(layer_scores)
    middle_start = max_layer // 3
    middle_end = (2 * max_layer) // 3

    for layer_idx, score_dict in layer_scores.items():
        scores = score_dict["audio_priority_score"]
        assert isinstance(scores, torch.Tensor)
        num_experts = scores.shape[0]
        this_ratio = protect_ratio
        if protect_middle_layers and middle_start <= layer_idx <= middle_end:
            this_ratio *= middle_layer_multiplier
        keep = max(1, min(num_experts, int(round(num_experts * this_ratio))))
        protected[layer_idx] = torch.argsort(scores, descending=True)[:keep].tolist()
    return protected



def save_scores(score_dir: Path, layer_scores: dict[int, dict[str, Any]], protected_experts: dict[int, list[int]]) -> None:
    score_dir.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, Any] = {"layers": {}}
    for layer_idx, score_dict in layer_scores.items():
        serializable["layers"][str(layer_idx)] = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in score_dict.items()
        }
        serializable["layers"][str(layer_idx)]["protected_experts"] = protected_experts.get(layer_idx, [])

    with (score_dir / "scores.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)
