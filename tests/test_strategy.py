from types import SimpleNamespace

import torch

from modality_reap.strategy import build_hybrid_compression_plan, build_layer_adaptive_schedule


def make_cluster_args(**overrides):
    defaults = {
        "compression_ratio": 0.5,
        "min_compression_ratio": 0.2,
        "max_compression_ratio": 0.8,
        "use_layer_adaptive_schedule": True,
        "protect_middle_layers": False,
        "middle_layer_multiplier": 1.5,
        "sensitivity_top1_weight": 0.35,
        "sensitivity_cv_weight": 0.3,
        "sensitivity_modal_gap_weight": 0.25,
        "sensitivity_shared_hotspot_weight": 0.1,
        "sensitivity_active_expert_weight": 0.2,
        "audio_keep_quantile": 0.75,
        "shared_keep_quantile": 0.75,
        "prune_audio_quantile": 0.25,
        "prune_shared_quantile": 0.25,
        "prune_conflict_quantile": 0.4,
        "use_router_analysis_prior": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_layer_adaptive_schedule_reduces_sensitive_layer_compression():
    cluster_args = make_cluster_args()
    layer_scores = {
        0: {
            "audio_activation_score": torch.tensor([0.9, 0.1, 0.0]),
            "text_activation_score": torch.tensor([0.0, 0.9, 0.1]),
            "generalist_score": torch.tensor([0.0, 0.1, 0.0]),
            "conflict_score": torch.tensor([0.9, 0.8, 0.2]),
            "audio_priority_score": torch.tensor([0.8, 0.2, 0.1]),
        },
        1: {
            "audio_activation_score": torch.tensor([0.34, 0.33, 0.33]),
            "text_activation_score": torch.tensor([0.34, 0.33, 0.33]),
            "generalist_score": torch.tensor([0.34, 0.33, 0.33]),
            "conflict_score": torch.tensor([0.1, 0.1, 0.1]),
            "audio_priority_score": torch.tensor([0.4, 0.35, 0.3]),
        },
    }

    schedule = build_layer_adaptive_schedule(layer_scores, cluster_args)
    assert schedule[0]["compression_ratio"] < schedule[1]["compression_ratio"]


def test_build_hybrid_compression_plan_keeps_shared_and_prunes_tail():
    cluster_args = make_cluster_args()
    layer_scores = {
        0: {
            "audio_activation_score": torch.tensor([0.7, 0.1, 0.15, 0.05]),
            "text_activation_score": torch.tensor([0.1, 0.7, 0.1, 0.1]),
            "generalist_score": torch.tensor([0.1, 0.7, 0.08, 0.01]),
            "conflict_score": torch.tensor([0.6, 0.4, 0.2, 0.01]),
            "audio_priority_score": torch.tensor([0.92, 0.58, 0.23, 0.03]),
        }
    }
    layer_schedule = {
        0: {
            "compression_ratio": 0.5,
            "target_experts": 3,
            "sensitivity_score": 0.4,
            "normalized_sensitivity": 0.5,
            "num_experts": 4,
            "top1_share": 0.7,
            "load_balance_cv": 0.1,
            "modal_gap": 0.3,
            "shared_hotspot_share": 0.8,
            "active_experts": 4,
            "active_ratio": 1.0,
        }
    }

    plans = build_hybrid_compression_plan(layer_scores, layer_schedule, cluster_args)
    plan = plans[0]

    assert 0 in plan.audio_core_experts
    assert 1 in plan.shared_experts
    assert 2 in plan.merge_experts
    assert 3 in plan.pruned_experts
    assert plan.decision_labels[0] == "keep_audio"
    assert plan.decision_labels[1] == "keep_shared"

