import torch

from modality_reap.scoring import score_experts, select_protected_experts


def test_score_experts_prefers_audio_signal():
    observation_sets = {
        "audio": {
            0: {
                "expert_frequency": torch.tensor([10.0, 1.0, 0.0]),
                "weighted_expert_frequency_sum": torch.tensor([9.0, 0.5, 0.0]),
            }
        },
        "text": {
            0: {
                "expert_frequency": torch.tensor([1.0, 8.0, 0.0]),
                "weighted_expert_frequency_sum": torch.tensor([0.5, 7.5, 0.0]),
            }
        },
    }

    scores = score_experts(observation_sets, use_router_analysis_prior=False)
    layer0 = scores[0]["audio_priority_score"]
    assert int(torch.argmax(layer0).item()) == 0


def test_select_protected_experts_returns_at_least_one():
    layer_scores = {
        0: {
            "audio_priority_score": torch.tensor([0.9, 0.2, 0.1]),
        }
    }
    protected = select_protected_experts(layer_scores, protect_ratio=0.1, protect_middle_layers=False, middle_layer_multiplier=1.5)
    assert protected[0] == [0]
