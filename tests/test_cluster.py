import torch

from modality_reap.cluster import apply_protected_expert_constraints


def test_apply_protected_expert_constraints_splits_protected_member():
    labels = {0: torch.tensor([0, 0, 1, 1])}
    protected = {0: [1]}
    adjusted = apply_protected_expert_constraints(labels, protected)
    assert adjusted[0][1].item() != adjusted[0][0].item()
