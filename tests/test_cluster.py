import torch

from modality_reap.cluster import apply_protected_expert_constraints, ensure_symmetric_distance_matrix


def test_apply_protected_expert_constraints_splits_protected_member():
    labels = {0: torch.tensor([0, 0, 1, 1])}
    protected = {0: [1]}
    adjusted = apply_protected_expert_constraints(labels, protected)
    assert adjusted[0][1].item() != adjusted[0][0].item()


def test_ensure_symmetric_distance_matrix_repairs_asymmetry_and_nans():
    matrix = torch.tensor(
        [
            [0.0, 1.0, float("nan")],
            [2.0, 0.0, 3.0],
            [4.0, 5.0, 0.0],
        ]
    )
    repaired = ensure_symmetric_distance_matrix(matrix)
    assert torch.allclose(repaired, repaired.transpose(0, 1))
    assert torch.isfinite(repaired).all()
    assert torch.allclose(torch.diag(repaired), torch.zeros(3))
