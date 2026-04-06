import torch

from modality_reap.merge import MoEExpertMerger


def test_conflict_aware_subspace_merge_preserves_anchor_under_high_conflict():
    anchor = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
    other = torch.tensor([[1.0, 0.0], [0.0, 4.0]])

    merged = MoEExpertMerger._conflict_aware_subspace_merge(
        tensors=[other],
        tensor_weights=torch.tensor([1.0]),
        base_tensor=anchor,
        cluster_conflict=1.0,
        conflict_anchor_strength=0.8,
        subspace_rank_ratio=0.5,
        min_subspace_rank=1,
    )

    average = 0.5 * (anchor + other)
    assert torch.norm(merged - anchor) < torch.norm(average - anchor)
