"""Equivalence test for the prune_gs voxel-grouping speedup.

prune_gs swapped `torch.unique(voxel_indices, dim=0)` (lexicographic row sort,
the 201s/step bottleneck) for a packed-int64-key 1-D unique. The merge math
downstream is unchanged and only consumes `inverse_indices`, so correctness
reduces to: do the two methods induce the *same partition* of points into
voxels? Group labels may be permuted; the grouping itself must be identical.
"""
import torch


def _inverse_old(voxel_indices):
    _, inverse = torch.unique(voxel_indices, dim=0, return_inverse=True)
    return inverse


def _inverse_new(voxel_indices):
    vmin = voxel_indices.min(dim=0).values
    shifted = voxel_indices - vmin
    sizes = shifted.max(dim=0).values + 1
    keys = (shifted[:, 0] * sizes[1] + shifted[:, 1]) * sizes[2] + shifted[:, 2]
    _, inverse = torch.unique(keys, return_inverse=True)
    return inverse


def _same_partition(a, b):
    """True iff inverse-index vectors a, b induce the same grouping."""
    # Two points are in the same group under `a` iff a[i] == a[j]; likewise b.
    # Equivalent partitions <=> the pair (a, b) has a 1:1 label correspondence.
    pair = a * (b.max() + 1) + b
    return torch.unique(pair).numel() == torch.unique(a).numel() == torch.unique(b).numel()


def test_grouping_matches_on_random_voxels():
    torch.manual_seed(0)
    # Mix of distinct and colliding voxels, including negative coords.
    voxels = torch.randint(-50, 50, (5000, 3))
    voxels[:1000] = voxels[1000:2000]  # force duplicate voxels (merges)
    assert _same_partition(_inverse_old(voxels), _inverse_new(voxels))


def test_grouping_matches_when_all_distinct():
    torch.manual_seed(1)
    voxels = torch.arange(300).view(100, 3)  # every row unique
    old, new = _inverse_old(voxels), _inverse_new(voxels)
    assert old.unique().numel() == 100
    assert _same_partition(old, new)


def test_grouping_matches_when_all_same():
    voxels = torch.zeros(64, 3, dtype=torch.long)
    old, new = _inverse_old(voxels), _inverse_new(voxels)
    assert old.unique().numel() == 1
    assert _same_partition(old, new)


def test_merge_result_invariant_to_label_permutation():
    """The actual merged output (scatter_sum by inverse) must agree as a set."""
    torch.manual_seed(2)
    n = 2000
    voxels = torch.randint(0, 30, (n, 3))
    means = torch.randn(n, 3)
    weights = torch.rand(n)

    def merge(inverse):
        ngroups = int(inverse.max()) + 1
        wsum = torch.zeros(ngroups).scatter_add_(0, inverse, weights)
        w = weights / wsum[inverse].clamp_min(1e-8)
        out = torch.zeros(ngroups, 3).scatter_add_(
            0, inverse.unsqueeze(-1).expand(-1, 3), means * w.unsqueeze(-1)
        )
        return out

    old = merge(_inverse_old(voxels))
    new = merge(_inverse_new(voxels))
    # Same set of merged gaussians, modulo group ordering.
    old_sorted = old[old[:, 0].argsort()]
    new_sorted = new[new[:, 0].argsort()]
    assert torch.allclose(old_sorted, new_sorted, atol=1e-5)
