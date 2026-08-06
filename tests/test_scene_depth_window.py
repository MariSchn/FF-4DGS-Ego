"""The scene-depth read at a hand pixel is contaminated in ONE direction only.

Task #63. The scene scale is s = med(z_hand / d_scene). Measured against the GT camera scale over
1884 segments, s_hand 0.6208 vs s_gt 1.0230, ratio 0.578: our scale is 42% too small. The hand
depth is not the culprit (C-MPJPE absolute ~36mm at ~0.7m, ~5%), so d_scene at those pixels reads
about 1.7x too far.

The mechanism is one-sided. At a hand joint's pixel the nearest visible surface is the hand, so
misregistration or silhouette blur can only blend in BACKGROUND, which is farther. d_scene can be
pushed up and essentially never down, so s is biased down. Under one-sided contamination the right
estimator is a low-order statistic over a small neighbourhood, not the mean-like bilinear blend.

These tests pin that claim on synthetic depth maps where the truth is known exactly, so the
estimator is validated independently of whether it later helps on real data.
"""
from __future__ import annotations

import pytest
import torch

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    sample_depth_at_joints,
)

HAND_D = 0.50      # metres, the hand surface
BG_D = 0.85        # metres, the background behind it


def _hand_on_background(res: int = 32, hand_px: int = 8) -> torch.Tensor:
    """A depth map that is background everywhere except a small square 'hand' patch."""
    d = torch.full((1, 1, res, res), BG_D)
    c = res // 2
    h = hand_px // 2
    d[..., c - h:c + h, c - h:c + h] = HAND_D
    return d


def _grid_at(res: int, row: float, col: float) -> torch.Tensor:
    """[B,S,H,J,2] normalised grid addressing depth-map pixel (row, col), pixel-centre convention."""
    return torch.tensor([(col + 0.5) / res, (row + 0.5) / res]).view(1, 1, 1, 1, 2)


def test_bilinear_read_at_the_hand_centre_is_exact():
    """Sanity: with no misregistration the plain read is already correct, so the window is only
    ever compensating for an actual offset, not papering over a broken projection."""
    res = 32
    d = _hand_on_background(res)
    g = _grid_at(res, res // 2, res // 2)
    got, in_frame = sample_depth_at_joints(d, g)
    assert bool(in_frame.all())
    assert abs(float(got.reshape(-1)[0]) - HAND_D) < 1e-5


def test_misregistered_read_is_biased_FAR_and_never_near():
    """A joint whose projection lands on the hand's edge reads too FAR, in the direction that
    biases the solved scale DOWN. This is the defect, reproduced.
    """
    res = 32
    d = _hand_on_background(res, hand_px=8)
    c = res // 2
    edge = c + 4                                  # one pixel outside the hand square
    got = float(sample_depth_at_joints(d, _grid_at(res, c, edge))[0].reshape(-1)[0])

    assert got > HAND_D, "fixture wrong: the offset read should not still be pure hand"
    assert got <= BG_D + 1e-6, "a read can never exceed the farthest surface present"
    # s = z/d, so an inflated d shrinks s. Quantify it the way the eval would.
    s_true = HAND_D / HAND_D
    s_biased = HAND_D / got
    assert s_biased < s_true, (
        f"contamination must bias the SCALE DOWN, got d={got:.4f} -> s={s_biased:.4f}")


def test_window_min_recovers_the_hand_depth_under_misregistration():
    """The fix: a low-order statistic over a small window recovers the near surface."""
    res = 32
    d = _hand_on_background(res, hand_px=8)
    c = res // 2
    edge = c + 4

    plain = float(sample_depth_at_joints(d, _grid_at(res, c, edge))[0].reshape(-1)[0])
    win = float(sample_depth_at_joints(d, _grid_at(res, c, edge),
                                       window=3, reduce="min")[0].reshape(-1)[0])

    assert win < plain, "window-min must pull the read toward the near surface"
    assert abs(win - HAND_D) < 1e-4, (
        f"window-min should recover the hand depth {HAND_D}, got {win:.4f}")


def test_window_min_does_not_corrupt_a_clean_read():
    """It must not drag a correct read toward some nearer surface that is not there.

    On a flat region every window sample is identical, so min is a no-op. Without this, a fix that
    simply always reports a nearer depth would pass the test above and be wrong everywhere else.
    """
    res = 32
    flat = torch.full((1, 1, res, res), 0.7)
    g = _grid_at(res, 10, 10)
    plain = float(sample_depth_at_joints(flat, g)[0].reshape(-1)[0])
    win = float(sample_depth_at_joints(flat, g, window=3, reduce="min")[0].reshape(-1)[0])
    assert abs(plain - 0.7) < 1e-6 and abs(win - 0.7) < 1e-6


def test_quantile_reduce_sits_between_min_and_plain():
    """qNN is the softer variant; it must actually behave like one rather than alias to min."""
    res = 32
    d = _hand_on_background(res, hand_px=8)
    c = res // 2
    g = _grid_at(res, c, c + 4)
    lo = float(sample_depth_at_joints(d, g, window=3, reduce="min")[0].reshape(-1)[0])
    mid = float(sample_depth_at_joints(d, g, window=3, reduce="q50")[0].reshape(-1)[0])
    hi = float(sample_depth_at_joints(d, g)[0].reshape(-1)[0])
    assert lo <= mid, f"q50 {mid} must not undercut min {lo}"
    assert mid <= hi + 1e-6 or abs(mid - hi) < 1e-6


def test_in_frame_is_unchanged_by_the_window():
    """Validity must depend on the joint, not on which estimator is used, or the two A/B arms
    would be scored over different joint populations and the comparison would be meaningless."""
    res = 32
    d = _hand_on_background(res)
    inside = _grid_at(res, 5, 5)
    outside = torch.tensor([1.4, 0.5]).view(1, 1, 1, 1, 2)
    for g, expect in ((inside, True), (outside, False)):
        a = sample_depth_at_joints(d, g)[1]
        b = sample_depth_at_joints(d, g, window=3, reduce="min")[1]
        assert bool(a.all()) is expect and bool(b.all()) is expect
        assert torch.equal(a, b)


def test_bad_arguments_are_rejected():
    res = 32
    d = _hand_on_background(res)
    g = _grid_at(res, 5, 5)
    with pytest.raises(ValueError, match="odd"):
        sample_depth_at_joints(d, g, window=2, reduce="min")
    with pytest.raises(ValueError, match="bilinear"):
        sample_depth_at_joints(d, g, window=3, reduce="bilinear")
    with pytest.raises(ValueError, match="unknown reduce"):
        sample_depth_at_joints(d, g, window=3, reduce="mean")
