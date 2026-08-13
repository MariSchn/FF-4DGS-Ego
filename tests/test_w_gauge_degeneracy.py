"""Pin the two failure modes of our W-MPJPE gauge that the Hand3R protocol exposed.

Hand3R reports Short Video and Long Video side by side, both with WA-MPJPE and W-MPJPE. To match
that layout we have to score a 30-frame segment, and at 30 frames our own gauge stops measuring
what its name says: the alignment window equals the segment, so the whole segment is rigidly fitted
and no drift can accumulate. Their two-frame gauge does not have this property at any length.

These tests fail if someone widens the alignment window towards the segment length, or narrows it
to two and calls the result ours.
"""
from __future__ import annotations

import torch

from scripts.eval_worldspace_baseline import HAND3R_W_ALIGN_FRAMES
from scripts.world_space_metrics import (
    solve_similarity,
    apply_similarity,
    w_mpjpe_first_window_aligned,
)


def _drifting(t: int = 100, j: int = 16, seed: int = 5):
    """A prediction that tracks the truth locally but drifts away globally, which is the only
    regime in which W and WA are supposed to differ.

    Drift and travel are per-frame RATES, not totals. An earlier version of this fixture spread a
    fixed total drift over ``t`` with ``linspace``, which made a 30-frame clip drift as far as a
    100-frame one and quietly destroyed the premise of the length test below.
    """
    g = torch.Generator().manual_seed(seed)
    frame = torch.arange(t, dtype=torch.float64)[:, None]
    shape = torch.randn(j, 3, generator=g).double() * 0.03
    travel = frame * 0.009 * torch.tensor([1.0, 0.15, 0.4], dtype=torch.float64)
    gt = shape[None] + travel[:, None, :] + torch.randn(t, j, 3, generator=g).double() * 0.002
    drift = frame * 0.0008 * torch.tensor([0.4, 1.0, -0.3], dtype=torch.float64)
    pred = gt + drift[:, None, :] + torch.randn(t, j, 3, generator=g).double() * 0.003
    return pred, gt


def test_our_gauge_degenerates_when_the_window_is_the_segment():
    """At segment 30 with wa_short 30, our W-MPJPE IS a full-segment rigid fit, not a drift measure.

    Proven by construction rather than by inspection: aligning the segment rigidly with a transform
    solved on the whole segment gives the same number, to the decimal.
    """
    pred, gt = _drifting(t=30)
    ours = w_mpjpe_first_window_aligned(pred, gt, None, wa_short=30)

    _, rot, _ = solve_similarity(pred.reshape(-1, 3), gt.reshape(-1, 3))
    trans = gt.reshape(-1, 3).mean(0) - rot @ pred.reshape(-1, 3).mean(0)
    rigid = float((apply_similarity(pred, torch.tensor(1.0, dtype=pred.dtype), rot, trans) - gt)
                  .norm(dim=-1).mean() * 1000.0)
    assert abs(ours - rigid) < 1e-6, (
        f"expected the 30/30 gauge to be a whole-segment rigid fit, got {ours:.4f} vs {rigid:.4f}")


def test_hand3r_gauge_still_measures_drift_at_thirty_frames():
    """Their gauge stays a drift measure where ours collapses, which is why the short-video column
    must use it."""
    pred, gt = _drifting(t=30)
    ours = w_mpjpe_first_window_aligned(pred, gt, None, wa_short=30)
    theirs = w_mpjpe_first_window_aligned(pred, gt, None, wa_short=HAND3R_W_ALIGN_FRAMES)
    assert theirs > ours, (
        f"a two-frame fit cannot beat a whole-segment fit on the segment it is scored on; "
        f"got theirs {theirs:.2f} vs ours {ours:.2f}")


def test_the_gap_widens_with_segment_length():
    """Drift accumulates, so the penalty for anchoring on two frames grows with the segment. If
    this ever flattens, the predictions under test are not drifting and the comparison is vacuous.
    """
    gaps = []
    for t in (30, 100):
        pred, gt = _drifting(t=t)
        gaps.append(w_mpjpe_first_window_aligned(pred, gt, None, HAND3R_W_ALIGN_FRAMES)
                    - w_mpjpe_first_window_aligned(pred, gt, None, 30))
    assert gaps[1] > gaps[0] > 0, f"gauge gap did not grow with segment length: {gaps}"


def test_baseline_scorer_emits_both_gauges():
    """The per-segment record carries both W columns, so a table can never silently mix them."""
    from scripts.eval_worldspace_baseline import aggregate
    rows = [{"W_MPJPE": 70.0, "W_MPJPE_h3r": 95.0, "WA_MPJPE_short": 27.0,
             "WA_MPJPE_long": 30.0, "C_MPJPE": 26.0, "C_MPJPE_abs": 35.0}]
    agg = aggregate(rows, seq_c_rows=None)
    assert agg["W_MPJPE"] == 70.0 and agg["W_MPJPE_h3r"] == 95.0
