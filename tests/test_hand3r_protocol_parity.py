"""Prove our torch port of Hand3R's protocol reproduces their NumPy reference scorer.

This test is the load-bearing part of the whole Hand3R comparison. Once it passes, any difference
between a Hand3R-protocol number and our own number is attributable to the protocol itself, not to
two teams having written Umeyama slightly differently. Without it, every downstream claim about
"the gauge costs X mm" is confounded.

The reference file is ``scripts/hand3r_protocol/reference_scorer.py``, vendored verbatim from the
bundle the Hand3R authors sent on 2026-08-13.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.hand3r_protocol import reference_scorer as ref
from scripts.hand3r_protocol.hand3r_metrics import score_clip_hand3r
from scripts.world_space_metrics import apply_similarity, solve_similarity


def _traj(seed: int, t: int = 120, j: int = 21):
    """A hand that both articulates and travels, with the prediction drifting away from truth.

    Constant offsets are useless here: a rigid gauge removes them exactly, so every metric would
    read zero and the test would pass on a broken implementation. The drift term grows with time,
    which is what makes W-MPJPE and WA-MPJPE disagree and therefore what the test is measuring.
    """
    g = torch.Generator().manual_seed(seed)
    shape = torch.randn(j, 3, generator=g) * 0.03
    travel = torch.linspace(0, 0.8, t)[:, None] * torch.tensor([1.0, 0.2, 0.5])
    gt = shape[None] + travel[:, None, :] + torch.randn(t, j, 3, generator=g) * 0.002
    drift = torch.linspace(0, 0.06, t)[:, None] * torch.tensor([0.3, 1.0, -0.4])
    pred = gt + drift[:, None, :] + torch.randn(t, j, 3, generator=g) * 0.004
    return pred.double(), gt.double()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alignment_algebra_is_identical(seed):
    """Our ``solve_similarity(src, dst)`` and their ``_align(target, source)`` are the same map."""
    pred, gt = _traj(seed, t=40)
    p, g = pred.reshape(-1, 3), gt.reshape(-1, 3)

    for fixed_scale in (False, True):
        s_ref, r_ref, t_ref = ref._align(g.numpy(), p.numpy(), fixed_scale=fixed_scale)
        if fixed_scale:
            # Ours has no fixed-scale flag; the caller pins s=1 and re-solves t, which is what
            # both hand3r_metrics and world_space_metrics do.
            _, r, _ = solve_similarity(p, g)
            s = torch.tensor(1.0, dtype=torch.float64)
            tt = g.mean(0) - (r.double() @ p.mean(0))
        else:
            s, r, tt = solve_similarity(p, g)
        assert np.isclose(float(s), s_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(r.double().numpy(), r_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(tt.double().numpy(), t_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("seed,chunk", [(0, 100), (1, 30), (2, 30), (3, 100)])
def test_torch_port_matches_reference_scorer(seed, chunk):
    """End to end: our torch protocol == their NumPy protocol, on the metrics we implement."""
    pred, gt = _traj(seed)
    ours = score_clip_hand3r(pred, gt, chunk_len=chunk)
    theirs = ref.score_clip(gt_cam=gt.numpy(), pred_cam=pred.numpy(),
                            gt_world=gt.numpy(), pred_world=pred.numpy(),
                            chunk_length=chunk, unit="m")
    assert theirs is not None and ours
    for our_key, their_key in (("C_MPJPE_abs", "C-MPJPE"), ("W_MPJPE", "W-MPJPE"),
                               ("WA_MPJPE", "WA-MPJPE"), ("MRE", "MRE")):
        assert np.isclose(ours[our_key], theirs[their_key], rtol=1e-6, atol=1e-6), (
            f"{our_key}: ours {ours[our_key]:.6f} vs reference {theirs[their_key]:.6f}")


def test_their_w_gauge_is_not_our_w_gauge():
    """The gauges genuinely differ, so the parity above is not passing by accident.

    Our W solves the rigid transform on the first 30 frames; theirs on the first 2. If a refactor
    ever made these agree, the protocol delta we report in the paper would silently become zero,
    so the difference is asserted rather than assumed.
    """
    pred, gt = _traj(7)
    theirs = score_clip_hand3r(pred, gt, chunk_len=100)["W_MPJPE"]

    from scripts.world_space_metrics import w_mpjpe_first_window_aligned
    ours = w_mpjpe_first_window_aligned(pred[:100].float(), gt[:100].float(), None, wa_short=30)

    assert abs(ours - theirs) > 1.0, (
        f"the two W gauges agreed to within {abs(ours - theirs):.3f} mm, which contradicts "
        f"Hand3R's PROTOCOL.md; check that the first-two-frames fit was not refactored away")


def test_short_tail_is_dropped():
    """Their rule: a chunk shorter than 10 frames is skipped. Ours would have scored it."""
    pred, gt = _traj(11, t=105)
    scored = score_clip_hand3r(pred, gt, chunk_len=100)
    # 105 frames at chunk 100 leaves a 5-frame tail, which contributes to neither chunk metric.
    assert scored["n_frames"] == 105
    solo = score_clip_hand3r(pred[:100], gt[:100], chunk_len=100)
    assert np.isclose(scored["W_MPJPE"], solo["W_MPJPE"], rtol=1e-9)
