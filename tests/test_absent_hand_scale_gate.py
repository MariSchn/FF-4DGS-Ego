"""The scene-scale solve must be able to exclude a hand slot the detector never filled (task #70).

WHAT THIS CAUGHT (2026-08-07), found while fixing the registration figure rather than by reading
the code. Un-rotating the depth panel so both hands' joints share one coordinate convention made a
second joint cluster visible that sits on a door, a floor, and a table, but on no hand.

MEASURED on five HOI4D dumps (regsteps_v2_seq0..4, scene ZY20210800001_H1_C12):

  * hand slot 0 supplied EXACTLY 96 accepted correspondences per 16-frame clip (6 per frame) at a
    median ratio of -0.019, with 50% of them behind the camera;
  * hand slot 1, the visible hand, sat at median 0.607-0.894 with 0-4.2% behind camera;
  * slot 0 accounted for 240 of 255 = 94.1% of every behind-camera correspondence.

The box store settles it: `hoi4d_detboxes_v3/ZY20210800001_H1_C12_N28_S200_s01_T2.pt` has
`valid[:, 0].sum() == 0` over all 300 frames. The detector never found a left hand. So this is not
a badly predicted real hand; the model emits a default MANO into an empty slot and the solve eats
it, because `ratio_validity_mask` was purely geometric and `predict_clip` was never given
`hand_valid` even though `eval_sequence` already reads it for conditioning.

The gate is asserted here at the mask level so the rule is testable without a GPU forward pass,
matching how `require_positive_z` is covered.
"""
from __future__ import annotations

import torch

from scripts.eval_world_space import ratio_validity_mask


def _geometrically_valid(S=4, H=2, J=16):
    """Correspondences that pass every geometric term, so only the hand gate can reject them."""
    z = torch.full((S, H, J), 0.5)
    sampled = torch.full((S, H, J), 1.0)
    in_frame = torch.ones((S, H, J), dtype=torch.bool)
    return z, sampled, in_frame


def test_absent_hand_is_dropped_when_gated():
    z, sampled, in_frame = _geometrically_valid()
    hand_valid = torch.ones((4, 2), dtype=torch.bool)
    hand_valid[:, 0] = False                       # the HOI4D case: slot 0 never detected

    ungated = ratio_validity_mask(z, sampled, in_frame)
    gated = ratio_validity_mask(z, sampled, in_frame, hand_present=hand_valid)

    assert int(ungated.sum()) == 4 * 2 * 16, "geometry alone must accept everything here"
    assert int(gated.sum()) == 4 * 1 * 16, "the absent slot's joints must be gone"
    assert not bool(gated[:, 0].any()), "no joint from the absent hand may survive"
    assert bool(gated[:, 1].all()), "the present hand must be untouched"


def test_gate_is_a_no_op_when_every_hand_is_present():
    """NEGATIVE CONTROL. Without this, a gate that drops everything would pass the test above."""
    z, sampled, in_frame = _geometrically_valid()
    all_present = torch.ones((4, 2), dtype=torch.bool)

    assert torch.equal(ratio_validity_mask(z, sampled, in_frame),
                       ratio_validity_mask(z, sampled, in_frame, hand_present=all_present))


def test_default_preserves_the_shipped_behaviour():
    """Every number reported before 2026-08-07 was computed with the absent hand INCLUDED.

    The gate must therefore default to off, so an eval rerun without the flag reproduces the old
    result exactly rather than silently changing the paper's W/WA.
    """
    z, sampled, in_frame = _geometrically_valid()
    hand_valid = torch.zeros((4, 2), dtype=torch.bool)          # would drop everything if applied

    assert int(ratio_validity_mask(z, sampled, in_frame).sum()) == 4 * 2 * 16


def test_gate_composes_with_the_behind_camera_guard():
    """#63 and #70 overlap heavily but are not the same rule, so they must compose, not collide."""
    z, sampled, in_frame = _geometrically_valid()
    z[:, 0] = -0.2                                  # absent hand also happens to be behind camera
    z[0, 1, 0] = -0.2                               # ...and one genuine joint is too
    hand_valid = torch.ones((4, 2), dtype=torch.bool)
    hand_valid[:, 0] = False

    both = ratio_validity_mask(z, sampled, in_frame, require_positive_z=True,
                               hand_present=hand_valid)
    assert not bool(both[:, 0].any()), "absent hand gone"
    assert not bool(both[0, 1, 0]), "the behind-camera joint of the REAL hand is gone too"
    assert int(both.sum()) == 4 * 16 - 1, "everything else on the real hand survives"


def test_broadcasts_over_a_leading_batch_axis():
    """`eval_sequence` holds hand_valid as [1,S,H]; the mask is [1,S,H,J]. Shapes must line up."""
    z = torch.full((1, 4, 2, 16), 0.5)
    sampled = torch.full((1, 4, 2, 16), 1.0)
    in_frame = torch.ones((1, 4, 2, 16), dtype=torch.bool)
    hand_valid = torch.ones((1, 4, 2), dtype=torch.bool)
    hand_valid[..., 0] = False

    gated = ratio_validity_mask(z, sampled, in_frame, hand_present=hand_valid)
    assert gated.shape == (1, 4, 2, 16)
    assert int(gated.sum()) == 4 * 16
