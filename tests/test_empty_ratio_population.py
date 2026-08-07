"""A clip with no usable correspondence is a FAILED solve, not a crash.

WHAT THIS CAUGHT (2026-08-07). The task #70 hand-gate arm of job 104381 died on all 30 sequences:

    UnboundLocalError: cannot access local variable '_s_raw' where it is not associated with a value
      File "scripts/eval_world_space.py", line 297, in predict_clip
        "s": s, "s_raw": _s_raw, "s_failed": s_failed,

`_s_raw` was assigned only inside `if bool(valid.any())`, while `steps_out` read it
unconditionally. The bug pre-dated the gate and was simply unreachable: the purely geometric mask
essentially never emptied, because it accepted joints from a hand slot the detector never filled
(which is task #70 itself). Gating on `hand_valid` makes the empty case ordinary - a clip in which
the detector found no hand in ANY frame has nothing to solve on - and one raised exception aborts
the whole sequence, so every sequence was skipped and the arm produced no JSON.

The correct behaviour is not to suppress the empty case but to CLASSIFY it: an empty population
means the scale was not solved, so `s_failed` must be true and the caller's `scale_fallback`
policy applies. Returning s=1.0 silently would be the worse failure, because 1.0 is a plausible
scale and nothing downstream would notice.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from scripts.eval_world_space import ratio_validity_mask

SRC = (Path(__file__).resolve().parents[1] / "scripts" / "eval_world_space.py").read_text()


def test_s_raw_is_bound_before_the_conditional_solve():
    """Source-level, because reproducing it needs a GPU forward pass.

    The assignment must appear before `if bool(valid.any())`, or the empty path raises.
    """
    init = SRC.find('_s_raw = float("nan")')
    guard = SRC.find("if bool(valid.any()):")
    assert init != -1, "_s_raw must be initialised unconditionally in predict_clip"
    assert guard != -1, "could not find the conditional solve"
    assert init < guard, (
        "_s_raw is initialised AFTER the conditional solve, so a clip with no valid "
        "correspondence raises UnboundLocalError when steps_out reads it")


def test_empty_population_is_marked_failed():
    """An empty population must set s_failed, not leave s=1.0 looking solved."""
    m = re.search(r"if not bool\(valid\.any\(\)\):(.{0,1200}?)if bool\(valid\.any\(\)\):",
                  SRC, re.S)
    assert m, "expected an explicit empty-population branch before the solve"
    assert "s_failed = True" in m.group(1), (
        "an empty ratio population means the scale was NOT solved; s_failed must be set so the "
        "caller's scale_fallback policy applies. s=1.0 is a plausible-looking value and would "
        "propagate silently into every world metric")


def test_gate_can_legitimately_empty_a_clip():
    """The behaviour that made the crash reachable, asserted directly.

    A clip where the detector filled no hand slot must produce an empty mask. This is not a
    defect to be engineered away: it is the correct answer for that clip.
    """
    z = torch.full((4, 2, 16), 0.5)
    sampled = torch.full((4, 2, 16), 1.0)
    in_frame = torch.ones((4, 2, 16), dtype=torch.bool)
    none_detected = torch.zeros((4, 2), dtype=torch.bool)

    assert int(ratio_validity_mask(z, sampled, in_frame).sum()) == 4 * 2 * 16
    assert int(ratio_validity_mask(z, sampled, in_frame,
                                   hand_present=none_detected).sum()) == 0


def test_partial_detection_keeps_the_detected_hand():
    """NEGATIVE CONTROL: the gate must not empty a clip that HAS a detected hand."""
    z = torch.full((4, 2, 16), 0.5)
    sampled = torch.full((4, 2, 16), 1.0)
    in_frame = torch.ones((4, 2, 16), dtype=torch.bool)
    hv = torch.zeros((4, 2), dtype=torch.bool)
    hv[:, 1] = True                      # the HOI4D case: slot 1 detected, slot 0 never

    gated = ratio_validity_mask(z, sampled, in_frame, hand_present=hv)
    assert int(gated.sum()) == 4 * 16, "the detected hand's joints must survive"
    assert bool(gated[:, 1].all()) and not bool(gated[:, 0].any())
