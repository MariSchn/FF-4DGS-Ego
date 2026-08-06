"""cam_extrinsics must not depend on the 2D-joint cache.

Task #63. The GT-scale check compares the similarity that maps our predicted (up-to-scale) camera
centres onto the GT metric centres against the scale we solve from hand depth. It is how we
learned our scale is 42% too small (s_hand 0.6208 vs s_gt 1.0230). It needs cam_extrinsics.

It was running on 12 of 1884 segments, and the reason had nothing to do with the diagnostic:
cam_extrinsics was attached to a clip only when gt_joints_2d loaded, and the cache block loads
from disk only when ALL THREE of {gt_joints_2d, cam_extrinsics, cam_intrinsics} exist. Measured on
hoi4d_test157: 157/157 sequences have cam_extrinsics_cache.pt, and 1/157 has
gt_joints_2d_cache.pt. So 156 sequences had usable extrinsics sitting on disk and discarded them
because an unrelated file was missing.

These tests pin the decoupling at the source level, which is the level the bug lived at: both the
loader's cache recovery and the two attachment points are textual conditions in
scripts/train_hand_head.py, and re-coupling them is a one-line regression that no numeric test
would catch until a diagnostic quietly went sparse again.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "scripts" / "train_hand_head.py").read_text()


def _block_after(anchor: str, n_lines: int = 12) -> str:
    i = SRC.index(anchor)
    return "\n".join(SRC[i:].splitlines()[:n_lines])


def test_extrinsics_are_attached_to_the_clip_independently():
    """clip['cam_extrinsics'] must sit under its own guard, not inside the gt_joints_2d branch."""
    assert 'if seq_cam_extrinsics is not None:' in SRC, (
        "cam_extrinsics must have its own None-guard before being attached to a clip")

    # The gt_joints_2d branch must no longer carry the extrinsics assignment with it.
    blk = _block_after("if seq_gt_joints_2d is not None:\n", 4)
    assert 'clip["cam_extrinsics"]' not in blk, (
        "clip['cam_extrinsics'] is assigned inside the `if seq_gt_joints_2d is not None:` branch, "
        "so a sequence without a 2D cache loses extrinsics it has on disk:\n" + blk)


def test_extrinsics_are_emitted_independently():
    """out['cam_extrinsics'] must not be gated on 'gt_joints_2d' in clip."""
    assert 'if "cam_extrinsics" in clip:' in SRC, (
        "out['cam_extrinsics'] must be guarded on cam_extrinsics, not on gt_joints_2d")

    blk = _block_after('if "gt_joints_2d" in clip:\n', 3)
    assert 'out["cam_extrinsics"]' not in blk, (
        "out['cam_extrinsics'] is emitted inside the `if \"gt_joints_2d\" in clip:` branch:\n" + blk)


def test_extrinsics_cache_is_recovered_when_the_2d_cache_is_absent():
    """The all-or-nothing cache load must have an additive fallback for extrinsics.

    Without it, the recompute path returns None for a store with no calibration and an existing
    cam_extrinsics_cache.pt on disk is thrown away.
    """
    assert re.search(
        r"if seq_cam_extrinsics is None and os\.path\.exists\(cam_extr_cache_path\)", SRC), (
        "no additive recovery for cam_extrinsics_cache.pt: a sequence whose gt_joints_2d cache is "
        "missing will discard extrinsics that exist on disk")


def test_intrinsics_stay_independent_too():
    """Guards the pattern that was already correct, so a future tidy-up cannot re-couple it.

    cam_intrinsics was deliberately decoupled earlier (the root anchor and metric losses need it
    without any 2D GT). That precedent is why the extrinsics coupling reads as an oversight rather
    than a design choice, and it should not silently regress either.
    """
    assert 'if "cam_intrinsics" in clip:' in SRC
    blk = _block_after('if "gt_joints_2d" in clip:\n', 3)
    assert 'out["cam_intrinsics"]' not in blk
