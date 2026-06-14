"""Tests for hand_joints_to_rgb_points (3D GT-vs-pred hand-placement scatter).

Pins the data prep behind the W&B Object3D logging: GT joints green, predicted
red, absent/non-finite joints dropped, GT/pred kept one-to-one.
"""
import sys
import types

import numpy as np

# hand_vis_utils pulls in heavy optional deps (decord, cv2) at import; stub any
# that are absent so the pure helper is importable in a minimal env.
for _name in ("decord", "cv2"):
    try:
        __import__(_name)
    except Exception:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("decord", types.ModuleType("decord")), "VideoReader"):
    sys.modules.setdefault("decord", types.ModuleType("decord")).VideoReader = object

from scripts.hand_vis_utils import hand_joints_to_rgb_points

GT_RGB = (0, 200, 0)
PRED_RGB = (220, 30, 30)


def _two_hands(fill_left=0.5, fill_right=0.5):
    # [H=2, J=3, 3] toy joints
    gt = np.full((2, 3, 3), fill_left, dtype=np.float32)
    gt[1] = fill_right
    return gt


def test_valid_joints_produce_colored_points():
    gt = _two_hands()
    pred = gt + 0.02
    pts = hand_joints_to_rgb_points(gt, pred)
    # 6 GT + 6 pred = 12 rows, 6 cols
    assert pts.shape == (12, 6), pts.shape
    assert np.allclose(pts[:6, 3:], GT_RGB)
    assert np.allclose(pts[6:, 3:], PRED_RGB)
    # GT xyz preserved
    assert np.allclose(pts[:6, :3], gt.reshape(-1, 3))


def test_absent_hand_zero_filler_dropped():
    gt = _two_hands()
    gt[1] = 0.0  # right hand absent (all-zero filler)
    pred = np.full((2, 3, 3), 0.4, dtype=np.float32)
    pts = hand_joints_to_rgb_points(gt, pred)
    # only the 3 valid left-hand joints survive -> 3 GT + 3 pred
    assert pts.shape == (6, 6), pts.shape
    assert np.allclose(pts[:3, :3], gt[0])


def test_non_finite_dropped():
    gt = _two_hands()
    gt[0, 0] = np.nan
    pred = gt + 0.01
    pts = hand_joints_to_rgb_points(gt, pred)
    assert np.isfinite(pts).all()
    assert pts.shape == (10, 6), pts.shape  # 5 valid GT + 5 pred


def test_all_absent_returns_none():
    gt = np.zeros((2, 3, 3), dtype=np.float32)
    pred = np.zeros((2, 3, 3), dtype=np.float32)
    assert hand_joints_to_rgb_points(gt, pred) is None


def test_shape_mismatch_returns_none():
    gt = _two_hands()
    pred = np.full((2, 4, 3), 0.5, dtype=np.float32)  # J differs
    assert hand_joints_to_rgb_points(gt, pred) is None


def test_pred_kept_only_where_gt_valid():
    # pred is finite everywhere, but GT's right hand is absent -> pred right
    # hand must also be dropped (one-to-one correspondence).
    gt = _two_hands()
    gt[1] = 0.0
    pred = np.full((2, 3, 3), 0.9, dtype=np.float32)
    pts = hand_joints_to_rgb_points(gt, pred)
    assert pts.shape == (6, 6)  # 3 GT + 3 pred, not 3 GT + 6 pred
