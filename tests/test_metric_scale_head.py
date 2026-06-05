"""CPU tests for the L2 MetricScaleHead (closed-form global metric scale).

Deterministic synthetic tensors only (no randomness). The geometry is grounded
in the shared helper ``hand_depth_sampling``: we place hand joints at known
camera-frame points, build ``gs_depth`` so the sampled scene depth at each joint
equals ``z / k``, and check that the solved scale recovers ``k``.
"""
from __future__ import annotations

import torch

from diffsynth.auxiliary_models.worldmirror.models.heads.metric_scale_head import (
    apply_metric_scale,
    solve_metric_scale,
)
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)

# Intrinsics in the locked 1408 square frame: [focal, cx, cy].
CAM_INTR = torch.tensor([[600.0, 704.0, 704.0]])
HD = WD = 64


def _single_point_joints(z: float, n_joints: int = 16) -> torch.Tensor:
    """[1,1,1,J,3] joints all at one in-frame camera point with depth ``z``."""
    pt = torch.tensor([0.05, -0.10, z])
    return pt.view(1, 1, 1, 1, 3).repeat(1, 1, 1, n_joints, 1)


def _spread_point_joints(z: float) -> torch.Tensor:
    """[1,1,1,9,3] joints on a 3x3 X/Y grid, all sharing depth ``z``.

    Distinct image cells let us corrupt a minority of samples independently.
    """
    coords = torch.linspace(-0.15, 0.15, 3)
    pts = [[float(x), float(y), z] for y in coords for x in coords]
    return torch.tensor(pts).view(1, 1, 1, 9, 3)


def _const_depth(value: float) -> torch.Tensor:
    """[1,1,1,Hd,Wd] constant positive scene depth map."""
    return torch.full((1, 1, 1, HD, WD), float(value))


def test_recovers_known_scale():
    # Arrange: gs_depth = z / k everywhere -> sampled == z / k at every joint.
    k = 3.0
    z = 0.40
    pred_joints = _single_point_joints(z)
    gs_depth = _const_depth(z / k)
    has_hand = torch.ones(1, 1, 1)

    # Act
    s = solve_metric_scale(pred_joints, gs_depth, has_hand, CAM_INTR)

    # Assert
    assert s.shape == ()
    assert torch.isclose(s, torch.tensor(k), atol=1e-4)


def test_clamp_upper_bound_respected():
    # Arrange: k = 50 sits above the default clamp ceiling of 10.
    # z is chosen so z/k (= 0.02) stays above depth_min (0.01) and isn't gated.
    z, k = 1.0, 50.0
    pred_joints = _single_point_joints(z)
    gs_depth = _const_depth(z / k)
    has_hand = torch.ones(1, 1, 1)

    # Act
    s = solve_metric_scale(pred_joints, gs_depth, has_hand, CAM_INTR)

    # Assert
    assert torch.isclose(s, torch.tensor(10.0), atol=1e-6)


def test_clamp_lower_bound_respected():
    # Arrange: k = 0.02 sits below the default clamp floor of 0.1.
    z, k = 0.40, 0.02
    pred_joints = _single_point_joints(z)
    gs_depth = _const_depth(z / k)
    has_hand = torch.ones(1, 1, 1)

    # Act
    s = solve_metric_scale(pred_joints, gs_depth, has_hand, CAM_INTR)

    # Assert
    assert torch.isclose(s, torch.tensor(0.1), atol=1e-6)


def test_median_rejects_outliers():
    # Arrange: 9 joints at z/k; corrupt 2 cells to wildly wrong depths.
    z, k = 0.5, 2.0
    pred_joints = _spread_point_joints(z)
    gs_depth = _const_depth(z / k)
    has_hand = torch.ones(1, 1, 1)

    grid_xy, _ = project_joints_to_norm_pixels(pred_joints, CAM_INTR)
    cells = (grid_xy[0, 0, 0] * HD).round().to(torch.int64)  # (x, y) per joint
    for j in (0, 8):  # minority of 9
        cx, cy = int(cells[j, 0]), int(cells[j, 1])
        gs_depth[0, 0, 0, max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3] = 10.0

    # Act
    s = solve_metric_scale(pred_joints, gs_depth, has_hand, CAM_INTR)

    # Assert: median is unmoved by the 2/9 outliers.
    assert torch.isclose(s, torch.tensor(k), atol=1e-4)


def test_apply_drives_residual_to_one():
    # Arrange
    z, k = 0.40, 4.0
    pred_joints = _single_point_joints(z)
    gs_depth = _const_depth(z / k)
    has_hand = torch.ones(1, 1, 1)
    preds = {"gs_depth": gs_depth}

    # Act: solve, apply, re-solve.
    s = solve_metric_scale(pred_joints, gs_depth, has_hand, CAM_INTR)
    scaled = apply_metric_scale(preds, s)
    s2 = solve_metric_scale(pred_joints, scaled["gs_depth"], has_hand, CAM_INTR)

    # Assert: after scaling, the residual scale collapses to ~1.
    assert torch.isclose(s, torch.tensor(k), atol=1e-4)
    assert torch.isclose(s2, torch.tensor(1.0), atol=1e-4)


def test_apply_does_not_mutate_input():
    # Arrange
    s = torch.tensor(2.0)
    gs_depth = _const_depth(0.5)
    camera_poses = torch.eye(4).view(1, 1, 4, 4).clone()
    camera_poses[0, 0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    preds = {"gs_depth": gs_depth, "camera_poses": camera_poses, "other": "keep"}

    gs_before = gs_depth.clone()
    poses_before = camera_poses.clone()

    # Act
    out = apply_metric_scale(preds, s)

    # Assert: input tensors untouched.
    assert torch.equal(gs_depth, gs_before)
    assert torch.equal(camera_poses, poses_before)
    assert out is not preds

    # gs_depth scaled by s.
    assert torch.allclose(out["gs_depth"], gs_before * s)

    # Camera translation scaled, rotation block untouched.
    assert torch.allclose(out["camera_poses"][0, 0, :3, 3], poses_before[0, 0, :3, 3] * s)
    assert torch.allclose(out["camera_poses"][..., :3, :3], poses_before[..., :3, :3])

    # Untouched keys pass through.
    assert out["other"] == "keep"


def test_apply_handles_missing_keys():
    # Arrange: neither gs_depth nor camera_poses present.
    s = torch.tensor(3.0)
    preds = {"foo": torch.ones(2)}

    # Act
    out = apply_metric_scale(preds, s)

    # Assert: passthrough, new dict, no error.
    assert out is not preds
    assert torch.equal(out["foo"], preds["foo"])


def test_empty_has_hand_returns_identity():
    # Arrange: no valid hands at all.
    pred_joints = _single_point_joints(0.40)
    gs_depth = _const_depth(0.10)
    has_hand = torch.zeros(1, 1, 1)

    # Act
    s = solve_metric_scale(pred_joints, gs_depth, has_hand, CAM_INTR)

    # Assert: identity scale, no NaN.
    assert torch.equal(s, torch.tensor(1.0))
    assert not torch.isnan(s)


def test_depth_min_gate_yields_identity():
    # Arrange: all sampled depths sit at/below depth_min -> no valid samples.
    pred_joints = _single_point_joints(0.40)
    gs_depth = _const_depth(0.005)  # below default depth_min = 0.01
    has_hand = torch.ones(1, 1, 1)

    # Act
    s = solve_metric_scale(pred_joints, gs_depth, has_hand, CAM_INTR)

    # Assert
    assert torch.equal(s, torch.tensor(1.0))


def test_conf_gate_excludes_low_confidence():
    # Arrange: valid geometry but confidence below threshold everywhere.
    z, k = 0.40, 3.0
    pred_joints = _single_point_joints(z)
    gs_depth = _const_depth(z / k)
    gs_conf = _const_depth(0.0)  # all conf == 0
    has_hand = torch.ones(1, 1, 1)

    # Act
    s = solve_metric_scale(
        pred_joints, gs_depth, has_hand, CAM_INTR,
        gs_depth_conf=gs_conf, conf_thresh=0.5,
    )

    # Assert: gated out -> identity.
    assert torch.equal(s, torch.tensor(1.0))

    # And with confidence above threshold, the scale is recovered.
    gs_conf_high = _const_depth(1.0)
    s_ok = solve_metric_scale(
        pred_joints, gs_depth, has_hand, CAM_INTR,
        gs_depth_conf=gs_conf_high, conf_thresh=0.5,
    )
    assert torch.isclose(s_ok, torch.tensor(k), atol=1e-4)
