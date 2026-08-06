"""Tests for the L1 HDGLA anchor loss (scripts/hand_depth_anchor_loss.py).

CPU-only, deterministic. Verifies the masking logic, the zero/offset residuals,
the all-invalid short-circuit, out-of-frame masking, the confidence gate, and
that gradient flows to ``gs_depth`` but NOT to ``pred_joints`` (the hand is
detached entirely — scene follows hand).
"""
import torch

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    IMAGE_WIDTH,
)
from scripts.hand_depth_anchor_loss import hand_depth_anchor_loss


# Physical principal point at the centre of a square W x W frame. The fixture previously used
# cx = cy = 0, which is not a realisable camera; the helper now derives the frame width from the
# intrinsics as 2*cx, so an origin principal point cannot express a width at all. See the note in
# test_hand_depth_sampling.py.
def _cam_intr() -> torch.Tensor:
    return torch.tensor(
        [[IMAGE_WIDTH, IMAGE_WIDTH / 2.0, IMAGE_WIDTH / 2.0]], dtype=torch.float64
    )


def _xy_for_norm(u_norm: float, v_norm: float, z):
    """Camera-frame (x, y) placing a joint at normalised (u_norm, v_norm) at depth z.

    Inverts col = f*x/z + cx -> v = col, and row = f*y/z + cy -> u = (W-1) - row.
    ``z`` may be a float or a tensor; the return follows it.
    """
    W = IMAGE_WIDTH
    f, cx, cy = W, W / 2.0, W / 2.0
    col = v_norm * W
    row = (W - 1.0) - u_norm * W
    return (col - cx) * z / f, (row - cy) * z / f


def _joints_at_center(z_values: torch.Tensor) -> torch.Tensor:
    """Build [B,S,H,J,3] joints all projecting to the frame center (u=v=0.5)
    with the given per-joint metric depth ``z_values`` ([B,S,H,J])."""
    z = z_values
    x, y = _xy_for_norm(0.5, 0.5, z)
    return torch.stack([x, y, z], dim=-1)


def test_constant_depth_equal_to_z_gives_zero_loss():
    z_const = 0.5
    B, S, H, J = 1, 1, 1, 4
    z_values = torch.full((B, S, H, J), z_const, dtype=torch.float64)
    joints = _joints_at_center(z_values)
    has_hand = torch.ones((B, S, H), dtype=torch.float64)
    gs_depth = torch.full((B, S, 1, 16, 16), z_const, dtype=torch.float64)

    loss, info = hand_depth_anchor_loss(joints, gs_depth, has_hand, _cam_intr())

    assert info["n_valid"] == B * S * H * J
    assert abs(info["hand_depth_residual_m"]) < 1e-6
    assert float(loss) < 1e-9


def test_known_offset_recovers_residual():
    z_const = 0.5
    offset = 0.1  # metres; sampled depth = z + offset
    B, S, H, J = 1, 1, 1, 3
    z_values = torch.full((B, S, H, J), z_const, dtype=torch.float64)
    joints = _joints_at_center(z_values)
    has_hand = torch.ones((B, S, H), dtype=torch.float64)
    gs_depth = torch.full((B, S, 1, 16, 16), z_const + offset, dtype=torch.float64)

    loss, info = hand_depth_anchor_loss(joints, gs_depth, has_hand, _cam_intr())

    assert info["n_valid"] == B * S * H * J
    assert abs(info["hand_depth_residual_m"] - offset) < 1e-6
    # smooth_l1 with beta=0.02 and |diff|=0.1 (> beta) -> |diff| - 0.5*beta.
    expected_loss = offset - 0.5 * 0.02
    assert abs(float(loss) - expected_loss) < 1e-6


def test_all_invalid_hands_gives_zero_loss_and_n_valid_zero():
    z_const = 0.5
    B, S, H, J = 1, 1, 2, 4
    z_values = torch.full((B, S, H, J), z_const, dtype=torch.float64)
    joints = _joints_at_center(z_values)
    has_hand = torch.zeros((B, S, H), dtype=torch.float64)
    gs_depth = torch.full((B, S, 1, 16, 16), z_const + 0.3, dtype=torch.float64)

    loss, info = hand_depth_anchor_loss(joints, gs_depth, has_hand, _cam_intr())

    assert info["n_valid"] == 0
    assert info["hand_depth_residual_m"] == 0.0
    assert float(loss) == 0.0
    assert not torch.isnan(loss)


def test_out_of_frame_joints_are_masked():
    """Two joints: one in-frame (z matches gs_depth -> 0 residual), one out of
    frame (huge offset). The out-of-frame joint must be excluded, so residual=0."""
    z_const = 0.5
    B, S, H = 1, 1, 1
    # in-frame joint at center, sampled == z.
    z = z_const
    in_x, in_y = _xy_for_norm(0.5, 0.5, z)
    in_joint = torch.tensor([in_x, in_y, z], dtype=torch.float64)
    # out-of-frame joint: u_norm = 1.5 -> outside.
    out_x, out_y = _xy_for_norm(1.5, 0.5, z)
    out_joint = torch.tensor([out_x, out_y, z], dtype=torch.float64)
    joints = torch.stack([in_joint, out_joint]).view(B, S, H, 2, 3)

    has_hand = torch.ones((B, S, H), dtype=torch.float64)
    gs_depth = torch.full((B, S, 1, 16, 16), z_const, dtype=torch.float64)

    loss, info = hand_depth_anchor_loss(joints, gs_depth, has_hand, _cam_intr())

    assert info["n_valid"] == 1  # only the in-frame joint counts
    assert abs(info["hand_depth_residual_m"]) < 1e-6
    assert float(loss) < 1e-9


def test_confidence_gate_excludes_low_conf_samples():
    z_const = 0.5
    B, S, H, J = 1, 1, 1, 2
    z_values = torch.full((B, S, H, J), z_const, dtype=torch.float64)
    joints = _joints_at_center(z_values)
    has_hand = torch.ones((B, S, H), dtype=torch.float64)
    gs_depth = torch.full((B, S, 1, 16, 16), z_const + 0.2, dtype=torch.float64)
    # All confidence below threshold -> nothing valid.
    gs_conf = torch.full((B, S, 1, 16, 16), 0.1, dtype=torch.float64)

    loss, info = hand_depth_anchor_loss(
        joints, gs_depth, has_hand, _cam_intr(),
        gs_depth_conf=gs_conf, conf_thresh=0.5,
    )
    assert info["n_valid"] == 0
    assert float(loss) == 0.0


def test_gradient_flows_to_gs_depth_not_pred_joints():
    """Scene-follows-hand: gradient reaches ``gs_depth``; ``pred_joints`` is
    detached entirely, so the anchor never perturbs the metric hand."""
    z_const = 0.5
    B, S, H, J = 1, 1, 1, 4
    z_values = torch.full((B, S, H, J), z_const, dtype=torch.float64)
    joints = _joints_at_center(z_values).clone().requires_grad_()
    has_hand = torch.ones((B, S, H), dtype=torch.float64)
    gs_depth = torch.full((B, S, 1, 16, 16), z_const + 0.05, dtype=torch.float64).requires_grad_()

    loss, info = hand_depth_anchor_loss(joints, gs_depth, has_hand, _cam_intr())
    assert info["n_valid"] > 0
    loss.backward()

    # The scene depth is what gets pulled.
    assert gs_depth.grad is not None
    assert torch.any(gs_depth.grad != 0)
    # The hand is fully detached: it receives no gradient at all.
    assert joints.grad is None


def test_pred_joints_fully_detached_under_varying_depth():
    """Even with a spatially-varying ``gs_depth`` (so the sampling-location path
    would otherwise be active), ``pred_joints`` receives no gradient — it is
    detached entirely — while ``gs_depth`` still does."""
    z_const = 0.5
    B, S, H, J = 1, 1, 1, 4
    z_values = torch.full((B, S, H, J), z_const, dtype=torch.float64)
    joints = _joints_at_center(z_values).clone().requires_grad_()
    has_hand = torch.ones((B, S, H), dtype=torch.float64)
    # Horizontal ramp (1.0..2.0) -> spatially varying AND offset from z=0.5 so the
    # smooth-L1 residual is well above beta and the scene gradient is non-trivial.
    ramp = 1.0 + torch.arange(16, dtype=torch.float64) / 15.0
    gs_depth = ramp.view(1, 1, 1, 1, 16).expand(B, S, 1, 16, 16).contiguous().requires_grad_()

    loss, info = hand_depth_anchor_loss(joints, gs_depth, has_hand, _cam_intr())
    assert info["n_valid"] > 0
    loss.backward()

    assert gs_depth.grad is not None
    assert torch.any(gs_depth.grad != 0)
    assert joints.grad is None
