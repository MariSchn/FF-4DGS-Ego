"""Tests for the P1 metric-coupling additions:

* Keypoint3DLoss(align_root=...) — absolute vs root-relative supervision.
* hand_depth_anchor_loss(direction=...) — gradient routing per direction.
"""
import torch

from scripts.hamer_losses import Keypoint3DLoss
from scripts.hand_depth_anchor_loss import hand_depth_anchor_loss


def _kp_inputs(B=2, S=2, N=5):
    pred = torch.randn(B, S, N, 3)
    gt_xyz = torch.randn(B, S, N, 3)
    conf = torch.ones(B, S, N, 1)
    gt = torch.cat([gt_xyz, conf], dim=-1)
    return pred, gt


def test_align_root_true_is_translation_invariant():
    crit = Keypoint3DLoss(loss_type="l2")
    pred, gt = _kp_inputs()
    base = crit(pred, gt, align_root=True)
    # Shift BOTH pred and gt by the same global translation.
    T = torch.tensor([0.3, -0.2, 0.5])
    pred_s = pred + T
    gt_s = gt.clone()
    gt_s[..., :3] = gt_s[..., :3] + T
    shifted = crit(pred_s, gt_s, align_root=True)
    assert torch.allclose(base, shifted, atol=1e-5), (base, shifted)


def test_align_root_false_penalizes_global_translation():
    crit = Keypoint3DLoss(loss_type="l2")
    pred, gt = _kp_inputs()
    # Make pred match gt exactly except for a global offset.
    pred = gt[..., :3].clone()
    loss_aligned = crit(pred, gt, align_root=True)
    pred_off = pred + torch.tensor([0.0, 0.0, 0.17])   # 17 cm depth error
    loss_abs = crit(pred_off, gt, align_root=False)
    # Root-relative loss is ~0 (only a global shift); absolute loss is large.
    assert loss_aligned.item() < 1e-6
    assert loss_abs.item() > 0.01


def _anchor_inputs():
    B, S, H, J = 1, 1, 2, 3
    pred = torch.zeros(B, S, H, J, 3)
    pred[..., 2] = 0.5                       # z = 0.5 m, near optical axis
    gs_depth = torch.full((B, S, 1, 16, 16), 0.6)   # scene depth 0.6 m
    has_hand = torch.ones(B, S, H)
    cam_intr = torch.tensor([[500.0, 704.0, 704.0]])
    return pred, gs_depth, has_hand, cam_intr


def _grad_norm(t):
    return 0.0 if t.grad is None else float(t.grad.abs().sum())


def test_direction_scene_follows_hand_routes_grad_to_scene():
    pred, gs_depth, has_hand, cam = _anchor_inputs()
    pred = pred.clone().requires_grad_(True)
    gs_depth = gs_depth.clone().requires_grad_(True)
    loss, info = hand_depth_anchor_loss(pred, gs_depth, has_hand, cam,
                                        direction="scene_follows_hand")
    loss.backward()
    assert info["n_valid"] > 0
    assert _grad_norm(gs_depth) > 0          # scene moves
    assert _grad_norm(pred) == 0             # hand fixed


def test_direction_hand_follows_scene_routes_grad_to_hand():
    pred, gs_depth, has_hand, cam = _anchor_inputs()
    pred = pred.clone().requires_grad_(True)
    gs_depth = gs_depth.clone().requires_grad_(True)
    loss, info = hand_depth_anchor_loss(pred, gs_depth, has_hand, cam,
                                        direction="hand_follows_scene")
    loss.backward()
    assert _grad_norm(pred) > 0              # hand moves
    assert _grad_norm(gs_depth) == 0         # scene fixed


def test_direction_bidirectional_routes_grad_to_both():
    pred, gs_depth, has_hand, cam = _anchor_inputs()
    pred = pred.clone().requires_grad_(True)
    gs_depth = gs_depth.clone().requires_grad_(True)
    loss, info = hand_depth_anchor_loss(pred, gs_depth, has_hand, cam,
                                        direction="bidirectional")
    loss.backward()
    assert _grad_norm(pred) > 0
    assert _grad_norm(gs_depth) > 0


def test_direction_loss_and_residual_are_value_identical():
    vals, resids = [], []
    for d in ("scene_follows_hand", "hand_follows_scene", "bidirectional"):
        pred, gs_depth, has_hand, cam = _anchor_inputs()
        loss, info = hand_depth_anchor_loss(pred, gs_depth, has_hand, cam, direction=d)
        vals.append(float(loss))
        resids.append(info["hand_depth_residual_m"])
    assert max(vals) - min(vals) < 1e-6, vals
    assert max(resids) - min(resids) < 1e-6, resids
    assert abs(resids[0] - 0.1) < 1e-4       # |0.6 - 0.5|


def test_unknown_direction_raises():
    pred, gs_depth, has_hand, cam = _anchor_inputs()
    try:
        hand_depth_anchor_loss(pred, gs_depth, has_hand, cam, direction="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown direction")
