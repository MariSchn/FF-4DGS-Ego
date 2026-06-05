"""L1 Hand-Depth Geometric Loss Anchor (HDGLA).

Pulls predicted Gaussian scene depth (``gs_depth``) toward the metric MANO hand
depth at the projected hand joints. This is the constraint that forces the
up-to-scale monocular scene to agree in *metric* scale with the trusted metric
hand (see docs/superpowers/specs/2026-06-05-hand-scene-metric-coupling-design.md,
Unit 1).

Gradient direction is **scene follows hand**: ``pred_joints`` is detached
entirely (both the sampling location and the depth target), so only ``gs_depth``
receives gradient from this term — the metric hand is never perturbed by it.

Projection and sampling are delegated to the shared, correctness-critical helper
``hand_depth_sampling`` (do NOT reimplement them here).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)


def hand_depth_anchor_loss(
    pred_joints: torch.Tensor,
    gs_depth: torch.Tensor,
    has_hand: torch.Tensor,
    cam_intr: torch.Tensor,
    *,
    margin: float = 0.02,
    depth_min: float = 0.01,
    gs_depth_conf: torch.Tensor | None = None,
    conf_thresh: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Smooth-L1 anchor between sampled scene depth and metric hand depth.

    Args:
        pred_joints: [B, S, H, J, 3] camera-frame metric joints (metres).
        gs_depth:    [B, S, 1, Hd, Wd] (or [B, S, Hd, Wd]) positive scene depth.
        has_hand:    [B, S, H] in {0, 1}; whether each hand is valid.
        cam_intr:    [B, 3] = [focal, cx, cy] in the square-frame pixel grid.
        margin:      smooth-L1 (Huber) beta in metres.
        depth_min:   reject sampled scene depth <= this value (invalid geometry).
        gs_depth_conf: optional [B, S, 1, Hd, Wd] confidence; gates samples.
        conf_thresh: keep a sample only if its sampled confidence > this value.

    Returns:
        loss_scalar: scalar tensor (0 when no valid joints; never NaN).
        info: {"hand_depth_residual_m": float, "n_valid": int}.
    """
    # Scene follows hand: the hand is the trusted metric anchor and must stay
    # completely untouched by this term. Detach pred_joints so BOTH the sampling
    # location (grid) and the depth target are fixed — gradient flows only to
    # gs_depth via `sampled`.
    grid_xy, z = project_joints_to_norm_pixels(pred_joints.detach(), cam_intr)
    sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy)

    # Broadcast has_hand [B,S,H] over the J joint axis -> [B,S,H,J].
    has_hand_j = has_hand.unsqueeze(-1).to(torch.bool).expand_as(sampled)
    valid = has_hand_j & in_frame & (sampled > depth_min)

    if gs_depth_conf is not None:
        conf, _ = sample_depth_at_joints(gs_depth_conf, grid_xy)
        valid = valid & (conf > conf_thresh)

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        loss = torch.zeros((), dtype=sampled.dtype, device=sampled.device)
        return loss, {"hand_depth_residual_m": 0.0, "n_valid": 0}

    # `z` and `grid_xy` derive from the detached pred_joints above, so the target
    # is fixed and the loss differentiates only w.r.t. gs_depth.
    per_joint = F.smooth_l1_loss(sampled, z, beta=margin, reduction="none")
    valid_f = valid.to(per_joint.dtype)
    loss = (per_joint * valid_f).sum() / valid_f.sum()

    with torch.no_grad():
        residual = ((sampled - z).abs() * valid_f).sum() / valid_f.sum()

    return loss, {
        "hand_depth_residual_m": float(residual.item()),
        "n_valid": n_valid,
    }
