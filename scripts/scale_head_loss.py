"""Supervision for the feedforward ScaleHead (the "scale head" route).

Trains the predicted global scale ``s`` so that ``s * gs_depth`` matches the
metric MANO hand depth at the projected hand joints. With the Gaussian backbone
frozen, both the sampled scene depth and the hand depth target are detached, so
gradient flows ONLY into ``pred_scale``; the smooth-L1 minimiser is exactly the
median ratio that ``solve_metric_scale`` computes in closed form — i.e. the head
learns to reproduce the closed-form metric coupling in a single forward pass.

Reuses the shared, correctness-critical projection + sampling helpers (never
reimplements them).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)


def scale_head_loss(
    pred_scale: torch.Tensor,
    pred_joints: torch.Tensor,
    gs_depth: torch.Tensor,
    has_hand: torch.Tensor,
    cam_intr: torch.Tensor,
    *,
    margin: float = 0.05,
    depth_min: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """Smooth-L1 between ``s * scene_depth`` and metric hand depth at the joints.

    Args:
        pred_scale:  [B] positive predicted global scale (requires grad).
        pred_joints: [B, S, H, J, 3] camera-frame metric joints (m), detached here.
        gs_depth:    [B, S, 1, Hd, Wd] (or [B, S, Hd, Wd]) scene depth, detached here.
        has_hand:    [B, S, H] in {0, 1}.
        cam_intr:    [B, 3] = [focal, cx, cy].
        margin:      smooth-L1 (Huber) beta in metres.
        depth_min:   reject sampled scene depth <= this.

    Returns:
        loss_scalar: scalar tensor (0 when no valid joints; never NaN).
        info: {"n_valid": int, "scale_residual_m": float, "scale_mean": float}.
    """
    grid_xy, z = project_joints_to_norm_pixels(pred_joints.detach(), cam_intr)
    sampled, in_frame = sample_depth_at_joints(gs_depth.detach(), grid_xy)

    has_hand_j = has_hand.unsqueeze(-1).to(torch.bool).expand_as(sampled)
    valid = (has_hand_j & in_frame & (sampled > depth_min)
             & torch.isfinite(z) & torch.isfinite(sampled))

    info_scale_mean = float(pred_scale.mean().item())
    if not bool(valid.any()):
        loss = torch.zeros((), dtype=gs_depth.dtype, device=gs_depth.device)
        return loss, {"n_valid": 0, "scale_residual_m": 0.0, "scale_mean": info_scale_mean}

    # Broadcast the per-clip scale [B] over [B, S, H, J]; only `s` carries gradient.
    s = pred_scale.view(-1, *([1] * (sampled.dim() - 1)))
    scaled = s * sampled.detach()
    tgt = z.detach()

    per = F.smooth_l1_loss(scaled, tgt, beta=margin, reduction="none")
    valid_f = valid.to(per.dtype)
    loss = (per * valid_f).sum() / valid_f.sum()

    with torch.no_grad():
        residual = ((scaled - tgt).abs() * valid_f).sum() / valid_f.sum()

    return loss, {
        "n_valid": int(valid.sum().item()),
        "scale_residual_m": float(residual.item()),
        "scale_mean": info_scale_mean,
    }
