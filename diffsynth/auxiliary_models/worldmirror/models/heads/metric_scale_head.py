"""L2 MetricScaleHead: closed-form global metric scale (inference solve).

Makes the up-to-scale monocular Gaussian scene agree in *metric* scale with the
metric MANO hand. We project the predicted metric hand joints into ``gs_depth``
(via the shared geometry helper), read the scene depth there, and solve a single
global scalar ``s`` such that ``s * gs_depth`` matches the hand's metric depth at
the joints. ``apply_metric_scale`` then multiplies ``gs_depth`` and the camera
translation by ``s`` so the rasterizer's internal scale ratio collapses to ~1.

Both L1 (training anchor) and L2 (this module) share the projection+sampling core
in ``..utils.hand_depth_sampling`` and never reimplement it.
"""
from __future__ import annotations

import torch

from ..utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)


def solve_metric_scale(
    pred_joints: torch.Tensor,
    gs_depth: torch.Tensor,
    has_hand: torch.Tensor,
    cam_intr: torch.Tensor,
    *,
    clamp: tuple[float, float] = (0.1, 10.0),
    depth_min: float = 0.01,
    gs_depth_conf: torch.Tensor | None = None,
    conf_thresh: float = 0.0,
) -> torch.Tensor:
    """Solve a single global metric scale ``s`` (closed form, L2 median).

    ``s`` is the median of ``z / sampled`` over valid joints, where ``z`` is the
    hand's metric camera-frame depth and ``sampled`` is the scene depth read from
    ``gs_depth`` at the projected joint pixel. Applying ``s`` to ``gs_depth``
    drives it toward the metric hand depth.

    Args:
        pred_joints:   [B, S, H, J, 3] camera-frame metric joints (metres).
        gs_depth:      [B, S, 1, Hd, Wd] (or [B, S, Hd, Wd]) positive scene depth.
        has_hand:      [B, S, H] in {0, 1}; per-hand validity.
        cam_intr:      [B, 3] = [focal, cx, cy].
        clamp:         (lo, hi) bounds applied to the solved scale.
        depth_min:     reject sampled scene depth <= this (metres).
        gs_depth_conf: optional [B, S, 1, Hd, Wd] (or [B, S, Hd, Wd]) confidence;
                       when given, gate samples on ``conf > conf_thresh``.
        conf_thresh:   confidence gate threshold.

    Returns:
        s: scalar tensor (0-dim). Identity ``1.0`` when no valid samples exist
           (never NaN).
    """
    grid_xy, z = project_joints_to_norm_pixels(pred_joints, cam_intr)
    sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy)

    # has_hand is [B, S, H]; broadcast over the J joint axis.
    hand_valid = has_hand.to(torch.bool).unsqueeze(-1)  # [B, S, H, 1]
    valid = hand_valid & in_frame & (sampled > depth_min)

    if gs_depth_conf is not None:
        conf, _ = sample_depth_at_joints(gs_depth_conf, grid_xy)
        valid = valid & (conf > conf_thresh)

    valid = valid & torch.isfinite(z) & torch.isfinite(sampled)

    if not bool(valid.any()):
        return torch.tensor(1.0, dtype=gs_depth.dtype, device=gs_depth.device)

    ratio = (z / sampled)[valid]
    s = torch.median(ratio).clamp(clamp[0], clamp[1])
    return s.reshape(())


def apply_metric_scale(preds: dict, s: torch.Tensor) -> dict:
    """Return a shallow copy of ``preds`` with metric scale ``s`` applied.

    Scales ``gs_depth`` and the camera-pose translation by ``s``; the input dict
    and its tensors are left unmutated (tensors are cloned before editing).

    Args:
        preds: prediction dict that may contain ``gs_depth`` [..., Hd, Wd] and
               ``camera_poses`` [..., 4, 4].
        s:     scalar tensor scale.

    Returns:
        A new dict with the same keys; ``gs_depth`` and ``camera_poses`` rescaled.
    """
    out = dict(preds)

    gs_depth = preds.get("gs_depth")
    if gs_depth is not None:
        out["gs_depth"] = gs_depth * s

    camera_poses = preds.get("camera_poses")
    if camera_poses is not None:
        scaled = camera_poses.clone()
        scaled[..., :3, 3] = scaled[..., :3, 3] * s
        out["camera_poses"] = scaled

    return out
