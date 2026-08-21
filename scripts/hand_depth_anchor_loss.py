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

_LOG_EPS = 1e-6   # metres; the log branch's floor on a depth that passed the gate but is tiny

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
    direction: str = "scene_follows_hand",
    space: str = "linear",
) -> tuple[torch.Tensor, dict]:
    """Smooth-L1 anchor between sampled scene depth and metric hand depth.

    The 2D sampling location is always taken from the *detached* hand projection
    so the grid is stable; ``direction`` only selects which depth carries the
    gradient:

    * ``"scene_follows_hand"`` (default): the hand is the trusted metric anchor;
      gradient flows only into ``gs_depth``. Use when the hand placement is more
      reliable than the monocular scene depth.
    * ``"hand_follows_scene"``: the scene depth is treated as the metric target;
      gradient flows only into the hand joints (i.e. back to the head's
      translation). Use when the (well-scaled) scene depth should correct an
      under-supervised absolute hand placement.
    * ``"bidirectional"``: both move toward each other (gradient into both).

    Args:
        pred_joints: [B, S, H, J, 3] camera-frame metric joints (metres).
        gs_depth:    [B, S, 1, Hd, Wd] (or [B, S, Hd, Wd]) positive scene depth.
        has_hand:    [B, S, H] in {0, 1}; whether each hand is valid.
        cam_intr:    [B, 3] = [focal, cx, cy] in the square-frame pixel grid.
        margin:      smooth-L1 (Huber) beta in metres.
        depth_min:   reject sampled scene depth <= this value (invalid geometry).
        gs_depth_conf: optional [B, S, 1, Hd, Wd] confidence; gates samples.
        conf_thresh: keep a sample only if its sampled confidence > this value.
        direction:   one of {scene_follows_hand, hand_follows_scene, bidirectional}.
        space:       "linear" compares metres, "log" compares log-depths. In linear space the
                     smooth-L1 saturates above ``margin``, so a depth that has escaped by orders of
                     magnitude pulls back no harder than one that is 3 cm off, and a run whose depth
                     diverged reached exp(20) with the anchor active. Log space penalises the RATIO,
                     which is what a scale error is, so the restoring force grows with the number of
                     decades. ``margin`` is then read in log units.

    Returns:
        loss_scalar: scalar tensor (0 when no valid joints; never NaN).
        info: {"hand_depth_residual_m": float, "n_valid": int}.
    """
    if direction not in ("scene_follows_hand", "hand_follows_scene", "bidirectional"):
        raise ValueError(f"unknown anchor direction: {direction!r}")

    # Sampling location is always the detached hand projection (stable 2D grid).
    grid_xy_det, z_det = project_joints_to_norm_pixels(pred_joints.detach(), cam_intr)
    sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy_det)

    # Broadcast has_hand [B,S,H] over the J joint axis -> [B,S,H,J].
    has_hand_j = has_hand.unsqueeze(-1).to(torch.bool).expand_as(sampled)
    valid = has_hand_j & in_frame & (sampled > depth_min)

    if gs_depth_conf is not None:
        conf, _ = sample_depth_at_joints(gs_depth_conf, grid_xy_det)
        valid = valid & (conf > conf_thresh)

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        loss = torch.zeros((), dtype=sampled.dtype, device=sampled.device)
        return loss, {"hand_depth_residual_m": 0.0, "n_valid": 0}

    # Select which side carries gradient. `z_grad` re-projects WITHOUT detach so
    # the hand depth differentiates back to the head; `z_det` is the fixed target.
    if direction == "scene_follows_hand":
        src, tgt = sampled, z_det                      # grad -> gs_depth only
    elif direction == "hand_follows_scene":
        _, z_grad = project_joints_to_norm_pixels(pred_joints, cam_intr)
        src, tgt = z_grad, sampled.detach()            # grad -> hand only
    else:  # bidirectional
        _, z_grad = project_joints_to_norm_pixels(pred_joints, cam_intr)
        src, tgt = sampled, z_grad                     # grad -> both

    if space not in ("linear", "log"):
        raise ValueError(f"unknown anchor space: {space!r}")
    if space == "log":
        # clamp_min guards the log against a non-positive sample that slipped the depth_min gate
        src, tgt = src.clamp_min(_LOG_EPS).log(), tgt.clamp_min(_LOG_EPS).log()
    per_joint = F.smooth_l1_loss(src, tgt, beta=margin, reduction="none")
    valid_f = valid.to(per_joint.dtype)
    loss = (per_joint * valid_f).sum() / valid_f.sum()

    with torch.no_grad():
        residual = ((sampled - z_det).abs() * valid_f).sum() / valid_f.sum()

    return loss, {
        "hand_depth_residual_m": float(residual.item()),
        "n_valid": n_valid,
    }
