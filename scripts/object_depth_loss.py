"""GT object-depth supervision loss (the direct metric-depth signal).

The hand anchor (``hand_depth_anchor_loss``) only constrains ``gs_depth`` AT the
hand, where the model is already trained to match the metric hand — so it cannot,
by itself, make the SCENE metric. B2 confirmed this: on the manipulated objects
the anchored scene depth was ~135 cm off. This term adds the missing signal: it
supervises ``gs_depth`` directly toward the GT metric depth of the objects,
rendered from the HOT3D object meshes + 6-DoF poses
(``scripts.b2_render_object_depth``). On the objects the model now has a real
metric target, not just the hand.

Gradient flows into ``gs_depth`` (and, when backbone blocks are unfrozen, back
into the encoder), which is exactly the "train the Gaussian head (and a few
layers) on the HOT3D metric depth" direction.

Sampling reuses the proven, resolution-independent normalized-grid path
(``sample_depth_at_joints``); ``gs_depth`` is never indexed directly, so its
internal resolution never has to match the render resolution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    sample_depth_at_joints,
)


def object_depth_loss(
    gs_depth: torch.Tensor,
    gt_obj_depth: torch.Tensor,
    gt_obj_mask: torch.Tensor,
    *,
    margin: float = 0.05,
    depth_min: float = 0.01,
    depth_max: float = 50.0,
) -> tuple[torch.Tensor, dict]:
    """Masked smooth-L1 between sampled ``gs_depth`` and GT object depth.

    Args:
        gs_depth:     [B, S, 1, Hd, Wd] (or [B, S, Hd, Wd]) positive scene depth (m).
        gt_obj_depth: [B, S, R, R] GT object depth (m); 0 where no object surface.
        gt_obj_mask:  [B, S, R, R] bool; True on valid (rendered) object pixels.
        margin:       smooth-L1 (Huber) beta in metres.
        depth_min:    reject sampled scene depth <= this (invalid geometry).
        depth_max:    reject sampled scene depth >= this (runaway predictions).

    Returns:
        loss_scalar: scalar tensor (0 when no valid object pixels; never NaN).
        info: {"n_valid": int, "obj_depth_residual_m": float}.
    """
    B, S = gt_obj_depth.shape[:2]
    R = gt_obj_depth.shape[-1]
    device = gs_depth.device

    losses: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    n_valid = 0

    for b in range(B):
        for s in range(S):
            m = gt_obj_mask[b, s]
            if not bool(m.any()):
                continue
            ys, xs = torch.where(m)
            od = gt_obj_depth[b, s][ys, xs].to(device)
            # Normalized [0,1] grid at the valid object pixels, sampled differentiably.
            grid = torch.stack([xs.float() / R, ys.float() / R], dim=-1)
            grid = grid.view(1, 1, 1, -1, 2).to(device)
            gs_at, in_fr = sample_depth_at_joints(gs_depth[b : b + 1, s : s + 1], grid)
            gs_at = gs_at.reshape(-1)
            in_fr = in_fr.reshape(-1)
            keep = in_fr & (gs_at > depth_min) & (gs_at < depth_max) & (od > depth_min)
            if not bool(keep.any()):
                continue
            losses.append(F.smooth_l1_loss(gs_at[keep], od[keep], beta=margin, reduction="none"))
            with torch.no_grad():
                residuals.append((gs_at[keep] - od[keep]).abs())
            n_valid += int(keep.sum().item())

    if not losses:
        loss = torch.zeros((), dtype=gs_depth.dtype, device=device)
        return loss, {"n_valid": 0, "obj_depth_residual_m": 0.0}

    loss = torch.cat(losses).mean()
    residual = float(torch.cat(residuals).mean().item())
    return loss, {"n_valid": n_valid, "obj_depth_residual_m": residual}
