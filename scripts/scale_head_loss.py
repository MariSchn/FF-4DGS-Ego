"""Supervision for the feedforward ScaleHead (the "scale head" route).

Two targets are available and they are not equivalent.

``scale_head_loss`` trains ``s * gs_depth`` toward the metric MANO hand depth at the projected
joints. Its optimum is the closed-form median ratio, but that ratio is a BIASED target: the frozen
backbone does not reconstruct the thin foreground hand, so the depth sampled at hand pixels is the
background behind it, about 1.37x too far. Measured with ground-truth hand depth, the resulting
scale lands near 0.73 against a true 1.02, and the bias is systematic rather than noise.

``scale_head_gt_loss`` trains the head against the scale that aligns the predicted camera
trajectory to the ground-truth one, which is what the world metrics actually use. It does not read
scene depth at all, so it does not inherit that bias.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)
from scripts.world_space_metrics import solve_similarity


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


def scale_head_gt_loss(
    pred_scale: torch.Tensor,
    pred_c2w: torch.Tensor,
    gt_w2c: torch.Tensor,
    *,
    beta: float = 0.1,
    min_baseline: float = 0.01,
    clamp: tuple[float, float] = (0.1, 10.0),
) -> tuple[torch.Tensor, dict]:
    """Smooth-L1 in log space between the predicted scale and the ground-truth camera scale.

    The target is the Umeyama scale taking the predicted camera centres onto the ground-truth
    ones, i.e. the same quantity ``eval_world_space`` reports as ``s_gt``. Log space because a
    scale is multiplicative: predicting 2x and predicting 0.5x are equally wrong.

    Args:
        pred_scale: [B] positive predicted scale, the only tensor carrying gradient.
        pred_c2w:   [B, S, 4, 4] predicted camera-to-world.
        gt_w2c:     [B, S, 4, 4] ground-truth world-to-camera, the store's convention.
        beta:       Huber beta, in log units.
        min_baseline: metres the ground-truth camera must travel for the scale to be
            determined. Measured over 16-frame HOI4D clips the median travel is 1.7 cm, so
            0.05 would reject 86% of them and most batches would carry no target at all.
        clamp:      reject targets outside this range rather than training toward them.

    Returns:
        loss_scalar: scalar tensor, 0 when no clip in the batch has a usable target.
        info: {"n_clips", "s_gt_mean", "s_pred_mean", "log_residual"}.
    """
    idx, targets = [], []
    for b in range(pred_c2w.shape[0]):
        pc = pred_c2w[b, :, :3, 3].detach().to(torch.float64)
        gc = torch.linalg.inv(gt_w2c[b].detach().to(torch.float64))[:, :3, 3]
        keep = torch.isfinite(pc).all(-1) & torch.isfinite(gc).all(-1)
        if int(keep.sum()) < 3:
            continue
        pc, gc = pc[keep], gc[keep]
        # A camera that does not move leaves the scale undetermined, and solve_similarity divides
        # by that near-zero variance. Worse, below three correspondences it returns 1.0, so an
        # unguarded call teaches the head to predict identity on exactly the clips carrying no
        # signal. Require real travel in the ground-truth trajectory instead.
        if float((gc - gc.mean(0)).norm(dim=-1).max()) < min_baseline:
            continue
        if float((pc - pc.mean(0)).norm(dim=-1).max()) < 1e-6:
            continue
        s_gt, _, _ = solve_similarity(pc, gc)
        s_gt = float(s_gt)
        if not (clamp[0] < s_gt < clamp[1]):
            continue
        idx.append(b)
        targets.append(s_gt)

    info = {"n_clips": len(idx),
            "s_pred_mean": float(pred_scale.mean().item()),
            "s_gt_mean": 0.0,
            "s_gt_std": 0.0,
            "log_residual": 0.0}
    if not idx:
        # Multiply rather than return a bare zero: backward() on a tensor with no grad_fn raises,
        # so a batch where every clip was rejected would abort the run instead of contributing
        # nothing. At 16 frames most HOI4D clips move the camera under 2 cm, so this is common.
        return pred_scale.sum() * 0.0, info

    sel = pred_scale[torch.tensor(idx, device=pred_scale.device)]
    tgt = torch.tensor(targets, dtype=sel.dtype, device=sel.device)
    log_pred = torch.log(sel.clamp_min(1e-6))
    log_tgt = torch.log(tgt)
    loss = F.smooth_l1_loss(log_pred, log_tgt, beta=beta)

    with torch.no_grad():
        info["s_gt_mean"] = float(tgt.mean().item())
        info["log_residual"] = float((log_pred - log_tgt).abs().mean().item())
        info["s_gt_std"] = float(tgt.std().item()) if tgt.numel() > 1 else 0.0
    return loss, info
