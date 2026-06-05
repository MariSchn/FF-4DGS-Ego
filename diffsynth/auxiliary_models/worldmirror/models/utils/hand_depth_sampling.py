"""Shared geometry helper: project metric MANO joints into the Gaussian depth
map and sample scene depth at those pixels.

This is the correctness-critical core shared by:
  * the L1 training anchor loss   (scripts/hand_depth_anchor_loss.py)
  * the L2 inference scale solve  (heads/metric_scale_head.py)

Coordinate convention (LOCKED against the existing pipeline)
-----------------------------------------------------------
Predicted MANO joints are in per-frame CAMERA space, metric metres
(train_hand_head.py:1247; the 2D-loss comment at :1292 confirms camera frame).
We project them with the SAME convention as the 2D reprojection loss
(train_hand_head.py:1304-1311), which mirrors project_joints_torch /
project_vertices (hand_vis_utils.py:317-343, "90 deg CW rotation to match MP4
video orientation"):

    col = f * X / z + cx          # horizontal camera projection  (= p[0])
    row = f * Y / z + cy          # vertical   camera projection  (= p[1])
    u   = (W - 1) - row           # rotated horizontal -> WIDTH  axis of the frame
    v   = col                     # rotated vertical   -> HEIGHT axis of the frame

The hand-bbox pipeline (train_hand_head.py:434-458) sets bbox x<-u, y<-v, and the
injection indexes features as [..., y1:y2, x1:x2] (hand_to_gs_injection.py); i.e.
the GS feature maps / gs_depth array [..., Hd, Wd] use WIDTH<-u, HEIGHT<-v. So for
F.grid_sample: grid[..., 0] (x) = u / W, grid[..., 1] (y) = v / W.

gs_depth shape is [B, S, 1, Hd, Wd] (DPTHead is_gsdpt, output_dim=2, 'exp'
activation -> positive depth), worldmirror.py:434-440.

Assumption (verified in design Phase 2): `imgs`/`gs_depth` are a full-frame
resize/center-crop of the square 1408x1408 Aria frame, so normalised [0,1] pixel
coordinates transfer directly into the depth map regardless of its (Hd, Wd).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGE_WIDTH: float = 1408.0  # square Aria RGB frame; see train_hand_head.py:1307
Z_MIN: float = 0.05          # 5 cm clamp on projection depth; train_hand_head.py:1304


def project_joints_to_norm_pixels(
    pred_joints: torch.Tensor,
    cam_intr: torch.Tensor,
    image_width: float = IMAGE_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-frame joints to normalised (x=width, y=height) pixels.

    Args:
        pred_joints: [B, S, H, J, 3] camera-frame metric joints (metres).
        cam_intr:    [B, 3] = [focal, cx, cy] in the ``image_width`` pixel frame.
        image_width: square frame size used for normalisation.

    Returns:
        grid_xy:  [B, S, H, J, 2] normalised pixel coords (x=width, y=height) in [0, 1].
        z_metric: [B, S, H, J] metric camera-frame depth (raw, unclamped) per joint.
    """
    if pred_joints.dim() != 5 or pred_joints.shape[-1] != 3:
        raise ValueError(f"pred_joints must be [B,S,H,J,3], got {tuple(pred_joints.shape)}")
    if cam_intr.dim() != 2 or cam_intr.shape[-1] != 3:
        raise ValueError(f"cam_intr must be [B,3], got {tuple(cam_intr.shape)}")

    B = pred_joints.shape[0]
    f = cam_intr[:, 0].view(B, 1, 1, 1)
    cx = cam_intr[:, 1].view(B, 1, 1, 1)
    cy = cam_intr[:, 2].view(B, 1, 1, 1)

    x = pred_joints[..., 0]
    y = pred_joints[..., 1]
    z = pred_joints[..., 2].clamp_min(Z_MIN)

    col = f * x / z + cx
    row = f * y / z + cy
    u = (image_width - 1.0) - row  # width axis
    v = col                        # height axis

    grid_xy = torch.stack([u, v], dim=-1) / image_width  # [B, S, H, J, 2] in [0, 1]
    return grid_xy, pred_joints[..., 2]


def sample_depth_at_joints(
    gs_depth: torch.Tensor,
    grid_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear-sample ``gs_depth`` at normalised joint pixel locations.

    Resolution-independent: ``grid_xy`` is normalised, so the result does not
    depend on ``gs_depth``'s (Hd, Wd).

    Args:
        gs_depth: [B, S, 1, Hd, Wd] or [B, S, Hd, Wd], positive depth.
        grid_xy:  [B, S, H, J, 2] normalised (x=width, y=height) in [0, 1].

    Returns:
        sampled:  [B, S, H, J] sampled depth.
        in_frame: [B, S, H, J] bool, True where the joint projects inside [0, 1]^2.
    """
    if gs_depth.dim() == 5:
        if gs_depth.shape[2] != 1:
            raise ValueError(f"expected gs_depth channel dim 1, got {tuple(gs_depth.shape)}")
        gs_depth = gs_depth[:, :, 0]
    if gs_depth.dim() != 4:
        raise ValueError(f"gs_depth must be [B,S,1,Hd,Wd] or [B,S,Hd,Wd], got {tuple(gs_depth.shape)}")

    B, S, Hd, Wd = gs_depth.shape
    H, J = grid_xy.shape[2], grid_xy.shape[3]

    x = grid_xy[..., 0]
    y = grid_xy[..., 1]
    in_frame = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)

    grid = torch.stack([x * 2.0 - 1.0, y * 2.0 - 1.0], dim=-1)  # [-1, 1]
    inp = gs_depth.reshape(B * S, 1, Hd, Wd)
    samp = F.grid_sample(
        inp, grid.reshape(B * S, H * J, 1, 2),
        mode="bilinear", align_corners=False, padding_mode="border",
    )
    return samp.reshape(B, S, H, J), in_frame
