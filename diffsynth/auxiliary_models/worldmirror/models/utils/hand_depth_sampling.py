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

Normalisation width (FIXED 2026-08-06, was a correctness bug)
-------------------------------------------------------------
`col`/`row` above are in the pixel frame that ``cam_intr`` is expressed in, so the
normalisation constant must be THAT frame's width, not a fixed one. This module
previously hardcoded 1408 (the square Aria frame it was written for) and took it as a
default, while HOI4D and H2O stores carry intrinsics rescaled to the packing
resolution: `preprocess_hoi4d.load_intrinsics` applies `s = res / crop`, and H2O is
packed at `--res 224`. On a 224 store a joint at the image centre normalised to
(0.9197, 0.0795) instead of (0.5, 0.5), i.e. the depth was sampled in the frame corner,
and `in_frame` still returned True so nothing shouted.

We now derive the width from the intrinsics as ``W = 2 * cx``, which is the same
invariant `eval_world_space._intr_3x3` already documents ("principal point cx~W/2 lets
us rescale to res"). This is store-agnostic and needs no caller changes. Pass
``image_width`` explicitly only to override.

The u/v swap above is NOT Aria-specific and is deliberately preserved: the hand-bbox
pipeline sets bbox x<-u, y<-v and the injection indexes [..., y1:y2, x1:x2], so the
rotation is a pipeline-wide locked convention rather than a sensor property.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

IMAGE_WIDTH: float = 1408.0  # square Aria RGB frame; ONLY a fallback, see note above
_OFFCENTRE_TOL_PX: float = 1.0   # how far 2*cx and 2*cy may disagree before the
                                 # centred-principal-point assumption is unsafe
_warned: dict[str, bool] = {"offcentre": False}
Z_MIN: float = 0.05          # 5 cm clamp on projection depth; train_hand_head.py:1304


def frame_width_from_intr(cam_intr: torch.Tensor) -> torch.Tensor:
    """LAST-RESORT estimate of the pixel-frame width, as ``2 * cx``.

    This is ONLY valid when the principal point sits at the frame centre, and that is not
    true of every store we train and evaluate on. HOI4D ships
    ``[f, cx, cy] = [219.92, 114.28, 108.52]`` for a frame that is really 224x224, so
    ``2*cx = 228.56`` overestimates the width by 2.0% while ``2*cy = 217.04`` disagrees with
    it by 11.5 px. The width is then NOT recoverable from the intrinsics alone.

    Prefer passing ``image_width`` explicitly. This function warns once when the intrinsics
    themselves show the assumption is unsafe.

    Args:
        cam_intr: [B, 3] = [focal, cx, cy].
    Returns:
        [B] frame widths in pixels, estimated as ``2 * cx``.
    """
    w_x = 2.0 * cam_intr[:, 1]
    w_y = 2.0 * cam_intr[:, 2]
    if bool((w_x - w_y).abs().max() > _OFFCENTRE_TOL_PX) and not _warned["offcentre"]:
        _warned["offcentre"] = True
        warnings.warn(
            f"frame_width_from_intr: the principal point is NOT centred "
            f"(2*cx={float(w_x.reshape(-1)[0]):.2f} vs 2*cy={float(w_y.reshape(-1)[0]):.2f}). "
            "The frame width cannot be derived from the intrinsics alone; pass image_width "
            "explicitly or every depth sample lands off-pixel.",
            RuntimeWarning, stacklevel=2)
    return w_x


def project_joints_to_norm_pixels(
    pred_joints: torch.Tensor,
    cam_intr: torch.Tensor,
    image_width: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-frame joints to normalised (x=width, y=height) pixels.

    Args:
        pred_joints: [B, S, H, J, 3] camera-frame metric joints (metres).
        cam_intr:    [B, 3] = [focal, cx, cy] in some square pixel frame.
        image_width: square frame size used for normalisation. ``None`` (the default and
            the correct choice for every store) derives it per batch element from the
            intrinsics as ``2 * cx``. Pass a float only to override deliberately.

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

    if image_width is None:
        W = frame_width_from_intr(cam_intr).view(B, 1, 1, 1)
    else:
        W = torch.full_like(cx, float(image_width))
    if not bool((W > 1.0).all()):
        raise ValueError(
            f"degenerate frame width from intrinsics: cx={cam_intr[:, 1].tolist()}. "
            "cam_intr must be [focal, cx, cy] in pixels with the principal point near the "
            "frame centre."
        )

    x = pred_joints[..., 0]
    y = pred_joints[..., 1]
    z = pred_joints[..., 2].clamp_min(Z_MIN)

    col = f * x / z + cx
    row = f * y / z + cy
    u = (W - 1.0) - row  # width axis
    v = col              # height axis

    # The +0.5 is the pixel-CENTRE offset, not a fudge. sample_depth_at_joints maps x in
    # [0,1] to g = 2x-1 and calls grid_sample(align_corners=False), which reads pixel
    # p = x*W - 0.5, so landing on the centre of pixel k requires x = (k + 0.5)/W. Without it
    # every sample sat half a pixel off, and pushing identical prediction and ground truth
    # through the object-depth path returned a non-zero residual.
    grid_xy = (torch.stack([u, v], dim=-1) + 0.5) / W.unsqueeze(-1)  # [B, S, H, J, 2] in [0, 1]
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
        if gs_depth.shape[2] == 1:
            gs_depth = gs_depth[:, :, 0]
        elif gs_depth.shape[4] == 1:
            gs_depth = gs_depth[:, :, :, :, 0]
        else:
            raise ValueError(f"expected gs_depth channel dim 1, got {tuple(gs_depth.shape)}")
    if gs_depth.dim() != 4:
        raise ValueError(f"gs_depth must be [B,S,1,Hd,Wd], [B,S,Hd,Wd,1] or [B,S,Hd,Wd], got {tuple(gs_depth.shape)}")

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
