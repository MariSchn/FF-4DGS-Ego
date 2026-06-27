"""Per-hand contact signal: the GT wrist sits ON the visible GT surface.

Used to gate the contact anchor where the scene depth is reliable. The
scale-source contact-stratified test (commit 291931d) found the hand recovers
the scene's metric scale almost perfectly within 5cm of true contact
(s_contact=1.005) but biased low elsewhere (s_noncontact=0.804) -- so the anchor
should only pull the hand root toward the scene depth at contact. Contact is
defined by GT only (GT wrist depth vs GT dense sensor depth), so it is
non-circular w.r.t. the predicted hand the anchor corrects.
"""
import torch

# Reuse the projection/sampling helpers (scripts.root_depth_anchor already imports
# them with a dev-machine fallback, so this module needs no import handling).
from scripts.root_depth_anchor import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)

WRIST_J = 0
DEPTH_MIN = 0.01


def is_contact(wrist_z, dense_at_wrist, in_frame, thresh_m: float = 0.05):
    """All [B, S, 2] (per hand). Contact iff the wrist is in-frame, the surface
    depth is valid, and |wrist_z - dense_at_wrist| < thresh_m. Returns bool [B,S,2]."""
    valid = (in_frame & (dense_at_wrist > DEPTH_MIN)
             & torch.isfinite(dense_at_wrist) & torch.isfinite(wrist_z))
    return valid & ((wrist_z - dense_at_wrist).abs() < thresh_m)


def wrist_contact_mask(wrist_cam, dense_depth, cam_intr, thresh_m: float = 0.05):
    """wrist_cam [B,S,2,3] camera-frame GT wrist (m); dense_depth [B,S,1,H,W] GT
    metric depth; cam_intr [B,3] = [focal, cx, cy]. Returns contact bool [B,S,2]."""
    grid_xy, z = project_joints_to_norm_pixels(wrist_cam.unsqueeze(3), cam_intr)  # [B,S,2,1,2],[B,S,2,1]
    d, in_frame = sample_depth_at_joints(dense_depth, grid_xy)                    # [B,S,2,1]
    return is_contact(z[..., 0], d[..., 0], in_frame[..., 0], thresh_m)
