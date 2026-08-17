#!/usr/bin/env python3
"""Convert the predicted Gaussians from the reconstructor's units into metres.

The world lift places the hand with ``X_world = R_t X_cam + s t_t``: the joints leave the hand head
already in metres, so only the camera position needs the scene scale. The Gaussians were never
given the same treatment. Their positions are ``R_t (D_t ray) + t_t`` with the predicted depth and
the camera translation both in the reconstructor's units, so the hand landed in a metric world
while the scene stayed in the original one.

This applies the same ``s`` to the scene. Note what it does NOT do: rendering is invariant under a
similarity applied to the scene AND its cameras, so this changes world coordinates and leaves every
rendered image, and therefore every PSNR, exactly as it was. What it buys is that a distance
measured between a hand joint and a Gaussian is in metres.

WHAT MUST SCALE, and the ones that are easy to miss:
  means            positions
  scales           extents, or the Gaussians land correctly and keep the wrong size
  forward/backward vel      velocities are length over time
  forward/backward scales   extents at the other timestamps

WHAT MUST NOT:
  rotations (unit quaternions), harmonics (colour), opacities, timestamps, life_span.
"""
from __future__ import annotations

_LENGTH_FIELDS = ("means", "scales", "forward_vel", "backward_vel",
                  "forward_scales", "backward_scales")


def scale_gaussians_(g, s: float):
    """Scale one Gaussians object into metres, in place. Returns the object.

    Every field that carries a length is multiplied; colour, opacity and orientation are not.
    """
    if s == 1.0:
        return g
    for name in _LENGTH_FIELDS:
        v = getattr(g, name, None)
        if v is not None:
            setattr(g, name, v * s)
    return g


def scale_splats_(splats, s: float):
    """Scale the nested ``List[List[Gaussians]]`` the model returns as ``preds['splats']``."""
    for per_batch in splats:
        for g in (per_batch if isinstance(per_batch, (list, tuple)) else [per_batch]):
            scale_gaussians_(g, s)
    return splats


def metric_world_from_predictions(preds, s: float):
    """Put the scene and the cameras that define it into metres together.

    The camera translations are scaled alongside the Gaussians so that the pair stays consistent.
    Scaling only one of the two would move the scene relative to the cameras and corrupt every
    render, which is the failure this function exists to avoid.
    """
    if s == 1.0:
        return preds
    if "splats" in preds:
        scale_splats_(preds["splats"], s)
    for key in ("rendered_extrinsics", "camera_poses"):
        e = preds.get(key)
        if e is not None:
            e[..., :3, 3] = e[..., :3, 3] * s
    return preds
