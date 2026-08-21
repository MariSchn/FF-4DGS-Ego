# Hand-scene registration in image space

## Problem

Nothing in the objective relates the reconstructed scene to the predicted hand. The two Gaussian
losses compare the render against the ground-truth frame over the whole image, so they are
satisfied by a good reconstruction regardless of where the hand head placed the hand. The coupling
we claim is architectural, through feature injection, and never supervised.

The existing dense registration (`scripts/hand_scene_registration_loss.py`) does relate them, but
in metric depth, so it needs the scene scale. That scale has median 1.00 and spread 0.55, and we
have measured that correcting it is worth under 1 mm of W-MPJPE. Paying for a noisy scale to get a
coupling term is the wrong trade.

## Proposal

Compare the predicted MANO surface against the rendered depth **after normalising both within the
hand's own window**. Local normalisation removes any global scale factor by construction, so the
term needs no `pred_scale` and inherits none of its noise.

Everything on both sides is predicted. No ground truth enters.

## What it measures

Whether the hand has the same relative depth structure as the scene where it projects: the same
tilt, the same ordering between near and far fingers, the same curvature. Not how far away it is.
Absolute distance is already anchored by `kp3d_abs`, and the two terms divide the work.

## Code

New file, `scripts/hand_scene_image_registration.py`.

```python
"""Scale-free hand-scene registration in image space.

The dense metric version needs pred_scale to reconcile normalised scene depth with metric MANO
depth. Normalising both within the hand's own window cancels any global factor, so this term
registers the two surfaces without ever seeing a scale.
"""
import torch
import torch.nn.functional as F

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)

VERT_STRIDE = 15   # 778 -> 52, the stride hand_scene_registration_loss already uses


def _robust_norm(x, valid, eps=1e-6):
    """Median-centre and MAD-scale over the valid entries of the last axis."""
    n = valid.sum(-1, keepdim=True).clamp_min(1)
    xm = torch.where(valid, x, torch.zeros_like(x))
    med = xm.sum(-1, keepdim=True) / n
    dev = torch.where(valid, (x - med).abs(), torch.zeros_like(x))
    mad = dev.sum(-1, keepdim=True) / n
    return (x - med) / (mad + eps)


def hand_scene_image_registration_loss(
    pred_verts, rendered_depth, has_hand, cam_intr, *,
    beta=0.5, depth_min=0.01, direction="bidirectional",
    depth_conf=None, conf_thresh=0.0,
):
    """Huber between the locally normalised rendered depth and MANO vertex depth.

    Args:
        pred_verts:     [B, S, H, 778, 3] camera-frame MANO vertices, metres.
        rendered_depth: [B, S, 1, Hd, Wd] depth from the Gaussian rasteriser.
        has_hand:       [B, S, H] in {0, 1}.
        cam_intr:       [B, 3] = [focal, cx, cy].
        direction:      hand_only | scene_only | bidirectional.
    """
    verts = pred_verts[..., ::VERT_STRIDE, :]

    # The sampling grid is detached on purpose. With gradient, the cheapest way to reduce the
    # residual is to slide vertices onto pixels whose depth already agrees, which optimises the
    # lookup rather than the geometry.
    grid_det, z_det = project_joints_to_norm_pixels(verts.detach(), cam_intr)
    sampled, in_frame = sample_depth_at_joints(rendered_depth, grid_det)

    valid = (has_hand.unsqueeze(-1).bool().expand_as(sampled)
             & in_frame & (sampled > depth_min)
             & torch.isfinite(sampled) & torch.isfinite(z_det))
    if depth_conf is not None:
        conf, _ = sample_depth_at_joints(depth_conf, grid_det)
        valid = valid & (conf > conf_thresh)

    info = {"n_valid": int(valid.sum()), "residual": 0.0}
    if not bool(valid.any()):
        return pred_verts.sum() * 0.0, info

    scene = sampled if direction in ("scene_only", "bidirectional") else sampled.detach()
    if direction in ("hand_only", "bidirectional"):
        _, hand = project_joints_to_norm_pixels(verts, cam_intr)
    else:
        hand = z_det

    scene_n = _robust_norm(scene, valid)
    hand_n = _robust_norm(hand, valid)

    per_vert = F.smooth_l1_loss(scene_n, hand_n, beta=beta, reduction="none")
    vf = valid.to(per_vert.dtype)
    loss = (per_vert * vf).sum() / vf.sum()

    with torch.no_grad():
        info["residual"] = float(((scene_n - hand_n).abs() * vf).sum() / vf.sum())
    return loss, info
```

## Integration

`render_views_from_predictions` in `scripts/gs_metrics.py` already receives the depth from the
rasteriser and discards it. It gains a flag:

```python
def render_views_from_predictions(model, preds, views, height, width, return_depth=False):
    rendered, depth, _alpha = model.gs_renderer.rasterizer.forward(...)
    rgb = rendered.clamp(0.0, 1.0).float()
    return (rgb, depth) if return_depth else rgb
```

Default unchanged, so no existing caller moves.

In `scripts/train_hand_head.py`, beside the other optional terms:

```python
loss_img_reg = torch.zeros((), device=device)
if w_img_reg > 0.0 and rendered_depth is not None:
    pred_verts = compute_vertices_from_batch(pred_params, mano_model, device)
    loss_img_reg, _reg_info = hand_scene_image_registration_loss(
        pred_verts, rendered_depth, has_hand, batch["cam_intrinsics"].to(device),
        direction=img_reg_direction,
    )
```

and one term in the weighted sum, one key in the accumulators, one field in the printed breakdown.

## Guards

Three, all copied from terms that already carry them because each one was paid for once.

Frames with no hand are dropped through `has_hand`. Without it an empty MANO in an unfilled hand
slot contributes a constant residual that never decays, which is what the joint loss comments
record and what the scale solve was found to be ingesting.

Vertices outside the frame, at degenerate depth, or non-finite are dropped. Optionally the
Gaussian head's own depth confidence gates the sample.

An empty valid set returns `pred_verts.sum() * 0.0` rather than a bare zero, because a tensor with
no `grad_fn` aborts `backward()` instead of contributing nothing.

## How to read it

Two arms, identical but for the weight, on the five-dataset pool. The term is new, so the readout
is not W-MPJPE, which we already know the scale barely moves. It is:

- `residual`, the normalised depth disagreement, which should fall if the term does anything,
- hand-region PSNR and LPIPS, which is where a real coupling should show,
- **absolute C-MPJPE, as the cheat detector.**

The last one is the point. The loss can be minimised by moving the hand instead of the scene, and
that is the cheapest route. If the residual falls while absolute C-MPJPE rises, the term is buying
consistency with accuracy and `kp3d_abs` needs more weight, or the direction should be
`scene_only`.

## Alternative considered

Comparing the projected MANO silhouette against the rasteriser's alpha is also scale-free and also
geometric, and it is simpler. It was not chosen because alpha is near-saturated in a densely
reconstructed scene, so the silhouette would land on a region that is already opaque with or
without the hand. Depth carries the discriminative signal that occupancy does not.
