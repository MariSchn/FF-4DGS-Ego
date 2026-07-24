"""Orchestration for the Phase-1 scene-depth root anchor: project the predicted
wrist, sample the (detached) metric gs_depth there, run RootDepthRefine, and apply
the per-hand depth shift to the joints. Shared by train_hand_head and
eval_world_space so train-time and eval-time corrections are identical.
"""
import os

import torch
import torch.nn.functional as F

try:
    from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
        project_joints_to_norm_pixels, sample_depth_at_joints,
    )
except Exception:  # dev machines lack diffsynth's heavy deps (modelscope); load the pure module directly
    import importlib.util as _ilu
    _p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "diffsynth/auxiliary_models/worldmirror/models/utils/hand_depth_sampling.py",
    )
    _spec = _ilu.spec_from_file_location("hand_depth_sampling", _p)
    _hds = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_hds)
    project_joints_to_norm_pixels = _hds.project_joints_to_norm_pixels
    sample_depth_at_joints = _hds.sample_depth_at_joints

WRIST_J = 0  # MANO joint 0 = wrist (pelvis_id used by the kp losses)


def apply_root_anchor(module, pred_joints, gs_depth, gs_depth_conf, cam_intr,
                      contact_mask=None, ref_d_scene=None):
    """pred_joints [B,S,2,J,3] camera-frame (m). gs_depth [B,S,1,Hd,Wd] (detached
    inside). cam_intr [B,3]. contact_mask [B,S,2] bool optional: when given it
    REPLACES the module's |disagree|<band_m proxy gate (fire only at true contact,
    where the scene depth is reliable). Returns (corrected_joints, delta_z, info).

    ref_d_scene [B,S,2] optional: an EXTERNAL metric depth-at-wrist reference (e.g.
    DA3METRIC-LARGE, precomputed per frame) that REPLACES the gs_depth sampling as the
    anchor's target. This is the "frozen feedforward metric model -> trainable refine"
    path (Cyrus): the head's wrist depth is pulled toward a strong external metric depth
    rather than the frozen backbone's gs_depth. gs_depth may be None when ref is given."""
    wrist = pred_joints[:, :, :, WRIST_J:WRIST_J + 1, :]          # [B,S,2,1,3]
    grid_xy, z = project_joints_to_norm_pixels(wrist, cam_intr)   # [B,S,2,1,2], [B,S,2,1]
    wrist_z = z[..., 0]
    if ref_d_scene is not None:
        d_scene = ref_d_scene.to(wrist_z.device, wrist_z.dtype)   # [B,S,2] external metric depth at wrist
        # Fixed global bias correction for the reference (DA3 on HOI4D reads ~0.892x
        # true depth). Without this every variant pulled the hand ~70mm too close.
        # ANCHOR_REF_SCALE (env) overrides the module constant so the TRAIN-split
        # refit value (scripts/fit_ref_scale.py, D2-4) can be applied without code
        # or config edits; unset -> module.ref_scale -> 1.0 as before.
        _rs = float(os.environ.get("ANCHOR_REF_SCALE",
                                   getattr(module, "ref_scale", 1.0)))
        if _rs != 1.0:
            d_scene = d_scene * _rs
        conf = torch.ones_like(d_scene)
        in_frame = torch.isfinite(d_scene) & (d_scene > 0.01) & torch.isfinite(wrist_z)
    else:
        d_scene, in_frame = sample_depth_at_joints(gs_depth.detach(), grid_xy)  # [B,S,2,1]
        if gs_depth_conf is not None:
            conf, _ = sample_depth_at_joints(gs_depth_conf.detach(), grid_xy)
        else:
            conf = torch.ones_like(d_scene)
        d_scene = d_scene[..., 0]
        conf = conf[..., 0]
        in_frame = in_frame[..., 0]
        in_frame = in_frame & (d_scene > 0.01) & torch.isfinite(d_scene) & torch.isfinite(wrist_z)

    # d_scene (esp. the DA3 ref cache) can be NaN where the wrist projects out of frame
    # or DA3 skipped it. in_frame already excludes those, but a NaN in the MLP input
    # poisons delta (NaN * gate = NaN, not 0) -> NaN pred_joints -> NaN kp losses at
    # step 0. Sanitize so gated-out positions get a finite 0 delta.
    d_scene = torch.nan_to_num(d_scene, nan=0.0, posinf=0.0, neginf=0.0)
    conf = torch.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    delta_z, gate = module(wrist_z, d_scene, conf, in_frame, contact=contact_mask)  # [B,S,2]
    if getattr(module, "correction", "z") == "ray":
        # A depth error on a correct 2D projection scales the position along the viewing
        # ray, so move the whole hand rigidly by root*(dz/z): x,y,z shift together and the
        # root stays on its ray. z-only shifts leave a lateral error of (r/Z)*dz that the
        # re-anchor oracle (full 3D) never had.
        root = pred_joints[:, :, :, WRIST_J, :]                        # [B,S,2,3]
        safe = wrist_z.clamp(min=0.05)
        ratio = torch.where(wrist_z > 0.05, delta_z / safe, torch.zeros_like(delta_z))
        corrected = pred_joints + (root * ratio.unsqueeze(-1)).unsqueeze(-2)
    else:
        corrected = pred_joints.clone()
        corrected[..., 2] = corrected[..., 2] + delta_z.unsqueeze(-1)  # z-only depth shift
    info = {"d_scene": d_scene, "wrist_z": wrist_z, "conf": conf, "gate": gate}
    return corrected, delta_z, info


def root_anchor_loss(corrected_wrist_z, d_scene, gate, has_hand, delta_m: float = 0.05):
    """Gated Huber pulling the corrected wrist depth toward the scene depth.
    All [B,S,2]. Zero (no grad) when nothing is gated, avoiding an empty-mean NaN."""
    mask = (gate & (has_hand > 0.5)).float()
    denom = mask.sum()
    if float(denom) < 1.0:
        return corrected_wrist_z.sum() * 0.0
    per = F.huber_loss(corrected_wrist_z, d_scene, reduction="none", delta=delta_m)
    return (mask * per).sum() / denom
