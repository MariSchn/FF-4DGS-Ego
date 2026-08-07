"""World-space hand-placement eval (workstream C1): W-MPJPE / WA-MPJPE.

Chains FF-4DGS-Ego per-clip camera-frame hand predictions into one world-space
trajectory over `segment_len`-frame segments, then compares to the cached GT world
joints. See report/world-space-eval-design.md.

Run on a gb10 node:
    python -m scripts.eval_world_space --config configs/exp_p3_scalehead.yaml \
        --data_root <hot3d_root> --max_seqs 4 --segment_len 128 \
        --clip_len 16 --stride 8 --wa_short 16 --out world_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import traceback

import torch
import yaml

from scripts.world_space_metrics import (
    apply_similarity,
    c_mpjpe,
    chain_trajectories_by_overlap,
    chain_trajectories_dense,
    chain_trajectories_global,
    chain_trajectories_linked,
    gravity_align_c2w,
    reanchor_to_gt,
    replace_root_with_gt_motion,
    smooth_root_trajectory,
    solve_similarity,
    solve_similarity_robust,
    w_mpjpe,
    w_mpjpe_first_window_aligned,
    wa_mpjpe,
)


# A checkpoint at this step or below cannot have learned anything; evaluating it reports
# untrained-model performance as if it were a result.
MIN_TRAINED_STEP = 10


def _assert_checkpoint_is_trained(path, ck):
    """Refuse to evaluate a checkpoint that was saved before training did anything.

    THIS EXISTS BECAUSE OF C-abs 725. The HOI4D-only control was evaluated from
    best_val_loss.pt, which sat at global_step=1: the config had val_every=3000 while the run
    only reached 2222 steps, so validation ran ONCE at the start and never again, and the
    step-1 model stayed 'best' forever. The eval then dutifully reported 725 mm - untrained
    performance - and nothing in the pipeline objected. Two days of hypotheses (dead kp3d_abs,
    data mixing, box conventions) were spent on what was a checkpoint-selection bug.
    """
    if not isinstance(ck, dict):
        return
    step = ck.get("global_step")
    epoch = ck.get("epoch")
    if step is None:
        return
    print(f"warm-start checkpoint: global_step={step} epoch={epoch}  ({os.path.basename(path)})",
          flush=True)
    if int(step) <= MIN_TRAINED_STEP:
        raise SystemExit(
            f"Refusing to evaluate {path}: global_step={step} means this checkpoint was saved "
            f"before training had progressed, so any metric from it is untrained-model noise "
            f"(this is exactly how C-abs 725 was produced).\n"
            f"  Likely cause: training ran fewer steps than training.val_every, so validation "
            f"never re-ran and the first checkpoint stayed 'best'.\n"
            f"  Fix: evaluate hand_head_final.pt / latest.pt instead of best_val_loss.pt."
        )


# The two irreplaceable heads live in a PRIVATE HF repo; the local copies were deleted once the
# upload verified, which is why a missing file here is a RESTORE, not a rebuild.
_CKPT_BACKUP_REPO = "mondraaa/worldhand4dgs-checkpoints"


def _require_checkpoint_present(path):
    """Fail loudly, with the restore command, instead of raising a bare FileNotFoundError.

    WHAT THIS CAUGHT (2026-08-07). zguard job 104380 ran all four arms of the #63 2x2, every one
    died on `FileNotFoundError: .../jitterrob10ep_best.pt`, and sacct still reported the job
    **COMPLETED** because the readout script ran afterwards and exited 0. The readout printed
    "MISSING" on all four rows and nobody was paged. A run that reports success while producing
    nothing is the same failure mode as the enable_gs eval trap and the C-abs-725 checkpoint.

    The bare exception was also unhelpful: it said the file was absent but not that it is
    recoverable, so the obvious reading was "the checkpoint is lost" rather than "run one command".
    """
    if os.path.exists(path):
        return
    name = os.path.basename(path)
    raise SystemExit(
        f"\n!! CHECKPOINT MISSING: {path}\n"
        f"   This is RECOVERABLE. The file lives in the private HF repo {_CKPT_BACKUP_REPO},\n"
        f"   because local copies were deleted after the upload verified.\n\n"
        f"   Restore it with (token on stdin, never in argv):\n"
        f"     printf '%s\\n' \"$HF_TOKEN\" | bash ~/hf_restore.sh   # edit the filename to {name}\n\n"
        f"   Do NOT retrain: the head behind the paper's headline numbers cannot be reproduced\n"
        f"   bit-for-bit. Exiting non-zero so a batch script cannot report success without it.")


def build_model(cfg, device):
    """Build WorldMirror from cfg, load the base checkpoint, warm-start the hand head."""
    from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
    mcfg = cfg["model"]
    model = WorldMirror(**{k: v for k, v in mcfg.items() if k != "checkpoint"})
    ckpt = torch.load(mcfg["checkpoint"], map_location=device)
    state = ckpt.get("state_dict", ckpt.get("reconstructor", ckpt))
    model.load_state_dict(state, strict=False)
    if mcfg.get("warm_start_hand_head"):
        _require_checkpoint_present(mcfg["warm_start_hand_head"])
        ws = torch.load(mcfg["warm_start_hand_head"], map_location=device)
        _assert_checkpoint_is_trained(mcfg["warm_start_hand_head"], ws)
        sd = ws["model_state_dict"] if isinstance(ws, dict) and "model_state_dict" in ws else ws
        hh_keys = set(model.hand_head.state_dict().keys())
        loaded = {k: v for k, v in sd.items() if k in hh_keys}
        if loaded:
            model.hand_head.load_state_dict(loaded, strict=False)
            print(f"Warm-start: loaded {len(loaded)}/{len(hh_keys)} hand_head tensors")
        else:
            # Full-model state dict (hand_head.* prefixed), e.g. saved by
            # train_hand_head.save_checkpoint. Load into the whole model.
            # Without this fallback the trained head is silently ignored and
            # the eval runs the untrained head.
            res = model.load_state_dict(sd, strict=False)
            n_hh = sum(1 for k in sd if k.startswith("hand_head."))
            print(f"Warm-start: full-model dict, loaded {n_hh} hand_head tensors "
                  f"(missing={len(res.missing_keys)})")
    model.to(device).eval()
    return model


def _world_from_cam(pj, c2w, s):
    """Place metric cam-frame joints ``pj`` [S,H,J,3] into the clip-local world via the up-to-scale
    camera poses ``c2w`` [S,4,4], scaling ONLY the camera *translation* by the scene scale ``s``
    (the hand is already metric; only the camera/scene trajectory is up-to-scale). Returns
    [S, H*J, 3]. Keeping this separate from the model forward lets one clip be re-placed under a
    per-clip OR a per-sequence scale without re-running the network."""
    sf, h, jn = pj.shape[0], pj.shape[1], pj.shape[2]
    world = torch.empty(sf, h * jn, 3, device=pj.device)
    for k in range(sf):
        rot, trans = c2w[k, :3, :3], c2w[k, :3, 3] * s
        world[k] = (rot @ pj[k].reshape(-1, 3).T).T + trans
    return world


def ratio_validity_mask(z, sampled, in_frame, require_positive_z=False, hand_present=None):
    """Which joint/scene-depth correspondences may feed the scene-scale median.

    ``z`` is the metric camera-frame hand depth, ``sampled`` the predicted scene depth read at
    the joint's projected pixel, ``in_frame`` whether that pixel landed inside the image.

    ``hand_present`` broadcasts the per-frame, per-hand detector flag over the joint axis
    (task #70). Every other term here is geometric, so a hand slot the detector never filled
    still passes: the model emits a default MANO for the empty slot, those joints project to
    plausible in-frame pixels, and the median consumes them. Measured on five HOI4D clips whose
    box store has ``valid[:, 0].sum() == 0`` over all 300 frames, the absent slot supplied 96
    accepted correspondences per clip at a median ratio of -0.019, and 94.1% of every
    behind-camera correspondence in the sample. This is the CAUSE that ``require_positive_z``
    only masks the symptom of, which is why both exist and why the A/B runs them separately.

    ``require_positive_z`` closes a hole that the other three terms do not cover. A joint behind
    the camera (z < 0) is projected by ``project_joints_to_norm_pixels`` with its depth clamped to
    ``Z_MIN`` = 5 cm, so it lands at a plausible in-frame pixel, while the depth returned for the
    ratio is the raw negative value. The result is a negative ratio that the median consumes,
    which is a candidate cause of the segments that solve exactly onto the 0.1 clamp floor
    (task #63). Kept behind a flag until the A/B measures it, so the change is evidence-led.

    Pulled out of ``predict_clip`` purely so this rule is testable without a GPU forward pass.
    """
    valid = in_frame & (sampled > 0.01) & torch.isfinite(z) & torch.isfinite(sampled)
    if require_positive_z:
        valid = valid & (z > 0)
    if hand_present is not None:
        # hand_present is [S,H] (or anything broadcastable); valid is [...,S,H,J]. Unsqueeze the
        # joint axis rather than reshaping, so a caller passing [1,S,H] still broadcasts.
        valid = valid & hand_present.unsqueeze(-1).to(valid.device).bool()
    return valid


def predict_clip(preds, mano_model, device, cam_intr, model=None, anchor_log=None, contact_mask=None,
                 ref_d_scene=None, depth_out=None, steps_out=None, require_positive_z=False,
                 scene_depth_window=1, scene_depth_reduce="bilinear", hand_valid=None,
                 gate_scale_on_hand_valid=False):
    """Run the hand head for one clip and gather its metric-scale correspondences.

    Returns ``(pj_cam, c2w, s_clip, ratios)``: ``pj_cam`` [S,H,J,3] metric camera-frame joints
    (m, CPU), ``c2w`` [S,4,4] up-to-scale cam->world poses (CPU), the per-clip scene scale
    ``s_clip`` = ``median(ratios)`` (== the closed-form ``solve_metric_scale``), and ``ratios``
    = the raw per-joint correspondences ``z_hand / scene_depth`` (1-D, CPU). Returning the raw
    ratios lets the caller pool them across the whole sequence for ONE sequence-level scale,
    instead of medianing per-clip scalars (a per-frame heuristic). World placement is deferred to
    ``_world_from_cam`` so a clip can be re-placed under any scale without re-running the network.

    Computes the ratios with the same projection/sampling core as ``solve_metric_scale`` (depth
    sampled at the projected joints), but inline here so the eval depends only on the low-level
    helpers already present on the cluster (the launcher symlinks the cluster ``diffsynth``).
    """
    from scripts.train_hand_head import compute_joints_from_batch
    from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
        project_joints_to_norm_pixels, sample_depth_at_joints)

    pred_joints = compute_joints_from_batch(preds["hand_joints"], mano_model, device)  # [1,S,H,J,3] cam (m)
    # cam->world (clip-local, up-to-scale). Only the world-space eval uses it; the
    # camera-frame anchor eval ignores c2w, and with GS off the model never renders
    # rendered_extrinsics — fall back to identity so cam-frame eval still runs.
    _c2w_raw = preds.get("rendered_extrinsics")
    if _c2w_raw is not None:
        c2w = _c2w_raw[0].float()                           # [S,4,4]
    else:
        _S = pred_joints.shape[1]
        c2w = torch.eye(4, device=pred_joints.device).unsqueeze(0).repeat(_S, 1, 1)
        # SHOUT. Identity poses mean the world lift has NO camera motion at all, so every
        # world/W metric degenerates to hand-joint chaining and every scene-scale variant
        # collapses to the same number. This used to pass silently and invalidated a whole
        # round of scale/linking experiments before the diag_cam probe caught it.
        if not getattr(predict_clip, "_warned_identity_c2w", False):
            predict_clip._warned_identity_c2w = True
            print("  !! NO rendered_extrinsics IN preds - falling back to IDENTITY camera poses. "
                  "World/W metrics from this run are NOT valid (no camera motion). Ensure the "
                  "model publishes camera_poses (see worldmirror gs_anchor_only fast path).",
                  flush=True)
    gs_depth = preds.get("gs_depth")
    # Dump support: a nearest-subsampled 32x32 per-frame scene depth (fp16). Nearest (not avg)
    # keeps real depth samples so offline scene-point backprojection has no flying-pixel blend.
    if depth_out is not None:
        d32 = None
        if gs_depth is not None:
            d = gs_depth[0].float()
            while d.dim() > 3:
                d = d.squeeze(1)
            d32 = torch.nn.functional.interpolate(
                d.unsqueeze(1), size=(32, 32), mode="nearest-exact").squeeze(1).half().cpu()
        depth_out.append(d32)

    ratios = torch.empty(0)
    s = 1.0
    s_failed = False
    if gs_depth is not None and cam_intr is not None:
        grid_xy, z = project_joints_to_norm_pixels(pred_joints, cam_intr.to(device))
        sampled, in_frame = sample_depth_at_joints(
            gs_depth, grid_xy, window=scene_depth_window, reduce=scene_depth_reduce)
        # MEASURED FIRST, gated second (task #63): the diagnostic is always recorded so the
        # prevalence of behind-camera joints is a number rather than a story, while the mask
        # change stays behind require_positive_z until the A/B says it helps.
        valid_unguarded = ratio_validity_mask(z, sampled, in_frame, require_positive_z=False)
        _hp = hand_valid if gate_scale_on_hand_valid else None
        valid = ratio_validity_mask(z, sampled, in_frame, require_positive_z=require_positive_z,
                                    hand_present=_hp)
        _nonpos = (~torch.isfinite(z)) | (z <= 0)
        _n_tot = int(valid_unguarded.numel())
        _frac_nonpos_valid = (float((valid_unguarded & _nonpos).sum())
                              / max(int(valid_unguarded.sum()), 1))
        # Always measured, gated separately (task #70). How much of the accepted population comes
        # from a hand slot the detector never filled is the number that decides whether the gate
        # is worth adopting, and it must be recorded under BOTH settings to stay comparable.
        _frac_absent_hand = None
        if hand_valid is not None:
            _absent = ~hand_valid.unsqueeze(-1).to(valid.device).bool()
            _frac_absent_hand = (float((valid_unguarded & _absent).sum())
                                 / max(int(valid_unguarded.sum()), 1))
        if steps_out is not None:
            steps_out["frac_nonpos_z_of_valid"] = _frac_nonpos_valid
            steps_out["n_joints_total"] = _n_tot
            steps_out["frac_absent_hand_of_valid"] = _frac_absent_hand
            steps_out["gate_scale_on_hand_valid"] = bool(gate_scale_on_hand_valid)
        if bool(valid.any()):
            ratios = (z / sampled)[valid].detach().float().cpu()    # [n_valid] z_hand/scene_depth
            _s_raw = float(ratios.median())
            s = float(min(max(_s_raw, 0.1), 10.0))
            # A solved scale sitting exactly on a clamp bound is not an estimate, it is the
            # bound. The true camera-centre scale on this data is ~1.0, so 0.1 or 10.0 means the
            # solve FAILED for this clip. Measured over 1884 segments: 15.7% land on the floor,
            # in the median case with EVERY clip of the segment at the bound and zero variance.
            # Record it so the caller can decide, instead of returning the bound as a number.
            s_failed = (_s_raw < 0.1) or (_s_raw > 10.0)
        # Registration-panel intermediates (supervisor request 2026-08-06: "visual intermediate
        # results for the registration steps"). Emitted from INSIDE the eval's own solve rather
        # than recomputed by the plotting script: an earlier standalone dumper re-implemented the
        # load-and-forward path, silently diverged from this one (no bf16 autocast, no cond_flags),
        # and produced a constant hand depth of -0.0205 m on every sequence. The figure has to show
        # what the eval does, so it is fed by the eval.
        if steps_out is not None:
            _d = gs_depth[0].float()
            if _d.dim() == 4:
                _d = _d[:, 0] if _d.shape[1] == 1 else _d[..., 0]
            steps_out.update({
                "gs_depth": _d.half().cpu(),                  # [S,Hd,Wd] R1: the scene depth map
                "grid_xy": grid_xy[0].float().cpu(),          # [S,H,J,2] R1: where joints project
                "hand_z": z[0].float().cpu(),                 # [S,H,J] R2: metric hand depth
                "scene_at_hand": sampled[0].float().cpu(),    # [S,H,J] R2: scene depth there
                "valid": valid[0].cpu(),
                "ratios": ratios,                             # R2: the population medianed
                "s": s, "s_raw": _s_raw, "s_failed": s_failed,
                "c2w": c2w.cpu(),                             # [S,4,4] R3: up-to-scale trajectory
            })

    # Contact Phase 1: post-hoc root-depth correction. Applied AFTER the scene-scale
    # solve (so the scene scale stays an independent property, no anchor->scale
    # feedback — the design's circularity guard) but before world placement, using
    # the same apply_root_anchor as training. Behind the enable flag; needs gs_depth.
    # C1: when ref_d_scene (DA3 metric wrist depth) is given it REPLACES gs_depth as the
    # anchor target, so the anchor applies even with GS off (gs_depth None).
    if (model is not None and getattr(model, "enable_root_anchor", False)
            and cam_intr is not None and (gs_depth is not None or ref_d_scene is not None)):
        from scripts.root_depth_anchor import apply_root_anchor
        pred_joints, _dz, _info = apply_root_anchor(
            model.root_depth_refine, pred_joints, gs_depth,
            preds.get("gs_depth_conf"), cam_intr.to(device),
            contact_mask=contact_mask,
            ref_d_scene=(ref_d_scene.to(device) if ref_d_scene is not None else None),
        )
        # Diagnostic: did the anchor fire (gate) and how big a correction (|dz|)?
        # A near-zero gate-rate means the scene-depth reference was never trusted
        # (HOT3D frozen gs_depth too weak / out of band); near-zero |dz| on a firing
        # gate means the module never learned. Either explains an inert anchor.
        if anchor_log is not None:
            gate = _info["gate"]
            gated = gate.float()
            n_gate = float(gated.sum())
            dz_gated = float((_dz.abs() * gated).sum() / n_gate) if n_gate > 0 else 0.0
            # Disagreement = |d_scene - wrist_z| where gated. This is the headroom Δz
            # converges toward under the consistency loss. If disagree >> |dz|, the
            # anchor is UNDERTRAINED (room to correct, hasn't learned). If disagree
            # ~= |dz| ~= 0, there is NOTHING to fix (head wrist depth already ~ gs_depth).
            disagree = (_info["d_scene"] - _info["wrist_z"]).abs()
            disagree_gated = float((disagree * gated).sum() / n_gate) if n_gate > 0 else 0.0
            anchor_log.append({
                "gate_rate": float(gated.mean()),
                "dz_gated_m": dz_gated,
                "dz_max_m": float(_dz.abs().max()),
                "disagree_gated_m": disagree_gated,
            })
    return pred_joints[0].float().cpu(), c2w.cpu(), s, ratios, s_failed


def _dense_scene_points(preds, cam_intr, pj_cam, s_clip, grid=24, hand_radius_m=0.15):
    """G1 dense-link: unproject the clip's gs_depth into per-frame static-scene points (scene
    units, camera frame), masking out hand pixels by 3D proximity to the predicted hand joints.

    Inverts the EXACT projection ``project_joints_to_norm_pixels`` uses to sample this depth map
    (hand_depth_sampling.py: col=f*x/z+cx, row=f*y/z+cy, u=(W-1)-row, v=col, normalized by
    W = 2*cx derived from the intrinsics) so the unprojection is self-consistent with the
    scale/anchor pipeline. W was hardcoded to 1408 here and in the forward projection until
    2026-08-06; on the 224-px HOI4D/H2O stores that sampled the frame corner instead of the
    hand. Both sides now derive W the same way. Hand masking
    is done by 3D distance (scene points scaled to metric via the clip scale) rather than 2D
    boxes - rotation-convention-free and it removes the held object's contact region too.
    Returns ``(pts [S,P,3] scene-units cam-frame CPU, valid [S,P] bool CPU)`` or ``None``.
    """
    # Function-local by design, matching predict_clip. A module-level import would make the whole
    # eval depend on the full `diffsynth` package at import time, and the launcher only symlinks
    # the cluster diffsynth. This name was previously referenced here with NO import in scope at
    # all, so the --dense_link path raised NameError the moment it ran (pyflakes, 2026-08-06).
    from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
        frame_width_from_intr)

    gs_depth = preds.get("gs_depth")
    if gs_depth is None or cam_intr is None:
        return None
    # Must match the forward projection exactly: derived from the intrinsics, not hardcoded.
    W_NORM = float(frame_width_from_intr(cam_intr.view(1, 3))[0])
    # gs_depth is [B,S,1,Hd,Wd] (channel-first) or [B,S,Hd,Wd,1] (channel-last), per
    # hand_depth_sampling.py. Drop the batch, then the singleton channel from EITHER position ->
    # [S,Hd,Wd]. (A `while d.dim()>3: d=d.squeeze(1)` spins forever on the channel-last layout,
    # since squeeze(1) is a no-op when dim 1 is not size 1 - that was the clip-1 hang.)
    d = gs_depth[0].float()                          # [S,1,Hd,Wd] or [S,Hd,Wd,1] or [S,Hd,Wd]
    if d.dim() == 4 and d.shape[1] == 1:
        d = d[:, 0]                                  # channel-first -> [S,Hd,Wd]
    elif d.dim() == 4 and d.shape[-1] == 1:
        d = d[..., 0]                                # channel-last  -> [S,Hd,Wd]
    if d.dim() != 3:
        raise ValueError(f"_dense_scene_points: expected gs_depth[0] -> [S,Hd,Wd], got {tuple(d.shape)}")
    dg = torch.nn.functional.interpolate(
        d.unsqueeze(1), size=(grid, grid), mode="nearest-exact").squeeze(1)   # [S,G,G]
    S = dg.shape[0]
    dev = dg.device
    f, cx, cy = [float(x) for x in cam_intr.view(-1)[:3]]
    iy, ix = torch.meshgrid(torch.arange(grid, device=dev), torch.arange(grid, device=dev),
                            indexing="ij")
    u01 = (ix.float() + 0.5) / grid                  # depth-map x axis = "u"
    v01 = (iy.float() + 0.5) / grid                  # depth-map y axis = "v"
    row = (W_NORM - 1.0) - u01 * W_NORM              # invert u=(W-1)-row
    col = v01 * W_NORM                               # invert v=col
    z = dg.reshape(S, -1)                            # [S,P] scene-unit depth
    x = (col.reshape(1, -1) - cx) * z / f
    y = (row.reshape(1, -1) - cy) * z / f
    pts = torch.stack([x, y, z], dim=-1)             # [S,P,3] scene units, cam frame
    valid = torch.isfinite(z) & (z > 0.05)
    # 3D hand mask: scene point (scaled to metric) within hand_radius_m of ANY predicted joint.
    j = torch.nan_to_num(pj_cam.reshape(S, -1, 3), nan=1e6).to(dev)          # [S,HJ,3] metric
    dist = torch.cdist(pts * float(s_clip), j)                               # [S,P,HJ]
    valid &= dist.min(dim=-1).values > hand_radius_m
    return pts.cpu(), valid.cpu()


def _intr_3x3(cam_intr, res, device):
    """Build a [3,3] pinhole K at the model's square `res` from cached (f, cx, cy). The cache is at
    the original pinhole resolution; principal point cx~W/2 lets us rescale to `res`."""
    f, cx, cy = [float(x) for x in cam_intr.view(-1)[:3]]
    fx = f * res / (2.0 * cx)
    fy = f * res / (2.0 * cy)
    k = torch.tensor([[fx, 0.0, res / 2.0], [0.0, fy, res / 2.0], [0.0, 0.0, 1.0]], device=device)
    return k


def eval_sequence(model, mano_model, device, seq_dir, cfg, segment_len, clip_len, stride, wa_short,
                  max_segs=0, feed_intrinsics=False, smooth_windows=None, dump_list=None,
                  refine_pose=False, refine_iters=40, refine_lr=3e-3, refine_frame_stride=1,
                  refine_sanity=False, robust_scale=False, scale_fallback="clamp",
                  da3_wrist_cache_dir=None, contact_cache_dir=None, contact_gate="off",
                  oracle_depth=False, dense_link=False,
                  gravity_oracle=False, gravity_axis=(0.0, 1.0, 0.0), dump_cam_dir=None,
                  diag_cam=False, require_positive_z=False,
                  scene_depth_window=1, scene_depth_reduce="bilinear",
                  gate_scale_on_hand_valid=False):
    """Eval all `segment_len` segments of one sequence; return list of per-segment metrics.

    ``feed_intrinsics``: condition the backbone on the *known* camera intrinsics (ray prior,
    ``cond_flags=[0,0,1]``) instead of the default identity dummy, to probe whether feeding true
    intrinsics improves the predicted per-clip geometry.

    ``smooth_windows``: optional list of odd window sizes. For each, temporally smooth the chained
    (per-seq-pooled) root track and record ``W_MPJPE_sm{w}`` / ``WA_MPJPE_long_sm{w}`` — the
    drift-vs-bias diagnostic. ``dump_list``: if given, append the raw per-segment pooled trajectory
    (pred/GT world joints + valid mask) so the smoothing sweep can be iterated offline on CPU.
    """
    from scripts.train_hand_head import HOT3DHandDataset, build_views

    mcfg = cfg["model"]
    ds = HOT3DHandDataset([seq_dir], mano_model, num_frames=clip_len, clip_stride=stride,
                          use_hand_crop=mcfg.get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5))
    if len(ds) == 0:
        return []
    hd = os.path.join(seq_dir, "hand_data")
    gt_world = torch.load(os.path.join(hd, "gt_joints_cache_world.pt"), map_location="cpu").float()  # [N,2,16,3]
    gt_cam = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), map_location="cpu").float()    # [N,2,16,3]
    cam_intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)
    bb = torch.load(os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt"), map_location="cpu")
    gt_valid = bb["valid"].bool()                           # [N,2]
    # C1 anchor references, keyed by seq basename, from /home (scratch is write-locked).
    _sq = os.path.basename(seq_dir.rstrip("/"))
    seq_da3 = None
    if da3_wrist_cache_dir is not None:
        _p = os.path.join(da3_wrist_cache_dir, f"{_sq}_da3_wrist.pt")
        if os.path.exists(_p):
            seq_da3 = torch.load(_p, map_location="cpu", weights_only=True).float()   # [N,2] m
    seq_contact = None
    if contact_gate == "oracle" and contact_cache_dir is not None:
        _p = os.path.join(contact_cache_dir, f"{_sq}_contact.pt")
        if os.path.exists(_p):
            seq_contact = torch.load(_p, map_location="cpu", weights_only=True).bool()  # [N,2]

    overlap = clip_len - stride
    clips_per_seg = segment_len // stride
    out = []
    # --dump_cam_preds: accumulate our per-frame cam-space hands ([N,2,16,3]) to compose with a
    # SLAM trajectory downstream ("ours + SLAM" lever 2). Same predict path as the world eval.
    cam_buf = torch.full_like(gt_cam, float("nan")) if dump_cam_dir else None
    val_buf = torch.zeros(gt_cam.shape[:2], dtype=torch.bool) if dump_cam_dir else None
    # ...and the chained WORLD track for the same frames, so the dump satisfies the full
    # eval_worldspace_baseline contract {cam_joints, world_joints, valid}. That lets our own
    # (online, self-chained) row be scored by the SAME scorer as the +SLAM / HaWoR rows instead
    # of by eval_world_space's own segmenter - the two enumerate segments differently (this file
    # drops the partial tail, the baseline scorer keeps it), which would otherwise make the rows
    # non-comparable. Frames outside a scored segment stay NaN and the scorer masks them out.
    world_buf = torch.full_like(gt_cam, float("nan")) if dump_cam_dir else None
    n_seg = len(ds) // clips_per_seg
    if max_segs > 0:
        n_seg = min(n_seg, max_segs)
    for seg in range(n_seg):
        base = seg * clips_per_seg
        clip_cams = []   # [(pj_cam [S,H,J,3], c2w [S,4,4], s_perclip), ...]
        clip_oracle = [] if oracle_depth else None  # per-clip GT-depth-anchored pj_cam (ceiling diag)
        clip_dense = [] if dense_link else None     # per-clip (scene pts, valid) for the G1 dense chain
        clip_grav = [] if gravity_oracle else None  # per-clip GT c2w for the gravity-view oracle
        clip_depths = [] if dump_list is not None else None  # per-clip 32x32 gs_depth (dump only)
        s_pairs = []     # (s_gt, s_hand) per clip: GT camera-trajectory scale vs our hand scale (b)
        anchor_log = []  # per-clip {gate_rate, dz_gated_m, dz_max_m} when the root anchor is active
        nonpos_log = []  # per-clip fraction of accepted correspondences that are behind-camera (#63)
        s_gt_by_clip = {}  # clip index -> GT camera-trajectory scale, for the GT-scale oracle (#38)
        for c in range(clips_per_seg):
            print(f"  [clip seg{seg} {c + 1}/{clips_per_seg}] fwd+lift", flush=True)
            batch = ds[base + c]
            imgs = batch["img"].unsqueeze(0).to(device)
            hb = batch["hand_bboxes"].unsqueeze(0).to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].unsqueeze(0).to(device) if "hand_valid" in batch else None
            views = build_views(imgs, clip_len, device, hb, hv)
            cond_flags = [0, 0, 0]
            if feed_intrinsics:
                res = imgs.shape[-1]
                k = _intr_3x3(cam_intr, res, device)
                views["camera_intrs"] = k.view(1, 1, 3, 3).expand(1, clip_len, 3, 3).contiguous()
                cond_flags = [0, 0, 1]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(views, cond_flags=cond_flags, is_inference=True, use_motion=False)
            # Per-clip camera-pose refinement: sharpen each frame's pose against the clip's own
            # static Gaussian map (the per-clip-pose bottleneck the oracle-cam diagnostic exposed).
            # Runs in fp32 outside the autocast block; overwrites the c2w predict_clip lifts with.
            if refine_pose and "splats" in preds and "rendered_extrinsics" in preds:
                from scripts.pose_refine import refine_clip_poses
                ref_ext, rinfo = refine_clip_poses(
                    model.gs_renderer.rasterizer, preds["splats"][0],
                    preds["rendered_extrinsics"][0].float(), preds["rendered_intrinsics"][0].float(),
                    imgs[0].float(), hand_bboxes=(hb[0].float() if hb is not None else None),
                    iters=refine_iters, lr=refine_lr, frame_stride=refine_frame_stride,
                    sanity=refine_sanity)
                if not refine_sanity:
                    preds["rendered_extrinsics"] = ref_ext.unsqueeze(0).to(
                        preds["rendered_extrinsics"].dtype)
                print(f"  [refine seg{seg} c{c + 1}] conv={rinfo.get('conv')} nG={rinfo.get('n_gauss')} "
                      f"psnr ff={rinfo.get('psnr_ff_mean', float('nan')):.1f}"
                      f"->ref={rinfo.get('psnr_ref_mean', float('nan')):.1f} "
                      f"improved={rinfo.get('improved')}/{rinfo.get('n_frames')}", flush=True)
            _cf = (base + c) * stride   # this clip's frame offset (dataset clip_stride == stride)
            _ref = None
            if seq_da3 is not None:
                _sl = seq_da3[_cf:_cf + clip_len]
                if _sl.shape[0] == clip_len:
                    _ref = _sl.unsqueeze(0).to(device)                  # [1,S,2] DA3 metric wrist depth
            _con = None
            if seq_contact is not None:
                _sl = seq_contact[_cf:_cf + clip_len]
                if _sl.shape[0] == clip_len:
                    _con = _sl.unsqueeze(0).to(device)                  # [1,S,2] contact gate
            _steps: dict = {}
            cc = predict_clip(preds, mano_model, device, cam_intr, model=model, anchor_log=anchor_log,
                              ref_d_scene=_ref, contact_mask=_con, depth_out=clip_depths,
                              steps_out=_steps, require_positive_z=require_positive_z,
                              scene_depth_window=scene_depth_window,
                              scene_depth_reduce=scene_depth_reduce,
                              hand_valid=hv, gate_scale_on_hand_valid=gate_scale_on_hand_valid)
            clip_cams.append(cc)
            # Task #63 diagnostic: what fraction of the correspondences the UNGUARDED mask
            # accepts are behind-camera joints. Recorded whether or not the guard is on, so the
            # two arms of the A/B are directly comparable.
            if "frac_nonpos_z_of_valid" in _steps:
                nonpos_log.append(_steps["frac_nonpos_z_of_valid"])

            if diag_cam:
                # The scene scale multiplies ONLY the camera translation, so its effect on the
                # world trajectory is proportional to how far the camera actually moves within a
                # clip. Every scale variant (per-clip / median / pooled) came out BIT-IDENTICAL on
                # all 314 real segments despite a per-clip scale std of 0.28, which is exactly what
                # a degenerate (near-zero) predicted camera translation would produce. Print the
                # predicted excursion next to the GT one to settle it.
                _pc = cc[1][:, :3, 3]                                   # predicted centres [S,3]
                _pe = float((_pc - _pc.mean(0)).norm(dim=-1).max())
                _ge = float("nan")
                if "cam_extrinsics" in batch:
                    _gc = torch.inverse(batch["cam_extrinsics"].float())[:, :3, 3]
                    _ge = float((_gc - _gc.mean(0)).norm(dim=-1).max())
                print(f"  [diagcam seg{seg} c{c + 1}] pred_cam_excursion={_pe:.6f} m | "
                      f"gt_cam_excursion={_ge:.6f} m | s_clip={float(cc[2]):.4f} | "
                      f"has_extrinsics={'cam_extrinsics' in batch}", flush=True)

            if dump_cam_dir is not None:
                pj = cc[0].detach().cpu().float()                      # [S,H,16,3] cam-frame joints
                e = min(_cf + pj.shape[0], cam_buf.shape[0])
                cam_buf[_cf:e] = pj[:e - _cf]
                val_buf[_cf:e] = torch.isfinite(pj[:e - _cf]).all(dim=-1).all(dim=-1)

            # G1 dense-link: unproject this clip's gs_depth into hand-masked static-scene points.
            if dense_link:
                clip_dense.append(_dense_scene_points(preds, cam_intr, cc[0], cc[2]))

            # ORACLE DEPTH CEILING: replace each frame's predicted wrist DEPTH with the GT
            # camera-frame wrist depth by scaling the whole hand along the camera ray (cam-frame
            # rays pass through the origin, so a uniform scale by gt_z/pred_z moves the wrist to
            # gt_z while leaving the 2D projection unchanged). This is the "perfect per-frame
            # metric depth anchor" ceiling - what an ideal DA3/MonST3R depth model would buy -
            # isolating the hand-depth term of W from camera-trajectory/scale drift. Where GT is
            # missing or a depth is non-positive, keep the prediction (ratio 1).
            if oracle_depth:
                pjo = cc[0].clone()                               # [S,H,J,3] cam-frame (CPU)
                gcl = gt_cam[_cf:_cf + clip_len]                  # [S,H,J,3] GT cam-frame
                if gcl.shape[0] == pjo.shape[0]:
                    pz = pjo[:, :, 0, 2]                          # [S,H] pred wrist z (MANO joint 0)
                    gz = gcl[:, :, 0, 2]                          # [S,H] gt wrist z
                    ok = (pz > 0.05) & torch.isfinite(gz) & (gz > 0.05)
                    ratio = torch.where(ok, gz / pz.clamp(min=0.05), torch.ones_like(pz))
                    pjo = pjo * ratio[:, :, None, None]
                clip_oracle.append(pjo)

            # (b) GROUND-TRUTH SCALE check: the world placement scales the up-to-scale camera
            # translation by our hand-derived s. The *true* metric scale is the similarity that maps
            # the predicted (up-to-scale) camera centers onto the GT metric camera centers. Comparing
            # the two says whether our hand scale is right in absolute terms (ratio==1) or biased/noisy.
            if "cam_extrinsics" in batch:
                c2w_gt = torch.inverse(batch["cam_extrinsics"].float())     # [S,4,4] metric cam->world
                if gravity_oracle:
                    clip_grav.append(c2w_gt)                                # aligned with clip_cams order
                pc = cc[1][:, :3, 3]                                        # pred centers (up-to-scale)
                gc = c2w_gt[:, :3, 3]                                       # gt centers (metric)
                if pc.shape[0] >= 3 and float((pc - pc.mean(0)).norm(dim=-1).max()) > 1e-4:
                    s_gt, _, _ = solve_similarity(pc, gc)
                    if torch.isfinite(s_gt) and float(s_gt) > 1e-6:
                        s_pairs.append((float(s_gt), float(cc[2])))
                        # Aligned with clip_cams by index, for the GT-SCALE oracle below.
                        s_gt_by_clip[len(clip_cams) - 1] = float(s_gt)

        seg_start = base * stride
        t_avail = min(segment_len, gt_world.shape[0] - seg_start)
        if t_avail < wa_short:
            continue
        gtw = gt_world[seg_start:seg_start + t_avail].reshape(t_avail, -1, 3)            # world
        val = gt_valid[seg_start:seg_start + t_avail].repeat_interleave(16, dim=1)       # [t,32]

        def _metrics(pred):
            t = min(pred.shape[0], t_avail)
            p, g, v = pred[:t], gtw[:t], val[:t]
            # W-MPJPE under the shared first-window RIGID gauge (rotation+translation, NO scale):
            # matches Hand3R's "first-window align, no scale" and avoids the Sim3 scale blow-up.
            # Predictions are already metric (in-scene hand anchor), so scale is never re-solved.
            # Helper lives in world_space_metrics so the offline smoothing sweep scores identically.
            w = w_mpjpe_first_window_aligned(p, g, v, wa_short)
            return {"W_MPJPE": w,
                    "WA_MPJPE_short": wa_mpjpe(p, g, window=wa_short, valid=v),
                    "WA_MPJPE_long": wa_mpjpe(p, g, window=t, valid=v), "frames": t}

        def _greedy(cw):
            return _metrics(chain_trajectories_by_overlap(cw, overlap=overlap))

        def _global(cw):
            return _metrics(chain_trajectories_global(cw, overlap=overlap, iters=8, robust=True))

        def _linked(cw, centers):
            # ICLR chunk linker: global BA + camera-centre correspondences + cross-fade seam fusion.
            return _metrics(chain_trajectories_linked(cw, overlap=overlap, clip_centers=centers,
                                                      iters=8, robust=True))

        # Three scene scales, each applied ONLY to the camera translation (the hand is already
        # metric). per-clip = the per-frame heuristic (one closed-form solve per clip; high
        # variance -> the chained absolute trajectory drifts -> large W-MPJPE). per-seq MEDIAN =
        # robust median of the per-clip scalars. per-seq POOLED = ONE median over EVERY valid
        # (z_hand / scene_depth) correspondence in the whole sequence -> the principled
        # sequence-level solve (every joint votes once, robust to clips with few valid joints).
        scales = torch.tensor([s for (_, _, s, _, _) in clip_cams], dtype=torch.float32)
        s_med = float(scales.median())
        s_std = float(scales.std()) if scales.numel() > 1 else 0.0
        pooled = [r for (_, _, _, r, _) in clip_cams if r.numel()]
        all_ratios = torch.cat(pooled) if pooled else torch.empty(0)
        if robust_scale and all_ratios.numel():
            # MAD outlier rejection before the median: the W-decomposition flagged the pooled
            # hand scale as biased (+14-27% vs the true cam-center scale) AND heavy-tailed, and the
            # world-lift multiplies camera translation by it. Tukey/MAD reject (k=3) trims the
            # heavy z_hand/scene_depth tails so a few bad depth samples stop dragging the median.
            med = all_ratios.median()
            mad = (all_ratios - med).abs().median().clamp_min(1e-6)
            keep = (all_ratios - med).abs() <= 3.0 * 1.4826 * mad
            rr = all_ratios[keep] if bool(keep.any()) else all_ratios
            s_pool = float(rr.median().clamp(0.1, 10.0))
        else:
            s_pool = float(all_ratios.median().clamp(0.1, 10.0)) if all_ratios.numel() else s_med

        if not all_ratios.numel():
            # No (z_hand / scene_depth) correspondences at all -> predict_clip fell back to s=1.0
            # for every clip, so the up-to-scale camera translation is never converted to metric
            # and every world/W metric is inflated. Almost always a config error (enable_gs=False
            # => no gs_depth). Shout: this used to pass silently as a plausible-looking W.
            print("  !! SCALE DEGENERATE (s=1.0, no gs_depth correspondences) - world/W metrics "
                  "are NOT metric. Set model.enable_gs=True in the eval config.", flush=True)

        # --- SCALE-SOLVE FALLBACK (task #63) ---------------------------------------------
        # 15.7% of segments have a per-clip solve that lands exactly on a clamp bound, which is
        # the bound rather than an estimate, and because the reported aggregate is a MEAN those
        # segments drag every world number. `pooled` substitutes the sequence-pooled scale for
        # the failed clips only; `clamp` keeps the historical behaviour. The rate is reported
        # either way so the failure can never be invisible again.
        n_failed = sum(1 for (_, _, _, _, f) in clip_cams if f)
        frac_failed = n_failed / max(len(clip_cams), 1)
        if scale_fallback == "pooled" and n_failed:
            clip_cams = [(pj, c2w, (s_pool if f else s), r, f)
                         for (pj, c2w, s, r, f) in clip_cams]
        frac_nonpos = (sum(nonpos_log) / len(nonpos_log)) if nonpos_log else 0.0
        if n_failed or frac_nonpos > 0:
            print(f"  [scale] {n_failed}/{len(clip_cams)} clips FAILED the solve "
                  f"({100*frac_failed:.1f}%), policy={scale_fallback}"
                  + (f" -> substituted s_pool={s_pool:.4f}" if scale_fallback == "pooled" else
                     " -> kept at the clamp bound")
                  + f" | behind-camera joints {100*frac_nonpos:.1f}% of accepted"
                  + (" (EXCLUDED)" if require_positive_z else " (INCLUDED)"), flush=True)

        # GT-SCALE ORACLE (task #38). s multiplies ONLY the camera translation, so replacing our
        # solved s with the GT trajectory scale while keeping our predicted rotation AND
        # translation DIRECTION isolates how much of the world error is scale MAGNITUDE as
        # opposed to direction. Without this the translation term (-47.8mm on lw60) cannot be
        # attributed, and 'fix the scale' stays a story rather than a ranked lever.
        worlds_gts = None
        if s_gt_by_clip:
            worlds_gts = [_world_from_cam(pj, c2w, s_gt_by_clip.get(i, s))
                          for i, (pj, c2w, s, _, _) in enumerate(clip_cams)]
        worlds_pc = [_world_from_cam(pj, c2w, s) for (pj, c2w, s, _, _) in clip_cams]       # per-clip
        worlds_md = [_world_from_cam(pj, c2w, s_med) for (pj, c2w, _, _, _) in clip_cams]   # per-seq median
        worlds_pl = [_world_from_cam(pj, c2w, s_pool) for (pj, c2w, _, _, _) in clip_cams]  # per-seq pooled
        g_pc = _greedy(worlds_pc)
        gl_pc = _global(worlds_pc)
        g_md = _greedy(worlds_md)
        g_gts = _greedy(worlds_gts) if worlds_gts is not None else None
        chain_pl = chain_trajectories_by_overlap(worlds_pl, overlap=overlap)  # pooled greedy traj
        g_pl = _metrics(chain_pl)
        if world_buf is not None:
            # chain_pl is [t, 2*16, 3] in the segment's world frame; write it back at absolute
            # frame indices so the dump lines up with cam_buf (both start at seg_start).
            tt = min(chain_pl.shape[0], t_avail, world_buf.shape[0] - seg_start)
            if tt > 0:
                world_buf[seg_start:seg_start + tt] = chain_pl[:tt].reshape(tt, 2, 16, 3).cpu()
        # ICLR chunk linker on the pooled-scale worlds. The camera centre for clip k under the pooled
        # scale is c2w[k, :3, 3] * s_pool (exactly the translation _world_from_cam applies), so joints
        # and centres share the clip-local world frame. Reported as the "_link" suffix -> a direct
        # greedy(_spool)-vs-linker(_link) comparison at the same scene scale.
        centers_pl = [c2w[:, :3, 3] * s_pool for (_, c2w, _, _, _) in clip_cams]
        gl_link = _linked(worlds_pl, centers_pl)      # linker WITH camera centres (needs reliable cam poses)
        gl_linkcf = _linked(worlds_pl, None)          # cross-fade only, NO centres (the safe variant)
        # RIGID linker variants (per_clip_scale=False): after the pooled scene scale, per-clip
        # Umeyama SCALE is solved on a degenerate ~10 cm joint cluster and mostly re-injects the
        # clip-to-clip scale jitter that pooling removed. Freezing s=1 isolates that effect.
        gl_linkr = _metrics(chain_trajectories_linked(
            worlds_pl, overlap=overlap, clip_centers=centers_pl, iters=8, robust=True,
            per_clip_scale=False))
        gl_linkrcf = _metrics(chain_trajectories_linked(
            worlds_pl, overlap=overlap, clip_centers=None, iters=8, robust=True,
            per_clip_scale=False))

        # Smoothing diagnostic: re-place the chained pooled root track through a temporal low-pass
        # (per-frame articulation held fixed) and re-score. W drops => high-freq drift a temporal
        # head will absorb; W flat => low-freq per-frame depth bias a head alone will not fix.
        sm_rows = {}
        if smooth_windows:
            for w in smooth_windows:
                sm = smooth_root_trajectory(chain_pl, window=int(w))
                m = _metrics(sm)
                sm_rows[f"W_MPJPE_sm{w}"] = m["W_MPJPE"]
                sm_rows[f"WA_MPJPE_long_sm{w}"] = m["WA_MPJPE_long"]

        # Velocity/trajectory-head CEILINGS (always on; cheap post-process on the chained pred).
        # W_velGT: GT root motion + pred articulation (perfect inter-frame motion) -> upper bound a
        # trajectory head could reach. W_re16/W_re32: rigid re-anchor to GT every 16/32 frames ->
        # bound for periodic absolute correction of our own relative motion.
        ttc = min(chain_pl.shape[0], t_avail)
        pc, gc, vc2 = chain_pl[:ttc], gtw[:ttc], val[:ttc]
        sm_rows["W_MPJPE_velGT"] = w_mpjpe_first_window_aligned(
            replace_root_with_gt_motion(pc, gc, vc2), gc, vc2, wa_short)
        sm_rows["W_MPJPE_re16"] = w_mpjpe(reanchor_to_gt(pc, gc, vc2, 16), gc, vc2)
        sm_rows["W_MPJPE_re32"] = w_mpjpe(reanchor_to_gt(pc, gc, vc2, 32), gc, vc2)

        # ORACLE DEPTH ceiling: lift the GT-wrist-depth-anchored per-clip hands under the SAME
        # pooled scene scale + chaining as the real trajectory, and score. W_depthOracle << W_spool
        # => per-frame hand absolute depth is the dominant W lever (a dense metric depth anchor
        # like DA3 has real headroom); W_depthOracle ~= W_spool => the bottleneck is the camera
        # trajectory / scene scale, not hand depth (depth anchoring won't move W).
        if oracle_depth and clip_oracle:
            worlds_or = [_world_from_cam(po, c2w, s_pool)
                         for po, (_, c2w, _, _, _) in zip(clip_oracle, clip_cams)]
            m_or = _metrics(chain_trajectories_by_overlap(worlds_or, overlap=overlap))
            sm_rows["W_MPJPE_depthOracle"] = m_or["W_MPJPE"]
            sm_rows["WA_MPJPE_short_depthOracle"] = m_or["WA_MPJPE_short"]

        # GVHMR gravity-view ORACLE: snap predicted camera tilt/roll to true gravity (keep predicted
        # yaw + translation) and re-chain -> gravOracle. fix_yaw=True replaces the full rotation with
        # GT -> rotOracle (camera-rotation ceiling). If gravOracle collapses toward re16, the long-
        # window W drift is tilt/roll (a gravity-view head would fix it); if only rotOracle collapses,
        # the residual is yaw; if neither, it is translation/scale. Uses GT extrinsics for gravity only.
        if gravity_oracle and clip_grav and len(clip_grav) == len(clip_cams):
            grav = torch.tensor(gravity_axis, dtype=torch.float32)
            worlds_gv = [_world_from_cam(pj, gravity_align_c2w(c2w, c2wg, grav, mode="tilt"), s_pool)
                         for (pj, c2w, *_g), c2wg in zip(clip_cams, clip_grav)]
            m_gv = _metrics(chain_trajectories_by_overlap(worlds_gv, overlap=overlap))
            sm_rows["W_MPJPE_gravOracle"] = m_gv["W_MPJPE"]
            worlds_ro = [_world_from_cam(pj, gravity_align_c2w(c2w, c2wg, grav, mode="rot"), s_pool)
                         for (pj, c2w, *_g), c2wg in zip(clip_cams, clip_grav)]
            m_ro = _metrics(chain_trajectories_by_overlap(worlds_ro, overlap=overlap))
            sm_rows["W_MPJPE_rotOracle"] = m_ro["W_MPJPE"]
            print(f"  [gravity seg{seg}] gravOracle={m_gv['W_MPJPE']:.1f} rotOracle={m_ro['W_MPJPE']:.1f}"
                  f" (global {gl_pc['W_MPJPE']:.1f}, re16 {sm_rows.get('W_MPJPE_re16', float('nan')):.1f})",
                  flush=True)

        # G1 DENSE CHAIN (MonST3R gate): re-chain the SAME pooled-scale clip worlds, but solve
        # every link from dense hand-masked static-scene correspondences (shared overlap frames,
        # same pixel grid) instead of the hand-joint cluster. W_dchainr << W_spool => dense scene
        # evidence fixes per-link rotation drift -> build the full windowed-graph optimization;
        # ~= W_spool => our per-clip dense geometry is drift-inconsistent too (not the lever).
        if dense_link and clip_dense and all(cd is not None for cd in clip_dense):
            dense_pl, dense_val = [], []
            for (dp, dv), (_, c2w, *_d) in zip(clip_dense, clip_cams):
                dense_pl.append(_world_from_cam((dp * s_pool).unsqueeze(1), c2w, s_pool))
                dense_val.append(dv)
            tr_r, diag_r = chain_trajectories_dense(worlds_pl, dense_pl, dense_val, overlap,
                                                    per_clip_scale=False, robust=True)
            m_r = _metrics(tr_r)
            tr_s, _ = chain_trajectories_dense(worlds_pl, dense_pl, dense_val, overlap,
                                               per_clip_scale=True, robust=True)
            m_s = _metrics(tr_s)
            sm_rows["W_MPJPE_dchainr"] = m_r["W_MPJPE"]
            sm_rows["W_MPJPE_dchain"] = m_s["W_MPJPE"]
            _res = sorted(x for x in diag_r["resid_mm"] if x == x)
            _nc = sorted(diag_r["n_corr"])
            print(f"  [dense seg{seg}] W dchainr={m_r['W_MPJPE']:.1f} dchain={m_s['W_MPJPE']:.1f} "
                  f"corr_med={_nc[len(_nc) // 2] if _nc else 0} "
                  f"resid_med={(_res[len(_res) // 2] if _res else float('nan')):.0f}mm "
                  f"fallback={diag_r['fallback']}", flush=True)
        if dump_list is not None:
            tt = min(chain_pl.shape[0], t_avail)
            dump_list.append({
                "seq": os.path.basename(seq_dir), "seg": seg, "wa_short": wa_short,
                "pred_world": chain_pl[:tt].cpu(), "gt_world": gtw[:tt].cpu(),
                "valid": val[:tt].cpu(),
                # Raw per-clip state so LINKER/scale variants iterate OFFLINE (no GPU rerun):
                # cam-frame joints + up-to-scale c2w re-derive clip worlds under ANY scale
                # (`_world_from_cam`); depth32+cam_intr backproject MonST3R-style scene points.
                "clip_pj_cam": [c[0].cpu() for c in clip_cams],
                "clip_c2w": [c[1].cpu() for c in clip_cams],
                "clip_scales": [float(c[2]) for c in clip_cams],
                "clip_ratios": [c[3].cpu() for c in clip_cams],
                "clip_depth32": clip_depths,
                "cam_intr": (cam_intr.detach().float().cpu() if cam_intr is not None else None),
                "s_pool": float(s_pool), "s_med": float(s_med),
                "overlap": int(overlap), "stride": int(stride), "base": int(base),
            })

        # Camera-frame C-MPJPE (right hand, RH=1) — scale-free, chaining-free: scores the hand head
        # directly in the camera frame (Hand3R's "C-MPJPE" axis). Dedupe overlapping clip frames so
        # each absolute frame is counted once.
        RH = 1
        pcf = {}
        # Starred, NOT a fixed arity: clip_cams grew from 4 to 5 fields when s_failed was
        # added and this line was missed, killing every seg100 sequence with
        # 'too many values to unpack'. seg30 never reached it on the old file, so it stayed
        # latent until a mid-run sync. Star it so the next field cannot break it.
        for c, (pj, *_rest) in enumerate(clip_cams):
            start = (base + c) * stride
            for kk in range(pj.shape[0]):
                f = start + kk
                if f not in pcf and f < gt_cam.shape[0]:
                    pcf[f] = pj[kk, RH]
        fr = sorted(pcf)
        if fr:
            pc_cam = torch.stack([pcf[f] for f in fr])                 # [T,16,3]
            gc_cam = gt_cam[fr, RH]                                    # [T,16,3]
            vc = gt_valid[fr, RH].unsqueeze(-1).expand(-1, 16)         # [T,16]
            c_rr = c_mpjpe(pc_cam, gc_cam, valid=vc, root_relative=True)
            c_ab = c_mpjpe(pc_cam, gc_cam, valid=vc, root_relative=False)
        else:
            c_rr = c_ab = float("nan")

        # (b) scale-vs-GT aggregate over this segment's clips (our hand scale vs the true camera scale)
        scale_gt = {}
        if s_pairs:
            sgt = torch.tensor([a for a, _ in s_pairs])
            shd = torch.tensor([b for _, b in s_pairs])
            ratio = shd / sgt.clamp_min(1e-6)
            scale_gt = {"s_gt_med": float(sgt.median()), "s_hand_med": float(shd.median()),
                        "scale_ratio_med": float(ratio.median()),
                        "scale_ratio_std": float(ratio.std()) if ratio.numel() > 1 else 0.0,
                        "abs_scale_err_med": float((shd - sgt).abs().median())}

        row = {"seq": os.path.basename(seq_dir), "seg": seg, "frames": g_pc["frames"],
               "s_med": s_med, "s_pool": s_pool, "s_clip_std": s_std,
               "s_clip_min": float(scales.min()), "s_clip_max": float(scales.max()),
               "C_MPJPE": c_rr, "C_MPJPE_abs": c_ab}
        for k in ("W_MPJPE", "WA_MPJPE_short", "WA_MPJPE_long"):
            row[k] = g_pc[k]                       # bare keys = per-clip scale, greedy (back-compat)
            row[k + "_global"] = gl_pc[k]
            row[k + "_smed"] = g_md[k]             # per-seq median scale
            row[k + "_spool"] = g_pl[k]            # per-seq pooled scale (principled, sequence-level)
            row[k + "_link"] = gl_link[k]          # pooled scale, linker WITH camera centres
            row[k + "_linkcf"] = gl_linkcf[k]      # pooled scale, linker cross-fade only (no centres, safe)
            row[k + "_linkr"] = gl_linkr[k]        # RIGID linker (s=1 per clip) WITH centres
            row[k + "_linkrcf"] = gl_linkrcf[k]    # RIGID linker, no centres
        row.update(sm_rows)
        row.update(scale_gt)
        if g_gts is not None:
            # Same chaining, same rotations, same translation DIRECTIONS - only the scale
            # magnitude is replaced by the GT one. W_MPJPE_gtscale vs W_MPJPE therefore
            # attributes the translation term to magnitude vs direction (task #38).
            for k in ("W_MPJPE", "WA_MPJPE_short", "WA_MPJPE_long"):
                row[k + "_gtscale"] = g_gts[k]
        anchor_str = ""
        if anchor_log:
            g = sum(a["gate_rate"] for a in anchor_log) / len(anchor_log)
            dz = sum(a["dz_gated_m"] for a in anchor_log) / len(anchor_log)
            dzmax = max(a["dz_max_m"] for a in anchor_log)
            disagree = sum(a.get("disagree_gated_m", 0.0) for a in anchor_log) / len(anchor_log)
            row["anchor_gate_rate"] = g
            row["anchor_dz_gated_mm"] = dz * 1000.0
            row["anchor_dz_max_mm"] = dzmax * 1000.0
            row["anchor_disagree_mm"] = disagree * 1000.0
            anchor_str = (f" | anchor gate={g * 100:.0f}% |dz|={dz * 1000:.1f}mm"
                          f"(max {dzmax * 1000:.1f}) Δgs={disagree * 1000:.1f}mm")
        out.append(row)
        sm_str = ""
        if smooth_windows:
            sm_str = " | Wsm " + " ".join(
                f"w{w}={sm_rows[f'W_MPJPE_sm{w}']:.1f}" for w in smooth_windows)
        print(f"[{os.path.basename(seq_dir)} seg{seg}] "
              f"W perclip={g_pc['W_MPJPE']:.1f} smed={g_md['W_MPJPE']:.1f} spool={g_pl['W_MPJPE']:.1f} "
              f"link(cf/wc)={gl_linkcf['W_MPJPE']:.1f}/{gl_link['W_MPJPE']:.1f} | "
              f"WA(s/l)={g_pc['WA_MPJPE_short']:.1f}/{g_pc['WA_MPJPE_long']:.1f} | "
              f"C(rr/abs)={c_rr:.1f}/{c_ab:.1f} | s med/pool={s_med:.3f}/{s_pool:.3f} ±{s_std:.3f} "
              f"({g_pc['frames']}f){sm_str}{anchor_str}", flush=True)
    if dump_cam_dir is not None:
        os.makedirs(dump_cam_dir, exist_ok=True)
        torch.save({"cam_joints": cam_buf, "world_joints": world_buf, "valid": val_buf},
                   os.path.join(dump_cam_dir, f"{_sq}.pt"))
        n_w = int(torch.isfinite(world_buf).all(-1).all(-1).sum())
        print(f"[{_sq}] dumped cam preds {tuple(cam_buf.shape)} valid={int(val_buf.sum())} "
              f"world_frames={n_w} -> {dump_cam_dir}", flush=True)
    return out


def eval_oracle_cam(model, mano_model, device, seq_dir, cfg, clip_len, stride, max_clips=16):
    """Diagnostic ceiling: place predicted cam-frame hands into the GLOBAL GT world using GT
    camera extrinsics (w2c, inverted). No scale, no chaining -> isolates pure hand-head world
    placement error. If this is small, all our drift is camera/chaining (fixable by a better
    global alignment); if large, the hand head itself is the limit.
    """
    from scripts.train_hand_head import HOT3DHandDataset, build_views, compute_joints_from_batch

    mcfg = cfg["model"]
    ds = HOT3DHandDataset([seq_dir], mano_model, num_frames=clip_len, clip_stride=stride,
                          use_hand_crop=mcfg.get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5))
    if len(ds) == 0:
        return None
    hd = os.path.join(seq_dir, "hand_data")
    gt_world = torch.load(os.path.join(hd, "gt_joints_cache_world.pt"), map_location="cpu").float()
    bb = torch.load(os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt"), map_location="cpu")
    gt_valid = bb["valid"].bool()
    n_clips = len(ds) if max_clips <= 0 else min(len(ds), max_clips)
    errs = []
    for j in range(n_clips):
        batch = ds[j]
        if "cam_extrinsics" not in batch:
            return None
        imgs = batch["img"].unsqueeze(0).to(device)
        hb = batch["hand_bboxes"].unsqueeze(0).to(device) if "hand_bboxes" in batch else None
        hv = batch["hand_valid"].unsqueeze(0).to(device) if "hand_valid" in batch else None
        views = build_views(imgs, clip_len, device, hb, hv)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(views, is_inference=True, use_motion=False)
        pj = compute_joints_from_batch(preds["hand_joints"], mano_model, device)[0].float().cpu()  # [S,2,16,3] cam (m)
        w2c = batch["cam_extrinsics"].float().cpu()                                                # [S,4,4] world->cam
        start = j * stride
        for k in range(pj.shape[0]):
            fidx = start + k
            if fidx >= gt_world.shape[0]:
                break
            c2w = torch.inverse(w2c[k])
            pw = (c2w[:3, :3] @ pj[k].reshape(-1, 3).T).T + c2w[:3, 3]   # [32,3] world (m)
            gw = gt_world[fidx].reshape(-1, 3)                          # [32,3]
            vmask = gt_valid[fidx].repeat_interleave(16)                # [32]
            if vmask.any():
                errs.append((pw - gw).norm(dim=-1)[vmask])
    if not errs:
        return None
    return float(torch.cat(errs).mean().item() * 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    # 0 = ALL sequences, and it is the DEFAULT. This used to default to 4 with 0 meaning
    # "nothing" (the slice is [start : start+max_seqs]), which silently produced 4-sequence
    # numbers that look like full results: the H2O zero-shot scored 4 of 177 and the 32-frame
    # arm 4 of 157, both reported as if complete. A quick probe should be opt-IN, not the default.
    ap.add_argument("--max_seqs", type=int, default=0,
                    help="how many sequences to evaluate; 0 (default) means ALL")
    ap.add_argument("--seq_start", type=int, default=0, help="skip the first N cached sequences")
    ap.add_argument("--segment_len", type=int, default=128)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--declare_protocol", default=None, metavar="REASON",
                    help="Record a non-16/8 clip/stride as a DELIBERATE ablation rather than an "
                         "accidental deviation, and write REASON into the result JSON. Use for "
                         "the window-length arm (e.g. --declare_protocol 'window-length ablation: "
                         "model trained at 32 frames'). It does NOT suppress the train/inference "
                         "mismatch warning, which is the check that actually invalidates numbers.")
    ap.add_argument("--wa_short", type=int, default=16)
    ap.add_argument("--max_segs", type=int, default=0, help="cap segments per sequence (0 = all)")
    ap.add_argument("--oracle_cam", action="store_true",
                    help="diagnostic: GT camera extrinsics + no chaining -> hand-head world ceiling")
    ap.add_argument("--max_clips", type=int, default=16, help="oracle_cam: clips per sequence")
    ap.add_argument("--feed_intrinsics", action="store_true",
                    help="condition backbone on known intrinsics (cond_flags=[0,0,1]) vs identity dummy")
    ap.add_argument("--smooth_windows", default="",
                    help="comma-separated odd window sizes for the root-smoothing diagnostic (e.g. 3,9,15,31)")
    ap.add_argument("--dump_traj", default="",
                    help="if set, torch.save per-segment pooled trajectories here for an offline smoothing sweep")
    ap.add_argument("--refine_pose", action="store_true",
                    help="per-clip camera-pose refinement (render-and-optimize against the clip's static map)")
    ap.add_argument("--refine_sanity", action="store_true",
                    help="refine_pose plumbing check: report feedforward-pose render PSNR only, no optimization")
    ap.add_argument("--refine_iters", type=int, default=40, help="pose-refine Adam iters per frame")
    ap.add_argument("--refine_lr", type=float, default=3e-3, help="pose-refine se3 learning rate")
    ap.add_argument("--refine_frame_stride", type=int, default=1, help="refine every Nth frame (speed)")
    ap.add_argument("--scale_fallback", choices=["clamp", "pooled"], default="clamp",
                    help="what to do when a clip's scale solve lands on a clamp bound (it failed): "
                         "'clamp' keeps the bound (historical behaviour); 'pooled' substitutes the "
                         "sequence-pooled scale for those clips only. The failure RATE is reported "
                         "under both policies.")
    ap.add_argument("--require_positive_z", action="store_true",
                    help="exclude behind-camera joints (z<=0) from the scene-scale solve. The "
                         "projection clamps depth to Z_MIN=5cm, so such a joint lands at a "
                         "plausible in-frame pixel while contributing a NEGATIVE ratio to the "
                         "median: a candidate cause of the segments that solve onto the 0.1 "
                         "clamp floor (task #63). OFF by default until the A/B measures it; the "
                         "behind-camera RATE is reported under both settings.")
    ap.add_argument("--gate_scale_on_hand_valid", action="store_true",
                    help="drop joints belonging to a hand slot the detector never filled from the "
                         "scene-scale solve (task #70). The model emits a default MANO for an "
                         "empty slot and those joints project to plausible in-frame pixels, so "
                         "the purely geometric mask accepts them: measured on 5 HOI4D clips whose "
                         "store has valid[:,0]==0 on all 300 frames, the absent hand supplied 96 "
                         "correspondences per clip at median ratio -0.019 and 94.1% of all "
                         "behind-camera samples. This is the CAUSE that --require_positive_z only "
                         "masks the symptom of. OFF by default until the A/B measures W/WA/s_med; "
                         "the absent-hand RATE is reported under both settings.")
    ap.add_argument("--scene_depth_window", type=int, default=1,
                    help="odd neighbourhood (in depth-map pixels) to read scene depth over "
                         "when solving the scene scale. 1 = the historical single-point read.")
    ap.add_argument("--scene_depth_reduce", default="bilinear",
                    help="how to reduce that neighbourhood: bilinear (single point), min, or "
                         "qNN such as q10. Contamination at a hand pixel is ONE-SIDED (any "
                         "misregistration blends in BACKGROUND, which is farther), so a "
                         "low-order statistic is the right robust estimator. Measured bias: "
                         "s_hand 0.6208 vs s_gt 1.0230, ratio 0.578 (task #63).")
    ap.add_argument("--robust_scale", action="store_true",
                    help="MAD-reject z_hand/scene_depth outliers before the per-seq pooled scale median")
    ap.add_argument("--da3_wrist_cache_dir", default=None,
                    help="C1: per-seq DA3 metric wrist-depth caches (<seq>_da3_wrist.pt) as the anchor ref")
    ap.add_argument("--contact_cache_dir", default=None,
                    help="per-seq contact caches (<seq>_contact.pt) for --contact_gate oracle")
    ap.add_argument("--contact_gate", choices=["off", "oracle"], default="off",
                    help="oracle = gate the anchor by the cached contact mask (needs --contact_cache_dir)")
    ap.add_argument("--oracle_depth", action="store_true",
                    help="add the W_MPJPE_depthOracle ceiling: replace pred wrist depth with GT wrist "
                         "depth (perfect metric depth anchor) to isolate the hand-depth term of W")
    ap.add_argument("--dense_link", action="store_true",
                    help="G1 MonST3R gate: chain clips via dense hand-masked static-scene "
                         "correspondences (gs_depth unprojection) -> W_MPJPE_dchain/dchainr")
    ap.add_argument("--gravity_oracle", action="store_true",
                    help="GVHMR-style gravity-view oracle: correct predicted camera tilt/roll to true "
                         "gravity (W_MPJPE_gravOracle) + full-rotation ceiling (W_MPJPE_rotOracle)")
    ap.add_argument("--gravity_axis", default="0,1,0",
                    help="world gravity direction, comma-separated (HOI4D world is Y-up -> 0,1,0)")
    ap.add_argument("--diag_cam", action="store_true",
                    help="print predicted vs GT per-clip camera-centre excursion (scale-effect diagnostic)")
    ap.add_argument("--dump_cam_preds", default="",
                    help="if set, dump per-seq per-frame cam-space hands {cam_joints[N,2,16,3],valid} "
                         "to this dir (for the 'ours + SLAM' world composition, lever 2)")
    ap.add_argument("--out", default="world_eval.json")
    args = ap.parse_args()

    smooth_windows = [int(x) for x in args.smooth_windows.split(",") if x.strip()] or None
    dump_list = [] if args.dump_traj else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Declare the TRUE pixel-frame width once, from the store resolution, so the depth-sampling
    # projection stops estimating it as 2*cx. That estimate assumes a centred principal point and
    # HOI4D violates it (cx=114.28, cy=108.52 for a 224x224 frame -> 2*cx overestimates by 2.0%),
    # which put every sampled depth ~4.5 px off at the frame edge. See task #64.
    from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
        set_default_frame_width)
    _res = cfg.get("data", {}).get("resolution")
    if _res:
        _w = int(_res[0]) if isinstance(_res, (list, tuple)) else int(_res)
        set_default_frame_width(_w)
        print(f"[frame width] declared {_w} px from data.resolution "
              f"(NOT estimated as 2*cx)", flush=True)
    else:
        print("[frame width] data.resolution absent; falling back to the 2*cx estimate, which is "
              "WRONG on stores with an off-centre principal point", flush=True)

    # PROTOCOL GUARD. The head is trained on 16-frame clips (data.num_frames), and the locked
    # paper protocol is 16-frame chunks with an overlap of 8, i.e. --clip_len 16 --stride 8
    # (which are also the argparse defaults). Evaluating at any other clip length runs the model
    # out of distribution: measured C-abs degrades monotonically 29.7 / 30.2 / 30.7 / 32.2 at
    # clip_len 16 / 32 / 48 / 64. An entire round of world numbers was silently produced at 32/16
    # before this was caught, so make any deviation impossible to miss.
    _tf = int(cfg.get("data", {}).get("num_frames", 16))
    if (args.clip_len, args.stride) != (16, 8):
        print("=" * 78, flush=True)
        if args.declare_protocol:
            # A declared window-length ablation is legitimate PROVIDED the model was trained at
            # this window. Declaring it does not excuse a train/inference mismatch, so the
            # out-of-distribution check below still fires independently.
            print(f"DECLARED PROTOCOL: clip_len={args.clip_len} stride={args.stride} "
                  f"-- {args.declare_protocol}", flush=True)
        else:
            print(f"!! PROTOCOL DEVIATION: clip_len={args.clip_len} stride={args.stride}, "
                  f"expected the locked 16/8.", flush=True)
            print("!! Numbers from this run are NOT paper-reportable unless this is a deliberate "
                  "probe. Pass --declare_protocol '<reason>' to record it as intentional.",
                  flush=True)
        print("=" * 78, flush=True)
    if args.clip_len != _tf:
        print(f"!! clip_len={args.clip_len} != training num_frames={_tf} -> the model is being "
              f"run OUT OF DISTRIBUTION. This is the mismatch that invalidated a whole round of "
              f"world numbers; --declare_protocol does NOT make it valid.", flush=True)
    from scripts.hand_vis_utils import MANOModel
    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])
    model = build_model(cfg, device)
    if not args.refine_pose:
        # This eval only consumes preds["gs_depth"] (from gs_head, produced BEFORE the splat
        # render): it drives the per-clip metric scale, the dense-link correspondences and the
        # depth anchor. Only --refine_pose needs the rendered splats. So skip the
        # gsplat/torch_scatter rasterization, which hangs on nodes without a compiled fast
        # rasterizer (the CPU fallback spins one core indefinitely).
        # NOTE: keep enable_gs=True in the config - with GS off there is no gs_depth at all and
        # the per-clip scale silently degrades to s=1.0, inflating every world/W metric.
        model.gs_anchor_only = True

    def _has_caches(seq_dir):
        hd = os.path.join(seq_dir, "hand_data")
        return os.path.exists(os.path.join(hd, "gt_joints_cache_world.pt"))

    all_seqs = sorted(os.path.join(args.data_root, d) for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d)))
    _usable = [s for s in all_seqs if _has_caches(s)]
    # max_seqs == 0 means ALL. Previously 0 sliced to an EMPTY list, so the intuitive "give me
    # everything" silently evaluated nothing.
    _end = len(_usable) if args.max_seqs <= 0 else args.seq_start + args.max_seqs
    seqs = _usable[args.seq_start:_end]
    print(f"evaluating {len(seqs)} of {len(_usable)} usable sequences "
          f"(max_seqs={args.max_seqs or 'ALL'}, seq_start={args.seq_start})", flush=True)
    if 0 < len(seqs) < 0.5 * len(_usable):
        print(f"!! PARTIAL EVAL: {len(seqs)}/{len(_usable)} sequences. Numbers from this run are "
              f"a PROBE, not a full result - do not put them in a table.", flush=True)
    print(f"Evaluating {len(seqs)}/{len(all_seqs)} sequences that have hand_data caches:",
          *[os.path.basename(s) for s in seqs], flush=True)
    if args.oracle_cam:
        ceil = []
        for sq in seqs:
            try:
                e = eval_oracle_cam(model, mano_model, device, sq, cfg,
                                    args.clip_len, args.stride, max_clips=args.max_clips)
            except Exception as ex:
                print(f"[skip {os.path.basename(sq)}] {type(ex).__name__}: {ex}", flush=True)
                traceback.print_exc()
                e = None
            if e is not None:
                ceil.append(e)
                print(f"[{os.path.basename(sq)}] oracle-cam W-MPJPE={e:.1f} mm", flush=True)
        if ceil:
            print(f"\nORACLE-CAM (GT poses, no chaining)  W-MPJPE={sum(ceil)/len(ceil):.1f} mm  "
                  f"(n={len(ceil)} seqs) -- hand-head world placement ceiling")
        else:
            print("No oracle-cam sequences evaluated.")
        return

    results = []
    for sq in seqs:
        try:
            results += eval_sequence(model, mano_model, device, sq, cfg,
                                     args.segment_len, args.clip_len, args.stride, args.wa_short,
                                     max_segs=args.max_segs, feed_intrinsics=args.feed_intrinsics,
                                     smooth_windows=smooth_windows, dump_list=dump_list,
                                     refine_pose=args.refine_pose, refine_iters=args.refine_iters,
                                     refine_lr=args.refine_lr, refine_frame_stride=args.refine_frame_stride,
                                     refine_sanity=args.refine_sanity, robust_scale=args.robust_scale,
                                     scale_fallback=args.scale_fallback,
                                     da3_wrist_cache_dir=args.da3_wrist_cache_dir,
                                     contact_cache_dir=args.contact_cache_dir,
                                     contact_gate=args.contact_gate,
                                     oracle_depth=args.oracle_depth, dense_link=args.dense_link,
                                     gravity_oracle=args.gravity_oracle,
                                     gravity_axis=tuple(float(x) for x in args.gravity_axis.split(",")),
                                     dump_cam_dir=(args.dump_cam_preds or None),
                                     diag_cam=args.diag_cam,
                                     require_positive_z=args.require_positive_z,
                                     scene_depth_window=args.scene_depth_window,
                                     scene_depth_reduce=args.scene_depth_reduce,
                                     gate_scale_on_hand_valid=args.gate_scale_on_hand_valid)
        except Exception as e:
            print(f"[skip {os.path.basename(sq)}] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    if dump_list is not None:
        torch.save(dump_list, args.dump_traj)
        print(f"Dumped {len(dump_list)} segment trajectories -> {args.dump_traj}", flush=True)

    valid = [r for r in results if r["W_MPJPE"] == r["W_MPJPE"]]  # drop nan (greedy W defines validity)
    if valid:
        def _mean(key):
            vals = [r[key] for r in valid if key in r and r[key] == r[key]]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        keys = ("W_MPJPE", "WA_MPJPE_short", "WA_MPJPE_long")
        agg = {}
        for k in keys:
            # "_gtscale" is the task #38 oracle: our rotations and translation DIRECTIONS with
            # the GT scale magnitude substituted. Comparing it against the bare key attributes
            # the translation term to magnitude vs direction, which is what decides whether
            # "fix the scale" is the top W lever or a minor one.
            for suf in ("", "_global", "_smed", "_spool", "_link", "_linkcf", "_linkr",
                        "_linkrcf", "_gtscale"):
                agg[k + suf] = _mean(k + suf)
        for k in ("C_MPJPE", "C_MPJPE_abs", "s_med", "s_pool", "s_clip_std"):
            agg[k] = _mean(k)
        if smooth_windows:
            for w in smooth_windows:
                agg[f"W_MPJPE_sm{w}"] = _mean(f"W_MPJPE_sm{w}")
                agg[f"WA_MPJPE_long_sm{w}"] = _mean(f"WA_MPJPE_long_sm{w}")
        for k in ("W_MPJPE_velGT", "W_MPJPE_re16", "W_MPJPE_re32",
                  "W_MPJPE_depthOracle", "WA_MPJPE_short_depthOracle",
                  "W_MPJPE_dchain", "W_MPJPE_dchainr"):
            agg[k] = _mean(k)
        for k in ("s_gt_med", "s_hand_med", "scale_ratio_med", "scale_ratio_std", "abs_scale_err_med"):
            agg[k] = _mean(k)
        agg["n_segments"] = len(valid)
        with open(args.out, "w") as f:
            # Stamp the protocol into the artifact. Without this a result JSON cannot be
            # audited after the fact - exactly how the provenance of the short-window world
            # table became unrecoverable (the producing job script was gone and the JSON did
            # not say what clip length it used).
            _tf_stamp = int(cfg.get("data", {}).get("num_frames", -1))
            protocol = {
                "clip_len": args.clip_len, "stride": args.stride,
                "segment_len": args.segment_len, "wa_short": args.wa_short,
                "train_num_frames": _tf_stamp,
                "matches_locked_protocol": (args.clip_len == 16 and args.stride == 8),
                # A declared ablation is NOT the same artifact as an accidental deviation, but it
                # is also not the locked protocol. Keep both facts: matches_locked_protocol stays
                # honestly false, and the reason this run is legitimate is recorded next to it.
                "declared_protocol": args.declare_protocol,
                # MEASURED training stores, next to the DECLARED ones. --declare_protocol is typed
                # by hand into an sbatch script and drifts the moment the mix changes: the varlen
                # arms were submitted declaring "ARCTIC+OakInk2+HOT3D" and then had HOT3D removed
                # from data_roots, so the string alone would have asserted a training set the run
                # never saw. A zero-shot claim rests entirely on which stores trained the model, so
                # it is read from the config the checkpoint was actually produced with instead of
                # being trusted. Compare the two fields before quoting any cross-dataset number.
                "train_stores_measured": sorted(
                    r.get("name", r.get("root", "?")) for r in
                    (cfg.get("data", {}).get("data_roots")
                     or ([{"root": cfg["data"]["data_root"]}]
                         if cfg.get("data", {}).get("data_root") else []))),
                "train_random_frames": cfg.get("data", {}).get("random_frames"),
                # The mismatch that actually invalidates numbers is inference != training, not
                # inference != 16. Record it explicitly so an audit needs no other file.
                "train_inference_match": (args.clip_len == _tf_stamp),
                "robust_scale": bool(args.robust_scale),
                "config": args.config, "data_root": args.data_root,
            }
            json.dump({"protocol": protocol, "aggregate": agg, "per_segment": results},
                      f, indent=2)
        print(f"\nOURS W-MPJPE  per-clip={agg['W_MPJPE']:.1f}  per-seq-median={agg['W_MPJPE_smed']:.1f}  "
              f"per-seq-pooled={agg['W_MPJPE_spool']:.1f}")
        print(f"OURS WA(short/long)  per-clip={agg['WA_MPJPE_short']:.1f}/{agg['WA_MPJPE_long']:.1f}  "
              f"per-seq-pooled={agg['WA_MPJPE_short_spool']:.1f}/{agg['WA_MPJPE_long_spool']:.1f}")
        print(f"OURS CHUNK LINKER (pooled scale, W-MPJPE)  greedy={agg['W_MPJPE_spool']:.1f}  "
              f"cross-fade-only={agg['W_MPJPE_linkcf']:.1f}  +camera-centres={agg['W_MPJPE_link']:.1f}\n"
              f"  WA(short/long)  cf-only={agg['WA_MPJPE_short_linkcf']:.1f}/{agg['WA_MPJPE_long_linkcf']:.1f}  "
              f"+centres={agg['WA_MPJPE_short_link']:.1f}/{agg['WA_MPJPE_long_link']:.1f}\n"
              f"  -> cross-fade-only is the SAFE linker (never worse than greedy in tests); camera "
              f"centres help ONLY if the predicted cam poses are reliable - compare, do not assume.")
        print(f"OURS C-MPJPE (cam)   root-rel={agg['C_MPJPE']:.1f}  abs={agg['C_MPJPE_abs']:.1f}  "
              f"| mean s med/pool={agg['s_med']:.3f}/{agg['s_pool']:.3f} (clip-std {agg['s_clip_std']:.3f})  "
              f"(n={agg['n_segments']} segs) -> {args.out}")
        if smooth_windows:
            curve = "  ".join(f"w{w}={agg[f'W_MPJPE_sm{w}']:.1f}" for w in smooth_windows)
            print(f"OURS W-MPJPE root-smoothing (pooled, baseline {agg['W_MPJPE_spool']:.1f}):  {curve}\n"
                  f"  -> W drops with window => DRIFT (build the temporal head); W flat => BIAS "
                  f"(head alone insufficient).")
        print(f"OURS velocity-head CEILINGS (pooled W {agg['W_MPJPE_spool']:.1f}):  "
              f"GT-velocity={agg['W_MPJPE_velGT']:.1f}  reanchor16={agg['W_MPJPE_re16']:.1f}  "
              f"reanchor32={agg['W_MPJPE_re32']:.1f}\n"
              f"  -> GT-velocity LOW => perfect relative motion collapses W, a trajectory head has "
              f"large headroom (BUILD); GT-velocity HIGH => motion is not the lever (don't).")
        if agg.get("W_MPJPE_depthOracle") == agg.get("W_MPJPE_depthOracle"):  # not nan
            print(f"OURS DEPTH-ANCHOR CEILING (pooled W {agg['W_MPJPE_spool']:.1f}):  "
                  f"GT-wrist-depth W={agg['W_MPJPE_depthOracle']:.1f}  "
                  f"WA_short={agg['WA_MPJPE_short_depthOracle']:.1f}\n"
                  f"  -> W_depthOracle << W_spool => per-frame hand ABSOLUTE DEPTH is the dominant W "
                  f"lever (dense metric depth / DA3 has real headroom); ~= W_spool => camera "
                  f"trajectory / scene scale is the bottleneck, not hand depth.")
        if agg.get("W_MPJPE_dchainr") == agg.get("W_MPJPE_dchainr"):  # not nan
            print(f"OURS DENSE-CHAIN GATE (pooled W {agg['W_MPJPE_spool']:.1f}):  "
                  f"rigid dense-link W={agg['W_MPJPE_dchainr']:.1f}  "
                  f"sim dense-link W={agg['W_MPJPE_dchain']:.1f}\n"
                  f"  -> dchainr << W_spool => dense static-scene seams fix per-link rotation drift "
                  f"(BUILD the MonST3R-style windowed-graph optimization); ~= W_spool => the "
                  f"per-clip dense geometry is drift-inconsistent too (test-time alignment dead).")
        if agg.get("s_gt_med") == agg.get("s_gt_med"):  # not nan
            print(f"OURS SCALE vs GT (b) [true scale = sim(pred cam centers -> GT metric centers)]:  "
                  f"s_hand={agg['s_hand_med']:.3f}  s_gt={agg['s_gt_med']:.3f}  "
                  f"ratio(hand/gt)={agg['scale_ratio_med']:.3f} ±{agg['scale_ratio_std']:.3f}  "
                  f"|s_hand-s_gt|={agg['abs_scale_err_med']:.3f}\n"
                  f"  -> ratio far from 1 => our hand scale is BIASED vs truth; large ± => NOISY; "
                  f"both feed world drift (the world-lift scales camera translation by s_hand).")
    else:
        # Exit NONZERO. Returning 0 here made a total failure look like a successful job: the
        # H2O run "completed" in 46 s having evaluated nothing (every sequence raised
        # KeyError: 'gt'), and its SLURM state was COMPLETED. A job that produced no measurement
        # must fail loudly, or a dependent job will happily chain off it.
        raise SystemExit(
            "No valid segments evaluated - every sequence was skipped. This is a FAILURE, not an "
            "empty result: check the per-sequence [skip ...] lines above for the cause (a missing "
            "cache key and a wrong data_root both look like this)."
        )


if __name__ == "__main__":
    main()
