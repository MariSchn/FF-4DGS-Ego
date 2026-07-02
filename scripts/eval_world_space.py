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
    chain_trajectories_global,
    reanchor_to_gt,
    replace_root_with_gt_motion,
    smooth_root_trajectory,
    solve_similarity,
    solve_similarity_robust,
    w_mpjpe,
    w_mpjpe_first_window_aligned,
    wa_mpjpe,
)


def build_model(cfg, device):
    """Build WorldMirror from cfg, load the base checkpoint, warm-start the hand head."""
    from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
    mcfg = cfg["model"]
    model = WorldMirror(**{k: v for k, v in mcfg.items() if k != "checkpoint"})
    ckpt = torch.load(mcfg["checkpoint"], map_location=device)
    state = ckpt.get("state_dict", ckpt.get("reconstructor", ckpt))
    model.load_state_dict(state, strict=False)
    if mcfg.get("warm_start_hand_head"):
        ws = torch.load(mcfg["warm_start_hand_head"], map_location=device)
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


def predict_clip(preds, mano_model, device, cam_intr, model=None, anchor_log=None, contact_mask=None,
                 ref_d_scene=None):
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
    c2w = preds["rendered_extrinsics"][0].float()           # [S,4,4] cam->world (clip-local, up-to-scale)
    gs_depth = preds.get("gs_depth")

    ratios = torch.empty(0)
    s = 1.0
    if gs_depth is not None and cam_intr is not None:
        grid_xy, z = project_joints_to_norm_pixels(pred_joints, cam_intr.to(device))
        sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy)
        valid = in_frame & (sampled > 0.01) & torch.isfinite(z) & torch.isfinite(sampled)
        if bool(valid.any()):
            ratios = (z / sampled)[valid].detach().float().cpu()    # [n_valid] z_hand/scene_depth
            s = float(ratios.median().clamp(0.1, 10.0))

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
    return pred_joints[0].float().cpu(), c2w.cpu(), s, ratios


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
                  refine_sanity=False, robust_scale=False):
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

    overlap = clip_len - stride
    clips_per_seg = segment_len // stride
    out = []
    n_seg = len(ds) // clips_per_seg
    if max_segs > 0:
        n_seg = min(n_seg, max_segs)
    for seg in range(n_seg):
        base = seg * clips_per_seg
        clip_cams = []   # [(pj_cam [S,H,J,3], c2w [S,4,4], s_perclip), ...]
        s_pairs = []     # (s_gt, s_hand) per clip: GT camera-trajectory scale vs our hand scale (b)
        anchor_log = []  # per-clip {gate_rate, dz_gated_m, dz_max_m} when the root anchor is active
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
            cc = predict_clip(preds, mano_model, device, cam_intr, model=model, anchor_log=anchor_log)
            clip_cams.append(cc)

            # (b) GROUND-TRUTH SCALE check: the world placement scales the up-to-scale camera
            # translation by our hand-derived s. The *true* metric scale is the similarity that maps
            # the predicted (up-to-scale) camera centers onto the GT metric camera centers. Comparing
            # the two says whether our hand scale is right in absolute terms (ratio==1) or biased/noisy.
            if "cam_extrinsics" in batch:
                c2w_gt = torch.inverse(batch["cam_extrinsics"].float())     # [S,4,4] metric cam->world
                pc = cc[1][:, :3, 3]                                        # pred centers (up-to-scale)
                gc = c2w_gt[:, :3, 3]                                       # gt centers (metric)
                if pc.shape[0] >= 3 and float((pc - pc.mean(0)).norm(dim=-1).max()) > 1e-4:
                    s_gt, _, _ = solve_similarity(pc, gc)
                    if torch.isfinite(s_gt) and float(s_gt) > 1e-6:
                        s_pairs.append((float(s_gt), float(cc[2])))

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

        # Three scene scales, each applied ONLY to the camera translation (the hand is already
        # metric). per-clip = the per-frame heuristic (one closed-form solve per clip; high
        # variance -> the chained absolute trajectory drifts -> large W-MPJPE). per-seq MEDIAN =
        # robust median of the per-clip scalars. per-seq POOLED = ONE median over EVERY valid
        # (z_hand / scene_depth) correspondence in the whole sequence -> the principled
        # sequence-level solve (every joint votes once, robust to clips with few valid joints).
        scales = torch.tensor([s for (_, _, s, _) in clip_cams], dtype=torch.float32)
        s_med = float(scales.median())
        s_std = float(scales.std()) if scales.numel() > 1 else 0.0
        pooled = [r for (_, _, _, r) in clip_cams if r.numel()]
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

        worlds_pc = [_world_from_cam(pj, c2w, s) for (pj, c2w, s, _) in clip_cams]       # per-clip
        worlds_md = [_world_from_cam(pj, c2w, s_med) for (pj, c2w, _, _) in clip_cams]   # per-seq median
        worlds_pl = [_world_from_cam(pj, c2w, s_pool) for (pj, c2w, _, _) in clip_cams]  # per-seq pooled
        g_pc = _greedy(worlds_pc)
        gl_pc = _global(worlds_pc)
        g_md = _greedy(worlds_md)
        chain_pl = chain_trajectories_by_overlap(worlds_pl, overlap=overlap)  # pooled greedy traj
        g_pl = _metrics(chain_pl)

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
        if dump_list is not None:
            tt = min(chain_pl.shape[0], t_avail)
            dump_list.append({
                "seq": os.path.basename(seq_dir), "seg": seg, "wa_short": wa_short,
                "pred_world": chain_pl[:tt].cpu(), "gt_world": gtw[:tt].cpu(),
                "valid": val[:tt].cpu(),
            })

        # Camera-frame C-MPJPE (right hand, RH=1) — scale-free, chaining-free: scores the hand head
        # directly in the camera frame (Hand3R's "C-MPJPE" axis). Dedupe overlapping clip frames so
        # each absolute frame is counted once.
        RH = 1
        pcf = {}
        for c, (pj, _, _, _) in enumerate(clip_cams):
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
        row.update(sm_rows)
        row.update(scale_gt)
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
              f"W perclip={g_pc['W_MPJPE']:.1f} smed={g_md['W_MPJPE']:.1f} spool={g_pl['W_MPJPE']:.1f} | "
              f"WA(s/l)={g_pc['WA_MPJPE_short']:.1f}/{g_pc['WA_MPJPE_long']:.1f} | "
              f"C(rr/abs)={c_rr:.1f}/{c_ab:.1f} | s med/pool={s_med:.3f}/{s_pool:.3f} ±{s_std:.3f} "
              f"({g_pc['frames']}f){sm_str}{anchor_str}", flush=True)
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
    ap.add_argument("--max_seqs", type=int, default=4)
    ap.add_argument("--seq_start", type=int, default=0, help="skip the first N cached sequences")
    ap.add_argument("--segment_len", type=int, default=128)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
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
    ap.add_argument("--robust_scale", action="store_true",
                    help="MAD-reject z_hand/scene_depth outliers before the per-seq pooled scale median")
    ap.add_argument("--out", default="world_eval.json")
    args = ap.parse_args()

    smooth_windows = [int(x) for x in args.smooth_windows.split(",") if x.strip()] or None
    dump_list = [] if args.dump_traj else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from scripts.hand_vis_utils import MANOModel
    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])
    model = build_model(cfg, device)

    def _has_caches(seq_dir):
        hd = os.path.join(seq_dir, "hand_data")
        return os.path.exists(os.path.join(hd, "gt_joints_cache_world.pt"))

    all_seqs = sorted(os.path.join(args.data_root, d) for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d)))
    seqs = [s for s in all_seqs if _has_caches(s)][args.seq_start:args.seq_start + args.max_seqs]
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
                                     refine_sanity=args.refine_sanity, robust_scale=args.robust_scale)
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
            for suf in ("", "_global", "_smed", "_spool"):
                agg[k + suf] = _mean(k + suf)
        for k in ("C_MPJPE", "C_MPJPE_abs", "s_med", "s_pool", "s_clip_std"):
            agg[k] = _mean(k)
        if smooth_windows:
            for w in smooth_windows:
                agg[f"W_MPJPE_sm{w}"] = _mean(f"W_MPJPE_sm{w}")
                agg[f"WA_MPJPE_long_sm{w}"] = _mean(f"WA_MPJPE_long_sm{w}")
        for k in ("W_MPJPE_velGT", "W_MPJPE_re16", "W_MPJPE_re32"):
            agg[k] = _mean(k)
        for k in ("s_gt_med", "s_hand_med", "scale_ratio_med", "scale_ratio_std", "abs_scale_err_med"):
            agg[k] = _mean(k)
        agg["n_segments"] = len(valid)
        with open(args.out, "w") as f:
            json.dump({"aggregate": agg, "per_segment": results}, f, indent=2)
        print(f"\nOURS W-MPJPE  per-clip={agg['W_MPJPE']:.1f}  per-seq-median={agg['W_MPJPE_smed']:.1f}  "
              f"per-seq-pooled={agg['W_MPJPE_spool']:.1f}  (Hand3R 125.8)")
        print(f"OURS WA(short/long)  per-clip={agg['WA_MPJPE_short']:.1f}/{agg['WA_MPJPE_long']:.1f}  "
              f"per-seq-pooled={agg['WA_MPJPE_short_spool']:.1f}/{agg['WA_MPJPE_long_spool']:.1f}")
        print(f"OURS C-MPJPE (cam)   root-rel={agg['C_MPJPE']:.1f}  abs={agg['C_MPJPE_abs']:.1f}  (Hand3R 42.6)  "
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
        if agg.get("s_gt_med") == agg.get("s_gt_med"):  # not nan
            print(f"OURS SCALE vs GT (b) [true scale = sim(pred cam centers -> GT metric centers)]:  "
                  f"s_hand={agg['s_hand_med']:.3f}  s_gt={agg['s_gt_med']:.3f}  "
                  f"ratio(hand/gt)={agg['scale_ratio_med']:.3f} ±{agg['scale_ratio_std']:.3f}  "
                  f"|s_hand-s_gt|={agg['abs_scale_err_med']:.3f}\n"
                  f"  -> ratio far from 1 => our hand scale is BIASED vs truth; large ± => NOISY; "
                  f"both feed world drift (the world-lift scales camera translation by s_hand).")
    else:
        print("No valid segments evaluated.")


if __name__ == "__main__":
    main()
