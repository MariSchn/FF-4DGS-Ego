"""B2 / E2: NON-circular scene-metric — is the predicted scene metric on OBJECTS?

The scene-metric thesis is currently proven only AT the hand (the anchor trains
there → semi-circular). This script proves it (or not) on the manipulated objects,
whose metric depth is GT (HOT3D object meshes + per-frame 6-DoF poses). It renders
GT object depth (scripts/b2_render_object_depth.py, validated to ~3 cm at hand
contact) and compares the model's ``gs_depth`` to it in OBJECT regions with the
HAND region excluded.

Protocol per clip:
  1. forward the model -> gs_depth + predicted metric hand joints.
  2. solve the global scale ``s`` on the HAND (the metric the coupling provides).
  3. render GT object depth for each frame (object pose + cam extrinsics + K).
  4. in (object pixels AND NOT hand pixels): error = |gs_depth*s - gt_obj_depth|.

Run once per checkpoint (no-anchor baseline vs our anchored warm-start). The
anchored model should have clearly lower object-region depth error / higher
δ<10cm if the hand anchor made the SCENE (not just the hand) metric.

Usage (gb10 / venv_gb10):
    python -m scripts.eval_scene_metric_gt \
        --config configs/exp_p2_pinhole_warmstart.yaml \
        --checkpoint checkpoints/p2_warmstart/best_mpjpe.pt \
        --objects_dir /work/scratch/dmonopoli/hot3d_assets --num_clips 60
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.auxiliary_models.worldmirror.models.heads.metric_scale_head import (
    solve_metric_scale,
)
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
)
from scripts.b2_render_object_depth import build_frame_objects, render_object_depth
from scripts.hand_depth_anchor_loss import hand_depth_anchor_loss
from scripts.hand_vis_utils import MANOModel
from scripts.train_hand_head import (
    HOT3DHandDataset,
    build_views,
    compute_joints_from_batch,
    discover_sequences,
)

DELTA = 0.10           # metres; "metric-correct on objects" threshold
HAND_EXCL_RADIUS = 12  # px (at gs_depth resolution) disk excluded around each hand joint


def _load_trained_weights(model: WorldMirror, path: str, device: str) -> None:
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    is_full = any(k.startswith(("hand_head.", "gs_head.")) for k in sd.keys())
    tgt = model if is_full else model.hand_head
    missing, unexpected = tgt.load_state_dict(sd, strict=False)
    print(f"Loaded {'full' if is_full else 'hand-head'} ckpt '{path}' "
          f"(missing={len(missing)}, unexpected={len(unexpected)}).")


def _raw_seq_dir(seq_path: str) -> str:
    """Map a preprocessed seq path to its raw HOT3D seq (object poses + meshes live there)."""
    return seq_path.replace("/preprocessed_pinhole_f609/", "/sequences/")


def _gs_depth_2d(gs_depth: torch.Tensor) -> torch.Tensor:
    """[B,S,1,Hd,Wd] or [B,S,Hd,Wd] (B=1) -> [S,Hd,Wd]."""
    g = gs_depth[0]
    if g.dim() == 4:
        g = g[:, 0]
    return g


def _hand_pixel_mask(pred_joints, cam_intr, S, Hd, Wd, device):
    """Disk mask around projected predicted hand joints (to EXCLUDE the hand region)."""
    grid_xy, _ = project_joints_to_norm_pixels(pred_joints.detach(), cam_intr)  # [1,S,H,J,2] in [0,1]
    gx = grid_xy[0, ..., 0].reshape(S, -1) * Wd
    gy = grid_xy[0, ..., 1].reshape(S, -1) * Hd
    mask = torch.zeros(S, Hd, Wd, dtype=torch.bool, device=device)
    ys = torch.arange(Hd, device=device).view(1, Hd, 1)
    xs = torch.arange(Wd, device=device).view(1, 1, Wd)
    for s in range(S):
        for j in range(gx.shape[1]):
            cx, cy = gx[s, j], gy[s, j]
            if not (0 <= cx < Wd and 0 <= cy < Hd):
                continue
            mask[s] |= ((xs - cx) ** 2 + (ys - cy) ** 2) <= HAND_EXCL_RADIUS ** 2
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_p2_pinhole_warmstart.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--objects_dir", default="/work/scratch/dmonopoli/hot3d_assets")
    ap.add_argument("--num_clips", type=int, default=60)
    ap.add_argument("--max_seqs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg, model_cfg, vis_cfg = cfg["data"], cfg["model"], cfg.get("visualization", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not model_cfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true (we need gs_depth).")

    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    model.load_state_dict(base.get("state_dict", base.get("reconstructor", base)), strict=False)
    _load_trained_weights(model, args.checkpoint, device)
    model.to(device).eval()
    model.gs_anchor_only = True

    mano_model = MANOModel(vis_cfg["mano_model_folder"])
    num_frames = data_cfg["num_frames"]
    res = tuple(data_cfg["resolution"])
    seqs = discover_sequences(data_cfg["data_root"])
    random.seed(args.seed)
    random.shuffle(seqs)
    seqs = seqs[:args.max_seqs]
    dataset = HOT3DHandDataset(
        seqs, mano_model, num_frames=num_frames, res=res,
        clip_stride=data_cfg.get("clip_stride", num_frames),
        use_hand_crop=model_cfg.get("use_hand_crop", False),
        rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 2.0),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    avail = {p[:-4] for p in os.listdir(args.objects_dir) if p.endswith(".glb")}

    seq_cache: dict[str, list] = {}   # raw_seq -> frame_objects
    clip_scales, errs, deltas, n_obj_px = [], [], [], 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if len(clip_scales) >= args.num_clips:
                break
            if "cam_intrinsics" not in batch or "cam_extrinsics" not in batch:
                continue
            clip = dataset.clips[idx]
            raw_seq = _raw_seq_dir(clip["seq_path"])
            frame_offset = clip["frame_offset"]
            if raw_seq not in seq_cache:
                try:
                    n_video = torch.load(
                        os.path.join(clip["seq_path"], "hand_data", "cam_extrinsics_cache.pt"),
                        map_location="cpu", weights_only=True).shape[0]
                    seq_cache[raw_seq] = build_frame_objects(raw_seq, n_video)
                except Exception as e:
                    print(f"  [skip seq] {raw_seq}: {e}")
                    seq_cache[raw_seq] = None
            frame_objects = seq_cache[raw_seq]
            if frame_objects is None:
                continue

            imgs = batch["img"].to(device)
            hb = batch["hand_bboxes"].to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].to(device) if "hand_valid" in batch else None
            views = build_views(imgs, num_frames, device, hb, hv)
            preds = model(views, is_inference=False, use_motion=False)
            gs = _gs_depth_2d(preds["gs_depth"]).float()          # [S, Hd, Wd]
            S, Hd, Wd = gs.shape
            pred_joints = compute_joints_from_batch(preds["hand_joints"], mano_model, device)
            has_hand = (hv.float() if hv is not None
                        else torch.ones(pred_joints.shape[:3], device=device))
            cam_intr = batch["cam_intrinsics"].to(device)          # [1,3] @ native frame
            cam_extr = clip["cam_extrinsics"]                      # [S,4,4] T_cam_world

            # metric scale from the hand (the coupling's metric); skip clip if no hand
            _, info = hand_depth_anchor_loss(pred_joints, preds["gs_depth"], has_hand, cam_intr)
            if info["n_valid"] == 0:
                continue
            s = float(solve_metric_scale(pred_joints, preds["gs_depth"], has_hand, cam_intr))

            # K scaled from the native cam_intr frame (~2*cx) to the gs_depth grid
            f0, cx0, cy0 = [float(x) for x in cam_intr[0].tolist()]
            scale = Wd / (2.0 * cx0)
            Kf = (f0 * scale, cx0 * scale, cy0 * scale)

            hand_mask = _hand_pixel_mask(pred_joints, cam_intr, S, Hd, Wd, device)
            clip_err, clip_npx = [], 0
            for sframe in range(S):
                objs = [(u, T) for (u, T) in frame_objects[frame_offset + sframe] if u in avail]
                if not objs:
                    continue
                T_cam_world = cam_extr[sframe].double().numpy()    # validated w2c
                od, om = render_object_depth(objs, T_cam_world, Kf, Hd, Wd, args.objects_dir, device=device)
                region = om & (~hand_mask[sframe]) & (gs[sframe] > 1e-3)
                if not region.any():
                    continue
                e = (gs[sframe][region] * s - od[region]).abs()
                clip_err.append(e)
                clip_npx += int(region.sum())
            if not clip_err:
                continue
            ce = torch.cat(clip_err)
            errs.append(ce.cpu())
            deltas.append((ce < DELTA).float().mean().item())
            clip_scales.append(s)
            n_obj_px += clip_npx
            if len(clip_scales) % 10 == 0:
                print(f"  ...{len(clip_scales)} clips | running obj-err median="
                      f"{torch.cat(errs).median().item()*100:.2f}cm", flush=True)

    if not errs:
        raise SystemExit("No object-region pixels evaluated (no visible objects/meshes?).")

    allp = torch.cat(errs).numpy()
    s_arr = np.array(clip_scales)
    cv = float(s_arr.std() / max(abs(np.median(s_arr)), 1e-6))
    summary = {
        "checkpoint": args.checkpoint,
        "clips": len(clip_scales),
        "object_pixels": int(n_obj_px),
        "obj_depth_err_mean_cm": float(allp.mean() * 100),
        "obj_depth_err_median_cm": float(np.median(allp) * 100),
        "delta_under_10cm_pct": float(100 * np.mean([d for d in deltas])),
        "hand_scale_median": float(np.median(s_arr)),
        "hand_scale_CV_pct": float(cv * 100),
    }
    print("\n========== B2: NON-circular scene-metric (object regions) ==========")
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print("\nReading: anchored model should have LOWER obj_depth_err + HIGHER "
          "delta_under_10cm than the no-anchor baseline -> the hand anchor made "
          "the SCENE (not just the hand) metric.")

    if args.out:
        try:
            import json
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved -> {args.out}")
        except Exception as e:
            # disk/inode quota etc. must NOT fail the run — the summary already printed above.
            print(f"\n[warn] could not save {args.out}: {e} (summary printed above)")


if __name__ == "__main__":
    main()
