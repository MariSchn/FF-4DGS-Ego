"""Dump the raw arrays behind each of the three registration steps, for the paper figure.

Supervisor request (2026-08-06): "can we have some visual intermediate results for the
registration steps?" The registration panel in Fig. 1 names three stages but shows none of
them, so a reader has to take the scale solve on faith. This dumps exactly what each stage
consumes and produces, on ONE clip, so the figure is made from the real forward pass rather
than drawn by hand.

  R1  sample scene depth      -> rgb, full-res gs_depth, projected joint pixels, and the
                                 depth read at each of those pixels
  R2  solve one scale s       -> the full per-joint ratio population z_hand / d_scene, plus
                                 the median and the MAD gate that eval_world_space applies
  R3  scale camera translation-> the up-to-scale c2w trajectory and the same trajectory with
                                 translation multiplied by s, so the effect is visible

Plotting is deliberately NOT done here: this runs on the GPU node, the plotting runs
locally off the .pt so the figure can be iterated without burning GPU.

Usage (student cluster, venv_gb10):
    python -m scripts.dump_registration_steps \
        --config /tmp/fix59.yaml --data_root /work/scratch/dmonopoli/hoi4d_test157 \
        --out /home/dmonopoli/results/registration_steps.pt
"""
from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    frame_width_from_intr,
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)
from scripts.hand_vis_utils import MANOModel
from scripts.train_hand_head import (
    HOT3DHandDataset,
    build_views,
    compute_joints_from_batch,
    discover_sequences,
)

DEPTH_MIN = 0.01   # metres, matches eval_world_space's gate exactly
CLAMP = (0.1, 10.0)


def _load_trained_weights(model: WorldMirror, path: str, device: str) -> None:
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    is_full = any(k.startswith(("hand_head.", "gs_head.")) for k in sd)
    tgt = model if is_full else model.hand_head
    missing, unexpected = tgt.load_state_dict(sd, strict=False)
    print(f"loaded {'full' if is_full else 'hand-head'} ckpt (missing={len(missing)}, "
          f"unexpected={len(unexpected)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None, help="overrides data.data_root")
    ap.add_argument("--seq_index", type=int, default=0, help="which discovered sequence to use")
    ap.add_argument("--n_clips", type=int, default=6, help="clips to accumulate for the ratio histogram")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg, model_cfg, vis_cfg = cfg["data"], cfg["model"], cfg.get("visualization", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_cfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true: without it there is no gs_depth and no scale solve.")

    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    model.load_state_dict(base.get("state_dict", base.get("reconstructor", base)), strict=False)
    if model_cfg.get("warm_start_hand_head"):
        _load_trained_weights(model, model_cfg["warm_start_hand_head"], device)
    model.to(device).eval()
    model.gs_anchor_only = True

    mano_model = MANOModel(vis_cfg["mano_model_folder"])
    num_frames, res = data_cfg["num_frames"], tuple(data_cfg["resolution"])
    root = args.data_root or data_cfg["data_root"]
    seqs = discover_sequences(root)
    if not seqs:
        raise SystemExit(f"no sequences under {root}")
    seq = seqs[args.seq_index]
    print(f"sequence: {seq}")

    ds = HOT3DHandDataset([seq], mano_model, num_frames=num_frames, res=res,
                          clip_stride=data_cfg.get("clip_stride", num_frames),
                          use_hand_crop=model_cfg.get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 2.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    panel, all_ratios = None, []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_clips:
                break
            if "cam_intrinsics" not in batch:
                continue
            imgs = batch["img"].to(device)
            hb = batch["hand_bboxes"].to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].to(device) if "hand_valid" in batch else None
            preds = model(build_views(imgs, num_frames, device, hb, hv),
                          is_inference=False, use_motion=False)

            gs_depth = preds.get("gs_depth")
            if gs_depth is None:
                raise SystemExit("no gs_depth in preds")
            cam_intr = batch["cam_intrinsics"].to(device)
            pj = compute_joints_from_batch(preds["hand_joints"], mano_model, device)

            grid_xy, z = project_joints_to_norm_pixels(pj, cam_intr)
            sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy)
            valid = in_frame & (sampled > DEPTH_MIN) & torch.isfinite(z) & torch.isfinite(sampled)
            if not bool(valid.any()):
                continue
            all_ratios.append((z / sampled)[valid].float().cpu())

            if panel is None:   # R1 + R3 come from the FIRST usable clip only
                c2w_raw = preds.get("rendered_extrinsics")
                if c2w_raw is None:
                    raise SystemExit("no rendered_extrinsics: world lift would be identity, "
                                     "the R3 panel would be meaningless")
                d = gs_depth[0].float()
                while d.dim() > 3:
                    d = d.squeeze(1)                          # [S,Hd,Wd]
                panel = {
                    "rgb": (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu(),  # [S,3,H,W]
                    "gs_depth": d.half().cpu(),                                        # [S,Hd,Wd]
                    "grid_xy": grid_xy[0].float().cpu(),      # [S,2,J,2] normalised, y-x per the helper
                    "hand_z": z[0].float().cpu(),             # [S,2,J] metric hand depth
                    "scene_at_hand": sampled[0].float().cpu(),# [S,2,J] predicted scene depth there
                    "valid": valid[0].cpu(),
                    "c2w": c2w_raw[0].float().cpu(),          # [S,4,4] UP TO SCALE
                    "cam_intr": cam_intr[0].float().cpu(),
                    "frame_width": float(frame_width_from_intr(cam_intr)[0]),
                }
            print(f"clip {i}: {int(valid.sum())} valid joint samples", flush=True)

    if panel is None:
        raise SystemExit("no usable clip")

    r = torch.cat(all_ratios)
    med = r.median()
    mad = (r - med).abs().median().clamp_min(1e-6)
    keep = (r - med).abs() <= 3.0 * 1.4826 * mad          # the same robust gate the eval uses
    s = float(r[keep].median().clamp(*CLAMP))

    torch.save({"panel": panel, "ratios": r, "ratios_kept": keep,
                "s": s, "s_raw_median": float(med), "mad": float(mad),
                "seq": seq, "n_clips": len(all_ratios)}, args.out)
    print(f"\nratios n={r.numel()}  raw median={float(med):.4f}  "
          f"robust s={s:.4f}  kept={int(keep.sum())}/{r.numel()}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
