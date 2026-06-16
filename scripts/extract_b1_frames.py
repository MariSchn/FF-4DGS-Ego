"""B1 stage 1 (gb10 / venv_gb10): dump per-clip frames + hand anchor targets.

This is the *extract* half of the "why not just a metric-depth foundation model?"
baseline (publication-plan E1). It runs our warm-start WorldMirror once per clip to
produce the anchored scene depth (``gs_depth``) and the metric MANO hand joints,
then saves a portable, architecture-agnostic ``.pt`` bundle that the x86 UniDepth
eval (``scripts/eval_unidepth_b1.py``) reads on a different node/venv.

For every clip we store, per frame:
  * ``rgb``        : [S, 3, H, W] the input image UniDepth will run on.
  * ``K``          : [3, 3] pinhole intrinsics at that resolution (UniDepth input).
  * ``grid_xy``    : [S, 2, J, 2] normalised [0, 1] hand-joint pixel locations.
  * ``hand_z``     : [S, 2, J] metric hand-joint depth (the anchor target, metres).
  * ``gs_at_hand`` : [S, 2, J] our anchored scene depth sampled at the hand.
  * ``valid``      : [S, 2, J] bool — visible hand, in-frame, positive depth.

The comparison the eval makes is apples-to-apples with our headline 4.5 cm number:
"given the trusted-metric hand, how far off is each depth source AT the hand?"
``gs_at_hand`` reproduces our anchor residual; UniDepth's depth sampled at the same
``grid_xy`` gives the FM's residual.

Usage (gb10 node, inside venv_gb10):
    python -m scripts.extract_b1_frames \
        --config configs/exp_p2_pinhole_warmstart.yaml \
        --checkpoint checkpoints/p2_warmstart/best_mpjpe.pt \
        --num_clips 60 --out outputs/b1_bundle.pt
"""
from __future__ import annotations

import argparse
import random

import torch
import yaml
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
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

DEPTH_MIN = 0.05  # metres; reject degenerate near-camera samples


def _load_trained_weights(model: WorldMirror, path: str, device: str) -> None:
    """Load a checkpoint, auto-detecting full-model vs hand-head-only (mirrors eval_metric_scale)."""
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    is_full = any(k.startswith(("hand_head.", "gs_head.")) for k in sd.keys())
    if is_full:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded full-model checkpoint '{path}' (missing={len(missing)}, unexpected={len(unexpected)}).")
    else:
        missing, unexpected = model.hand_head.load_state_dict(sd, strict=False)
        print(f"Loaded hand-head-only checkpoint '{path}' (missing={len(missing)}, unexpected={len(unexpected)}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_p2_pinhole_warmstart.yaml")
    ap.add_argument("--checkpoint", required=True,
                    help="Our warm-start checkpoint (its gs_depth is the anchored scene depth).")
    ap.add_argument("--num_clips", type=int, default=60)
    ap.add_argument("--max_seqs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/b1_bundle.pt")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg, model_cfg, vis_cfg = cfg["data"], cfg["model"], cfg.get("visualization", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_cfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true in the config (we need gs_depth).")

    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    base_sd = base.get("state_dict", base.get("reconstructor", base))
    model.load_state_dict(base_sd, strict=False)
    _load_trained_weights(model, args.checkpoint, device)
    model.to(device).eval()
    model.gs_anchor_only = True  # fast path: gs_depth only, skip prune/render

    mano_model = MANOModel(vis_cfg["mano_model_folder"])

    num_frames = data_cfg["num_frames"]
    res = tuple(data_cfg["resolution"])
    seqs = discover_sequences(data_cfg["data_root"])
    if not seqs:
        raise SystemExit(f"No sequences under {data_cfg['data_root']}")
    random.seed(args.seed)
    random.shuffle(seqs)
    seqs = seqs[:args.max_seqs]
    print(f"Building dataset from {len(seqs)} sequence(s).")
    dataset = HOT3DHandDataset(
        seqs, mano_model,
        num_frames=num_frames, res=res,
        clip_stride=data_cfg.get("clip_stride", num_frames),
        use_hand_crop=model_cfg.get("use_hand_crop", False),
        rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 2.0),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    clips = []
    print(f"\nExtracting up to {args.num_clips} clips ({len(dataset.clips)} available) ...\n")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(clips) >= args.num_clips:
                break
            if "cam_intrinsics" not in batch:
                continue
            imgs = batch["img"].to(device)              # [1, S, 3, H, W]
            hb = batch["hand_bboxes"].to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].to(device) if "hand_valid" in batch else None
            views = build_views(imgs, num_frames, device, hb, hv)

            preds = model(views, is_inference=False, use_motion=False)
            gs_depth = preds.get("gs_depth")
            if gs_depth is None:
                raise SystemExit("Model produced no gs_depth — check enable_gs / the GS head.")
            pred_joints = compute_joints_from_batch(preds["hand_joints"], mano_model, device)
            has_hand = (hv.float() if hv is not None
                        else torch.ones(pred_joints.shape[:3], device=device))
            cam_intr = batch["cam_intrinsics"].to(device)   # [1, 3] = [f, cx, cy]

            grid_xy, z = project_joints_to_norm_pixels(pred_joints.detach(), cam_intr)
            sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy)
            has_hand_j = has_hand.unsqueeze(-1).to(torch.bool).expand_as(sampled)
            valid = has_hand_j & in_frame & (sampled > DEPTH_MIN) & (z > DEPTH_MIN)
            if int(valid.sum().item()) == 0:
                continue

            f = float(cam_intr[0, 0]); cx = float(cam_intr[0, 1]); cy = float(cam_intr[0, 2])
            K = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)

            clips.append({
                "rgb": imgs[0].detach().cpu().float(),        # [S, 3, H, W]
                "K": K,                                       # [3, 3]
                "grid_xy": grid_xy[0].detach().cpu().float(), # [S, 2, J, 2] in [0,1]
                "hand_z": z[0].detach().cpu().float(),        # [S, 2, J]
                "gs_at_hand": sampled[0].detach().cpu().float(),
                "valid": valid[0].detach().cpu().bool(),
            })
            n = int(valid.sum().item())
            res_cm = float(((sampled - z).abs() * valid.float()).sum() / valid.float().sum()) * 100
            print(f"clip {i:4d} | n_valid={n:4d} | gs_at_hand residual={res_cm:6.2f}cm", flush=True)

    if not clips:
        raise SystemExit("No valid clips extracted (no visible hands / no intrinsics).")

    meta = {"resolution": list(res), "num_frames": num_frames,
            "checkpoint": args.checkpoint, "n_clips": len(clips)}
    torch.save({"clips": clips, "meta": meta}, args.out)
    print(f"\nSaved {len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
