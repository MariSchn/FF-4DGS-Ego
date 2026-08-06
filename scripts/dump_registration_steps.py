"""Dump the raw arrays behind each of the three registration steps, for the paper figure.

Supervisor request (2026-08-06): "can we have some visual intermediate results for the
registration steps?" The registration panel in Fig. 1 names three stages but shows none of
them, so a reader has to take the scale solve on faith. This dumps exactly what each stage
consumes and produces so the figure is made from the real forward pass rather than drawn by hand.

  R1  sample scene depth      -> rgb, full-res gs_depth, projected joint pixels, and the
                                 depth read at each of those pixels
  R2  solve one scale s       -> the full per-joint ratio population z_hand / d_scene, plus
                                 the median and the MAD gate
  R3  scale camera translation-> the up-to-scale c2w trajectory and the same trajectory with
                                 translation multiplied by s, so the effect is visible

WHY THIS FILE OWNS NO FORWARD PASS. Its first version built the model, ran the model, and
re-derived the projection itself. That duplicate path drifted from scripts/eval_world_space.py
in two ways nobody noticed until the output was inspected: it omitted the bfloat16 autocast and
it never passed cond_flags. The result was a constant hand depth of -0.0205 m on all three
sequences, i.e. every hand behind the camera, which would have put a garbage panel in front of
the supervisor. The figure's whole purpose is to show what the EVAL does, so the arrays now come
out of eval_world_space's own build_model + predict_clip via the steps_out hook, and this file
only selects a clip and saves. If the two ever disagree again it is because the eval changed,
which is the correct coupling.

Plotting is deliberately NOT done here: this runs on the GPU node, the plotting runs locally off
the .pt so the figure can be iterated without burning GPU.

Usage (student cluster, venv_gb10):
    python -m scripts.dump_registration_steps \
        --config /tmp/fix59.yaml --data_root /work/scratch/dmonopoli/hoi4d_test157 \
        --out /home/dmonopoli/results/registration_steps.pt
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

from scripts.eval_world_space import build_model, predict_clip
from scripts.hand_vis_utils import MANOModel
from scripts.train_hand_head import (
    HOT3DHandDataset,
    build_views,
    discover_sequences,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None, help="overrides data.data_root")
    ap.add_argument("--seq_index", type=int, default=0, help="which discovered sequence to use")
    ap.add_argument("--n_clips", type=int, default=6, help="clips to scan for a usable one")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg, model_cfg, vis_cfg = cfg["data"], cfg["model"], cfg.get("visualization", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_cfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true: without it there is no gs_depth and no scale solve.")

    # Same builder the eval uses, so the trained-checkpoint guard and the warm-start fallback
    # both apply here too.
    model = build_model(cfg, device)
    model.gs_anchor_only = True
    mano_model = MANOModel(vis_cfg["mano_model_folder"])

    num_frames = data_cfg["num_frames"]
    root = args.data_root or data_cfg["data_root"]
    seqs = discover_sequences(root)
    if not seqs:
        raise SystemExit(f"no sequences under {root}")
    seq = seqs[args.seq_index]
    print(f"sequence: {seq}", flush=True)

    # Index the dataset DIRECTLY and unsqueeze, exactly as the eval does. An earlier version
    # wrapped this in a DataLoader(num_workers=0) and the job blocked indefinitely on the first
    # batch with zero CPU time, so the proven access pattern is used verbatim.
    ds = HOT3DHandDataset([seq], mano_model, num_frames=num_frames,
                          clip_stride=data_cfg.get("clip_stride", num_frames),
                          use_hand_crop=model_cfg.get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5))
    if len(ds) == 0:
        raise SystemExit(f"dataset empty for {seq}")
    print(f"clips available: {len(ds)}", flush=True)

    panel = None
    with torch.no_grad():
        for i in range(min(args.n_clips, len(ds))):
            batch = ds[i]
            if "cam_intrinsics" not in batch:
                print(f"  [clip {i}] no cam_intrinsics, skipping", flush=True)
                continue
            imgs = batch["img"].unsqueeze(0).to(device)
            hb = batch["hand_bboxes"].unsqueeze(0).to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].unsqueeze(0).to(device) if "hand_valid" in batch else None
            views = build_views(imgs, num_frames, device, hb, hv)
            # cond_flags and the bf16 autocast are BOTH part of the eval's forward. Omitting
            # either is what broke the first version of this script.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(views, cond_flags=[0, 0, 0], is_inference=True, use_motion=False)

            steps: dict = {}
            cam_intr = batch["cam_intrinsics"].view(1, 3)
            predict_clip(preds, mano_model, device, cam_intr, model=model, steps_out=steps)

            n_valid = int(steps["valid"].sum()) if "valid" in steps else 0
            frac_bad = steps.get("frac_nonpos_z_of_valid", float("nan"))
            print(f"  [clip {i}] valid={n_valid} s={steps.get('s', float('nan')):.4f} "
                  f"raw={steps.get('s_raw', float('nan')):.4f} "
                  f"failed={steps.get('s_failed')} frac_z<=0={frac_bad:.3f}", flush=True)
            if n_valid == 0:
                continue
            # Prefer a clip whose solve did NOT hit the clamp: the figure is meant to show the
            # method working, and a clamped clip is a failure case that belongs in the appendix
            # discussion of task #63, not in the method figure.
            if panel is None or (panel.get("s_failed") and not steps.get("s_failed")):
                steps["rgb"] = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                steps["cam_intr"] = cam_intr.float().cpu()
                steps["clip_index"] = i
                panel = steps
                if not steps.get("s_failed"):
                    break

    if panel is None:
        raise SystemExit("no usable clip")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save({"panel": panel, "seq": seq}, args.out)
    print(f"\nchosen clip {panel['clip_index']}: s={panel['s']:.4f} "
          f"(raw {panel['s_raw']:.4f}, clamped={panel['s_failed']})")
    print(f"hand_z median  = {float(panel['hand_z'][panel['valid']].median()):.4f} m")
    print(f"scene@hand med = {float(panel['scene_at_hand'][panel['valid']].median()):.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
