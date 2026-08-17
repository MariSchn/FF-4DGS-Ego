#!/usr/bin/env python3
"""Run MoVieS's released checkpoint on our exported clips and keep renders we can score.

Their infer_davis_nvs.py renders two things and neither is the input video: every timestep from
camera 0, and every camera at the last timestep. Scoring needs the diagonal, camera i at timestep i,
because that is the only combination for which a ground-truth frame exists.

    python run_movies.py --ckpt <file> --npz_root <dir> --out <dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch


def render_diagonal(model, images, c2w, fxfycxcy, device):
    """Return (F, H, W, 3) uint8 RGB, frame i seen from camera i at timestep i."""
    n = images.shape[1]
    steps = torch.linspace(0, 1, steps=n).unsqueeze(0)

    def bf(x):
        return x.to(device=device, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_outputs, pred_motions, pred_motion_gs = model.backbone(
            bf(images), bf(c2w), bf(fxfycxcy), bf(steps), bf(steps), frames_chunk_size=16)

        frames = []
        for i in range(n):
            if pred_motions is not None:
                backbone_outputs["offset"] = pred_motions[:, i, :, :3, ...]
            if pred_motion_gs is not None:
                backbone_outputs.update(pred_motion_gs[i])
            out = model.gs_renderer.render(
                backbone_outputs, bf(c2w), bf(fxfycxcy),
                bf(c2w[:, i:i + 1, ...]), bf(fxfycxcy[:, i:i + 1, ...]))
            frames.append(out["image"][0, 0].float().clamp(0, 1).cpu())

    arr = torch.stack(frames).permute(0, 2, 3, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/cluster/scratch/dmonopoli/MoVieS")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz_root", required=True, help="<root>/<seq>.npz from ours_store_to_movies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.chdir(a.repo)
    sys.path.insert(0, a.repo)
    sys.path.insert(0, os.path.join(a.repo, "extensions", "vggt"))
    from safetensors.torch import load_file
    from src.models import SplatRecon
    from src.options import opt_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SplatRecon(opt_dict["movies"], load_lpips=False)
    model.load_state_dict(load_file(a.ckpt), strict=True)
    model = model.eval().to(device)

    names = sorted(f[:-4] for f in os.listdir(a.npz_root) if f.endswith(".npz"))
    if a.limit:
        names = names[:a.limit]
    if not names:
        raise SystemExit(f"no npz under {a.npz_root}")

    os.makedirs(a.out, exist_ok=True)
    report, ok, fail = [], 0, 0
    for name in names:
        try:
            npz = np.load(os.path.join(a.npz_root, name + ".npz"))
            if "idx" not in npz.files:
                raise KeyError("npz has no idx, re-export with ours_store_to_movies")
            idx = npz["idx"]
            images = torch.from_numpy(npz["images"]).float().unsqueeze(0)
            rgb = render_diagonal(model, images,
                                  torch.from_numpy(npz["C2W"]).float().unsqueeze(0),
                                  torch.from_numpy(npz["fxfycxcy"]).float().unsqueeze(0),
                                  device)
            if len(rgb) != len(idx):
                raise RuntimeError(f"rendered {len(rgb)} frames for {len(idx)} indices")

            rdir = os.path.join(a.out, name, "renders")
            os.makedirs(rdir, exist_ok=True)
            for k, frame in zip(idx, rgb):
                cv2.imwrite(os.path.join(rdir, f"{int(k):06d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            ok += 1
            report.append({"seq": name, "frames": len(idx), "hw": list(rgb.shape[1:3])})
            print(f"OK {name}  {len(idx)} frames  {rgb.shape[1]}x{rgb.shape[2]}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad sequence must not stop a set
            fail += 1
            report.append({"seq": name, "error": f"{type(e).__name__}: {e}"})
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)

    with open(os.path.join(a.out, "_movies_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"MOVIES_RENDER done ok={ok} fail={fail} -> {a.out}", flush=True)
    if not ok:
        raise SystemExit("no sequence rendered")


if __name__ == "__main__":
    main()
