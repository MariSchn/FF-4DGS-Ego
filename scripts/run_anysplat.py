#!/usr/bin/env python3
"""Run AnySplat's released checkpoint on our exported sequences and keep the rendered frames.

Their inference.py cannot be used as is for two reasons. It calls from_pretrained on a hub id,
and Euler compute nodes have no internet, so the checkpoint has to come from the local directory.
And it writes an interpolated fly-through video, which no metric can be computed against. We need
renders at the input viewpoints so PSNR, SSIM and LPIPS can be scored against the real frames.

Input is the images/ directory of the 4DGT export, which is plain PNG and is the same pixel data
every other row in the table receives.

    python run_anysplat.py --ckpt <dir> --export_root <dir> --out <dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/cluster/scratch/dmonopoli/AnySplat")
    ap.add_argument("--ckpt", required=True, help="local HF-format directory")
    ap.add_argument("--export_root", required=True, help="<root>/<seq>/images/*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_views", type=int, default=32,
                    help="frames per sequence; AnySplat is a static reconstructor and its cost "
                         "grows with view count, so a whole 300-frame clip does not fit")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from src.model.model.anysplat import AnySplat
    from src.utils.image import process_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AnySplat.from_pretrained(a.ckpt).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"loaded AnySplat from {a.ckpt} on {device}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    seqs = sorted(d for d in os.listdir(a.export_root)
                  if os.path.isdir(os.path.join(a.export_root, d, "images")))
    if a.limit:
        seqs = seqs[:a.limit]

    report, ok, fail = {}, 0, 0
    for s in seqs:
        idir = os.path.join(a.export_root, s, "images")
        files = sorted(f for f in os.listdir(idir) if f.lower().endswith((".png", ".jpg")))
        # Uniform subsample rather than the first N, so the views span the whole clip and the
        # reconstruction is not asked to cover the scene from one end of the trajectory.
        if len(files) > a.max_views:
            idx = np.linspace(0, len(files) - 1, a.max_views).round().astype(int)
            files = [files[i] for i in idx]
        try:
            imgs = torch.stack([process_image(os.path.join(idir, f)) for f in files], 0)
            imgs = imgs.unsqueeze(0).to(device)
            with torch.no_grad():
                gaussians, pose = model.inference((imgs + 1) * 0.5)
            od = os.path.join(a.out, s)
            os.makedirs(od, exist_ok=True)

            # Render at the INPUT viewpoints, through AnySplat's own decoder rather than a
            # reimplementation, so the row is scored on what their model actually produces. The
            # near and far planes and the colour clip match their save_interpolated_video.
            ext, intr = pose["extrinsic"], pose["intrinsic"]
            nv = ext.shape[1]
            hh, ww = imgs.shape[-2], imgs.shape[-1]
            with torch.no_grad():
                dec = model.decoder.forward(
                    gaussians, ext, intr.float(),
                    torch.ones(1, nv, device=device) * 0.1,
                    torch.ones(1, nv, device=device) * 100.0,
                    (hh, ww),
                )
            color = dec.color[0].clamp(0, 1).cpu()
            rdir = os.path.join(od, "renders")
            os.makedirs(rdir, exist_ok=True)
            for i, fn in enumerate(files):
                arr = (color[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                cv2.imwrite(os.path.join(rdir, fn), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

            torch.save({"extrinsic": pose["extrinsic"].cpu(),
                        "intrinsic": pose["intrinsic"].cpu(),
                        "means": gaussians.means[0].cpu(),
                        "scales": gaussians.scales[0].cpu(),
                        "rotations": gaussians.rotations[0].cpu(),
                        "harmonics": gaussians.harmonics[0].cpu(),
                        "opacities": gaussians.opacities[0].cpu(),
                        "frames": files},
                       os.path.join(od, "anysplat.pt"))
            report[s] = {"views": len(files), "gaussians": int(gaussians.means.shape[1])}
            ok += 1
            print(f"{s}: {len(files)} views -> {gaussians.means.shape[1]} gaussians", flush=True)
        except Exception as e:
            fail += 1
            report[s] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{s}: FAILED {type(e).__name__}: {e}", flush=True)

    with open(os.path.join(a.out, "_anysplat_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"ANYSPLAT_DONE ok={ok} fail={fail} out={a.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
