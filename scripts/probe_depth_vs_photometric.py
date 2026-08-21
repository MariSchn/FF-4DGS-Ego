#!/usr/bin/env python3
"""Does the photometric loss reward pushing the Gaussian depth to infinity?

The Gaussian depth of a photometrically trained head sits at exp(20) in every pixel. Two
explanations survive: the loss surface actively rewards large depth (a degenerate optimum), or
nothing constrains the depth and it drifted for free. They differ in a curve this script measures.

Forward one clip once, then rescale `gs_depth` by a range of factors and re-render at each. A
monotone L1 improvement toward large depth confirms the degenerate optimum. A flat curve says the
loss never depended on depth at all.

Training builds its views with identity poses and identity intrinsics
(train_hand_head.py:1262-1263), under which unprojecting and reprojecting through the same camera
returns the original pixel for any depth, so the rendered geometry can only depend on depth through
the projected splat footprint. This probe reproduces those training-time views by default.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from scripts.eval_hand_head import build_model, load_hand_head
from scripts.gs_metrics import render_views_from_predictions
from scripts.run_ours_gs import load_clip
from scripts.train_hand_head import build_views

BOX_FILE = "hand_bboxes_v2_rf1.5_res224x224.pt"
FACTORS = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--export_root", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--couple_scales", action="store_true",
                    help="scale splat scales with the positions, i.e. a pure gauge change")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg, a.device)
    if a.ckpt:
        load_hand_head(model, a.ckpt, a.device)
    model.eval()

    imgs, frames = load_clip(os.path.join(a.export_root, a.seq, "images"), a.n_views)
    imgs = imgs.unsqueeze(0).to(a.device)
    n = imgs.shape[1]
    bb = torch.load(os.path.join(a.store, a.seq, "hand_data", BOX_FILE), map_location="cpu")
    fi = torch.tensor(frames, dtype=torch.long)
    hb = bb["bboxes"][fi].unsqueeze(0).to(a.device)
    hv = bb["valid"][fi].bool().unsqueeze(0).to(a.device)
    views = build_views(imgs, n, a.device, hb, hv, frame_index=fi.unsqueeze(0))

    with torch.no_grad():
        preds = model(views, is_inference=False, use_motion=False)
    d0 = preds["gs_depth"]
    print(f"gs_depth shape {tuple(d0.shape)}  median {float(d0.median()):.6g}  "
          f"min {float(d0.min()):.6g}  max {float(d0.max()):.6g}", flush=True)
    print(f"fraction at exp(20): {float((d0 >= 4.8516e8).float().mean()):.4f}", flush=True)

    # The rasterizer reads preds["splats"], which the forward already built from gs_depth
    # (rasterization.py:721-738). Rewriting preds["gs_depth"] after the forward changes nothing:
    # the first version of this probe did exactly that and reported an identical render, to the
    # last digit, across twenty decades. Scale the splat positions themselves instead.
    splats = preds["splats"]
    orig = [[(sp.means.clone(), sp.scales.clone()) for sp in per_b] for per_b in splats]
    sc0 = torch.cat([sp.scales.reshape(-1) for per_b in splats for sp in per_b])
    print(f"splat scales: median {float(sc0.median()):.5g}  max {float(sc0.max()):.5g}", flush=True)

    H, W = imgs.shape[-2], imgs.shape[-1]
    rows = []
    for k in FACTORS:
        for per_b, per_b0 in zip(splats, orig):
            for sp, (m0, s0) in zip(per_b, per_b0):
                sp.means = m0 * k          # (u*d, v*d, d) scales as a whole under identity K
                if a.couple_scales:
                    # Scaling positions and extents together is a similarity transform of the whole
                    # cloud, so the projected footprint is unchanged. If the render survives this
                    # but not the uncoupled sweep, the model is intact up to an overall gauge and
                    # only its absolute scale is wrong.
                    sp.scales = s0 * k
        with torch.no_grad():
            r = render_views_from_predictions(model, preds, views, H, W)
        gt = imgs[0].permute(0, 2, 3, 1)                       # [S,H,W,3] to match the render
        l1 = float((r[0] - gt).abs().mean())
        mse = float(((r[0] - gt) ** 2).mean())
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / max(mse, 1e-12))
        zmed = float(torch.cat([sp.means[:, 2] for per_b in splats for sp in per_b]).median())
        rows.append({"factor": k, "median_depth": zmed,
                     "l1": l1, "psnr": psnr, "render_mean": float(r.mean())})
        print(f"  x{k:<8g} splat_z={rows[-1]['median_depth']:<12.5g} "
              f"L1={l1:.5f}  PSNR={psnr:6.2f}  render_mean={rows[-1]['render_mean']:.4f}", flush=True)

    means = sorted({round(r["render_mean"], 9) for r in rows})
    if len(means) == 1:
        raise SystemExit(
            f"render_mean is identical ({means[0]}) at every factor, so the rescale never reached "
            "the rasterizer and this probe measured nothing. Do not read the numbers above.")

    best = min(rows, key=lambda r: r["l1"])
    mono = all(rows[i]["l1"] >= rows[i + 1]["l1"] for i in range(len(rows) - 1))
    spread = max(r["l1"] for r in rows) - min(r["l1"] for r in rows)
    print(f"\nbest L1 at factor {best['factor']:g} (depth {best['median_depth']:.5g})", flush=True)
    print(f"monotone improving with depth: {mono}   L1 spread across 12 decades: {spread:.5f}",
          flush=True)
    print("VERDICT: " + ("loss REWARDS large depth -> degenerate optimum" if best["factor"] >= 1e2
                        else "loss does NOT reward large depth -> free drift"), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"seq": a.seq, "ckpt": a.ckpt, "config": a.config, "n_views": n,
                   "gs_depth_median_raw": float(d0.median()),
                   "frac_at_clamp": float((d0 >= 4.8516e8).float().mean()),
                   "rows": rows, "monotone": mono, "l1_spread": spread}, fh, indent=2)


if __name__ == "__main__":
    main()
