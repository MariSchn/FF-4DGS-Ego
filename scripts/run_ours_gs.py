#!/usr/bin/env python3
"""Render our Gaussians at the input viewpoints of the Gaussian-table sequences.

Produces the same PNG layout the baseline runners produce, so `scripts.score_gs_renders` scores our
row and theirs with one code path. `scripts.eval_gs_head` already re-renders input views, but it is
tied to a HOT3D split and reports its own aggregate, which would put our row and the baselines'
under two different frame selections.

Without --ckpt the base reconstructor runs untouched, which is the NeoVerse row. With it, the
trained hand head and the injection are loaded, which is ours.

    python -m scripts.run_ours_gs --config configs/exp_gsinj2_on.yaml \
        --export_root <gs_export/4dgt> --store <hoi4d_test157_hd> --out <dir> [--ckpt <pt>]
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch
import yaml

from scripts.eval_hand_head import build_model, load_hand_head
from scripts.gs_metrics import render_views_from_predictions
from scripts.train_hand_head import build_views

BOX_FILE = "hand_bboxes_v2_rf1.5_res224x224.pt"


def load_clip(img_dir: str, n_views: int):
    """Return (imgs [S,3,H,W] in [0,1], frame indices). Same uniform rule the baselines use."""
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    if not files:
        raise FileNotFoundError(img_dir)
    if len(files) > n_views:
        sel = np.linspace(0, len(files) - 1, n_views).round().astype(int)
        files = [files[i] for i in sel]
    imgs = []
    for f in files:
        bgr = cv2.imread(os.path.join(img_dir, f))
        if bgr is None:
            raise RuntimeError(f"unreadable {os.path.join(img_dir, f)}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(imgs), [int(os.path.splitext(f)[0]) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="must set model.enable_gs true")
    ap.add_argument("--ckpt", default=None, help="omit for the base reconstructor row")
    ap.add_argument("--export_root", required=True, help="<root>/<seq>/images/*.png")
    ap.add_argument("--store", required=True, help="our store, for the detbox v3 boxes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--static", action="store_true",
                    help="fuse the clip into one cloud, the regime the splatting baselines use")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if not cfg["model"].get("enable_gs"):
        raise SystemExit(f"{a.config} has enable_gs false, so the model emits no splats")

    model = build_model(cfg, a.device)
    if a.ckpt:
        load_hand_head(model, a.ckpt, a.device)
    model.eval()

    seqs = sorted(d for d in os.listdir(a.export_root)
                  if os.path.isdir(os.path.join(a.export_root, d, "images")))
    if a.limit:
        seqs = seqs[:a.limit]
    if not seqs:
        raise SystemExit(f"no sequences under {a.export_root}")

    os.makedirs(a.out, exist_ok=True)
    report, ok, fail = [], 0, 0
    for s in seqs:
        try:
            imgs, frames = load_clip(os.path.join(a.export_root, s, "images"), a.n_views)
            imgs = imgs.unsqueeze(0).to(a.device)
            n = imgs.shape[1]

            bb = torch.load(os.path.join(a.store, s, "hand_data", BOX_FILE), map_location="cpu")
            fi = torch.tensor(frames, dtype=torch.long)
            hb = bb["bboxes"][fi].unsqueeze(0).to(a.device)
            hv = bb["valid"][fi].bool().unsqueeze(0).to(a.device)

            views = build_views(imgs, n, a.device, hb, hv, frame_index=fi.unsqueeze(0))
            if a.static:
                # One fused cloud at timestamp -1 instead of one set per frame. Without it the
                # rasterizer draws a Gaussian only into the view whose timestamp matches
                # (`rasterization.py:321`), so every frame is rendered from its own unprojection
                # and the score is close to a copy of the input. AnySplat builds a single scene
                # from all views and renders them from it, so only this setting compares like
                # with like.
                views["is_static"] = torch.ones_like(views["is_static"])
            with torch.no_grad():
                preds = model(views, is_inference=False, use_motion=False)
                rendered = render_views_from_predictions(
                    model, preds, views, imgs.shape[-2], imgs.shape[-1])

            rdir = os.path.join(a.out, s, "renders")
            os.makedirs(rdir, exist_ok=True)
            arr = (rendered[0].clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
            for k, frame in zip(frames, arr):
                cv2.imwrite(os.path.join(rdir, f"{k:06d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            ok += 1
            report.append({"seq": s, "frames": len(frames), "hw": list(arr.shape[1:3])})
            print(f"OK {s}  {len(frames)} frames  {arr.shape[1]}x{arr.shape[2]}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad sequence must not stop a set
            fail += 1
            report.append({"seq": s, "error": f"{type(e).__name__}: {e}"})
            print(f"FAIL {s}: {type(e).__name__}: {e}", flush=True)

    with open(os.path.join(a.out, "_ours_gs_report.json"), "w") as fh:
        json.dump({"ckpt": a.ckpt, "config": a.config, "seqs": report}, fh, indent=2)
    print(f"OURS_GS_RENDER done ok={ok} fail={fail} -> {a.out}", flush=True)
    if not ok:
        raise SystemExit("no sequence rendered")


if __name__ == "__main__":
    main()
