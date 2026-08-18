#!/usr/bin/env python3
"""Score a baseline's rendered frames against ground truth, whole-frame and over the hand region.

This only loads pixels and builds the region mask. Every metric comes from `scripts.gs_metrics`,
the same module the training-time validation uses, so a baseline row and our own row are produced
by one implementation. Rewriting PSNR/SSIM/LPIPS here would make the two sets of numbers
incomparable in exactly the way this table exists to avoid.

The hand region is the union of the store's own per-sequence boxes, the file
`hand_bboxes_v2_rf1.5_res224x224.pt`, which is what our own world evaluation reads too, so every
row here and our row in the pose tables share one region definition. It is NOT the separate
`hoi4d_detboxes_v3` store injected into the baseline pose pipelines: those boxes differ, they run
about 3-6% wider and carry a `det_hit` flag this file has no equivalent of. The values are
normalized [cx, cy, w, h] per hand and are NOT clamped to the frame, so they are clipped here.

    python -m scripts.score_gs_renders --renders <root> --gt_root <gs_export/4dgt> \
        --store <hoi4d_test157_detv3> --method anysplat --out results/gs_anysplat.json
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from scripts.gs_metrics import (LPIPSScorer, metric_chunks_from_batch,
                                metrics_from_chunks, region_metric_chunks_from_batch)


def _load_png(path: str) -> torch.Tensor:
    """[3,H,W] float in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable frame {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def hand_mask(boxes: torch.Tensor, valid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Union of the valid hand boxes for one frame, as an [H,W] bool mask."""
    m = torch.zeros(h, w, dtype=torch.bool)
    for hand in range(boxes.shape[0]):
        if not bool(valid[hand]):
            continue
        cx, cy, bw, bh = [float(v) for v in boxes[hand]]
        x0 = max(0, int(round((cx - bw / 2) * w)))
        x1 = min(w, int(round((cx + bw / 2) * w)))
        y0 = max(0, int(round((cy - bh / 2) * h)))
        y1 = min(h, int(round((cy + bh / 2) * h)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def score_sequence(rdir, gdir, bpath, scorer, device):
    """Return (full_chunk, hand_chunk, n_frames) for one sequence, or None if nothing lines up."""
    bb = torch.load(bpath, map_location="cpu")
    boxes, valid = bb["bboxes"], bb["valid"].bool()

    preds, gts, masks = [], [], []
    for fn in sorted(f for f in os.listdir(rdir) if f.endswith(".png")):
        gpath = os.path.join(gdir, fn)
        if not os.path.exists(gpath):
            continue
        # The frame index is the filename, which the export and the render share, so a method that
        # renders a subset of frames still lines up with its own boxes.
        idx = int(os.path.splitext(fn)[0])
        if idx >= boxes.shape[0]:
            continue
        gt = _load_png(gpath)
        pred = _load_png(os.path.join(rdir, fn))
        if pred.shape != gt.shape:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]
        preds.append(pred)
        gts.append(gt)
        masks.append(hand_mask(boxes[idx], valid[idx], gt.shape[-2], gt.shape[-1]))

    if not preds:
        return None
    # gs_metrics expects rendered as [B,S,H,W,3] and gt as [B,S,3,H,W].
    rendered = torch.stack(preds).permute(0, 2, 3, 1).unsqueeze(0)
    gt_imgs = torch.stack(gts).unsqueeze(0)
    region = torch.stack(masks).unsqueeze(0)
    full = metric_chunks_from_batch(rendered, gt_imgs, None, scorer, device)
    hand = region_metric_chunks_from_batch(rendered, gt_imgs, region, scorer, device)
    return full, hand, len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="<root>/<seq>/renders/*.png")
    ap.add_argument("--gt_root", required=True, help="<root>/<seq>/images/*.png")
    ap.add_argument("--store", required=True, help="our store, for the detbox v3 boxes")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    scorer = LPIPSScorer(device=a.device)
    seqs = sorted(d for d in os.listdir(a.renders)
                  if os.path.isdir(os.path.join(a.renders, d, "renders")))
    if not seqs:
        raise SystemExit(f"no sequences with renders/ under {a.renders}")

    full_chunks, hand_chunks, per_seq = [], [], {}
    for s in seqs:
        rdir = os.path.join(a.renders, s, "renders")
        gdir = os.path.join(a.gt_root, s, "images")
        bpath = os.path.join(a.store, s, "hand_data", "hand_bboxes_v2_rf1.5_res224x224.pt")
        if not (os.path.isdir(gdir) and os.path.exists(bpath)):
            print(f"{s}: SKIP, missing ground truth or boxes", flush=True)
            continue
        got = score_sequence(rdir, gdir, bpath, scorer, a.device)
        if got is None:
            print(f"{s}: SKIP, no frame pairs", flush=True)
            continue
        full, hand, n = got
        full_chunks.append(full)
        hand_chunks.append(hand)
        per_seq[s] = {"frames": n,
                      "full": metrics_from_chunks([full]),
                      "hand": metrics_from_chunks([hand])}
        print(f"{s}: {n} frames  PSNR {per_seq[s]['full']['PSNR']:.2f}  "
              f"hand {per_seq[s]['hand']['PSNR']}", flush=True)

    agg = {"full": metrics_from_chunks(full_chunks), "hand": metrics_from_chunks(hand_chunks),
           "n_seqs": len(per_seq)}
    out = {"method": a.method, "aggregate": agg, "per_seq": per_seq,
           "protocol": {"renders": a.renders, "gt_root": a.gt_root, "store": a.store,
                        "metrics": "scripts.gs_metrics, the module the training validation uses",
                        "hand_region": "union of detbox v3 boxes, clipped to frame"}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)

    f_, h_ = agg["full"], agg["hand"]
    def fmt(d, k):
        v = d.get(k)
        return f"{v:.4f}" if isinstance(v, float) else "n/a"
    print(f"\n{a.method}: whole PSNR {fmt(f_,'PSNR')} SSIM {fmt(f_,'SSIM')} "
          f"LPIPS {fmt(f_,'LPIPS')} (N={f_['num_valid_frames']})")
    print(f"{' ' * len(a.method)}  hand  PSNR {fmt(h_,'PSNR')} SSIM {fmt(h_,'SSIM')} "
          f"LPIPS {fmt(h_,'LPIPS')} (N={h_['num_valid_frames']})")
    print(f"GS_SCORE_DONE {a.method} -> {a.out}")
    return 0 if per_seq else 2


if __name__ == "__main__":
    raise SystemExit(main())
