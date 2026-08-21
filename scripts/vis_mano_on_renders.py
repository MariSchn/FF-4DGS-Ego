#!/usr/bin/env python3
"""Overlay the predicted MANO mesh on our own Gaussian renders, using only predicted quantities.

Answers two separate questions with one forward pass:

  1. Does the Gaussian head reconstruct the hand at all? Panel B renders the predicted splats at
     the input viewpoint, so a hand-shaped region either is there or is not.
  2. Is the Gaussian depth sampled where we think it is? `project_joints_to_norm_pixels` applies a
     90-degree rotation because `gs_depth` is held in a rotated layout, and the same file records
     438 mm of error from sampling an unrotated map with the rotated grid. Panels C and D draw
     both grids on the depth map, and the per-convention median of z_hand / d_scene says which one
     is reading the hand.

No ground truth is loaded anywhere: camera, Gaussians, depth and MANO all come out of the model.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import yaml

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
    set_default_frame_width,
)
from scripts.eval_hand_head import build_model, load_hand_head
from scripts.gs_metrics import render_views_from_predictions
from scripts.hand_vis_utils import MANOModel, render_mesh_overlay
from scripts.run_ours_gs import load_clip
from scripts.train_hand_head import (build_views, compute_vertices_from_batch,
                                     _intr_to_render_frame, _video_wh)

BOX_FILE = "hand_bboxes_v2_rf1.5_res224x224.pt"
LEFT_COLOR = (255, 150, 50)     # BGR
RIGHT_COLOR = (50, 165, 255)


_T0 = time.time()


def stage(msg: str) -> None:
    print(f"[+{time.time() - _T0:7.1f}s] {msg}", flush=True)


def _to_bgr(img_chw_or_hwc: torch.Tensor) -> np.ndarray:
    a = img_chw_or_hwc.detach().float().cpu().numpy()
    if a.shape[0] == 3 and a.ndim == 3:
        a = a.transpose(1, 2, 0)
    return cv2.cvtColor((a.clip(0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _depth_to_bgr(d: torch.Tensor) -> np.ndarray:
    """Percentile-normalised colormap. Absolute values are meaningless up to scale anyway."""
    a = d.detach().float().cpu().numpy()
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    n = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="must set model.enable_gs true")
    ap.add_argument("--ckpt", default=None, help="omit for the untrained-head (frozen NeoVerse) arm")
    ap.add_argument("--export_root", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", required=True, nargs="+", help="one or more; the model loads once")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", type=int, default=32)
    ap.add_argument("--frames", type=int, default=6, help="how many of the clip's views to draw")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if not cfg["model"].get("enable_gs"):
        raise SystemExit(f"{a.config} has enable_gs false, so the model emits no splats")

    # HOI4D ships an off-centre principal point, so the 2*cx fallback overestimates the frame by
    # 2% and every sample lands off-pixel. train_hand_head and eval_world_space both declare it.
    res = cfg.get("data", {}).get("resolution")
    if not res:
        raise SystemExit("config has no data.resolution, so the projection frame width is unknown")
    res_i = int(res[0]) if isinstance(res, (list, tuple)) else int(res)
    set_default_frame_width(res_i)

    stage("building model")
    model = build_model(cfg, a.device)
    if a.ckpt:
        stage(f"loading ckpt {a.ckpt}")
        load_hand_head(model, a.ckpt, a.device)
    model.eval()
    stage("model ready; loading MANO")
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])
    faces = {0: mano.left.faces.astype(np.int32), 1: mano.right.faces.astype(np.int32)}

    for seq in a.seq:
        imgs, frames = load_clip(os.path.join(a.export_root, seq, "images"), a.n_views)
        imgs = imgs.unsqueeze(0).to(a.device)
        n = imgs.shape[1]
        sd = os.path.join(a.store, seq, "hand_data")
        bb = torch.load(os.path.join(sd, BOX_FILE), map_location="cpu")
        fi = torch.tensor(frames, dtype=torch.long)
        hb, hv = bb["bboxes"][fi].unsqueeze(0).to(a.device), bb["valid"][fi].bool().unsqueeze(0).to(a.device)
        cam_intr = torch.load(os.path.join(sd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)
        # The cache is at the source resolution; the frames are cover-and-centre-cropped.
        # A no-op on HOI4D and H2O, whose videos are already square at the render size.
        cam_intr = _intr_to_render_frame(
            cam_intr.view(3), _video_wh(os.path.join(a.store, seq, "video_main_rgb.mp4")),
            (res_i, res_i)).view(1, 3)

        stage(f"[{seq}] forward")
        views = build_views(imgs, n, a.device, hb, hv, frame_index=fi.unsqueeze(0))
        with torch.no_grad():
            preds = model(views, is_inference=False, use_motion=False)
            stage(f"[{seq}] rasterizing")
            rendered = render_views_from_predictions(model, preds, views, imgs.shape[-2], imgs.shape[-1])
            stage(f"[{seq}] MANO vertices")
            verts = compute_vertices_from_batch(preds["hand_joints"], mano, a.device)   # [1,S,2,778,3]

        gs_depth = preds.get("gs_depth")
        if gs_depth is None:
            raise SystemExit("preds has no gs_depth: nothing to check the sampling convention against")

        H, W = imgs.shape[-2], imgs.shape[-1]
        if H != W:
            print(f"NOTE non-square frame {H}x{W}; the grid normalises by width for both axes")

        # Both conventions, on the identical vertex cloud.
        grids, ratios = {}, {}
        stage(f"[{seq}] projecting both conventions")
        for name, rot in (("rotated", True), ("unrotated", False)):
            g, z = project_joints_to_norm_pixels(verts, cam_intr.to(a.device), rotated=rot)
            d, in_frame = sample_depth_at_joints(gs_depth, g)
            # Gate on the per-hand detector flag. Without it the median pools the absent hand's
            # unconstrained vertices, which project anywhere and sample unrelated scene depth.
            hv_v = hv.unsqueeze(-1).bool().expand_as(d)
            ok = hv_v & in_frame & (d > 0.01) & torch.isfinite(z) & torch.isfinite(d) & (z > 0)
            grids[name] = (g, z)
            r = (z[ok] / d[ok]) if bool(ok.any()) else torch.empty(0)
            ratios[name] = {
                "median_z_over_d": float(r.median()) if r.numel() else float("nan"),
                "n_valid": int(ok.sum()),
                "n_valid_ungated": int((in_frame & (d > 0.01) & torch.isfinite(z)
                                        & torch.isfinite(d) & (z > 0)).sum()),
                "median_gaussian_depth": float(d[ok].median()) if r.numel() else float("nan"),
                "median_z_hand": float(z[ok].median()) if r.numel() else float("nan"),
            }
            # Reprojection agreement, as a number rather than an eyeball: distance from the
            # projected vertex centroid to the centre of the detector box that defined the crop.
            off = []
            for si in range(n):
                for hi in (0, 1):
                    if not bool(hv[0, si, hi]):
                        continue
                    c = (g[0, si, hi].mean(0) * W).tolist()
                    bx = hb[0, si, hi].tolist()
                    off.append(((c[0] - (bx[0] + bx[2]) / 2 * W) ** 2
                                + (c[1] - (bx[1] + bx[3]) / 2 * W) ** 2) ** 0.5)
            ratios[name]["centroid_to_boxcentre_px_med"] = (
                float(np.median(off)) if off else float("nan"))
            ratios[name]["box_diag_px"] = float(np.median(
                [((hb[0, si, hi, 2] - hb[0, si, hi, 0]).item() ** 2
                  + (hb[0, si, hi, 3] - hb[0, si, hi, 1]).item() ** 2) ** 0.5 * W
                 for si in range(n) for hi in (0, 1) if bool(hv[0, si, hi])] or [float("nan")]))

        os.makedirs(a.out, exist_ok=True)
        # gs_depth is channel-first [S,1,Hd,Wd] or channel-last [S,Hd,Wd,1]. Drop the singleton
        # from whichever side carries it: a `while dim>3: squeeze(1)` loop hangs on channel-last,
        # which is the same defect eval_world_space.py:421 records as "the clip-1 hang".
        dep = gs_depth[0].float()
        stage(f"[{seq}] gs_depth shape {tuple(gs_depth.shape)}")
        if dep.dim() == 4 and dep.shape[1] == 1:
            dep = dep[:, 0]                          # channel-first -> [S,Hd,Wd]
        elif dep.dim() == 4 and dep.shape[-1] == 1:
            dep = dep[..., 0]                        # channel-last  -> [S,Hd,Wd]
        if dep.dim() != 3:
            raise SystemExit(f"gs_depth[0] is {tuple(dep.shape)}, expected [S,Hd,Wd]")
        Hd, Wd = dep.shape[-2], dep.shape[-1]

        # Frames the detector actually found a hand in. A uniform pick lands on hand-free frames,
        # where the overlay is skipped and the panel shows nothing.
        with_hand = torch.nonzero(hv[0].any(-1), as_tuple=False).flatten().cpu().numpy()
        if with_hand.size == 0:
            stage(f"[{seq}] no frame has a valid hand box, skipping")
            continue
        sel = with_hand[np.linspace(0, with_hand.size - 1,
                                    min(a.frames, with_hand.size)).round().astype(int)]
        stage(f"[{seq}] {with_hand.size}/{n} frames have a hand; drawing {len(sel)}")
        stage(f"[{seq}] drawing {len(sel)} panels")
        for s in sel:
            base_in, base_gs = _to_bgr(imgs[0, s]), _to_bgr(rendered[0, s])
            dmap = cv2.resize(_depth_to_bgr(dep[s]), (W, H), interpolation=cv2.INTER_NEAREST)
            row = []
            for name in ("unrotated", "rotated"):
                g, z = grids[name]
                over_in, over_gs, over_d = base_in.copy(), base_gs.copy(), dmap.copy()
                for h in (0, 1):
                    # Per hand, not per frame. An absent hand slot still carries head output, and
                    # drawing it paints unconstrained vertices across the whole frame.
                    if not bool(hv[0, s, h]):
                        continue
                    px = (g[0, s, h].detach().cpu().numpy() * W)      # [778,2] in frame pixels
                    zz = z[0, s, h].detach().cpu().numpy()
                    valid = np.isfinite(px).all(1) & (zz > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) \
                        & (px[:, 1] >= 0) & (px[:, 1] < H)
                    col = LEFT_COLOR if h == 0 else RIGHT_COLOR
                    over_in = render_mesh_overlay(over_in, px, faces[h], zz, valid, col, a.alpha, False)
                    over_gs = render_mesh_overlay(over_gs, px, faces[h], zz, valid, col, a.alpha, False)
                    for p in px[valid][::12].astype(np.int32):
                        cv2.circle(over_d, tuple(p), 1, col, -1)
                row.append(np.hstack([
                    _label(over_in, f"{name[:5]}: MANO / input"),
                    _label(over_gs, f"{name[:5]}: MANO / our render"),
                    _label(over_d, f"{name[:5]}: samples / gaussian depth"),
                ]))
            panel = np.vstack(row)
            cv2.imwrite(os.path.join(a.out, f"{seq}_f{frames[s]:06d}.png"), panel)

        info = {"seq": seq, "ckpt": a.ckpt, "config": a.config, "n_views": n,
                "frame_hw": [H, W], "gs_depth_hw": [Hd, Wd],
                "cam_intr_f_cx_cy": [float(x) for x in cam_intr.view(-1)],
                "render_mean": float(rendered.mean()), "ratios": ratios}
        with open(os.path.join(a.out, f"_vis_{seq}.json"), "w") as fh:
            json.dump(info, fh, indent=2)
        print(json.dumps(info, indent=2), flush=True)
        print(f"VIS seq done -> {a.out}", flush=True)

    print(f"VIS ALL done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
