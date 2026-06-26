"""Scale-source ablation — THE thesis test: which metric-scale source best places the
up-to-scale GS scene depth onto HOI4D's dense GT sensor depth?

We take the FROZEN model's up-to-scale gs_depth and apply ONE global scale s, then score
|s*gs_depth - gt_dense_depth| over NON-HAND scene pixels (so the hand cannot trivially fit
its own anchor -> non-circular). Scale sources:
  - none   : s = 1 (raw frozen gs_depth residual; ~ the 20.5 cm no-supervision baseline).
  - hand   : s from the in-scene MANO hand (solve_metric_scale at the GT wrist). OURS.
  - oracle : s = median(gt_depth / gs_depth) over the scored pixels. Upper bound.
  - fm     : s from a metric-depth FM (UniDepth-V2) over the masked (non-hand) scene -- the
             standard "mask the hand, scale to a depth net" recipe (HaWoR / EgoGrasp).
             OPTIONAL: used only if UniDepth imports on this node; else reported skipped.

Claim to establish: hand approx oracle, and hand < fm  =>  the hand drives the coupling.

Run (gb10 / venv_gb10):
    python -m scripts.eval_scale_source --config configs/exp_hoi4d_depth.yaml \
        --pp /tmp/hoi4d_pp --raw /work/scratch/dmonopoli/hoi4d --out scale_source.json [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import yaml

from scripts.train_hoi4d_depth import build_model
from scripts.train_hand_head import build_views
from diffsynth.auxiliary_models.worldmirror.models.heads.metric_scale_head import solve_metric_scale
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels, sample_depth_at_joints,
)

RH = 1               # right hand index in the [N,2,16,3] GT joint cache
HAND_EXCL_R = 14     # px radius around projected hand joints to EXCLUDE from scene scoring
DMIN, DMAX = 0.05, 10.0


def _center_square(a):
    h, w = a.shape[:2]
    s = min(h, w)
    return a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def _load_img(path, res):
    im = cv2.cvtColor(cv2.resize(_center_square(cv2.imread(path)), (res, res),
                                 interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,res,res]


def _load_depth(path, res):
    d = _center_square(cv2.imread(path, cv2.IMREAD_ANYDEPTH)).astype(np.float32) / 1000.0  # mm->m
    d = cv2.resize(d, (res, res), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(d)  # [res,res] metres


def _hand_mask(joints_1S2J3, cam_intr, S, H, W, device):
    """[H,W per frame] bool True inside a disk around each projected hand joint."""
    grid, _ = project_joints_to_norm_pixels(joints_1S2J3, cam_intr)  # [1,S,2,J,2] in [0,1]
    gx = (grid[0, ..., 0].reshape(S, -1) * W)
    gy = (grid[0, ..., 1].reshape(S, -1) * H)
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    masks = torch.zeros(S, H, W, dtype=torch.bool, device=device)
    for s in range(S):
        for j in range(gx.shape[1]):
            cx, cy = float(gx[s, j]), float(gy[s, j])
            if 0 <= cx < W and 0 <= cy < H:
                masks[s] |= ((xs - cx) ** 2 + (ys - cy) ** 2) <= HAND_EXCL_R ** 2
    return masks


def _scene_err(gs, gt, valid, s):
    """mean |s*gs - gt| over valid pixels, in cm. gs,gt,valid: [P]."""
    e = (s * gs[valid] - gt[valid]).abs()
    return float(e.mean() * 100.0) if e.numel() else float("nan")


def _median_ratio(num, den, valid):
    r = (num[valid] / den[valid].clamp_min(1e-4))
    r = r[torch.isfinite(r) & (r > 0)]
    return float(r.median().clamp(0.05, 20.0)) if r.numel() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_hoi4d_depth.yaml")
    ap.add_argument("--pp", required=True, help="preprocessed root: <seq>/hand_data/{gt_joints_cache_cam_v2.pt,cam_intrinsics.pt}")
    ap.add_argument("--raw", required=True, help="raw HOI4D root: <seq>/{images,raw_depth}/")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max_clips_per_seq", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="1 seq, 2 clips")
    ap.add_argument("--out", default="scale_source.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["model"], device).eval()
    if not cfg["model"].get("enable_gs", False):
        raise SystemExit("enable_gs must be true (need gs_depth).")

    # Optional metric-depth FM (UniDepth-V2). Only if it imports on THIS node.
    fm = None
    try:
        from unidepth.models import UniDepthV2
        fm = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device).eval()
        print("[fm] UniDepth-V2 loaded on this node.", flush=True)
    except Exception as e:
        print(f"[fm] UniDepth unavailable here ({type(e).__name__}); the 'fm' source is SKIPPED "
              f"(run it on the x86 venv_unidepth stage).", flush=True)

    seqs = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(args.raw, "*")))
            if os.path.isdir(os.path.join(d, "raw_depth"))
            and os.path.exists(os.path.join(args.pp, os.path.basename(d), "hand_data", "gt_joints_cache_cam_v2.pt"))]
    if args.smoke:
        seqs = seqs[:1]
    print(f"[scale-source] {len(seqs)} seqs with BOTH raw_depth and hand caches", flush=True)

    acc = {k: [] for k in ("none", "hand", "oracle", "fm")}
    s_acc = {k: [] for k in ("hand", "oracle", "fm")}
    per_seq = []
    S, res = args.num_frames, args.res
    for sq in seqs:
        hd = os.path.join(args.pp, sq, "hand_data")
        gtj = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), map_location="cpu").float()  # [N,2,16,3]
        cam_intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)
        imgs_f = sorted(glob.glob(os.path.join(args.raw, sq, "images", "*.png")))
        deps_f = sorted(glob.glob(os.path.join(args.raw, sq, "raw_depth", "*.png")))
        n = min(len(imgs_f), len(deps_f), gtj.shape[0])
        seq_err = {k: [] for k in acc}
        clips = list(range(0, n - S + 1, args.stride))[: (2 if args.smoke else args.max_clips_per_seq)]
        for t in clips:
            img = torch.stack([_load_img(imgs_f[t + k], res) for k in range(S)]).unsqueeze(0).to(device)  # [1,S,3,res,res]
            gtd = torch.stack([_load_depth(deps_f[t + k], res) for k in range(S)]).to(device)             # [S,res,res] m
            j = gtj[t:t + S].unsqueeze(0).to(device)                                                       # [1,S,2,16,3]
            ci = cam_intr.to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(build_views(img, S, device, None, None), is_inference=False, use_motion=False)
            gsd = preds.get("gs_depth")
            if gsd is None:
                continue
            gsd = gsd.float().reshape(S, res, res)                                                         # up-to-scale depth
            has_hand = torch.zeros(1, S, 2, device=device)
            has_hand[..., RH] = 1.0          # HOI4D releases the RIGHT hand only; the left
            #                                  slot is empty -> marking it valid collapses the
            #                                  scale median onto the 0.1 clamp floor.
            s_hand = float(solve_metric_scale(j, gsd.reshape(1, S, 1, res, res), has_hand, ci))
            hm = _hand_mask(j, ci, S, res, res, device)                                                    # [S,res,res] hand region
            # FM scale (masked scene): scale gs to the FM's metric depth over non-hand pixels.
            s_fm = float("nan")
            if fm is not None:
                K3 = torch.tensor([[float(ci[0, 0]) * res / (2 * float(ci[0, 1])), 0, res / 2],
                                   [0, float(ci[0, 0]) * res / (2 * float(ci[0, 2])), res / 2],
                                   [0, 0, 1]], device=device)
                fmd = torch.stack([fm.infer((img[0, k] * 255).clamp(0, 255), K3)["depth"].squeeze().float()
                                   for k in range(S)])                                                     # [S,res,res] metric
                v_fm = (~hm) & (fmd > DMIN) & (fmd < DMAX) & (gsd > DMIN)
                s_fm = _median_ratio(fmd.reshape(-1), gsd.reshape(-1), v_fm.reshape(-1))
            # scoring region: non-hand, valid GT + gs.
            v = ((~hm) & (gtd > DMIN) & (gtd < DMAX) & (gsd > DMIN)).reshape(-1)
            gsf, gtf = gsd.reshape(-1), gtd.reshape(-1)
            s_oracle = _median_ratio(gtf, gsf, v)
            seq_err["none"].append(_scene_err(gsf, gtf, v, 1.0))
            seq_err["hand"].append(_scene_err(gsf, gtf, v, s_hand))
            seq_err["oracle"].append(_scene_err(gsf, gtf, v, s_oracle))
            if fm is not None:
                seq_err["fm"].append(_scene_err(gsf, gtf, v, s_fm))
            s_acc["hand"].append(s_hand); s_acc["oracle"].append(s_oracle)
            if fm is not None:
                s_acc["fm"].append(s_fm)
        row = {"seq": sq, "n_clips": len(clips)}
        for k in acc:
            vals = [e for e in seq_err[k] if e == e]
            if vals:
                row[k + "_cm"] = float(np.mean(vals)); acc[k].extend(vals)
        per_seq.append(row)
        print(f"[{sq}] " + " ".join(f"{k}={row.get(k+'_cm', float('nan')):.1f}cm" for k in acc), flush=True)

    agg = {k + "_cm": (float(np.mean(acc[k])) if acc[k] else float("nan")) for k in acc}
    for k in s_acc:
        agg["s_" + k + "_med"] = (float(np.median(s_acc[k])) if s_acc[k] else float("nan"))
    agg["n_seqs"] = len(per_seq)
    agg["fm_available"] = fm is not None
    json.dump({"aggregate": agg, "per_seq": per_seq}, open(args.out, "w"), indent=2)
    print("\n==== SCALE-SOURCE ABLATION (non-hand scene depth error, cm; lower better) ====")
    print(f"  none(s=1)={agg['none_cm']:.1f}  HAND={agg['hand_cm']:.1f}  oracle={agg['oracle_cm']:.1f}  "
          f"fm(UniDepth)={agg.get('fm_cm', float('nan')):.1f}  (fm_available={fm is not None})")
    print(f"  scales: s_hand={agg['s_hand_med']:.3f}  s_oracle={agg['s_oracle_med']:.3f}  s_fm={agg.get('s_fm_med', float('nan')):.3f}")
    print(f"  -> CLAIM holds if HAND approx oracle AND HAND < fm. n={agg['n_seqs']} seqs -> {args.out}")


if __name__ == "__main__":
    main()
