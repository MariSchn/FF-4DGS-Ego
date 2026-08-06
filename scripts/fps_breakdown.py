#!/usr/bin/env python3
"""Per-component inference-cost breakdown, in the format ICLR/NeurIPS reviewers in this area
actually accept.

WHY THIS EXISTS. `fps_probe.py` reports a single number (6.59 FPS) measured with the Gaussian
branch OFF, at the superseded clip length T=16, and with detection excluded. That is not the
configuration the paper describes: the title promises 4D Gaussian reconstruction, and the world
evaluation *requires* gs_depth for its scene-scale solve, so the timed path provably is not the
path that produced the world numbers.

This is the single highest-yield attack in this literature, verified against six real OpenReview
threads (2026-08-05):
  - Human3R (ICLR 2026) reviewer N5HW ran the released code and reported "the inference speed ...
    falls far short of real-time, even significantly below the 5 FPS reported in Table 4". It cost
    roughly four points and dominated the meta-review.
  - A runtime/efficiency breakdown was demanded by reviewers in 5 of the 6 threads examined.
  - The repair that worked, in every case, was a PER-COMPONENT TABLE (Human3R Tab. 5-7,
    GVHMR/TRAM stage tables, Fuse-and-Refine's static/streaming breakdowns).

So this script produces that table rather than a headline number.

WHAT IT MEASURES. Two configurations, end to end, on identical clips:
  (a) hands only          - enable_gs False, the configuration `fps_probe.py` timed
  (b) hands + Gaussians   - enable_gs True, the configuration the paper actually claims
and reports the delta as the cost of the Gaussian branch. Detection is timed separately if a
detector is supplied, and otherwise reported as EXCLUDED rather than silently omitted, because
every baseline we compare against either includes its detector in its own number or states that
it does not.

USAGE
    python -m scripts.fps_breakdown --config <cfg.yaml> --data_root <store> \
        --clip_len 32 --iters 30 --out fps_breakdown.json

Emits JSON plus a LaTeX table body ready to paste into the paper.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import yaml

from scripts.eval_world_space import build_model
from scripts.train_hand_head import build_views, compute_joints_from_batch, HOT3DHandDataset
from scripts.hand_vis_utils import MANOModel


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _load_clips(cfg, mcfg, data_root, clip_len, stride, need, mano_model):
    clips = []
    for d in sorted(os.listdir(data_root)):
        sd = os.path.join(data_root, d)
        if not (os.path.isdir(sd)
                and os.path.exists(os.path.join(sd, "hand_data", "gt_joints_cache_world.pt"))):
            continue
        ds = HOT3DHandDataset([sd], mano_model, num_frames=clip_len, clip_stride=stride,
                              use_hand_crop=mcfg.get("use_hand_crop", False),
                              rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5))
        for i in range(len(ds)):
            clips.append(ds[i])
            if len(clips) >= need:
                return clips
    return clips


def _time_config(cfg, enable_gs, clips, clip_len, warmup, iters, device, mano_model):
    """Build the model with the Gaussian branch on or off and time the online path."""
    cfg = copy.deepcopy(cfg)
    cfg["model"]["enable_gs"] = bool(enable_gs)
    model = build_model(cfg, device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _one(batch):
        imgs = batch["img"].unsqueeze(0).to(device)
        hb = batch["hand_bboxes"].unsqueeze(0).to(device) if "hand_bboxes" in batch else None
        hv = batch["hand_valid"].unsqueeze(0).to(device) if "hand_valid" in batch else None
        views = build_views(imgs, clip_len, device, hb, hv)
        _sync(device); t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(views, cond_flags=[0, 0, 0], is_inference=True, use_motion=False)
        _sync(device); t1 = time.perf_counter()
        _ = compute_joints_from_batch(preds["hand_joints"], mano_model, device)
        _sync(device); t2 = time.perf_counter()
        return (t1 - t0), (t2 - t1)

    for b in clips[:warmup]:
        _one(b)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fwd, lift = [], []
    for b in clips[warmup:warmup + iters]:
        f, l = _one(b)
        fwd.append(f); lift.append(l)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    fwd, lift = np.array(fwd), np.array(lift)
    tot = fwd + lift
    return {
        "enable_gs": bool(enable_gs),
        "params_total_M": n_total / 1e6,
        "params_trainable_M": n_train / 1e6,
        "trainable_pct": 100.0 * n_train / max(n_total, 1),
        "forward_s_mean": float(fwd.mean()), "forward_s_std": float(fwd.std()),
        "lift_s_mean": float(lift.mean()), "lift_s_std": float(lift.std()),
        "total_s_mean": float(tot.mean()), "total_s_std": float(tot.std()),
        "fps": float(clip_len / tot.mean()),
        "peak_mem_GiB": float(peak),
        "n_timed": int(len(tot)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--clip_len", type=int, default=32,
                    help="MUST match the clip length behind the reported tables")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--detector_s_per_frame", type=float, default=None,
                    help="measured detector cost per frame; if omitted it is reported EXCLUDED")
    ap.add_argument("--out", default="fps_breakdown.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    clips = _load_clips(cfg, cfg["model"], a.data_root, a.clip_len, a.stride,
                        a.warmup + a.iters, mano_model)
    if len(clips) < a.warmup + a.iters:
        print(f"[fps_breakdown] WARN only {len(clips)} clips; wanted {a.warmup + a.iters}", flush=True)
    print(f"[fps_breakdown] GPU={gpu} T={a.clip_len} clips={len(clips)}", flush=True)

    rows = {}
    for name, gs in (("hands_only", False), ("hands_plus_gaussians", True)):
        print(f"[fps_breakdown] timing {name} (enable_gs={gs}) ...", flush=True)
        try:
            rows[name] = _time_config(cfg, gs, clips, a.clip_len, a.warmup, a.iters,
                                      device, mano_model)
        except Exception as e:                      # a config may not support the GS branch
            rows[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[fps_breakdown] {name} FAILED: {e}", flush=True)

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if "hands_only" in ok and "hands_plus_gaussians" in ok:
        d = ok["hands_plus_gaussians"]["total_s_mean"] - ok["hands_only"]["total_s_mean"]
        rows["gaussian_branch_cost_s"] = float(d)
        rows["gaussian_branch_pct"] = 100.0 * d / ok["hands_plus_gaussians"]["total_s_mean"]

    rows["_meta"] = {
        "gpu": gpu, "clip_len": a.clip_len, "config": a.config,
        "detection": ("EXCLUDED from all rows; shared detbox v3 across every compared method"
                      if a.detector_s_per_frame is None
                      else f"{a.detector_s_per_frame*1000:.1f} ms/frame, INCLUDED as a separate row"),
        "note": ("FPS = clip_len frames / mean per-clip latency. The model consumes a whole clip in "
                 "one alternating-attention pass, so throughput is frames per clip over clip latency."),
    }
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)

    # ---- LaTeX body, paste-ready
    print("\n% ---- paste into the paper ----")
    print("\\begin{tabular}{@{}lccc@{}}")
    print("\\toprule")
    print("Configuration & latency (ms) & FPS $\\uparrow$ & peak mem (GiB) \\\\")
    print("\\midrule")
    for key, label in (("hands_only", "Hands only (no scene)"),
                       ("hands_plus_gaussians", "Hands $+$ 4D Gaussians")):
        r = rows.get(key, {})
        if "error" in r:
            print(f"{label} & \\multicolumn{{3}}{{c}}{{not runnable: {r['error'][:40]}}} \\\\")
        else:
            print(f"{label} & {1000*r['total_s_mean']:.1f} & {r['fps']:.2f} & {r['peak_mem_GiB']:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("% Detection: " + rows["_meta"]["detection"])
    print(f"\nFPS_BREAKDOWN_DONE T={a.clip_len} out={a.out}")


if __name__ == "__main__":
    main()
