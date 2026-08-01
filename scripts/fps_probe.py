#!/usr/bin/env python3
"""Inference-throughput (FPS) probe for the feedforward hand model, for the world-comparison
table's FPS column. Times the SAME online path the eval runs - backbone forward
(model(views, is_inference=True, use_motion=False), GS off) + hand-head joint lift
(compute_joints_from_batch) - on real clips, CUDA-synchronized, with warmup. Reports
frames/second so it is directly comparable to the SLAM baselines' wall-clock FPS (~2.3).

FPS is defined as clip_len frames / mean per-clip forward+lift latency: the model consumes a
whole clip of clip_len frames in one alternating-attention pass, so throughput is frames per
clip divided by clip latency. Detection is excluded (negligible next to the backbone, and shared
across all methods via detbox v3); SLAM is excluded because our method has none - that is the
point of the column.
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--clip_len", type=int, default=16, help="frames per forward pass (eval default 16)")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30, help="timed clips")
    ap.add_argument("--out", default="fps_probe.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])
    model = build_model(cfg, device)

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[fps_probe] GPU={gpu} clip_len={a.clip_len} params total={n_total/1e6:.1f}M "
          f"trainable={n_trainable/1e6:.1f}M", flush=True)

    # gather clips from real sequences until we have warmup+iters
    need = a.warmup + a.iters
    clips = []
    for d in sorted(os.listdir(a.data_root)):
        sd = os.path.join(a.data_root, d)
        hd = os.path.join(sd, "hand_data")
        if not (os.path.isdir(sd) and os.path.exists(os.path.join(hd, "gt_joints_cache_world.pt"))):
            continue
        ds = HOT3DHandDataset([sd], mano_model, num_frames=a.clip_len, clip_stride=a.stride,
                              use_hand_crop=mcfg.get("use_hand_crop", False),
                              rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5))
        for i in range(len(ds)):
            clips.append(ds[i])
            if len(clips) >= need:
                break
        if len(clips) >= need:
            break
    if len(clips) < need:
        print(f"[fps_probe] WARN only {len(clips)} clips available (< {need}); using what we have", flush=True)
    print(f"[fps_probe] {len(clips)} clips from {a.data_root}", flush=True)

    def _forward(batch, time_lift):
        imgs = batch["img"].unsqueeze(0).to(device)
        hb = batch["hand_bboxes"].unsqueeze(0).to(device) if "hand_bboxes" in batch else None
        hv = batch["hand_valid"].unsqueeze(0).to(device) if "hand_valid" in batch else None
        views = build_views(imgs, a.clip_len, device, hb, hv)
        _sync(device); t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(views, cond_flags=[0, 0, 0], is_inference=True, use_motion=False)
        _sync(device); t1 = time.perf_counter()
        # hand-head joint lift (MANO fk from the predicted params) - the rest of the online path
        _ = compute_joints_from_batch(preds["hand_joints"], mano_model, device)
        _sync(device); t2 = time.perf_counter()
        return (t1 - t0), (t2 - t1)

    # warmup
    for i in range(min(a.warmup, len(clips))):
        _forward(clips[i], True)
    # timed
    fwd, lift = [], []
    timed = clips[a.warmup:a.warmup + a.iters] if len(clips) > a.warmup else clips
    for b in timed:
        tf, tl = _forward(b, True)
        fwd.append(tf); lift.append(tl)

    fwd = np.array(fwd); lift = np.array(lift); tot = fwd + lift
    def stats(x):
        return {"mean_ms": float(x.mean() * 1e3), "median_ms": float(np.median(x) * 1e3),
                "p90_ms": float(np.percentile(x, 90) * 1e3)}
    fps_fwd = a.clip_len / fwd.mean()
    fps_tot = a.clip_len / tot.mean()
    res = {
        "gpu": gpu, "clip_len": a.clip_len, "n_timed": int(len(timed)),
        "params_total_M": round(n_total / 1e6, 2), "params_trainable_M": round(n_trainable / 1e6, 2),
        "forward": stats(fwd), "lift": stats(lift), "forward_plus_lift": stats(tot),
        "fps_forward_only": round(float(fps_fwd), 2),
        "fps_forward_plus_lift": round(float(fps_tot), 2),
    }
    import json
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nFPS_PROBE_DONE gpu={gpu} clip_len={a.clip_len} "
          f"fwd={res['forward']['mean_ms']:.1f}ms lift={res['lift']['mean_ms']:.1f}ms "
          f"-> FPS(fwd)={res['fps_forward_only']} FPS(fwd+lift)={res['fps_forward_plus_lift']} "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
