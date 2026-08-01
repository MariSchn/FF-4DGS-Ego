#!/usr/bin/env python3
"""How many frames can the NeoVerse/WorldMirror reconstructor take in ONE forward pass on this
GPU before it OOMs? Answers the "can it ingest 128 frames?" question empirically. Builds the model
once, then runs a single forward at increasing clip lengths, catching CUDA OOM, reporting
fit/OOM + latency + peak memory per size. GS off, is_inference, use_motion=False (our eval path).
"""
import argparse
import time
import torch
import yaml

from scripts.eval_world_space import build_model
from scripts.train_hand_head import build_views, HOT3DHandDataset
from scripts.hand_vis_utils import MANOModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sizes", default="16,32,48,64,96,128")
    ap.add_argument("--out", default="max_frames_probe.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])
    model = build_model(cfg, device)
    print(f"[max_frames] GPU={gpu}", flush=True)

    sizes = [int(x) for x in a.sizes.split(",")]
    maxN = max(sizes)
    # grab one sequence long enough, load maxN frames worth of a clip by building a big dataset item
    import os
    seq = None
    for d in sorted(os.listdir(a.data_root)):
        if os.path.exists(os.path.join(a.data_root, d, "hand_data", "gt_joints_cache_world.pt")):
            seq = os.path.join(a.data_root, d); break
    ds = HOT3DHandDataset([seq], mano, num_frames=maxN, clip_stride=maxN,
                          use_hand_crop=mcfg.get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5))
    big = ds[0]
    imgs_all = big["img"]                      # [maxN,3,H,W]
    hb_all = big.get("hand_bboxes"); hv_all = big.get("hand_valid")
    results = {}
    for n in sizes:
        if imgs_all.shape[0] < n:
            results[n] = {"status": "not_enough_frames"}; print(f"  n={n}: not enough frames", flush=True); continue
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        imgs = imgs_all[:n].unsqueeze(0).to(device)
        hb = hb_all[:n].unsqueeze(0).to(device) if hb_all is not None else None
        hv = hv_all[:n].unsqueeze(0).to(device) if hv_all is not None else None
        views = build_views(imgs, n, device, hb, hv)
        try:
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(views, cond_flags=[0, 0, 0], is_inference=True, use_motion=False)
            torch.cuda.synchronize() if device == "cuda" else None
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            results[n] = {"status": "OK", "latency_s": round(dt, 2), "peak_gb": round(peak, 2),
                          "fps": round(n / dt, 2)}
            print(f"  n={n}: OK {dt:.2f}s peak={peak:.1f}GB fps={n/dt:.1f}", flush=True)
        except RuntimeError as e:
            msg = str(e)[:120]
            results[n] = {"status": "OOM" if "out of memory" in str(e).lower() else "ERR", "msg": msg}
            print(f"  n={n}: {results[n]['status']} {msg}", flush=True)
            torch.cuda.empty_cache()
    import json
    json.dump({"gpu": gpu, "results": results}, open(a.out, "w"), indent=2)
    ok = [n for n, r in results.items() if r.get("status") == "OK"]
    print(f"MAX_FRAMES_DONE gpu={gpu} max_fit={max(ok) if ok else 0} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
