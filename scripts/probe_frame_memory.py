"""Feasibility gate for single-pass long-window inference: how many frames fit?

The variable-length proposal (train n ~ U[2,32], infer at 100 frames in ONE pass) rests on an
assumption nobody has measured: that NeoVerse can take 100 frames at all. Its alternating
attention is O((S*P)^2) in the number of frames S times patches P, so the answer is not obvious
and the whole idea dies if it is no.

This probe answers it before any retraining is committed. It builds the real model, runs a real
forward pass at increasing frame counts, and records peak GPU memory, catching OOM as a RESULT
rather than a crash. It needs no trained checkpoint: memory depends on architecture and shapes,
not on weight values.

    python -m scripts.probe_frame_memory --config configs/train_hoi4d_32f.yaml \
        --frames 16,32,48,64,80,100,128 --out probe.json
"""
from __future__ import annotations

import argparse
import json

import torch
import yaml

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from scripts.train_hand_head import build_views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", default="16,32,48,64,80,100,128")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--enable_gs", action="store_true",
                    help="probe WITH the Gaussian branch; default probes the hands-only path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA: a memory probe on CPU measures nothing")

    cfg = yaml.safe_load(open(args.config))
    model_cfg = dict(cfg["model"])
    model_cfg["enable_gs"] = bool(args.enable_gs)

    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location="cpu")
    model.load_state_dict(base.get("state_dict", base.get("reconstructor", base)), strict=False)
    model.cuda().eval()
    model.gs_anchor_only = True

    dev = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU={dev}  total={total:.1f} GiB  enable_gs={args.enable_gs}  res={args.res}")

    rows = []
    for S in [int(x) for x in args.frames.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        imgs = torch.rand(1, S, 3, args.res, args.res, device="cuda")
        hb = torch.tensor([[[[0.3, 0.3, 0.7, 0.7], [0.3, 0.3, 0.7, 0.7]]] * S], device="cuda")
        hv = torch.ones(1, S, 2, device="cuda")
        try:
            with torch.no_grad():
                _ = model(build_views(imgs, S, "cuda", hb, hv), is_inference=False, use_motion=False)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            rows.append({"frames": S, "ok": True, "peak_GiB": round(peak, 2)})
            print(f"  S={S:4d}  OK    peak={peak:6.2f} GiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            rows.append({"frames": S, "ok": False, "peak_GiB": None})
            print(f"  S={S:4d}  OOM", flush=True)
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            rows.append({"frames": S, "ok": False, "peak_GiB": None})
            print(f"  S={S:4d}  OOM (RuntimeError)", flush=True)
            torch.cuda.empty_cache()
        finally:
            del imgs, hb, hv

    ok = [r for r in rows if r["ok"]]
    json.dump({"gpu": dev, "total_GiB": round(total, 1), "enable_gs": bool(args.enable_gs),
               "res": args.res, "rows": rows,
               "max_frames_ok": max((r["frames"] for r in ok), default=0)},
              open(args.out, "w"), indent=2)
    print(f"\nMAX FRAMES THAT FIT: {max((r['frames'] for r in ok), default=0)}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
