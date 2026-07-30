#!/usr/bin/env python3
"""Measure the largest training clip length (num_frames) this hardware can sustain.

NeoVerse trains at 81 frames; we currently train at 16 and have never measured our ceiling.
The binding cost is the backbone's attention over num_frames * patches_per_frame tokens, which
grows quadratically, so the ceiling has to be measured rather than reasoned about.

For each candidate F this runs the REAL model (same build_model + build_views path as training
and eval) on a synthetic batch of the correct shape, does a forward AND a backward through the
trainable head, and records peak GPU memory and step time. OOM is caught per setting so one
failure does not end the sweep.

Synthetic images are fine here: memory and time depend on tensor shapes and the graph, not on
pixel values. What this does NOT measure is accuracy at longer F - see the ctxgate sweep for
that (C-abs degraded 29.7 -> 32.2 going 16 -> 64 because the head only ever saw 16 frames).

Usage:
  python -m scripts.probe_max_frames --config <train cfg> [--frames 16,24,32,48,64] [--batch 1]
"""
import argparse
import time

import torch
import yaml


def _fmt_gb(x):
    return f"{x / (1024 ** 3):6.2f} GB"


def probe_one(cfg, device, num_frames, batch, res, do_backward=True):
    """Return a dict describing one (num_frames, batch) setting, or an 'oom'/'error' marker."""
    from scripts.eval_world_space import build_model
    from scripts.train_hand_head import build_views

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build_model(cfg, device)
        # Trainable parameters only: the backbone is frozen, so the backward graph is the head's.
        train_params = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in train_params)

        imgs = torch.randn(batch, num_frames, 3, res, res, device=device)
        hb = torch.tensor([0.3, 0.3, 0.7, 0.7], device=device).view(1, 1, 1, 4) \
                  .expand(batch, num_frames, 2, 4).contiguous()
        hv = torch.ones(batch, num_frames, 2, dtype=torch.bool, device=device)
        views = build_views(imgs, num_frames, device, hb, hv)

        opt = torch.optim.AdamW(train_params, lr=1e-4) if (do_backward and train_params) else None

        # one warm-up step (allocator + any lazy init), then a timed step
        for it in range(2):
            if it == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(views, cond_flags=[0, 0, 1], is_inference=False, use_motion=False)
                loss = sum(v.float().square().mean() for v in preds.values()
                           if torch.is_tensor(v) and v.is_floating_point() and v.requires_grad)
            if opt is not None and torch.is_tensor(loss) and loss.requires_grad:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated()
        del model, views, imgs, opt
        torch.cuda.empty_cache()
        return {"ok": True, "peak": peak, "sec": dt, "n_train": n_train}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"ok": False, "why": "OOM"}
    except Exception as e:                                   # noqa: BLE001 - report, keep sweeping
        torch.cuda.empty_cache()
        return {"ok": False, "why": f"{type(e).__name__}: {e}"[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", default="16,24,32,48,64,81")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no_backward", action="store_true",
                    help="inference-only ceiling (no optimizer / backward graph)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = "cuda"
    res = int(cfg.get("data", {}).get("resolution", [224, 224])[0])
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[probe] device={torch.cuda.get_device_name(0)} total={_fmt_gb(total)} "
          f"res={res} batch={a.batch} backward={not a.no_backward}", flush=True)

    rows = []
    for f in [int(x) for x in a.frames.split(",")]:
        r = probe_one(cfg, device, f, a.batch, res, do_backward=not a.no_backward)
        if r["ok"]:
            print(f"[probe] F={f:3d}  peak={_fmt_gb(r['peak'])}  "
                  f"({100 * r['peak'] / total:5.1f}% of card)  step={r['sec']:6.2f}s  "
                  f"trainable={r['n_train'] / 1e6:.1f}M", flush=True)
        else:
            print(f"[probe] F={f:3d}  FAILED: {r['why']}", flush=True)
        rows.append((f, r))

    ok = [f for f, r in rows if r["ok"]]
    print("\n=== MAX-FRAMES SUMMARY ===")
    print(f"largest num_frames that fits at batch {a.batch}: {max(ok) if ok else 'none'}")
    for f, r in rows:
        if r["ok"]:
            print(f"  F={f:3d}  {_fmt_gb(r['peak'])}  {r['sec']:6.2f} s/step")
        else:
            print(f"  F={f:3d}  {r['why']}")
    print("NOTE: fitting is not the same as being worth it - changing num_frames invalidates the")
    print("feature cache (keyed <seq>_<frame_offset>) and the ctxgate sweep showed C-abs DEGRADES")
    print("with longer clips for a head trained at 16.")


if __name__ == "__main__":
    main()
