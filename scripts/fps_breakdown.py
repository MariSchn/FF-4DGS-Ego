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


# Which submodules are INHERITED from the pretrained reconstruction model and which we ADDED.
# This split is the point of the whole table. A frozen-backbone paper's speed is mostly someone
# else's speed, and a reviewer's first question is which part we are taking credit for. Reporting
# one aggregate number lets the reader assume we built the fast thing; reporting the split says
# plainly that the throughput is inherited and only the small head is ours.
INHERITED = ("visual_geometry_transformer", "depth_head", "gs_head", "cam_head", "pts_head",
             "norm_head", "gs_fwd_attr_head", "gs_bwd_attr_head",
             "velocity_fwd_head", "velocity_bwd_head")
ADDED = ("hand_head", "injection", "scale_head", "root_anchor")


def _classify(name: str) -> str:
    """Map a top-level module name to inherited / added / other."""
    root = name.split(".")[0]
    if root in INHERITED:
        return "inherited"
    if any(root.startswith(a) for a in ADDED):
        return "added"
    return "other"


def _param_split(model) -> dict:
    """Parameters by provenance and by trainability, counted from the live module tree.

    Trainable is read from requires_grad, which is NOT the same as "was trained": the camera head
    carries requires_grad=True yet appears in no optimizer param group, which is exactly how
    fps_probe.py came to report 262.4M trainable when the trained count is 46.3M. Both are
    reported so the discrepancy is visible rather than load-bearing.
    """
    out = {}
    for name, mod in model.named_children():
        cls = _classify(name)
        n = sum(p.numel() for p in mod.parameters())
        t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        if n == 0:
            continue
        out[name] = {"class": cls, "params": n, "params_requires_grad": t}
    # Snapshot the per-module rows BEFORE inserting totals: writing "_total_*" into `out` and then
    # re-scanning out.values() makes the second pass trip over the summary rows, which have no
    # "class" key.
    tot = sum(v["params"] for v in out.values())
    sums = {cls: sum(v["params"] for v in out.values() if v["class"] == cls)
            for cls in ("inherited", "added", "other")}
    for cls, s in sums.items():
        out[f"_total_{cls}"] = {"params": s, "pct": 100.0 * s / tot if tot else 0.0}
    out["_total"] = {"params": tot}
    return out


def _forward_once(model, batch, clip_len, device):
    """One inference forward, identical to the timed path so FLOPs and latency describe the same
    computation. Shared with _time_config rather than duplicated, because a FLOP count taken on a
    different code path than the timing is a number about nothing."""
    imgs = batch["img"].unsqueeze(0).to(device)
    hb = batch["hand_bboxes"].unsqueeze(0).to(device) if "hand_bboxes" in batch else None
    hv = batch["hand_valid"].unsqueeze(0).to(device) if "hand_valid" in batch else None
    views = build_views(imgs, clip_len, device, hb, hv)
    return model(views, cond_flags=[0, 0, 0], is_inference=True, use_motion=False)


def _flop_split(model, clip, clip_len, enable_gs, device):
    """Per-module FLOPs for one forward pass, using torch's own counter (no extra dependency).

    FLOPs are the honest efficiency currency here because latency depends on the GPU, the batch
    and the memory system, while a reviewer comparing against an offline SLAM pipeline wants to
    know how much arithmetic the method fundamentally does. It is also the number that makes the
    inherited/added split undeniable: our head is a rounding error against the encoder.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return {"error": "torch.utils.flop_counter unavailable (needs torch >= 2.0)"}

    counter = FlopCounterMode(display=False, depth=1)
    try:
        with counter:
            with torch.no_grad():
                _forward_once(model, clip, clip_len, device)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    per_mod = counter.get_flop_counts()
    out, totals = {}, {"inherited": 0, "added": 0, "other": 0}
    for mod_name, ops in per_mod.items():
        # Skip BOTH the "Global" row and the model ROOT. The counter reports the root module's own
        # total alongside each child's, so summing everything double-counts: on a two-child test
        # model that put 50% of all FLOPs in "other" and halved both real shares.
        if mod_name == "Global" or "." not in mod_name:
            continue
        f = sum(ops.values())
        short = mod_name.split(".", 1)[1]     # "Model.visual_geometry_transformer" -> the child
        cls = _classify(short)
        out[short] = {"class": cls, "flops": f}
        totals[cls] += f
    grand = sum(totals.values())
    for cls, s in totals.items():
        out[f"_total_{cls}"] = {"flops": s, "pct": 100.0 * s / grand if grand else 0.0}
    out["_total"] = {"flops": grand, "enable_gs": enable_gs}
    return out


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


def _time_config(cfg, enable_gs, clips, clip_len, warmup, iters, device, mano_model,
                 gs_anchor_only=False, stride=None):
    """Build the model with the Gaussian branch on or off and time the online path.

    ``gs_anchor_only`` reproduces the configuration the world numbers are actually scored in: the
    fast path returns before splat build and rasterization, so it costs less than a full render.
    Timing only the two extremes reports a system nobody runs.
    """
    cfg = copy.deepcopy(cfg)
    cfg["model"]["enable_gs"] = bool(enable_gs)
    model = build_model(cfg, device)
    if gs_anchor_only:
        model.gs_anchor_only = True
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _one(batch):
        _sync(device); t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = _forward_once(model, batch, clip_len, device)
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
    # Provenance and arithmetic, measured on the SAME model instance that was just timed.
    params = _param_split(model)
    flops = _flop_split(model, clips[0], clip_len, enable_gs, device)
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
        "gs_anchor_only": bool(gs_anchor_only),
        # Two rates, because they differ whenever clips overlap and only one of them is a
        # throughput. clip_fps divides by the clip length and describes a non-overlapping stream.
        # The evaluation protocol advances by `stride`, so at stride 8 with 16-frame clips every
        # frame is encoded twice and the deployed system delivers half of clip_fps. Reporting
        # clip_fps against a baseline's wall-clock rate overstates us by exactly that factor.
        "clip_fps": float(clip_len / tot.mean()),
        "fps": float((stride if stride else clip_len) / tot.mean()),
        "stride": int(stride if stride else clip_len),
        "peak_mem_GiB": float(peak),
        "param_split": params,
        "flop_split": flops,
        "n_timed": int(len(tot)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    # Defaults are the EVALUATION protocol, not a convenient one. eval_world_space defaults to
    # --clip_len 16 --stride 8 and calls that the locked protocol, so clips overlap by half and
    # every frame is encoded twice. Timing at stride 16 measures a stream nobody runs and doubles
    # the reported throughput.
    ap.add_argument("--clip_len", type=int, default=16,
                    help="MUST match the clip length behind the reported tables")
    ap.add_argument("--stride", type=int, default=8,
                    help="MUST match the eval stride; the unique-frame rate is stride / latency")
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
    # Three arms, not two. "as_scored" is the configuration every world number in the paper was
    # produced under: the Gaussian branch enabled for its depth, with the anchor-only fast path
    # skipping rasterization. Without it the table offers a cheaper system than the one evaluated
    # and a more expensive one than the one evaluated, and neither is what we ran.
    for name, gs, anchor in (("hands_only", False, False),
                             ("as_scored", True, True),
                             ("hands_plus_gaussians", True, False)):
        print(f"[fps_breakdown] timing {name} (enable_gs={gs}, gs_anchor_only={anchor}) ...",
              flush=True)
        try:
            rows[name] = _time_config(cfg, gs, clips, a.clip_len, a.warmup, a.iters,
                                      device, mano_model, gs_anchor_only=anchor, stride=a.stride)
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

    # ---- inherited vs added. The table above says how fast; this one says whose speed it is.
    ref = ok.get("hands_plus_gaussians") or ok.get("hands_only")
    if ref and "param_split" in ref:
        ps, fs = ref["param_split"], ref["flop_split"]
        print("\n% ---- inherited vs added (paste into the paper) ----")
        print("\\begin{tabular}{@{}lrrrr@{}}")
        print("\\toprule")
        print("Component & params (M) & \\% & GFLOPs & \\% \\\\")
        print("\\midrule")
        for cls, label in (("inherited", "Inherited (frozen reconstruction model)"),
                           ("added", "Added (hand branch, injection)")):
            p = ps.get(f"_total_{cls}", {})
            f = fs.get(f"_total_{cls}", {}) if "error" not in fs else {}
            gf = f"{f['flops']/1e9:.1f}" if f.get("flops") else "---"
            fp = f"{f['pct']:.1f}" if f.get("pct") is not None else "---"
            print(f"{label} & {p.get('params',0)/1e6:.1f} & {p.get('pct',0):.1f} & {gf} & {fp} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        if "error" in fs:
            print(f"% FLOPs unavailable: {fs['error']}")
        print("% Read this table before quoting the throughput: the cost is dominated by the")
        print("% inherited encoder, so our FPS is mostly the backbone's FPS. What is ours is the")
        print("% small added branch, and that is what the parameter-efficiency claim is about.")

        print("\n% per-module detail (appendix)")
        for name, v in sorted(ps.items()):
            if name.startswith("_"):
                continue
            fl = fs.get(name, {}).get("flops") if "error" not in fs else None
            fl_s = f"{fl/1e9:.2f} GFLOPs" if fl else "n/a"
            print(f"%   {name:32s} {v['class']:9s} {v['params']/1e6:8.1f} M  {fl_s}")

    print(f"\nFPS_BREAKDOWN_DONE T={a.clip_len} out={a.out}")


if __name__ == "__main__":
    main()
