#!/usr/bin/env python3
"""Downsample diagnostic: represent a LONG HOI4D video with N frames sampled UNIFORMLY across
its whole duration, vs the same N frames taken CONSECUTIVELY (a short window), and compare
absolute camera-frame accuracy (C_abs) + world trajectory error (W). Real detector boxes
(detbox v3) when a box_root is given, else GT boxes. Live backbone forward (no feature cache).

Answers: does spreading the input frames across the full video (few-frame long-horizon input)
change absolute placement vs a dense short clip?
"""
import argparse
import os
import numpy as np
import torch
import yaml

from scripts.eval_world_space import build_model, predict_clip, _world_from_cam
from scripts.train_hand_head import build_views
from scripts.hand_vis_utils import MANOModel
from diffsynth.utils.auxiliary import load_video
import torchvision.transforms.functional as TVF


def _load_frames(video_path, num_frames, sampling, res=224):
    """Load num_frames via the codebase load_video (matches training center-crop).
    sampling='uniform' -> linspace(0,total-1,N) ; 'first' -> range(0,N)."""
    pil = load_video(video_path, num_frames=num_frames, resolution=(res, res),
                     resize_mode="center_crop", sampling=sampling)
    return torch.stack([TVF.to_tensor(p) for p in pil])   # [N,3,res,res]


def _c_abs(pj_cam, gt_cam, gt_valid):
    """Mean absolute per-joint camera-frame error (mm) over valid hands/frames."""
    # pj_cam [S,H,J,3] m ; gt_cam [S,H,J,3] m ; gt_valid [S,H] bool
    d = (pj_cam - gt_cam).norm(dim=-1)                   # [S,H,J]
    m = gt_valid.unsqueeze(-1).expand_as(d)
    if m.sum() == 0:
        return float("nan")
    return float((d[m].mean()) * 1000.0)


def _rigid_align(P, Q):
    """Kabsch: rotation+translation (no scale) mapping P->Q. P,Q [M,3]. Returns aligned P."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device))
    R = Vt.T @ D @ U.T
    return (R @ (P - Pc).T).T + Qc


def _w_mpjpe(pj_cam, c2w, s, gt_world, gt_valid, wa_short):
    """W-MPJPE: lift cam->world, rigid-align on the first `wa_short` frames, error over all."""
    Sp = pj_cam.shape[0]
    world = _world_from_cam(pj_cam, c2w, s)              # [S, H*J, 3]
    S = min(Sp, gt_world.shape[0])
    world = world[:S]
    gtw = gt_world[:S].reshape(S, -1, 3)
    val = gt_valid[:S].repeat_interleave(pj_cam.shape[2], dim=1)   # [S, H*J]
    k = min(wa_short, S)
    pm = val[:k].reshape(-1)
    Pa, Qa = world[:k].reshape(-1, 3)[pm], gtw[:k].reshape(-1, 3)[pm]
    if Pa.shape[0] < 3:
        return float("nan")
    # solve alignment on window, apply to all
    Pc, Qc = Pa.mean(0), Qa.mean(0)
    H = (Pa - Pc).T @ (Qa - Qc)
    U, _, Vt = torch.linalg.svd(H)
    dsign = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, dsign], device=Pa.device))
    R = Vt.T @ D @ U.T
    wa = (R @ (world.reshape(-1, 3) - Pc).T).T + Qc
    wa = wa.reshape(S, -1, 3)
    d = (wa - gtw).norm(dim=-1)
    m = val
    if m.sum() == 0:
        return float("nan")
    return float(d[m].mean() * 1000.0)


def run_clip(model, mano_model, device, cfg, video_path, idx, sampling, hd, box_root, seq, cam_intr):
    N = len(idx)
    imgs = _load_frames(video_path, N, sampling).unsqueeze(0).to(device)   # [1,N,3,224,224]
    # boxes: detector (box_root) if present, else GT
    bpath = os.path.join(box_root or hd, "hand_bboxes_v2_rf1.5_res224x224.pt")
    bb = torch.load(bpath, map_location="cpu")
    if isinstance(bb, dict):
        boxk = next((k for k in ("bboxes", "boxes", "bbox") if k in bb), None)
        box_t = bb[boxk] if boxk else None
        val_t = bb.get("valid")
    else:  # raw tensor [N,2,4]
        box_t, val_t = bb, None
    idx_t = torch.as_tensor(idx)
    hb = box_t[idx_t].unsqueeze(0).to(device) if box_t is not None else None
    hv = val_t[idx_t].unsqueeze(0).to(device) if val_t is not None else None
    views = build_views(imgs, N, device, hb, hv)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(views, cond_flags=[0, 0, 0], is_inference=True, use_motion=False)
    # Starred: predict_clip's tuple has grown twice (ratios, then s_failed) and a fixed-arity
    # unpack here is a latent ValueError that only fires when this probe runs. See #70.
    pj_cam, c2w, s, *_rest = predict_clip(preds, mano_model, device, cam_intr, model=model)
    return pj_cam, c2w, s, (hv[0].cpu() if hv is not None else torch.ones(N, 2, dtype=torch.bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--box_root", default=None, help="dir with per-seq detector boxes; None=GT boxes in-place")
    ap.add_argument("--num_frames", type=int, default=30)
    ap.add_argument("--max_seqs", type=int, default=8)
    ap.add_argument("--min_len", type=int, default=90, help="only use sequences at least this long")
    ap.add_argument("--wa_short", type=int, default=30)
    ap.add_argument("--out", default="downsample_probe.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])
    model = build_model(cfg, device)

    from decord import VideoReader
    seqs = []
    for d in sorted(os.listdir(a.data_root)):
        sd = os.path.join(a.data_root, d)
        hd = os.path.join(sd, "hand_data")
        vp = os.path.join(sd, "video_main_rgb.mp4")
        if not (os.path.isdir(sd) and os.path.exists(os.path.join(hd, "gt_joints_cache_world.pt"))
                and os.path.exists(vp)):
            continue
        n = len(VideoReader(vp))
        if n >= a.min_len:
            seqs.append((d, sd, hd, vp, n))
    seqs = seqs[:a.max_seqs]
    print(f"[downsample_probe] {len(seqs)} seqs (>= {a.min_len} frames), N={a.num_frames}, "
          f"boxes={'detbox:'+a.box_root if a.box_root else 'GT-in-place'}", flush=True)

    rows = []
    for name, sd, hd, vp, n in seqs:
        box_dir = os.path.join(a.box_root, name, "hand_data") if a.box_root else hd
        gt_cam = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), map_location="cpu").float()
        gt_world = torch.load(os.path.join(hd, "gt_joints_cache_world.pt"), map_location="cpu").float()
        cam_intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)
        N = a.num_frames
        uni = list(np.linspace(0, n - 1, N, dtype=int))               # spread across whole video
        con = list(range(0, min(N, n)))                                # first N consecutive
        try:
            for tag, idx, samp in (("uniform", uni, "uniform"), ("consec", con, "first")):
                pj, c2w, s, hv = run_clip(model, mano_model, device, cfg, vp, idx, samp, hd, box_dir, name, cam_intr)
                gc = gt_cam[idx][:pj.shape[0]]
                gw = gt_world[idx][:pj.shape[0]]
                cab = _c_abs(pj, gc, hv[:pj.shape[0]])
                w = _w_mpjpe(pj, c2w, s, gw, hv[:pj.shape[0]], a.wa_short)
                rows.append({"seq": name, "n_video": n, "mode": tag, "C_abs": cab, "W": w})
                print(f"  [{name} n={n}] {tag:8s} C_abs={cab:.1f}  W={w:.1f}", flush=True)
        except Exception as e:
            import traceback
            print(f"[skip {name}] {type(e).__name__}: {e}", flush=True); traceback.print_exc()

    import json
    def _agg(mode, key):
        v = [r[key] for r in rows if r["mode"] == mode and np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")
    summary = {"n_seqs": len({r["seq"] for r in rows}), "num_frames": a.num_frames,
               "uniform_C_abs": _agg("uniform", "C_abs"), "consec_C_abs": _agg("consec", "C_abs"),
               "uniform_W": _agg("uniform", "W"), "consec_W": _agg("consec", "W"), "rows": rows}
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDOWNSAMPLE_PROBE_DONE uniform C_abs={summary['uniform_C_abs']:.1f} / "
          f"consec C_abs={summary['consec_C_abs']:.1f} | uniform W={summary['uniform_W']:.1f} / "
          f"consec W={summary['consec_W']:.1f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
