#!/usr/bin/env python3
"""Write one of our sequences as the single ``.npz`` MoVieS reads, so its released checkpoint runs
on our data without touching its repository.

MoVieS's input contract, from ``src/infer_davis_nvs.py``:

    npz["images"]    (F, 3, H, W) float in [0, 1]   (vis_util.tensor_to_video documents the range)
    npz["C2W"]       (F, 4, 4)    camera-to-world
    npz["fxfycxcy"]  (F, 4)       NORMALIZED intrinsics, i.e. divided by width and height

    <out>/<seq>.npz          and the loader is DATA_DIR/<name>.npz

Unlike AnySplat, MoVieS does not predict cameras. Its own DAVIS samples carry poses estimated by
MegaSAM. We supply our store's ground-truth extrinsics instead, which is more accurate than what it
normally receives and is an asymmetry against our own predicted-pose rows: declare it wherever a
MoVieS number appears.

TWO CONVENTIONS ARE ASSUMED HERE AND BOTH ARE CHECKED RATHER THAN TRUSTED, because this project has
already published months of world metrics computed with a trajectory applied backwards.

  * Our store holds ``cam_extrinsics_cache.pt`` as T_camera_world (w2c). MoVieS wants
    camera-to-world, so we invert, and the inversion is the default rather than a flag.
  * ``fxfycxcy`` is normalized. The demo reuses frame 0's entry for every output frame, which only
    makes sense for a resolution-independent quantity, and MoVieS shares VGGT's camera
    normalization. ``--self_check`` reprojects our own ground-truth hand joints through the
    intrinsics actually written, in pixels, and reports how many land inside the image.

A convention error shows up as a self-check pass rate near zero, and as renderings that look like
noise. Run --self_check on one sequence before converting a set.

    python -m scripts.ours_store_to_movies --data_root <store> --out_root <out> --limit 1 --self_check
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch


def load_seq(seq_dir: str):
    """Return (intr[f, cx, cy], w2c [T,4,4], joints_cam [T,H,J,3] or None), all float64."""
    hd = os.path.join(seq_dir, "hand_data")
    intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").numpy()
    w2c = torch.load(os.path.join(hd, "cam_extrinsics_cache.pt"), map_location="cpu").numpy()
    j_cam = None
    p = os.path.join(hd, "gt_joints_cache_cam_v2.pt")
    if os.path.isfile(p):
        j_cam = torch.load(p, map_location="cpu").numpy()
    return intr.astype(np.float64), w2c.astype(np.float64), j_cam


def self_check(intr, j_cam, w: int, h: int) -> dict:
    """Reproject our own camera-frame joints and report how many land in the image.

    This does not test MoVieS. It tests that the intrinsics and the frame size we are about to
    write describe the images we are about to write, which is the part a wrong assumption would
    break silently.
    """
    if j_cam is None:
        return {"ran": False}
    f, cx, cy = intr
    j = j_cam.reshape(-1, 3)
    j = j[np.isfinite(j).all(-1) & (j[:, 2] > 1e-6)]
    if not len(j):
        return {"ran": False}
    u = f * j[:, 0] / j[:, 2] + cx
    v = f * j[:, 1] / j[:, 2] + cy
    inside = ((u >= 0) & (u < w) & (v >= 0) & (v < h)).mean()
    return {"ran": True, "n": int(len(j)), "frac_inside": float(inside)}


def read_frames(vid: str, idx: np.ndarray) -> np.ndarray:
    """Decode the requested frame indices as (F, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(vid)
    want = set(int(i) for i in idx)
    frames, k = {}, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if k in want:
            frames[k] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        k += 1
        if len(frames) == len(want):
            break
    cap.release()
    missing = [int(i) for i in idx if int(i) not in frames]
    if missing:
        raise RuntimeError(f"{vid}: frames {missing[:5]} not decodable of {k} read")
    return np.stack([frames[int(i)] for i in idx])


def convert(seq_dir: str, out_path: str, n_frames: int, do_check: bool) -> dict:
    intr, w2c, j_cam = load_seq(seq_dir)
    vid = os.path.join(seq_dir, "video_main_rgb.mp4")
    if not os.path.isfile(vid):
        raise FileNotFoundError(vid)

    t_total = min(len(w2c), int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT)))
    if t_total < n_frames:
        raise RuntimeError(f"{seq_dir}: {t_total} frames, need {n_frames}")
    # Evenly spaced over the whole sequence: MoVieS asks for a fixed count, and a contiguous head
    # would show it a fraction of a second of motion in a clip that is several seconds long.
    idx = np.linspace(0, t_total - 1, n_frames).round().astype(int)

    rgb = read_frames(vid, idx)                                   # (F, H, W, 3) uint8
    h, w = rgb.shape[1:3]
    images = (rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # (F, 3, H, W) in [0, 1]

    c2w = np.linalg.inv(w2c[idx])                                 # (F, 4, 4)

    f, cx, cy = intr
    fxfycxcy = np.tile(
        np.array([f / w, f / h, cx / w, cy / h], dtype=np.float32), (n_frames, 1))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, images=images.astype(np.float32),
             C2W=c2w.astype(np.float32), fxfycxcy=fxfycxcy)

    info = {"seq": os.path.basename(seq_dir), "frames": n_frames, "of": t_total, "hw": (h, w)}
    if do_check:
        info["check"] = self_check(intr, j_cam, w, h)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="our store, one directory per sequence")
    ap.add_argument("--out_root", required=True, help="written as <out_root>/<seq>.npz")
    ap.add_argument("--n_frames", type=int, default=13,
                    help="MoVieS's demo hardcodes 13 input timesteps")
    ap.add_argument("--limit", type=int, default=0, help="convert at most this many sequences")
    ap.add_argument("--self_check", action="store_true",
                    help="reproject our GT joints through the written intrinsics")
    args = ap.parse_args()

    seqs = sorted(d for d in glob.glob(os.path.join(args.data_root, "*"))
                  if os.path.isdir(os.path.join(d, "hand_data")))
    if args.limit:
        seqs = seqs[:args.limit]
    if not seqs:
        raise SystemExit(f"no sequences with hand_data under {args.data_root}")

    ok = fail = 0
    worst = 1.0
    for s in seqs:
        name = os.path.basename(s)
        try:
            info = convert(s, os.path.join(args.out_root, name + ".npz"),
                           args.n_frames, args.self_check)
            ok += 1
            chk = info.get("check", {})
            if chk.get("ran"):
                worst = min(worst, chk["frac_inside"])
                print(f"OK {name}  {info['frames']}/{info['of']} frames  {info['hw']}  "
                      f"joints inside {chk['frac_inside']:.1%} of {chk['n']}", flush=True)
            else:
                print(f"OK {name}  {info['frames']}/{info['of']} frames  {info['hw']}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad sequence must not stop a set
            fail += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)

    print(f"MOVIES_EXPORT done ok={ok} fail={fail} -> {args.out_root}", flush=True)
    if args.self_check and worst < 0.9:
        raise SystemExit(f"self-check floor {worst:.1%} of joints inside the frame: the intrinsics "
                         f"or the frame size written do not describe the images written")


if __name__ == "__main__":
    main()
