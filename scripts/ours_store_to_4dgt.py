#!/usr/bin/env python3
"""Write one of our sequences in the format 4DGT reads, so its released checkpoint can be run on
our data without touching its repository.

4DGT's input contract, from docs/data.md, is a directory holding `images/` and a `transforms.json`
whose "frames" list carries per-frame fx, fy, cx, cy, w, h, image_path, transform_matrix (4x4) and
timestamp. That is all it needs: no VRS, no Aria tooling, no rectification, because our store is
already a plain pinhole with no distortion.

    <out>/<seq>/images/000000.png ...
    <out>/<seq>/transforms.json

THE ONE THING THAT CAN SILENTLY RUIN THIS IS THE POSE CONVENTION, and this project has already
paid for that lesson once: every world metric we reported for months applied the camera trajectory
backwards, because a tensor labelled camera-to-world was world-to-camera. So nothing here is
assumed.

  * Our store holds `cam_extrinsics_cache.pt` as T_camera_world (w2c). transforms.json wants
    camera-to-world, so we invert, and the inversion is the DEFAULT rather than a flag.
  * transforms.json descends from the NeRF/instant-ngp convention, whose camera looks down -z with
    +y up (OpenGL), while ours looks down +z with +y down (OpenCV). --axes opengl applies
    diag(1,-1,-1) on the right, which is that change of basis; --axes opencv writes the matrix
    unchanged. Neither is guessed: --self_check reprojects our own ground-truth hand joints
    through the matrix actually written and reports how many land inside the image, which is the
    same test that resolved the Re:InterHand layout.

A convention error shows up as a self-check pass rate near zero, and as 4DGT renderings that look
like noise. Run --self_check on one sequence before converting a set.

    python -m scripts.ours_store_to_4dgt --data_root <store> --out_root <out> --limit 1 --self_check
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

# NeRF/OpenGL cameras look down -z with +y up; ours look down +z with +y down. Right-multiplying a
# camera-to-world matrix by this flips the camera's own y and z axes, which is exactly that change
# of basis and leaves the camera CENTRE untouched.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def load_seq(seq_dir: str):
    hd = os.path.join(seq_dir, "hand_data")
    intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").numpy()
    w2c = torch.load(os.path.join(hd, "cam_extrinsics_cache.pt"), map_location="cpu").numpy()
    j_cam = None
    p = os.path.join(hd, "gt_joints_cache_cam_v2.pt")
    if os.path.isfile(p):
        j_cam = torch.load(p, map_location="cpu").numpy()
    return intr.astype(np.float64), w2c.astype(np.float64), j_cam


def self_check(intr, w2c, j_cam, w, h) -> dict:
    """Reproject our own camera-frame joints and report how many land in the image.

    This does not test 4DGT. It tests that the intrinsics and the frame size we are about to write
    describe the images we are about to write, which is the part a wrong assumption would break
    silently.
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


def convert(seq_dir: str, out_dir: str, axes: str, fps: float, do_check: bool) -> dict:
    intr, w2c, j_cam = load_seq(seq_dir)
    f, cx, cy = float(intr[0]), float(intr[1]), float(intr[2])

    vid = os.path.join(seq_dir, "video_main_rgb.mp4")
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        return {"seq": os.path.basename(seq_dir), "error": f"cannot open {vid}"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    frames, i = [], 0
    while i < len(w2c):
        ok, img = cap.read()
        if not ok:
            break
        name = f"{i:06d}.png"
        cv2.imwrite(os.path.join(img_dir, name), img)
        c2w = np.linalg.inv(w2c[i])
        if axes == "opengl":
            c2w = c2w @ CV_TO_GL
        frames.append({
            "fx": f, "fy": f, "cx": cx, "cy": cy, "w": w, "h": h,
            "image_path": f"images/{name}",
            "transform_matrix": c2w.tolist(),
            "timestamp": float(i) / fps * 1e9,   # nanoseconds, as in their Aria examples
        })
        i += 1
    cap.release()

    with open(os.path.join(out_dir, "transforms.json"), "w") as fh:
        json.dump({"frames": frames}, fh, indent=1)

    rep = {"seq": os.path.basename(seq_dir), "n_frames": len(frames),
           "n_extrinsics": int(len(w2c)), "w": w, "h": h,
           "fx": f, "cx": cx, "cy": cy, "axes": axes}
    if len(frames) != len(w2c):
        # Same rule as every store we write: the frame index must mean the same thing in the video
        # and in the poses, so a mismatch is recorded rather than absorbed.
        rep["truncated"] = True
    if do_check:
        rep["self_check"] = self_check(intr, w2c, j_cam, w, h)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="our store, e.g. hoi4d_test157_detv3")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--axes", choices=["opengl", "opencv"], default="opengl",
                    help="opengl (NeRF/instant-ngp convention, the transforms.json default) or "
                         "opencv (write our matrix unchanged). If 4DGT renders noise, this is the "
                         "first thing to flip.")
    ap.add_argument("--self_check", action="store_true")
    a = ap.parse_args()

    seqs = sorted(d for d in os.listdir(a.data_root)
                  if os.path.isdir(os.path.join(a.data_root, d, "hand_data")))
    if a.limit:
        seqs = seqs[:a.limit]
    os.makedirs(a.out_root, exist_ok=True)

    reports = []
    for s in seqs:
        r = convert(os.path.join(a.data_root, s), os.path.join(a.out_root, s),
                    a.axes, a.fps, a.self_check)
        reports.append(r)
        chk = r.get("self_check") or {}
        extra = ""
        if chk.get("ran"):
            extra = f" | joints inside image: {chk['frac_inside']*100:.1f}% of {chk['n']}"
        print(f"{r['seq']}: {r.get('n_frames', 0)} frames {r.get('w')}x{r.get('h')}"
              f"{' TRUNCATED' if r.get('truncated') else ''}{extra}", flush=True)

    with open(os.path.join(a.out_root, "_4dgt_export_report.json"), "w") as fh:
        json.dump(reports, fh, indent=1)
    print(f"\nwrote {len(reports)} sequences -> {a.out_root}")
    print("If any self-check pass rate is near zero, the intrinsics do not describe these images "
          "and nothing downstream is worth running.")


if __name__ == "__main__":
    main()
