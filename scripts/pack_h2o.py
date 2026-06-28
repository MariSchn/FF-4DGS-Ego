"""Stream-pack H2O sequences into one .npz per sequence (the inode fix).

H2O stores ~2,700 tiny files per sequence; the cluster caps scratch at ~100k
inodes (venv alone is 64k), so raw H2O cannot live there. This reads the ego
tar as a STREAM (``curl ... | python -m scripts.pack_h2o``), decoding each member
in memory and writing ONE compressed ``.npz`` per sequence — ~47 inodes for a whole
subject instead of ~127k, and the 35 GB tar never lands on disk.

Needs only PIL + numpy (no torch/cv2), so it runs on the interactive-cpu partition
(or the login node) concurrently with GPU training.

Per .npz (sequence): rgb[N,R,R,3] uint8, depth[N,R,R] f16 (metres), joints[N,128]
(H2O hand_pose: per hand valid+21x3, mm), mano[N,124] (valid+trans+pose48+beta10),
cam_pose[N,4,4] (cam_to_world), K[6] (fx,fy,cx,cy,w,h).

Usage:
    curl -sL -u "$U:$P" https://h2odataset.ethz.ch/data/dataset/subject1_ego_v1_1.tar.gz \
      | python3 -m scripts.pack_h2o --out /work/scratch/dmonopoli/h2o_packed --res 224
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile

import numpy as np
from PIL import Image


def _center_square_resize(img: Image.Image, res: int, nearest: bool = False) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((res, res), Image.NEAREST if nearest else Image.BILINEAR)


# ---------------------------------------------------------------------------
# High-res hand crop (the pose-resolution fix). Crops a square padded around the
# hand from the ORIGINAL full-res RGB (not the 224 px frame) so the hand encoder
# sees native detail. The bbox is derived from the GT 3D joints projected with K
# — mirrors scripts.eval_cmpjpe.bboxes_from_joints, but in full-frame pixels
# (no center-square-crop adjustment, since we crop the original image).
# ---------------------------------------------------------------------------

H2O_FULL_W, H2O_FULL_H = 1280, 720


def _hand_joints_from_vec(j128: np.ndarray):
    """H2O hand_pose [128] -> (left[21,3], right[21,3], lvalid, rvalid), metres, cam frame."""
    lvalid, rvalid = float(j128[0]), float(j128[64])
    left = j128[1:64].reshape(21, 3)
    right = j128[65:128].reshape(21, 3)
    return left, right, lvalid, rvalid


def _bbox_from_joints_fullres(joints21, K, pad: float = 1.6):
    """Project [21,3] cam-frame joints (m) with full-res K -> square px bbox.

    Returns (cx, cy, side) in original full-frame pixels, or None if degenerate.
    """
    fx, fy, cx_k, cy_k = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    z = np.clip(joints21[:, 2], 1e-3, None)
    u = fx * joints21[:, 0] / z + cx_k
    v = fy * joints21[:, 1] / z + cy_k
    x1, x2 = float(u.min()), float(u.max())
    y1, y2 = float(v.min()), float(v.max())
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * pad
    if not np.isfinite(side) or side < 2.0:
        return None
    return cx, cy, side


def _crop_square(img: Image.Image, cx: float, cy: float, side: float, out_px: int) -> np.ndarray:
    """Crop a square (cx,cy,side) px box from `img`, pad out-of-frame, resize to out_px."""
    half = side / 2.0
    left, top = int(round(cx - half)), int(round(cy - half))
    right, bot = int(round(cx + half)), int(round(cy + half))
    crop = img.crop((left, top, right, bot))  # PIL pads out-of-frame with black
    crop = crop.resize((out_px, out_px), Image.BILINEAR)
    return np.asarray(crop, np.uint8)


def _hand_crops_for_frame(rgb_full: Image.Image, j128: np.ndarray, K, out_px: int):
    """Return (crop_L[out_px,out_px,3], crop_R[...]) uint8 for one frame.

    Empty (zeros) crop where the hand is absent or the bbox is degenerate.
    """
    left, right, lvalid, rvalid = _hand_joints_from_vec(j128)
    zeros = np.zeros((out_px, out_px, 3), np.uint8)
    cl, cr = zeros, zeros
    if K is not None and np.asarray(K).size >= 4:
        if lvalid > 0.5:
            bb = _bbox_from_joints_fullres(left, K)
            if bb is not None:
                cl = _crop_square(rgb_full, *bb, out_px)
        if rvalid > 0.5:
            bb = _bbox_from_joints_fullres(right, K)
            if bb is not None:
                cr = _crop_square(rgb_full, *bb, out_px)
    return cl, cr


def _flush(seq: str, buf: dict, K, out_dir: str, res: int, hand_crop_px: int = 0) -> None:
    if not buf:
        return
    frames = sorted(buf.keys())
    n = len(frames)
    rgb = np.zeros((n, res, res, 3), np.uint8)
    depth = np.zeros((n, res, res), np.float16)
    joints = np.zeros((n, 128), np.float32)
    mano = np.zeros((n, 124), np.float32)
    cam_pose = np.zeros((n, 4, 4), np.float32)
    save_hires = hand_crop_px > 0
    if save_hires:
        hand_crop_L = np.zeros((n, hand_crop_px, hand_crop_px, 3), np.uint8)
        hand_crop_R = np.zeros((n, hand_crop_px, hand_crop_px, 3), np.uint8)
    for i, f in enumerate(frames):
        b = buf[f]
        if "rgb" in b: rgb[i] = b["rgb"]
        if "depth" in b: depth[i] = b["depth"]
        if "joints" in b: joints[i, : b["joints"].size] = b["joints"]
        if "mano" in b: mano[i, : b["mano"].size] = b["mano"]
        if "cam_pose" in b: cam_pose[i] = b["cam_pose"].reshape(4, 4)
        # High-res hand crops from the original full-res RGB kept in the buffer.
        if save_hires and "rgb_full" in b and "joints" in b:
            cl, cr = _hand_crops_for_frame(b["rgb_full"], joints[i], K, hand_crop_px)
            hand_crop_L[i], hand_crop_R[i] = cl, cr
    name = seq.replace("/", "_")
    arrays = dict(
        rgb=rgb, depth=depth, joints=joints, mano=mano, cam_pose=cam_pose,
        K=np.asarray(K, np.float32) if K is not None else np.zeros(6, np.float32),
    )
    if save_hires:
        arrays["hand_crop_L"] = hand_crop_L
        arrays["hand_crop_R"] = hand_crop_R
    np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
    print(f"packed {name}: {n} frames" + (f" (+hires {hand_crop_px}px)" if save_hires else ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--hand_crop_px", type=int, default=0,
                    help="if >0, also save native hand crops (hand_crop_L/R) at this px "
                         "(the hi-res hand branch; e.g. 256). 0 = off (224 path unchanged).")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = args.res
    HC = args.hand_crop_px

    tar = tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz")
    cur_seq, buf, K = None, {}, None
    npacked = 0
    for m in tar:
        if not m.isfile():
            continue
        parts = m.name.split("/")
        if "cam4" not in parts:
            continue
        ci = parts.index("cam4")
        seq = "/".join(parts[:ci])              # e.g. subject1_ego/h1/0
        modality = parts[ci + 1] if len(parts) > ci + 1 else ""
        fname = parts[ci + 2] if len(parts) > ci + 2 else ""

        if seq != cur_seq:
            if cur_seq is not None:
                _flush(cur_seq, buf, K, args.out, R, HC); npacked += 1
            cur_seq, buf, K = seq, {}, None

        try:
            data = tar.extractfile(m).read()
        except Exception:
            continue

        if modality == "cam_intrinsics.txt":
            try: K = np.loadtxt(io.StringIO(data.decode()))
            except Exception: K = None
            continue
        frame = re.sub(r"\.[^.]+$", "", fname)
        if not frame:
            continue
        b = buf.setdefault(frame, {})
        try:
            if modality == "rgb":
                full = Image.open(io.BytesIO(data)).convert("RGB")
                b["rgb"] = np.asarray(_center_square_resize(full, R), np.uint8)
                if HC > 0:
                    # Keep the original full-res frame so _flush can crop the hand
                    # from native pixels once the joints for this frame arrive.
                    b["rgb_full"] = full
            elif modality == "depth":
                d = _center_square_resize(Image.open(io.BytesIO(data)), R, nearest=True)
                b["depth"] = (np.asarray(d, np.float32) / 1000.0).astype(np.float16)  # mm -> m
            elif modality == "hand_pose":
                b["joints"] = np.loadtxt(io.StringIO(data.decode()), dtype=np.float32)
            elif modality == "hand_pose_mano":
                b["mano"] = np.loadtxt(io.StringIO(data.decode()), dtype=np.float32)
            elif modality == "cam_pose":
                b["cam_pose"] = np.loadtxt(io.StringIO(data.decode()), dtype=np.float32)
        except Exception:
            pass

    if cur_seq is not None:
        _flush(cur_seq, buf, K, args.out, R, HC); npacked += 1
    print(f"=== PACK DONE: {npacked} sequences -> {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
