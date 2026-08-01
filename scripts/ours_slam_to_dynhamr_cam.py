"""Our SLAM camera trajectory -> Dyn-HaMR `cameras.npz`, so Dyn-HaMR can be run with OUR
camera and skip its DROID-SLAM stage entirely (the DROID CUDA compile is the build blocker).

Dyn-HaMR loads the camera from `<root>/dynhamr/cameras/<seq>/shot-<idx>/cameras.npz` via
`load_cameras_npz` (dyn-hamr/data/dataset.py), which reads keys:
    height (int), width (int), focal (float), w2c [N,4,4] (world->cam), optional intrins [N,4] = [fx,fy,cx,cy].
It then sets cam_R = w2c[:, :3, :3], cam_t = w2c[:, :3, 3].

Our camera is the SAME trajectory the composed "<method>+SLAM" rows use: it is recovered
per-frame from our completed HaWoR run by Kabsch on HaWoR's own cam_joints <-> world_joints
(world = R @ cam + t; residual ~0). That gives a cam->world rotation R and translation t, so
    w2c = [[R^T, -R^T t], [0, 1]].
Intrinsics come from the HOI4D per-seq cache (`hand_data/cam_intrinsics.pt` = [focal, cx, cy]).

Usage:
    python -m scripts.ours_slam_to_dynhamr_cam \
        --slam_pred_dir <hawor_preds> --test_root <hoi4d test seqs> \
        --out_root <root>/dynhamr/cameras --shot_idx 0
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from scripts.build_slam_baseline import recover_se3


def w2c_from_cam2world(R: torch.Tensor, t: torch.Tensor, ok: torch.Tensor) -> np.ndarray:
    """R[N,3,3], t[N,3] with world = R@cam + t  ->  w2c[N,4,4] (world->cam).
    Frames where the trajectory is unavailable are filled with the last valid pose so the
    sequence stays contiguous (Dyn-HaMR keys frames by index, not by validity)."""
    n = R.shape[0]
    w2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    last = np.eye(4, dtype=np.float32)
    for k in range(n):
        if bool(ok[k]):
            Rk = R[k].double()
            Rt = Rk.T
            tk = t[k].double()
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = Rt.numpy().astype(np.float32)
            M[:3, 3] = (-Rt @ tk).numpy().astype(np.float32)
            last = M
        w2c[k] = last
    return w2c


def intrinsics_for(seq_dir: str) -> tuple[float, int, int]:
    """Decoded-frame intrinsics matching build_native_baseline_preds + dynhamr_export_from_boxes:
    the store-res focal f is rescaled to the decoded frame (focal = f*Wimg/(2*cx)) and the principal
    point is the decoded image centre (cx=Wimg/2, cy=Himg/2). Returns (focal, Wimg, Himg). This MUST
    agree with the images Dyn-HaMR reads and the 2D-keypoint projection, else the optimisation is fed
    inconsistent camera/image geometry."""
    k = torch.load(os.path.join(seq_dir, "hand_data", "cam_intrinsics.pt"),
                   map_location="cpu").float().flatten()
    f, cx0 = float(k[0]), float(k[1])
    cap = cv2.VideoCapture(os.path.join(seq_dir, "video_main_rgb.mp4"))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode video for {seq_dir}")
    Himg, Wimg = fr.shape[:2]
    focal = f * (Wimg / (2.0 * cx0))
    return focal, int(Wimg), int(Himg)


def convert_seq(seq: str, slam_pred_dir: str, test_root: str, out_root: str, shot_idx: int) -> dict | None:
    pred_path = os.path.join(slam_pred_dir, seq + ".pt")
    seq_dir = os.path.join(test_root, seq)
    if not (os.path.exists(pred_path) and os.path.isdir(seq_dir)):
        return None
    hw = torch.load(pred_path, map_location="cpu", weights_only=False)
    R, t, ok, res = recover_se3(hw["cam_joints"].float(), hw["world_joints"].float(), hw["valid"].bool())
    if res > 1e-3:
        print(f"SEQ_WARN {seq}: HaWoR SE(3) residual {res*1000:.2f} mm > 1 mm", flush=True)
    w2c = w2c_from_cam2world(R, t, ok)
    focal, width, height = intrinsics_for(seq_dir)
    cx, cy = width / 2.0, height / 2.0
    n = w2c.shape[0]
    intrins = np.tile(np.array([focal, focal, cx, cy], np.float32), (n, 1))

    out_dir = os.path.join(out_root, seq, f"shot-{shot_idx}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "cameras.npz"),
             w2c=w2c, focal=np.float32(focal), width=np.int32(width),
             height=np.int32(height), intrins=intrins)
    return {"seq": seq, "N": n, "valid": int(ok.sum()), "res_mm": res * 1000, "wh": (width, height)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slam_pred_dir", required=True, help="HaWoR preds dir (SLAM-trajectory source)")
    ap.add_argument("--test_root", required=True, help="HOI4D test seqs (for per-seq intrinsics)")
    ap.add_argument("--out_root", required=True, help="<root>/dynhamr/cameras")
    ap.add_argument("--shot_idx", type=int, default=0)
    a = ap.parse_args()

    seqs = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.slam_pred_dir, "*.pt")))
    os.makedirs(a.out_root, exist_ok=True)
    done, worst = 0, 0.0
    for sq in seqs:
        r = convert_seq(sq, a.slam_pred_dir, a.test_root, a.out_root, a.shot_idx)
        if r:
            done += 1
            worst = max(worst, r["res_mm"])
            print(f"[{done}] {r['seq']} N={r['N']} valid={r['valid']} wh={r['wh']} res={r['res_mm']:.3f}mm", flush=True)
    print(f"OURS_SLAM_TO_DYNHAMR_CAM_DONE wrote {done}/{len(seqs)} cameras.npz -> {a.out_root} "
          f"(max SE3 residual {worst:.3f} mm)", flush=True)


if __name__ == "__main__":
    main()
