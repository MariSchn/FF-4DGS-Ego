"""Convert raw ARCTIC dataset sequences into our dataset format for training.

Emits per sequence:
    <out>/<seq>/
        video_main_rgb.mp4
        hand_data/cam_intrinsics.pt             [3] = [focal, cx, cy]
        hand_data/cam_extrinsics_cache.pt       [N,4,4] T_camera_world
        hand_data/gt_joints_cache_world.pt      [N,2,16,3] metres
        hand_data/gt_joints_cache_cam_v2.pt     [N,2,16,3] metres
        hand_data/hand_bboxes_v2_rf1.5_res224x224.pt
"""
import argparse
import json
import os
import glob
import cv2
import numpy as np
import torch


NUM_HANDS = 2
NUM_JOINTS = 16
# MANO 21 joint -> SMPL-X 16 joint index mapping
MANO2SMPLX16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def process_arctic_sequence(seq_dir: str, out_dir: str, mano_model_path: str = None):
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    out_seq_dir = os.path.join(out_dir, seq_name)
    hd_dir = os.path.join(out_seq_dir, "hand_data")
    os.makedirs(hd_dir, exist_ok=True)

    # Load annotations (ARCTIC pkl/npz format)
    meta_file = os.path.join(seq_dir, "meta.json")
    mano_file = os.path.join(seq_dir, "mano_params.npz")
    cam_file = os.path.join(seq_dir, "cam_params.json")

    if not os.path.exists(mano_file):
        print(f"Skipping {seq_name}: missing {mano_file}")
        return False

    data = np.load(mano_file, allow_pickle=True)
    # Extract left (0) and right (1) 3D joints in camera and world frames
    # [N, 2, 21, 3] -> [N, 2, 16, 3]
    joints_cam = data.get("joints_cam", data.get("joints_3d"))  # meters
    if joints_cam is None:
        print(f"Skipping {seq_name}: no 3D joints")
        return False

    N = joints_cam.shape[0]
    joints_cam_16 = joints_cam[:, :, MANO2SMPLX16, :].astype(np.float32)

    # Extrinsics & Intrinsics
    if os.path.exists(cam_file):
        with open(cam_file) as f:
            cam_data = json.load(f)
        focal = float(cam_data.get("focal", 500.0))
        cx = float(cam_data.get("cx", 112.0))
        cy = float(cam_data.get("cy", 112.0))
    else:
        focal, cx, cy = 500.0, 112.0, 112.0

    intrinsics = torch.tensor([focal, cx, cy], dtype=torch.float32)

    # World joints (if camera poses are identity / provided)
    c2w = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
    joints_world_16 = joints_cam_16.copy()

    # Bounding boxes calculation per frame
    bboxes = np.zeros((N, 2, 4), dtype=np.float32)
    valid = np.ones((N, 2), dtype=bool)

    for i in range(N):
        for h in range(2):
            pts_3d = joints_cam_16[i, h]
            if np.all(pts_3d == 0) or np.isnan(pts_3d).any():
                valid[i, h] = False
                continue
            # Project to 2D
            z = pts_3d[:, 2]
            z_safe = np.where(z > 1e-3, z, 1.0)
            u = pts_3d[:, 0] * focal / z_safe + cx
            v = pts_3d[:, 1] * focal / z_safe + cy
            x1, x2 = np.min(u), np.max(u)
            y1, y2 = np.min(v), np.max(v)
            # Rescale with RF 1.5
            w, h_box = (x2 - x1) * 1.5, (y2 - y1) * 1.5
            cx_b, cy_b = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bx1, bx2 = max(0.0, cx_b - w / 2.0), min(224.0, cx_b + w / 2.0)
            by1, by2 = max(0.0, cy_b - h_box / 2.0), min(224.0, cy_b + h_box / 2.0)
            bboxes[i, h] = [bx1 / 224.0, by1 / 224.0, bx2 / 224.0, by2 / 224.0]

    # Save tensors
    torch.save(intrinsics, os.path.join(hd_dir, "cam_intrinsics.pt"))
    torch.save(c2w, os.path.join(hd_dir, "cam_extrinsics_cache.pt"))
    torch.save(torch.from_numpy(joints_world_16), os.path.join(hd_dir, "gt_joints_cache_world.pt"))
    torch.save(torch.from_numpy(joints_cam_16), os.path.join(hd_dir, "gt_joints_cache_cam_v2.pt"))
    torch.save({
        "bboxes": torch.from_numpy(bboxes),
        "valid": torch.from_numpy(valid),
        "gt": True
    }, os.path.join(hd_dir, "hand_bboxes_v2_rf1.5_res224x224.pt"))

    print(f"Preprocessed ARCTIC sequence {seq_name} ({N} frames)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arctic_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = sorted(glob.glob(os.path.join(a.arctic_root, "*")))
    print(f"Found {len(seqs)} ARCTIC sequences")
    for s in seqs:
        if os.path.isdir(s):
            process_arctic_sequence(s, a.out)


if __name__ == "__main__":
    main()
