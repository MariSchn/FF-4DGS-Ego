"""Convert raw OakInk2 dataset sequences (CVPR 2024) into our dataset format for held-out evaluation.

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


def process_oakink2_sequence(seq_dir: str, out_dir: str):
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    out_seq_dir = os.path.join(out_dir, seq_name)
    hd_dir = os.path.join(out_seq_dir, "hand_data")
    os.makedirs(hd_dir, exist_ok=True)

    anno_file = os.path.join(seq_dir, "anno.pkl")
    if not os.path.exists(anno_file):
        # Check npz or json alternative
        anno_file = os.path.join(seq_dir, "anno.npz")
        if not os.path.exists(anno_file):
            print(f"Skipping {seq_name}: no annotation file found")
            return False

    if anno_file.endswith(".npz"):
        data = np.load(anno_file, allow_pickle=True)
    else:
        import pickle
        with open(anno_file, "rb") as f:
            data = pickle.load(f)

    # Extract 3D joints in camera coordinates: shape [N, 2, 21, 3] in meters
    joints_cam = data.get("joints_cam", data.get("hand_pts_3d"))
    if joints_cam is None:
        print(f"Skipping {seq_name}: missing 3D joint data")
        return False

    N = joints_cam.shape[0]
    joints_cam_16 = joints_cam[:, :, MANO2SMPLX16, :].astype(np.float32)

    # Extract intrinsics
    cam_intr = data.get("cam_intr", [500.0, 112.0, 112.0])
    if isinstance(cam_intr, (list, np.ndarray)):
        cam_intr = np.asarray(cam_intr).flatten()
        focal, cx, cy = float(cam_intr[0]), float(cam_intr[1]), float(cam_intr[2])
    else:
        focal, cx, cy = 500.0, 112.0, 112.0

    intrinsics = torch.tensor([focal, cx, cy], dtype=torch.float32)
    c2w = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
    joints_world_16 = joints_cam_16.copy()

    # Bounding boxes and validity flags
    bboxes = np.zeros((N, 2, 4), dtype=np.float32)
    valid = np.ones((N, 2), dtype=bool)

    for i in range(N):
        for h in range(2):
            pts_3d = joints_cam_16[i, h]
            if np.all(pts_3d == 0) or np.isnan(pts_3d).any():
                valid[i, h] = False
                continue
            z = pts_3d[:, 2]
            z_safe = np.where(z > 1e-3, z, 1.0)
            u = pts_3d[:, 0] * focal / z_safe + cx
            v = pts_3d[:, 1] * focal / z_safe + cy
            x1, x2 = np.min(u), np.max(u)
            y1, y2 = np.min(v), np.max(v)
            w, h_box = (x2 - x1) * 1.5, (y2 - y1) * 1.5
            cx_b, cy_b = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bx1, bx2 = max(0.0, cx_b - w / 2.0), min(224.0, cx_b + w / 2.0)
            by1, by2 = max(0.0, cy_b - h_box / 2.0), min(224.0, cy_b + h_box / 2.0)
            bboxes[i, h] = [bx1 / 224.0, by1 / 224.0, bx2 / 224.0, by2 / 224.0]

    # Save output tensors
    torch.save(intrinsics, os.path.join(hd_dir, "cam_intrinsics.pt"))
    torch.save(c2w, os.path.join(hd_dir, "cam_extrinsics_cache.pt"))
    torch.save(torch.from_numpy(joints_world_16), os.path.join(hd_dir, "gt_joints_cache_world.pt"))
    torch.save(torch.from_numpy(joints_cam_16), os.path.join(hd_dir, "gt_joints_cache_cam_v2.pt"))
    torch.save({
        "bboxes": torch.from_numpy(bboxes),
        "valid": torch.from_numpy(valid),
        "gt": True
    }, os.path.join(hd_dir, "hand_bboxes_v2_rf1.5_res224x224.pt"))

    print(f"Preprocessed OakInk2 sequence {seq_name} ({N} frames)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oakink2_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = sorted(glob.glob(os.path.join(a.oakink2_root, "*")))
    print(f"Found {len(seqs)} OakInk2 sequences")
    for s in seqs:
        if os.path.isdir(s):
            process_oakink2_sequence(s, a.out)


if __name__ == "__main__":
    main()
