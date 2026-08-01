"""Dyn-HaMR output -> eval_worldspace_baseline prediction contract.

Converts Dyn-HaMR per-sequence output pickles/tensors into standard evaluation files:
    <seq>.pt containing:
        "cam_joints"   : [N, 2, 16, 3] metres, camera frame, SMPL-X 16 joint order
        "world_joints" : [N, 2, 16, 3] metres, world frame,  SMPL-X 16 joint order
        "valid"        : [N, 2] bool
"""
import argparse
import glob
import os
import torch
import numpy as np

RH = 1  # Right hand slot
NUM_JOINTS = 16
# MANO 21 -> SMPL-X 16 order
MANO2SMPLX16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def convert_seq(seq_name: str, pred_path: str, gt_dir: str, out_pred_dir: str):
    gt_cam_path = os.path.join(gt_dir, seq_name, "hand_data", "gt_joints_cache_cam_v2.pt")
    if not os.path.exists(gt_cam_path):
        return False
    gt_cam = torch.load(gt_cam_path, weights_only=True)
    N = gt_cam.shape[0]

    cam_joints = torch.full((N, 2, NUM_JOINTS, 3), float("nan"), dtype=torch.float32)
    world_joints = torch.full((N, 2, NUM_JOINTS, 3), float("nan"), dtype=torch.float32)
    valid = torch.zeros(N, 2, dtype=torch.bool)

    if os.path.exists(pred_path):
        pred_data = torch.load(pred_path, map_location="cpu")
        j_cam = pred_data.get("cam_joints", pred_data.get("pred_joints_cam"))
        j_world = pred_data.get("world_joints", pred_data.get("pred_joints_world"))

        if j_cam is not None:
            j_cam_np = np.asarray(j_cam, np.float32)
            T_pred = min(N, j_cam_np.shape[0])
            cam_joints[:T_pred, RH] = torch.from_numpy(j_cam_np[:T_pred, MANO2SMPLX16])
            valid[:T_pred, RH] = True

        if j_world is not None:
            j_world_np = np.asarray(j_world, np.float32)
            T_pred = min(N, j_world_np.shape[0])
            world_joints[:T_pred, RH] = torch.from_numpy(j_world_np[:T_pred, MANO2SMPLX16])

    out_file = os.path.join(out_pred_dir, f"{seq_name}.pt")
    torch.save({
        "cam_joints": cam_joints,
        "world_joints": world_joints,
        "valid": valid
    }, out_file)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynhamr_out", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pred_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    seqs = sorted(os.listdir(args.data_root))
    print(f"Converting Dyn-HaMR predictions across {len(seqs)} sequences...", flush=True)

    count = 0
    for seq in seqs:
        p_file = os.path.join(args.dynhamr_out, f"{seq}.pt")
        if convert_seq(seq, p_file, args.data_root, args.pred_dir):
            count += 1
    print(f"Converted {count} sequence predictions -> {args.pred_dir}", flush=True)


if __name__ == "__main__":
    main()
