"""Run the full HaWoR pipeline on one video and dump per-frame hand joints.

Reuses HaWoR's own functions (detect_track -> motion_estimation -> SLAM -> infiller),
exactly as demo.py, then converts the world-frame MANO joints to the CAMERA frame
using the SLAM cameras and saves them for C-MPJPE. No rendering / aitviewer needed.

Output npz (per video):
  joints_world [2,T,21,3]  world-frame 21-joint MANO (16 base + 5 tips), metres
  joints_cam   [2,T,21,3]  camera-frame (R_w2c @ Xw + t_w2c)
  valid        [2,T]       HaWoR per-frame validity
  img_focal    scalar      focal HaWoR estimated/used
  hand_order   "[left=0,right=1]"
Run with cwd = HaWoR repo root.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam

# MANO-21 (smplx joints + 5 tip vertices) order produced by run_mano's joints3d.
# run_mano returns joints3d as the 16 kinematic joints already? We extract 21 to match
# GT (16 base + 5 fingertips). MANO tip vertex indices on the MANO mesh:
MANO_TIP_VERTS = [745, 317, 444, 556, 673]  # thumb,index,middle,ring,pinky (standard)


def joints21_from_mano(out):
    """out from run_mano/run_mano_left -> [T,21,3]: 16 kinematic joints + 5 tip verts.

    run_mano returns 'joints3d' [B,T,21,3] already in the manopth-21 layout for HaWoR
    (16 base + 5 tips); if only 'joints' (16) present, append tip verts. We detect by
    shape.
    """
    j = out["joints3d"] if "joints3d" in out else out["joints"]
    j = torch.as_tensor(j)
    if j.dim() == 4:      # [B,T,J,3] -> drop B
        j = j[0]
    if j.shape[-2] == 21:
        return j
    # 16 joints + append tip verts from 'vertices'
    v = torch.as_tensor(out["vertices"])
    if v.dim() == 4:
        v = v[0]
    tips = v[:, MANO_TIP_VERTS, :]
    return torch.cat([j, tips], dim=-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--img_focal", type=float, default=None)
    ap.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    ap.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.input_type = "file"
    args.vis_mode = "cam"

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all)

    # world-frame MANO joints for both hands. hand2idx: left=0, right=1
    T = pred_trans.shape[1]
    out_world = np.zeros((2, T, 21, 3), np.float32)
    for hand_idx, runner in ((0, run_mano_left), (1, run_mano)):
        g = runner(pred_trans[hand_idx:hand_idx + 1], pred_rot[hand_idx:hand_idx + 1],
                   pred_hand_pose[hand_idx:hand_idx + 1], betas=pred_betas[hand_idx:hand_idx + 1])
        out_world[hand_idx] = joints21_from_mano(g).cpu().numpy()

    # camera frame: Xc = R_w2c @ Xw + t_w2c (SLAM cams already scaled to metres)
    Rwc = R_w2c.float().cpu().numpy()   # [T,3,3]
    twc = t_w2c.float().cpu().numpy()   # [T,3]
    Tc = min(T, Rwc.shape[0])
    out_cam = np.zeros((2, Tc, 21, 3), np.float32)
    for h in range(2):
        for t in range(Tc):
            out_cam[h, t] = (Rwc[t] @ out_world[h, t].T).T + twc[t]

    valid = np.asarray(pred_valid.cpu().numpy() if torch.is_tensor(pred_valid) else pred_valid)
    np.savez_compressed(
        args.out,
        joints_world=out_world[:, :Tc],
        joints_cam=out_cam,
        valid=valid,
        img_focal=float(img_focal),
        start_idx=int(start_idx), end_idx=int(end_idx),
        hand_order="left=0,right=1",
    )
    print(f"[infer] wrote {args.out}  T={Tc} focal={img_focal}")


if __name__ == "__main__":
    main()
