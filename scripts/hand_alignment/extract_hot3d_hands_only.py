"""
extract_hot3d_hands_only.py

Generate hand_meshes.pt for an EXISTING scene reconstruction, without re-running
NeoVerse (no GPU) and without projectaria's heavy model deps. Reuses the saved
predicted cameras (camera_params.json) from a prior scene export and warps the GT
MANO hands into that scene frame with the trajectory-based similarity scale.

This is the hand half of align_hand_to_neoverse.py, sliced out so it runs CPU-only
on the login node (projectaria has x86 wheels; the raw HOT3D sequence is on disk).

Usage (in the /tmp projectaria venv):
  python scripts/hand_alignment/extract_hot3d_hands_only.py \
    --seq /work/courses/3dv/team25/data/hot3d_aria/sequences/P0001_10a27bf7 \
    --scene_dir /work/scratch/dmonopoli/output_test_P0001_10a27bf7 \
    --out /tmp/hand_meshes.pt
"""
import argparse
import inspect
import json
import os
import sys

# chumpy (MANO loader dep) uses APIs removed in Python 3.11+. Shim before smplx.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
import torch

REPO = "/home/dmonopoli/FF-4DGS-Ego"
sys.path.insert(0, REPO)

from scripts.hand_vis_utils import (  # noqa: E402
    MANOModel, setup_vis_context, _frame_to_timecode, find_closest,
)
from projectaria_tools.core.sophus import SE3  # noqa: E402

# 90-deg sensor->display rotation about camera Z (matches align_hand_to_neoverse).
R_SENSOR_TO_DISPLAY = np.array([[0.0, -1.0, 0.0],
                                [1.0,  0.0, 0.0],
                                [0.0,  0.0, 1.0]])


def gt_camera_poses(ctx, frame_indices):
    headset_ts = ctx["headset_ts_sorted"]
    headset_poses = ctx["headset_poses"]
    hand_ts = ctx["hand_ts_sorted"]
    n_video = ctx["n_video"]
    T_dc = ctx["T_device_camera"].to_matrix()
    out = np.zeros((len(frame_indices), 4, 4), dtype=np.float64)
    for k, fi in enumerate(frame_indices):
        qtc = _frame_to_timecode(fi, n_video, hand_ts)
        h = find_closest(headset_ts, qtc)
        t_wd, q_wd = headset_poses[h]
        T_wd = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0].to_matrix()
        out[k] = T_wd @ T_dc
    return out


def recover_scale_from_camera_pairs(c2w_gt, c2w_pred):
    p_gt = c2w_gt[:, :3, 3]
    p_pred = c2w_pred[:, :3, 3]
    d_gt = np.linalg.norm(p_gt[:, None] - p_gt[None], axis=-1)
    d_pred = np.linalg.norm(p_pred[:, None] - p_pred[None], axis=-1)
    N = len(p_gt)
    span = float(np.linalg.norm(p_gt.max(0) - p_gt.min(0)))
    iu, ju = np.triu_indices(N, 1)
    mask = d_gt[iu, ju] > 0.05 * span
    if mask.sum() < 5:
        mask = np.ones_like(mask, dtype=bool)
    ratios = d_pred[iu[mask], ju[mask]] / d_gt[iu[mask], ju[mask]]
    return float(np.median(ratios)), float(np.std(ratios)), span


def per_frame_decompose(verts_world, k, gt_c2w, pred_c2w):
    w2c = np.linalg.inv(gt_c2w[k])
    c2w_p = pred_c2w[k]
    V = verts_world.shape[0]
    vh = np.concatenate([verts_world, np.ones((V, 1))], axis=1)
    vc = (w2c @ vh.T).T[:, :3]
    vd = (R_SENSOR_TO_DISPLAY @ vc.T).T
    disp = (c2w_p[:3, :3] @ vd.T).T
    return c2w_p[:3, 3].astype(np.float32), disp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="raw HOT3D sequence dir")
    ap.add_argument("--scene_dir", required=True, help="dir with camera_params.json (pred cams)")
    ap.add_argument("--mano", default=os.path.join(REPO, "models/MANO"))
    ap.add_argument("--frame_start", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mano = MANOModel(a.mano)
    ctx = setup_vis_context(a.seq, mano_model=mano)
    if ctx is None:
        raise SystemExit(f"setup_vis_context failed for {a.seq}")
    frames = list(range(a.frame_start, a.frame_start + a.num_frames))

    cam = json.load(open(os.path.join(a.scene_dir, "camera_params.json")))
    pred_c2w = np.array([np.asarray(e["matrix"], np.float64) for e in cam["extrinsics"]])
    if len(pred_c2w) != a.num_frames:
        print(f"  WARN: scene has {len(pred_c2w)} cams, using num_frames={a.num_frames}")

    gt_c2w = gt_camera_poses(ctx, frames)
    s, sstd, gt_span = recover_scale_from_camera_pairs(gt_c2w, pred_c2w)
    print(f"GT camera span = {gt_span:.4f} m  (if <0.05 m the scale is unreliable)")
    print(f"trajectory scale s = {s:.5f}  (std {sstd:.5f})")

    hand_ts = ctx["hand_ts_sorted"]
    hand_poses = ctx["hand_poses"]
    n_video = ctx["n_video"]
    meshes = []
    for k, fi in enumerate(frames):
        qtc = _frame_to_timecode(fi, n_video, hand_ts)
        hd = hand_poses[find_closest(hand_ts, qtc)]
        for hk, side in [("0", "left"), ("1", "right")]:
            if hk not in hd:
                continue
            verts, faces = mano.get_mesh(hd[hk], is_right=(side == "right"))
            pos, disp = per_frame_decompose(verts, k, gt_c2w, pred_c2w)
            vn = (pos + s * disp).astype(np.float32)
            meshes.append({
                "frame_offset": k, "frame_global": int(fi), "side": side,
                "pos_pred": pos, "displacement": disp, "vertices": vn,
                "default_scale": float(s), "faces": faces.astype(np.int32),
            })
    torch.save(meshes, a.out)
    print(f"wrote {len(meshes)} hand mesh(es) -> {a.out}")


if __name__ == "__main__":
    main()
