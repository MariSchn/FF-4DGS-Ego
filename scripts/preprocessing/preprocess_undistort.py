"""
preprocess_undistort.py — Undistort Hot3D to pinhole + rebuild lens-dependent caches.

Walks every sequence under --source_root and writes a parallel structure under
--dest_root, with:
  * video_main_rgb.mp4 — fisheye → pinhole at the chosen focal length
  * hand_data/gt_joints_2d_cache.pt — 2D joints projected via the new pinhole
  * hand_data/hand_bboxes_v2_rf{X}_res{H}x{W}.pt — bbox from pinhole-projected mesh
  * hand_data/cam_intrinsics.pt — pinhole [f, cx, cy] (overwritten, not symlinked)
  * Symlinks for every lens-independent file (trajectories, JSONL hands, world-/
    camera-frame 3D joint caches, T_camera_world cache, calibration JSONL)
  * .preprocessing_meta.json marker for downstream sanity checks

`train_hand_head.py` consumes this exactly like the raw dataset — every cache
is precomputed so `HOT3DHandDataset.__init__` never falls back to rebuilding
with the fisheye projection.

Usage:
    python scripts/preprocessing/preprocess_undistort.py \\
        --source_root /home/marian/3dv/data/hot3d_aria/raw_data \\
        --dest_root   /home/marian/3dv/data/hot3d_aria/preprocessed_pinhole_f609 \\
        --focal 609.78 --rescale_factor 1.5 --resolution 224
"""

import argparse
import bisect
import datetime
import inspect
import json
import os
import sys
from pathlib import Path

# chumpy (pulled in by smplx when unpickling the MANO .pkl) still calls
# inspect.getargspec, removed in Python 3.11+. Restore it before any MANO load
# so chumpy imports on the py3.12 preprocessing venv. Standard chumpy shim.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import torch
from decord import VideoReader
from projectaria_tools.core.sophus import SE3

from scripts.preprocessing.pinhole_undistort_ops import (
    ARIA_FRAME_SIZE,
    build_pinhole_target,
    undistort_mp4_frame,
)
from scripts.hand_vis_utils import (
    MANOModel,
    find_closest,
    load_camera_calibration,
    load_hand_poses,
    load_headset_trajectory,
    project_vertices,
)


SCRIPT_VERSION = "1.0"
NUM_HANDS = 2
HAND_PARAM_DIM = 32


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source_root", required=True,
                   help="Root containing raw Hot3D sequences (each is its own subdir).")
    p.add_argument("--dest_root", required=True,
                   help="Destination root for the preprocessed dataset.")
    p.add_argument("--focal", type=float, default=609.78,
                   help="Pinhole focal length (px) at 1408x1408. Default = Aria GT focal.")
    p.add_argument("--rescale_factor", type=float, default=1.5,
                   help="Hand-bbox padding factor (HaMeR default 2.0; "
                        "train_hand_head.yaml default 1.5).")
    p.add_argument("--resolution", type=int, default=224,
                   help="Training resolution (used for bbox cache filename).")
    p.add_argument("--sequences", default="all",
                   help="'all' or comma-separated list e.g. P0001_10a27bf7,P0002_016222d1.")
    p.add_argument("--mano_model_folder", default="models/MANO")
    p.add_argument("--video_fps", type=int, default=30,
                   help="FPS for the output MP4. Aria captures at 30.")
    p.add_argument("--force", action="store_true",
                   help="Re-preprocess sequences whose marker matches the params.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (replicate dataset cache-building logic, but lens-aware)
# ---------------------------------------------------------------------------

def build_gt_per_frame_world(hand_poses, hand_ts_sorted, n_video):
    """Build [n_video, 64] world-frame MANO params, one per video frame via
    linear interpolation of the JSONL timestamps (same mapping as
    HOT3DHandDataset.__init__).
    """
    ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]

    def _closest_ts(frame_i):
        frac = frame_i / max(n_video - 1, 1)
        query = int(ts_start + frac * (ts_end - ts_start))
        idx = bisect.bisect_left(hand_ts_sorted, query)
        if idx == 0:
            return hand_ts_sorted[0]
        if idx >= len(hand_ts_sorted):
            return hand_ts_sorted[-1]
        before, after = hand_ts_sorted[idx - 1], hand_ts_sorted[idx]
        return before if (query - before) <= (after - query) else after

    def _hand_to_vec(hand_data):
        vecs = []
        for hand_id in ["0", "1"]:
            h = hand_data.get(hand_id, {})
            if h:
                vecs.append(torch.cat([
                    torch.tensor(h["wrist_xform"]["t_xyz"],  dtype=torch.float32),
                    torch.tensor(h["wrist_xform"]["q_wxyz"], dtype=torch.float32),
                    torch.tensor(h["pose"],                  dtype=torch.float32),
                    torch.tensor(h["betas"],                 dtype=torch.float32),
                ]))
            else:
                vecs.append(torch.zeros(HAND_PARAM_DIM))
        return torch.cat(vecs)

    out = []
    for frame_i in range(n_video):
        ts = _closest_ts(frame_i)
        out.append(_hand_to_vec(hand_poses[ts]))
    return out  # list of [64] tensors


def compute_seq_joints_world(mano, gt_per_frame_world):
    """Run MANO per (frame, hand) on world-frame params → [N, 2, 16, 3] joints."""
    out = []
    for frame_p in gt_per_frame_world:
        frame = []
        for h_idx in range(NUM_HANDS):
            p = frame_p[h_idx * HAND_PARAM_DIM:(h_idx + 1) * HAND_PARAM_DIM]
            if p.abs().sum() < 1e-6:
                j3d = np.zeros((16, 3), dtype=np.float32)
            else:
                j3d = mano.get_joints_from_tensor(p, is_right=(h_idx == 1), return_tensor=False)
                if isinstance(j3d, np.ndarray) and j3d.ndim == 3:
                    j3d = j3d.squeeze(0)
                elif torch.is_tensor(j3d) and j3d.dim() == 3:
                    j3d = j3d.squeeze(0)
            frame.append(torch.as_tensor(j3d, dtype=torch.float32))
        out.append(torch.stack(frame))
    return torch.stack(out)  # [N, 2, 16, 3]


def frame_camera_transforms(frame_i, n_video, hand_ts_sorted,
                            headset_poses, headset_ts, T_device_camera):
    """T_world_device (SE3) and T_camera_world ([4,4] np.float64) for one frame."""
    ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]
    frac = frame_i / max(n_video - 1, 1)
    query_tc = int(ts_start + frac * (ts_end - ts_start))

    closest_ht = find_closest(headset_ts, query_tc)
    t_wd, q_wd = headset_poses[closest_ht]
    T_world_device = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0]
    T_camera_world = (
        T_device_camera.inverse().to_matrix()
        @ T_world_device.inverse().to_matrix()
    )
    return T_world_device, T_camera_world


def compute_2d_cam_data_pinhole(
    n_video, hand_ts_sorted, seq_gt_joints_world,
    T_device_camera, pinhole_calib, headset_poses, headset_ts,
    image_width=ARIA_FRAME_SIZE,
):
    """Mirror of HOT3DHandDataset._compute_2d_cam_data, but using pinhole_calib.

    Returns:
        gt_joints_2d   [N, 2, 16, 3]  — (u, v, confidence) in pixel coords
        cam_extrinsics [N, 4, 4]       — T_camera_world per frame
    """
    gt_2d_list, extr_list = [], []

    for frame_i in range(n_video):
        T_world_device, T_cam_world_np = frame_camera_transforms(
            frame_i, n_video, hand_ts_sorted,
            headset_poses, headset_ts, T_device_camera,
        )
        extr_list.append(torch.tensor(T_cam_world_np, dtype=torch.float32))

        frame_joints_3d = seq_gt_joints_world[frame_i]   # [2, 16, 3]
        frame_2d = torch.zeros(NUM_HANDS, 16, 3)
        for h_idx in range(NUM_HANDS):
            joints_w = frame_joints_3d[h_idx].numpy()
            if np.abs(joints_w).sum() < 1e-6:
                continue
            pixels, _, valid = project_vertices(
                joints_w, T_world_device, T_device_camera, pinhole_calib,
                image_width=image_width,
            )
            frame_2d[h_idx, :, 0] = torch.from_numpy(pixels[:, 0].astype(np.float32))
            frame_2d[h_idx, :, 1] = torch.from_numpy(pixels[:, 1].astype(np.float32))
            frame_2d[h_idx, :, 2] = torch.from_numpy(valid.astype(np.float32))
        gt_2d_list.append(frame_2d)

    return torch.stack(gt_2d_list), torch.stack(extr_list)


def compute_projected_bboxes_pinhole(
    n_video, hand_ts_sorted, gt_per_frame_world,
    T_device_camera, pinhole_calib, headset_poses, headset_ts,
    hand_poses, hand_ts_data, mano, rescale_factor,
    image_width=ARIA_FRAME_SIZE,
):
    """Mirror of HOT3DHandDataset._compute_projected_bboxes, but using pinhole_calib.

    Returns:
        bboxes_list  : list of [2, 4] float tensors (x1,y1,x2,y2 in [0,1])
        valid_list   : list of [2]    bool  tensors
    """
    ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]
    bboxes_list, valid_list = [], []

    for frame_i in range(n_video):
        frac = frame_i / max(n_video - 1, 1)
        query_tc = int(ts_start + frac * (ts_end - ts_start))

        closest_ht = find_closest(headset_ts, query_tc)
        t_wd, q_wd = headset_poses[closest_ht]
        T_world_device = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0]

        closest_hand_ts = find_closest(hand_ts_data, query_tc)
        hand_data = hand_poses[closest_hand_ts]

        frame_bboxes = torch.zeros(NUM_HANDS, 4)
        frame_valid = torch.zeros(NUM_HANDS, dtype=torch.bool)

        for h_idx in range(NUM_HANDS):
            hand_key = str(h_idx)
            is_right = h_idx == 1
            if hand_key not in hand_data or not hand_data[hand_key]:
                frame_bboxes[h_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                continue
            try:
                vertices, _ = mano.get_mesh(hand_data[hand_key], is_right)
                pixels, _, valid_mask = project_vertices(
                    vertices, T_world_device, T_device_camera, pinhole_calib,
                    image_width=image_width,
                )
                if valid_mask.sum() < 10:
                    frame_bboxes[h_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                    continue
                valid_px = pixels[valid_mask]
                u_min, v_min = valid_px.min(axis=0)
                u_max, v_max = valid_px.max(axis=0)
                cu = (u_min + u_max) / 2.0
                cv = (v_min + v_max) / 2.0
                bbox_w = (u_max - u_min) * rescale_factor
                bbox_h = (v_max - v_min) * rescale_factor
                bbox_size = max(bbox_w, bbox_h)
                x1 = (cu - bbox_size / 2.0) / image_width
                y1 = (cv - bbox_size / 2.0) / image_width
                x2 = (cu + bbox_size / 2.0) / image_width
                y2 = (cv + bbox_size / 2.0) / image_width
                x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in (x1, y1, x2, y2))
                if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                    frame_bboxes[h_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                    continue
                frame_bboxes[h_idx] = torch.tensor([x1, y1, x2, y2])
                frame_valid[h_idx] = True
            except Exception:
                frame_bboxes[h_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])

        bboxes_list.append(frame_bboxes)
        valid_list.append(frame_valid)

    return bboxes_list, valid_list


def transform_gt_per_frame_to_camera(
    gt_per_frame_world, n_video, hand_ts_sorted, T_device_camera,
    headset_poses, headset_ts, mano,
):
    """Lens-independent. Mirror of _transform_gt_to_crop_local.

    MANO's transl is an offset from the beta-specific canonical wrist
    (`joint_0_final = joint_0_canonical(betas) + transl`), so we cannot just
    rotate transl by R_cw. Apply the correct chain per frame per hand,
    in-place on a deep copy of gt_per_frame.
    """
    from scipy.spatial.transform import Rotation
    gt_per_frame = [vec.clone() for vec in gt_per_frame_world]

    mano._ensure_device(torch.device("cpu"))
    canon_cache = {}

    def canonical_joint_0(betas_np, is_right):
        key = (is_right, tuple(np.round(betas_np, 5)))
        if key in canon_cache:
            return canon_cache[key]
        layer = mano.right if is_right else mano.left
        out = layer(
            betas=torch.tensor([betas_np], dtype=torch.float32),
            global_orient=torch.zeros(1, 3),
            hand_pose=torch.zeros(1, 15),
            transl=torch.zeros(1, 3),
            return_verts=True,
        )
        j0 = out.joints[0, 0].detach().numpy()  # [3]
        canon_cache[key] = j0
        return j0

    for frame_i in range(n_video):
        _T_world_device, T_camera_world = frame_camera_transforms(
            frame_i, n_video, hand_ts_sorted,
            headset_poses, headset_ts, T_device_camera,
        )
        R_cw = T_camera_world[:3, :3]
        t_cw = T_camera_world[:3, 3]

        for h_idx in range(NUM_HANDS):
            offset = h_idx * HAND_PARAM_DIM
            p = gt_per_frame[frame_i][offset:offset + HAND_PARAM_DIM]
            if p.abs().sum() < 1e-6:
                continue
            transl_w = p[0:3].numpy().astype(np.float64)
            q_wxyz   = p[3:7].numpy().astype(np.float64)
            betas    = p[22:32].numpy().astype(np.float64)

            j0_canon = canonical_joint_0(betas, is_right=(h_idx == 1))
            j0_world = j0_canon + transl_w
            j0_cam   = R_cw @ j0_world + t_cw
            transl_cam = j0_cam - j0_canon

            # Rotation: R_cam = R_cw @ R_world (quaternion composition).
            R_world = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            R_cam   = R_cw @ R_world
            q_cam   = Rotation.from_matrix(R_cam).as_quat()  # xyzw
            q_cam_wxyz = np.array([q_cam[3], q_cam[0], q_cam[1], q_cam[2]])

            p[0:3] = torch.tensor(transl_cam, dtype=torch.float32)
            p[3:7] = torch.tensor(q_cam_wxyz, dtype=torch.float32)
            # pose (7:22) and betas (22:32) unchanged
    return gt_per_frame


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def link_safe(src_path, dst_path):
    """Symlink src → dst, replacing dst if it exists."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    dst_path.symlink_to(src_path.resolve())
    return True


def undistort_video(src_video, dst_video, fisheye_calib, pinhole_calib, fps):
    """Read every frame of src_video, undistort to pinhole, write to dst_video."""
    vr = VideoReader(str(src_video))
    n_frames = len(vr)
    # First frame to get dims.
    first = vr[0].asnumpy()  # RGB
    H, W = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (W, H))
    try:
        for i in range(n_frames):
            mp4_rgb = vr[i].asnumpy()
            und_rgb = undistort_mp4_frame(mp4_rgb, fisheye_calib, pinhole_calib)
            writer.write(cv2.cvtColor(und_rgb, cv2.COLOR_RGB2BGR))
            if (i + 1) % 500 == 0 or i + 1 == n_frames:
                print(f"      {i+1}/{n_frames} frames")
    finally:
        writer.release()


def process_sequence(src_seq, dst_seq, args, mano):
    seq_id = src_seq.name
    print(f"\n=== {seq_id} ===")

    calib_path   = src_seq / "mps_slam_calibration" / "online_calibration.jsonl"
    headset_path = src_seq / "ground_truth" / "headset_trajectory.csv"
    jsonl_path   = src_seq / "hand_data" / "mano_hand_pose_trajectory.jsonl"
    video_path   = src_seq / "video_main_rgb.mp4"
    if not all(p.exists() for p in [calib_path, headset_path, jsonl_path, video_path]):
        print("  SKIP — missing required source files")
        return

    meta_path = dst_seq / ".preprocessing_meta.json"
    if meta_path.exists() and not args.force:
        try:
            with open(meta_path) as f:
                existing = json.load(f)
            if (existing.get("focal") == float(args.focal)
                and existing.get("rescale_factor") == float(args.rescale_factor)
                and existing.get("resolution") == int(args.resolution)):
                print("  SKIP — already preprocessed with matching params (use --force to redo)")
                return
        except Exception:
            pass  # treat as not preprocessed

    dst_seq.mkdir(parents=True, exist_ok=True)
    (dst_seq / "hand_data").mkdir(exist_ok=True)
    (dst_seq / "ground_truth").mkdir(exist_ok=True)
    (dst_seq / "mps_slam_calibration").mkdir(exist_ok=True)

    print("  Loading calibration ...")
    T_device_camera, fisheye_calib = load_camera_calibration(str(calib_path))
    pinhole_calib = build_pinhole_target(fisheye_calib, args.focal)
    headset_poses = load_headset_trajectory(str(headset_path))
    headset_ts = sorted(headset_poses.keys())
    hand_poses_data = load_hand_poses(str(jsonl_path))
    hand_ts_data = sorted(hand_poses_data.keys())

    # 1. Undistort video.
    dst_video = dst_seq / "video_main_rgb.mp4"
    if dst_video.exists() and not args.force:
        print(f"  Video already exists, reusing: {dst_video.name}")
    else:
        print("  Undistorting video ...")
        undistort_video(video_path, dst_video, fisheye_calib, pinhole_calib, fps=args.video_fps)

    n_video = len(VideoReader(str(dst_video)))

    # 2. Symlink lens-independent files.
    print("  Linking lens-independent files ...")
    link_safe(src_seq / "ground_truth" / "headset_trajectory.csv",
              dst_seq / "ground_truth" / "headset_trajectory.csv")
    link_safe(src_seq / "mps_slam_calibration" / "online_calibration.jsonl",
              dst_seq / "mps_slam_calibration" / "online_calibration.jsonl")
    link_safe(src_seq / "hand_data" / "mano_hand_pose_trajectory.jsonl",
              dst_seq / "hand_data" / "mano_hand_pose_trajectory.jsonl")
    # mps_slam_trajectories isn't used by the dataset but symlink for completeness if present
    src_traj = src_seq / "mps_slam_trajectories"
    if src_traj.exists():
        link_safe(src_traj, dst_seq / "mps_slam_trajectories")

    # 3. Build / load world-frame gt_per_frame and world-frame 3D joints cache.
    gt_per_frame_world = build_gt_per_frame_world(hand_poses_data, hand_ts_data, n_video)

    world_cache_dst = dst_seq / "hand_data" / "gt_joints_cache_world.pt"
    world_cache_src = src_seq / "hand_data" / "gt_joints_cache_world.pt"
    if world_cache_src.exists():
        link_safe(world_cache_src, world_cache_dst)
        seq_gt_joints_world = torch.load(world_cache_src, weights_only=True)
    else:
        print("  Computing world-frame 3D joint cache (not present at source) ...")
        seq_gt_joints_world = compute_seq_joints_world(mano, gt_per_frame_world)
        torch.save(seq_gt_joints_world, world_cache_dst)

    # 4. cam_extrinsics_cache.pt and cam_intrinsics.pt (pinhole intrinsics).
    extr_cache_dst = dst_seq / "hand_data" / "cam_extrinsics_cache.pt"
    extr_cache_src = src_seq / "hand_data" / "cam_extrinsics_cache.pt"
    if extr_cache_src.exists():
        link_safe(extr_cache_src, extr_cache_dst)
        cam_extrinsics = torch.load(extr_cache_src, weights_only=True)
    else:
        cam_extrinsics = None  # will be filled below via 2D-joint pass

    # Pinhole intrinsics — always written fresh.
    fp = pinhole_calib.get_focal_lengths()
    pp = pinhole_calib.get_principal_point()
    cam_intrinsics_pinhole = torch.tensor(
        [float(fp[0]), float(pp[0]), float(pp[1])], dtype=torch.float32,
    )
    torch.save(cam_intrinsics_pinhole, dst_seq / "hand_data" / "cam_intrinsics.pt")

    # 5. Recompute 2D-joint cache with pinhole projection.
    print("  Recomputing 2D joints cache (pinhole projection) ...")
    gt_2d_pinhole, extr_computed = compute_2d_cam_data_pinhole(
        n_video, hand_ts_data, seq_gt_joints_world,
        T_device_camera, pinhole_calib, headset_poses, headset_ts,
    )
    torch.save(gt_2d_pinhole, dst_seq / "hand_data" / "gt_joints_2d_cache.pt")
    if cam_extrinsics is None:
        torch.save(extr_computed, extr_cache_dst)

    # 6. gt_joints_cache_cam_v2.pt — lens-independent (3D joints in metric
    #    sensor camera frame). Symlink if present, otherwise compute.
    cam_v2_dst = dst_seq / "hand_data" / "gt_joints_cache_cam_v2.pt"
    cam_v2_src = src_seq / "hand_data" / "gt_joints_cache_cam_v2.pt"
    if cam_v2_src.exists():
        link_safe(cam_v2_src, cam_v2_dst)
    else:
        print("  Computing 3D camera-frame joints cache ...")
        # gt_per_frame in camera frame, then run MANO again to get camera-frame joints.
        gt_per_frame_cam = transform_gt_per_frame_to_camera(
            gt_per_frame_world, n_video, hand_ts_data,
            T_device_camera, headset_poses, headset_ts, mano,
        )
        seq_gt_joints_cam = compute_seq_joints_world(mano, gt_per_frame_cam)
        torch.save(seq_gt_joints_cam, cam_v2_dst)

    # 7. hand_bboxes cache (pinhole-projected bbox, lens-independent gt+R_cw transform).
    bbox_name = f"hand_bboxes_v2_rf{args.rescale_factor}_res{args.resolution}x{args.resolution}.pt"
    bbox_dst = dst_seq / "hand_data" / bbox_name
    print(f"  Computing bbox cache (pinhole) → {bbox_name}")
    bbox_frames, valid_frames = compute_projected_bboxes_pinhole(
        n_video, hand_ts_data, gt_per_frame_world,
        T_device_camera, pinhole_calib, headset_poses, headset_ts,
        hand_poses_data, hand_ts_data, mano, args.rescale_factor,
    )
    gt_per_frame_cam_for_bbox = transform_gt_per_frame_to_camera(
        gt_per_frame_world, n_video, hand_ts_data,
        T_device_camera, headset_poses, headset_ts, mano,
    )
    torch.save({
        "bboxes": torch.stack(bbox_frames),
        "valid":  torch.stack(valid_frames),
        "gt":     torch.stack(gt_per_frame_cam_for_bbox),
    }, bbox_dst)

    # 8. Marker file.
    meta = {
        "focal": float(args.focal),
        "rescale_factor": float(args.rescale_factor),
        "resolution": int(args.resolution),
        "source_root": str(Path(args.source_root).resolve()),
        "source_seq_path": str(src_seq.resolve()),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "pinhole_image_size": [int(ARIA_FRAME_SIZE), int(ARIA_FRAME_SIZE)],
        "pinhole_principal_point": [float(pp[0]), float(pp[1])],
        "pinhole_focal_length": [float(fp[0]), float(fp[1])],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Done. Wrote marker → {meta_path}")


def main():
    args = parse_args()
    src = Path(args.source_root)
    dst = Path(args.dest_root)
    if not src.exists():
        raise SystemExit(f"--source_root not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    if args.sequences == "all":
        seq_ids = sorted(d.name for d in src.iterdir() if d.is_dir())
    else:
        seq_ids = [s.strip() for s in args.sequences.split(",") if s.strip()]

    print(f"Processing {len(seq_ids)} sequence(s) from {src}")
    print(f"  → {dst}")
    print(f"  focal={args.focal}  rescale_factor={args.rescale_factor}  res={args.resolution}")

    mano = MANOModel(args.mano_model_folder)

    for seq_id in seq_ids:
        try:
            process_sequence(src / seq_id, dst / seq_id, args, mano)
        except Exception as e:
            print(f"  FAILED on {seq_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
