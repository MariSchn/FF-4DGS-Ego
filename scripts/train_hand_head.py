"""
Train MANO hand parameter prediction head on Hot3D Aria data.

Example:
    python -m scripts.train_hand_head --config configs/train_hand_head.yaml
"""

import argparse
import bisect
import json
import os
import random
import sys
import time

import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
import wandb
import yaml
from decord import VideoReader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF
from tqdm import tqdm

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.utils.auxiliary import load_video

from scripts.hamer_losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from scripts.hand_depth_anchor_loss import hand_depth_anchor_loss
from scripts.hand_metrics import metric_chunks_from_batch, metrics_from_chunks
from scripts.gs_metrics import (
    LPIPSScorer,
    render_views_from_predictions,
    metric_chunks_from_batch as gs_metric_chunks_from_batch,
    metrics_from_chunks as gs_metrics_from_chunks,
    region_metric_chunks_from_batch as gs_region_chunks_from_batch,
)


def _hand_region_mask(hand_bboxes, hand_valid, H, W):
    """Per-frame boolean mask [B, S, H, W] filled inside each valid hand box.

    hand_bboxes: [B, S, 2, 4] normalised xyxy in [0, 1]. hand_valid: [B, S, 2]
    (or None). Used to restrict GS metrics to the hand region.
    """
    B, S = hand_bboxes.shape[0], hand_bboxes.shape[1]
    mask = torch.zeros(B, S, H, W, dtype=torch.bool)
    bb = hand_bboxes.clamp(0.0, 1.0).cpu()
    hv = hand_valid.cpu() if hand_valid is not None else None
    for b in range(B):
        for s in range(S):
            for h in range(bb.shape[2]):
                if hv is not None and float(hv[b, s, h]) < 0.5:
                    continue
                x1, y1, x2, y2 = bb[b, s, h].tolist()
                xi1, xi2 = int(x1 * W), int(round(x2 * W))
                yi1, yi2 = int(y1 * H), int(round(y2 * H))
                if xi2 > xi1 and yi2 > yi1:
                    mask[b, s, yi1:yi2, xi1:xi2] = True
    return mask


HAND_PARAM_DIM = 32  # per hand: pos(3) + rot(4) + pose(15) + betas(10)
NUM_HANDS = 2


# ------------------------------------------------------------------
# Bounding-box utilities
# ------------------------------------------------------------------



def default_full_image_bboxes(num_frames: int) -> tuple:
    """Fallback: return full-image bboxes (no effective cropping).

    Useful as a baseline or when real bboxes are not yet available.
    """
    bboxes = torch.zeros(num_frames, NUM_HANDS, 4)
    bboxes[:, :, 2] = 1.0  # x2
    bboxes[:, :, 3] = 1.0  # y2
    valid = torch.ones(num_frames, NUM_HANDS, dtype=torch.bool)
    return bboxes, valid


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class HOT3DHandDataset(Dataset):
    """Sliding-window clips over a list of sequences."""

    def __init__(self, seq_dirs, mano_model, num_frames=16, res=(224, 224), clip_stride=None,
                 use_hand_crop=False, rescale_factor=2.0):
        self.num_frames = num_frames
        self.mano_model = mano_model
        self.res = res
        self.use_hand_crop = use_hand_crop
        self.rescale_factor = rescale_factor
        self.clips = []
        from collections import OrderedDict
        self.video_readers = OrderedDict()
        
        if clip_stride is None:
            clip_stride = num_frames

        for seq_path in tqdm(seq_dirs):
            video_path = os.path.join(seq_path, "video_main_rgb.mp4")
            hand_data_root = os.path.join(seq_path, "hand_data")
            jsonl_path = os.path.join(hand_data_root, "mano_hand_pose_trajectory.jsonl")
            # The joint cache depends on whether we've rewritten the params into
            # camera frame. Bump the filename so stale world-frame caches don't
            # get silently reused after switching crop mode.
            # v2 suffix = MANO transl semantics fixed (joint_0_canonical-aware).
            joint_cache_name = (
                "gt_joints_cache_cam_v2.pt" if self.use_hand_crop
                else "gt_joints_cache_world.pt"
            )
            joint_cache_path = os.path.join(hand_data_root, joint_cache_name)

            if not os.path.exists(video_path) or not os.path.exists(jsonl_path):
                print(f"Skipping {seq_path} because it doesn't have a video or jsonl file")
                continue
            
            # if seq_path[0:5] in ["P0004", "P0005", "P0006", "P0008", "P0016", "P0020"]:
            #     print(f"Skipping {seq_path} as it doesn't have MANO data")
            #     continue

            # Load all JSONL entries keyed by timestamp
            hand_entries = {}  # timestamp_ns -> hand_poses dict
            with open(jsonl_path) as f:
                for line in f:
                    entry = json.loads(line)
                    hand_entries[entry["timestamp_ns"]] = entry["hand_poses"]
            hand_ts_sorted = sorted(hand_entries.keys())

            if len(hand_ts_sorted) < 2:
                continue

            n_video = len(VideoReader(video_path))
            if n_video < num_frames:
                continue

            # Map each video frame to the closest JSONL entry via linear interpolation
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

            def _hand_to_vec(hand_poses):
                vecs = []
                for hand_id in ["0", "1"]:
                    hand = hand_poses.get(hand_id, {})
                    if hand:
                        vecs.append(torch.cat([
                            torch.tensor(hand["wrist_xform"]["t_xyz"],  dtype=torch.float32),
                            torch.tensor(hand["wrist_xform"]["q_wxyz"], dtype=torch.float32),
                            torch.tensor(hand["pose"],                  dtype=torch.float32),
                            torch.tensor(hand["betas"],                 dtype=torch.float32),
                        ]))
                    else:
                        vecs.append(torch.zeros(HAND_PARAM_DIM))
                return torch.cat(vecs)

            gt_per_frame = []
            for frame_i in range(n_video):
                ts = _closest_ts(frame_i)
                gt_per_frame.append(_hand_to_vec(hand_entries[ts]))

            # Handle 2D GT joints + camera data. If we have to compute this
            # from scratch, we need world-frame joints (project_vertices applies
            # its own world→camera transform). If the cache exists we skip that
            # entirely, which matters because after use_hand_crop below we
            # rewrite gt_per_frame into camera frame.
            cam_2d_cache_path   = os.path.join(hand_data_root, "gt_joints_2d_cache.pt")
            cam_extr_cache_path = os.path.join(hand_data_root, "cam_extrinsics_cache.pt")
            cam_intr_cache_path = os.path.join(hand_data_root, "cam_intrinsics.pt")

            if (os.path.exists(cam_2d_cache_path)
                    and os.path.exists(cam_extr_cache_path)
                    and os.path.exists(cam_intr_cache_path)):
                seq_gt_joints_2d   = torch.load(cam_2d_cache_path,   weights_only=True)
                seq_cam_extrinsics = torch.load(cam_extr_cache_path,  weights_only=True)
                seq_cam_intrinsics = torch.load(cam_intr_cache_path,  weights_only=True)
                print(f"Loaded 2D GT joints + cam data for {seq_path}.")
            else:
                # gt_per_frame is still in world frame here (crop transform runs below),
                # so joints computed from it are in world frame as project_vertices expects.
                seq_gt_joints_world = self._compute_seq_joints_from_params(gt_per_frame)
                seq_gt_joints_2d, seq_cam_extrinsics, seq_cam_intrinsics = (
                    HOT3DHandDataset._compute_2d_cam_data(
                        seq_path, n_video, hand_ts_sorted, seq_gt_joints_world
                    )
                )
                if seq_gt_joints_2d is not None:
                    torch.save(seq_gt_joints_2d,   cam_2d_cache_path)
                    torch.save(seq_cam_extrinsics, cam_extr_cache_path)
                    torch.save(seq_cam_intrinsics, cam_intr_cache_path)
                    print(f"Computed and saved 2D GT joints + cam data for {seq_path}.")
                else:
                    print(f"No calibration for {seq_path} — 2D loss unavailable for this sequence.")

            # Handle Bounding Boxes (rewrites gt_per_frame into camera frame on a hit).
            if self.use_hand_crop:
                cache_name = f"hand_bboxes_v2_rf{self.rescale_factor}_res{res[0]}x{res[1]}.pt"
                bbox_cache_path = os.path.join(seq_path, "hand_data", cache_name)

                if os.path.exists(bbox_cache_path):
                    cached = torch.load(bbox_cache_path, weights_only=True)
                    bbox_frames = list(cached["bboxes"])
                    valid_frames = list(cached["valid"])
                    gt_per_frame[:] = list(cached["gt"])
                else:
                    bbox_frames, valid_frames = HOT3DHandDataset._compute_projected_bboxes(
                        seq_path, n_video, hand_ts_sorted, gt_per_frame,
                        rescale_factor=self.rescale_factor,
                    )
                    if bbox_frames is None:
                        print(f"Skipping {seq_path}: missing calibration for hand crop")
                        continue

                    # world → camera frame so predicted and ground-truth params
                    # (and the joints we derive from them) live in the same frame.
                    ok = self._transform_gt_to_crop_local(
                        seq_path, n_video, hand_ts_sorted, gt_per_frame,
                        bbox_frames, valid_frames, res=res,
                    )
                    if not ok:
                        print(f"Skipping {seq_path}: missing calibration for GT crop-local transform")
                        continue

                    torch.save({
                        "bboxes": torch.stack(bbox_frames),
                        "valid": torch.stack(valid_frames),
                        "gt": torch.stack(gt_per_frame),
                    }, bbox_cache_path)
                    print(f"Cached hand bboxes -> {bbox_cache_path}")
            else:
                bbox_frames = valid_frames = None

            # Build GT joints from the (possibly camera-frame) params.
            if os.path.exists(joint_cache_path):
                seq_gt_joints = torch.load(joint_cache_path, weights_only=True)
                print(f"Loaded GT joints for {seq_path}.")
            else:
                seq_gt_joints = self._compute_seq_joints_from_params(gt_per_frame)
                torch.save(seq_gt_joints, joint_cache_path)
                print(f"Computed and saved GT joints for {seq_path}.")

            # Create Sliding Window Clips
            for start in range(0, n_video - num_frames + 1, clip_stride):
                end = start + num_frames
                clip = {
                    "video_path":   video_path,
                    "gt_frames":    gt_per_frame[start : end],
                    "gt_joints":    seq_gt_joints[start : end].clone(),
                    "frame_offset": start,
                    "seq_path":     seq_path,
                }
                if self.use_hand_crop:
                    clip["hand_bboxes"] = [b.clone() for b in bbox_frames[start : start + num_frames]]
                    clip["hand_valid"]  = [v.clone() for v in valid_frames[start : start + num_frames]]
                if seq_gt_joints_2d is not None:
                    clip["gt_joints_2d"]   = seq_gt_joints_2d[start : end].clone()    # [S, 2, 16, 3]
                    clip["cam_extrinsics"] = seq_cam_extrinsics[start : end].clone()  # [S, 4, 4]
                    clip["cam_intrinsics"] = seq_cam_intrinsics                # [3]
                self.clips.append(clip)
        self.mano_model = None

    def _compute_seq_joints_from_params(self, gt_per_frame):
        """Run MANO once per (frame, hand) to produce [N, 2, 16, 3] joints.

        Used for both the world-frame joints fed to the 2D cache generator and
        the camera-frame joints that become the 3D-loss ground truth.
        """
        seq_list = []
        for frame_p in gt_per_frame:
            frame_joints = []
            for h_idx in range(NUM_HANDS):
                offset = h_idx * HAND_PARAM_DIM
                p = frame_p[offset : offset + HAND_PARAM_DIM]
                if p.abs().sum() < 1e-6:
                    j3d = np.zeros((16, 3), dtype=np.float32)
                else:
                    j3d = self.mano_model.get_joints_from_tensor(
                        p, is_right=(h_idx == 1), return_tensor=False,
                    )
                    if isinstance(j3d, np.ndarray) and j3d.ndim == 3:
                        j3d = j3d.squeeze(0)
                    elif torch.is_tensor(j3d) and j3d.dim() == 3:
                        j3d = j3d.squeeze(0)
                frame_joints.append(torch.as_tensor(j3d, dtype=torch.float32))
            seq_list.append(torch.stack(frame_joints))
        return torch.stack(seq_list)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        video_path = clip["video_path"]
        video_reader = VideoReader(video_path)
        
        pil_images = load_video(
            video_reader,
            num_frames=self.num_frames,
            resolution=self.res,
            sampling="first",
            frame_offset=clip["frame_offset"],
        )
        
        imgs = torch.stack([TVF.to_tensor(img) for img in pil_images])
        gt_params = torch.stack(clip["gt_frames"])
        gt_joints = clip["gt_joints"] 

        # # Pre-compute GT joints for the whole clip
        # all_gt_joints = []
        # for frame_p in gt_params:
        #     frame_joints = []
        #     for h_idx in range(NUM_HANDS):
        #         offset = h_idx * HAND_PARAM_DIM
        #         p = frame_p[offset : offset + HAND_PARAM_DIM]
                
        #         # Use CPU/Numpy version for the Dataset
        #         if p.abs().sum() < 1e-6:
        #             j3d = np.zeros((21, 3), dtype=np.float32)
        #         else:
        #             j3d = self.mano_model.get_joints_from_tensor(p, is_right=(h_idx==1), return_tensor=False)
        #         frame_joints.append(torch.from_numpy(j3d))
        #     all_gt_joints.append(torch.stack(frame_joints))
            

        # gt_joints_3d = torch.stack(all_gt_joints) 
        # gt_joints_3d = torch.squeeze(gt_joints_3d, 2)
        # # [16, 2, 21, 3], [num_frames, num_hands, num_joints, dims]
        
        # print(f"gt_joints_3d.shape: {gt_joints_3d.shape}")

        out = {
            "img": imgs, 
            "gt": gt_params, 
            "gt_joints": gt_joints
        }

        if self.use_hand_crop:
            out["hand_bboxes"] = torch.stack(clip["hand_bboxes"])  # [S, 2, 4]
            out["hand_valid"]  = torch.stack(clip["hand_valid"])   # [S, 2]

        if "gt_joints_2d" in clip:
            out["gt_joints_2d"]   = clip["gt_joints_2d"]    # [S, 2, 16, 3]
            out["cam_extrinsics"] = clip["cam_extrinsics"]  # [S, 4, 4]
            out["cam_intrinsics"] = clip["cam_intrinsics"]  # [3]

        return out

    @staticmethod
    def _compute_projected_bboxes(seq_path, n_video, hand_ts_sorted, gt_per_frame,
                                   rescale_factor=2.0, **_kwargs):
        """Compute per-frame hand bboxes by projecting the full MANO mesh.

        Follows the same approach as HaMeR's hand detection pipeline:
        1. Project all hand mesh vertices to 2D (like HaMeR uses ViTPose keypoints)
        2. Compute a tight bounding box around all valid projected vertices
        3. Apply a rescale factor to pad the box (HaMeR default: 2.0x)

        This produces bboxes that tightly enclose the visible hand and are
        centered on the hand (not just the wrist), matching HaMeR's ViTPose
        keypoint-based bbox extraction.

        Returns lists of [2, 4] bbox tensors (normalised x1,y1,x2,y2 in [0,1])
        and [2] bool valid tensors, or (None, None) if calibration files are missing.
        """
        import numpy as np
        from projectaria_tools.core.sophus import SE3
        from scripts.hand_vis_utils import (
            load_camera_calibration, load_headset_trajectory, find_closest,
            load_hand_poses, MANOModel, project_vertices,
        )

        calib_path   = os.path.join(seq_path, "mps_slam_calibration", "online_calibration.jsonl")
        headset_path = os.path.join(seq_path, "ground_truth", "headset_trajectory.csv")
        jsonl_path   = os.path.join(seq_path, "hand_data", "mano_hand_pose_trajectory.jsonl")
        mano_folder  = os.path.join(os.path.dirname(os.path.dirname(seq_path)),
                                     "models", "MANO")

        # Also accept mano_folder from the repo root (common layout)
        if not os.path.exists(mano_folder):
            # Try relative to the FF-4DGS-Ego repo root
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mano_folder = os.path.join(repo_root, "models", "MANO")

        for p in [calib_path, headset_path, jsonl_path]:
            if not os.path.exists(p):
                return None, None
        if not os.path.exists(mano_folder):
            print(f"[WARN] MANO model folder not found at {mano_folder}, "
                  "cannot compute mesh-based bboxes")
            return None, None

        T_device_camera, cam_calib = load_camera_calibration(calib_path)
        headset_poses = load_headset_trajectory(headset_path)
        headset_ts    = sorted(headset_poses.keys())

        hand_poses_data = load_hand_poses(jsonl_path)
        hand_ts_data    = sorted(hand_poses_data.keys())

        mano_model = MANOModel(mano_folder)

        ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]
        IMAGE_WIDTH = 1408  # Aria sensor resolution before resize

        bboxes_list = []
        valid_list  = []

        for frame_i in range(len(gt_per_frame)):
            frac     = frame_i / max(n_video - 1, 1)
            query_tc = int(ts_start + frac * (ts_end - ts_start))

            # Find closest headset pose
            closest_ht = find_closest(headset_ts, query_tc)
            t_wd, q_wd = headset_poses[closest_ht]
            T_world_device = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0]

            # Find closest hand pose entry (raw JSONL data with full MANO params)
            closest_hand_ts = find_closest(hand_ts_data, query_tc)
            hand_data = hand_poses_data[closest_hand_ts]

            frame_bboxes = torch.zeros(NUM_HANDS, 4)
            frame_valid  = torch.zeros(NUM_HANDS, dtype=torch.bool)

            for hand_idx in range(NUM_HANDS):
                hand_key = str(hand_idx)  # "0" = left, "1" = right
                is_right = hand_idx == 1

                if hand_key not in hand_data or not hand_data[hand_key]:
                    # Hand absent: safe fallback bbox
                    frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                    continue

                try:
                    # Generate full MANO mesh in world coordinates
                    vertices, _faces = mano_model.get_mesh(hand_data[hand_key], is_right)

                    # Project all 778 vertices to 2D using the same projection
                    # as hand_vis_utils (fisheye + 90° rotation)
                    pixels, depths, valid_mask = project_vertices(
                        vertices, T_world_device, T_device_camera, cam_calib,
                        image_width=IMAGE_WIDTH,
                    )

                    if valid_mask.sum() < 10:
                        frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                        continue

                    # Tight bbox around all valid projected vertices (in pixels)
                    valid_pixels = pixels[valid_mask]  # [N_valid, 2] — (u, v) in pixel coords
                    u_min, v_min = valid_pixels.min(axis=0)
                    u_max, v_max = valid_pixels.max(axis=0)

                    # Compute center and size (Hamer-style: center + scale)
                    center_u = (u_min + u_max) / 2.0
                    center_v = (v_min + v_max) / 2.0
                    bbox_w = u_max - u_min
                    bbox_h = v_max - v_min

                    # Apply rescale factor (Hamer default = 2.0) to pad the bbox
                    # This matches HaMeR's ViTDetDataset: scale = rescale_factor * (box_size) / 200
                    # but we directly expand the bbox by the factor
                    bbox_w *= rescale_factor
                    bbox_h *= rescale_factor

                    # Make square (take max side, like Hamer's expand_to_aspect_ratio)
                    bbox_size = max(bbox_w, bbox_h)

                    # Convert to normalised [0,1] coords (x1, y1, x2, y2)
                    x1 = (center_u - bbox_size / 2.0) / IMAGE_WIDTH
                    y1 = (center_v - bbox_size / 2.0) / IMAGE_WIDTH
                    x2 = (center_u + bbox_size / 2.0) / IMAGE_WIDTH
                    y2 = (center_v + bbox_size / 2.0) / IMAGE_WIDTH

                    # Clamp to [0, 1]
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))

                    # Sanity check: box must have positive area
                    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                        frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                        continue

                    frame_bboxes[hand_idx] = torch.tensor([x1, y1, x2, y2])
                    frame_valid[hand_idx] = True

                except Exception as e:
                    # Fallback on MANO mesh generation failure
                    frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                    continue

            bboxes_list.append(frame_bboxes)
            valid_list.append(frame_valid)

        return bboxes_list, valid_list

    @staticmethod
    def _load_camera_seq_data(seq_path):
        """Load per-sequence camera calibration + headset trajectory.

        Returns (T_device_camera, cam_calib, headset_poses, headset_ts, calib_path)
        or None if the required calibration files are missing.
        """
        from scripts.hand_vis_utils import load_camera_calibration, load_headset_trajectory

        calib_path = os.path.join(seq_path, "mps_slam_calibration", "online_calibration.jsonl")
        headset_path = os.path.join(seq_path, "ground_truth", "headset_trajectory.csv")
        if not os.path.exists(calib_path) or not os.path.exists(headset_path):
            return None

        T_device_camera, cam_calib = load_camera_calibration(calib_path)
        headset_poses = load_headset_trajectory(headset_path)
        headset_ts = sorted(headset_poses.keys())
        return T_device_camera, cam_calib, headset_poses, headset_ts, calib_path

    @staticmethod
    def _frame_camera_transforms(frame_i, n_video, hand_ts_sorted,
                                 headset_poses, headset_ts, T_device_camera):
        """Build T_world_device and T_camera_world (numpy [4, 4]) for one frame."""
        from projectaria_tools.core.sophus import SE3
        from scripts.hand_vis_utils import find_closest

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

    @staticmethod
    def _compute_2d_cam_data(seq_path, n_video, hand_ts_sorted, seq_gt_joints_3d):
        """Compute GT 2D keypoints and per-frame camera extrinsics for the 2D loss.

        Projects all GT 3D joints to pixel coordinates using project_vertices and
        records the world-to-camera extrinsic matrix for each frame so that predicted
        joints can be projected differentiably at training time.

        Returns:
            gt_joints_2d   (torch.Tensor | None): [N, 2, 16, 3]  — (u, v, confidence)
            cam_extrinsics (torch.Tensor | None): [N, 4, 4]       — T_camera_world per frame
            cam_intrinsics (torch.Tensor | None): [3]             — [f, cx, cy]
            or (None, None, None) if calibration files are missing.
        """
        import numpy as np
        from scripts.hand_vis_utils import project_vertices

        seq_data = HOT3DHandDataset._load_camera_seq_data(seq_path)
        if seq_data is None:
            return None, None, None
        T_device_camera, cam_calib, headset_poses, headset_ts, calib_path = seq_data

        # FISHEYE624 params layout: [f, cx, cy, k1..k6, p1, p2, s0..s3]
        with open(calib_path) as fh:
            entry = json.loads(fh.readline())
            for cam in entry["CameraCalibrations"]:
                if cam["Label"] == "camera-rgb":
                    raw_params = np.array(cam["Projection"]["Params"], dtype=np.float64)
                    break
        cam_intrinsics = torch.tensor(
            [raw_params[0], raw_params[1], raw_params[2]], dtype=torch.float32
        )  # [f, cx, cy]

        IMAGE_WIDTH = 1408

        gt_joints_2d_list   = []
        cam_extrinsics_list = []

        for frame_i in range(n_video):
            T_world_device, T_cam_world_np = HOT3DHandDataset._frame_camera_transforms(
                frame_i, n_video, hand_ts_sorted, headset_poses, headset_ts, T_device_camera,
            )
            cam_extrinsics_list.append(
                torch.tensor(T_cam_world_np, dtype=torch.float32)
            )

            # Project all joints for this frame
            frame_joints_3d = seq_gt_joints_3d[frame_i]   # [2, 16, 3]
            frame_joints_2d = torch.zeros(NUM_HANDS, 16, 3)  # [u, v, confidence]

            for h_idx in range(NUM_HANDS):
                joints_w = frame_joints_3d[h_idx].numpy()  # [16, 3]
                if np.abs(joints_w).sum() < 1e-6:
                    continue  # hand absent — confidence stays 0

                pixels, _, valid = project_vertices(
                    joints_w, T_world_device, T_device_camera, cam_calib, IMAGE_WIDTH
                )
                frame_joints_2d[h_idx, :, 0] = torch.from_numpy(pixels[:, 0].astype(np.float32))
                frame_joints_2d[h_idx, :, 1] = torch.from_numpy(pixels[:, 1].astype(np.float32))
                frame_joints_2d[h_idx, :, 2] = torch.from_numpy(valid.astype(np.float32))

            gt_joints_2d_list.append(frame_joints_2d)

        gt_joints_2d   = torch.stack(gt_joints_2d_list)    # [N, 2, 16, 3]
        cam_extrinsics = torch.stack(cam_extrinsics_list)  # [N, 4, 4]
        return gt_joints_2d, cam_extrinsics, cam_intrinsics

    def _transform_gt_to_crop_local(self, seq_path, n_video, hand_ts_sorted, gt_per_frame,
                                     bbox_frames, valid_frames, res=(224, 224)):
        """Transform GT wrist position and orientation from world to camera frame.

        MANO's convention is `joint_0_final = joint_0_canonical(betas) + transl`,
        i.e. `transl` is an offset from the beta-specific canonical wrist location,
        NOT an absolute world position. So naively doing `R_cw @ transl + t_cw`
        places joint 0 at `joint_0_canonical + R_cw @ transl + t_cw` in camera
        frame, which differs from the true camera-frame wrist position by
        `(I - R_cw) @ joint_0_canonical`. This manifested as a visible
        rotation+mirror offset in the GT-vs-pred overlay.

        Correct transform (per frame per hand):
            1. joint_0_world = joint_0_canonical(betas) + transl_world
            2. joint_0_cam   = R_cw @ joint_0_world + t_cw
            3. transl_cam    = joint_0_cam - joint_0_canonical(betas)
            4. R_cam         = R_cw @ R_world   (unchanged)

        Modifies gt_per_frame in-place.
        Returns True on success, False if calibration files are missing.
        """
        from scipy.spatial.transform import Rotation

        seq_data = HOT3DHandDataset._load_camera_seq_data(seq_path)
        if seq_data is None:
            return False
        T_device_camera, _cam_calib, headset_poses, headset_ts, _calib_path = seq_data

        # Cache joint_0_canonical per (is_right, betas) since betas are usually
        # constant across a sequence and MANO is fairly slow.
        self.mano_model._ensure_device(torch.device("cpu"))
        canon_cache = {}

        def canonical_joint_0(betas_np, is_right):
            key = (is_right, tuple(np.round(betas_np, 5)))
            if key in canon_cache:
                return canon_cache[key]
            layer = self.mano_model.right if is_right else self.mano_model.left
            out = layer(
                betas=torch.tensor([betas_np], dtype=torch.float32),
                global_orient=torch.zeros(1, 3),
                hand_pose=torch.zeros(1, 15),
                transl=torch.zeros(1, 3),
                return_verts=True,
            )
            j0 = out.joints[0, 0].detach().numpy().astype(np.float64)
            canon_cache[key] = j0
            return j0

        for frame_i in range(len(gt_per_frame)):
            _T_world_device, T_camera_world = HOT3DHandDataset._frame_camera_transforms(
                frame_i, n_video, hand_ts_sorted, headset_poses, headset_ts, T_device_camera,
            )
            R_cw = T_camera_world[:3, :3]
            t_cw = T_camera_world[:3, 3]

            gt_vec = gt_per_frame[frame_i]  # [64] = 2 hands x 32

            for hand_idx in range(NUM_HANDS):
                off = hand_idx * HAND_PARAM_DIM
                t_world = gt_vec[off:off + 3].numpy().astype(np.float64)
                q_wxyz  = gt_vec[off + 3:off + 7].numpy().astype(np.float64)
                betas   = gt_vec[off + 22:off + 32].numpy().astype(np.float64)

                if np.abs(t_world).sum() < 1e-8 and np.abs(q_wxyz).sum() < 1e-8:
                    continue

                j0_canon = canonical_joint_0(betas, is_right=(hand_idx == 1))

                # Joint 0 in world → camera → back to transl-offset form
                j0_world = j0_canon + t_world
                j0_cam   = R_cw @ j0_world + t_cw
                t_cam    = j0_cam - j0_canon

                gt_vec[off]     = float(t_cam[0])
                gt_vec[off + 1] = float(t_cam[1])
                gt_vec[off + 2] = float(t_cam[2])

                # Rotation transforms as before
                q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                R_world = Rotation.from_quat(q_xyzw).as_matrix()
                R_cam = R_cw @ R_world
                q_cam_xyzw = Rotation.from_matrix(R_cam).as_quat()
                q_cam_wxyz = np.array([
                    q_cam_xyzw[3], q_cam_xyzw[0], q_cam_xyzw[1], q_cam_xyzw[2]
                ])

                gt_vec[off + 3:off + 7] = torch.from_numpy(q_cam_wxyz.astype(np.float32))

        return True


def discover_sequences(data_root):
    seqs = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path):
            continue
        if (os.path.exists(os.path.join(path, "video_main_rgb.mp4")) and
                os.path.exists(os.path.join(path, "hand_data/mano_hand_pose_trajectory.jsonl"))):
            seqs.append(path)
    return seqs


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------

def build_views(imgs, num_frames, device, hand_bboxes=None, hand_valid=None,
                 crop_local_output=False):
    B, _, _, H, W = imgs.shape
    views = {
        "img":          imgs,
        "is_target":    torch.zeros((B, num_frames), dtype=torch.bool, device=device),
        "timestamp":    torch.arange(num_frames, device=device).unsqueeze(0).expand(B, -1),
        "is_static":    torch.zeros((B, num_frames), dtype=torch.bool, device=device),
        "valid_mask":   torch.ones((B, num_frames, H, W), dtype=torch.bool, device=device),
        "camera_poses": torch.eye(4, device=device).view(1, 1, 4, 4).expand(B, num_frames, 4, 4),
        "camera_intrs": torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, num_frames, 3, 3),
        "depthmap":     torch.ones((B, num_frames, H, W), device=device),
    }
    if hand_bboxes is not None:
        views["hand_bboxes"] = hand_bboxes
    if hand_valid is not None:
        views["hand_valid"] = hand_valid
    if crop_local_output:
        views["crop_local_output"] = True
    return views


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------

def setup_vis_items(dataset, num_vis_frames, seq_cache, mano_model, preload=False):
    """Set up visualization entries for a dataset.

    Args:
        dataset: HOT3DHandDataset
        preload: if True, also load img/gt tensors (for train vis)

    Returns:
        List of dicts with 'clip_idx', 'ctx', and optionally 'img'/'gt'.
    """
    from scripts.hand_vis_utils import setup_vis_context

    n = len(dataset.clips)
    step = max(1, n // num_vis_frames)
    items = []
    for clip_idx in torch.arange(0, n, step).tolist()[:num_vis_frames]:
        clip = dataset.clips[clip_idx]
        seq_path = clip["seq_path"]
        if seq_path not in seq_cache:
            try:
                seq_cache[seq_path] = setup_vis_context(seq_path, mano_model=mano_model)
            except Exception as e:
                print(f"[VIS] WARNING: Skipping visualization context for {seq_path} due to missing dependency: {e}")
                seq_cache[seq_path] = None
        ctx = seq_cache[seq_path]
        if ctx is None:
            continue

        entry = {
            "clip_idx": clip_idx,
            "ctx": {**ctx, "frame_offset": clip["frame_offset"]},
        }
        if preload:
            data = dataset[clip_idx]
            entry["img"] = data["img"]
            entry["gt"] = data["gt"]
            if "hand_bboxes" in data:
                entry["hand_bboxes"] = data["hand_bboxes"]
                entry["hand_valid"] = data["hand_valid"]
        items.append(entry)
    return items


def render_vis_list(vis_items, gt_pred_pairs, render_fn):
    """Render visualization images from gt/pred pairs.

    Args:
        vis_items: list of dicts with 'ctx' (containing frame_offset)
        gt_pred_pairs: list of (gt_tensor, pred_tensor) aligned with vis_items
        render_fn: render_hand_comparison function

    Returns:
        List of wandb.Image objects.
    """
    images = []
    for i, (item, (gt, pred)) in enumerate(zip(vis_items, gt_pred_pairs)):
        ctx = item["ctx"]
        vis_img = render_fn(ctx, ctx["frame_offset"], gt, pred)
        if vis_img is not None:
            images.append(wandb.Image(vis_img, caption=f"Frame {i}: Solid=GT, Wireframe=Pred"))
    return images


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def run_validation(model, val_loader, num_frames, device, criterion_kp3d, criterion_kp2d, criterion_param, mano_model, loss_weights, vis_clip_indices=None, lpips_scorer=None, max_batches=None):
    """Run validation and optionally capture gt/pred at specific clip indices.

    Returns (val_loss, val_terms, captured, hand_metrics, gs_metrics) where
    `hand_metrics` is the same HaMeR-style metric dict produced by
    `scripts.eval_hand_head` (computed via
    `scripts.hand_metrics.metrics_from_chunks`) and `gs_metrics` is the
    PSNR / SSIM / LPIPS dict from `scripts.gs_metrics.metrics_from_chunks`
    (or None if the model has `enable_gs=False`). Both match their
    standalone-eval counterparts bit-for-bit given the same val sequences.
    """
    model.eval()
    val_loss = 0.0
    val_terms = {
        "transl": 0.0, "global_orient": 0.0, "hand_pose": 0.0, "betas": 0.0,
        "kp3d": 0.0, "kp3d_abs": 0.0, "kp2d": 0.0,
        "gs_l1": 0.0, "gs_lpips": 0.0,
        "hand_depth_anchor": 0.0, "hand_depth_residual_m": 0.0,
    }
    captured = {}
    gs_captured = {}
    metric_chunks = []
    gs_chunks = []
    gs_hand_chunks = []   # region-masked (hand bbox) GS metrics
    eval_gs = bool(getattr(model, "enable_gs", False)) and (lpips_scorer is not None)
    batch_size = val_loader.batch_size
    n_processed = 0
    with torch.no_grad():
        for batch_idx, vbatch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
            # Cap the number of val batches: PSNR/MPJPE over a few dozen clips
            # tracks fine, and a full pass renders thousands of frames (hours).
            if max_batches is not None and batch_idx >= max_batches:
                break
            n_processed += 1
            imgs = vbatch["img"].to(device)
            gt = vbatch["gt"].to(device)
            hb = vbatch["hand_bboxes"].to(device) if "hand_bboxes" in vbatch else None
            hv = vbatch["hand_valid"].to(device)  if "hand_valid"  in vbatch else None
            views = build_views(imgs, num_frames, device, hb, hv)
            preds = model(views, is_inference=False, use_motion=False)
            pred_params = preds["hand_joints"]

            metric_chunks.append(metric_chunks_from_batch(
                pred_params, gt, hv, mano_model, device,
            ))

            loss_gs_l1 = torch.zeros((), device=device)
            loss_gs_lpips = torch.zeros((), device=device)
            if eval_gs:
                H_img, W_img = imgs.shape[-2:]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    rendered = render_views_from_predictions(
                        model, preds, views, height=H_img, width=W_img,
                    )
                gs_chunks.append(gs_metric_chunks_from_batch(
                    rendered, imgs, None, lpips_scorer, device,
                ))
                # Region-masked metrics over the hand bounding boxes. Full-frame
                # PSNR is background-dominated; this isolates the hand region the
                # prior acts on. Guarded so an eval-only error never kills a run.
                if hb is not None:
                    try:
                        hand_mask = _hand_region_mask(hb, hv, rendered.shape[2], rendered.shape[3])
                        gs_hand_chunks.append(gs_region_chunks_from_batch(
                            rendered, imgs, hand_mask, lpips_scorer, device,
                        ))
                    except Exception as _e:  # pragma: no cover
                        tqdm.write(f"[region-metrics] skipped: {type(_e).__name__}: {_e}")
                B_r, S_r = rendered.shape[:2]
                pred_chw = rendered.permute(0, 1, 4, 2, 3).reshape(B_r * S_r, 3, H_img, W_img)
                gt_chw = imgs.reshape(B_r * S_r, 3, H_img, W_img)
                loss_gs_l1 = F.l1_loss(pred_chw.float(), gt_chw.float())
                lpips_model = lpips_scorer._ensure()
                loss_gs_lpips = lpips_model(
                    pred_chw.float() * 2.0 - 1.0, gt_chw.float() * 2.0 - 1.0
                ).mean()
                if vis_clip_indices:
                    for item_idx in range(imgs.shape[0]):
                        clip_idx = batch_idx * batch_size + item_idx
                        if clip_idx in vis_clip_indices:
                            gs_captured[clip_idx] = {
                                "rendered": rendered[item_idx].float().cpu(),  # [S, H, W, 3]
                                "gt":       imgs[item_idx].float().cpu(),       # [S, 3, H, W]
                            }

            pred_joints = compute_joints_from_batch(pred_params, mano_model, device)
            B, S, H, J, _ = pred_joints.shape

            if hv is not None:
                has_hand = hv.float()
            else:
                gt_pack = gt.view(*gt.shape[:-1], NUM_HANDS, HAND_PARAM_DIM)
                has_hand = (gt_pack.abs().sum(dim=-1) > 1e-6).float()

            param_losses = criterion_param(pred_params, gt, has_hand)

            gt_joints = vbatch["gt_joints"].to(device)
            gt_conf = has_hand.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, J, 1)
            gt_input = torch.cat([gt_joints, gt_conf], dim=-1)
            pred_flat = pred_joints.view(B * S * H, 1, J, 3)
            gt_flat   = gt_input.view(B * S * H, 1, J, 4)
            loss_kp3d = criterion_kp3d(pred_flat, gt_flat, pelvis_id=0)
            loss_kp3d_abs = criterion_kp3d(pred_flat, gt_flat, pelvis_id=0, align_root=False)

            loss_kp2d = torch.zeros((), device=device)
            if "gt_joints_2d" in vbatch:
                cam_intr = vbatch["cam_intrinsics"].to(device)            # [B, 3]
                N = B * S
                pred_j  = pred_joints.view(N, H, J, 3)
                focal   = cam_intr[:, 0].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cx      = cam_intr[:, 1].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cy      = cam_intr[:, 2].unsqueeze(1).expand(B, S).reshape(N, 1, 1)

                # Clamp at 5 cm (below camera focal distance is nonphysical)
                # to keep focal*x/z finite when early pred params are degenerate.
                z = pred_j[..., 2].clamp_min(0.05)
                col = focal * pred_j[..., 0] / z + cx
                row = focal * pred_j[..., 1] / z + cy
                IMAGE_WIDTH = 1408.0
                # 90° CW to match project_vertices: (col, row) → (W-1-row, col)
                u = (IMAGE_WIDTH - 1.0) - row
                v = col
                pred_2d = torch.stack([u, v], dim=-1)

                pred_2d_norm = pred_2d / IMAGE_WIDTH - 0.5
                gt_2d        = vbatch["gt_joints_2d"].to(device)
                gt_2d_norm   = gt_2d.clone()
                gt_2d_norm[..., :2] = gt_2d[..., :2] / IMAGE_WIDTH - 0.5
                gt_2d_norm[..., 2]  = gt_2d_norm[..., 2] * has_hand.unsqueeze(-1)

                pred_2d_flat = pred_2d_norm.view(N * H, 1, J, 2)
                gt_2d_flat   = gt_2d_norm.view(N * H, 1, J, 3)
                loss_kp2d    = criterion_kp2d(pred_2d_flat, gt_2d_flat)

            # L1 HDGLA metric anchor (mirror of the train loop; default kwargs).
            loss_hand_anchor = torch.zeros((), device=device)
            anchor_residual_m = 0.0
            gs_depth_pred = preds.get("gs_depth")
            if gs_depth_pred is not None and "cam_intrinsics" in vbatch:
                loss_hand_anchor, _anchor_info = hand_depth_anchor_loss(
                    pred_joints, gs_depth_pred, has_hand, vbatch["cam_intrinsics"].to(device),
                )
                anchor_residual_m = _anchor_info["hand_depth_residual_m"]

            loss = (
                loss_weights["transl"]        * param_losses["transl"]
                + loss_weights["global_orient"] * param_losses["global_orient"]
                + loss_weights["hand_pose"]     * param_losses["hand_pose"]
                + loss_weights["betas"]         * param_losses["betas"]
                + loss_weights["kp3d"]          * loss_kp3d
                + loss_weights.get("kp3d_abs", 0.0) * loss_kp3d_abs
                + loss_weights["kp2d"]          * loss_kp2d
                + loss_weights.get("gs_l1", 0.0)    * loss_gs_l1
                + loss_weights.get("gs_lpips", 0.0) * loss_gs_lpips
                + loss_weights.get("hand_depth_anchor", 0.0) * loss_hand_anchor
            )

            val_loss += loss.item()
            for k in ("transl", "global_orient", "hand_pose", "betas"):
                val_terms[k] += param_losses[k].item()
            val_terms["kp3d"]  += loss_kp3d.item()
            val_terms["kp3d_abs"] += loss_kp3d_abs.item()
            val_terms["kp2d"]  += loss_kp2d.item()
            val_terms["gs_l1"]    += loss_gs_l1.item()
            val_terms["gs_lpips"] += loss_gs_lpips.item()
            val_terms["hand_depth_anchor"]    += loss_hand_anchor.item()
            val_terms["hand_depth_residual_m"] += anchor_residual_m

            # Hand-overlay capture needs a second, camera-space forward pass.
            # Only run it when this batch actually holds a target clip — avoids
            # a wasted 2x forward on every other val batch.
            batch_clip_idxs = [batch_idx * batch_size + i for i in range(imgs.shape[0])]
            if vis_clip_indices and any(ci in vis_clip_indices for ci in batch_clip_idxs):
                vis_preds = model(build_views(imgs, num_frames, device, hb, hv, crop_local_output=False), is_inference=False, use_motion=False)
                for item_idx, clip_idx in enumerate(batch_clip_idxs):
                    if clip_idx in vis_clip_indices:
                        captured[clip_idx] = {
                            "gt": gt[item_idx, 0].cpu(),
                            "pred": vis_preds["hand_joints"][item_idx, 0].cpu(),
                        }

    n = max(n_processed, 1)
    val_terms = {k: v / n for k, v in val_terms.items()}
    hand_metrics = metrics_from_chunks(metric_chunks)
    gs_metrics = gs_metrics_from_chunks(gs_chunks) if eval_gs else None
    if gs_metrics is not None and gs_hand_chunks:
        gs_hand = gs_metrics_from_chunks(gs_hand_chunks)
        gs_metrics["PSNR_hand"]  = gs_hand["PSNR"]
        gs_metrics["SSIM_hand"]  = gs_hand["SSIM"]
        gs_metrics["LPIPS_hand"] = gs_hand["LPIPS"]
        gs_metrics["num_valid_frames_hand"] = gs_hand["num_valid_frames"]
    return val_loss / n, val_terms, captured, hand_metrics, gs_metrics, gs_captured


def build_hand_pointcloud_3d(captured):
    """One wandb.Object3D per captured vis clip: GT hand joints (green) vs
    predicted joints (red) in camera frame. The 2D overlay can't show the depth
    axis -- where the placement error lives -- so this makes the residual
    inspectable in the W&B 3D viewer. Best-effort: skips clips with no valid
    joints. `captured[idx]` holds gt/pred tensors of shape [H, J, 3]."""
    from scripts.hand_vis_utils import hand_joints_to_rgb_points
    objs = []
    for clip_idx in sorted(captured):
        cap = captured[clip_idx]
        pts = hand_joints_to_rgb_points(cap["gt"].cpu().numpy(), cap["pred"].cpu().numpy())
        if pts is not None:
            objs.append(wandb.Object3D(pts))
    return objs


def build_gs_vis_videos(gs_captured, fps=8):
    """Build wandb.Video side-by-side (rendered | GT) clips over all frames
    for each captured val clip. Returns [] if nothing was captured.

    Iterates gs_captured directly so it does not depend on the projectaria
    hand-vis context (which is unavailable on aarch64 cluster nodes)."""
    videos = []
    for clip_idx in sorted(gs_captured.keys()):
        cap = gs_captured[clip_idx]
        rendered = cap["rendered"]                            # [S, H, W, 3] in [0, 1]
        gt = cap["gt"].permute(0, 2, 3, 1)                    # [S, H, W, 3] in [0, 1]
        side_by_side = torch.cat([rendered, gt], dim=2)       # [S, H, 2W, 3]
        # wandb.Video wants [T, C, H, W] uint8
        frames = (side_by_side.clamp(0, 1) * 255).to(torch.uint8).permute(0, 3, 1, 2).numpy()
        videos.append(wandb.Video(
            frames, fps=fps, format="mp4",
            caption=f"Clip {clip_idx}: Rendered | GT",
        ))
    return videos


def render_train_vis(model, train_vis_items, num_frames, device, render_fn):
    """Forward-pass fixed train clips and render visualizations."""
    model.eval()
    with torch.no_grad():
        imgs = torch.stack([it["img"] for it in train_vis_items]).to(device)
        hb = torch.stack([it["hand_bboxes"] for it in train_vis_items]).to(device) if "hand_bboxes" in train_vis_items[0] else None
        hv = torch.stack([it["hand_valid"]  for it in train_vis_items]).to(device) if "hand_valid"  in train_vis_items[0] else None
        # crop_local_output=False so predictions are in camera space for rendering
        preds = model(build_views(imgs, num_frames, device, hb, hv, crop_local_output=False), is_inference=False, use_motion=False)
        pairs = [
            (item["gt"][0], preds["hand_joints"][i, 0].cpu())
            for i, item in enumerate(train_vis_items)
        ]
    return render_vis_list(train_vis_items, pairs, render_fn)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def compute_joints_from_batch(params, mano_model, device):
    """Differentiable, batched conversion of 32-D MANO params to 3D joints.

    Args:
        params: [B, S, 64] — two hands packed as (left[32], right[32]).
        mano_model: MANOModel wrapper.

    Returns:
        [B, S, 2, 16, 3] joint tensor with autograd linked back to `params`.
    """
    B, S, D = params.shape
    assert D == NUM_HANDS * HAND_PARAM_DIM
    N = B * S
    flat = params.view(N, NUM_HANDS, HAND_PARAM_DIM)  # [N, 2, 32]
    left  = mano_model.get_joints_batched(flat[:, 0], is_right=False, device=device)  # [N, 16, 3]
    right = mano_model.get_joints_batched(flat[:, 1], is_right=True,  device=device)
    joints = torch.stack([left, right], dim=1)  # [N, 2, 16, 3]
    return joints.view(B, S, NUM_HANDS, joints.shape[-2], 3)


def _apply_overrides(cfg, overrides):
    """Apply dotted-key overrides like 'training.lr=3e-4' to a nested dict."""
    for ov in overrides:
        key, val = ov.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)



def train():

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hand_head.yaml")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VAL",
                        help="Config overrides, e.g. training.lr=3e-4 model.hamer_head_kwargs.depth=4")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        _apply_overrides(cfg, args.overrides)
        print(f"Config overrides: {args.overrides}")

    data_cfg     = cfg["data"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    wandb_cfg    = cfg.get("wandb", {})
    debug_cfg    = cfg.get("debug", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model ---
    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    checkpoint = torch.load(model_cfg["checkpoint"], map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("reconstructor", checkpoint))
    missing, _ = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. New (hand head) keys: {len(missing)}")
    model.to(device)

    vis_cfg = cfg.get("visualization", {})
    mano_folder = vis_cfg.get("mano_model_folder")
    if not mano_folder:
        raise RuntimeError("MANO model folder must be specified in config for training and visualization")
    from scripts.hand_vis_utils import MANOModel
    mano_model = MANOModel(mano_folder)

    criterion_kp3d  = Keypoint3DLoss(loss_type='l2').to(device)
    criterion_kp2d  = Keypoint2DLoss(loss_type='l1').to(device)
    criterion_param = ParameterLoss().to(device)

    # Optimizer parameter groups. Hand head is always trained. When the GS
    # branch is enabled we additionally train gs_head and (if built) the
    # hand→GS injection convs, in a single param group at the same LR.
    hand_params = list(model.hand_head.parameters())
    gs_head_params = (
        list(model.gs_head.parameters()) if getattr(model, "enable_gs", False) else []
    )
    injection_params = (
        list(model.hand_to_gs_injection.parameters())
        if getattr(model, "hand_to_gs_injection", None) is not None
        else []
    )
    trainable_params = hand_params + gs_head_params + injection_params
    print(
        "Trainable parameters: "
        f"hand={sum(p.numel() for p in hand_params):,} "
        f"gs_head={sum(p.numel() for p in gs_head_params):,} "
        f"injection={sum(p.numel() for p in injection_params):,} "
        f"total={sum(p.numel() for p in trainable_params):,}"
    )

    # --- Data ---
    all_seqs = discover_sequences(data_cfg["data_root"])
    if not all_seqs:
        raise RuntimeError(f"No sequences found in {data_cfg['data_root']}")
    print(f"Found {len(all_seqs)} sequences")

    if debug_cfg.get("enabled", False):
        max_seqs = debug_cfg.get("max_sequences", 5)
        all_seqs = all_seqs[:max_seqs]
        print(f"[DEBUG] Limited to {len(all_seqs)} sequences")

    num_frames       = data_cfg["num_frames"]
    res              = tuple(data_cfg["resolution"])
    clip_stride      = data_cfg.get("clip_stride", num_frames)
    batch_size       = training_cfg.get("batch_size", 2)
    grad_accum_steps = training_cfg.get("grad_accum_steps", 1)
    num_workers      = data_cfg.get("num_workers", 4)

    # Hand-crop dataset options (mirror the model flag)
    use_hand_crop = model_cfg.get("hand_head_type") == "hand_crop" or model_cfg.get("use_hand_crop", False)
    rescale_factor = cfg.get("hand_crop", {}).get("rescale_factor", 2.0)

    # L1 HDGLA metric anchor config (pulls gs_depth toward metric hand depth).
    anchor_cfg = cfg.get("hand_depth_anchor", {})
    w_anchor = cfg["loss_weights"].get("hand_depth_anchor", 0.0)
    anchor_margin = anchor_cfg.get("margin", 0.02)
    anchor_depth_min = anchor_cfg.get("depth_min", 0.01)
    anchor_conf_thresh = anchor_cfg.get("conf_thresh", 0.0)
    anchor_warmup_steps = int(anchor_cfg.get("warmup_steps", 800))
    anchor_direction = anchor_cfg.get("direction", "scene_follows_hand")
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 10.0))
    kp3d_abs_warmup_steps = int(training_cfg.get("kp3d_abs_warmup_steps", 0))

    ds_kwargs = dict(
        num_frames=num_frames, res=res, clip_stride=clip_stride,
        use_hand_crop=use_hand_crop, rescale_factor=rescale_factor,
    )

    if debug_cfg.get("single_frame", False):
        # Overfit on a single clip from the middle of the first sequence
        single_set = HOT3DHandDataset(all_seqs[:1], mano_model, **ds_kwargs)
        mid = len(single_set.clips) // 2
        single_set.clips = [single_set.clips[mid]]
        train_set = val_set = single_set
        print(f"[DEBUG] Single-frame overfit: seq={os.path.basename(all_seqs[0])}, clip offset={single_set.clips[0]['frame_offset']}")
    else:
        random.seed(training_cfg.get("seed", 42))
        random.shuffle(all_seqs)
        n_val = int(len(all_seqs) * float(data_cfg.get("val_split", 0.1)))
        # Hold out at least one sequence whenever we have more than one, so a
        # small val_split (e.g. 0.01) on a handful of sequences still produces
        # validation media + a best checkpoint instead of silently disabling both.
        if n_val == 0 and len(all_seqs) > 1:
            n_val = 1
        if n_val == 0:
            print("[WARN] No validation sequences — validation disabled, no best checkpoint will be saved")
        val_seqs, train_seqs = all_seqs[:n_val], all_seqs[n_val:]
        train_set = HOT3DHandDataset(train_seqs, mano_model, **ds_kwargs)
        val_set = HOT3DHandDataset(val_seqs, mano_model, **ds_kwargs) if val_seqs else None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    if val_set is not None and len(val_set.clips) > 0:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
        print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")
    else:
        val_set = None
        val_loader = None
        print(f"Train clips: {len(train_set)} | Val clips: 0")

    # --- Visualization setup ---
    num_vis_frames = vis_cfg.get("num_vis_frames", 4)
    render_fn = None
    val_vis_items = []
    train_vis_items = []

    has_val_clips = val_set is not None and len(val_set.clips) > 0
    if mano_folder and (has_val_clips or len(train_set.clips) > 0):
        from scripts.hand_vis_utils import render_hand_comparison
        
        render_fn = render_hand_comparison
        seq_cache = {}

        if has_val_clips:
            val_vis_items = setup_vis_items(val_set, num_vis_frames, seq_cache, mano_model)
        train_vis_items = setup_vis_items(train_set, num_vis_frames, seq_cache, mano_model, preload=True)

        if val_vis_items or train_vis_items:
            print(f"[VIS] {len(val_vis_items)} val + {len(train_vis_items)} train frames across {len(seq_cache)} sequences")

    val_vis_clip_indices = {it["clip_idx"] for it in val_vis_items}

    # GS overlay videos (rendered | GT) only need the rendered frames + GT
    # images that validation already captures — NOT the projectaria/Aria hand
    # context. Select a few val clips directly so the GS overlay still logs on
    # nodes where projectaria_tools (`_core_pybinds`) or Aria calib files are
    # unavailable and setup_vis_context returns None for every sequence.
    gs_vis_clip_indices = set()
    if val_set is not None and len(val_set.clips) > 0 and model_cfg.get("enable_gs", False):
        # First N clips → all land in the first batch(es), so they're captured
        # even when validation is capped to a few batches (val_max_batches).
        gs_vis_clip_indices = set(range(min(len(val_set.clips), num_vis_frames)))

    # Union drives validation capture; each media path renders only what it can.
    capture_clip_indices = (val_vis_clip_indices | gs_vis_clip_indices) or None

    # --- Optimizer & scheduler ---
    epochs     = training_cfg["epochs"]
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = epochs * steps_per_epoch
    optimizer  = Adam(trainable_params, lr=float(training_cfg["lr"]))
    scheduler  = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(training_cfg.get("min_lr", 1e-6)))

    log_every  = training_cfg.get("log_every", 500)
    val_every  = training_cfg.get("val_every", 2000)
    val_max_batches = training_cfg.get("val_max_batches", None)  # None = full val set
    # PROFILE_STEPS=N prints a CUDA-synced forward/render/backward breakdown for
    # the first N steps, then trains normally. Zero overhead when unset.
    profile_steps = int(os.environ.get("PROFILE_STEPS", "0"))
    use_amp = bool(training_cfg.get("use_amp", True))  # bf16 autocast on the forward
    save_every = training_cfg.get("save_every", 2000)
    keep_last_checkpoints = training_cfg.get("keep_last_checkpoints", 2)
    output_dir = training_cfg.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    best_val_mpjpe = float("inf")  # tracked separately: val_loss best != MPJPE best

    print(f"Training on {device} | {epochs} epochs | batch_size={batch_size} | grad_accum_steps={grad_accum_steps} | amp_bf16={use_amp}")

    # --- LPIPS scorer for GS-head image-quality metrics ---
    # Built once and reused across every validation run (and across train_gs
    # debug scripts). Skipped when the model was built with enable_gs=False.
    lpips_scorer = LPIPSScorer(device=device) if model_cfg.get("enable_gs", False) else None

    # --- W&B ---
    use_wandb = wandb_cfg.get("enabled", False)
    if use_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "hand-head-training"),
            entity=wandb_cfg.get("entity") or None,
            name=wandb_cfg.get("run_name") or None,
            tags=wandb_cfg.get("tags") or [],
            notes=wandb_cfg.get("notes") or None,
            config=cfg,
        )

    # --- Diagnostic: first-batch GT translation stats (sanity check the
    # crop-local / camera-frame transform after the fisheye fix). ---
    _diag_batch = next(iter(train_loader))
    _diag_gt = _diag_batch["gt"]  # [B, S, 64]
    for _hand_idx, _name in enumerate(("left", "right")):
        _off = _hand_idx * HAND_PARAM_DIM
        _t = _diag_gt[..., _off:_off + 3]
        _nz = _t.abs().sum(dim=-1) > 1e-6
        if _nz.any():
            _tv = _t[_nz]
            print(f"[DIAG] {_name} hand GT t_cam: "
                  f"min={_tv.min().item():.4f} max={_tv.max().item():.4f} "
                  f"mean={_tv.mean().item():.4f} std={_tv.std().item():.4f} "
                  f"(N={_nz.sum().item()})")
        else:
            print(f"[DIAG] {_name} hand GT t_cam: all-zero in first batch")
    del _diag_batch, _diag_gt

    # --- Sanity check: MANO path must be differentiable w.r.t. pred_params.
    # If this assert fires, loss_kp3d / loss_kp2d will silently contribute no
    # gradient to the head and training will look fine on the param losses only.
    _probe = 0.01 * torch.randn(1, num_frames, NUM_HANDS * HAND_PARAM_DIM,
                                device=device)
    _probe = _probe.detach().clone().requires_grad_(True)
    _probe_joints = compute_joints_from_batch(_probe, mano_model, device)
    _probe_joints.sum().backward()
    assert _probe.grad is not None and _probe.grad.abs().sum().item() > 0, (
        "MANO joint computation is not differentiable — check "
        "MANOModel.get_joints_batched / quat_wxyz_to_axis_angle_torch."
    )
    del _probe, _probe_joints

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    def _prune_numbered_checkpoints(keep_last):
        # Each checkpoint is ~5.7 GB; an unbounded save_every=100 run fills the
        # /work/scratch quota in ~12 saves (Errno 122 killed the P1b run). Keep
        # only the last `keep_last` numbered checkpoints; best_*/latest/final are
        # named differently and never pruned here.
        numbered = []
        for f in os.listdir(output_dir):
            if f.startswith("checkpoint_") and f.endswith(".pt"):
                try:
                    numbered.append((int(f[len("checkpoint_"):-len(".pt")]), f))
                except ValueError:
                    continue
        for _, f in sorted(numbered)[:-keep_last] if keep_last > 0 else sorted(numbered):
            try:
                os.remove(os.path.join(output_dir, f))
            except OSError:
                pass

    def save_checkpoint(step, epoch_idx, is_best=False, name=None, best_metric=None):
        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": step,
            "epoch": epoch_idx,
            "best_val_loss": best_val_loss,
        }
        if name is not None:
            filename = name
        elif best_metric is not None:
            filename = f"best_{best_metric}.pt"
        elif is_best:
            filename = "best_val_loss.pt"
        else:
            filename = f"checkpoint_{step}.pt"
        path = os.path.join(output_dir, filename)
        torch.save(checkpoint_state, path)
        torch.save(checkpoint_state, os.path.join(output_dir, "latest.pt"))
        if name is None and not is_best and best_metric is None:
            _prune_numbered_checkpoints(keep_last_checkpoints)

    latest_path = os.path.join(output_dir, "latest.pt")
    if os.path.exists(latest_path):
        print(f"Resuming from checkpoint: {latest_path}")
        ckpt_state = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt_state["model_state_dict"])
        optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])
        global_step = ckpt_state["global_step"]
        start_epoch = ckpt_state["epoch"]
        best_val_loss = ckpt_state["best_val_loss"]
        print(f"Resumed successfully. global_step={global_step}, start_epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")
    elif cfg["model"].get("warm_start_hand_head"):
        # Initialise the hand head from a prior checkpoint (e.g. the 50mm run)
        # instead of training it from scratch, so the abs-3D loss only has to
        # refine an already-converged placement. Only when NOT resuming.
        ws_path = cfg["model"]["warm_start_hand_head"]
        print(f"Warm-starting hand head from {ws_path}")
        ws = torch.load(ws_path, map_location=device)
        sd = ws["model_state_dict"] if isinstance(ws, dict) and "model_state_dict" in ws else ws
        hh_keys = set(model.hand_head.state_dict().keys())
        loaded = {k: v for k, v in sd.items() if k in hh_keys}
        if loaded:
            res = model.hand_head.load_state_dict(loaded, strict=False)
            print(f"Warm-start: loaded {len(loaded)}/{len(hh_keys)} hand_head tensors "
                  f"(missing={len(res.missing_keys)}, unexpected=0)")
        else:
            res = model.load_state_dict(sd, strict=False)
            print(f"Warm-start: no bare hand_head keys matched; loaded full-model dict "
                  f"(missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")

    # Opt-in autograd anomaly detection (DETECT_ANOMALY=1): raises at the first
    # op that produces a non-finite value in the backward pass, with a traceback
    # to the forward call site. Used to root-cause the P1a NaN-grad freeze.
    if os.environ.get("DETECT_ANOMALY", "0") == "1":
        torch.autograd.set_detect_anomaly(True)
        tqdm.write("[anomaly] set_detect_anomaly(True): backward will raise at the first non-finite op")

    # --- Per-term NaN isolation (P1a NaN hunt, round 2) ---
    # The quat-degenerate guard removed one NaN source, but grads still go
    # non-finite near full abs-loss ramp (zombie runs: steps 87 / 179 / 186).
    # Screen the total grad after every batch backward; on the first
    # finite->non-finite transition, switch to isolation mode: on subsequent
    # batches, backward each weighted loss term ALONE and report which one
    # produces non-finite grads, then halt (exit 3) instead of zombie-training
    # with every optimizer step skipped.
    def _grads_finite(params):
        with torch.no_grad():
            total = torch.zeros((), device=device)
            for p in params:
                if p.grad is not None:
                    total += p.grad.float().pow(2).sum()
            return bool(torch.isfinite(total)), float(total.sqrt())

    def _isolate_nan_terms(named_terms, params):
        """Backward each weighted term alone (retain_graph). True if culprit found."""
        culprit = False
        for name, term in named_terms:
            if not torch.is_tensor(term) or not term.requires_grad:
                tqdm.write(f"[nan-isolate]   {name}: value={float(term):.6f} (no graph, skipped)")
                continue
            if not torch.isfinite(term):
                tqdm.write(f"[nan-isolate]   {name}: FORWARD value non-finite ({term.item()})")
                culprit = True
                continue
            optimizer.zero_grad(set_to_none=True)
            term.backward(retain_graph=True)
            ok, gn = _grads_finite(params)
            # Flush per term: if a later backward OOMs, partial verdicts survive.
            tqdm.write(f"[nan-isolate]   {name}: value={term.item():.6f} grad_norm={gn:.3e} finite={ok}")
            sys.stdout.flush()
            if not ok:
                culprit = True
        optimizer.zero_grad(set_to_none=True)
        return culprit

    nan_isolate_pending = 0      # batches left in isolation mode (0 = normal training)
    NAN_ISOLATE_BATCHES = 32     # give up after this many isolation batches
    NAN_GUARD_HALT_AFTER = 5     # consecutive skipped steps before halting
    prev_batch_grads_finite = True
    consec_nan_guard = 0

    # --- Training loop ---
    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs"):
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_terms = {
            "transl": 0.0, "global_orient": 0.0, "hand_pose": 0.0, "betas": 0.0,
            "kp3d": 0.0, "kp3d_abs": 0.0, "kp2d": 0.0,
            "gs_l1": 0.0, "gs_lpips": 0.0,
            "hand_depth_anchor": 0.0, "hand_depth_residual_m": 0.0,
        }

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}", leave=False)):
            if epoch == start_epoch and batch_idx < (global_step % steps_per_epoch) * grad_accum_steps:
                continue
            imgs = batch["img"].to(device)
            gt_params = batch["gt"].to(device)
            gt_joints = batch["gt_joints"].to(device)

            hb = batch["hand_bboxes"].to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].to(device)  if "hand_valid"  in batch else None
            
            # print(f"gt_params: {gt_params}")
            # print(f"gt_joints: {gt_joints}")

            _prof = profile_steps and global_step < profile_steps
            def _lap():
                if _prof:
                    torch.cuda.synchronize()
                    return time.perf_counter()
                return 0.0
            _t0 = _lap()

            views_train = build_views(imgs, num_frames, device, hb, hv)

            # PROFILE_TORCH=1 (with PROFILE_STEPS>0): op-level breakdown of ONE
            # forward to find what the flat-under-bf16 20s actually is. Prints
            # top ops by CUDA and CPU self-time, then exits.
            if os.environ.get("PROFILE_TORCH") and _prof:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True) as _tp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        _ = model(views_train, is_inference=False, use_motion=False)
                    torch.cuda.synchronize()
                ka = _tp.key_averages()
                tqdm.write("\n[TORCH-PROFILE] ===== top 18 by CUDA self-time =====")
                tqdm.write(ka.table(sort_by="self_cuda_time_total", row_limit=18))
                tqdm.write("\n[TORCH-PROFILE] ===== top 18 by CPU self-time =====")
                tqdm.write(ka.table(sort_by="self_cpu_time_total", row_limit=18))
                raise SystemExit("[TORCH-PROFILE] done — remove PROFILE_TORCH to train")

            # Backbone forward dominates the step (~20s fp32). It's frozen, so
            # running it under bf16 autocast is safe and ~2-3x faster. Upcast the
            # float outputs back to fp32 so the MANO joint solve, metric anchor,
            # and rasterizer all see fp32 (matching the no-AMP numerics).
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                preds = model(views_train, is_inference=False, use_motion=False)
            if use_amp:
                preds = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                         for k, v in preds.items()}
            pred_params = preds["hand_joints"] # [B, S, 64]
            _t_fwd = _lap()

            pred_joints = compute_joints_from_batch(pred_params, mano_model, device)

            # GS reconstruction loss (L1 + LPIPS) on the rendered views.
            # Gated on model.enable_gs (so disabled-GS configs are bit-for-bit
            # identical to the no-GS codepath) and on lpips_scorer presence.
            loss_gs_l1 = torch.zeros((), device=device)
            loss_gs_lpips = torch.zeros((), device=device)
            rendered = None
            if getattr(model, "enable_gs", False) and lpips_scorer is not None:
                H_img, W_img = imgs.shape[-2:]
                rendered = render_views_from_predictions(
                    model, preds, views_train, height=H_img, width=W_img,
                )
            _t_render = _lap()
            if rendered is not None:
                B_r, S_r = rendered.shape[:2]
                pred_chw = rendered.permute(0, 1, 4, 2, 3).reshape(B_r * S_r, 3, H_img, W_img)
                gt_chw = imgs.reshape(B_r * S_r, 3, H_img, W_img)
                loss_gs_l1 = F.l1_loss(pred_chw, gt_chw)
                lpips_model = lpips_scorer._ensure()
                loss_gs_lpips = lpips_model(pred_chw * 2.0 - 1.0, gt_chw * 2.0 - 1.0).mean()

            loss_kp2d = torch.zeros((), device=device)

            # Per-hand presence mask — prefer the dataset's hand_valid (derived from
            # the projected-bbox pipeline); fall back to a params-are-nonzero check.
            if hv is not None:
                has_hand = hv.to(device).float()                 # [B, S, 2]
            else:
                gt_pack = gt_params.view(*gt_params.shape[:-1], NUM_HANDS, HAND_PARAM_DIM)
                has_hand = (gt_pack.abs().sum(dim=-1) > 1e-6).float()

            # Parameter loss — split per MANO key, each masked per-hand.
            # Returns dict with 'transl', 'global_orient', 'hand_pose', 'betas'.
            param_losses = criterion_param(pred_params, gt_params, has_hand)

            # 3D keypoint loss. Confidence is 1 where the hand is present, 0 otherwise —
            # absent hands have zero GT joints but MANO's default pose would otherwise
            # produce a constant ~0.2 residual that never decays.
            B, S, H, J, _ = pred_joints.shape
            gt_conf = has_hand.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, J, 1)
            gt_input = torch.cat([gt_joints, gt_conf], dim=-1)                # [B, S, H, J, 4]
            pred_flat = pred_joints.view(B * S * H, 1, J, 3)
            gt_flat   = gt_input.view(B * S * H, 1, J, 4)
            loss_kp3d = criterion_kp3d(pred_flat, gt_flat, pelvis_id=0)
            # Absolute (non-root-relative) 3D loss: the root-relative loss above
            # is translation-invariant, leaving global hand placement supervised
            # only by the MANO transl term + 2D reprojection. This term directly
            # penalises absolute camera-frame position (metric depth).
            loss_kp3d_abs = criterion_kp3d(pred_flat, gt_flat, pelvis_id=0, align_root=False)

            # 2D reprojection loss. pred_joints are camera-frame (post-transform),
            # so skip the world→camera extrinsic and project with intrinsics only.
            if "gt_joints_2d" in batch:
                cam_intr = batch["cam_intrinsics"].to(device)              # [B, 3]
                N = B * S
                pred_j  = pred_joints.view(N, H, J, 3)                     # [N, H, J, 3]
                focal   = cam_intr[:, 0].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cx      = cam_intr[:, 1].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cy      = cam_intr[:, 2].unsqueeze(1).expand(B, S).reshape(N, 1, 1)

                # Clamp at 5 cm (below camera focal distance is nonphysical)
                # to keep focal*x/z finite when early pred params are degenerate.
                z = pred_j[..., 2].clamp_min(0.05)
                col = focal * pred_j[..., 0] / z + cx
                row = focal * pred_j[..., 1] / z + cy
                IMAGE_WIDTH = 1408.0
                # 90° CW to match project_vertices: (col, row) → (W-1-row, col)
                u = (IMAGE_WIDTH - 1.0) - row
                v = col
                pred_2d = torch.stack([u, v], dim=-1)                      # [N, H, J, 2]

                # Normalize to [-0.5, 0.5] so residuals match HaMeR's convention.
                pred_2d_norm = pred_2d / IMAGE_WIDTH - 0.5
                gt_2d        = batch["gt_joints_2d"].to(device)            # [B, S, H, J, 3]
                gt_2d_norm   = gt_2d.clone()
                gt_2d_norm[..., :2] = gt_2d[..., :2] / IMAGE_WIDTH - 0.5
                gt_2d_norm[..., 2]  = gt_2d_norm[..., 2] * has_hand.unsqueeze(-1)

                pred_2d_flat = pred_2d_norm.view(N * H, 1, J, 2)
                gt_2d_flat   = gt_2d_norm.view(N * H, 1, J, 3)
                loss_kp2d    = criterion_kp2d(pred_2d_flat, gt_2d_flat)

            # L1 HDGLA metric anchor: pull gs_depth toward metric hand depth at the joints.
            loss_hand_anchor = torch.zeros((), device=device)
            anchor_residual_m = 0.0
            gs_depth_pred = preds.get("gs_depth")
            if w_anchor > 0.0 and gs_depth_pred is not None and "cam_intrinsics" in batch:
                loss_hand_anchor, _anchor_info = hand_depth_anchor_loss(
                    pred_joints, gs_depth_pred, has_hand, batch["cam_intrinsics"].to(device),
                    margin=anchor_margin, depth_min=anchor_depth_min,
                    gs_depth_conf=preds.get("gs_depth_conf"), conf_thresh=anchor_conf_thresh,
                    direction=anchor_direction,
                )
                anchor_residual_m = _anchor_info["hand_depth_residual_m"]
            anchor_ramp = min(1.0, global_step / anchor_warmup_steps) if anchor_warmup_steps > 0 else 1.0
            abs_ramp = min(1.0, global_step / kp3d_abs_warmup_steps) if kp3d_abs_warmup_steps > 0 else 1.0

            w = cfg["loss_weights"]
            loss = (
                w["transl"]        * param_losses["transl"]
                + w["global_orient"] * param_losses["global_orient"]
                + w["hand_pose"]     * param_losses["hand_pose"]
                + w["betas"]         * param_losses["betas"]
                + w["kp3d"]          * loss_kp3d
                + w.get("kp3d_abs", 0.0) * abs_ramp * loss_kp3d_abs
                + w["kp2d"]          * loss_kp2d
                + w.get("gs_l1", 0.0)    * loss_gs_l1
                + w.get("gs_lpips", 0.0) * loss_gs_lpips
                + w.get("hand_depth_anchor", 0.0) * anchor_ramp * loss_hand_anchor
            )

            # Isolation mode: backward each weighted term alone on this batch,
            # ordered most-suspicious-first so an OOM mid-diagnostic still
            # leaves verdicts for the likely culprits in the log.
            if nan_isolate_pending > 0:
                nan_isolate_pending -= 1
                named_terms = [
                    ("kp3d_abs(w*ramp)",          w.get("kp3d_abs", 0.0) * abs_ramp * loss_kp3d_abs),
                    ("hand_depth_anchor(w*ramp)", w.get("hand_depth_anchor", 0.0) * anchor_ramp * loss_hand_anchor),
                    ("kp2d(w)",                   w["kp2d"] * loss_kp2d),
                    ("gs_lpips(w)",               w.get("gs_lpips", 0.0) * loss_gs_lpips),
                    ("gs_l1(w)",                  w.get("gs_l1", 0.0) * loss_gs_l1),
                    ("kp3d(w)",                   w["kp3d"] * loss_kp3d),
                    ("transl(w)",                 w["transl"] * param_losses["transl"]),
                    ("global_orient(w)",          w["global_orient"] * param_losses["global_orient"]),
                    ("hand_pose(w)",              w["hand_pose"] * param_losses["hand_pose"]),
                    ("betas(w)",                  w["betas"] * param_losses["betas"]),
                ]
                tqdm.write(f"[nan-isolate] batch {batch_idx} step {global_step} "
                           f"seq={batch.get('seq_path', '?')} offset={batch.get('frame_offset', '?')}")
                found = _isolate_nan_terms(named_terms, trainable_params)
                if not found:
                    # No single term bad in isolation -- check the combination.
                    loss.backward()
                    ok, gn = _grads_finite(trainable_params)
                    tqdm.write(f"[nan-isolate]   COMBINED: grad_norm={gn:.3e} finite={ok}")
                    optimizer.zero_grad(set_to_none=True)
                    found = not ok
                sys.stdout.flush()
                if found:
                    tqdm.write("[nan-isolate] culprit(s) reported above; halting (exit 3)")
                    sys.stdout.flush()
                    sys.exit(3)
                if nan_isolate_pending == 0:
                    tqdm.write("[nan-isolate] no culprit in isolation window; halting (exit 3)")
                    sys.stdout.flush()
                    sys.exit(3)
                continue  # skip the normal accum path while isolating

            _t_loss = _lap()
            (loss / grad_accum_steps).backward()
            _t_bwd = _lap()
            if _prof:
                tqdm.write(
                    f"[PROFILE] step {global_step}: "
                    f"fwd={_t_fwd-_t0:.2f}s render={_t_render-_t_fwd:.2f}s "
                    f"loss={_t_loss-_t_render:.2f}s bwd={_t_bwd-_t_loss:.2f}s "
                    f"| total={_t_bwd-_t0:.2f}s"
                )

            # Per-batch screen: catch the exact batch whose backward first turns
            # the accumulated grads non-finite, then enter isolation mode.
            batch_grads_ok, _ = _grads_finite(trainable_params)
            if not batch_grads_ok and prev_batch_grads_finite:
                tqdm.write(f"[nan-isolate] first non-finite grad after batch {batch_idx} "
                           f"(step {global_step}); per-term isolation for next {NAN_ISOLATE_BATCHES} batches")
                sys.stdout.flush()
                nan_isolate_pending = NAN_ISOLATE_BATCHES
            prev_batch_grads_finite = batch_grads_ok

            accum_loss += loss.item()
            for k in ("transl", "global_orient", "hand_pose", "betas"):
                accum_terms[k] += param_losses[k].item()
            accum_terms["kp3d"]  += loss_kp3d.item()
            accum_terms["kp3d_abs"] += loss_kp3d_abs.item()
            accum_terms["kp2d"]  += loss_kp2d.item()
            accum_terms["gs_l1"]    += loss_gs_l1.item()
            accum_terms["gs_lpips"] += loss_gs_lpips.item()
            accum_terms["hand_depth_anchor"]    += loss_hand_anchor.item()
            accum_terms["hand_depth_residual_m"] += anchor_residual_m

            if (batch_idx + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                # NaN guard: a single bad batch can produce non-finite grads; applying
                # them poisons the weights (then SVD/quat ops crash). Skip the step.
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    scheduler.step()
                    consec_nan_guard = 0
                else:
                    consec_nan_guard += 1
                    tqdm.write(f"[nan-guard] non-finite grad_norm at step {global_step}; "
                               f"skipping optimizer step ({consec_nan_guard} consecutive)")
                    if consec_nan_guard >= NAN_GUARD_HALT_AFTER:
                        tqdm.write(f"[nan-guard] {NAN_GUARD_HALT_AFTER} consecutive non-finite steps; "
                                   f"halting instead of zombie-training (exit 4)")
                        sys.stdout.flush()
                        sys.exit(4)
                optimizer.zero_grad()
                avg_loss = accum_loss / grad_accum_steps
                avg_terms = {k: v / grad_accum_steps for k, v in accum_terms.items()}
                accum_loss = 0.0
                accum_terms = {
                    "transl": 0.0, "global_orient": 0.0, "hand_pose": 0.0, "betas": 0.0,
                    "kp3d": 0.0, "kp3d_abs": 0.0, "kp2d": 0.0,
                    "gs_l1": 0.0, "gs_lpips": 0.0,
                    "hand_depth_anchor": 0.0, "hand_depth_residual_m": 0.0,
                }
                global_step += 1

                # --- Train logging ---
                if use_wandb:
                    wandb.log({"train/loss": avg_loss,
                               "train/loss_transl":        avg_terms["transl"],
                               "train/loss_global_orient": avg_terms["global_orient"],
                               "train/loss_hand_pose":     avg_terms["hand_pose"],
                               "train/loss_betas":         avg_terms["betas"],
                               "train/loss_kp3d":          avg_terms["kp3d"],
                               "train/loss_kp3d_abs":      avg_terms["kp3d_abs"],
                               "train/loss_kp2d":          avg_terms["kp2d"],
                               "train/loss_gs_l1":         avg_terms["gs_l1"],
                               "train/loss_gs_lpips":      avg_terms["gs_lpips"],
                               "train/loss_hand_depth_anchor": avg_terms["hand_depth_anchor"],
                               "train/hand_depth_residual_m":  avg_terms["hand_depth_residual_m"],
                               "train/grad_norm":          grad_norm.item(),
                               "lr": scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % log_every == 0 or global_step == 1:
                    lr = scheduler.get_last_lr()[0]
                    tqdm.write(
                        f"  step {global_step} | train_loss={avg_loss:.4f} "
                        f"(t={avg_terms['transl']:.4f} o={avg_terms['global_orient']:.4f} "
                        f"p={avg_terms['hand_pose']:.4f} b={avg_terms['betas']:.4f} "
                        f"kp3d={avg_terms['kp3d']:.4f} kp2d={avg_terms['kp2d']:.4f} "
                        f"gs_l1={avg_terms['gs_l1']:.4f} gs_lpips={avg_terms['gs_lpips']:.4f}) "
                        f"| grad_norm={grad_norm.item():.4f} | lr={lr:.2e}"
                    )
                    if use_wandb and train_vis_items:
                        train_images = render_train_vis(model, train_vis_items, num_frames, device, render_fn)
                        if train_images:
                            wandb.log({"media/train_hand_overlay": train_images}, step=global_step)
                        model.train()

                # --- Validation ---
                if val_loader and (global_step % val_every == 0 or global_step == 1):
                    val_loss, val_terms, captured, hand_metrics, gs_metrics, gs_captured = run_validation(model, val_loader, num_frames, device, criterion_kp3d, criterion_kp2d, criterion_param, mano_model, cfg["loss_weights"], capture_clip_indices, lpips_scorer=lpips_scorer, max_batches=val_max_batches)
                    tqdm.write(
                        f"  step {global_step} | val_loss={val_loss:.4f} "
                        f"(t={val_terms['transl']:.4f} o={val_terms['global_orient']:.4f} "
                        f"p={val_terms['hand_pose']:.4f} b={val_terms['betas']:.4f} "
                        f"kp3d={val_terms['kp3d']:.4f} kp2d={val_terms['kp2d']:.4f} "
                        f"gs_l1={val_terms['gs_l1']:.4f} gs_lpips={val_terms['gs_lpips']:.4f})"
                    )
                    hm_all = hand_metrics.get("all")
                    if hm_all is not None:
                        tqdm.write(
                            f"  hand_metrics(all): MPJPE={hm_all['MPJPE']:.2f}mm "
                            f"PA={hm_all['PA_MPJPE']:.2f}mm "
                            f"MPVPE={hm_all['MPVPE']:.2f}mm "
                            f"PA={hm_all['PA_MPVPE']:.2f}mm "
                            f"AUC_J={hm_all['AUC_J']:.3f} AUC_V={hm_all['AUC_V']:.3f}"
                        )
                    if gs_metrics is not None and gs_metrics["num_valid_frames"] > 0:
                        tqdm.write(
                            f"  gs_metrics: PSNR={gs_metrics['PSNR']:.2f}dB "
                            f"SSIM={gs_metrics['SSIM']:.4f} "
                            f"LPIPS={gs_metrics['LPIPS']:.4f} "
                            f"(N={gs_metrics['num_valid_frames']})"
                        )

                    if use_wandb:
                        log_dict = {"val/loss": val_loss,
                                    "val/loss_transl":        val_terms["transl"],
                                    "val/loss_global_orient": val_terms["global_orient"],
                                    "val/loss_hand_pose":     val_terms["hand_pose"],
                                    "val/loss_betas":         val_terms["betas"],
                                    "val/loss_kp3d":          val_terms["kp3d"],
                                    "val/loss_kp3d_abs":      val_terms["kp3d_abs"],
                                    "val/loss_kp2d":          val_terms["kp2d"],
                                    "val/loss_gs_l1":         val_terms["gs_l1"],
                                    "val/loss_gs_lpips":      val_terms["gs_lpips"],
                                    "val/loss_hand_depth_anchor": val_terms["hand_depth_anchor"],
                                    "val/hand_depth_residual_m":  val_terms["hand_depth_residual_m"]}
                        for side_label in ("left", "right", "all"):
                            side_metrics = hand_metrics.get(side_label)
                            if side_metrics is None:
                                continue
                            for k, v in side_metrics.items():
                                log_dict[f"hand_metrics/{side_label}/{k}"] = v
                        log_dict["hand_metrics/num_valid_hands"] = hand_metrics["num_valid_hands"]
                        if gs_metrics is not None and gs_metrics["num_valid_frames"] > 0:
                            log_dict["gaussian_metrics/PSNR"]  = gs_metrics["PSNR"]
                            log_dict["gaussian_metrics/SSIM"]  = gs_metrics["SSIM"]
                            log_dict["gaussian_metrics/LPIPS"] = gs_metrics["LPIPS"]
                            log_dict["gaussian_metrics/num_valid_frames"] = gs_metrics["num_valid_frames"]
                            if gs_metrics.get("num_valid_frames_hand", 0) > 0:
                                log_dict["gaussian_metrics/PSNR_hand"]  = gs_metrics["PSNR_hand"]
                                log_dict["gaussian_metrics/SSIM_hand"]  = gs_metrics["SSIM_hand"]
                                log_dict["gaussian_metrics/LPIPS_hand"] = gs_metrics["LPIPS_hand"]
                                log_dict["gaussian_metrics/num_valid_frames_hand"] = gs_metrics["num_valid_frames_hand"]
                        # Media is best-effort: a missing ffmpeg (or any encode
                        # error) must never crash a training run. Metrics are
                        # logged regardless.
                        try:
                            if val_vis_items:
                                pairs = [
                                    (captured[it["clip_idx"]]["gt"], captured[it["clip_idx"]]["pred"])
                                    for it in val_vis_items if it["clip_idx"] in captured
                                ]
                                val_images = render_vis_list(val_vis_items, pairs, render_fn)
                                if val_images:
                                    log_dict["media/val_hand_overlay"] = val_images
                        except Exception as _e:
                            tqdm.write(f"[media] hand overlay skipped: {type(_e).__name__}: {_e}")
                        try:
                            if gs_captured:
                                gs_videos = build_gs_vis_videos(gs_captured)
                                if gs_videos:
                                    log_dict["media/val_gs_overlay"] = gs_videos
                        except Exception as _e:
                            tqdm.write(f"[media] gs overlay skipped: {type(_e).__name__}: {_e}")
                        try:
                            if captured:
                                hand_pc = build_hand_pointcloud_3d(captured)
                                if hand_pc:
                                    log_dict["media/val_hand_pointcloud"] = hand_pc
                        except Exception as _e:
                            tqdm.write(f"[media] 3d hand pointcloud skipped: {type(_e).__name__}: {_e}")
                        try:
                            wandb.log(log_dict, step=global_step)
                        except Exception as _e:
                            tqdm.write(f"[wandb] log with media failed ({type(_e).__name__}); metrics-only")
                            wandb.log({k: v for k, v in log_dict.items()
                                       if not k.startswith("media/")}, step=global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(global_step, epoch, is_best=True)
                        tqdm.write("  -> New best val_loss. Saved.")
                    # MPJPE is the actual objective and does NOT track val_loss
                    # (the ramping abs-3D term inflates val_loss while MPJPE
                    # drops). Retain the best-MPJPE checkpoint by its own name so
                    # it survives numbered-checkpoint pruning. P1a's best (92.8mm
                    # @ step 300) would otherwise have been lost to retention.
                    hm_all_best = hand_metrics.get("all") if hand_metrics else None
                    if hm_all_best is not None and hm_all_best["MPJPE"] < best_val_mpjpe:
                        best_val_mpjpe = hm_all_best["MPJPE"]
                        save_checkpoint(global_step, epoch, best_metric="mpjpe")
                        tqdm.write(f"  -> New best MPJPE {best_val_mpjpe:.2f}mm. Saved.")
                    model.train()

                if global_step % save_every == 0:
                    save_checkpoint(global_step, epoch)

        # Flush leftover gradients from an incomplete accumulation window
        if (batch_idx + 1) % grad_accum_steps != 0:
            optimizer.zero_grad()

    # --- Save final ---
    save_checkpoint(global_step, epoch, name="hand_head_final.pt")
    print(f"Final weights saved to: {os.path.join(output_dir, 'hand_head_final.pt')}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
