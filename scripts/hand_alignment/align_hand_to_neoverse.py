"""
align_hand_to_neoverse.py

Take a short Hot3D clip, run NeoVerse on it, then warp the GT MANO hand into
NeoVerse's normalized scene frame using a closed-form similarity transform
recovered from the camera-pose pairs.

Within one 16-frame chunk, NeoVerse's per-batch pose normalization (see
diffsynth/auxiliary_models/worldmirror/models/utils/priors.py:normalize_poses)
makes the GT and predicted camera trajectories differ by a single similarity
(uniform scale s, rotation R, translation t). R and t are fully determined by
the per-frame camera-pose pairs — only the scale s is genuinely ambiguous.

This is the Phase-1 verification script. If the visualization looks right, the
single-similarity assumption holds and the GT supervision can be rewritten in
the normalized frame. If it looks wrong, residuals printed here indicate
whether rotation alignment or the scalar scale is at fault.

Outputs (mirrors scripts/reconstruct_4dgs.py layout so view_4dgs.py also works):
  gaussians.pt            - list of per-group Gaussian dicts
  camera_params.json      - per-frame intrinsics + cam2world (NeoVerse frame)
  gaussians_frame0000.ply - frame-0 Gaussians PLY
  hand_meshes.pt          - per-frame, per-hand MANO meshes in NeoVerse frame
  hand_meshes/*.obj       - same meshes as standalone OBJ files
  alignment.json          - {s, R, t, per-frame & aggregate residuals}
  camera_trajectories.npz - GT/pred c2w + camera centers (for debugging)

Usage:
  python scripts/align_hand_to_neoverse.py
  python scripts/align_hand_to_neoverse.py --sequence_id P0001_10a27bf7 --frame_start 0
  python scripts/align_hand_to_neoverse.py --launch_viser
"""

import argparse
import json
import os
import sys

# Resolve repo root (../../) so `from scripts.X import ...` works from a subdir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import trimesh
from torchvision.transforms import functional as TVF
from projectaria_tools.core.sophus import SE3

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.utils.auxiliary import load_video
from projectaria_tools.core import calibration as aria_calib
from diffsynth.auxiliary_models.worldmirror.utils.save_utils import (
    save_gs_ply,
    save_camera_params,
)
from scripts.hand_vis_utils import (
    MANOModel,
    setup_vis_context,
    _frame_to_timecode,
    find_closest,
)


HOT3D_RAW_DATA_DIR = "/home/marian/3dv/data/hot3d_aria/raw_data"
SH_C0 = 0.28209479177387814
ARIA_FRAME_SIZE = 1408  # native Aria fisheye resolution


def build_pinhole_target(fisheye_calib, focal):
    """Build a linear (pinhole) camera matching the source's image size and
    physical position (T_Device_Camera), in SENSOR orientation, with the
    given focal length."""
    return aria_calib.get_linear_camera_calibration(
        ARIA_FRAME_SIZE, ARIA_FRAME_SIZE, float(focal), "pinhole_target",
        fisheye_calib.get_transform_device_camera(),
    )


def undistort_mp4_frame(frame_mp4_rgb, fisheye_calib, pinhole_calib):
    """Warp an MP4 (display-orientation) fisheye frame to MP4-orientation pinhole.

    MP4 is rotated 90° CW from the sensor orientation that the calibration
    describes, so we rotate CCW first to align with the calibration, undistort,
    then rotate CW to put the result back in upright/MP4 orientation.
    """
    frame_sensor = np.rot90(frame_mp4_rgb, k=1).copy()
    und_sensor = aria_calib.distort_by_calibration(
        frame_sensor, pinhole_calib, fisheye_calib,
    )
    return np.rot90(und_sensor, k=-1).copy()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequence_id", default=None,
                   help="Hot3D sequence folder name (e.g. P0001_10a27bf7). "
                        "Default: first alphabetically under --data_root.")
    p.add_argument("--data_root", default=HOT3D_RAW_DATA_DIR)
    p.add_argument("--frame_start", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")
    p.add_argument("--mano_model_folder", default="models/MANO")
    p.add_argument("--undistort", action="store_true",
                   help="Undistort the Aria fisheye frames to a pinhole camera "
                        "before feeding to NeoVerse. Removes the fisheye-vs-pinhole "
                        "bias that affects scale recovery and the pinhole 2D overlay.")
    p.add_argument("--undistort_focal", type=float, default=280.0,
                   help="Focal length (px) of the target pinhole at 1408x1408. "
                        "Lower = wider FOV, more retained scene but more edge stretching.")
    p.add_argument("--output_dir", default=None,
                   help="Default: outputs/hand_alignment/<sequence>_f<start>")
    p.add_argument("--height", type=int, default=336)
    p.add_argument("--width", type=int, default=336)
    p.add_argument("--launch_viser", action="store_true",
                   help="Launch view_4dgs_with_hand.py at the end")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


# -----------------------------------------------------------------------------
# GT camera poses (cam2world in Hot3D world frame, metric)
# -----------------------------------------------------------------------------

def gt_camera_poses(ctx, frame_indices):
    """Return per-frame T_world_camera (cam2world) as a [N, 4, 4] numpy array.

    Uses the same frame→timecode mapping + headset trajectory + device-camera
    extrinsic as scripts/hand_vis_utils.project_vertices.
    """
    n_video = ctx["n_video"]
    headset_ts = ctx["headset_ts_sorted"]
    headset_poses = ctx["headset_poses"]
    hand_ts = ctx["hand_ts_sorted"]
    T_dc_mat = ctx["T_device_camera"].to_matrix()                    # [4, 4]

    out = np.zeros((len(frame_indices), 4, 4), dtype=np.float64)
    for k, frame_i in enumerate(frame_indices):
        query_tc = _frame_to_timecode(frame_i, n_video, hand_ts)
        h_ts = find_closest(headset_ts, query_tc)
        t_wd, q_wd = headset_poses[h_ts]
        T_wd = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0].to_matrix()
        out[k] = T_wd @ T_dc_mat
    return out


# -----------------------------------------------------------------------------
# Per-frame similarity from camera-pose pairs (only the scalar `s` is shared)
# -----------------------------------------------------------------------------

def recover_scale_from_wrist_depth(
    gt_c2w, gs_depth, depth_conf,
    mano, ctx, frame_indices,
    src_image_size=1408,
    pinhole_calib=None,
):
    """Phase-2 scale recovery from NeoVerse's depth map at the hand pixels.

    NeoVerse's predicted intrinsics describe a ~70° pinhole FOV, but the MP4
    is an Aria 180° fisheye. So we cannot project GT 3D points through the
    pinhole intrinsics — they'd land outside the 336×336 image. Instead we
    project through the Aria fisheye calibration (the same code path used by
    scripts/hand_vis_utils.project_vertices), apply the 90° image rotation,
    subtract the center-crop offset, and look up depth at that pixel.

    We don't restrict to the wrist either — many wrists are cropped out by
    the central 70°. Iterate over all 16 MANO joints per hand and keep
    every one that lands in-frame with finite depth. Each in-frame joint
    gives one measurement of s = depth_pred / depth_gt_metric.

    Returns:
        s (float), diagnostics dict.
    """
    n_video = ctx["n_video"]
    hand_ts_sorted = ctx["hand_ts_sorted"]
    hand_poses = ctx["hand_poses"]
    # When the image fed to NeoVerse was undistorted, project the GT hand
    # through the SAME pinhole calibration we used for undistortion. Else,
    # use the original Aria fisheye projection.
    projection_calib = pinhole_calib if pinhole_calib is not None else ctx["cam_calib"]

    H, W = gs_depth.shape[2], gs_depth.shape[3]
    # diffsynth.utils.auxiliary.center_crop first resizes the image to cover
    # the target resolution, then crops. For square Aria frames going to a
    # square target, the resize produces an image already at target size and
    # the crop is a no-op — i.e. the whole 1408x1408 FOV is rescaled to
    # 336x336, NOT center-cropped. So map sensor pixels by scaling, not by
    # subtracting an offset. The same scaling holds for undistorted frames
    # since we resize them the same way.
    pixel_scale = W / float(src_image_size)

    samples = []  # (frame, side, joint_idx, gt_z, pred_depth, conf, s_i)
    debug_attempts = []
    in_frame_per_frame = {}

    for k, frame_i in enumerate(frame_indices):
        query_tc = _frame_to_timecode(frame_i, n_video, hand_ts_sorted)
        hand_data = hand_poses[find_closest(hand_ts_sorted, query_tc)]
        w2c_gt = np.linalg.inv(gt_c2w[k])
        in_frame_per_frame.setdefault(k, 0)

        for hand_key, side in [("0", "left"), ("1", "right")]:
            if hand_key not in hand_data:
                continue
            joints_world = mano.get_joints(hand_data[hand_key],
                                           is_right=(side == "right"),
                                           return_tensor=False)
            joints_world = np.asarray(joints_world)
            if joints_world.ndim == 3:
                joints_world = joints_world[0]

            for j, joint_world in enumerate(joints_world):
                joint_sensor = (w2c_gt @ np.append(joint_world, 1.0))[:3]
                if joint_sensor[2] <= 0.05:
                    continue

                # Project through the right camera model (fisheye or pinhole).
                # Both return sensor-orientation (col, row).
                p = projection_calib.project(joint_sensor)
                if p is None:
                    continue

                # 90° image rotation that aligns with how MP4 is stored.
                u_mp4 = (src_image_size - 1) - p[1]    # column in displayed image
                v_mp4 = p[0]                            # row in displayed image
                u_336 = u_mp4 * pixel_scale
                v_336 = v_mp4 * pixel_scale
                ui, vi = int(round(u_336)), int(round(v_336))
                if not (0 <= ui < W and 0 <= vi < H):
                    if j == 0:  # log once-per-hand for wrist
                        debug_attempts.append({
                            "frame": k, "side": side, "joint": j,
                            "z": float(joint_sensor[2]),
                            "u": float(u_336), "v": float(v_336),
                            "reason": "out_of_crop",
                        })
                    continue

                d = float(gs_depth[0, k, vi, ui, 0])
                c = float(depth_conf[0, k, vi, ui])
                if not np.isfinite(d) or d <= 0 or not np.isfinite(c):
                    continue
                samples.append((k, side, j, float(joint_sensor[2]), d, c,
                                d / float(joint_sensor[2])))
                in_frame_per_frame[k] += 1
                if j == 0:
                    debug_attempts.append({
                        "frame": k, "side": side, "joint": j,
                        "z": float(joint_sensor[2]),
                        "u": float(u_336), "v": float(v_336),
                        "reason": "ok",
                    })

    if not samples:
        return None, {"sample_count": 0, "debug_attempts": debug_attempts}

    ratios = np.array([row[6] for row in samples])
    confs = np.array([row[5] for row in samples])
    q1, q3 = np.percentile(ratios, [25, 75])
    iqr = q3 - q1
    inlier_mask = (ratios >= q1 - 1.5 * iqr) & (ratios <= q3 + 1.5 * iqr)
    inlier_ratios = ratios[inlier_mask]
    inlier_confs = confs[inlier_mask]
    if inlier_confs.sum() > 0:
        s = float((inlier_ratios * inlier_confs).sum() / inlier_confs.sum())
    else:
        s = float(np.median(inlier_ratios))

    return s, {
        "sample_count": len(samples),
        "inlier_count": int(inlier_mask.sum()),
        "scale_median": float(np.median(ratios)),
        "scale_mean_inliers": float(inlier_ratios.mean()),
        "scale_conf_weighted_inliers": s,
        "scale_min": float(np.min(ratios)),
        "scale_max": float(np.max(ratios)),
        "scale_iqr": float(iqr),
        "in_frame_joints_per_frame": dict(in_frame_per_frame),
        "per_frame_samples": [
            {"frame_offset": k, "side": side, "joint": j, "gt_depth_m": z,
             "pred_depth": d, "conf": c, "scale": r}
            for (k, side, j, z, d, c, r) in samples
        ],
        "debug_attempts": debug_attempts,
    }


def recover_scale_from_camera_pairs(c2w_gt, c2w_pred):
    """
    Per-frame transformation: only the scalar scale `s` is shared.

    For every frame i:
        T_i = C2W_pred[i] · diag(s, s, s, 1) · W2C_gt[i]

    Per-frame (R, t) come directly from the camera-pose pairs (exact, no
    averaging), so we don't lose precision when the model's predicted frame
    drifts across the clip. Only `s` is genuinely ambiguous, and we recover
    it from the trajectory positions:

        || p_pred[i] − ( s · R_i · p_gt[i] + t_i ) ||²  =  0  by construction,
    so positions alone don't pin down `s`. Instead we constrain `s` using the
    GT-to-pred translation magnitude ratio across pose pairs:

        For each pair, transforming GT-camera-frame origin (which is p_gt) to
        pred coords gives  C2W_pred[i] · (s · W2C_gt[i] · p_gt[i])
                         = C2W_pred[i] · (s · 0)  + C2W_pred[i] · (s · 0)
        i.e. the camera origin doesn't constrain `s`. We need an off-origin
        reference point — the simplest is the cross-frame displacement:

            p_pred[j] - p_pred[i]  ≈  s · (R_pred[j] · W2C_gt[i]) ·
                                       (p_gt[j] - p_gt[i]) · ...

    In practice the cleanest scalar `s` is the median ratio of pairwise
    inter-camera-center distances:

        s ≈ median_ij  || p_pred[i] − p_pred[j] ||  /  || p_gt[i] − p_gt[j] ||

    This is scale-only and rotation/translation-invariant — exactly what we
    need.

    Args:
        c2w_gt   : [N, 4, 4] camera-to-world in Hot3D world frame, metric.
        c2w_pred : [N, 4, 4] camera-to-world in NeoVerse normalized frame.

    Returns:
        s (float), diagnostics dict.
    """
    p_gt = c2w_gt[:, :3, 3]      # [N, 3]
    p_pred = c2w_pred[:, :3, 3]  # [N, 3]

    diff_gt = p_gt[:, None, :] - p_gt[None, :, :]                   # [N, N, 3]
    diff_pred = p_pred[:, None, :] - p_pred[None, :, :]
    d_gt = np.linalg.norm(diff_gt, axis=-1)                          # [N, N]
    d_pred = np.linalg.norm(diff_pred, axis=-1)

    # Use only pairs with non-trivial baseline in GT — short baselines are
    # dominated by trajectory + model noise.
    N = p_gt.shape[0]
    gt_span = float(np.linalg.norm(p_gt.max(0) - p_gt.min(0)))
    threshold = 0.05 * gt_span
    iu, ju = np.triu_indices(N, k=1)
    mask = d_gt[iu, ju] > threshold
    if mask.sum() < 5:
        # Fallback: use all pairs.
        mask = np.ones_like(mask, dtype=bool)
    ratios = d_pred[iu[mask], ju[mask]] / d_gt[iu[mask], ju[mask]]
    s = float(np.median(ratios))

    # Quality diagnostic: how consistent are the per-pair scale estimates?
    # If they vary wildly, no single scale fits.
    s_std = float(np.std(ratios))
    s_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    return s, {
        "scale_pair_count": int(mask.sum()),
        "scale_median": s,
        "scale_std": s_std,
        "scale_iqr": s_iqr,
        "scale_min": float(np.min(ratios)),
        "scale_max": float(np.max(ratios)),
    }


# Rotation from Aria sensor 3D camera frame to MP4-display 3D camera frame.
# project_vertices() in scripts/hand_vis_utils.py applies a 90° image rotation
# (`u = (W-1) - p[1]; v = p[0]`) to align with how the MP4 is stored. The 3D
# analog is a 90° rotation about the camera's Z-axis (forward).
#
# Sensor frame:  x_s right (sensor col), y_s down (sensor row), z_s forward.
# Display frame: x_d right (mp4 col),    y_d down (mp4 row),    z_d forward.
# Verified by projecting a sensor-frame point (1, 0, 1) through the documented
# 2D mapping: it lands at display-frame (0, 1, 1), matching this rotation.
R_SENSOR_TO_DISPLAY = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
])


def per_frame_decompose_vertices(verts_world, frame_idx, c2w_gt, c2w_pred):
    """Decompose per-frame alignment into a scale-independent displacement
    from the predicted camera center, plus the camera center itself.

    hand_in_pred_world = pos_pred[i] + s · displacement

    where
        displacement = R_pred[i] · R_sd · (W2C_gt[i] · vertex_world)[:3]

    is scale-invariant. This lets the viewer apply scale interactively.

    Returns:
        pos_pred:    [3]    predicted camera center for this frame.
        displacement: [V, 3] scale-1.0 displacement of each vertex.
    """
    w2c_gt = np.linalg.inv(c2w_gt[frame_idx])
    c2w_p = c2w_pred[frame_idx]

    V = verts_world.shape[0]
    verts_h = np.concatenate([verts_world, np.ones((V, 1))], axis=1)
    verts_cam_sensor = (w2c_gt @ verts_h.T).T[:, :3]
    verts_cam_display = (R_SENSOR_TO_DISPLAY @ verts_cam_sensor.T).T
    # Rotate (but don't translate) into pred world frame: only the rotation
    # part of C2W_pred matters here, since the translation is added separately.
    R_pred = c2w_p[:3, :3]
    displacement = (R_pred @ verts_cam_display.T).T
    pos_pred = c2w_p[:3, 3]
    return pos_pred.astype(np.float32), displacement.astype(np.float32)


def per_frame_align_vertices(verts_world, frame_idx, c2w_gt, c2w_pred, s):
    """Apply per-frame alignment with a specific scale `s`.

    Thin wrapper around per_frame_decompose_vertices for callers that want a
    pre-scaled mesh (e.g. OBJ export).
    """
    pos_pred, displacement = per_frame_decompose_vertices(
        verts_world, frame_idx, c2w_gt, c2w_pred,
    )
    return (pos_pred + s * displacement).astype(np.float32)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()

    if args.sequence_id is None:
        seqs = sorted(
            d for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        )
        if not seqs:
            raise RuntimeError(f"No sequences in {args.data_root}")
        args.sequence_id = seqs[0]
        print(f"No --sequence_id given; using first: {args.sequence_id}")

    seq_path = os.path.join(args.data_root, args.sequence_id)
    if not os.path.isdir(seq_path):
        raise RuntimeError(f"Sequence not found: {seq_path}")

    if args.output_dir is None:
        suffix = f"_undist_f{int(args.undistort_focal)}" if args.undistort else ""
        args.output_dir = os.path.join(
            "outputs/hand_alignment",
            f"{args.sequence_id}_f{args.frame_start:04d}{suffix}",
        )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "hand_meshes"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1) Hot3D context: MANO model, camera calibration, headset trajectory,
    #    JSONL hand poses. setup_vis_context bails with None if any file is
    #    missing.
    # ------------------------------------------------------------------
    print(f"Loading Hot3D context for {args.sequence_id} ...")
    mano = MANOModel(args.mano_model_folder)
    ctx = setup_vis_context(seq_path, mano_model=mano)
    if ctx is None:
        raise RuntimeError(f"setup_vis_context failed for {seq_path}")

    n_video = ctx["n_video"]
    if args.frame_start + args.num_frames > n_video:
        raise RuntimeError(
            f"Clip [{args.frame_start}, {args.frame_start + args.num_frames}) "
            f"exceeds video length {n_video}"
        )
    frame_indices = list(range(args.frame_start, args.frame_start + args.num_frames))

    # ------------------------------------------------------------------
    # 2) Load and preprocess RGB for NeoVerse.
    # ------------------------------------------------------------------
    pinhole_calib = None
    if args.undistort:
        # Build a pinhole target camera in sensor orientation with the chosen
        # focal length; undistort each MP4 frame fisheye→pinhole; then resize
        # to NeoVerse's input resolution. Feeding NeoVerse a pinhole image
        # means its pinhole-camera assumption is no longer violated.
        from PIL import Image as PILImage
        from decord import VideoReader

        print(f"Loading {args.num_frames} undistorted frames at {args.width}x{args.height} "
              f"(focal={args.undistort_focal}) ...")
        pinhole_calib = build_pinhole_target(ctx["cam_calib"], args.undistort_focal)
        vr = VideoReader(ctx["video_path"])
        frame_indices_native = list(range(args.frame_start,
                                          args.frame_start + args.num_frames))
        pil_images = []
        for fi in frame_indices_native:
            mp4_rgb = vr[fi].asnumpy()
            und_rgb = undistort_mp4_frame(mp4_rgb, ctx["cam_calib"], pinhole_calib)
            und_pil = PILImage.fromarray(und_rgb).resize(
                (args.width, args.height), resample=PILImage.LANCZOS,
            )
            pil_images.append(und_pil)
    else:
        print(f"Loading {args.num_frames} fisheye frames at {args.width}x{args.height} ...")
        pil_images = load_video(
            ctx["video_path"],
            num_frames=args.num_frames,
            resolution=(args.width, args.height),
            resize_mode="center_crop",
            sampling="first",
            frame_offset=args.frame_start,
        )
    assert len(pil_images) == args.num_frames, (
        f"got {len(pil_images)} frames, expected {args.num_frames}"
    )

    img_tensor = torch.stack(
        [TVF.to_tensor(img)[None] for img in pil_images], dim=1
    ).to(device)

    # ------------------------------------------------------------------
    # 3) Load reconstructor and run inference.
    #
    # Bypass diffsynth.models.ModelManager — its init_weights_on_device() +
    # to_empty(device=...) path leaves the un-checkpointed hand_head params
    # with uninitialized memory, which corrupts the forward pass and yields
    # all-NaN predictions across every head. Instantiate WorldMirror directly
    # with enable_hand=False (we use GT hand meshes here, not predicted ones)
    # so no head is missing its weights.
    # ------------------------------------------------------------------
    print(f"Loading reconstructor from {args.reconstructor_path} ...")
    reconstructor = WorldMirror(enable_norm=False, enable_hand=False)
    ckpt = torch.load(args.reconstructor_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("reconstructor", ckpt))
    missing, unexpected = reconstructor.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    reconstructor.to(device).eval()

    views = {
        "img":       img_tensor,
        "is_target": torch.zeros((1, args.num_frames), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, args.num_frames), dtype=torch.bool, device=device),
        "timestamp": torch.arange(args.num_frames, dtype=torch.int64,
                                  device=device).unsqueeze(0),
    }

    print("Running NeoVerse forward pass ...")

    # Sanity-check the input first — should be float in [0, 1], no NaN.
    print(f"  input img: shape={tuple(img_tensor.shape)} dtype={img_tensor.dtype}")
    print(f"    range=[{img_tensor.min().item():.3f}, {img_tensor.max().item():.3f}]"
          f"  any_nan={torch.isnan(img_tensor).any().item()}")
    # Sanity-check the model parameters too — easy mode-check for a corrupted load.
    bad_params = [n for n, p in reconstructor.named_parameters()
                  if not torch.isfinite(p).all()]
    if bad_params:
        print(f"  WARNING: {len(bad_params)} parameters contain NaN/Inf, e.g. {bad_params[:3]}")
    else:
        print(f"  model parameters all finite ({sum(1 for _ in reconstructor.parameters())} tensors)")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = reconstructor(views, is_inference=True, use_motion=False)

    # Per-prediction NaN audit — narrows down which head is failing.
    print()
    print("Prediction NaN audit:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            n_nan = torch.isnan(value).sum().item()
            n_inf = torch.isinf(value).sum().item()
            tag = "OK" if (n_nan == 0 and n_inf == 0) else f"BAD ({n_nan} NaN, {n_inf} Inf)"
            print(f"  {key:<28s} {tuple(value.shape)} {value.dtype}  {tag}")
        else:
            print(f"  {key:<28s} (non-tensor: {type(value).__name__})")

    gaussian_list = predictions["splats"][0]
    pred_c2w = predictions["rendered_extrinsics"][0].float().cpu().numpy()
    pred_K = predictions["rendered_intrinsics"][0].float().cpu().numpy()
    gs_depth_np = predictions["gs_depth"].float().cpu().numpy()        # [1, S, H, W, 1]
    gs_depth_conf_np = predictions["gs_depth_conf"].float().cpu().numpy()  # [1, S, H, W]

    # ------------------------------------------------------------------
    # 4) GT camera poses for the same frames (Hot3D world frame, metric).
    # ------------------------------------------------------------------
    gt_c2w = gt_camera_poses(ctx, frame_indices)

    # ------------------------------------------------------------------
    # Validate predicted camera poses. NeoVerse occasionally returns NaN
    # / near-zero matrices (see the emergency fix-up in reconstruct_4dgs.py
    # around line 202). If that happens here, the similarity solve blows
    # up — print useful diagnostics and bail.
    # ------------------------------------------------------------------
    bad = []
    for i in range(args.num_frames):
        m = pred_c2w[i]
        if not np.isfinite(m).all():
            bad.append((i, "non-finite"))
        elif np.abs(m).sum() < 1e-3:
            bad.append((i, "near-zero"))
        else:
            det = float(np.linalg.det(m[:3, :3]))
            if abs(det - 1.0) > 0.1:
                bad.append((i, f"non-rotation det(R)={det:.4f}"))

    if bad:
        print()
        print("ERROR: NeoVerse returned invalid camera pose(s):")
        for i, reason in bad:
            print(f"  frame {i:>2d}: {reason}")
            print(f"    pred_c2w[{i}] =\n{pred_c2w[i]}")
        print()
        print("First and last predicted c2w for reference:")
        print(f"  pred_c2w[0] =\n{pred_c2w[0]}")
        print(f"  pred_c2w[-1] =\n{pred_c2w[-1]}")
        print()
        print("Possible causes:")
        print("  - bfloat16 numerical issues on this clip")
        print("  - too little camera motion in the 16-frame window")
        print("  - the reconstructor checkpoint is mismatched")
        print()
        print("Try:")
        print("  - a different --frame_start (skip the static intro)")
        print("  - a different --sequence_id")
        print("  - increasing --num_frames so the model sees more motion")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 5) Closed-form similarity GT → NeoVerse.
    #
    # Diagnose trajectory motion first — if either trajectory barely moves
    # across the 16 frames, scale is ill-determined (the closed form will
    # collapse toward zero) and any downstream check is meaningless.
    # Egocentric clips at the start of a recording often have zero GT motion
    # because the user is putting on the headset.
    # ------------------------------------------------------------------
    gt_centers = gt_c2w[:, :3, 3]
    pred_centers = pred_c2w[:, :3, 3]
    gt_span = float(np.linalg.norm(gt_centers.max(0) - gt_centers.min(0)))
    pred_span = float(np.linalg.norm(pred_centers.max(0) - pred_centers.min(0)))
    print(f"Camera-trajectory motion (norm of XYZ range):")
    print(f"  GT:   {gt_span:.4f} m")
    print(f"  Pred: {pred_span:.6f} (normalized units)")
    if gt_span < 0.05:
        print()
        print(f"ERROR: GT trajectory motion is {gt_span:.4f} m — too small to "
              "constrain scale. Try a later --frame_start or a clip with "
              "more camera motion.")
        raise SystemExit(2)
    if pred_span < 1e-3:
        print()
        print(f"ERROR: predicted trajectory motion is {pred_span:.6f} — model "
              "isn't predicting meaningful camera motion. Try a different clip.")
        raise SystemExit(2)

    s_traj, traj_diag = recover_scale_from_camera_pairs(gt_c2w, pred_c2w)
    s_depth, depth_diag = recover_scale_from_wrist_depth(
        gt_c2w, gs_depth_np, gs_depth_conf_np, mano, ctx, frame_indices,
        pinhole_calib=pinhole_calib,
    )

    # Per-frame in-frame joint counts — quick "how many usable samples did
    # we get?" health-check.
    if "in_frame_joints_per_frame" in depth_diag:
        counts = depth_diag["in_frame_joints_per_frame"]
        print()
        print(f"In-frame MANO joints per frame (out of 32 = 16×2 hands):")
        for k in sorted(counts):
            print(f"  f{k:>2d}: {counts[k]}")

    print()
    print("Scale estimates:")
    print(f"  trajectory (camera-pair distances): s = {s_traj:.6f}  "
          f"(IQR={traj_diag['scale_iqr']:.4f} over {traj_diag['scale_pair_count']} pairs)")
    if s_depth is not None:
        print(f"  depth-map  (wrist depth at pixel): s = {s_depth:.6f}  "
              f"(IQR={depth_diag['scale_iqr']:.4f} over {depth_diag['inlier_count']} / "
              f"{depth_diag['sample_count']} wrist samples)")
        s = s_depth
        chosen = "depth-map"
    else:
        print("  depth-map: no usable wrist samples (all out-of-frame or behind camera)")
        s = s_traj
        chosen = "trajectory"
    print(f"  -> using {chosen} scale s = {s:.6f}")
    print()
    print("Per-frame (R, t) come from camera-pose pairs (exact, no averaging).")
    print()
    scale_diag = {"chosen_method": chosen,
                  "trajectory": traj_diag,
                  "depth_map": depth_diag if s_depth is not None else None}

    # ------------------------------------------------------------------
    # 6) Recover GT hand meshes in world frame; warp per-frame into
    #    NeoVerse frame using each frame's own camera-pose pair + shared s.
    # ------------------------------------------------------------------
    print("Recovering GT hand meshes ...")
    hand_ts_sorted = ctx["hand_ts_sorted"]
    hand_poses = ctx["hand_poses"]
    hand_meshes = []

    for k, frame_i in enumerate(frame_indices):
        query_tc = _frame_to_timecode(frame_i, n_video, hand_ts_sorted)
        hand_data = hand_poses[find_closest(hand_ts_sorted, query_tc)]
        for hand_key, side in [("0", "left"), ("1", "right")]:
            if hand_key not in hand_data:
                continue
            try:
                verts_world, faces = mano.get_mesh(
                    hand_data[hand_key], is_right=(side == "right"),
                )
            except Exception as e:
                print(f"  frame {frame_i} {side}: MANO mesh failed: {e}")
                continue
            pos_pred_frame, displacement = per_frame_decompose_vertices(
                verts_world, frame_idx=k, c2w_gt=gt_c2w, c2w_pred=pred_c2w,
            )
            verts_neoverse = (pos_pred_frame + s * displacement).astype(np.float32)
            hand_meshes.append({
                "frame_offset": k,
                "frame_global": int(frame_i),
                "side": side,
                # Decomposed form so the viewer can tune scale interactively:
                # verts(s) = pos_pred + s * displacement
                "pos_pred": pos_pred_frame,
                "displacement": displacement.astype(np.float32),
                # Pre-scaled vertices at the auto-recovered s, for OBJ + back-compat.
                "vertices": verts_neoverse,
                "default_scale": float(s),
                "faces": faces.astype(np.int32),
            })
            obj_path = os.path.join(
                args.output_dir, "hand_meshes", f"frame{k:04d}_{side}.obj",
            )
            trimesh.Trimesh(vertices=verts_neoverse, faces=faces).export(obj_path)

    print(f"Recovered {len(hand_meshes)} hand mesh(es) across {args.num_frames} frames")

    # ------------------------------------------------------------------
    # 7) Save outputs.
    # ------------------------------------------------------------------
    print("Saving outputs ...")
    cam_path = save_camera_params(
        extrinsics=pred_c2w, intrinsics=pred_K, target_dir=args.output_dir,
    )

    splats_path = os.path.join(args.output_dir, "gaussians.pt")
    torch.save(
        [
            {
                "means":     gs.means.cpu(),
                "harmonics": gs.harmonics.cpu(),
                "opacities": gs.opacities.cpu(),
                "scales":    gs.scales.cpu(),
                "rotations": gs.rotations.cpu(),
                "timestamp": gs.timestamp,
            }
            for gs in gaussian_list
        ],
        splats_path,
    )

    gs0 = gaussian_list[0]
    save_gs_ply(
        path=os.path.join(args.output_dir, "gaussians_frame0000.ply"),
        means=gs0.means.float(),
        scales=gs0.scales.float(),
        rotations=gs0.rotations.float(),
        rgbs=(0.5 + SH_C0 * gs0.harmonics[..., 0, :]).clamp(0, 1).float(),
        opacities=gs0.opacities.float(),
    )

    hand_meshes_path = os.path.join(args.output_dir, "hand_meshes.pt")
    torch.save(hand_meshes, hand_meshes_path)

    alignment = {
        "alignment_mode": "per_frame_camera_pair",
        "scale": float(s),
        "sequence_id": args.sequence_id,
        "frame_start": int(args.frame_start),
        "num_frames": int(args.num_frames),
        "undistorted": bool(args.undistort),
        "undistort_focal": float(args.undistort_focal) if args.undistort else None,
        **scale_diag,
    }
    align_path = os.path.join(args.output_dir, "alignment.json")
    with open(align_path, "w") as f:
        json.dump(alignment, f, indent=2)

    # GT camera centers in pred frame, via per-frame similarity. By
    # construction this puts each GT camera center exactly on its predicted
    # counterpart — the visible offset between blue/orange clouds in the
    # viewer was a global-fit artifact and disappears under per-frame
    # alignment. We keep the GT-as-origin variant for sanity-check overlay.
    gt_centers_in_pred = np.zeros_like(pred_c2w[:, :3, 3])
    for i in range(args.num_frames):
        gt_centers_in_pred[i] = per_frame_align_vertices(
            gt_c2w[i, :3, 3][None, :], frame_idx=i,
            c2w_gt=gt_c2w, c2w_pred=pred_c2w, s=s,
        )[0]
    np.savez(
        os.path.join(args.output_dir, "camera_trajectories.npz"),
        gt_c2w=gt_c2w,
        pred_c2w=pred_c2w,
        gt_camera_centers_world=gt_c2w[:, :3, 3],
        pred_camera_centers=pred_c2w[:, :3, 3],
        gt_camera_centers_in_pred=gt_centers_in_pred,
    )

    print()
    print("Outputs:")
    for p in [splats_path, cam_path, hand_meshes_path, align_path]:
        print(f"  {p}")
    print()
    print("View with:")
    print(f"  python scripts/view_4dgs_with_hand.py --output_dir {args.output_dir}")

    if args.launch_viser:
        from scripts.view_4dgs_with_hand import run_viewer
        run_viewer(
            output_dir=args.output_dir,
            host="localhost",
            port=args.port,
            opacity_thresh=0.05,
            show_frustums=False,
            show_camera_centers=True,
        )


if __name__ == "__main__":
    main()
