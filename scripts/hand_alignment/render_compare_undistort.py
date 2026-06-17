"""
render_compare_undistort.py

2×2 comparison of the alignment pipeline run with and without fisheye→pinhole
undistortion. Inputs are two alignment-output directories produced by
align_hand_to_neoverse.py — one with --undistort (fixed focal) and one without.

Panel layout (per frame):
    ┌────────────────────────┬────────────────────────┐
    │ GT (fisheye)           │ Pred (no undist.)      │
    │ input + hand           │ NeoVerse on raw        │
    │ (fisheye proj)         │ fisheye + hand pinhole │
    ├────────────────────────┼────────────────────────┤
    │ GT (undistorted)       │ Pred (undist., fixed   │
    │ input + hand           │ focal) + hand pinhole  │
    │ (pinhole proj)         │                        │
    └────────────────────────┴────────────────────────┘

Top row = the unmodified pipeline (fisheye input, NeoVerse's predicted K).
Bottom row = the corrected pipeline (fisheye undistorted to a fixed-focal
pinhole before NeoVerse, then GT projected through the same pinhole).

Usage:
    python scripts/render_compare_undistort.py \\
        --fisheye_dir   outputs/hand_alignment/<seq>_f<start> \\
        --undistort_dir outputs/hand_alignment/<seq>_f<start>_undist_f<focal>
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve repo root (../../) so `from scripts.X import ...` works from a subdir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import cv2
from PIL import Image

from diffsynth.utils.auxiliary import load_video
from decord import VideoReader
from PIL import Image as PILImage

from scripts.hand_alignment.render_2d_overlay import (
    load_alignment_dir,
    attach_world_meshes,
    render_gaussians_panel,
    overlay_hand_via_pinhole,
    overlay_hand_via_fisheye,
    project_pinhole,
    SIDE_COLOR_BGR,
)
from scripts.hand_alignment.align_hand_to_neoverse import (
    build_pinhole_target,
    undistort_mp4_frame,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fisheye_dir", required=True,
                   help="Alignment-output dir for the fisheye pipeline.")
    p.add_argument("--undistort_dir", required=True,
                   help="Alignment-output dir for the undistorted pipeline.")
    p.add_argument("--data_root", default="/home/marian/3dv/data/hot3d_aria/raw_data")
    p.add_argument("--frames", default="every:16",
                   help="Comma-separated frame indices, 'all', or 'every:N'.")
    p.add_argument("--opacity_thresh", type=float, default=0.05)
    p.add_argument("--hand_alpha", type=float, default=0.5)
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def overlay_hand_via_chosen_pinhole(canvas_rgb, hand_meshes, ctx, frame_idx,
                                     pinhole_calib, alpha,
                                     src_image_size=1408):
    """Project the GT hand through a SPECIFIC pinhole calibration (not
    NeoVerse's K_pred) and overlay on canvas. Used for the bottom-left
    panel (undistorted input frame, projected through our undistortion
    pinhole)."""
    H, W = canvas_rgb.shape[:2]
    pixel_scale = W / float(src_image_size)

    for m in hand_meshes:
        if m["frame_offset"] != frame_idx:
            continue
        verts_world = m.get("verts_world")
        if verts_world is None:
            continue
        faces = np.asarray(m["faces"], dtype=np.int32)
        T_wd = m["T_world_device"]
        T_dc = ctx["T_device_camera"]
        T_camera_world = T_dc.inverse().to_matrix() @ T_wd.inverse().to_matrix()

        N = verts_world.shape[0]
        verts_h = np.concatenate([verts_world, np.ones((N, 1))], axis=1)
        verts_cam = (T_camera_world @ verts_h.T).T[:, :3]

        pixels = np.full((N, 2), -1.0, dtype=np.float32)
        valid = np.zeros(N, dtype=bool)
        for i in range(N):
            if verts_cam[i, 2] <= 0.01:
                continue
            p = pinhole_calib.project(verts_cam[i])
            if p is None:
                continue
            # Sensor-orientation pinhole pixel → MP4 display pixel (90° CW)
            u = ((src_image_size - 1) - p[1]) * pixel_scale
            v = p[0] * pixel_scale
            pixels[i] = (u, v)
            valid[i] = True

        if not valid.any():
            continue

        depths = verts_cam[:, 2]
        face_depths = depths[faces].mean(axis=1)
        order = np.argsort(-face_depths)

        color_bgr = SIDE_COLOR_BGR[m["side"]]
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
        overlay = canvas_bgr.copy()
        drew = False
        for fi in order:
            face = faces[fi]
            if not (valid[face[0]] and valid[face[1]] and valid[face[2]]):
                continue
            tri = pixels[face].astype(np.int32)
            if (tri < -50).any() or (tri[:, 0] > W + 50).any() or (tri[:, 1] > H + 50).any():
                continue
            cv2.fillPoly(overlay, [tri.reshape(-1, 1, 2)], color_bgr)
            drew = True
        if drew:
            blended = cv2.addWeighted(overlay, alpha, canvas_bgr, 1 - alpha, 0)
            canvas_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return canvas_rgb


def load_undistorted_input_frames(alignment, ctx, data_root, num_frames):
    """Re-derive the 336x336 undistorted frames that NeoVerse received."""
    seq_path = os.path.join(data_root, alignment["sequence_id"])
    video_path = os.path.join(seq_path, "video_main_rgb.mp4")
    focal = alignment.get("undistort_focal", 280.0)
    pinhole_calib = build_pinhole_target(ctx["cam_calib"], focal)
    vr = VideoReader(video_path)
    out = []
    for fi in range(alignment["frame_start"],
                    alignment["frame_start"] + num_frames):
        mp4 = vr[fi].asnumpy()
        und = undistort_mp4_frame(mp4, ctx["cam_calib"], pinhole_calib)
        und_pil = PILImage.fromarray(und).resize((336, 336), resample=PILImage.LANCZOS)
        out.append(np.asarray(und_pil))
    return out, pinhole_calib


def load_fisheye_input_frames(alignment, data_root, num_frames):
    seq_path = os.path.join(data_root, alignment["sequence_id"])
    video_path = os.path.join(seq_path, "video_main_rgb.mp4")
    pil = load_video(
        video_path, num_frames=num_frames, resolution=(336, 336),
        resize_mode="center_crop", sampling="first",
        frame_offset=alignment["frame_start"],
    )
    return [np.asarray(p) for p in pil]


def annotate(panel, text, frame_label=None):
    cv2.putText(panel, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    if frame_label:
        H, W = panel.shape[:2]
        cv2.putText(panel, frame_label, (W - 60, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main():
    args = parse_args()

    # --- Load both alignment runs ---
    print(f"Loading fisheye pipeline:    {args.fisheye_dir}")
    fish_splats, fish_c2w, fish_K, fish_meshes, fish_align = load_alignment_dir(args.fisheye_dir)
    print(f"Loading undistorted pipeline: {args.undistort_dir}")
    und_splats, und_c2w, und_K, und_meshes, und_align = load_alignment_dir(args.undistort_dir)

    S_fish = fish_c2w.shape[0]
    S_und = und_c2w.shape[0]
    if S_fish != S_und:
        raise RuntimeError(f"Frame counts differ ({S_fish} vs {S_und}); use the same clip.")
    S = S_fish

    # --- Shared context for both pipelines ---
    from scripts.hand_vis_utils import setup_vis_context, MANOModel
    seq_path = os.path.join(args.data_root, fish_align["sequence_id"])
    mano = MANOModel("models/MANO")
    ctx = setup_vis_context(seq_path, mano_model=mano)
    if ctx is None:
        raise RuntimeError(f"setup_vis_context failed for {seq_path}")
    attach_world_meshes(fish_meshes, ctx, fish_align, S)
    attach_world_meshes(und_meshes, ctx, und_align, S)

    # --- Pre-load input frames + pinhole calib for undistort path ---
    print("Loading fisheye input frames ...")
    fisheye_inputs = load_fisheye_input_frames(fish_align, args.data_root, S)
    print("Re-running undistortion to recover input frames ...")
    und_inputs, pinhole_calib = load_undistorted_input_frames(
        und_align, ctx, args.data_root, S,
    )

    # --- Parse frame selection ---
    if args.frames == "all":
        frame_indices = list(range(S))
    elif args.frames.startswith("every:"):
        N = int(args.frames.split(":")[1])
        frame_indices = list(range(0, S, N))
    else:
        frame_indices = [int(x) for x in args.frames.split(",")]

    out_dir = Path(args.fisheye_dir) / "compare_undistort"
    out_dir.mkdir(exist_ok=True)
    H, W = 336, 336

    focal = und_align.get("undistort_focal", 280.0)
    print(f"Rendering {len(frame_indices)} frame(s) ...")
    images = []
    for i in frame_indices:
        # --- GT-FISHEYE (top-left): fisheye input + GT hand via Aria fisheye ---
        gt_fisheye = fisheye_inputs[i].copy()
        gt_fisheye = overlay_hand_via_fisheye(gt_fisheye, fish_meshes, ctx, i,
                                              args.hand_alpha)
        gt_fisheye = annotate(gt_fisheye, "GT (fisheye)", f"f{i:03d}")

        # --- GT-UNDIST (bottom-left): undistorted input + GT hand via pinhole_calib ---
        gt_undist = und_inputs[i].copy()
        gt_undist = overlay_hand_via_chosen_pinhole(gt_undist, und_meshes, ctx, i,
                                                    pinhole_calib, args.hand_alpha)
        gt_undist = annotate(gt_undist, f"GT (undist. f={focal:.0f})", f"f{i:03d}")

        # --- PRED-NOFIX (cols 2 of both rows): NeoVerse on RAW fisheye, no fixed focal ---
        pred_nofix = render_gaussians_panel(fish_splats, fish_K[i], fish_c2w[i],
                                             i, H, W, args.opacity_thresh)
        pred_nofix = overlay_hand_via_pinhole(pred_nofix, fish_meshes, fish_K[i],
                                               fish_c2w[i], i,
                                               fish_align.get("scale", 1.0),
                                               args.hand_alpha)
        pred_nofix = annotate(pred_nofix, "Pred (no fixed focal)")

        # --- PRED-FIXED (cols 3 of both rows): NeoVerse on undistorted at fixed focal ---
        pred_fixed = render_gaussians_panel(und_splats, und_K[i], und_c2w[i],
                                             i, H, W, args.opacity_thresh)
        pred_fixed = overlay_hand_via_pinhole(pred_fixed, und_meshes, und_K[i],
                                               und_c2w[i], i,
                                               und_align.get("scale", 1.0),
                                               args.hand_alpha)
        pred_fixed = annotate(pred_fixed, f"Pred (fixed focal={focal:.0f})")

        top = np.concatenate([gt_fisheye, pred_nofix], axis=1)
        bot = np.concatenate([gt_undist,  pred_fixed], axis=1)
        combined = np.concatenate([top, bot], axis=0)

        png_path = out_dir / f"frame{i:04d}.png"
        Image.fromarray(combined).save(str(png_path))
        images.append(combined)
        if i % 16 == 0 or i == frame_indices[-1]:
            print(f"  rendered frame {i}")

    print(f"Saved {len(images)} PNG(s) to {out_dir}")

    if len(images) > 1:
        video_path = out_dir / "compare.mp4"
        h, w = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        for img in images:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
