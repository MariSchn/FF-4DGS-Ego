"""
render_2d_overlay.py

Project NeoVerse's predicted Gaussians + the warped GT MANO hand into the
predicted camera's 2D image plane, frame by frame, and save a side-by-side
panel with the original Aria input frame.

The point of this script: the 3D viewer can't disambiguate scale, since a
hand pulled twice as far from the camera along the same ray still projects
to the same pixel under a pinhole camera. The 2D rendering removes the
viewing-angle freedom you have in 3D, so you can directly check whether
the GT hand mesh lands at the same pixels as the visible hand in the
reconstructed Gaussian scene.

What you'll see per frame (left → right):
  1. Original Aria 336x336 input (with GT hand wireframe overlaid via Aria
     fisheye projection — confirms GT MANO is itself correct).
  2. Gaussian-point re-render through NeoVerse's predicted pinhole camera,
     with the GT hand mesh projected through the SAME pinhole camera.
     If our per-frame R, t alignment is correct, the hand pixels here
     should sit on top of the hand-shaped Gaussian cluster.

The hand position in panel 2 is scale-invariant (pinhole math), so this
view ONLY validates rotation/translation. For scale verification, use the
3D viewer's Hand scale slider.

Usage:
  python scripts/render_2d_overlay.py --output_dir outputs/hand_alignment/<dir>
"""

import argparse
import json
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


SH_C0 = 0.28209479177387814
HOT3D_RAW_DATA_DIR = "/home/marian/3dv/data/hot3d_aria/raw_data"
SIDE_COLOR_BGR = {  # cv2 wants BGR
    "left":  (110, 110, 220),
    "right": (220, 180, 110),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True,
                   help="A directory produced by align_hand_to_neoverse.py.")
    p.add_argument("--scale", type=float, default=None,
                   help="Hand scale. Default: the saved default_scale "
                        "(but 2D projection is scale-invariant for pinhole — see docstring).")
    p.add_argument("--scales", default=None,
                   help="Comma-separated scales to render side-by-side, e.g. "
                        "'0.4,0.8,1.85,3.0'. Overrides --scale. Outputs go to "
                        "overlay_2d/multiscale/.")
    p.add_argument("--frames", default="all",
                   help="Comma-separated frame indices, 'all', or 'every:N'.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--opacity_thresh", type=float, default=0.05,
                   help="Skip Gaussians with opacity below this when drawing.")
    p.add_argument("--hand_alpha", type=float, default=0.5)
    p.add_argument("--data_root", default=HOT3D_RAW_DATA_DIR)
    p.add_argument("--skip_input_panel", action="store_true",
                   help="Skip the left (original Aria input + fisheye-projected hand) panel.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def project_pinhole(points_3d, K, c2w):
    """Pinhole projection of [N, 3] world points to [N, 2] pixels.

    Returns pixels [N, 2], depths [N], in-front-of-camera mask [N].
    """
    c2w_inv = np.linalg.inv(c2w)
    N = points_3d.shape[0]
    pts_h = np.concatenate([points_3d, np.ones((N, 1), dtype=np.float64)], axis=1)
    pts_cam = (c2w_inv @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    in_front = z > 0.01
    safe_z = np.where(in_front, z, 1.0)
    u = K[0, 0] * pts_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / safe_z + K[1, 2]
    return np.stack([u, v], axis=1), z, in_front


def render_gaussians_panel(splats, K, c2w, frame_idx, H, W, opacity_thresh):
    """Render Gaussian centers as colored 1-px dots with a per-pixel z-buffer."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Collect all points (static + dynamic for this frame).
    all_means, all_colors = [], []
    for gs in splats:
        ts = gs["timestamp"]
        if ts != -1 and ts != frame_idx:
            continue
        opac = gs["opacities"].cpu().numpy()
        mask = opac > opacity_thresh
        if not mask.any():
            continue
        means = gs["means"].cpu().numpy()[mask]
        harmonics = gs["harmonics"].cpu().numpy()[mask]
        rgb = np.clip(0.5 + SH_C0 * harmonics[:, 0, :3], 0, 1) * 255
        all_means.append(means)
        all_colors.append(rgb)

    if not all_means:
        return canvas

    means = np.concatenate(all_means, axis=0)
    rgbs = np.concatenate(all_colors, axis=0).astype(np.uint8)

    pixels, z, in_front = project_pinhole(means, K, c2w)
    ui = pixels[:, 0].round().astype(np.int64)
    vi = pixels[:, 1].round().astype(np.int64)
    in_frame = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not in_frame.any():
        return canvas

    ui = ui[in_frame]
    vi = vi[in_frame]
    z = z[in_frame]
    rgbs = rgbs[in_frame]

    # Per-pixel z-buffer via lexsort: sort by (flat_pixel asc, z asc), then keep
    # the FIRST occurrence of each unique pixel — that's the closest point.
    flat_idx = vi * W + ui
    order = np.lexsort((z, flat_idx))
    flat_sorted = flat_idx[order]
    rgb_sorted = rgbs[order]
    uniq, first_idx = np.unique(flat_sorted, return_index=True)

    canvas_flat = canvas.reshape(-1, 3)
    canvas_flat[uniq] = rgb_sorted[first_idx]
    # OpenCV/cv2 uses BGR for everything else in this script; the Gaussian dots
    # are stored RGB above. We'll convert once at the end of compose.
    return canvas


def overlay_hand_via_pinhole(canvas_rgb, hand_meshes, K, c2w, frame_idx,
                              scale, alpha):
    """Overlay GT hand mesh on top of the Gaussian re-render, projected
    through NeoVerse's predicted pinhole intrinsics K_pred.

    The mesh lands at the pinhole projection of the GT 3D hand position,
    which differs from the visible hand in the Gaussian render (the
    visible content is at fisheye pixels). The offset grows with off-axis
    angle and demonstrates the fisheye-vs-pinhole mismatch.
    """
    H, W = canvas_rgb.shape[:2]
    for m in hand_meshes:
        if m["frame_offset"] != frame_idx:
            continue
        if "pos_pred" in m and "displacement" in m:
            verts = (np.asarray(m["pos_pred"], dtype=np.float64)
                     + scale * np.asarray(m["displacement"], dtype=np.float64))
        else:
            verts = np.asarray(m["vertices"], dtype=np.float64)
        faces = np.asarray(m["faces"], dtype=np.int32)

        pixels, z, in_front = project_pinhole(verts, K, c2w)
        face_in = in_front[faces].all(axis=1)
        if not face_in.any():
            continue
        face_z = z[faces].mean(axis=1)

        order = np.argsort(-face_z)
        color_bgr = SIDE_COLOR_BGR[m["side"]]
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
        overlay = canvas_bgr.copy()
        any_drawn = False
        for fi in order:
            if not face_in[fi]:
                continue
            tri = pixels[faces[fi]].astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [tri], color_bgr)
            any_drawn = True
        if any_drawn:
            blended = cv2.addWeighted(overlay, alpha, canvas_bgr, 1 - alpha, 0)
            canvas_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return canvas_rgb


def overlay_hand_via_fisheye(canvas_rgb, hand_meshes, ctx, frame_idx,
                              alpha, src_image_size=1408):
    """Overlay GT hand mesh on top of the Gaussian re-render, projected
    through Aria fisheye (NOT NeoVerse's pinhole intrinsics).

    Why: the Gaussian re-render through pinhole K_pred recovers the fisheye
    pixel where the input image content lives (NeoVerse pinhole-back-projects
    each fisheye pixel to 3D, and re-projecting through the same K returns
    the same pixel). So content in the Gaussian render is at fisheye pixel
    coordinates. To make the GT hand mesh sit on top of the visible hand
    in the Gaussian render, project the same way — through Aria fisheye.

    Projecting through pinhole instead introduces a fisheye-vs-pinhole bias
    that grows with off-axis angle, pulling the hand mesh further from
    center than the visible hand actually sits.
    """
    H, W = canvas_rgb.shape[:2]
    cam_calib = ctx["cam_calib"]
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
            p = cam_calib.project(verts_cam[i])
            if p is None:
                continue
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


# -----------------------------------------------------------------------------
# Input-image panel (original Aria frame with GT hand projected via fisheye)
# -----------------------------------------------------------------------------

def load_input_frames(alignment, data_root, num_frames):
    """Re-derive the exact 336x336 input frames NeoVerse received."""
    seq_path = os.path.join(data_root, alignment["sequence_id"])
    video_path = os.path.join(seq_path, "video_main_rgb.mp4")
    pil_images = load_video(
        video_path,
        num_frames=num_frames,
        resolution=(336, 336),
        resize_mode="center_crop",
        sampling="first",
        frame_offset=alignment["frame_start"],
    )
    return [np.asarray(img) for img in pil_images]  # RGB uint8


def overlay_hand_on_input(input_rgb, hand_meshes, frame_idx, ctx,
                          src_image_size=1408):
    """Project GT hand mesh through Aria fisheye and draw it on the input frame,
    so we can confirm the GT MANO mesh itself is consistent with the image.

    Uses the same projection logic as scripts/hand_vis_utils.project_vertices.
    """
    out = input_rgb.copy()
    cam_calib = ctx["cam_calib"]
    H, W = out.shape[:2]
    # See note in align_hand_to_neoverse.recover_scale_from_wrist_depth:
    # load_video's center_crop is actually a resize for square frames, so we
    # map sensor pixels to displayed pixels by scaling, not by subtracting
    # an offset.
    pixel_scale = W / float(src_image_size)

    # We need the hand mesh in WORLD frame for project_vertices. The saved
    # hand_meshes are in pred-NeoVerse coords, so we re-derive vertices in
    # GT-world from the JSONL on the fly via setup_vis_context — see main.
    # We'll be passed the world-frame meshes directly via the dict below.
    # (See main: we attach 'verts_world' to each mesh entry.)
    for m in hand_meshes:
        if m["frame_offset"] != frame_idx:
            continue
        verts_world = m.get("verts_world")
        if verts_world is None:
            continue
        faces = np.asarray(m["faces"])
        # Project world → sensor (via T_camera_world from frame's T_world_device).
        T_wd = m["T_world_device"]
        T_dc = ctx["T_device_camera"]
        T_camera_world = T_dc.inverse().to_matrix() @ T_wd.inverse().to_matrix()
        N = verts_world.shape[0]
        verts_h = np.concatenate([verts_world, np.ones((N, 1))], axis=1)
        verts_cam = (T_camera_world @ verts_h.T).T[:, :3]

        # Aria fisheye projection + 90° MP4 rotation + center-crop offset.
        pixels = np.full((N, 2), -1.0, dtype=np.float32)
        valid = np.zeros(N, dtype=bool)
        for i in range(N):
            if verts_cam[i, 2] <= 0.01:
                continue
            p = cam_calib.project(verts_cam[i])
            if p is None:
                continue
            u = ((src_image_size - 1) - p[1]) * pixel_scale
            v = p[0] * pixel_scale
            pixels[i] = (u, v)
            valid[i] = True

        # Draw filled triangles where all three vertices are valid + on-screen.
        color_bgr = SIDE_COLOR_BGR[m["side"]]
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        overlay = out_bgr.copy()
        drew = False
        # depth for ordering
        depths = verts_cam[:, 2]
        face_depths = depths[faces].mean(axis=1)
        order = np.argsort(-face_depths)
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
            blended = cv2.addWeighted(overlay, 0.45, out_bgr, 0.55, 0)
            out = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return out


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_alignment_dir(output_dir):
    out_dir = Path(output_dir)
    splats = torch.load(out_dir / "gaussians.pt", weights_only=False)
    with open(out_dir / "camera_params.json") as f:
        cam_data = json.load(f)
    c2w = np.array([e["matrix"] for e in cam_data["extrinsics"]], dtype=np.float64)
    K = np.array([e["matrix"] for e in cam_data["intrinsics"]], dtype=np.float64)
    hand_meshes = torch.load(out_dir / "hand_meshes.pt", weights_only=False)
    with open(out_dir / "alignment.json") as f:
        alignment = json.load(f)
    return splats, c2w, K, hand_meshes, alignment


def attach_world_meshes(hand_meshes, ctx, alignment, num_frames):
    """For the input-image panel we need the hand mesh in GT WORLD coords (so we
    can project it through Aria fisheye). Re-derive from the JSONL params.

    Also attach the per-frame T_world_device so the projection has all it needs.
    """
    from scripts.hand_vis_utils import _frame_to_timecode, find_closest, MANOModel
    from projectaria_tools.core.sophus import SE3
    mano = ctx["mano_model"]

    n_video = ctx["n_video"]
    hand_ts = ctx["hand_ts_sorted"]
    headset_ts = ctx["headset_ts_sorted"]
    headset_poses = ctx["headset_poses"]
    hand_poses = ctx["hand_poses"]

    frame_indices = list(range(alignment["frame_start"],
                               alignment["frame_start"] + num_frames))
    per_frame_T_wd = {}
    for k, fi in enumerate(frame_indices):
        query_tc = _frame_to_timecode(fi, n_video, hand_ts)
        h_ts = find_closest(headset_ts, query_tc)
        t_wd, q_wd = headset_poses[h_ts]
        per_frame_T_wd[k] = SE3.from_quat_and_translation(
            q_wd[0], q_wd[1:], t_wd
        )[0]

    for m in hand_meshes:
        fi = alignment["frame_start"] + m["frame_offset"]
        query_tc = _frame_to_timecode(fi, n_video, hand_ts)
        hand_data = hand_poses[find_closest(hand_ts, query_tc)]
        side = m["side"]
        hand_key = "0" if side == "left" else "1"
        if hand_key not in hand_data:
            m["verts_world"] = None
            m["T_world_device"] = per_frame_T_wd[m["frame_offset"]]
            continue
        verts_world, _faces = mano.get_mesh(
            hand_data[hand_key], is_right=(side == "right"),
        )
        m["verts_world"] = verts_world
        m["T_world_device"] = per_frame_T_wd[m["frame_offset"]]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    splats, c2w, K, hand_meshes, alignment = load_alignment_dir(args.output_dir)

    S = c2w.shape[0]
    if args.scales is not None:
        scales = [float(x.strip()) for x in args.scales.split(",")]
    elif args.scale is not None:
        scales = [float(args.scale)]
    else:
        scales = [float(alignment.get("scale", 1.0))]
    print(f"Rendering at scale(s): {scales}")

    if args.frames == "all":
        frame_indices = list(range(S))
    elif args.frames.startswith("every:"):
        N = int(args.frames.split(":")[1])
        frame_indices = list(range(0, S, N))
    else:
        frame_indices = [int(x) for x in args.frames.split(",")]

    H, W = 336, 336

    # We always need ctx (camera calibration + headset trajectory + MANO) because
    # the right panel now projects the GT hand through Aria fisheye too.
    from scripts.hand_vis_utils import setup_vis_context, MANOModel
    seq_path = os.path.join(args.data_root, alignment["sequence_id"])
    mano = MANOModel("models/MANO")
    ctx = setup_vis_context(seq_path, mano_model=mano)
    if ctx is None:
        raise RuntimeError(f"setup_vis_context failed for {seq_path}")
    attach_world_meshes(hand_meshes, ctx, alignment, S)

    # Optionally load the original Aria frames for the left reference panel.
    input_frames = None
    if not args.skip_input_panel:
        try:
            input_frames = load_input_frames(alignment, args.data_root, S)
        except Exception as e:
            print(f"WARN: couldn't load input frames ({e}) — skipping left panel.")
            input_frames = None

    out_dir = Path(args.output_dir)
    overlay_dir = out_dir / "overlay_2d"
    overlay_dir.mkdir(exist_ok=True)
    multiscale = len(scales) > 1
    if multiscale:
        ms_dir = overlay_dir / "multiscale"
        ms_dir.mkdir(exist_ok=True)

    print(f"Rendering {len(frame_indices)} frame(s) × {len(scales)} scale(s) ...")

    # Cache per-frame left panels (Aria fisheye) and base Gaussian re-renders
    # (scale-invariant) so multi-scale rendering doesn't redo the same work.
    base_gaussian_panels = {}
    input_panels = {}

    images_by_scale = {s: [] for s in scales}
    for i in frame_indices:
        if i not in base_gaussian_panels:
            base_gaussian_panels[i] = render_gaussians_panel(
                splats, K[i], c2w[i], i, H, W, args.opacity_thresh,
            )
        if input_frames is not None and i not in input_panels:
            panel = overlay_hand_on_input(input_frames[i], hand_meshes, i, ctx)
            cv2.putText(panel, "Aria input + hand (fisheye)", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"f{i:03d}", (W - 60, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            input_panels[i] = panel

        # Three panels (left → right):
        #   1. Aria input + hand projected through fisheye  (reference: GT MANO + image)
        #   2. Gaussian re-render + hand through pinhole    (shows fisheye-pinhole bias)
        #   3. Gaussian re-render + hand through fisheye    (hand lands on visible content)
        for s in scales:
            pinhole_panel = overlay_hand_via_pinhole(
                base_gaussian_panels[i].copy(), hand_meshes, K[i], c2w[i],
                i, s, args.hand_alpha,
            )
            cv2.putText(pinhole_panel, f"Gaussians + hand (pinhole)  s={s:.3f}",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

            fisheye_panel = overlay_hand_via_fisheye(
                base_gaussian_panels[i].copy(), hand_meshes, ctx, i,
                args.hand_alpha,
            )
            cv2.putText(fisheye_panel, "Gaussians + hand (fisheye)",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

            if input_frames is not None:
                combined = np.concatenate(
                    [input_panels[i], pinhole_panel, fisheye_panel], axis=1,
                )
            else:
                cv2.putText(pinhole_panel, f"f{i:03d}", (W - 60, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                combined = np.concatenate([pinhole_panel, fisheye_panel], axis=1)

            target_dir = ms_dir if multiscale else overlay_dir
            png_path = target_dir / f"frame{i:04d}_s{s:.3f}.png"
            Image.fromarray(combined).save(str(png_path))
            images_by_scale[s].append(combined)

        if i % 16 == 0 or i == frame_indices[-1]:
            print(f"  rendered frame {i}")

    total_pngs = sum(len(v) for v in images_by_scale.values())
    target_dir = ms_dir if multiscale else overlay_dir
    print(f"Saved {total_pngs} PNG(s) to {target_dir}")

    for s, imgs in images_by_scale.items():
        if len(imgs) <= 1:
            continue
        video_path = target_dir / f"overlay_s{s:.3f}.mp4"
        h, w = imgs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        for img in imgs:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
