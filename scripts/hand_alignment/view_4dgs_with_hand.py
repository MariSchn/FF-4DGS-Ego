"""
Interactive 4DGS + MANO-hand viewer.

Extends scripts/view_4dgs.py: in addition to the predicted Gaussian splats, it
loads a hand_meshes.pt file produced by scripts/align_hand_to_neoverse.py and
renders the warped GT MANO meshes in the same scene, bound to the frame
slider so the hand only appears on the frame it was recovered from.

Usage:
  python scripts/view_4dgs_with_hand.py --output_dir outputs/hand_alignment/<sequence>_f<start>
  python scripts/view_4dgs_with_hand.py --output_dir <dir> --show_camera_centers
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Resolve repo root (../../) so `from scripts.X import ...` works from a subdir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import viser
import viser.transforms as vtf

from scripts.view_4dgs import (
    load_gaussians,
    load_cameras,
    splats_to_viser,
)


# Same world-frame fix-up as view_4dgs.splats_to_viser:
#   Reconstruction world: X=right, Y=forward, Z=up
#   Viser world:          X=right, Y=up,      Z=backward
# Rotate -90° around X  →  (x, y, z) → (x, z, -y).
VISER_FIXUP_R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hand mesh loading + frame fix-up
# ---------------------------------------------------------------------------

def load_hand_meshes(pt_path):
    """Load hand_meshes.pt produced by align_hand_to_neoverse.py.

    Returns (entries, default_scale) where each entry has the decomposed form
    pos_pred + displacement (both already rotated into viser's world frame):

        vertices(s) = pos_pred + s * displacement

    For old hand_meshes.pt files that only have pre-scaled `vertices`, we fall
    back to a fixed-scale entry (scale slider will be disabled).
    """
    raw = torch.load(pt_path, weights_only=False)
    entries = []
    default_scale = 1.0
    for m in raw:
        entry = {
            "frame_offset": int(m["frame_offset"]),
            "frame_global": int(m.get("frame_global", m["frame_offset"])),
            "side": str(m["side"]),
            "faces": np.asarray(m["faces"], dtype=np.int32),
        }
        if "pos_pred" in m and "displacement" in m:
            entry["pos_pred"] = (
                np.asarray(m["pos_pred"], dtype=np.float32) @ VISER_FIXUP_R.T
            )
            entry["displacement"] = (
                np.asarray(m["displacement"], dtype=np.float32) @ VISER_FIXUP_R.T
            )
            entry["decomposed"] = True
        else:
            entry["vertices"] = (
                np.asarray(m["vertices"], dtype=np.float32) @ VISER_FIXUP_R.T
            )
            entry["decomposed"] = False
        if "default_scale" in m:
            default_scale = float(m["default_scale"])
        entries.append(entry)
    return entries, default_scale


def vertices_at_scale(entry, scale):
    """Compute vertices in viser world frame at the given scale.

    For decomposed entries: vertices = pos_pred + scale * displacement.
    For legacy fixed entries: scale is ignored.
    """
    if entry["decomposed"]:
        return entry["pos_pred"] + scale * entry["displacement"]
    return entry["vertices"]


def load_camera_trajectories(npz_path):
    """Load camera_trajectories.npz produced by align_hand_to_neoverse.py.

    Returns (gt_centers_aligned, pred_centers) in viser world frame.
    """
    data = np.load(npz_path)
    gt_aligned = np.asarray(data["gt_camera_centers_in_pred"], dtype=np.float32)
    pred = np.asarray(data["pred_camera_centers"], dtype=np.float32)
    return gt_aligned @ VISER_FIXUP_R.T, pred @ VISER_FIXUP_R.T


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def run_viewer(output_dir, host="localhost", port=8080,
               opacity_thresh=0.05, show_frustums=False,
               show_camera_centers=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir)

    print("Loading Gaussians ...")
    gaussian_list = load_gaussians(out_dir / "gaussians.pt", device)

    print("Loading camera parameters ...")
    cam2world, intrinsics = load_cameras(out_dir / "camera_params.json")
    S = cam2world.shape[0]

    hand_meshes_path = out_dir / "hand_meshes.pt"
    if hand_meshes_path.exists():
        hand_meshes, default_scale = load_hand_meshes(hand_meshes_path)
    else:
        hand_meshes, default_scale = [], 1.0
    decomposable = bool(hand_meshes) and all(m["decomposed"] for m in hand_meshes)
    if hand_meshes:
        print(f"Loaded {len(hand_meshes)} hand mesh(es) "
              f"across frames {min(m['frame_offset'] for m in hand_meshes)} "
              f"to {max(m['frame_offset'] for m in hand_meshes)}  "
              f"(default scale = {default_scale:.4f}, scale slider "
              f"{'enabled' if decomposable else 'disabled — legacy format'})")
    else:
        print(f"No hand_meshes.pt found in {out_dir} — running splats-only.")

    cam_traj_path = out_dir / "camera_trajectories.npz"
    if show_camera_centers and cam_traj_path.exists():
        gt_centers_aligned, pred_centers = load_camera_trajectories(cam_traj_path)
    else:
        gt_centers_aligned = pred_centers = None

    dynamic_timestamps = sorted(
        set(gs.timestamp for gs in gaussian_list if gs.timestamp >= 0)
    )
    num_frames = len(dynamic_timestamps) if dynamic_timestamps else 1

    # ------------------------------------------------------------------
    # Server + GUI
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=host, port=port)
    host_display = host if host != "0.0.0.0" else "<remote-ip>"
    print(f"\nOpen your browser at  http://{host_display}:{port}\n")

    time_slider = server.gui.add_slider(
        "Frame", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0,
    )
    opacity_slider = server.gui.add_slider(
        "Opacity threshold", min=0.0, max=0.5, step=0.01,
        initial_value=opacity_thresh,
    )
    show_hands_cb = server.gui.add_checkbox("Show hand meshes", True)
    show_all_hands_cb = server.gui.add_checkbox(
        "Show hands from all frames", False,
        hint="Off = hand visible only on its source frame.",
    )
    # Scale slider — only meaningful for decomposed entries.
    if decomposable:
        scale_slider = server.gui.add_slider(
            "Hand scale", min=0.0, max=5.0, step=0.01,
            initial_value=float(default_scale),
            hint="vertices = pos_pred + scale * displacement. "
                 "Tune until the hand mesh lines up with the visible hand "
                 "inside the Gaussian scene.",
        )
    else:
        scale_slider = None

    # ------------------------------------------------------------------
    # Camera frustums (optional)
    # ------------------------------------------------------------------
    if show_frustums:
        for i in range(S):
            R_mat = cam2world[i, :3, :3]
            tt = cam2world[i, :3, 3]
            wxyz = vtf.SO3.from_matrix(R_mat).wxyz
            fy = intrinsics[i, 1, 1]
            H = intrinsics[i, 1, 2] * 2
            server.scene.add_camera_frustum(
                f"/cameras/frame_{i:04d}",
                fov=float(2 * np.arctan2(H / 2, fy)),
                aspect=float(intrinsics[i, 0, 0] / fy),
                scale=0.05,
                wxyz=wxyz,
                position=tt,
                color=(180, 180, 255),
            )

    # ------------------------------------------------------------------
    # Camera centers (optional — aligned GT vs predicted)
    # ------------------------------------------------------------------
    if gt_centers_aligned is not None and pred_centers is not None:
        server.scene.add_point_cloud(
            "/cam_centers/pred",
            points=pred_centers,
            colors=np.tile(np.array([[80, 80, 255]], dtype=np.uint8),
                           (len(pred_centers), 1)),
            point_size=0.01,
        )
        server.scene.add_point_cloud(
            "/cam_centers/gt_aligned",
            points=gt_centers_aligned,
            colors=np.tile(np.array([[255, 200, 80]], dtype=np.uint8),
                           (len(gt_centers_aligned), 1)),
            point_size=0.01,
        )

    # ------------------------------------------------------------------
    # Pre-create one mesh handle per (frame, side) so we can toggle visibility
    # cheaply on slider updates instead of re-uploading geometry.
    # ------------------------------------------------------------------
    SIDE_COLOR = {"left": (220, 110, 110), "right": (110, 180, 220)}
    # List of mutable dicts so the scale-change handler can swap in a new
    # handle after re-adding the mesh under the same name.
    mesh_handles = []
    initial_scale = float(default_scale)
    for m in hand_meshes:
        name = f"/hands/frame{m['frame_offset']:04d}_{m['side']}"
        h = server.scene.add_mesh_simple(
            name=name,
            vertices=vertices_at_scale(m, initial_scale),
            faces=m["faces"],
            color=SIDE_COLOR[m["side"]],
            opacity=0.85,
            flat_shading=False,
        )
        h.visible = False
        mesh_handles.append({
            "name": name,
            "frame_offset": m["frame_offset"],
            "entry": m,
            "handle": h,
        })

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------
    state = {"dirty": True, "scale_dirty": False, "current_scale": initial_scale}

    @time_slider.on_update
    def _(_): state["dirty"] = True

    @opacity_slider.on_update
    def _(_): state["dirty"] = True

    @show_hands_cb.on_update
    def _(_): state["dirty"] = True

    @show_all_hands_cb.on_update
    def _(_): state["dirty"] = True

    if scale_slider is not None:
        @scale_slider.on_update
        def _(_):
            state["dirty"] = True
            state["scale_dirty"] = True

    @server.on_client_connect
    def _(client):
        state["dirty"] = True

    print("Viewer running. Press Ctrl-C to quit.")
    try:
        while True:
            if state["dirty"]:
                state["dirty"] = False
                frame_idx = int(time_slider.value)
                timestamp = dynamic_timestamps[frame_idx] if dynamic_timestamps else 0
                thresh = float(opacity_slider.value)

                splat_data = splats_to_viser(gaussian_list, timestamp, thresh)
                if splat_data is not None:
                    n = len(splat_data["centers"])
                    print(f"  Frame {frame_idx} (ts={timestamp}): "
                          f"rendering {n:,} Gaussians", end="\r")
                    server.scene.add_gaussian_splats(
                        "/splats",
                        centers=splat_data["centers"],
                        covariances=splat_data["covariances"],
                        rgbs=splat_data["rgbs"],
                        opacities=splat_data["opacities"],
                    )

                # Rebuild mesh geometry if scale changed (only for decomposable
                # entries; the legacy fixed-vertex path skips this).
                if state.get("scale_dirty") and scale_slider is not None:
                    state["scale_dirty"] = False
                    new_scale = float(scale_slider.value)
                    state["current_scale"] = new_scale
                    for mh in mesh_handles:
                        entry = mh["entry"]
                        if not entry["decomposed"]:
                            continue
                        was_visible = mh["handle"].visible
                        mh["handle"] = server.scene.add_mesh_simple(
                            name=mh["name"],
                            vertices=vertices_at_scale(entry, new_scale),
                            faces=entry["faces"],
                            color=SIDE_COLOR[entry["side"]],
                            opacity=0.85,
                            flat_shading=False,
                        )
                        mh["handle"].visible = was_visible

                # Hand visibility.
                show_hands = bool(show_hands_cb.value)
                show_all = bool(show_all_hands_cb.value)
                for mh in mesh_handles:
                    mh["handle"].visible = (
                        show_hands
                        and (show_all or mh["frame_offset"] == frame_idx)
                    )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down viewer.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True,
                   help="Directory produced by align_hand_to_neoverse.py "
                        "(or reconstruct_4dgs.py — hand meshes are optional).")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--opacity_thresh", type=float, default=0.05)
    p.add_argument("--show_frustums", action="store_true")
    p.add_argument("--show_camera_centers", action="store_true",
                   help="Show aligned-GT and predicted camera centers as point clouds.", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_viewer(
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        opacity_thresh=args.opacity_thresh,
        show_frustums=args.show_frustums,
        show_camera_centers=args.show_camera_centers,
    )


if __name__ == "__main__":
    main()
