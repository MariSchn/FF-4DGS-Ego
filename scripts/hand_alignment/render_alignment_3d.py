"""
render_alignment_3d.py

Publication-quality, fully head-less renders of the MANO-hand <-> Gaussian
alignment, for figures / videos in the final report.

Unlike scripts/view_4dgs_with_hand.py (an interactive viser viewer) this script
rasterizes the Gaussians offscreen with gsplat - the real splatting renderer -
so it runs over SSH with no browser, at arbitrary resolution, and composites the
GT MANO mesh into the same image with depth-correct occlusion and Lambert
shading.

Three render modes (--mode):

  turntable  Orbit a virtual camera around the hand and render the Gaussian
             scene with the hand mesh composited in. Outputs a looping mp4 plus
             a handful of evenly-spaced high-res hero stills. This is the figure
             that *demonstrates the alignment* - a single angle can hide a depth
             offset, an orbit cannot.

  camera     Render from the ACTUAL reconstructed camera pose(s) in
             camera_params.json, super-sampled to high resolution: a clean,
             high-quality reconstruction image from the true viewpoint (with or
             without the hand overlay).

  both       Do both (default).

Examples:
  # everything, hand at frame 64
  python scripts/hand_alignment/render_alignment_3d.py \
      --output_dir outputs/hand_alignment/P0001_10a27bf7_f1000_undist_f609 \
      --mode both --frame 64

  # just a clean high-res reconstruction of the input view, no hand
  python scripts/hand_alignment/render_alignment_3d.py \
      --output_dir <dir> --mode camera --frame 64 --no_hand --res 1600
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
import imageio.v2 as imageio
import gsplat


SH_C0 = 0.28209479177387814
SIDE_COLOR = {  # RGB
    "left":  (224, 102, 102),   # warm red
    "right": (102, 158, 224),   # cool blue
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scene(output_dir, device):
    out = Path(output_dir)

    raw = torch.load(out / "gaussians.pt", map_location=device, weights_only=False)
    # One group per timestamp; index by timestamp for O(1) frame lookup.
    groups = {}
    for d in raw:
        groups[int(d["timestamp"])] = d

    with open(out / "camera_params.json") as f:
        cam = json.load(f)
    c2w = np.array([e["matrix"] for e in cam["extrinsics"]], dtype=np.float64)
    K = np.array([e["matrix"] for e in cam["intrinsics"]], dtype=np.float64)

    hand_path = out / "hand_meshes.pt"
    hands_by_frame = {}
    default_scale = 1.0
    if hand_path.exists():
        for m in torch.load(hand_path, weights_only=False):
            hands_by_frame.setdefault(int(m["frame_offset"]), []).append(m)
            default_scale = float(m.get("default_scale", default_scale))

    al_path = out / "alignment.json"
    if al_path.exists():
        with open(al_path) as f:
            default_scale = float(json.load(f).get("scale", default_scale))

    return groups, c2w, K, hands_by_frame, default_scale


def gaussians_for_frame(groups, frame, opacity_thresh, device):
    """Return gsplat-ready tensors for one frame's Gaussians (opacity-filtered)."""
    if frame not in groups:
        avail = sorted(groups)
        raise SystemExit(f"Frame {frame} not in gaussians.pt (have {avail[0]}..{avail[-1]}).")
    g = groups[frame]
    opac = g["opacities"].to(device).float()
    mask = opac > opacity_thresh
    means = g["means"].to(device).float()[mask]
    quats = g["rotations"].to(device).float()[mask]
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = g["scales"].to(device).float()[mask]
    harmonics = g["harmonics"].to(device).float()[mask]
    rgb = (0.5 + SH_C0 * harmonics[:, 0, :3]).clamp(0.0, 1.0)
    return dict(means=means, quats=quats, scales=scales,
                opac=opac[mask], rgb=rgb)


def hand_vertices(entry, scale):
    """World-frame vertices = pos_pred + scale * displacement (reconstruction frame)."""
    if "pos_pred" in entry and "displacement" in entry:
        return (np.asarray(entry["pos_pred"], dtype=np.float64)
                + scale * np.asarray(entry["displacement"], dtype=np.float64))
    return np.asarray(entry["vertices"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Cameras (OpenCV convention: +x right, +y down, +z forward — matches the
# stored extrinsics and what gsplat expects)
# ---------------------------------------------------------------------------

WORLD_UP = np.array([0.0, 0.0, 1.0])  # reconstruction frame: Z = up


def look_at_c2w(eye, target, world_up=WORLD_UP):
    z = target - eye
    z = z / np.linalg.norm(z)
    # Egocentric views look almost straight down, which makes z nearly parallel
    # to world-up and the cross product degenerate — fall back to world-Y then.
    up = world_up if abs(float(z @ world_up)) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)            # +y down (OpenCV)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def intrinsics_for(res, vfov_deg):
    f = (res / 2.0) / np.tan(np.deg2rad(vfov_deg) / 2.0)
    return np.array([[f, 0, res / 2.0], [0, f, res / 2.0], [0, 0, 1.0]])


def rot_axis_angle(axis, ang):
    """Rodrigues rotation matrix for `ang` radians about unit-ish `axis`."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_splats(g, c2w, K, W, H, bg, device, rasterize_mode="classic"):
    """gsplat render -> (rgb float[H,W,3] in [0,1], depth float[H,W]).

    We render on a black background and alpha-composite over `bg` ourselves so
    the background colour is exact and unfilled pixels are clean.

    rasterize_mode defaults to "classic" to match the model's own renderer
    (rasterization.py). "antialiased" lowers the opacity of tiny per-pixel
    Gaussians, which makes this feed-forward splat field stop tiling and show a
    woven/screen-door texture.
    """
    viewmat = torch.inverse(torch.tensor(c2w, dtype=torch.float32, device=device))[None]
    Ks = torch.tensor(K, dtype=torch.float32, device=device)[None]
    colors, alphas, _meta = gsplat.rasterization(
        means=g["means"], quats=g["quats"], scales=g["scales"],
        opacities=g["opac"], colors=g["rgb"],
        viewmats=viewmat, Ks=Ks, width=W, height=H,
        render_mode="RGB+ED", rasterize_mode=rasterize_mode, near_plane=0.01,
    )
    rgb = colors[0, :, :, :3].clamp(0, 1).cpu().numpy()
    a = alphas[0].cpu().numpy()                       # [H, W, 1]
    bg_arr = np.array(bg, dtype=np.float32)[None, None, :]
    img = rgb * a + bg_arr * (1 - a)
    depth = colors[0, :, :, 3].cpu().numpy()
    return img.clip(0, 1), depth


def project(verts_w, c2w, K):
    w2c = np.linalg.inv(c2w)
    N = len(verts_w)
    vh = np.concatenate([verts_w, np.ones((N, 1))], axis=1)
    vc = (w2c @ vh.T).T[:, :3]
    z = vc[:, 2]
    in_front = z > 1e-3
    sz = np.where(in_front, z, 1.0)
    u = K[0, 0] * vc[:, 0] / sz + K[0, 2]
    v = K[1, 1] * vc[:, 1] / sz + K[1, 2]
    return np.stack([u, v], axis=1), z, in_front


def composite_hands(img01, splat_depth, entries, scale, c2w, K,
                    alpha, occlude):
    """Draw shaded, translucent, depth-tested hand mesh(es) over a splat image.

    Returns uint8 RGB [H, W, 3].
    """
    H, W = img01.shape[:2]
    out = (img01 * 255).astype(np.uint8).copy()
    overlay = out.copy()
    mask = np.zeros((H, W), dtype=np.float32)
    eye = c2w[:3, 3]
    cam_fwd, cam_up, cam_right = c2w[:3, 2], -c2w[:3, 1], c2w[:3, 0]
    # Directional key light from the upper-front-left (in camera space), like a
    # studio 3-point key. Gives a smooth bright-top -> dark-bottom gradient.
    light = -cam_fwd + 0.6 * cam_up - 0.35 * cam_right
    light /= np.linalg.norm(light)
    AMBIENT = 0.4

    for m in entries:
        verts = hand_vertices(m, scale)
        faces = np.asarray(m["faces"], dtype=np.int32)
        px, z, in_front = project(verts, c2w, K)

        # Smooth (Gouraud-ish) shading: average face normals into per-vertex
        # normals so the lighting varies smoothly across the mesh instead of
        # looking faceted, then shade each face by its averaged vertex normal.
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)               # area-weighted face normals
        vn = np.zeros_like(verts)
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-8)
        nf = vn[faces].mean(axis=1)
        nf /= np.linalg.norm(nf, axis=1, keepdims=True).clip(1e-8)
        # Orient normals toward the camera so lit faces are the visible ones.
        fcent = verts[faces].mean(axis=1)
        view_dir = eye[None, :] - fcent
        nf *= np.sign((nf * view_dir).sum(1))[:, None]
        diffuse = (nf @ light).clip(0, 1)
        shade = AMBIENT + (1 - AMBIENT) * diffuse
        base = np.array(SIDE_COLOR[m["side"]], dtype=np.float32)
        face_col = (base[None, :] * shade[:, None]).clip(0, 255).astype(np.uint8)

        face_z = z[faces].mean(axis=1)
        order = np.argsort(-face_z)  # painter's: far -> near
        for fi in order:
            f = faces[fi]
            if not in_front[f].all():
                continue
            tri = px[f].astype(np.int32)
            if occlude:
                cx = int(np.clip(tri[:, 0].mean(), 0, W - 1))
                cy = int(np.clip(tri[:, 1].mean(), 0, H - 1))
                sd = splat_depth[cy, cx]
                # Solid scene clearly in front of this face -> face is occluded.
                if sd > 1e-3 and sd < face_z[fi] - 0.02:
                    continue
            cv2.fillPoly(overlay, [tri.reshape(-1, 1, 2)],
                         tuple(int(c) for c in face_col[fi]))
            cv2.fillPoly(mask, [tri.reshape(-1, 1, 2)], 1.0)

    a = (alpha * mask)[:, :, None]
    return (out * (1 - a) + overlay * a).astype(np.uint8)


def resize_to(img, out_w, out_h):
    """Resize an image; area-average when shrinking, Lanczos when enlarging."""
    h, w = img.shape[:2]
    if (w, h) == (out_w, out_h):
        return img
    interp = cv2.INTER_AREA if out_w < w else cv2.INTER_LANCZOS4
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


def write_video(path, frames_rgb, fps):
    h, w = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames_rgb:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_turntable(args, g, hand_entries, scale, cam_c2w, out_dir, device):
    # Rasterize near the Gaussians' native angular density, then image-upscale to
    # the output size. These feed-forward per-pixel surfels only tile at the
    # native resolution; rasterizing at high res exposes sub-pixel gaps as a
    # furry/cracked texture (the wandb render is crisp precisely because it is
    # native-res). So we keep the raster small and crisp, then enlarge.
    render_res = args.render_res or 220
    Wr = Hr = render_res
    bg = [float(x) for x in args.bg.split(",")]

    all_verts = np.concatenate([hand_vertices(m, scale) for m in hand_entries], axis=0)
    target = all_verts.mean(axis=0)
    extent = np.linalg.norm(all_verts - target, axis=1).max()

    # Orbit by RIGIDLY rotating the input reconstruction camera about the hand:
    # take its full pose (R0, c0) and apply a small rotation Rd to BOTH the
    # position and the orientation (R = Rd @ R0). Because every frame is a tiny
    # continuous rotation of a known-good camera, it can never roll or flip.
    # The hand is centred by shifting the principal point (a 2-D crop), NOT by
    # re-pointing the camera with look-at — re-pointing is what flipped before.
    R0 = cam_c2w[:3, :3].astype(np.float64)
    c0 = cam_c2w[:3, 3].astype(np.float64)
    cam_up = -R0[:, 1]    # camera up in world (OpenCV y is down)
    cam_right = R0[:, 0]
    cam_dist = np.linalg.norm(c0 - target)
    unit_dir = (c0 - target) / cam_dist
    half = np.deg2rad(args.vfov) / 2.0
    f = (render_res / 2.0) / np.tan(half)
    sweep = np.deg2rad(args.sweep)
    tilt = np.deg2rad(args.tilt)

    # Auto-fit the distance so the hand(s)' bounding sphere fills `--fill` of the
    # half-FOV — keeps both hands in frame regardless of how far apart they are.
    radius_fit = extent / (np.tan(half) * args.fill)
    radius = (args.radius if args.radius is not None else radius_fit) * args.zoom
    vec0 = unit_dir * radius

    # Fixed principal point: centre the hand(s) using the UNROTATED (centre)
    # frame and keep it constant across the sweep. A per-frame principal point
    # would pan the crop as the camera yaws, making the background appear to
    # swirl; fixing it makes the motion read as a clean, gentle parallax.
    eye0 = target + vec0
    tc0 = R0.T @ (target - eye0)
    cx = render_res / 2.0 - f * tc0[0] / tc0[2]
    cy = render_res / 2.0 - f * tc0[1] / tc0[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

    print(f"[turntable] target={np.round(target,3)} hand_extent={extent:.3f}m "
          f"cam_dist={cam_dist:.3f}m fit_radius={radius:.3f}m "
          f"sweep=+-{args.sweep}deg tilt=+-{args.tilt}deg  "
          f"{args.views} views, raster {render_res}px -> out {args.res}px")

    hero_idx = set(np.linspace(0, args.views, args.hero + 1)[:-1].astype(int))

    frames = []
    for i in range(args.views):
        # Sinusoidal ping-pong: rigidly rotate the input camera about the hand by
        # a small azimuth (about its up axis) and tilt (about its right axis).
        # R = Rd @ R0 keeps it a small continuous rotation of a known-good pose,
        # so it can never roll or flip; loops seamlessly.
        th = 2 * np.pi * i / args.views
        Rd = rot_axis_angle(cam_up, sweep * np.sin(th)) @ \
             rot_axis_angle(cam_right, tilt * np.sin(th))
        R = Rd @ R0                                  # rigid orientation, no flip
        eye = target + Rd @ vec0                     # rigid position (auto-fit dist)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, eye
        img01, depth = render_splats(g, c2w, K, Wr, Hr, bg, device, args.rasterize_mode)
        if hand_entries and not args.no_hand:
            img = composite_hands(img01, depth, hand_entries, scale, c2w, K,
                                  args.hand_alpha, not args.no_occlude)
        else:
            img = (img01 * 255).astype(np.uint8)
        img = resize_to(img, args.res, args.res)
        frames.append(img)
        if i in hero_idx:
            imageio.imwrite(out_dir / f"hero_{i:03d}.png", img)
        print(f"  view {i+1}/{args.views}", end="\r")
    print()

    write_video(out_dir / "turntable.mp4", frames, args.fps)
    print(f"[turntable] wrote {out_dir/'turntable.mp4'} and {len(hero_idx)} hero stills")


def run_camera(args, g, hand_entries, scale, c2w_all, K_all, out_dir, device):
    frame = args.frame
    c2w = c2w_all[frame]
    K_src = K_all[frame]
    H_src = int(round(K_src[1, 2] * 2))
    W_src = int(round(K_src[0, 2] * 2))
    # Rasterize at the camera's native resolution (crisp, matches the model's own
    # render) then image-upscale to the requested output size. Rasterizing at
    # high res instead would expose the surfel gaps as a cracked texture.
    render_res = args.render_res or max(H_src, W_src)
    up = render_res / max(H_src, W_src)
    Hr, Wr = int(round(H_src * up)), int(round(W_src * up))
    K = K_src.copy()
    K[:2, :] *= up
    bg = [float(x) for x in args.bg.split(",")]
    # Output size preserves the source aspect ratio.
    o_scale = args.res / max(Wr, Hr)
    out_w, out_h = int(round(Wr * o_scale)), int(round(Hr * o_scale))

    print(f"[camera] frame {frame}: raster {Wr}x{Hr} -> out {out_w}x{out_h}")
    img01, depth = render_splats(g, c2w, K, Wr, Hr, bg, device, args.rasterize_mode)
    plain = resize_to((img01 * 255).astype(np.uint8), out_w, out_h)
    imageio.imwrite(out_dir / f"recon_frame{frame:04d}.png", plain)
    print(f"[camera] wrote {out_dir/f'recon_frame{frame:04d}.png'}")

    if hand_entries and not args.no_hand:
        img = composite_hands(img01, depth, hand_entries, scale, c2w, K,
                              args.hand_alpha, not args.no_occlude)
        img = resize_to(img, out_w, out_h)
        imageio.imwrite(out_dir / f"recon_frame{frame:04d}_hand.png", img)
        print(f"[camera] wrote {out_dir/f'recon_frame{frame:04d}_hand.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True,
                   help="Directory produced by align_hand_to_neoverse.py.")
    p.add_argument("--mode", choices=["turntable", "camera", "both"], default="both")
    p.add_argument("--frame", type=int, default=None,
                   help="Frame offset to render the hand at. Default: middle frame.")
    p.add_argument("--scale", type=float, default=None,
                   help="Hand scale (vertices = pos_pred + scale*displacement). "
                        "Default: the saved alignment scale.")
    p.add_argument("--sides", default="both",
                   help="'both', 'left', or 'right' — which hand mesh(es) to draw. "
                        "'both' centres the orbit between the two hands.")
    p.add_argument("--no_hand", action="store_true", help="Render Gaussians only.")
    p.add_argument("--no_occlude", action="store_true",
                   help="Disable depth test (always draw the whole mesh on top).")

    p.add_argument("--res", type=int, default=1080,
                   help="Output (upscaled) resolution, long side, px.")
    p.add_argument("--render_res", type=int, default=None,
                   help="Rasterization resolution. Default: native (camera mode) "
                        "or 360 (turntable). Raise it for a sharper but more "
                        "cracked look — these per-pixel splats only tile at "
                        "native res.")
    p.add_argument("--rasterize_mode", choices=["classic", "antialiased"],
                   default="classic",
                   help="Match the model's renderer ('classic'). 'antialiased' "
                        "lowers tiny-Gaussian opacity and looks screen-doored.")
    p.add_argument("--bg", default="1,1,1", help="Background RGB in [0,1], e.g. '1,1,1'.")
    p.add_argument("--hand_alpha", type=float, default=0.8)
    p.add_argument("--opacity_thresh", type=float, default=0.05)

    # turntable-only
    p.add_argument("--views", type=int, default=180, help="Turntable frame count.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--sweep", type=float, default=22.0,
                   help="Azimuth half-range (deg) of the rocking turntable. Larger "
                        "= more parallax / more oblique angle, but eventually shows "
                        "the sheet's edge.")
    p.add_argument("--tilt", type=float, default=9.0,
                   help="Vertical rock half-range (deg) about the camera's right "
                        "axis, for a touch more 3-D parallax. 0 = pure horizontal.")
    p.add_argument("--fill", type=float, default=0.55,
                   help="Fraction of the half-FOV the hands' bounding sphere fills "
                        "(smaller = more scene context around the hands).")
    p.add_argument("--zoom", type=float, default=1.0,
                   help="Extra multiplier on the auto-fit distance. >1 backs off "
                        "for more context; <1 moves closer.")
    p.add_argument("--vfov", type=float, default=50.0, help="Orbit vertical FOV (deg).")
    p.add_argument("--radius", type=float, default=None,
                   help="Override orbit radius (m). Default: cam_dist * zoom.")
    p.add_argument("--hero", type=int, default=4, help="Number of hero stills to save.")

    p.add_argument("--out_subdir", default="report_3d")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available — gsplat requires a CUDA GPU and will "
              "likely fail on CPU.")

    groups, c2w_all, K_all, hands_by_frame, default_scale = load_scene(
        args.output_dir, device)

    n_frames = max(groups) + 1
    frame = args.frame if args.frame is not None else n_frames // 2
    scale = args.scale if args.scale is not None else default_scale

    entries = hands_by_frame.get(frame, [])
    if args.sides != "both":
        entries = [m for m in entries if m["side"] == args.sides]
    if not entries and not args.no_hand:
        print(f"WARNING: no hand mesh at frame {frame} (sides={args.sides}); "
              f"rendering Gaussians only.")

    g = gaussians_for_frame(groups, frame, args.opacity_thresh, device)
    print(f"Frame {frame}: {g['means'].shape[0]:,} Gaussians, scale={scale:.4f}, "
          f"{len(entries)} hand mesh(es)")

    out_dir = Path(args.output_dir) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("turntable", "both"):
        if not entries:
            raise SystemExit("turntable needs a hand mesh to orbit; pick a --frame "
                             "that has one, or use --mode camera.")
        run_turntable(args, g, entries, scale, c2w_all[frame], out_dir, device)
    if args.mode in ("camera", "both"):
        run_camera(args, g, entries, scale, c2w_all, K_all, out_dir, device)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
