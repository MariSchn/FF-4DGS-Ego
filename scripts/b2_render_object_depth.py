"""B2 geometry: render GT object depth from posed HOT3D meshes (non-circular scene-metric).

This is the *ground-truth depth* half of publication-plan E2. The scene-metric
thesis ("the hand anchor makes the SCENE metric, not just the hand") is currently
only proven AT the hand (semi-circular, the anchor trains there). E2 proves it in
NON-HAND regions by comparing our anchored ``gs_depth`` to real metric depth on the
manipulated objects, whose 6-DoF poses + meshes are GT in HOT3D.

This module is pure geometry (numpy / torch / trimesh) — NO model, NO GPU — so it
runs and self-validates on the login node. ``scripts/eval_scene_metric_gt.py`` then
combines it with a model forward (gs_depth) on a gb10 node.

Inputs per sequence (raw HOT3D seq dir):
  * ground_truth/dynamic_objects.csv : per-(uid, timestamp) T_world_object (t_wo, q_wo).
  * hand_data/cam_extrinsics_cache.pt: [N, 4, 4] per-frame camera extrinsics.
  * hand_data/cam_intrinsics.pt      : [3] = [f, cx, cy] in the pinhole frame.
Object meshes: <objects_dir>/<uid>.glb (HOT3D object library, metres, canonical frame).

Self-test:
    python scripts/b2_render_object_depth.py \
        --seq /work/courses/3dv/team25/data/hot3d_aria/sequences/P0001_10a27bf7 \
        --objects_dir /work/scratch/dmonopoli/hot3d_assets --frames 0,100,500
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache

import numpy as np
import torch


# ------------------------------------------------------------------ poses
def quat_wxyz_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Unit-quaternion (w, x, y, z) -> 3x3 rotation."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _load_dyn_objects(seq_dir: str):
    """Parse dynamic_objects.csv -> ({ts: [(uid, T_wo 4x4)]}, sorted-ts ndarray)."""
    import pandas as pd
    csv = os.path.join(seq_dir, "ground_truth", "dynamic_objects.csv")
    df = pd.read_csv(csv)
    # Columns (by position): object_uid, timestamp[ns], t_wo_{x,y,z}[m], q_wo_{w,x,y,z}.
    uids = df.iloc[:, 0].to_numpy()
    tss = df.iloc[:, 1].to_numpy(dtype=np.int64)
    two = df.iloc[:, 2:5].to_numpy(dtype=np.float64)
    qwo = df.iloc[:, 5:9].to_numpy(dtype=np.float64)
    by_ts: dict[int, list] = {}
    for r in range(len(df)):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quat_wxyz_to_R(qwo[r, 0], qwo[r, 1], qwo[r, 2], qwo[r, 3])
        T[:3, 3] = two[r]
        by_ts.setdefault(int(tss[r]), []).append((str(int(uids[r])), T))
    return by_ts, np.array(sorted(by_ts.keys()), dtype=np.int64)


def _hand_ts_sorted(seq_dir: str):
    """Sorted device timestamps from the MANO JSONL (the frame clock the caches use)."""
    import json
    jl = os.path.join(seq_dir, "hand_data", "mano_hand_pose_trajectory.jsonl")
    ts = {json.loads(ln)["timestamp_ns"] for ln in open(jl) if ln.strip()}
    return sorted(ts)


def build_frame_objects(seq_dir: str, n_video: int):
    """Frame-indexed object poses, aligned EXACTLY as preprocess_undistort did.

    Reproduces ``frame_camera_transforms``: frame i -> query timestamp by linear
    interpolation between the first/last MANO-JSONL timestamps, then nearest
    object-pose timestamp. This guarantees frame_objects[i] lines up with the
    cached cam_extrinsics[i] used by the model.

    Returns: list of length n_video; entry i = [(uid:str, T_world_object 4x4), ...].
    """
    import bisect
    hand_ts = _hand_ts_sorted(seq_dir)
    ts0, ts1 = hand_ts[0], hand_ts[-1]
    by_ts, dyn_ts = _load_dyn_objects(seq_dir)
    dyn_ts_list = dyn_ts.tolist()
    out: list[list] = []
    for fi in range(n_video):
        frac = fi / max(n_video - 1, 1)
        q = int(ts0 + frac * (ts1 - ts0))
        j = bisect.bisect_left(dyn_ts_list, q)
        cands = []
        if j < len(dyn_ts_list):
            cands.append(dyn_ts_list[j])
        if j > 0:
            cands.append(dyn_ts_list[j - 1])
        best = min(cands, key=lambda t: abs(t - q))
        out.append(by_ts[best])
    return out


@lru_cache(maxsize=128)
def _load_mesh_vertices(glb_path: str) -> np.ndarray:
    """Load a .glb as a single mesh (baked transforms) -> [V, 3] float64 vertices (metres)."""
    import trimesh
    m = trimesh.load(glb_path, force="mesh", process=False)
    return np.asarray(m.vertices, dtype=np.float64)


# ------------------------------------------------------------------ render
def render_object_depth(frame_objects, T_cam_world: np.ndarray, K, H: int, W: int,
                        objects_dir: str, device: str = "cpu"):
    """Front-most object depth map via vertex projection + per-pixel z-min.

    Vertices (true surface points) are projected and reduced with amin per pixel —
    an occlusion-approximate, GL-free depth that is dense for HOT3D meshes.

    Args:
        frame_objects: [(uid, T_world_object 4x4), ...] present this frame.
        T_cam_world:   4x4 world->camera (X_cam = T_cam_world @ X_world).
        K:             (f, cx, cy) pinhole intrinsics for the [H, W] frame.
        Returns: (depth [H,W] torch, mask [H,W] bool torch); depth=inf where no object.
    """
    f, cx, cy = float(K[0]), float(K[1]), float(K[2])
    depth = torch.full((H * W,), float("inf"), dtype=torch.float32, device=device)
    for uid, T_wo in frame_objects:
        glb = os.path.join(objects_dir, f"{uid}.glb")
        if not os.path.exists(glb):
            continue
        try:
            V = _load_mesh_vertices(glb)                               # [V,3] world-canonical
        except Exception:
            continue  # a single unreadable mesh must not crash the whole eval
        T_co = T_cam_world @ T_wo                                      # cam<-object
        Vc = (T_co[:3, :3] @ V.T + T_co[:3, 3:4]).T                    # [V,3] camera frame
        Z = Vc[:, 2]
        keep = Z > 1e-3
        if not keep.any():
            continue
        Xc, Yc, Zc = Vc[keep, 0], Vc[keep, 1], Vc[keep, 2]
        u = np.round(f * Xc / Zc + cx).astype(np.int64)
        v = np.round(f * Yc / Zc + cy).astype(np.int64)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inb.any():
            continue
        pix = torch.from_numpy((v[inb] * W + u[inb])).to(device)
        z = torch.from_numpy(Zc[inb].astype(np.float32)).to(device)
        depth.scatter_reduce_(0, pix, z, reduce="amin", include_self=True)
    depth = depth.view(H, W)
    return depth, torch.isfinite(depth)


# ------------------------------------------------------------------ extrinsics convention
def resolve_extrinsics_convention(cam_extr: torch.Tensor, joints_world: torch.Tensor,
                                  joints_cam: torch.Tensor) -> str:
    """Decide whether cam_extr is world->cam ('w2c') or cam->world ('c2w').

    Checks which transform maps the GT world hand joints onto the cached GT
    camera-frame joints. Returns 'w2c' or 'c2w'.
    """
    # pick a frame with a valid (non-zero) hand
    N = cam_extr.shape[0]
    for i in range(0, N, max(1, N // 50)):
        jw = joints_world[i].reshape(-1, 3)
        jc = joints_cam[i].reshape(-1, 3)
        if jw.abs().sum() < 1e-6 or jc.abs().sum() < 1e-6:
            continue
        E = cam_extr[i].double()
        jw_h = torch.cat([jw.double(), torch.ones(jw.shape[0], 1, dtype=torch.float64)], -1)
        cam_w2c = (E @ jw_h.T).T[:, :3]
        cam_c2w = (torch.linalg.inv(E) @ jw_h.T).T[:, :3]
        e_w2c = (cam_w2c - jc.double()).abs().mean().item()
        e_c2w = (cam_c2w - jc.double()).abs().mean().item()
        return "w2c" if e_w2c < e_c2w else "c2w", (e_w2c, e_c2w)
    return "w2c", (float("nan"), float("nan"))


# ------------------------------------------------------------------ self-test
def _selftest(seq_dir: str, objects_dir: str, frames):
    hd = os.path.join(seq_dir, "hand_data")
    cam_extr = torch.load(os.path.join(hd, "cam_extrinsics_cache.pt"), map_location="cpu").float()
    cam_intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float()
    jc = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), map_location="cpu").float()
    jw_path = os.path.join(hd, "gt_joints_cache_world.pt")
    jw = torch.load(jw_path, map_location="cpu").float() if os.path.exists(jw_path) else None
    N = cam_extr.shape[0]
    f, cx, cy = cam_intr.tolist()
    # frame size: infer from principal point if square (2*cx ~ W)
    W = H = int(round(2 * cx))
    print(f"seq frames N={N} | cam_intr f={f:.2f} cx={cx:.2f} cy={cy:.2f} -> assuming {W}x{H}")

    if jw is not None:
        conv, (e_w2c, e_c2w) = resolve_extrinsics_convention(cam_extr, jw, jc)
        print(f"extrinsics convention: {conv}  (err w2c={e_w2c*1000:.2f}mm  c2w={e_c2w*1000:.2f}mm)")
    else:
        conv = "w2c"
        print("no world joints cache; assuming w2c")

    frame_objs = build_frame_objects(seq_dir, N)
    print(f"dynamic_objects aligned to {N} frames via interp+nearest (preprocess mapping)")
    avail = {p[:-4] for p in os.listdir(objects_dir) if p.endswith(".glb")}
    print(f"object meshes available: {len(avail)}")

    for fi in frames:
        if fi >= N or fi >= len(frame_objs):
            print(f"frame {fi}: out of range"); continue
        E = cam_extr[fi].double().numpy()
        T_cam_world = E if conv == "w2c" else np.linalg.inv(E)
        objs = [(u, T) for (u, T) in frame_objs[fi] if u in avail]
        depth, mask = render_object_depth(objs, T_cam_world, (f, cx, cy), H, W, objects_dir)
        npix = int(mask.sum())
        # STRONG validation: object surface depth sampled at the projected hand joints
        # vs the independently-known metric hand depth. When the hand contacts an
        # object, these agree within object thickness (~few cm) ONLY if the whole
        # chain (object pose + frame alignment + extrinsics + intrinsics) is correct.
        jcf = jc[fi].reshape(-1, 3)
        valid_j = jcf.abs().sum(-1) > 1e-6
        hz = jcf[:, 2][valid_j]
        hand_obj_err = float("nan")
        if npix and valid_j.any():
            u = torch.round(f * jcf[:, 0] / jcf[:, 2].clamp(min=1e-3) + cx).long()
            v = torch.round(f * jcf[:, 1] / jcf[:, 2].clamp(min=1e-3) + cy).long()
            inb = valid_j & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if inb.any():
                od = depth[v[inb].clamp(0, H - 1), u[inb].clamp(0, W - 1)]
                hit = torch.isfinite(od)
                if hit.any():
                    hand_obj_err = float((od[hit] - jcf[:, 2][inb][hit]).abs().median()) * 100
        dvals = depth[mask]
        print(f"frame {fi}: objs={len(objs)}/{len(frame_objs[fi])} | obj_px={npix} "
              f"({100*npix/(H*W):.1f}%) | obj_depth med={dvals.median().item() if npix else float('nan'):.3f}m | "
              f"hand_z med={hz.median().item() if hz.numel() else float('nan'):.3f}m | "
              f"|obj_depth - hand_z|@hand median={hand_obj_err:.1f}cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="raw HOT3D sequence dir")
    ap.add_argument("--objects_dir", required=True)
    ap.add_argument("--frames", default="0,100,500")
    args = ap.parse_args()
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    _selftest(args.seq, args.objects_dir, frames)


if __name__ == "__main__":
    main()
