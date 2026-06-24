"""Convert a raw HOI4D sequence into our pinhole training format.

Emits, per sequence, the exact cache set HOT3DHandDataset loads on the cache-HIT
path (so the dataset never falls into the HOT3D-specific compute paths):

    <out>/<seq>/
        video_main_rgb.mp4                      square-cropped RGB
        hand_data/cam_intrinsics.pt             [3] = [focal, cx, cy] (square frame)
        hand_data/cam_extrinsics_cache.pt       [N,4,4] T_camera_world
        hand_data/mano_hand_pose_trajectory.jsonl
        hand_data/gt_joints_cache_world.pt      [N,2,16,3]
        hand_data/gt_joints_cache_cam_v2.pt     [N,2,16,3]
        hand_data/gt_joints_2d_cache.pt         [N,2,16,3] (u,v,conf)
        hand_data/hand_bboxes_v2_rf{rf}_res{H}x{W}.pt

HOI4D facts used (see report/hoi4d-port-plan.md): RGB 1920x1080@15fps; depth 16-bit
mm aligned to RGB; intrinsics camera_params/<cam>/intrin.npy (3x3); extrinsics
3Dseg/output.log (Open3D per-frame 4x4); hands handpose/refinehandpose_right/.../*.pickle
MANO {poseCoeff[48]=3 global aa + 45 full pose, beta[10], trans (camera frame, m)}.

!!! THREE THINGS TO VERIFY against the first real sequence (marked VERIFY below):
  1. output.log pose direction (world->cam vs cam->world) — back-project depth to check.
  2. MANO 45-full -> 15-PCA conversion (flat_hand_mean handling) — compare joints.
  3. camera-frame -> world-frame lift of the hand global pose.

Usage (gb10 / venv_gb10):
    python -m scripts.preprocessing.preprocess_hoi4d \
        --hoi4d_root /work/scratch/dmonopoli/hoi4d \
        --seq ZY.../H1/C1/N1/S1/s1/T1 \
        --out /work/scratch/dmonopoli/hoi4d_preprocessed \
        --mano_model models/MANO --res 1080 --limit 64
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch

NUM_HANDS = 2
NUM_JOINTS = 16
HAND_PARAM_DIM = 32  # [transl 3, quat_wxyz 4, pose 15 (PCA), betas 10]


# ------------------------------------------------------------------ geometry
def _aa_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> unit quaternion (w,x,y,z)."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / theta
    w = np.cos(theta / 2.0)
    xyz = axis * np.sin(theta / 2.0)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def _aa_to_R(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(aa).as_matrix()


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


# ------------------------------------------------------------------ HOI4D I/O
def load_K(hoi4d_root: str, cam_id: str, intrin_npy=None, intrin_vals=None) -> np.ndarray:
    """Resolve the 3x3 intrinsic K (@ full_w x full_h). Priority: explicit npy path,
    then explicit 'fx fy cx cy' string, else the default HOI4D
    camera_params/<cam_id>/intrin.npy. The explicit forms let layouts without a
    camera_params/ tree (e.g. the Livioni mirror) pass the known ZY-camera intrinsic."""
    if intrin_npy:
        return np.load(intrin_npy)
    if intrin_vals:
        fx, fy, cx, cy = [float(x) for x in intrin_vals.replace(",", " ").split()]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return np.load(os.path.join(hoi4d_root, "camera_params", cam_id, "intrin.npy"))


def load_intrinsics(K: np.ndarray, full_w: int, full_h: int, res: int):
    """3x3 K (@ full_w x full_h) -> [focal, cx, cy] for a CENTER-SQUARE crop -> res.

    Our pipeline is single-focal square-pinhole. HOI4D is 1920x1080 with fx!=fy, so
    we center-crop to (full_h x full_h), shift cx by the crop offset, take fx as the
    focal, then scale to `res`.
    """
    fx, cx, fy, cy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
    crop = min(full_w, full_h)
    x0 = (full_w - crop) // 2
    cx_c = cx - x0                                   # crop to square (height-limited)
    s = res / float(crop)
    return torch.tensor([fx * s, cx_c * s, cy * s], dtype=torch.float32)


def load_extrinsics(seq_dir: str, n: int) -> torch.Tensor:
    """3Dseg/output.log (Open3D trajectory) -> [N,4,4] T_camera_world.

    VERIFY(1): Open3D `extrinsic` is world->cam, but the Redwood .log content is
    conventionally cam->world; loaders disagree. We assume the loaded `extrinsic`
    is already T_camera_world (world->cam). If a depth back-projection lands in the
    scene point cloud only after inverting, flip `invert=True`.
    """
    # Parse the Redwood/Open3D .log directly (no open3d dep — no aarch64 wheel in
    # venv_gb10). Format = blocks of 5 lines: 1 metadata line + 4 matrix rows.
    log = os.path.join(seq_dir, "3Dseg", "output.log")
    rows = [ln.strip() for ln in open(log) if ln.strip()]
    mats = []
    for b in range(0, len(rows) - 4, 5):
        m = [list(map(float, rows[b + 1 + r].split())) for r in range(4)]
        mats.append(np.asarray(m, dtype=np.float64))
    if not mats:
        raise SystemExit(f"No poses parsed from {log}")
    if len(mats) < n:
        mats += [mats[-1]] * (n - len(mats))         # pad short trajectories
    return torch.from_numpy(np.stack(mats[:n])).float()


def load_mano_frame(handpose_root: str, seq: str, frame: int):
    """<handpose_root>/<seq>/<frame>.pickle -> (global_aa[3], pose45[45], beta[10], trans[3]) or None.

    ``handpose_root`` already includes the per-hand subfolder (e.g.
    ``.../handpose_right_hand`` on the Livioni mirror, or ``.../refinehandpose_right``
    on the official release); ``seq`` is the nested ZY.../H*/C*/... path.
    """
    p = os.path.join(handpose_root, seq, f"{frame}.pickle")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    pose = np.asarray(d["poseCoeff"], dtype=np.float64)   # 48 = 3 global + 45 pose
    return pose[:3], pose[3:48], np.asarray(d["beta"], dtype=np.float64), np.asarray(d["trans"], dtype=np.float64)


# ------------------------------------------------------------------ MANO pose basis
def pose45_to_pca15(pose45: np.ndarray, mano_model, is_right: bool) -> np.ndarray:
    """Project HOI4D's 45 full (no-PCA, flat-hand-mean) pose onto our 15 PCA coeffs.

    VERIFY(2): assumes the smplx MANO layer exposes `hands_components` (45x45) and
    `hands_mean` (45). HOI4D uses flat_hand_mean=True (no mean added), so we subtract
    our layer's mean before projecting. Compare resulting joints to the manopth joints
    on one frame before trusting at scale.
    """
    layer = mano_model.right if is_right else mano_model.left
    comps = getattr(layer, "hands_components", None)
    mean = getattr(layer, "hands_mean", None)
    if comps is None:
        return pose45[:15]                            # fallback: truncate (flagged)
    comps = comps.detach().cpu().numpy() if torch.is_tensor(comps) else np.asarray(comps)
    mean = (mean.detach().cpu().numpy() if torch.is_tensor(mean)
            else (np.zeros(45) if mean is None else np.asarray(mean)))
    return (pose45 - mean) @ np.linalg.pinv(comps)[:, :15]


# ------------------------------------------------------------------ video
def _list_frame_images(d: str) -> list[str]:
    """Sorted frame paths in a directory of per-frame images (Livioni `images/`)."""
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(d) if f.lower().endswith(exts)]

    def _key(f: str) -> int:
        digits = "".join(ch for ch in os.path.splitext(f)[0] if ch.isdigit())
        return int(digits) if digits else -1

    return [os.path.join(d, f) for f in sorted(files, key=_key)]


def _crop_resize_square(img, res: int):
    h, w = img.shape[:2]
    s = min(h, w); y0, x0 = (h - s) // 2, (w - s) // 2
    import cv2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (res, res), interpolation=cv2.INTER_AREA)


def write_square_video(src: str, dst_mp4: str, res: int, limit: int | None) -> int:
    """Center-square-crop + resize the HOI4D RGB to dst_mp4. Returns frame count.

    `src` is either an `align_rgb/image.mp4` (official HOI4D) or a directory of
    per-frame images (the Livioni HF mirror ships `images/00000.jpg ...`).
    """
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if os.path.isdir(src):                                    # Livioni `images/` frame dir
        frames = _list_frame_images(src)
        if limit:
            frames = frames[:limit]
        vw = cv2.VideoWriter(dst_mp4, fourcc, 15.0, (res, res))
        written = 0
        for p in frames:
            img = cv2.imread(p)
            if img is None:
                break
            vw.write(_crop_resize_square(img, res)); written += 1
        vw.release()
        return written

    cap = cv2.VideoCapture(src)                               # official `image.mp4`
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(total, limit) if limit else total
    vw = cv2.VideoWriter(dst_mp4, fourcc, fps, (res, res))
    written = 0
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        vw.write(_crop_resize_square(img, res)); written += 1
    cap.release(); vw.release()
    return written


# ------------------------------------------------------------------ intrinsic recovery
def recover_K_from_kps2d(handpose_root: str, seq: str, mano, full_w: int, full_h: int,
                         n_use: int = 8, is_right: bool = True):
    """Recover the full-res pinhole intrinsic K by least-squares from HOI4D's GT kps2D.

    This mirror ships no camera_params/, but every handpose pickle carries
    ``kps2D[21,2]`` — the 2D projection of the 21 MANO joints. With the MANO 3D joints
    (camera frame, from poseCoeff/beta/trans) and their 2D projections, a skewless
    distortion-free pinhole solves directly:  u = fx*(X/Z)+cx,  v = fy*(Y/Z)+cy.
    Each axis is an independent 2-unknown least-squares over all joints/frames, so a
    handful of frames is hugely overdetermined. The camera is fixed across HOI4D, so
    one sequence's recovery serves all of them; reproj RMSE validates joint ordering.

    Returns (K[3x3] @ full_w x full_h, reproj_rmse_px, n_frames_used).
    """
    import glob
    pk_dir = os.path.join(handpose_root, seq)
    frames = sorted(int(os.path.basename(p)[:-7])
                    for p in glob.glob(os.path.join(pk_dir, "[0-9]*.pickle")))
    rows_u, rhs_u, rows_v, rhs_v = [], [], [], []
    used = 0
    for fi in frames:
        m = load_mano_frame(handpose_root, seq, fi)
        if m is None:
            continue
        with open(os.path.join(pk_dir, f"{fi}.pickle"), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        kps = d.get("kps2D")
        if kps is None:
            continue
        kps = np.asarray(kps, dtype=np.float64)               # [21,2] @ full res
        if kps.shape != (21, 2):
            continue
        g_aa, pose45, beta, trans = m
        pose15 = pose45_to_pca15(pose45, mano, is_right=is_right)
        q_cam = _R_to_quat_wxyz(_aa_to_R(g_aa))
        param = np.concatenate([trans, q_cam, pose15, beta]).astype(np.float32)  # [32]
        # 16-joint MANO (committed get_joints_batched). kps2D[:16] are the same
        # kinematic joints in MANO order; the 5 fingertips (kps2D[16:]) are the
        # convention-sensitive part, so dropping them avoids any tip-order mismatch
        # and is plenty (16 joints x 2 axes x n_use frames >> 4 unknowns).
        jc = mano.get_joints_batched(torch.from_numpy(param).unsqueeze(0),
                                     is_right=is_right)[0].detach().cpu().numpy()  # [16,3] cam
        z = np.clip(jc[:, 2], 1e-3, None)
        for i in range(jc.shape[0]):
            rows_u.append([jc[i, 0] / z[i], 1.0]); rhs_u.append(kps[i, 0])
            rows_v.append([jc[i, 1] / z[i], 1.0]); rhs_v.append(kps[i, 1])
        used += 1
        if used >= n_use:
            break
    if used == 0:
        raise RuntimeError(f"recover_K: no usable kps2D frames under {pk_dir}")
    Au, bu = np.asarray(rows_u), np.asarray(rhs_u)
    Av, bv = np.asarray(rows_v), np.asarray(rhs_v)
    (fx, cx), *_ = np.linalg.lstsq(Au, bu, rcond=None)
    (fy, cy), *_ = np.linalg.lstsq(Av, bv, rcond=None)
    rmse = float(np.sqrt(np.mean((Au @ [fx, cx] - bu) ** 2 + (Av @ [fy, cy] - bv) ** 2)))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, rmse, used


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoi4d_root", required=True, help="annotation root holding camera_params/, handpose/, and <seq>/3Dseg/output.log")
    ap.add_argument("--rgb_root", default=None, help="root holding RGB (defaults to hoi4d_root). Livioni mirror = <rgb_root>/<seq_underscored>/images/")
    ap.add_argument("--handpose_root", default=None, help="root holding <seq nested>/<frame>.pickle (default: <hoi4d_root>/handpose/refinehandpose_right). Livioni mirror = <gt>/Hand_pose/handpose_right_hand")
    ap.add_argument("--intrin_npy", default=None, help="explicit 3x3 intrinsic .npy at full_w x full_h (overrides camera_params lookup)")
    ap.add_argument("--intrin_vals", default=None, help="explicit 'fx fy cx cy' at full_w x full_h (overrides camera_params lookup)")
    ap.add_argument("--recover_k", action="store_true", help="recover the full-res intrinsic by least-squares from the pickles' kps2D vs MANO 3D joints (no camera_params needed)")
    ap.add_argument("--recover_k_frames", type=int, default=8, help="how many handpose frames to use for --recover_k")
    ap.add_argument("--seq", required=True, help="sequence subpath ZY.../H*/C*/N*/S*/s*/T*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mano_model", default="models/MANO")
    ap.add_argument("--res", type=int, default=1080, help="square output resolution")
    ap.add_argument("--full_w", type=int, default=1920)
    ap.add_argument("--full_h", type=int, default=1080)
    ap.add_argument("--rescale_factor", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=None, help="cap frames (debug)")
    args = ap.parse_args()

    from scripts.hand_vis_utils import MANOModel
    mano = MANOModel(args.mano_model)

    seq_dir = os.path.join(args.hoi4d_root, args.seq)
    cam_id = args.seq.split("/")[0]                   # ZY... camera id is the first path token
    handpose_root = args.handpose_root or os.path.join(args.hoi4d_root, "handpose", "refinehandpose_right")
    out_seq = os.path.join(args.out, args.seq.replace("/", "_"))
    hd = os.path.join(out_seq, "hand_data")
    os.makedirs(hd, exist_ok=True)

    # 1) RGB -> square video. Prefer official align_rgb/image.mp4; fall back to the
    #    Livioni mirror's per-frame images/ dir (keyed by underscored seq name).
    rgb_root = args.rgb_root or args.hoi4d_root
    seq_us = args.seq.replace("/", "_")
    mp4 = os.path.join(rgb_root, args.seq, "align_rgb", "image.mp4")
    rgb_src = mp4 if os.path.exists(mp4) else os.path.join(rgb_root, seq_us, "images")
    n = write_square_video(rgb_src, os.path.join(out_seq, "video_main_rgb.mp4"), args.res, args.limit)
    print(f"[rgb] {n} frames -> {args.res}x{args.res}  (src={rgb_src})")

    # 2) intrinsics + extrinsics
    if args.recover_k and not (args.intrin_npy or args.intrin_vals):
        K, k_rmse, k_n = recover_K_from_kps2d(handpose_root, args.seq, mano,
                                              args.full_w, args.full_h, n_use=args.recover_k_frames)
        flag = "" if k_rmse < 10.0 else "  [WARN rmse>10px: check kps2D joint order/res]"
        print(f"[intrin] recovered K from kps2D over {k_n} frames  rmse={k_rmse:.2f}px  "
              f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}{flag}", flush=True)
    else:
        K = load_K(args.hoi4d_root, cam_id, args.intrin_npy, args.intrin_vals)
    cam_intr = load_intrinsics(K, args.full_w, args.full_h, args.res)
    # Extrinsics (3Dseg/output.log) are ONLY needed for the world-frame cache. When
    # they are absent (e.g. the Livioni HF mirror has images + depth but no poses),
    # fall back to building CAMERA-FRAME caches only — enough for camera-frame
    # C-MPJPE / C-abs (the HOI4D MANO is already camera-frame, so no extrinsics
    # are required to get gt_joints_cache_cam_v2.pt).
    try:
        cam_extr = load_extrinsics(seq_dir, n)        # [N,4,4] T_camera_world (VERIFY direction)
        have_extr = True
    except (FileNotFoundError, OSError) as e:
        cam_extr = None
        have_extr = False
        print(f"[warn] no extrinsics ({type(e).__name__}); building CAMERA-FRAME caches "
              f"only (skipping gt_joints_cache_world.pt + cam_extrinsics_cache.pt)")
    torch.save(cam_intr, os.path.join(hd, "cam_intrinsics.pt"))
    if have_extr:
        torch.save(cam_extr, os.path.join(hd, "cam_extrinsics_cache.pt"))
    f0, cx0, cy0 = [float(x) for x in cam_intr.tolist()]

    # 3) hands: pickle -> world-frame params + joints + 2d + bboxes
    jl = []                                                    # jsonl entries
    j_world = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    j_cam = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    j_2d = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    bboxes = torch.zeros(n, NUM_HANDS, 4)
    valid = torch.zeros(n, NUM_HANDS, dtype=torch.bool)
    gt64 = torch.zeros(n, NUM_HANDS * HAND_PARAM_DIM)
    RH = 1                                                     # HOI4D releases right hand -> index 1

    for fi in range(n):
        hp = {}
        for h in range(NUM_HANDS):
            if h != RH:
                continue
            m = load_mano_frame(handpose_root, args.seq, fi)
            if m is None:
                continue
            g_aa, pose45, beta, trans = m                     # camera frame
            pose15 = pose45_to_pca15(pose45, mano, is_right=True)

            off = h * HAND_PARAM_DIM
            R_cam = _aa_to_R(g_aa); t_cam = trans
            if have_extr:
                # VERIFY(3): lift camera-frame global pose -> world frame via T_world_camera.
                T_cw = cam_extr[fi].double().numpy()              # world->cam (assumed)
                T_wc = np.linalg.inv(T_cw)                        # cam->world
                R_world = T_wc[:3, :3] @ R_cam
                t_world = T_wc[:3, :3] @ t_cam + T_wc[:3, 3]
                q_world = _R_to_quat_wxyz(R_world)
                hp[str(h)] = {
                    "wrist_xform": {"t_xyz": t_world.tolist(), "q_wxyz": q_world.tolist()},
                    "pose": [float(x) for x in pose15.tolist()],
                    "betas": [float(x) for x in beta.tolist()],
                }
                gt64[fi, off:off + 3] = torch.tensor(t_world, dtype=torch.float32)
                gt64[fi, off + 3:off + 7] = torch.tensor(q_world, dtype=torch.float32)
                gt64[fi, off + 7:off + 22] = torch.tensor(pose15, dtype=torch.float32)
                gt64[fi, off + 22:off + 32] = torch.tensor(beta, dtype=torch.float32)
                # joints via OUR MANO (consistent 16-joint convention), world frame
                jw = mano.get_joints_from_tensor(gt64[fi, off:off + HAND_PARAM_DIM],
                                                 is_right=True, return_tensor=False)
                jw = np.asarray(jw).reshape(-1, 3)[:NUM_JOINTS]
                j_world[fi, h] = torch.tensor(jw, dtype=torch.float32)
                # camera frame = world joints transformed back via T_cw
                jc = (T_cw[:3, :3] @ jw.T + T_cw[:3, 3:4]).T
            else:
                # No extrinsics: the HOI4D MANO is already camera-frame, so build the
                # camera-frame joints directly. (Provably identical to the roundtrip
                # above: the T_cw/T_wc terms cancel to R_cam @ J + t_cam.)
                q_cam = _R_to_quat_wxyz(R_cam)
                gt64[fi, off:off + 3] = torch.tensor(t_cam, dtype=torch.float32)
                gt64[fi, off + 3:off + 7] = torch.tensor(q_cam, dtype=torch.float32)
                gt64[fi, off + 7:off + 22] = torch.tensor(pose15, dtype=torch.float32)
                gt64[fi, off + 22:off + 32] = torch.tensor(beta, dtype=torch.float32)
                hp[str(h)] = {
                    "wrist_xform": {"t_xyz": [float(x) for x in t_cam], "q_wxyz": q_cam.tolist()},
                    "pose": [float(x) for x in pose15.tolist()],
                    "betas": [float(x) for x in beta.tolist()],
                }
                jc = mano.get_joints_from_tensor(gt64[fi, off:off + HAND_PARAM_DIM],
                                                 is_right=True, return_tensor=False)
                jc = np.asarray(jc).reshape(-1, 3)[:NUM_JOINTS]
            j_cam[fi, h] = torch.tensor(jc, dtype=torch.float32)
            # 2d projection (pinhole)
            z = np.clip(jc[:, 2], 1e-3, None)
            u = f0 * jc[:, 0] / z + cx0
            v = f0 * jc[:, 1] / z + cy0
            j_2d[fi, h, :, 0] = torch.tensor(u, dtype=torch.float32)
            j_2d[fi, h, :, 1] = torch.tensor(v, dtype=torch.float32)
            j_2d[fi, h, :, 2] = 1.0
            inb = (u >= 0) & (u < args.res) & (v >= 0) & (v < args.res)
            valid[fi, h] = bool(inb.any())
            if inb.any():
                x1, x2 = float(u[inb].min()), float(u[inb].max())
                y1, y2 = float(v[inb].min()), float(v[inb].max())
                cx_b, cy_b = (x1 + x2) / 2, (y1 + y2) / 2
                w_b = (x2 - x1) * args.rescale_factor
                h_b = (y2 - y1) * args.rescale_factor
                bboxes[fi, h] = torch.tensor([
                    (cx_b - w_b / 2) / args.res, (cy_b - h_b / 2) / args.res,
                    (cx_b + w_b / 2) / args.res, (cy_b + h_b / 2) / args.res,
                ], dtype=torch.float32)
        jl.append({"timestamp_ns": fi, "hand_poses": hp})

    with open(os.path.join(hd, "mano_hand_pose_trajectory.jsonl"), "w") as f:
        for e in jl:
            f.write(json.dumps(e) + "\n")
    if have_extr:
        torch.save(j_world, os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(j_cam, os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(j_2d, os.path.join(hd, "gt_joints_2d_cache.pt"))
    bbox_name = f"hand_bboxes_v2_rf{args.rescale_factor}_res{args.res}x{args.res}.pt"
    torch.save({"bboxes": bboxes, "valid": valid, "gt": gt64}, os.path.join(hd, bbox_name))

    print(f"[done] {out_seq}: {n} frames, valid-hand frames={int(valid.any(-1).sum())}")
    print("VERIFY: (1) extrinsics direction, (2) 45->15 PCA pose, (3) cam->world lift "
          "— check gt_joints_2d overlays the hand in the RGB before scaling up.")


if __name__ == "__main__":
    main()
