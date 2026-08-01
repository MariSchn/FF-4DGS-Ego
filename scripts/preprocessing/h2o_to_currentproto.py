"""Convert packed H2O .npz sequences (scripts.pack_h2o) into the current-protocol
per-sequence layout consumed by scripts.train_hand_head (HOT3DHandDataset) and
scripts.eval_hand_cam_anchor — making H2O a true second dataset next to HOI4D.

Emits, per sequence (exactly the cache-HIT set, mirroring preprocess_hoi4d):

    <out>/<seq>/
        video_main_rgb.mp4                       224x224 (re-encoded npz rgb)
        hand_data/cam_intrinsics.pt              [3] = [focal, cx, cy] float32
        hand_data/cam_extrinsics_cache.pt        [N,4,4] float32 T_camera_world
        hand_data/mano_hand_pose_trajectory.jsonl
        hand_data/gt_joints_cache_world.pt       [N,2,16,3] float32 world, metres
        hand_data/gt_joints_cache_cam_v2.pt      [N,2,16,3] float32 camera, metres
        hand_data/gt_joints_2d_cache.pt          [N,2,16,3] float32 (u,v,conf) px
        hand_data/hand_bboxes_v2_rf{rf}_res{R}x{R}.pt
                                                 {bboxes [N,2,4] norm xyxy f32,
                                                  valid [N,2] bool, gt [N,64] f32}

SCHEMA DECISIONS (each traced to the reference implementation)
  * Hand order: LEFT = index 0, RIGHT = index 1 — matches HOT3D ("0"=left,
    "1"=right, train_hand_head.py:626-628) and eval_cmpjpe.h2o_gt_joints16's
    stack([left16, right16]). HOI4D only filled index 1; H2O fills BOTH, with
    per-hand `valid` from H2O's hand_pose valid flags (j128[0], j128[64]).
  * Joints: H2O hand_pose 21 joints are camera-frame METRES (verified in
    eval_cmpjpe.py — do NOT /1000). 21 -> smplx-16 remap is H2O_TO_MANO[:16]
    (fixed 2026-07-18; the old MANO21_TO_16 composition scrambled the joints);
    the copy below is asserted identical to scripts.eval_cmpjpe inside --validate,
    and gate 4 checks anatomical bone lengths so an order bug cannot pass again.
  * Intrinsics: npz K=[fx,fy,cx,cy,w,h] at 1280x720. The pack center-square-
    cropped to 720^2 then resized to 224 (pack_h2o._center_square_resize), so
    f = fx*s, cx' = (cx - (w-h)//2)*s, cy' = cy*s with s = res/h — the same
    adjustment as eval_cmpjpe.bboxes_from_joints (eval_cmpjpe.py:87-89). Our
    protocol is single-focal (preprocess_hoi4d.load_intrinsics takes fx);
    gate 1 measures the fx!=fy anisotropy and fails if it exceeds 1px median.
  * Extrinsics: H2O cam_pose is cam->world (pack_h2o docstring); the cache
    stores its INVERSE, T_camera_world (world->cam), the convention asserted
    at train_hand_head.py:526 ("[S,4,4] T_cam_world (validated w2c)") and
    produced by _compute_2d_cam_data (train_hand_head.py:733-736,749).
    gt_joints_cache_world = cam_pose @ cam-frame joints.
  * 2D cache: plain single-focal pinhole u = f*x/z + cx (u,v,conf), conf=1 for
    a present hand — byte-for-byte the preprocess_hoi4d.py:438-443 convention.
    NOTE the kp2d loss path (train_hand_head.py:1156-1160) hardcodes the Aria
    1408px + 90-degree rotation, so kp2d must stay DISABLED for H2O exactly as
    for HOI4D; kp3d/kp3d_abs use only the joint caches.
  * Bboxes: H2O npz carries no MANO vertices, so the tight box is over the
    projected 21 H2O joints (all 21 incl. fingertips, in-frame subset) instead
    of the 778 projected vertices — consistent for BOTH train and eval since
    both read this same cache (no asymmetry). Protocol = HOT3D reference
    (train_hand_head.py:650-680): tight box -> x rescale_factor (1.5) ->
    SQUARE to max side -> normalize by res -> clamp [0,1].
    KNOWN DISCREPANCY: preprocess_hoi4d.py:446-455 does NOT square and does
    NOT clamp (per-axis w/h x1.5). We follow the HOT3D square protocol per
    the porting decision; flagged so a cross-dataset run knows the difference.
  * MANO jsonl (best-effort): H2O mano[124] = per hand [valid, trans(3),
    pose48(3 global aa + 45 full aa, no PCA), beta(10)]. We emit CAMERA-frame
    wrist t_xyz/q_wxyz (q from the 3-dim global axis-angle) + the lossy 45->15
    PCA projection via preprocess_hoi4d.pose45_to_pca15 + betas — the exact
    fields HOT3DHandDataset._hand_to_vec parses (train_hand_head.py:194-207).
    Camera frame matches the use_hand_crop crop-local semantics and the HOI4D
    no-extrinsics path (preprocess_hoi4d.py:417-430). LOSS IMPACT: kp3d /
    kp3d_abs / kp2d read only the joint caches and are UNAFFECTED; MANO-param
    losses (criterion_param on gt64) see the lossy PCA15 + an UNVERIFIED H2O
    trans/axis-angle convention — the --validate MANO forward check reports
    the resulting joint error so this stays measured, not assumed.

CONVENTION RISKS that can only be verified on-cluster (real npz):
  1. H2O mano trans units/semantics vs smplx `transl` (metres assumed, like
     the joints). Check: --validate "[mano]" forward-kinematics report.
  2. H2O mano full-45 pose flat_hand_mean convention (HOI4D was
     flat_hand_mean=True; hand_vis_utils.py:105-116). Same check as (1).
  3. cam_pose translation units (metres assumed). Check: --validate prints
     median |t| and gate 4 round-trips the transform.
  4. fx vs fy anisotropy of the H2O ego camera. Check: gate 1.

VALIDATION GATES (--validate; MANDATORY before training on the output):
  1. Reprojection: project gt_joints_cache_cam_v2 with cam_intrinsics
     (single focal) -> median px distance to (a) gt_joints_2d_cache < 1e-3
     (internal consistency) and (b) the TRUE fx/fy projection built
     independently per eval_cmpjpe's K adjustment < 1.0 px.
  2. Round-trip vs the old pipeline: (a) eval_cmpjpe.h2o_gt_joints16(npz
     joints) == cam_v2 cache within 1e-4 m; (b) GT-as-prediction through OUR
     eval loading path (HOT3DHandDataset clips -> overlap-dedup, the
     eval_hand_cam_anchor recipe) scored with world_space_metrics.c_mpjpe
     == ~0 (< 1e-3 mm), plus video frame count == N and a first-clip decode.
  3. Bbox sanity: recomputed-from-npz box == stored (1e-5), square before
     clamping, inside [0,1], contains every in-frame projected joint.
  4. Extrinsics direction: T_cw @ cam_pose == I (1e-4); cam -> world -> cam
     joint round-trip < 1e-5 m; world cache == cam_pose-lifted cam cache.
  A failed gate exits non-zero and names the sequence + numbers. Do NOT
  loosen a gate to make it pass — report the failure.

Usage (gb10 / venv_gb10; see configs/_h2o_currentproto.sbatch):
    python -m scripts.preprocessing.h2o_to_currentproto \
        --h2o /work/scratch/dmonopoli/h2o_packed \
        --out /work/scratch/dmonopoli/h2o_currentproto \
        --mano_model models/MANO --max_seqs 8 --validate
Resume: already-complete sequence dirs are skipped; partial output is built in
a .tmp_ dir and atomically renamed, so re-running after a quota kill is safe.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

from scripts.preprocessing.preprocess_hoi4d import _aa_to_quat_wxyz, pose45_to_pca15

NUM_HANDS = 2
NUM_JOINTS = 16
HAND_PARAM_DIM = 32          # [transl 3, quat_wxyz 4, pose 15 (PCA), betas 10]

# Copy of scripts.eval_cmpjpe H2O_TO_MANO (kept import-light here: eval_cmpjpe pulls
# in WorldMirror at module top). --validate asserts it equals the original.
# Positions 0-15 are the smplx-16 kinematic order; 16-20 are the fingertips.
# BUG FIX 2026-07-18: the old H2O16_IDX composed H2O_TO_MANO with a 21->16 selector
# written for the H2O-native tips-interleaved layout, producing scrambled 16-joint
# caches (thumb slots held fingertips, ring slot held thumb CMC, three MCPs missing).
# Confirmed by anatomical bone lengths on the converted store; caught by gate 4 below.
H2O_TO_MANO = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
H2O16_IDX = H2O_TO_MANO[:16]

CACHE_FILES = [
    "cam_intrinsics.pt", "cam_extrinsics_cache.pt", "mano_hand_pose_trajectory.jsonl",
    "gt_joints_cache_world.pt", "gt_joints_cache_cam_v2.pt", "gt_joints_2d_cache.pt",
]


# ------------------------------------------------------------------ npz access
def load_npz_checked(path: str) -> dict:
    """Load one packed sequence and validate every array shape at the boundary."""
    d = np.load(path)
    rgb = d["rgb"]
    n, res = int(rgb.shape[0]), int(rgb.shape[1])
    if rgb.shape != (n, res, res, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"rgb shape/dtype off: {rgb.shape} {rgb.dtype}")
    joints = np.asarray(d["joints"], np.float64)
    mano = np.asarray(d["mano"], np.float64)
    cam_pose = np.asarray(d["cam_pose"], np.float64)
    K = np.asarray(d["K"], np.float64)
    if joints.shape != (n, 128):
        raise ValueError(f"joints shape {joints.shape} != ({n},128)")
    if mano.shape != (n, 124):
        raise ValueError(f"mano shape {mano.shape} != ({n},124)")
    if cam_pose.shape != (n, 4, 4):
        raise ValueError(f"cam_pose shape {cam_pose.shape} != ({n},4,4)")
    if K.size < 6 or K[0] <= 0 or K[1] <= 0 or K[5] <= 0 or K[4] < K[5]:
        raise ValueError(f"K invalid (need [fx,fy,cx,cy,w,h], w>=h>0): {K}")
    return {"rgb": rgb, "joints": joints, "mano": mano, "cam_pose": cam_pose,
            "K": K, "n": n, "res": res}


def hand21_from_vec(j128: np.ndarray, hand: int):
    """One frame's [128] hand_pose -> ([21,3] m cam frame, valid) for hand 0/1."""
    off = 0 if hand == 0 else 64
    return j128[off + 1:off + 64].reshape(21, 3), float(j128[off]) > 0.5


def mano_from_vec(m124: np.ndarray, hand: int):
    """One frame's [124] mano -> (valid, trans[3], global_aa[3], pose45[45], beta[10])."""
    off = 62 * hand
    v = m124[off:off + 62]
    return float(v[0]) > 0.5, v[1:4], v[4:7], v[7:52], v[52:62]


def square_intrinsics(K: np.ndarray, res: int):
    """[fx,fy,cx,cy,w,h] full-frame -> single-focal [f,cx,cy] tensor for the
    center-square crop (h x h) + resize to res. eval_cmpjpe.py:87-89 convention;
    f = fx (preprocess_hoi4d.load_intrinsics precedent). Returns (tensor,
    (fx_s, fy_s, cx_s, cy_s)) with the anisotropic pair kept for gate 1."""
    fx, fy, cx, cy, w, h = [float(x) for x in K[:6]]
    x0 = (int(w) - int(h)) // 2
    s = res / h
    fx_s, fy_s, cx_s, cy_s = fx * s, fy * s, (cx - x0) * s, cy * s
    return torch.tensor([fx_s, cx_s, cy_s], dtype=torch.float32), (fx_s, fy_s, cx_s, cy_s)


# ------------------------------------------------------------------ geometry
def _is_rigid(T: np.ndarray) -> bool:
    """True if T is a proper rigid cam->world transform (bottom row 0001, det R ~1)."""
    return bool(np.allclose(T[3], [0, 0, 0, 1], atol=1e-4)
                and abs(np.linalg.det(T[:3, :3]) - 1.0) <= 1e-2)


def sanitize_cam_pose(cam_pose: np.ndarray):
    """[N,4,4] cam->world -> (filled cam_pose [N,4,4], T_cw [N,4,4] world->cam,
    filled_idx list).

    H2O occasionally records an all-zero cam_pose for isolated frames (no camera
    pose tracked that frame — seen on subject3 *_1 takes at frame 0). Those are
    NOT rigid transforms and cannot be inverted. Rather than drop the whole
    sequence (which would also discard the camera-frame caches that never use
    cam_pose), carry-forward the nearest valid pose (previous if available, else
    next) into the invalid frames and record their indices. This keeps cam_pose
    and its inverse T_cw mutually consistent (so gate 4 stays meaningful) and the
    N-frame video alignment intact. The world-frame joints of a filled frame use
    a neighbour's pose (slightly wrong in WORLD frame only — the camera-frame
    training signal is exact); the count is surfaced by the caller, never silent.
    A sequence with NO valid pose at all is a hard failure."""
    n = cam_pose.shape[0]
    valid = np.array([_is_rigid(cam_pose[i]) for i in range(n)])
    if not valid.any():
        raise ValueError("cam_pose has no rigid frame at all (fully untracked "
                         "sequence) — refusing to convert")
    filled = cam_pose.astype(np.float64).copy()
    filled_idx = [int(i) for i in np.nonzero(~valid)[0]]
    if filled_idx:
        valid_pos = np.nonzero(valid)[0]
        for i in filled_idx:
            prev = valid_pos[valid_pos < i]
            src = prev[-1] if prev.size else valid_pos[valid_pos > i][0]
            filled[i] = cam_pose[src]
    T_cw = np.stack([np.linalg.inv(filled[i]) for i in range(n)])
    return filled, T_cw, filled_idx


def project_single_focal(j: np.ndarray, f: float, cx: float, cy: float):
    """[K,3] cam-frame m -> (u[K], v[K]) px, preprocess_hoi4d.py:438-441 form."""
    z = np.clip(j[:, 2], 1e-3, None)
    return f * j[:, 0] / z + cx, f * j[:, 1] / z + cy


def bbox_square_from_uv(u, v, inb, res: int, rf: float):
    """In-frame projected joints -> HOT3D-protocol box (train_hand_head.py:650-680):
    tight -> x rf -> square to max side -> /res -> clamp [0,1].
    Returns (clamped[4], unclamped[4]) normalized xyxy, or None if nothing in-frame."""
    if not inb.any():
        return None
    x1, x2 = float(u[inb].min()), float(u[inb].max())
    y1, y2 = float(v[inb].min()), float(v[inb].max())
    cxb, cyb = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max((x2 - x1) * rf, (y2 - y1) * rf)
    raw = np.array([(cxb - side / 2) / res, (cyb - side / 2) / res,
                    (cxb + side / 2) / res, (cyb + side / 2) / res], np.float64)
    return np.clip(raw, 0.0, 1.0), raw


# ------------------------------------------------------------------ video
def write_video(rgb: np.ndarray, dst_mp4: str, fps: float) -> None:
    """Encode [N,R,R,3] uint8 RGB -> mp4. Backend is chosen by AVAILABILITY UP
    FRONT (never by catching mid-write exceptions — imageio's FFMPEG plugin only
    probes for the ffmpeg exe lazily on the first append_data, which made an
    exception-driven fallback skip cv2 entirely and fail every sequence on
    gb10). cv2 mp4v is primary: it is the preprocess_hoi4d writer, proven in
    venv_gb10 and the backend the local validation gates exercised. imageio/
    ffmpeg (libx264 yuv420p) is used only when cv2 is absent, with the exe
    resolved eagerly. Frame count is re-verified after encoding — a muxer that
    drops frames would silently misalign every cache."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        vw = cv2.VideoWriter(dst_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (rgb.shape[2], rgb.shape[1]))
        if not vw.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {dst_mp4}")
        for fr in rgb:
            vw.write(fr[:, :, ::-1])                    # RGB -> BGR
        vw.release()
    else:
        import imageio.v2 as imageio
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()                 # loud, EARLY exe check
        writer = imageio.get_writer(dst_mp4, format="FFMPEG", mode="I", fps=fps,
                                    codec="libx264", pixelformat="yuv420p",
                                    quality=9, macro_block_size=None)
        for fr in rgb:
            writer.append_data(fr)
        writer.close()
    n_out = count_video_frames(dst_mp4)
    if n_out != rgb.shape[0]:
        raise ValueError(f"video frame count {n_out} != {rgb.shape[0]} after encode")


def count_video_frames(path: str) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    except ImportError:
        import imageio.v2 as imageio
        return imageio.get_reader(path).count_frames()


# ------------------------------------------------------------------ conversion
def convert_seq(npz_path: str, out_seq: str, rf: float, fps: float, mano) -> dict:
    """One packed npz -> the per-seq current-protocol directory (built in place;
    caller handles tmp-dir atomicity). Returns per-seq stats."""
    d = load_npz_checked(npz_path)
    n, res = d["n"], d["res"]
    cam_intr, _ = square_intrinsics(d["K"], res)
    f0, cx0, cy0 = [float(x) for x in cam_intr.tolist()]
    cam_pose_f, T_cw, filled_idx = sanitize_cam_pose(d["cam_pose"])  # world->cam
    # Bound: isolated missing poses (H2O drops the odd frame) are carried forward;
    # a large invalid fraction signals a convention/parse problem, not a dropout,
    # so refuse rather than carry-forward-fill a whole sequence's world frame.
    if len(filled_idx) > max(3, int(0.25 * n)):
        raise ValueError(f"{len(filled_idx)}/{n} cam_pose frames non-rigid "
                         f"(> 25%) — likely a convention/parse issue, not dropouts")

    j_cam = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    j_world = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    j_2d = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    bboxes = torch.zeros(n, NUM_HANDS, 4)
    valid = torch.zeros(n, NUM_HANDS, dtype=torch.bool)
    gt64 = torch.zeros(n, NUM_HANDS * HAND_PARAM_DIM)
    jl = []
    n_mano_gap = 0                                       # joints valid, mano invalid

    for fi in range(n):
        T_wc = cam_pose_f[fi]                            # cam->world (carry-forward filled)
        hp = {}
        for h in range(NUM_HANDS):
            j21, jv = hand21_from_vec(d["joints"][fi], h)
            if jv and (not np.isfinite(j21).all() or np.abs(j21).sum() < 1e-8):
                raise ValueError(f"frame {fi} hand {h}: valid flag set but joints "
                                 f"non-finite/zero — refusing to convert silently")
            if jv:
                j16 = j21[H2O16_IDX]                     # smplx-16, cam frame, m
                j_cam[fi, h] = torch.tensor(j16, dtype=torch.float32)
                jw = (T_wc[:3, :3] @ j16.T + T_wc[:3, 3:4]).T
                j_world[fi, h] = torch.tensor(jw, dtype=torch.float32)
                u16, v16 = project_single_focal(j16, f0, cx0, cy0)
                j_2d[fi, h, :, 0] = torch.tensor(u16, dtype=torch.float32)
                j_2d[fi, h, :, 1] = torch.tensor(v16, dtype=torch.float32)
                j_2d[fi, h, :, 2] = 1.0
                u21, v21 = project_single_focal(j21, f0, cx0, cy0)
                inb = (u21 >= 0) & (u21 < res) & (v21 >= 0) & (v21 < res)
                bb = bbox_square_from_uv(u21, v21, inb, res, rf)
                if bb is not None:
                    bboxes[fi, h] = torch.tensor(bb[0], dtype=torch.float32)
                    valid[fi, h] = True

            mv, trans, g_aa, pose45, beta = mano_from_vec(d["mano"][fi], h)
            if jv and not mv:
                n_mano_gap += 1
            if mv:
                q = _aa_to_quat_wxyz(g_aa)
                pose15 = (pose45_to_pca15(pose45, mano, is_right=(h == 1))
                          if mano is not None else pose45[:15])
                off = h * HAND_PARAM_DIM
                gt64[fi, off:off + 3] = torch.tensor(trans, dtype=torch.float32)
                gt64[fi, off + 3:off + 7] = torch.tensor(q, dtype=torch.float32)
                gt64[fi, off + 7:off + 22] = torch.tensor(pose15, dtype=torch.float32)
                gt64[fi, off + 22:off + 32] = torch.tensor(beta, dtype=torch.float32)
                hp[str(h)] = {
                    "wrist_xform": {"t_xyz": [float(x) for x in trans],
                                    "q_wxyz": [float(x) for x in q]},
                    "pose": [float(x) for x in pose15],
                    "betas": [float(x) for x in beta],
                }
        jl.append({"timestamp_ns": fi, "hand_poses": hp})

    hd = os.path.join(out_seq, "hand_data")
    os.makedirs(hd, exist_ok=True)
    write_video(d["rgb"], os.path.join(out_seq, "video_main_rgb.mp4"), fps)
    torch.save(cam_intr, os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.from_numpy(T_cw).float(), os.path.join(hd, "cam_extrinsics_cache.pt"))
    with open(os.path.join(hd, "mano_hand_pose_trajectory.jsonl"), "w") as fjl:
        for e in jl:
            fjl.write(json.dumps(e) + "\n")
    torch.save(j_world, os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(j_cam, os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(j_2d, os.path.join(hd, "gt_joints_2d_cache.pt"))
    torch.save({"bboxes": bboxes, "valid": valid, "gt": gt64},
               os.path.join(hd, bbox_cache_name(rf, res)))
    if n_mano_gap:
        print(f"  [warn] {n_mano_gap} (frame,hand) with valid joints but invalid "
              f"MANO — param-loss target is zeros there (kp3d caches unaffected)")
    if filled_idx:
        print(f"  [warn] {len(filled_idx)} frame(s) had a missing/zero cam_pose "
              f"(idx {filled_idx[:8]}{'...' if len(filled_idx) > 8 else ''}); "
              f"carried forward the nearest valid pose (WORLD cache only — camera-"
              f"frame caches exact; frame/video alignment preserved)")
    return {"frames": n, "res": res,
            "valid_L": int(valid[:, 0].sum()), "valid_R": int(valid[:, 1].sum()),
            "mano_gap": n_mano_gap, "cam_pose_filled": len(filled_idx)}


def bbox_cache_name(rf: float, res: int) -> str:
    return f"hand_bboxes_v2_rf{rf}_res{res}x{res}.pt"


def seq_is_complete(out_seq: str, rf: float, res: int) -> bool:
    if not os.path.exists(os.path.join(out_seq, "video_main_rgb.mp4")):
        return False
    hd = os.path.join(out_seq, "hand_data")
    need = CACHE_FILES + [bbox_cache_name(rf, res)]
    return all(os.path.exists(os.path.join(hd, f)) for f in need)


# ------------------------------------------------------------------ validation
class GateFailure(Exception):
    pass


def _gate(name: str, ok: bool, detail: str):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        raise GateFailure(f"{name}: {detail}")


def validate_seq(npz_path: str, out_seq: str, rf: float, mano) -> None:
    """Run the 4 mandatory gates for one converted sequence. Raises GateFailure."""
    d = load_npz_checked(npz_path)
    n, res = d["n"], d["res"]
    hd = os.path.join(out_seq, "hand_data")
    cam_intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), weights_only=True)
    T_cw = torch.load(os.path.join(hd, "cam_extrinsics_cache.pt"), weights_only=True).double().numpy()
    j_cam = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), weights_only=True)
    j_world = torch.load(os.path.join(hd, "gt_joints_cache_world.pt"), weights_only=True)
    j_2d = torch.load(os.path.join(hd, "gt_joints_2d_cache.pt"), weights_only=True)
    bb = torch.load(os.path.join(hd, bbox_cache_name(rf, res)), weights_only=True)
    valid = bb["valid"].bool()
    f0, cx0, cy0 = [float(x) for x in cam_intr.tolist()]
    _, (fx_s, fy_s, cx_s, cy_s) = square_intrinsics(d["K"], res)

    # --- gate 0: our remap constants == scripts.eval_cmpjpe's (heavy import;
    # available on-cluster where validation runs).
    from scripts.eval_cmpjpe import H2O_TO_MANO as REF_A
    from scripts.eval_cmpjpe import h2o_gt_joints16
    _gate("gate0 remap constants", H2O_TO_MANO == REF_A and H2O16_IDX == REF_A[:16],
          "match scripts.eval_cmpjpe")

    sel = valid.numpy()
    if not sel.any():
        raise GateFailure("no valid hands in sequence — nothing to validate")

    # --- gate 1: reprojection.
    jc = j_cam.double().numpy()
    z = np.clip(jc[..., 2], 1e-3, None)
    u1 = f0 * jc[..., 0] / z + cx0                       # single focal (protocol)
    v1 = f0 * jc[..., 1] / z + cy0
    d_int = np.hypot(u1 - j_2d[..., 0].numpy(), v1 - j_2d[..., 1].numpy())[sel]
    _gate("gate1a 2D cache internal", float(np.median(d_int)) < 1e-3,
          f"median {np.median(d_int):.2e} px (vs stored gt_joints_2d_cache)")
    u2 = fx_s * jc[..., 0] / z + cx_s                    # true fx/fy, eval_cmpjpe form
    v2 = fy_s * jc[..., 1] / z + cy_s
    d_true = np.hypot(u1 - u2, v1 - v2)[sel]
    _gate("gate1b vs true fx/fy projection", float(np.median(d_true)) < 1.0,
          f"median {np.median(d_true):.3f} px (fx/fy anisotropy + crop math)")

    # --- gate 2a: cam cache == eval_cmpjpe's remapped GT.
    ref16, ref_valid = h2o_gt_joints16(torch.from_numpy(d["joints"]).float())  # [N,2,16,3]
    m = (ref_valid > 0.5).numpy()
    dj = (ref16 - j_cam).abs().max(dim=-1).values.max(dim=-1).values.numpy()[m]
    _gate("gate2a cam cache vs eval_cmpjpe GT", float(dj.max()) < 1e-4,
          f"max |diff| {dj.max():.2e} m over {int(m.sum())} valid (frame,hand)")
    _gate("gate2a valid flags", bool((sel <= m).all()),
          "every cache-valid hand is H2O-valid (cache may drop out-of-frame hands)")

    # --- gate 4: anatomical joint order. Reprojection/round-trip gates are blind to
    # joint ORDER (a scrambled cache projects to a scrambled 2D cache and round-trips
    # to itself). Median wrist->MCP distances must be anatomically plausible for the
    # smplx-16 layout: index/middle/pinky/ring MCPs ~75-100 mm, thumb CMC ~30-50 mm.
    # The 2026-07-18 scramble measured 89/125/123/40/157 mm here and would fail.
    MCP_GATES = {1: ("index MCP", 55.0, 115.0), 4: ("middle MCP", 60.0, 120.0),
                 7: ("pinky MCP", 45.0, 100.0), 10: ("ring MCP", 50.0, 110.0),
                 13: ("thumb CMC", 15.0, 65.0)}
    for h in range(NUM_HANDS):
        hv = sel[:, h]
        if hv.sum() < 10:
            continue
        for jix, (jname, lo, hi) in MCP_GATES.items():
            dist = float(np.median(np.linalg.norm(
                jc[hv, h, jix] - jc[hv, h, 0], axis=-1))) * 1000.0
            _gate(f"gate4 bone hand{h} {jname}", lo <= dist <= hi,
                  f"median wrist->{jname} {dist:.1f} mm (expected {lo:.0f}-{hi:.0f})")

    # --- gate 2b: GT-as-prediction through OUR eval loading path == ~0.
    from scripts.train_hand_head import HOT3DHandDataset
    from scripts.world_space_metrics import c_mpjpe
    clip_len = min(16, n)
    ds = HOT3DHandDataset([out_seq], None, num_frames=clip_len,
                          clip_stride=max(clip_len // 2, 1),
                          use_hand_crop=True, rescale_factor=rf, res=(res, res))
    _gate("gate2b dataset load", len(ds) > 0 and ds.clips[0]["n_video"] == n,
          f"{len(ds)} clips, n_video={ds.clips[0]['n_video'] if len(ds) else '-'} (N={n})")
    first = ds[0]                                        # exercises video decode
    _gate("gate2b video decode", tuple(first["img"].shape) == (clip_len, 3, res, res),
          f"first clip img {tuple(first['img'].shape)}")
    pcf = {}
    for j, clip in enumerate(ds.clips):                  # eval_hand_cam_anchor dedupe
        start = clip["frame_offset"]
        for kk in range(clip_len):
            fidx = start + kk
            if fidx not in pcf and fidx < n:
                pcf[fidx] = clip["gt_joints"][kk]        # [2,16,3] "prediction" = GT
    fr = sorted(pcf)
    errs = []
    for h in range(NUM_HANDS):
        vc = valid[fr, h].unsqueeze(-1).expand(-1, NUM_JOINTS)
        if vc.any():
            pred = torch.stack([pcf[fx_][h] for fx_ in fr])
            errs.append(float(c_mpjpe(pred, j_cam[fr, h], valid=vc, root_relative=False)))
            errs.append(float(c_mpjpe(pred, j_cam[fr, h], valid=vc, root_relative=True)))
    _gate("gate2b GT-as-prediction C-MPJPE", bool(errs) and max(errs) < 1e-3,
          f"max {max(errs) if errs else float('nan'):.2e} mm over {len(fr)} frames")

    # --- gate 3: bbox sanity (recompute from npz, compare to stored).
    worst_dev, worst_sq, n_checked, contained = 0.0, 0.0, 0, True
    for fi in range(n):
        for h in range(NUM_HANDS):
            if not sel[fi, h]:
                continue
            j21, _ = hand21_from_vec(d["joints"][fi], h)
            u21, v21 = project_single_focal(j21, f0, cx0, cy0)
            inb = (u21 >= 0) & (u21 < res) & (v21 >= 0) & (v21 < res)
            rec = bbox_square_from_uv(u21, v21, inb, res, rf)
            if rec is None:
                raise GateFailure(f"bbox: frame {fi} hand {h} valid but no in-frame joints")
            clamped, raw = rec
            stored = bb["bboxes"][fi, h].double().numpy()
            worst_dev = max(worst_dev, float(np.abs(stored - clamped).max()))
            worst_sq = max(worst_sq, abs((raw[2] - raw[0]) - (raw[3] - raw[1])))
            x1, y1, x2, y2 = stored
            if not (-1e-6 <= x1 <= x2 <= 1 + 1e-6 and -1e-6 <= y1 <= y2 <= 1 + 1e-6):
                raise GateFailure(f"bbox outside [0,1]: frame {fi} hand {h} {stored}")
            uin, vin = u21[inb] / res, v21[inb] / res
            eps = 1e-6
            if ((uin < x1 - eps) | (uin > x2 + eps) | (vin < y1 - eps) | (vin > y2 + eps)).any():
                contained = False
            n_checked += 1
    _gate("gate3 bbox recompute", worst_dev < 1e-5,
          f"max |stored-recomputed| {worst_dev:.2e} over {n_checked} boxes")
    _gate("gate3 bbox square (pre-clamp)", worst_sq < 1e-9,
          f"max |w-h| {worst_sq:.2e} normalized")
    _gate("gate3 bbox contains joints", contained, "all in-frame projected joints inside box")

    # --- gate 4: extrinsics direction round-trip. Sanitize d["cam_pose"] the
    # SAME way convert_seq did (carry-forward isolated missing poses) so the
    # stored T_cw is checked against the pose it was actually inverted from —
    # otherwise a dropped frame's raw zero would false-fail T_cw @ cam_pose == I.
    cam_pose_f, _, filled_idx = sanitize_cam_pose(d["cam_pose"])
    eye_err = float(np.abs(np.einsum("nij,njk->nik", T_cw, cam_pose_f)
                           - np.eye(4)).max())
    _gate("gate4 T_cw @ cam_pose == I", eye_err < 1e-4,
          f"max |dev| {eye_err:.2e} ({len(filled_idx)} frame(s) carried-forward)")
    jc4 = j_cam.double().numpy()
    R_wc, t_wc = cam_pose_f[:, :3, :3], cam_pose_f[:, :3, 3]
    jw = np.einsum("nij,nhkj->nhki", R_wc, jc4) + t_wc[:, None, None, :]
    dw = np.abs(jw - j_world.double().numpy()).max(-1)[sel]
    _gate("gate4 world cache == lifted cam cache", float(dw.max()) < 1e-5,
          f"max |diff| {dw.max():.2e} m")
    jr = np.einsum("nij,nhkj->nhki", T_cw[:, :3, :3], jw) + T_cw[:, None, None, :3, 3]
    dr = np.abs(jr - jc4).max(-1)[sel]
    _gate("gate4 cam->world->cam round-trip", float(dr.max()) < 1e-5,
          f"max |diff| {dr.max():.2e} m")
    t_mag = float(np.median(np.linalg.norm(cam_pose_f[:, :3, 3], axis=-1)))
    print(f"  [info] median |cam_pose t| = {t_mag:.3f} (metres expected; >100 suspicious)")

    # --- informational: MANO forward kinematics vs GT joints (risks 1+2).
    if mano is not None:
        errs_full, errs_pca = [], []
        step = max(n // 8, 1)
        gt64 = bb["gt"]
        for fi in range(0, n, step):
            for h in range(NUM_HANDS):
                mv, trans, g_aa, pose45, beta = mano_from_vec(d["mano"][fi], h)
                if not (mv and sel[fi, h]):
                    continue
                jf = mano.get_joints_full_pose(g_aa, pose45, beta, trans, is_right=(h == 1))
                errs_full.append(1000 * np.linalg.norm(
                    np.asarray(jf).reshape(-1, 3)[:NUM_JOINTS] - j_cam[fi, h].numpy(),
                    axis=-1).mean())
                p32 = gt64[fi, h * HAND_PARAM_DIM:(h + 1) * HAND_PARAM_DIM].view(1, -1)
                jp = mano.get_joints_batched(p32, is_right=(h == 1), device="cpu")
                errs_pca.append(1000 * np.linalg.norm(
                    jp[0].detach().numpy() - j_cam[fi, h].numpy(), axis=-1).mean())
        if errs_full:
            mf, mp = float(np.mean(errs_full)), float(np.mean(errs_pca))
            tag = "" if mf < 15.0 else "  [WARN >15mm: H2O MANO trans/aa convention off — mano-param losses unreliable; kp3d caches unaffected]"
            print(f"  [mano] full-45 fwd vs GT joints: {mf:.1f} mm | jsonl PCA15 path: {mp:.1f} mm{tag}")
        else:
            print("  [mano] no valid MANO frames sampled — convention unverified")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h2o", required=True, help="dir of packed H2O .npz (scripts.pack_h2o)")
    ap.add_argument("--out", required=True, help="output root for per-seq dirs")
    ap.add_argument("--mano_model", default=None,
                    help="models/MANO folder for the 45->15 PCA jsonl pose + the "
                         "--validate MANO check; omitted -> pose truncation (flagged)")
    ap.add_argument("--rescale_factor", type=float, default=1.5)
    ap.add_argument("--fps", type=float, default=30.0, help="H2O ego is 30 fps")
    ap.add_argument("--max_seqs", type=int, default=0,
                    help="cap NEWLY converted sequences this run (0 = no cap); "
                         "resume-by-skipping makes batched runs safe on tight quota")
    ap.add_argument("--only", default=None, help="substring filter on npz basename")
    ap.add_argument("--validate", action="store_true",
                    help="run the 4 mandatory gates on every seq touched this run")
    ap.add_argument("--validate_only", action="store_true",
                    help="skip conversion; gate-check already-converted seqs")
    ap.add_argument("--tolerate_convert_failures", type=int, default=0,
                    help="exit 0 if at most N sequences fail to CONVERT (e.g. bad "
                         "extrinsics / >25%% non-rigid cam_pose) and NO gate fails; "
                         "the failed seqs are simply dropped from the store. Gate "
                         "failures on produced seqs stay always fatal. The camera "
                         "-frame C_abs protocol does not use extrinsics, so dropping "
                         "extrinsics-broken takes is safe for that eval.")
    args = ap.parse_args()

    mano = None
    if args.mano_model:
        from scripts.hand_vis_utils import MANOModel
        mano = MANOModel(args.mano_model)
    elif not args.validate_only:
        print("[warn] no --mano_model: jsonl 'pose' = 45->15 TRUNCATION (mano-param "
              "losses degraded; kp3d/kp3d_abs/kp2d caches unaffected)")

    npz_files = sorted(glob.glob(os.path.join(args.h2o, "*.npz")))
    if args.only:
        npz_files = [f for f in npz_files if args.only in os.path.basename(f)]
    if not npz_files:
        raise SystemExit(f"no .npz under {args.h2o} (filter={args.only})")
    os.makedirs(args.out, exist_ok=True)

    converted, skipped, failed, touched = 0, 0, [], []
    for npz_path in npz_files:
        name = os.path.basename(npz_path)[:-4]
        out_seq = os.path.join(args.out, name)
        tmp_seq = os.path.join(args.out, f".tmp_{name}")
        if args.validate_only:
            if seq_is_complete(out_seq, args.rescale_factor, _peek_res(npz_path)):
                touched.append((npz_path, out_seq))
            else:
                failed.append((name, "not converted (validate_only found no complete dir)"))
            continue
        if seq_is_complete(out_seq, args.rescale_factor, _peek_res(npz_path)):
            skipped += 1
            continue
        if args.max_seqs and converted >= args.max_seqs:
            continue
        print(f"[convert] {name}")
        if os.path.isdir(tmp_seq):
            shutil.rmtree(tmp_seq)                      # stale partial from a kill
        if os.path.isdir(out_seq):
            shutil.rmtree(out_seq)                      # incomplete final dir
        try:
            stats = convert_seq(npz_path, tmp_seq, args.rescale_factor, args.fps, mano)
            os.rename(tmp_seq, out_seq)                 # atomic completion marker
            converted += 1
            touched.append((npz_path, out_seq))
            print(f"  done: {stats['frames']} frames @ {stats['res']}px, "
                  f"valid L/R = {stats['valid_L']}/{stats['valid_R']}")
        except Exception as e:                          # keep batch going, fail loudly at exit
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  [FAILED] {name}: {type(e).__name__}: {e}")
            if os.path.isdir(tmp_seq):
                shutil.rmtree(tmp_seq)

    gate_failed = []
    if args.validate or args.validate_only:
        for npz_path, out_seq in touched:
            print(f"[validate] {os.path.basename(out_seq)}")
            try:
                validate_seq(npz_path, out_seq, args.rescale_factor, mano)
            except GateFailure as e:
                gate_failed.append((os.path.basename(out_seq), str(e)))
                print(f"  [GATE FAILED] {e}")

    print(f"\n=== h2o_to_currentproto: {converted} converted, {skipped} skipped "
          f"(complete), {len(failed)} failed, {len(gate_failed)} gate failures ===")
    for nm, msg in failed:
        print(f"  convert-FAIL {nm}: {msg}")
    for nm, msg in gate_failed:
        print(f"  gate-FAIL   {nm}: {msg}")
    if gate_failed or len(failed) > args.tolerate_convert_failures:
        raise SystemExit(1)
    if failed:
        print(f"[TOLERATED] {len(failed)} convert failure(s) <= "
              f"--tolerate_convert_failures={args.tolerate_convert_failures}; "
              f"dropped from store, exiting 0.")


def _peek_res(npz_path: str) -> int:
    with np.load(npz_path, mmap_mode="r") as d:
        return int(d["rgb"].shape[1])


if __name__ == "__main__":
    main()
