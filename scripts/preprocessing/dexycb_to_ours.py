"""DexYCB (NVIDIA, CVPR 2021) -> our HOI4D-style per-sequence hand store.

THE TARGET FORMAT IS DERIVED FROM THE CONSUMER, not invented. Everything below is what
scripts/train_hand_head.py's HOT3DHandDataset reads off a sequence directory on the cache-HIT
path. Line numbers are that file as of 2026-08-04 and it is under active edit, so each one is
paired with the text it points at - re-find by the quote if the number has drifted.

    <out_root>/<subject>_<capture>_<serial>/
        video_main_rgb.mp4                            REQUIRED, seq skipped without it
                                                      (:281 `video_path = ...`, :301 the skip)
        hand_data/cam_intrinsics.pt                   [3] float32 = [f, cx, cy]
                                                      (:375, and :408-410 which loads it even when
                                                      the 2D/extrinsics caches are absent)
        hand_data/cam_extrinsics_cache.pt             [N,4,4] float32 T_camera_world = w2c
                                                      (:374; the direction is pinned by :760
                                                      `cam_extr = clip["cam_extrinsics"]  # [S,4,4]
                                                      T_cam_world (validated w2c)`)
        hand_data/gt_joints_2d_cache.pt               [N,2,16,3] float32 (u, v, conf) px
                                                      (:373, sliced at :522 `# [S, 2, 16, 3]`)
        hand_data/gt_joints_cache_cam_v2.pt           [N,2,16,3] float32 CAMERA frame, METRES
                                                      (:288-292 picks the _cam_v2 name whenever
                                                      use_hand_crop is on, which is our protocol)
        hand_data/gt_joints_cache_world.pt            [N,2,16,3] float32 world frame, metres
        hand_data/mano_hand_pose_trajectory.jsonl     optional once the .pt caches exist (:294-307);
                                                      written anyway so a fresh rf/res cache-miss can
                                                      rebuild, exactly as arctic/oakink2/h2o do
        hand_data/hand_bboxes_v2_rf1.5_res224x224.pt  {"bboxes": [N,2,4] normalised xyxy,
                                                       "valid":  [N,2] bool,
                                                       "gt":     [N,64] float32 camera-frame MANO}
                                                      name = f"hand_bboxes_v2_rf{rescale_factor}_
                                                      res{res[0]}x{res[1]}.pt" (:436-437), so the
                                                      "res224x224" is the TRAINING crop res, not the
                                                      video res - the boxes themselves are normalised
                                                      and resolution-free. Required keys per
                                                      scripts/verify_box_store.py:32.

SCHEMA DECISIONS, each traced to source rather than assumed
  * Hand axis: index 0 = LEFT, 1 = RIGHT (train_hand_head.py:861 '"0" = left, "1" = right').
    DexYCB captures are SINGLE-HAND (meta.yml `mano_sides` has one entry), so the other slot is
    filled with the ABSENT-HAND representation the consumer expects: ZEROS everywhere and
    valid=False. NOT NaN. This is load-bearing: Keypoint3DLoss multiplies the residual by the
    per-joint confidence (scripts/hamer_losses.py:105), and NaN * 0 = NaN, so a NaN-filled absent
    hand poisons every gradient in the batch. preprocess_hoi4d.py:372-377 (also a single-hand
    dataset) and train_hand_head.py:569-570 (`if p.abs().sum() < 1e-6: j3d = np.zeros((16, 3))`)
    both use zeros; arctic_to_ours/oakink2_to_ours fill NaN instead, which is a latent hazard on
    those stores, not a precedent to copy.
  * Joint order: DexYCB's 21 joints are the manopth ManoLayer output order, documented verbatim as
    dex_ycb_toolkit/dex_ycb.py `_MANO_JOINTS` = [wrist, thumb mcp/pip/dip/tip, index mcp/pip/dip/tip,
    middle ..., ring ..., little ...] and produced by manopth/manolayer.py:260's reorder
    `[0,13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20]`. Our 16-joint layout is
    [wrist, index x3, middle x3, pinky x3, ring x3, thumb x3] (scripts/egoexo4d_to_ours.py:48-49).
    The resulting map is DEXYCB21_TO_MANO16 below, and it is BIT-IDENTICAL to H2O's
    H2O16_IDX = H2O_TO_MANO[:16] because H2O publishes the same OpenPose-style hand order; that
    equality is asserted at import time so the 2026-07-18 H2O scramble (which corrupted every H2O
    number until anatomical bone lengths caught it) cannot silently repeat here.
  * Units: METRES. dex_ycb_toolkit/layers/mano_layer.py:60-62 divides the manopth output by 1000
    (manopth/manolayer.py:272-273 scales to mm), and hpe_eval.py multiplies label['joint_3d'] by
    1000 to reach the mm the FreiHAND evaluator wants - so the stored joint_3d is metres. Gate 5
    re-measures this on real data (a metre-scale hand is ~0.1 m wrist-to-knuckle, a mm-scale one
    would be ~100), so the unit is verified, not trusted.
  * Joints source: the per-frame labels_%06d.npz['joint_3d'], which is [n_hands,21,3] in the
    CAMERA frame of that camera, metres, with an all -1 sentinel for "no annotation"
    (hpe_eval.py:70-74 `if np.all(joint_3d == -1): continue`). Using it instead of MANO-FK'ing
    pose.npz removes both the MANO-convention risk and the extrinsic-direction risk from the
    camera-frame cache, which is the cache every headline metric (C_abs / C_rr) reads.
  * Extrinsics: calibration/extrinsics_<id>/extrinsics.yml maps CAMERA -> WORLD. Concluded from
    dex_ycb_toolkit/sequence_loader.py:_deproject_depth_and_filter_points, which back-projects a
    depth image into camera-frame points and then does
        p = torch.addmm(self._t[c].unsqueeze(1), self._R[c], p)      # p_world = R_c p_cam + t_c
    with R_c, t_c read straight from that yml, and confirmed by transform_ycb's
    `if camera_to_world: R_c = self._R[c]` branch. Our cache stores the INVERSE (T_camera_world,
    w2c). Gate 2 re-derives this from the data instead of trusting the reading: two cameras of the
    same capture must map their own joint_3d to the SAME world point, which only happens under the
    correct direction, and it refuses to convert if the flipped hypothesis wins.
    The yml holds one entry per camera plus an 'apriltag' entry (9 lines for the 8-camera rig), and
    the master camera's entry is the identity - i.e. "world" IS the master camera frame.
  * MANO params (gt64, the ParameterLoss target only): labels_%06d.npz['pose_m'] is [n_hands,51] in
    the CAMERA frame - visualize_pose.py:63-77 feeds it straight to ManoLayer and renders with the
    camera at the identity pose. Layout is [0:3] global axis-angle, [3:48] 45 PCA coefficients
    (ManoLayer(use_pca=True, ncomps=45, flat_hand_mean=False)), [48:51] translation in metres.
    Our pose15 slot takes coefficients [3:18] DIRECTLY: our pose45_to_pca15 computes
    (p45 - hand_mean) @ pinv(components)[:, :15], and for p45 = hand_mean + c @ components with a
    full-rank 45x45 basis that collapses to exactly c[:15]. This holds only if manopth and smplx
    read the same `hands_components` matrix out of the same MANO pkl, which they do, but it is an
    ASSUMPTION not a measurement - flagged here because it is the one number in the store that no
    gate below can check. It feeds ParameterLoss's `hand_pose` term only; kp3d / kp3d_abs / C_abs
    read the joint caches and are unaffected.
  * 2D cache: single-focal pinhole u = f x/z + cx, the preprocess_hoi4d.py:438-443 convention.
    NOTE the kp2d loss path (train_hand_head.py:2600, `IMAGE_WIDTH = 1408.0`, and its val twin at
    :1469) hardcodes the Aria 1408 px frame and a 90-degree rotation, so kp2d must stay DISABLED
    for DexYCB exactly as it is for HOI4D and H2O.
  * Boxes: scripts.arctic_to_ours.joints_to_bbox, rf 1.5, RECTANGULAR per-axis, UNCLAMPED. Never
    reimplemented locally: the square+clamped variant silently cost ~290-380 mm of C-abs on H2O
    (scripts/verify_box_store.py:5-9). Run verify_box_store.py on the output before training.
  * Video: 8 static RGB-D views per capture, so ONE output sequence per (capture, camera serial).
    640x480 is centre-square-cropped to 480x480 with cx shifted by the crop offset, following the
    two most carefully built stores in the repo (preprocess_hoi4d 1920x1080 -> 1080^2, h2o
    1280x720 -> 720^2). The alternative - keeping 4:3 and letting load_video squash it to 224x224 -
    would feed the head anisotropically distorted crops that no other store has. The cost is real
    and is MEASURED, not hidden: `frac_hand_cropped_out` is printed per sequence, and --no_square_crop
    turns the crop off.

VALIDATION GATES (all run during a normal conversion; --validate runs them and writes nothing)
  1. fx/fy anisotropy: our store format is single-focal, so the fy we discard must be worth less
     than --max_aniso_px of vertical error over the observed joints, else refuse.
  2. Extrinsic direction (once per extrinsics id, needs >= 2 cameras): cross-camera world
     agreement under the camera->world hypothesis must beat the flipped one and be under 2 cm.
  3. Anatomy: median wrist->knuckle bone lengths in 2-15 cm, each phalanx beyond it inside its own
     band, bones plausibly shortening outward along each finger, and no joint further than 30 cm
     from the wrist. This is the check that caught the H2O 21->16 scramble; a mapping error fails
     it on essentially every sequence, so a failure rate above --max_gate_fail aborts the whole run
     rather than quietly writing a partial store.
  4. Frame alignment: exactly one written video frame per cache row, re-counted after encoding.
  5. Units + transl semantics: joint_3d[wrist] - pose_m[48:51] must be CONSTANT across a sequence.
     manopth leaves the root joint at its shaped-template position and only then adds trans, so
     wrist = root_j(betas) + trans with root_j pose-independent; in a camera frame that difference
     is R_w2c @ root_j, which is fixed BECAUSE DEXYCB'S CAMERAS ARE STATIC (do not lift this gate
     to a moving-camera dataset unchanged). A non-constant difference means pose_m and joint_3d
     are not in the same frame; a difference ~1000x too large (~90 instead of ~0.09) means the
     labels are millimetres and every metric depth in the store would be wrong.

Usage:
    python -m scripts.preprocessing.dexycb_to_ours \
        --dexycb_root /cluster/scratch/dmonopoli/dexycb \
        --out_root /cluster/scratch/dmonopoli/dexycb_ours \
        --cameras 932122060857,836212060125 --limit 1        # smoke run: one sequence
    python -m scripts.preprocessing.dexycb_to_ours --dexycb_root ... --out_root ... --validate
Then, before any training on the result:
    python -m scripts.verify_box_store --data_root /cluster/scratch/dmonopoli/dexycb_ours
    python -m scripts.preprocessing.dexycb_depth_stat --store /cluster/scratch/dmonopoli/dexycb_ours
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np
import torch
import yaml

from scripts.arctic_to_ours import joints_to_bbox, project_single_focal
from scripts.preprocessing.preprocess_hoi4d import _aa_to_quat_wxyz

NUM_HANDS = 2
NUM_JOINTS = 16
HAND_PARAM_DIM = 32          # [transl 3, quat_wxyz 4, pose15 (PCA) 15, betas 10]
DEXYCB_W, DEXYCB_H = 640, 480          # colour stream, dex_ycb.py:_h/_w
DEXYCB_NJ = 21

# DexYCB 21 (OpenPose hand order, dex_ycb.py:_MANO_JOINTS) -> our smplx-16
# [wrist, index x3, middle x3, pinky x3, ring x3, thumb x3]. See the docstring for the
# derivation; the assert below pins it against the H2O mapping that the same order produced.
DEXYCB21_TO_MANO16 = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
# The tripwire must compare against the ACTUAL H2O constant, not a local copy of it. It used to
# be `_H2O16_IDX = [ ...same literal... ]`, which compares the literal to itself and can never
# fire - a disabled guard for exactly the joint-scramble class it was written to catch.
try:
    from scripts.preprocessing.h2o_to_currentproto import H2O16_IDX as _H2O16_IDX
except ImportError:  # allow running this converter without the H2O module present
    _H2O16_IDX = None
if _H2O16_IDX is not None and list(DEXYCB21_TO_MANO16) != list(_H2O16_IDX):
    # a raise, not an assert: -O must not strip this
    raise RuntimeError(
        f"DexYCB joint remap drifted from the verified H2O one:\n"
        f"  dexycb_to_ours.DEXYCB21_TO_MANO16 = {list(DEXYCB21_TO_MANO16)}\n"
        f"  h2o_to_currentproto.H2O16_IDX     = {list(_H2O16_IDX)}")

# Slot layout of our 16 joints, used by the anatomical gate.
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
MAX_JOINT_RADIUS_M = 0.30    # no hand joint sits further than 30 cm from its own wrist
BONE_MIN_M, BONE_MAX_M = 0.02, 0.15   # egoexo4d_to_ours.py:237 band for wrist->knuckle
# Bands for the two phalanges beyond the knuckle. The wrist->knuckle band alone is NOT enough:
# a remap that puts a fingertip in the middle slot leaves slot 1 correct and only shows up as an
# over-long second segment (0.085 m where a proximal phalanx is ~0.04). Generous by ~2x against
# real anatomy so a small hand cannot false-fail.
SEG1_BAND = (0.015, 0.070)
SEG2_BAND = (0.010, 0.060)

CACHE_FILES = ("cam_intrinsics.pt", "cam_extrinsics_cache.pt", "gt_joints_2d_cache.pt",
               "gt_joints_cache_cam_v2.pt", "gt_joints_cache_world.pt",
               "mano_hand_pose_trajectory.jsonl")


class EmptySequence(ValueError):
    """No frame of this (capture, camera) has a usable hand.

    Kept distinct from a gate failure on purpose. A gate failure means the DATA is wrong and a
    high rate of them means our joint mapping is wrong, which must abort the run. An empty
    sequence just means this camera never saw the hand: writing it would add clips that consume
    sampler weight and produce zero gradient, so it is skipped, but it says nothing about the
    mapping and must not count toward the abort threshold."""


# ------------------------------------------------------------------ calibration
def load_intrinsics(calib_dir: str, serial: str) -> dict:
    """calibration/intrinsics/<serial>_640x480.yml -> the ['color'] block.

    RealSense naming: fx, fy, ppx, ppy (dex_ycb.py:196-201 reads exactly intr['color'])."""
    p = os.path.join(calib_dir, "intrinsics", f"{serial}_{DEXYCB_W}x{DEXYCB_H}.yml")
    with open(p) as f:
        intr = yaml.load(f, Loader=yaml.FullLoader)["color"]
    for k in ("fx", "fy", "ppx", "ppy"):
        if k not in intr:
            raise ValueError(f"{p}: intrinsics block has no '{k}'")
    return intr


def load_extrinsics(calib_dir: str, extr_id: str) -> tuple[dict, str]:
    """calibration/extrinsics_<id>/extrinsics.yml -> ({serial: [3,4] camera->world}, master).

    The file carries one 12-number row-major 3x4 per camera PLUS an 'apriltag' row (9 rows for
    the 8-camera rig), and a separate 'master' key naming the camera whose frame IS the world
    frame - its own row is therefore the identity. Both facts are checked by the caller rather
    than assumed, because a silently transposed or inverted extrinsic produces a world cache
    that looks plausible and is wrong."""
    p = os.path.join(calib_dir, f"extrinsics_{extr_id}", "extrinsics.yml")
    with open(p) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    T = {k: np.asarray(v, np.float64).reshape(3, 4) for k, v in d["extrinsics"].items()}
    return T, str(d.get("master", ""))


def load_mano_betas(calib_dir: str, mano_calib_id: str) -> np.ndarray:
    """calibration/mano_<id>/mano.yml -> [10] betas (dex_ycb.py:230-235)."""
    p = os.path.join(calib_dir, f"mano_{mano_calib_id}", "mano.yml")
    with open(p) as f:
        return np.asarray(yaml.load(f, Loader=yaml.FullLoader)["betas"], np.float32).reshape(10)


def rt34_to_w2c(rt: np.ndarray) -> np.ndarray:
    """[3,4] CAMERA->WORLD -> [4,4] WORLD->CAMERA, our cam_extrinsics_cache convention."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], T[:3, 3] = rt[:, :3], rt[:, 3]
    return np.linalg.inv(T)


# ------------------------------------------------------------------ raw frame access
def label_path(cam_dir: str, i: int) -> str:
    return os.path.join(cam_dir, f"labels_{i:06d}.npz")


def load_label(cam_dir: str, i: int) -> tuple[np.ndarray, np.ndarray]:
    """labels_%06d.npz -> (joint_3d [n_hands,21,3] camera-frame metres, pose_m [n_hands,51]).

    joint_3d uses an all -1 sentinel for an unannotated hand (hpe_eval.py:70-74); pose_m uses
    all-zeros (sequence_loader.py:213 `np.any(mano_pose != 0.0, axis=2)`). Both are handled by
    the caller, which must not confuse "sentinel" with "a hand at the origin"."""
    d = np.load(label_path(cam_dir, i))
    j3 = np.asarray(d["joint_3d"], np.float64).reshape(-1, DEXYCB_NJ, 3)
    pm = np.asarray(d["pose_m"], np.float64).reshape(-1, 51)
    return j3, pm


def hand_is_annotated(j21: np.ndarray) -> bool:
    """True when this hand's joint_3d row is a real annotation and not the -1 sentinel."""
    return bool(np.isfinite(j21).all() and not np.all(j21 == -1))


# ------------------------------------------------------------------ gates
def anatomy_report(j16: np.ndarray, ok: np.ndarray) -> dict:
    """Per-finger median bone lengths + max joint radius over the valid frames of one hand.

    j16 [N,16,3] metres, ok [N] bool. Slot k = 1 + 3*fi is finger fi's knuckle, k+1 / k+2 the
    next two joints outward. A scrambled 21->16 remap moves a fingertip into a knuckle slot and
    shows up here immediately - which is exactly how the H2O scramble was caught."""
    out = {"n": int(ok.sum())}
    if not ok.any():
        return out
    j = j16[ok]
    out["max_radius_m"] = float(np.linalg.norm(j - j[:, :1], axis=-1).max())
    for fi, fname in enumerate(FINGERS):
        k = 1 + 3 * fi
        seg = [float(np.median(np.linalg.norm(j[:, a] - j[:, b], axis=-1)))
               for a, b in ((k, 0), (k + 1, k), (k + 2, k + 1))]
        out[fname] = [round(s, 4) for s in seg]
    return out


def anatomy_gate(rep: dict) -> str | None:
    """None when the hand is anatomically plausible, else the reason it is not."""
    if rep.get("n", 0) == 0:
        return None                                   # nothing annotated: nothing to gate
    if rep["max_radius_m"] > MAX_JOINT_RADIUS_M:
        return (f"a joint sits {rep['max_radius_m']:.3f} m from its wrist "
                f"(> {MAX_JOINT_RADIUS_M} m): joints are scrambled, in the wrong frame, "
                f"or not in metres")
    for fname in FINGERS:
        b0, b1, b2 = rep[fname]
        if not (BONE_MIN_M < b0 < BONE_MAX_M):
            return f"{fname} wrist->knuckle {b0:.3f} m outside {BONE_MIN_M}-{BONE_MAX_M} m"
        if not (SEG1_BAND[0] < b1 < SEG1_BAND[1]):
            return f"{fname} proximal phalanx {b1:.3f} m outside {SEG1_BAND} m"
        if not (SEG2_BAND[0] < b2 < SEG2_BAND[1]):
            return f"{fname} middle phalanx {b2:.3f} m outside {SEG2_BAND} m"
        # Bones shorten outward along a finger (metacarpal > proximal > middle phalanx). 25%
        # slack absorbs real anatomy and fit noise. The THUMB is excluded from the monotonic
        # part on purpose: its CMC sits close to the wrist, so wrist->knuckle is legitimately
        # SHORTER than the next segment and a strict check would reject correct data.
        if fname != "thumb" and not (b0 >= 0.75 * b1 and b1 >= 0.75 * b2):
            return f"{fname} bones {b0:.3f}/{b1:.3f}/{b2:.3f} m not decreasing outward"
    return None


def gate_extrinsic_direction(capture_dir: str, serials: list, T: dict,
                             n_frames: int) -> tuple[float, float, int]:
    """Cross-camera check of the extrinsic DIRECTION. Returns (err_c2w, err_w2c, n_pairs) metres.

    Two cameras observing the same hand must agree on where it is in the WORLD once each maps
    its own camera-frame joint_3d out with its own extrinsic. Under the correct direction the
    two world points coincide (the labels come from one global fit); under the flipped one they
    do not. This turns "which way round is extrinsics.yml" from a reading of someone else's code
    into a measurement on our actual bytes."""
    errs_c2w, errs_w2c, n_pairs = [], [], 0
    for i in range(0, n_frames, max(1, n_frames // 8)):
        pts = []
        for s in serials:
            cam_dir = os.path.join(capture_dir, s)
            if s not in T or not os.path.exists(label_path(cam_dir, i)):
                continue
            j3, _ = load_label(cam_dir, i)
            if len(j3) == 0 or not hand_is_annotated(j3[0]):
                continue
            R, t = T[s][:, :3], T[s][:, 3]
            fwd = (R @ j3[0].T).T + t                              # hypothesis: T is cam->world
            Ri = np.linalg.inv(R)
            inv = (Ri @ (j3[0] - t).T).T                           # hypothesis: T is world->cam
            pts.append((fwd, inv))
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                errs_c2w.append(float(np.linalg.norm(pts[a][0] - pts[b][0], axis=-1).mean()))
                errs_w2c.append(float(np.linalg.norm(pts[a][1] - pts[b][1], axis=-1).mean()))
                n_pairs += 1
    if not n_pairs:
        return float("nan"), float("nan"), 0
    return float(np.median(errs_c2w)), float(np.median(errs_w2c)), n_pairs


# ------------------------------------------------------------------ output geometry
def output_geometry(square_crop: bool, res: int) -> tuple[int, int, int, int, int, float]:
    """One place that decides what a written frame looks like, so the video and the intrinsics
    can never disagree about it.

    Returns (x0, crop_w, crop_h, out_w, out_h, scale). The resize scale is UNIFORM: a
    non-uniform one would need two focals, which a [f, cx, cy] store cannot express."""
    x0 = (DEXYCB_W - DEXYCB_H) // 2 if square_crop else 0
    crop_w = DEXYCB_H if square_crop else DEXYCB_W
    crop_h = DEXYCB_H
    s = (res / crop_h) if res else 1.0
    return x0, crop_w, crop_h, int(round(crop_w * s)), int(round(crop_h * s)), s


# ------------------------------------------------------------------ video
def write_video(cam_dir: str, n: int, dst_mp4: str, geom: tuple,
                fps: float) -> tuple[int, int, int]:
    """color_%06d.jpg -> mp4. Returns (n_written, W, H) of the written frames.

    EXACTLY one written frame per cache row: HOT3DHandDataset indexes every cache by video frame
    number, so a dropped or extra frame is a silent, systematic image/label mismatch. A missing
    jpg is a hard error rather than a skip for the same reason."""
    x0, crop_w, _crop_h, out_w, out_h, _s = geom
    vw = None
    written = 0
    for i in range(n):
        p = os.path.join(cam_dir, f"color_{i:06d}.jpg")
        im = cv2.imread(p)
        if im is None:
            raise ValueError(f"missing colour frame {p} (would misalign every cache)")
        if im.shape[1] != DEXYCB_W or im.shape[0] != DEXYCB_H:
            raise ValueError(f"{p} is {im.shape[1]}x{im.shape[0]}, expected "
                             f"{DEXYCB_W}x{DEXYCB_H} (intrinsics would not match)")
        im = im[:, x0:x0 + crop_w]
        if (im.shape[1], im.shape[0]) != (out_w, out_h):
            im = cv2.resize(im, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if vw is None:
            vw = cv2.VideoWriter(dst_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
            if not vw.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed to open {dst_mp4}")
        vw.write(im)
        written += 1
    if vw is not None:
        vw.release()
    # Re-count: a muxer that drops frames would misalign every cache and exit 0 doing it.
    cap = cv2.VideoCapture(dst_mp4)
    n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_out != written:
        raise ValueError(f"encoded {n_out} frames but wrote {written} (misalignment)")
    return written, out_w, out_h


# ------------------------------------------------------------------ conversion
def convert_seq(capture_dir: str, serial: str, out_seq: str, T: dict, calib_dir: str,
                meta: dict, args, write: bool = True) -> dict:
    """One (capture, camera) -> one store sequence directory. Raises on any gate failure.

    Every gate runs BEFORE the first byte is written, so a rejected sequence leaves no store
    behind to be discovered later; `write=False` runs the same gates and reports only."""
    cam_dir = os.path.join(capture_dir, serial)
    n = int(meta["num_frames"])
    if args.max_frames:
        n = min(n, args.max_frames)
    sides = list(meta["mano_sides"])                      # e.g. ['right']; our slot 0=left, 1=right
    slot_of = [0 if str(s).lower().startswith("l") else 1 for s in sides]
    betas = load_mano_betas(calib_dir, str(meta["mano_calib"][0]))

    # Intrinsics for the frame we actually write. Crop first (exact: cx just loses the crop
    # offset), then resize. The resize form is (c + 0.5) * s - 0.5, NOT c * s: under the
    # pixel-centre convention plain multiplication lands the principal point half a pixel out and
    # biases every crop box (the reasoning egoexo4d_to_ours.py:136-144 had to spell out).
    geom = output_geometry(args.square_crop, args.res)
    x0, _cw, _ch, out_w, out_h, s = geom
    intr = load_intrinsics(calib_dir, serial)
    fx, fy = float(intr["fx"]) * s, float(intr["fy"]) * s
    px = (float(intr["ppx"]) - x0 + 0.5) * s - 0.5
    py = (float(intr["ppy"]) + 0.5) * s - 0.5
    cam_intr = torch.tensor([fx, px, py], dtype=torch.float32)     # single-focal store format

    j_cam = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    j_world = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    j_2d = torch.zeros(n, NUM_HANDS, NUM_JOINTS, 3)
    bboxes = torch.zeros(n, NUM_HANDS, 4)
    valid = torch.zeros(n, NUM_HANDS, dtype=torch.bool)
    gt64 = torch.zeros(n, NUM_HANDS * HAND_PARAM_DIM)
    w2c = torch.from_numpy(rt34_to_w2c(T[serial]).astype(np.float32)).unsqueeze(0).repeat(n, 1, 1)
    R_c2w, t_c2w = T[serial][:, :3], T[serial][:, 3]

    jl = []
    n_annot = n_offscreen = n_mano_gap = 0
    wrist_minus_trans = []
    aniso_px = []

    for fi in range(n):
        j3, pm = load_label(cam_dir, fi)
        hp = {}
        for hand_i, slot in enumerate(slot_of):
            if hand_i >= len(j3) or not hand_is_annotated(j3[hand_i]):
                continue                                  # absent hand stays ZEROS + valid False
            n_annot += 1
            # DexYCB derives joint_3d from pose_m, so the two are present together. If they are
            # not, DROP the frame instead of writing joints with an all-zero gt64: the param loss
            # is masked by `valid`, so a valid-but-zero MANO row would train the model to predict
            # zero MANO with a healthy-looking curve (the hazard train_hand_head.py:444-453 calls
            # out at `if "gt" in cached`). Counted, never silent.
            row = pm[hand_i] if hand_i < len(pm) else np.zeros(51)
            if not np.any(row != 0.0):
                n_mano_gap += 1
                continue
            j16 = j3[hand_i][DEXYCB21_TO_MANO16]                    # metres, camera frame
            j_cam[fi, slot] = torch.tensor(j16, dtype=torch.float32)
            j_world[fi, slot] = torch.tensor((R_c2w @ j16.T).T + t_c2w, dtype=torch.float32)

            # Single-focal projection is what the whole pipeline uses; measure what discarding
            # fy costs in pixels so gate 1 can refuse a camera where it matters.
            u, v = project_single_focal(j16[None], fx, px, py)
            u, v = u[0], v[0]
            z = np.clip(j16[:, 2], 1e-3, None)
            aniso_px.append(float(np.abs((fy - fx) * j16[:, 1] / z).max()))
            j_2d[fi, slot, :, 0] = torch.tensor(u, dtype=torch.float32)
            j_2d[fi, slot, :, 1] = torch.tensor(v, dtype=torch.float32)
            j_2d[fi, slot, :, 2] = 1.0

            # A hand is usable only if it is in front of the camera AND leaves at least one joint
            # inside the (possibly cropped) frame - the preprocess_hoi4d.py:444-445 definition.
            # After a centre-square crop this is also the honest record of what the crop threw
            # away, which is why frac_hand_cropped_out is reported rather than swallowed.
            infront = bool((j16[:, 2] > 1e-2).all())
            inb = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
            if infront and not inb.any():
                n_offscreen += 1
            if not (infront and inb.any()):
                continue

            b = joints_to_bbox(j16, fx, px, py, float(out_w), float(out_h),
                               rf=args.rescale_factor)
            if b is None or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
                continue
            bboxes[fi, slot] = torch.from_numpy(np.asarray(b, np.float32))
            valid[fi, slot] = True

            # gt64, camera frame: [transl 3, quat_wxyz 4, pose15 (PCA) 15, betas 10].
            trans = row[48:51]
            wrist_minus_trans.append(j16[0] - trans)                # gate 5 (module docstring)
            off = slot * HAND_PARAM_DIM
            q = _aa_to_quat_wxyz(row[0:3])
            gt64[fi, off:off + 3] = torch.tensor(trans, dtype=torch.float32)
            gt64[fi, off + 3:off + 7] = torch.tensor(q, dtype=torch.float32)
            gt64[fi, off + 7:off + 22] = torch.tensor(row[3:18], dtype=torch.float32)
            gt64[fi, off + 22:off + 32] = torch.tensor(betas, dtype=torch.float32)
            hp[str(slot)] = {
                "wrist_xform": {"t_xyz": [float(x) for x in trans],
                                "q_wxyz": [float(x) for x in q]},
                "pose": [float(x) for x in row[3:18]],
                "betas": [float(x) for x in betas],
            }
        jl.append({"timestamp_ns": fi, "hand_poses": hp})

    # ---- gate 1: fx/fy anisotropy (our store cannot express two focals)
    aniso = float(np.median(aniso_px)) if aniso_px else 0.0
    if aniso > args.max_aniso_px:
        raise ValueError(f"single-focal error {aniso:.2f} px > {args.max_aniso_px} px "
                         f"(fx={fx:.2f} fy={fy:.2f}): this camera is too anisotropic for a "
                         f"[f, cx, cy] store")

    # ---- gate 3: anatomy, per hand, BEFORE anything is written
    stats = {"n_frames": n, "n_annotated": n_annot, "n_mano_gap": n_mano_gap,
             "frac_hand_cropped_out": round(n_offscreen / max(1, n_annot), 4),
             "aniso_px": round(aniso, 3)}
    for slot in set(slot_of):
        rep = anatomy_report(j_cam[:, slot].numpy(), valid[:, slot].numpy())
        stats[f"anatomy_{slot}"] = rep
        why = anatomy_gate(rep)
        if why is not None:
            raise ValueError(f"anatomical gate: {why}")

    # ---- gate 5: units + transl semantics
    if wrist_minus_trans:
        wmt = np.stack(wrist_minus_trans)
        spread = float(np.linalg.norm(wmt - np.median(wmt, 0), axis=-1).max())
        stats["root_offset_m"] = round(float(np.linalg.norm(np.median(wmt, 0))), 4)
        stats["root_offset_spread_m"] = round(spread, 6)
        if spread > 1e-3:
            raise ValueError(
                f"joint_3d[wrist] - pose_m.trans varies by {spread:.4f} m across the sequence. "
                f"manopth leaves the root joint at its shaped-template position and then adds "
                f"trans, so for a STATIC camera this difference is a constant (R_w2c @ root_j). "
                f"A varying one means pose_m and joint_3d are not in the same frame - do not "
                f"train on this store")
        if stats["root_offset_m"] > 1.0:
            raise ValueError(f"canonical root offset {stats['root_offset_m']} m: the labels look "
                             f"like MILLIMETRES, not metres - every depth in the store is wrong")

    stats["valid_rate"] = round(float(valid.any(-1).float().mean()), 4)
    if not write:
        return stats
    if stats["valid_rate"] <= args.min_valid_rate:
        raise EmptySequence(f"valid_rate {stats['valid_rate']} <= {args.min_valid_rate} "
                            f"({n_annot} annotated, {n_offscreen} of them outside the frame)")

    # ---- write
    hd = os.path.join(out_seq, "hand_data")
    os.makedirs(hd, exist_ok=True)
    written, W, H = write_video(cam_dir, n, os.path.join(out_seq, "video_main_rgb.mp4"),
                                geom, args.fps)
    if written != n:                                     # gate 4
        raise ValueError(f"wrote {written} video frames != {n} cache rows (misalignment)")
    stats["video_hw"] = [H, W]
    torch.save(cam_intr, os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(w2c, os.path.join(hd, "cam_extrinsics_cache.pt"))
    torch.save(j_cam, os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(j_world, os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(j_2d, os.path.join(hd, "gt_joints_2d_cache.pt"))
    with open(os.path.join(hd, "mano_hand_pose_trajectory.jsonl"), "w") as f:
        for e in jl:
            f.write(json.dumps(e) + "\n")
    bbox_name = f"hand_bboxes_v2_rf{args.rescale_factor}_res224x224.pt"
    torch.save({"bboxes": bboxes, "valid": valid, "gt": gt64,
                "convention": f"joints_to_bbox rf{args.rescale_factor} rectangular unclamped "
                              f"(dexycb)"},
               os.path.join(hd, bbox_name))
    return stats


def is_complete(out_seq: str, rescale_factor: float) -> bool:
    """A resumable run must not half-rewrite a finished sequence (the store is ~8000 dirs and a
    cluster reaper will interrupt it). Complete = video + every cache present."""
    hd = os.path.join(out_seq, "hand_data")
    need = list(CACHE_FILES) + [f"hand_bboxes_v2_rf{rescale_factor}_res224x224.pt"]
    return (os.path.exists(os.path.join(out_seq, "video_main_rgb.mp4"))
            and all(os.path.exists(os.path.join(hd, f)) for f in need))


def find_captures(dexycb_root: str, subjects: list | None) -> list:
    """<root>/<date>-subject-NN/<capture>/ dirs that hold a meta.yml."""
    out = []
    for sub in sorted(os.listdir(dexycb_root)):
        if "subject" not in sub or not os.path.isdir(os.path.join(dexycb_root, sub)):
            continue
        if subjects and sub not in subjects:
            continue
        for cap in sorted(glob.glob(os.path.join(dexycb_root, sub, "*"))):
            if os.path.exists(os.path.join(cap, "meta.yml")):
                out.append(cap)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dexycb_root", required=True,
                    help="DexYCB root holding calibration/, models/ and <date>-subject-NN/")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--cameras", default="",
                    help="comma-separated camera serials to convert (default: every serial in "
                         "meta.yml). The 8 views of one capture are near-duplicate content, so a "
                         "mixture usually wants a subset")
    ap.add_argument("--subjects", default="", help="comma-separated subject dir names to restrict to")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N output sequences (smoke run on one sequence: --limit 1)")
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames per sequence (debug)")
    ap.add_argument("--rescale_factor", type=float, default=1.5,
                    help="box padding; 1.5 is the training convention and the cache filename")
    ap.add_argument("--res", type=int, default=0,
                    help="resize the written frame to res x res (0 = keep the native crop, which "
                         "avoids any resampling of the intrinsics)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no_square_crop", dest="square_crop", action="store_false",
                    help="keep the native 4:3 frame instead of the 480x480 centre crop. Costs no "
                         "pixels but hands the head anisotropically squashed crops")
    ap.add_argument("--max_aniso_px", type=float, default=2.0,
                    help="refuse a camera whose fx/fy gap costs more than this many pixels")
    ap.add_argument("--min_valid_rate", type=float, default=0.0,
                    help="skip a sequence whose usable-frame fraction is <= this (default: keep "
                         "only sequences with at least one usable frame)")
    ap.add_argument("--max_gate_fail", type=float, default=0.10,
                    help="abort the run if more than this fraction of sequences fail a gate; a "
                         "high rate means the joint MAPPING is wrong, not the source data")
    ap.add_argument("--validate", action="store_true",
                    help="run every gate on the first requested camera of each capture and report "
                         "the numbers, writing nothing (--limit then counts captures)")
    ap.set_defaults(square_crop=True)
    a = ap.parse_args()

    calib_dir = os.path.join(a.dexycb_root, "calibration")
    if not os.path.isdir(calib_dir):
        raise SystemExit(f"no calibration dir at {calib_dir}")
    want = [s for s in a.cameras.split(",") if s]
    subjects = [s for s in a.subjects.split(",") if s]
    captures = find_captures(a.dexycb_root, subjects)
    print(f"found {len(captures)} captures under {a.dexycb_root}", flush=True)

    os.makedirs(a.out_root, exist_ok=True)
    n_done = n_skip = n_fail = n_gate = n_empty = n_valid = 0
    checked_extr: dict = {}
    for cap in captures:
        if a.limit and n_done >= a.limit:
            break
        with open(os.path.join(cap, "meta.yml")) as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        serials = [str(s) for s in meta["serials"]]
        try:
            T, master = load_extrinsics(calib_dir, str(meta["extrinsics"]))
        except (OSError, KeyError) as e:
            print(f"CAP_FAIL {cap}: extrinsics unreadable ({e})", flush=True)
            n_fail += 1
            continue

        # Rig sanity + the direction gate. Both are properties of the CALIBRATION, not of the
        # capture, so they run once per extrinsics id (10 of them across the dataset) rather than
        # once per capture - the gate reads label npz files, and 1000 redundant repeats of it
        # would cost more IO than the conversion.
        eid = str(meta["extrinsics"])
        if eid not in checked_extr:
            n_cam_rows = len([k for k in T if k != "apriltag"])
            master_is_I = (master in T and np.allclose(T[master], np.eye(4)[:3], atol=1e-6))
            e_c2w, e_w2c, n_pairs = gate_extrinsic_direction(cap, serials, T,
                                                             int(meta["num_frames"]))
            if n_pairs and not (e_c2w < e_w2c and e_c2w < 0.02):
                raise SystemExit(
                    f"EXTRINSIC DIRECTION GATE FAILED on {cap}: cross-camera world disagreement "
                    f"is {e_c2w*1000:.1f} mm assuming camera->world and {e_w2c*1000:.1f} mm "
                    f"assuming world->camera ({n_pairs} pairs). The converter assumes "
                    f"camera->world (dex_ycb_toolkit sequence_loader.py deprojection). Do NOT "
                    f"write a world cache until this is resolved - the camera-frame cache is "
                    f"unaffected either way")
            print(f"[calib {eid}] cams={n_cam_rows} master_identity={master_is_I} "
                  f"xcam_err_c2w={e_c2w*1000:.2f}mm w2c={e_w2c*1000:.2f}mm ({n_pairs} pairs)",
                  flush=True)
            if not n_pairs:
                print("  WARNING: extrinsic direction UNVERIFIED (needs two cameras with labels "
                      "in this capture). The camera-frame caches do not use extrinsics and are "
                      "unaffected; treat gt_joints_cache_world.pt as unvalidated until a "
                      "multi-camera capture has been converted", flush=True)
            checked_extr[eid] = True
        if a.validate:
            # Same gates, no bytes written: anatomy + units + transl semantics + anisotropy on
            # the first requested camera of this capture.
            serial = next((s for s in serials if not want or s in want), None)
            if serial is not None:
                try:
                    st = convert_seq(cap, serial, "", T, calib_dir, meta, a, write=False)
                    print(f"  VALIDATE {serial}: {json.dumps(st, sort_keys=True)}", flush=True)
                except (ValueError, OSError, KeyError) as e:
                    n_gate += 1
                    print(f"  VALIDATE {serial}: FAILED {type(e).__name__}: {e}", flush=True)
            # Count validations separately. Folding these into n_done made the summary print
            # "wrote=1" after a dry run that wrote nothing, which reads as success and is how a
            # validate-only invocation gets mistaken for a completed conversion.
            n_valid += 1
            continue

        for serial in serials:
            if want and serial not in want:
                continue
            if a.limit and n_done >= a.limit:
                break
            sub = os.path.basename(os.path.dirname(cap))
            seq_id = f"{sub}_{os.path.basename(cap)}_{serial}"
            out_seq = os.path.join(a.out_root, seq_id)
            if is_complete(out_seq, a.rescale_factor):
                n_skip += 1
                continue
            # Build into a .tmp_ dir and rename, so an interrupted run never leaves a partial
            # sequence that is_complete() would later have to adjudicate.
            tmp = os.path.join(a.out_root, f".tmp_{seq_id}")
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                st = convert_seq(cap, serial, tmp, T, calib_dir, meta, a)
                shutil.rmtree(out_seq, ignore_errors=True)
                os.replace(tmp, out_seq)
                n_done += 1
                print(f"[{n_done}] {seq_id} N={st['n_frames']} annot={st['n_annotated']} "
                      f"valid={st['valid_rate']} croppedout={st['frac_hand_cropped_out']} "
                      f"root_off={st.get('root_offset_m')}", flush=True)
            except EmptySequence as e:                   # must precede the ValueError clause
                shutil.rmtree(tmp, ignore_errors=True)
                n_empty += 1
                print(f"SEQ_EMPTY {seq_id}: {e}", flush=True)
            except ValueError as e:
                shutil.rmtree(tmp, ignore_errors=True)
                n_gate += 1
                print(f"SEQ_GATE {seq_id}: {e}", flush=True)
            except (OSError, KeyError, RuntimeError) as e:
                shutil.rmtree(tmp, ignore_errors=True)
                n_fail += 1
                print(f"SEQ_FAIL {seq_id}: {type(e).__name__}: {e}", flush=True)

    if a.validate:
        print(f"DEXYCB_TO_OURS_VALIDATE_ONLY validated={n_valid} gate_failed={n_gate} "
              f"errors={n_fail} -- NO BYTES WRITTEN (dry run)", flush=True)
    else:
        print(f"DEXYCB_TO_OURS_DONE wrote={n_done} skipped={n_skip} empty={n_empty} "
              f"gate_failed={n_gate} errors={n_fail} -> {a.out_root}", flush=True)
    total = n_done + n_gate
    if n_gate and total and n_gate / total > a.max_gate_fail:
        # A few rejects means bad source frames and is healthy. A LARGE fraction means the joint
        # MAPPING or the frame convention is wrong, which is the H2O-scramble signature and must
        # stop everything rather than leave a plausible-looking partial store behind.
        raise SystemExit(
            f"ABORT: {n_gate}/{total} sequences failed a gate ({n_gate/total:.1%} > "
            f"{a.max_gate_fail:.0%}). At that rate suspect the 21->16 remap or the label frame, "
            f"not the data. Do NOT train on this store until it is re-verified")


if __name__ == "__main__":
    main()
