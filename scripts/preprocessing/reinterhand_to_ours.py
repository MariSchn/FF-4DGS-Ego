#!/usr/bin/env python3
"""Re:InterHand (EGOCENTRIC split) -> our HOT3DHandDataset per-sequence store.

Dataset: Moon et al., "A Dataset of Relighted 3D Interacting Hands", NeurIPS 2023 (D&B track).
LICENSE: CC-BY-NC 4.0. Non-commercial use only; that term travels with the converted store,
so anything derived from it (caches, checkpoints trained on it, figures) inherits the licence.

WHY THIS DATASET. Our locked pool needs depth diversity, not just more frames: the transfer law
we fit is C_abs ~= 60 + 0.50 * |depth shift| (report/open-lines-tracker.md, 2026-08-04), so a
store earns its place by where its hands sit in camera depth, not by size. Measure that with
scripts/preprocessing/reinterhand_depth_stat.py BEFORE adding it to a mixture.

WHAT THE DOWNLOAD ACTUALLY CONTAINS (verified on the Euler copy at
/cluster/scratch/dmonopoli/reinterhand, 10 capture ids, ego split 147,745 images / 217 GB):

    <capture_id>/mano_fits/params/<frameid>_<left|right>.json
    <capture_id>/mano_fits/meshes/
    <capture_id>/Ego_cameras/envmap_per_frame.tar.gz{aa,ab,...}      # split parts

  A params JSON has exactly three keys, shapes verified on a real sample:
      pose (48,)   shape (10,)   trans (3,)

  `trans` reads z ~= 1.08 m on the inspected sample, i.e. CAPTURE/WORLD space, NOT camera space.
  Camera-frame joints therefore REQUIRE the ego extrinsics, which live in the Ego_cameras payload
  that was STILL DOWNLOADING when this converter was written.

  ==> EVERYTHING ABOUT THE Ego_cameras LAYOUT IS UNVERIFIED. It is confined to exactly two
      clearly-marked functions, `load_ego_cameras` and `resolve_ego_image`, each with its
      expected input documented. CONFIRM BOTH against the extracted Ego_cameras directory
      before the first real run. Nothing else in this file depends on that layout.

OUR TARGET STORE FORMAT, derived from the consumer (scripts/train_hand_head.py,
class HOT3DHandDataset), not invented. Line numbers are as of 2026-08-04 and that file is under
active edit, so each one also quotes the symbol it points at:

    <out>/<capture>__<ego_cam>/
        video_main_rgb.mp4                          REQUIRED; :300-301 ("Skipping {seq_path}
                                                    because it has no video file") drops any
                                                    sequence without it. Frame t of the video
                                                    MUST be row t of every cache below.
        hand_data/cam_intrinsics.pt                 [3] = [focal, cx, cy]  (:375 cam_intr_cache_path)
        hand_data/cam_extrinsics_cache.pt           [N,4,4] T_camera_world (:374 cam_extr_cache_path)
        hand_data/gt_joints_cache_cam_v2.pt         [N,2,16,3] METRES, CAMERA frame (:288-292)
        hand_data/gt_joints_cache_world.pt          [N,2,16,3] METRES, world frame (:288-292)
        hand_data/gt_joints_2d_cache.pt             [N,2,16,3] (u,v,conf) px (:373 cam_2d_cache_path)
        hand_data/mano_hand_pose_trajectory.jsonl   raw MANO trajectory (:283, parsed by
                                                    _hand_to_vec at :338)
        hand_data/hand_bboxes_v2_rf1.5_res224x224.pt
            {"bboxes": [N,2,4] normalised xyxy, "valid": [N,2] bool, "gt": [N,64] float32}
            All three keys are REQUIRED: :441-449 reads cached["bboxes"], cached["valid"] and
            cached["gt"], and a regeneration that dropped "gt" broke every eval on the H2O store
            (scripts/verify_box_store.py:10-13, REQUIRED_KEYS at :32).

    The joint-cache name is chosen by the consumer at train_hand_head.py:288-292:
    use_hand_crop=True -> "gt_joints_cache_cam_v2.pt", else "gt_joints_cache_world.pt".
    We train with crops, so _cam_v2 is the load-bearing one. Shape [n_frames, n_hands,
    n_joints, 3] is fixed by _compute_seq_joints_from_params (:557) -> [N,2,16,3], and the
    metric/anchor code reads metres throughout (scripts/fit_ref_scale.py:33-36 bands wrist z in
    0.05-5 m).

THREE THINGS THAT ARE SILENT IF WRONG, so each is pinned to evidence rather than assumed:

  1. HAND AXIS ORDER. Index 0 = LEFT, index 1 = RIGHT. Evidence: train_hand_head.py:861
     ('hand_key = str(hand_idx)  # "0" = left, "1" = right') and :862 ('is_right = hand_idx == 1'),
     the same convention as _compute_seq_joints_from_params (:573, is_right=(h_idx == 1)) and as
     every other converter (scripts/arctic_to_ours.py:22,192; scripts/egoexo4d_to_ours.py:52;
     RH = 1 in eval_worldspace_baseline / eval_world_space / fit_ref_scale).
     Re:InterHand ships SEPARATE left/right JSONs per frame, so this mapping is applied at the
     one point where the filename side is read - swapping it is invisible until metrics rot.

  2. JOINT ORDER / COUNT. 16 smplx-kinematic joints: wrist, index x3, middle x3, pinky x3,
     ring x3, thumb x3. We do NOT re-derive the reduction here: `_to_smplx16` is imported from
     scripts/arctic_to_ours.py (:70-75), the same function ARCTIC and OakInk2 went through, so
     this store is correct-by-construction relative to them. A locally re-written 21->16 selector
     is exactly the H2O scramble that voided an entire dataset's numbers
     (h2o_to_currentproto.py:120-125). `anatomy_gate` below re-checks it anyway.

  3. BOX CONVENTION. `joints_to_bbox` is IMPORTED from scripts/arctic_to_ours.py:111-118:
     rectangular per-axis x1.5, UNCLAMPED. A local square+clamped reimplementation cost
     ~290-380 mm C-abs on H2O and read as a generalisation failure
     (scripts/verify_box_store.py:5-9). Run scripts/verify_box_store.py on the output.

MANO PARAMETER SPLIT. pose (48,) = 3 global orient + 45 hand pose, axis-angle, no PCA. Confirmed
against the two other axis-angle paths in this repo: h2o_to_currentproto.py:55 ("pose48(3 global
aa + 45 full aa, no PCA)") and oakink2_to_ours.py:60-61 (aa[:,0] -> global_orient,
aa[:,1:] -> hand_pose[45]). FK uses build_mano (use_pca=False, flat_hand_mean=False), which is
also what InterHand2.6M's own MANO fitting uses - but see --fix_left_shapedirs.

UNITS. Our stores are METRES end to end. MANO FK output is metres by construction; only `trans`
and the camera translation carry a unit choice, and InterHand2.6M-derived data has historically
mixed metre MANO params with MILLIMETRE camera positions. Both are auto-detected by magnitude,
both are printed, and the physical wrist-depth gate below catches a 1000x mismatch that slips
through.

Usage (an smplx-capable env with our MANO models):
    python -m scripts.preprocessing.reinterhand_to_ours \
        --reinterhand_root /cluster/scratch/dmonopoli/reinterhand \
        --mano_dir models/MANO --out_root $S/reinterhand_ours \
        --ego_cam_json '{capture}/Ego_cameras/cam_params.json' \
        --ego_image_pattern '{capture}/Ego_cameras/envmap_per_frame/{cam}/{frame}.jpg' \
        --limit 1 --max_frames 64            # smoke run first, always
Then:
    python -m scripts.verify_box_store --data_root $S/reinterhand_ours
    python -m scripts.preprocessing.reinterhand_depth_stat --data_root $S/reinterhand_ours
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import types
from dataclasses import dataclass

import cv2
import numpy as np
import torch

# Every shared convention comes from the ARCTIC converter rather than being restated here, so
# Re:InterHand cannot silently drift from ARCTIC/OakInk2 on joint order, box shape or the
# camera-frame wrist-parameter identity. See the module docstring for why that matters.
from scripts.arctic_to_ours import (HAND_PARAM_DIM, NUM_HANDS, NUM_JOINTS,
                                    _to_smplx16, apply_se3, build_mano,
                                    camera_frame_wrist_params, fk_world_joints,
                                    joints_to_bbox, pose45_to_pca15,
                                    project_single_focal)

# Hand axis. Index 0 = LEFT, 1 = RIGHT (train_hand_head.py:861-862). Re:InterHand encodes the
# side in the params FILENAME, so this dict is the single place the two conventions meet.
SIDE_TO_HAND_IDX = {"left": 0, "right": 1}
HAND_IDX_TO_SIDE = {0: "left", 1: "right"}

RF = 1.5                       # box rescale factor; the store filename encodes it
BOX_RES = 224                  # box cache filename only (boxes are normalised, res-independent)
VIDEO_FPS = 30

# Physical wrist-depth band, metres. Same lower bound as scripts/fit_ref_scale.py:35-36, tighter
# at the top (3 m rather than 5) to match the band reinterhand_depth_stat.py rejects outside, so
# the converter cannot emit a sequence the depth statistic would then throw away. A
# metre/millimetre mix-up between the MANO `trans` and the camera translation lands ~1000x outside
# this band, which is the whole point of gating on it.
Z_MIN_M, Z_MAX_M = 0.05, 3.0

# Anatomical gate. Median wrist->proximal distances for the smplx-16 layout, millimetres. Bands
# copied from h2o_to_currentproto.py's gate 4, which is what caught the 2026-07-18 joint scramble
# (it measured 89/125/123/40/157 mm and would have failed here).
MCP_GATES = {1: ("index MCP", 55.0, 115.0), 4: ("middle MCP", 60.0, 120.0),
             7: ("pinky MCP", 45.0, 100.0), 10: ("ring MCP", 50.0, 110.0),
             13: ("thumb CMC", 15.0, 65.0)}
# No joint of a hand may sit further than this from its own wrist. A real hand spans ~200 mm
# wrist-to-middle-tip; 300 mm is a generous ceiling that still catches a scrambled or
# wrongly-scaled cache instantly. Isolated violations are dropped as frames; above this rate the
# sequence is refused, because a systematic violation is a convention bug and not bad fits.
MAX_JOINT_TO_WRIST_M = 0.30
SPAN_BAD_FRAC_MAX = 0.01
# A few bad captures are source noise; a large fraction means the MAPPING is wrong, which is the
# H2O-scramble signature and must stop the run (same split as egoexo4d_to_ours.py:380-395).
GATE_ABORT_FRAC = 0.10


class GateFailure(Exception):
    """A converted sequence failed a correctness gate; it is NOT written."""


class EgoLayoutUnknown(Exception):
    """The Ego_cameras payload does not match any layout this converter knows how to read."""


# ---------------------------------------------------------------------------------------------
# UNVERIFIED REGION - Ego_cameras payload
#
# The two functions below are the ONLY places that touch the Ego_cameras download. It had not
# finished extracting when this was written, so their input layout is an EXPECTATION, not a
# measurement. Confirm both against the extracted directory before the first real run; if the
# real layout differs, only this region changes.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EgoCamera:
    """One egocentric camera of one capture, in OUR conventions.

    cam_id : str            camera name as it appears in the image directory
    K      : [3,3] float64  pinhole intrinsics in pixels, at the ego IMAGE resolution
    w2c    : [4,4] float64  world -> camera rigid transform, METRES (X_cam = R @ X_world + t).
                            This is the convention every store uses for cam_extrinsics_cache
                            (train_hand_head.py:760, "T_cam_world (validated w2c)").
    per_frame : dict|None   {frame_id: [4,4]} when the ego camera MOVES. None => static rig,
                            `w2c` applies to every frame.
    """
    cam_id: str
    K: np.ndarray
    w2c: np.ndarray
    per_frame: dict | None = None

    def w2c_at(self, frame_id: str) -> np.ndarray | None:
        """world->camera for one frame, or None if this frame has no pose (drop the frame)."""
        if self.per_frame is None:
            return self.w2c
        return self.per_frame.get(frame_id)


def load_ego_cameras(cam_json_path: str, trans_units: str = "auto") -> dict[str, EgoCamera]:
    """Read the ego camera parameters of one capture.  *** LAYOUT UNVERIFIED - CONFIRM FIRST ***

    EXPECTED INPUT: a JSON file whose top level is one of the two InterHand2.6M-family layouts
    below. Re:InterHand is rendered from InterHand2.6M captures, so its camera file is expected
    to follow the InterHand2.6M `cam_params.json` schema - expected, not verified, because the
    Ego_cameras archive had not finished downloading.

      (A) InterHand2.6M cam_params.json, camera-centre form, keyed by camera id:
            {"campos":  {"<cam>": [x, y, z]},          # camera CENTRE in world coords
             "camrot":  {"<cam>": [[3x3]]},            # world -> camera ROTATION
             "focal":   {"<cam>": [fx, fy]},
             "princpt": {"<cam>": [px, py]}}
          -> w2c = [R | -R @ campos].  This is InterHand2.6M's own convention
             (cam_coord = (world_coord - campos) @ camrot.T), NOT a plain [R|t].

      (B) explicit 4x4 form, keyed by camera id, optionally per frame:
            {"<cam>": {"K": [[3x3]], "w2c": [[4x4]]}}
            {"<cam>": {"K": [[3x3]], "w2c": {"<frame_id>": [[4x4]]}}}

    UNITS: `campos` / the 4x4 translation are auto-detected by magnitude (InterHand2.6M publishes
    camera positions in MILLIMETRES while its MANO fits are in metres - mixing them is a silent
    1000x error, which is why this is detected and printed rather than assumed). Force with
    trans_units="m" or "mm".

    Anything else raises EgoLayoutUnknown naming the keys actually found, so the fix is a
    five-minute edit to THIS function and nothing else.
    """
    with open(cam_json_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise EgoLayoutUnknown(f"{cam_json_path}: top level is {type(raw).__name__}, expected a dict")

    if {"campos", "camrot"} <= set(raw):                       # layout (A)
        cams = sorted(raw["camrot"].keys())
        centres = np.asarray([np.asarray(raw["campos"][c], np.float64).reshape(3) for c in cams])
        scale = _translation_scale(centres, trans_units, what=f"ego campos ({os.path.basename(cam_json_path)})")
        out = {}
        for c in cams:
            R = np.asarray(raw["camrot"][c], np.float64).reshape(3, 3)
            pos = np.asarray(raw["campos"][c], np.float64).reshape(3) * scale
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = -R @ pos                              # t = -R @ C, the camera-centre form
            fx, fy = (np.asarray(raw["focal"][c], np.float64).reshape(2)
                      if "focal" in raw else (np.nan, np.nan))
            px, py = (np.asarray(raw["princpt"][c], np.float64).reshape(2)
                      if "princpt" in raw else (np.nan, np.nan))
            K = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]], np.float64)
            out[c] = EgoCamera(cam_id=c, K=K, w2c=w2c)
        return out

    first = next(iter(raw.values()))
    if isinstance(first, dict) and {"K", "w2c"} <= set(first):  # layout (B)
        out = {}
        for c, e in raw.items():
            K = np.asarray(e["K"], np.float64).reshape(3, 3)
            w2c_raw = e["w2c"]
            if isinstance(w2c_raw, dict):                       # moving ego camera, per frame
                per = {str(k): np.asarray(v, np.float64).reshape(4, 4) for k, v in w2c_raw.items()}
                ts = np.asarray([m[:3, 3] for m in per.values()])
                s = _translation_scale(ts, trans_units, what=f"ego w2c t [{c}]")
                per = {k: _scale_w2c(m, s) for k, m in per.items()}
                out[c] = EgoCamera(cam_id=c, K=K, w2c=np.eye(4), per_frame=per)
            else:
                m = np.asarray(w2c_raw, np.float64).reshape(4, 4)
                s = _translation_scale(m[None, :3, 3], trans_units, what=f"ego w2c t [{c}]")
                out[c] = EgoCamera(cam_id=c, K=K, w2c=_scale_w2c(m, s))
        return out

    raise EgoLayoutUnknown(
        f"{cam_json_path}: unrecognised ego camera layout. Top-level keys = "
        f"{sorted(raw)[:12]}; first value type = {type(first).__name__}"
        + (f", its keys = {sorted(first)[:12]}" if isinstance(first, dict) else "")
        + ". Extend load_ego_cameras() - do NOT guess elsewhere in this file.")


def resolve_ego_image(pattern: str, root: str, capture: str, cam: str, frame_id: str) -> str | None:
    """Path of one ego RGB frame.  *** LAYOUT UNVERIFIED - CONFIRM FIRST ***

    EXPECTED INPUT: `pattern` is a path template, relative to --reinterhand_root, using the
    placeholders {capture} {cam} {frame}, e.g.
        '{capture}/Ego_cameras/envmap_per_frame/{cam}/{frame}.jpg'
    The Ego_cameras images ship as split tarballs (envmap_per_frame.tar.gz{aa,ab,...}) which must
    be concatenated and extracted first; the directory tree inside them is NOT yet known, so the
    pattern is a CLI argument rather than a hard-coded path. Zero-padded numeric variants are
    tried as a convenience.

    Returns the first existing path, or None (the caller black-pads that frame so that video
    frame t == cache row t is preserved - the contract HOT3DHandDataset assumes).
    """
    digits = "".join(ch for ch in frame_id if ch.isdigit())
    candidates = [pattern.format(capture=capture, cam=cam, frame=frame_id)]
    if digits:
        for width in (5, 6, 8):
            candidates.append(pattern.format(capture=capture, cam=cam, frame=digits.zfill(width)))
        candidates.append(pattern.format(capture=capture, cam=cam, frame=str(int(digits))))
    for rel in candidates:
        p = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def diagnose_image_pattern(root: str, capture: str, pattern: str) -> str:
    """One-shot explanation of WHY no ego image resolved, so the operator fixes the pattern once.

    Lists what is really on disk for the first three levels under the capture directory, so the
    correct pattern can be read straight off the failure message. Cheaper than three round-trips
    to the cluster.
    """
    probe = os.path.join(root, capture)
    lines = [f"    pattern      : {pattern}", f"    probing under: {probe}"]
    cur = probe if os.path.isdir(probe) else root
    for depth in range(3):
        if not os.path.isdir(cur):
            lines.append(f"    [{depth}] {cur} is not a directory")
            break
        entries = sorted(os.listdir(cur))[:10]
        lines.append(f"    [{depth}] {cur} -> {entries}")
        sub = [e for e in entries if os.path.isdir(os.path.join(cur, e))]
        if not sub:
            break
        cur = os.path.join(cur, sub[0])
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# END OF UNVERIFIED REGION. Everything below reads only mano_fits/params, which IS verified.
# ---------------------------------------------------------------------------------------------


def _scale_w2c(m: np.ndarray, s: float) -> np.ndarray:
    out = m.copy()
    out[:3, 3] = out[:3, 3] * s
    return out


def _translation_scale(t: np.ndarray, units: str, what: str) -> float:
    """Factor converting a translation array to METRES, from an explicit choice or by magnitude.

    Auto-detection is by median |t|: a capture-scale rig sits within ~10 m of its origin, so a
    median in 0.02-10 is metres and a median 1000x larger (20-10000) is millimetres. The gap
    between the two bands is refused rather than rounded to the nearer guess - InterHand2.6M
    ships MILLIMETRE camera positions next to METRE MANO fits, and silently mixing them is a
    1000x error that still produces a perfectly plausible-looking tensor.
    """
    if units == "m":
        return 1.0
    if units == "mm":
        return 1e-3
    mag = float(np.median(np.linalg.norm(np.asarray(t, np.float64).reshape(-1, 3), axis=-1)))
    if 0.02 <= mag <= 10.0:
        print(f"    units[{what}]: median |t| = {mag:.4f} -> METRES (x1)")
        return 1.0
    if 20.0 <= mag <= 10000.0:
        print(f"    units[{what}]: median |t| = {mag:.1f} -> MILLIMETRES (x1e-3)")
        return 1e-3
    raise GateFailure(f"{what}: median |t| = {mag:.4g} is neither a metre- nor a millimetre-scale "
                      f"capture; pass --mano_units / --cam_units explicitly after checking the source")


# ---------------------------------------------------------------------------- MANO fits (verified)

def parse_params_dir(params_dir: str) -> dict[str, dict[int, dict]]:
    """Read <capture>/mano_fits/params/<frameid>_<left|right>.json -> {frame_id: {hand_idx: params}}.

    VERIFIED layout: filename is '<frameid>_<side>.json' and each JSON has exactly the three keys
    pose (48,), shape (10,), trans (3,). The SIDE COMES FROM THE FILENAME and is mapped through
    SIDE_TO_HAND_IDX here and nowhere else, so hand-axis order has exactly one definition site.
    """
    out: dict[str, dict[int, dict]] = {}
    n_bad = 0
    for path in glob.glob(os.path.join(params_dir, "*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if "_" not in stem:
            n_bad += 1
            continue
        frame_id, side = stem.rsplit("_", 1)
        hi = SIDE_TO_HAND_IDX.get(side.lower())
        if hi is None:
            n_bad += 1
            continue
        try:
            with open(path) as f:
                p = json.load(f)
            rec = {"pose": np.asarray(p["pose"], np.float64).reshape(48),
                   "shape": np.asarray(p["shape"], np.float64).reshape(10),
                   "trans": np.asarray(p["trans"], np.float64).reshape(3)}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A fit that is absent or null for this frame is normal (the hand left the volume);
            # it must be a MISSING hand, never zeros - zeros in `gt` would train the model to
            # predict a zero MANO with a healthy-looking loss curve (train_hand_head.py:448-453).
            n_bad += 1
            continue
        if not (np.isfinite(rec["pose"]).all() and np.isfinite(rec["trans"]).all()
                and np.isfinite(rec["shape"]).all()):
            n_bad += 1
            continue
        out.setdefault(frame_id, {})[hi] = rec
    if n_bad:
        print(f"    params: skipped {n_bad} unusable/absent fit file(s)")
    return out


def sort_frame_ids(frame_ids) -> list[str]:
    """Chronological order. Frame ids are numeric-with-optional-prefix ('21940', 'image21940'),
    so sort on the embedded integer and fall back to lexicographic when there is none."""
    def key(fid: str):
        digits = "".join(ch for ch in fid if ch.isdigit())
        return (0, int(digits), fid) if digits else (1, 0, fid)
    return sorted(frame_ids, key=key)


def split_pose48(pose48: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(48,) axis-angle -> global_orient (3,) + hand_pose (45,).

    Confirmed against the repo's two other axis-angle MANO paths: h2o_to_currentproto.py:55
    ("pose48(3 global aa + 45 full aa, no PCA)") and oakink2_to_ours.py:60-61 (aa[:,0] is the
    global orient, aa[:,1:] the 15x3 finger pose). Not a guess about Re:InterHand specifically -
    it is the only split that is dimensionally consistent with MANO's 16-joint kinematic tree.
    """
    return pose48[:3].copy(), pose48[3:].copy()


def maybe_fix_left_shapedirs(mano: dict) -> bool:
    """Apply the known smplx LEFT-MANO shapedirs sign bug fix, if this smplx build has the bug.

    Older smplx releases ship MANO_LEFT with the first shape component's sign flipped; InterHand2.6M's
    own code patches it before fitting. Our ARCTIC and OakInk2 stores were built WITHOUT the patch,
    so applying it here makes Re:InterHand's left hand inconsistent with them - which is why this is
    opt-in (--fix_left_shapedirs) and reported, not silently on. Effect is a few mm on the left hand
    only. Returns True if a patch was actually applied.
    """
    left, right = mano["left"], mano["right"]
    sd_l, sd_r = left.shapedirs, right.shapedirs
    if sd_l.shape != sd_r.shape:
        return False
    if float(torch.sum(torch.abs(sd_l[:, 0, :] - sd_r[:, 0, :]))) < 1.0:
        with torch.no_grad():
            sd_l[:, 0, :] *= -1
        return True
    return False


# ------------------------------------------------------------------------------------- the gate

def anatomy_gate(j_cam: np.ndarray, valid: np.ndarray, tag: str) -> np.ndarray:
    """Refuse joints that are not a hand. Returns an [N] mask of frames to INVALIDATE; raises
    GateFailure when the problem is systematic.

    Two independent checks, because each is blind to the other's failure:
      * SPAN - no joint further than MAX_JOINT_TO_WRIST_M from its own wrist. Catches unit errors
        and a wrist written into a finger slot. A handful of glitchy fits in a 15k-frame capture
        is source noise, so those FRAMES are dropped; only a systematic rate (> SPAN_BAD_FRAC_MAX)
        fails the sequence, because that means the units or the frame convention are wrong, not
        the data.
      * BONE LENGTHS - median wrist->proximal distance per finger inside MCP_GATES, measured on
        the surviving frames. Catches a joint ORDER scramble, which the span check happily passes
        (a permuted hand is still a hand-sized point cloud). This is the exact check that caught
        the 2026-07-18 H2O 21->16 scramble AFTER it had voided every H2O number, so it runs
        before anything is written and it always fails the whole sequence: a scramble cannot be
        confined to a few frames.

    j_cam [N,16,3] metres, valid [N] bool, for ONE hand.
    """
    drop = np.zeros(len(valid), bool)
    m = valid & np.isfinite(j_cam).all((1, 2))
    if m.sum() < 10:
        return drop                             # too few samples for a meaningful median
    idx = np.flatnonzero(m)
    d = np.linalg.norm(j_cam[m] - j_cam[m][:, :1], axis=-1)          # [n,16] joint -> own wrist
    over = d.max(axis=1) > MAX_JOINT_TO_WRIST_M
    frac = float(over.mean())
    if frac > SPAN_BAD_FRAC_MAX:
        raise GateFailure(f"{tag}: {frac:.1%} of frames have a joint further than "
                          f"{MAX_JOINT_TO_WRIST_M * 1000:.0f} mm from its wrist (worst "
                          f"{float(d.max()) * 1000:.0f} mm) - wrong units, wrong frame, or a "
                          f"scrambled joint slot")
    if over.any():
        drop[idx[over]] = True
        print(f"    {tag}: dropped {int(over.sum())} frame(s) whose joints span > "
              f"{MAX_JOINT_TO_WRIST_M * 1000:.0f} mm")
    d = d[~over]
    if d.shape[0] < 10:
        return drop
    for jix, (jname, lo, hi) in MCP_GATES.items():
        dist = float(np.median(d[:, jix])) * 1000.0
        if not (lo <= dist <= hi):
            raise GateFailure(f"{tag}: median wrist->{jname} {dist:.1f} mm outside {lo:.0f}-{hi:.0f} mm "
                              f"- the 16-joint ORDER is wrong (H2O-scramble signature)")
    return drop


def depth_gate(j_cam: np.ndarray, valid: np.ndarray, tag: str) -> float:
    """Median wrist depth must be physically possible. Returns it (metres).

    This is the backstop for a metre/millimetre mix between the MANO `trans` and the ego camera
    translation: both are individually plausible, their COMBINATION is not.
    """
    m = valid & np.isfinite(j_cam[:, 0, 2])
    if m.sum() < 10:
        return float("nan")
    z = float(np.median(j_cam[m, 0, 2]))
    if not (Z_MIN_M < z < Z_MAX_M):
        raise GateFailure(f"{tag}: median wrist depth {z:.4f} m outside ({Z_MIN_M}, {Z_MAX_M}) - "
                          f"almost certainly a metre/millimetre mismatch between the MANO trans "
                          f"and the ego camera translation")
    return z


# -------------------------------------------------------------------------------- the conversion

def fill_hand_caches(fits: dict, frame_ids: list[str], world2cam: np.ndarray, mano: dict,
                     device: str, intr: tuple[float, float, float], mano_scale: float,
                     tag: str) -> tuple[np.ndarray, ...]:
    """FK both hands of every frame -> (cam_j, wld_j, j2d, gt64, valid), our cache tensors.

    cam_j / wld_j  [N,2,16,3] metres, ZEROS where the hand has no fit
    j2d            [N,2,16,3] (u, v, conf) px, single-focal pinhole
    gt64           [N,64] camera-frame [transl 3, quat_wxyz 4, pose15 PCA, betas 10] per hand
    valid          [N,2] bool

    THE ABSENT HAND IS ZEROS, NOT NaN, and that is load-bearing rather than cosmetic.
    Keypoint3DLoss weights the residual by a confidence channel (scripts/hamer_losses.py:105,
    `conf * self.loss_fn(pred, gt)`) which the train loop derives from gt64 being non-zero
    (train_hand_head.py:1442, 1448). But 0 * NaN = NaN in torch, verified, so a NaN-filled absent
    hand makes the loss NaN for the WHOLE BATCH no matter what the confidence says. Re:InterHand
    is two-handed but its frames are not: either hand can be unfitted. preprocess_hoi4d.py:372-373
    (also a one-hand-per-frame store) fills zeros for exactly this reason; arctic_to_ours.py:187-188
    and oakink2_to_ours.py:99-100 fill NaN, which is a latent hazard on THOSE stores and not a
    precedent to copy.

    Hand slot hi comes from HAND_IDX_TO_SIDE, i.e. 0 -> left, 1 -> right; that is the ONLY place
    Re:InterHand's per-file side label meets our hand axis.
    """
    f, cx, cy = intr
    N = len(frame_ids)
    mano_ns = types.SimpleNamespace(right=mano["right"], left=mano["left"])
    cam_j = np.zeros((N, NUM_HANDS, NUM_JOINTS, 3), np.float32)
    wld_j = np.zeros((N, NUM_HANDS, NUM_JOINTS, 3), np.float32)
    j2d = np.zeros((N, NUM_HANDS, NUM_JOINTS, 3), np.float32)
    gt64 = np.zeros((N, NUM_HANDS * HAND_PARAM_DIM), np.float32)
    valid = np.zeros((N, NUM_HANDS), bool)

    for hi in range(NUM_HANDS):
        side = HAND_IDX_TO_SIDE[hi]                                   # 0 -> left, 1 -> right
        rows, rot, pose, trans, shape = [], [], [], [], []
        for t, fid in enumerate(frame_ids):
            rec = fits.get(fid, {}).get(hi)
            if rec is None:
                continue
            g, hp = split_pose48(rec["pose"])
            rows.append(t)
            rot.append(g)
            pose.append(hp)
            trans.append(rec["trans"])
            shape.append(rec["shape"])
        if not rows:
            print(f"    {tag}: no {side}-hand fits")
            continue
        rot = np.asarray(rot, np.float64)
        pose = np.asarray(pose, np.float64)
        shape = np.asarray(shape, np.float64)
        trans = np.asarray(trans, np.float64) * mano_scale            # -> metres

        jw = fk_world_joints(mano[side], rot.astype(np.float32), pose.astype(np.float32),
                             trans.astype(np.float32), shape.astype(np.float32), device)
        w2c_rows = world2cam[rows]
        jc = apply_se3(w2c_rows.astype(np.float32), jw)
        jw16, jc16 = _to_smplx16(jw), _to_smplx16(jc)                 # [n,16,3] metres

        # Camera-frame wrist transform for the 32-dim param vector. Same exact identity ARCTIC and
        # OakInk2 use (transl_cam = R_wc @ trans_w + t_wc, R_cam = R_wc @ R_global).
        transl_cam, quat_cam = camera_frame_wrist_params(rot, trans, w2c_rows, mano_ns,
                                                         is_right=(hi == 1))
        pose15 = pose45_to_pca15(pose, mano_ns, is_right=(hi == 1))   # lossy 45->15, gt64 slot only

        off = hi * HAND_PARAM_DIM
        for k, t in enumerate(rows):
            finite = bool(np.isfinite(jc16[k]).all())
            # In front of the camera by >1 cm. Behind-camera joints have no image evidence and
            # blow up the 2D projection, so they are never supervised (arctic_to_ours.py:202-205).
            infront = bool((jc16[k, :, 2] > 1e-2).all())
            valid[t, hi] = finite and infront
            if not (finite and infront):
                continue                      # leave the row ZERO, never NaN (see the docstring)
            cam_j[t, hi] = jc16[k]
            wld_j[t, hi] = jw16[k]
            u, v = project_single_focal(jc16[k][None], f, cx, cy)
            j2d[t, hi, :, 0] = u[0]
            j2d[t, hi, :, 1] = v[0]
            j2d[t, hi, :, 2] = 1.0
            gt64[t, off:off + 3] = transl_cam[k]
            gt64[t, off + 3:off + 7] = quat_cam[k]
            gt64[t, off + 7:off + 22] = pose15[k].astype(np.float32)
            gt64[t, off + 22:off + 32] = shape[k, :10].astype(np.float32)
    return cam_j, wld_j, j2d, gt64, valid


def write_ego_video(out_seq: str, paths: list, tag: str) -> tuple[int, int, int]:
    """Write video_main_rgb.mp4, EXACTLY one frame per cache row. -> (W, H, n_black).

    A missing image is black-padded, never skipped: skipping shifts every later frame against its
    label, and "video frame t == cache row t" is the contract HOT3DHandDataset assumes when it
    slices clips by frame_offset. Same two-pass shape as oakink2_to_ours.
    """
    first = next((p for p in paths if p), None)
    if first is None:
        raise GateFailure(f"{tag}: no ego image resolved for any of {len(paths)} frames")
    H, W = cv2.imread(first).shape[:2]
    vw = cv2.VideoWriter(os.path.join(out_seq, "video_main_rgb.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (W, H))
    n_black = 0
    for p in paths:
        im = cv2.imread(p) if p else None
        if im is None:
            im = np.zeros((H, W, 3), np.uint8)
            n_black += 1
        vw.write(im)
    vw.release()
    return W, H, n_black


def convert_capture_camera(capture: str, cam: EgoCamera, fits: dict, frame_ids: list[str],
                           mano: dict, args, out_root: str, device: str) -> dict:
    """Convert one (capture, ego camera) pair into one sequence directory. Raises GateFailure.

    Returns the sequence's stats, or None if it was already complete and was skipped. The ego
    split is 217 GB / 147,745 images, so this WILL be interrupted: output is built in a .tmp_ dir
    and atomically renamed, and a complete dir is skipped on re-run. Same resume contract as
    h2o_to_currentproto, and it also means a sequence that fails the post-video K gate below
    leaves no half-written directory behind for the trainer to trip over.
    """
    N = len(frame_ids)
    seq_id = f"{capture}__{cam.cam_id}"
    tag = seq_id
    out_seq_final = os.path.join(out_root, seq_id)
    if (not args.overwrite
            and os.path.exists(os.path.join(out_seq_final, "hand_data",
                                            "gt_joints_cache_cam_v2.pt"))):
        print(f"    {tag}: already complete, skipped (--overwrite to redo)")
        return None

    # Per-frame world->camera. Frames the camera has no pose for are dropped up front rather than
    # written with an identity extrinsic: an identity camera pose silently made every world/W
    # metric meaningless once already (memory: identity-camera-pose-bug, fixed 9dd474d).
    w2c_list, kept = [], []
    for t, fid in enumerate(frame_ids):
        m = cam.w2c_at(fid)
        if m is None:
            continue
        w2c_list.append(m)
        kept.append(fid)
    if not kept:
        raise GateFailure(f"{tag}: no frame has an ego camera pose")
    if len(kept) != N:
        print(f"    {tag}: {N - len(kept)}/{N} frames dropped (no ego camera pose)")
    frame_ids = kept
    N = len(frame_ids)
    world2cam = np.asarray(w2c_list, np.float64)                      # [N,4,4]

    K = cam.K
    if not np.isfinite(K).all():
        raise GateFailure(f"{tag}: ego intrinsics contain non-finite entries {K.tolist()}")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Our store format is SINGLE-FOCAL ([f, cx, cy]); preprocess_hoi4d and h2o_to_currentproto
    # both take fx. Report the anisotropy instead of hiding it - h2o gate 1 fails above 1 px.
    aniso = abs(fx - fy) / max(fx, 1e-9)
    if aniso > 0.02:
        print(f"    {tag}: WARNING fx/fy anisotropy {aniso:.3%} (fx={fx:.2f} fy={fy:.2f}); the store "
              f"is single-focal and keeps fx")
    f = fx

    cam_j, wld_j, j2d, gt64, valid = fill_hand_caches(
        fits, frame_ids, world2cam, mano, device, (f, cx, cy), args._mano_scale, tag)

    if not valid.any():
        raise GateFailure(f"{tag}: no valid hand in any frame")

    # GATES BEFORE ANY WRITE, so the store only ever contains validated sequences. The anatomy
    # gate may invalidate individual frames; the depth gate then runs on what survived.
    stats = {"seq": seq_id, "N": N}
    for hi in range(NUM_HANDS):
        if valid[:, hi].sum() < 10:
            continue
        side = HAND_IDX_TO_SIDE[hi]
        valid[anatomy_gate(cam_j[:, hi], valid[:, hi], f"{tag}[{side}]"), hi] = False
        stats[f"median_wrist_z_{side}"] = round(depth_gate(cam_j[:, hi], valid[:, hi],
                                                           f"{tag}[{side}]"), 4)
    if not valid.any():
        raise GateFailure(f"{tag}: the anatomical gate invalidated every hand")

    # Video. REQUIRED by the consumer; see write_ego_video for the row-alignment contract.
    out_seq = os.path.join(out_root, f".tmp_{seq_id}")            # renamed into place at the end
    shutil.rmtree(out_seq, ignore_errors=True)
    hd = os.path.join(out_seq, "hand_data")
    os.makedirs(hd, exist_ok=True)
    paths = [resolve_ego_image(args.ego_image_pattern, args.reinterhand_root, capture,
                               cam.cam_id, fid) for fid in frame_ids]
    try:
        W, H, n_black = write_ego_video(out_seq, paths, tag)
    except GateFailure as e:
        # The image layout is the second UNVERIFIED input, so say exactly what was tried and
        # what is really on disk instead of leaving the operator to guess the pattern.
        raise GateFailure(
            f"{e}. The Ego_cameras image layout is UNVERIFIED - fix --ego_image_pattern.\n"
            + diagnose_image_pattern(args.reinterhand_root, capture, args.ego_image_pattern))
    if n_black:
        print(f"    {tag}: black-padded {n_black}/{N} frames (image missing; row alignment kept)")
    stats["black_frames"] = n_black
    stats["res"] = (W, H)

    # Intrinsics must describe the IMAGE WE ACTUALLY HAVE. Ego_cameras may publish K at the render
    # resolution while the shipped frames are downscaled; that mismatch is a silent 12.5%-class
    # error on every crop box (egoexo4d_to_ours.py:114-122). We cannot rescale without knowing the
    # published resolution, so we CHECK and refuse rather than quietly proceed.
    implied_w, implied_h = 2.0 * cx, 2.0 * cy
    if abs(implied_w - W) > 0.02 * W or abs(implied_h - H) > 0.02 * H:
        raise GateFailure(
            f"{tag}: ego K implies a {implied_w:.0f}x{implied_h:.0f} image but the frames are "
            f"{W}x{H}. Rescale K to the shipped resolution (see egoexo4d_to_ours.rescale_K_to_video) "
            f"once the true render resolution is known - do NOT store a K that disagrees with the "
            f"pixels.")

    # Boxes, in the ONE training convention: imported joints_to_bbox, rectangular per-axis x1.5,
    # UNCLAMPED. Never reimplement this locally (verify_box_store.py:5-9).
    boxes = np.zeros((N, NUM_HANDS, 4), np.float32)
    for t in range(N):
        for hi in range(NUM_HANDS):
            if not valid[t, hi]:
                continue
            b = joints_to_bbox(cam_j[t, hi], f, cx, cy, float(W), float(H), rf=RF)
            if b is None or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
                valid[t, hi] = False
                continue
            boxes[t, hi] = b

    # EVERY cache row of an invalid hand is zeroed, in one place, after the last thing that can
    # invalidate a hand (the box builder above). Two separate reasons, both load-bearing:
    #   * gt64 zeros are how the consumer detects an absent hand - the train loop derives the
    #     keypoint confidence from `gt_pack.abs().sum(dim=-1) > 1e-6` (train_hand_head.py:1442)
    #     and its fallback FK from `p.abs().sum() < 1e-6` (:569). A non-zero param row behind
    #     valid=False would resurrect a hand the store says is not there.
    #   * joint zeros (never NaN) keep the loss finite: 0 * NaN = NaN, see fill_hand_caches.
    cam_j[~valid] = 0.0
    wld_j[~valid] = 0.0
    j2d[~valid] = 0.0
    for hi in range(NUM_HANDS):
        off = hi * HAND_PARAM_DIM
        gt64[~valid[:, hi], off:off + HAND_PARAM_DIM] = 0.0
    if not np.isfinite(cam_j).all() or not np.isfinite(gt64).all():
        raise GateFailure(f"{tag}: non-finite value survived into the caches - refusing to write, "
                          f"a single NaN turns the whole training batch's loss into NaN")

    # jsonl: the raw MANO trajectory the consumer parses on a bbox cache MISS
    # (train_hand_head.py:283, 338). hand ids "0" = left, "1" = right, matching _hand_to_vec.
    jl = []
    for t in range(N):
        hp = {}
        for hi in range(NUM_HANDS):
            if not valid[t, hi]:
                continue
            off = hi * HAND_PARAM_DIM
            v = gt64[t, off:off + HAND_PARAM_DIM]
            hp[str(hi)] = {"wrist_xform": {"t_xyz": [float(x) for x in v[:3]],
                                           "q_wxyz": [float(x) for x in v[3:7]]},
                           "pose": [float(x) for x in v[7:22]],
                           "betas": [float(x) for x in v[22:32]]}
        jl.append({"timestamp_ns": t, "hand_poses": hp})

    torch.save(torch.tensor([f, cx, cy], dtype=torch.float32), os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.from_numpy(world2cam.astype(np.float32)),
               os.path.join(hd, "cam_extrinsics_cache.pt"))
    torch.save(torch.from_numpy(cam_j), os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(torch.from_numpy(wld_j), os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(torch.from_numpy(j2d), os.path.join(hd, "gt_joints_2d_cache.pt"))
    with open(os.path.join(hd, "mano_hand_pose_trajectory.jsonl"), "w") as fjl:
        for e in jl:
            fjl.write(json.dumps(e) + "\n")
    # All three keys are required; "gt" carries the camera-frame MANO params and dropping it
    # KeyErrors every eval on the store (verify_box_store.py:10-13).
    torch.save({"bboxes": torch.from_numpy(boxes),
                "valid": torch.from_numpy(valid),
                "gt": torch.from_numpy(gt64),
                "convention": "joints_to_bbox rf1.5 rectangular unclamped (reinterhand)"},
               os.path.join(hd, f"hand_bboxes_v2_rf{RF}_res{BOX_RES}x{BOX_RES}.pt"))

    # Durable record of what was ASSUMED, so a later reader does not mistake it for measurement.
    with open(os.path.join(out_seq, ".reinterhand_meta.json"), "w") as fm:
        json.dump({"dataset": "Re:InterHand (Moon et al., NeurIPS 2023) - egocentric split",
                   "license": "CC-BY-NC 4.0 (non-commercial; inherited by anything derived)",
                   "capture": capture, "ego_cam": cam.cam_id, "frames": N,
                   "frame_ids": [frame_ids[0], frame_ids[-1]],
                   "hand_axis": "index 0 = LEFT, 1 = RIGHT (train_hand_head.py:861-862)",
                   "joints": "smplx-16 via scripts.arctic_to_ours._to_smplx16",
                   "boxes": "joints_to_bbox rf1.5 rectangular UNCLAMPED",
                   "units": "metres",
                   "mano_trans_scale_applied": args._mano_scale,
                   "left_shapedirs_patched": bool(args._left_patched),
                   "UNVERIFIED": ["ego camera JSON layout (load_ego_cameras)",
                                  "ego image path layout (resolve_ego_image / --ego_image_pattern)"],
                   "ego_cam_json": args.ego_cam_json,
                   "ego_image_pattern": args.ego_image_pattern},
                  fm, indent=2)

    # Atomic publish: until this line the sequence does not exist as far as the trainer is
    # concerned, so an interrupted run can never leave a partially-cached sequence in the store.
    shutil.rmtree(out_seq_final, ignore_errors=True)
    os.replace(out_seq, out_seq_final)

    stats["valid_rate"] = float(valid.mean())
    stats["boxes"] = int(valid.sum())
    return stats


def validate_capture(capture: str, fits: dict, mano: dict, device: str, mano_scale: float) -> None:
    """No-write sanity pass: FK a handful of frames per hand and print bone lengths.

    Re:InterHand publishes no precomputed joints, so - exactly as for OakInk2 - the MANO
    convention is checked anatomically. This needs NO ego camera data, so it can be run while
    the Ego_cameras payload is still downloading. Bone lengths are frame-invariant, so a pass
    here validates the MANO half of the pipeline on its own.
    """
    for hi in range(NUM_HANDS):
        side = HAND_IDX_TO_SIDE[hi]
        rot, pose, trans, shape = [], [], [], []
        for fid in sort_frame_ids(fits.keys())[:32]:
            rec = fits[fid].get(hi)
            if rec is None:
                continue
            g, hp = split_pose48(rec["pose"])
            rot.append(g)
            pose.append(hp)
            trans.append(rec["trans"] * mano_scale)
            shape.append(rec["shape"])
        if not rot:
            print(f"VALIDATE {capture}[{side}]: no fits")
            continue
        jw = fk_world_joints(mano[side], np.asarray(rot, np.float32), np.asarray(pose, np.float32),
                             np.asarray(trans, np.float32), np.asarray(shape, np.float32), device)
        j16 = _to_smplx16(jw)
        d = np.linalg.norm(j16 - j16[:, :1], axis=-1)
        bones = {name: f"{float(np.median(d[:, jix])) * 1000:.1f}"
                 for jix, (name, _, _) in MCP_GATES.items()}
        span = float(np.median(d.max(axis=1))) * 1000
        print(f"VALIDATE {capture}[{side}]: n={len(rot)} bones(mm)={bones} "
              f"max joint->wrist {span:.1f} mm (expect 150-230); median |trans| "
              f"{float(np.median(np.linalg.norm(np.asarray(trans), axis=-1))):.4f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reinterhand_root", required=True, help="download root holding the capture dirs")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--mano_dir", required=True, help="MANO model dir (smplx.create model_path)")
    ap.add_argument("--captures", default="", help="comma-separated capture ids (default: all found)")
    ap.add_argument("--ego_cam_json", default="{capture}/Ego_cameras/cam_params.json",
                    help="ego camera JSON, relative to --reinterhand_root, {capture} placeholder. "
                         "LAYOUT UNVERIFIED - see load_ego_cameras()")
    ap.add_argument("--ego_image_pattern",
                    default="{capture}/Ego_cameras/envmap_per_frame/{cam}/{frame}.jpg",
                    help="ego image path template with {capture} {cam} {frame}, relative to "
                         "--reinterhand_root. LAYOUT UNVERIFIED - see resolve_ego_image()")
    ap.add_argument("--ego_cams", default="", help="comma-separated ego camera ids (default: all)")
    ap.add_argument("--mano_units", default="auto", choices=["auto", "m", "mm"],
                    help="units of the MANO `trans` field (auto-detected by magnitude)")
    ap.add_argument("--cam_units", default="auto", choices=["auto", "m", "mm"],
                    help="units of the ego camera translation (InterHand2.6M publishes mm)")
    ap.add_argument("--fix_left_shapedirs", action="store_true",
                    help="patch the known smplx LEFT-MANO shapedirs sign bug. OFF by default "
                         "because ARCTIC/OakInk2 were built without it and consistency across the "
                         "mixture matters more than a few mm on one hand")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke run: handle at most N sequences (0 = all); already-complete "
                         "sequences count toward it, so a re-run with --limit stays a no-op")
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames per sequence (0 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="reconvert sequences that are already complete (default: resume)")
    ap.add_argument("--validate", action="store_true",
                    help="MANO-only anatomical check, writes nothing, needs no ego camera data")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano = build_mano(args.mano_dir, device)
    args._left_patched = maybe_fix_left_shapedirs(mano) if args.fix_left_shapedirs else False
    if args.fix_left_shapedirs:
        print(f"left shapedirs patch applied = {args._left_patched} (store diverges from "
              f"ARCTIC/OakInk2 when True)")

    captures = ([c for c in args.captures.split(",") if c] or
                sorted(d for d in os.listdir(args.reinterhand_root)
                       if os.path.isdir(os.path.join(args.reinterhand_root, d, "mano_fits"))))
    if not captures:
        raise SystemExit(f"no capture dirs with mano_fits/ under {args.reinterhand_root}")
    print(f"captures: {len(captures)} -> {captures}")
    want_cams = [c for c in args.ego_cams.split(",") if c]
    os.makedirs(args.out_root, exist_ok=True)

    n_ok, n_gate, n_fail, n_skip, done = 0, 0, 0, 0, []
    for capture in captures:
        cap_dir = os.path.join(args.reinterhand_root, capture)
        print(f"[{capture}]")
        fits = parse_params_dir(os.path.join(cap_dir, "mano_fits", "params"))
        if not fits:
            print("    no MANO fits, skipped")
            continue
        frame_ids = sort_frame_ids(fits.keys())
        if args.max_frames:
            frame_ids = frame_ids[:args.max_frames]

        # MANO trans units, detected once per capture over every fit it holds. A capture whose
        # translations are neither metre- nor millimetre-scale is skipped rather than converted
        # under a guessed scale - a 1000x error here is invisible in every downstream tensor.
        all_trans = np.asarray([r["trans"] for fr in fits.values() for r in fr.values()])
        try:
            args._mano_scale = _translation_scale(all_trans, args.mano_units,
                                                  what=f"mano trans [{capture}]")
        except GateFailure as e:
            print(f"    GATE_FAIL {capture}: {e}")
            n_gate += 1
            continue

        if args.validate:
            validate_capture(capture, fits, mano, device, args._mano_scale)
            continue

        cam_json = os.path.join(args.reinterhand_root,
                                args.ego_cam_json.format(capture=capture))
        try:
            cams = load_ego_cameras(cam_json, trans_units=args.cam_units)
        except (OSError, EgoLayoutUnknown, GateFailure) as e:
            # The ego camera payload is the one UNVERIFIED input. Say so loudly instead of
            # emitting a store with identity extrinsics, which reads as a bad model, not bad data.
            print(f"    EGO CAMERA UNAVAILABLE/UNKNOWN: {e}")
            print(f"    -> confirm the Ego_cameras layout and fix load_ego_cameras(); "
                  f"NOTHING was written for {capture}")
            n_fail += 1
            continue
        for cam_id in sorted(cams):
            if want_cams and cam_id not in want_cams:
                continue
            if args.limit and (n_ok + n_skip) >= args.limit:
                break
            try:
                st = convert_capture_camera(capture, cams[cam_id], fits, frame_ids, mano,
                                            args, args.out_root, device)
                if st is None:                           # already complete, resume path
                    n_skip += 1
                    continue
                n_ok += 1
                done.append(st)
                print(f"    [{n_ok}] {st['seq']} N={st['N']} res={st['res']} "
                      f"valid={st['valid_rate']:.2f} z_med="
                      f"{st.get('median_wrist_z_right', st.get('median_wrist_z_left', float('nan')))}")
            except GateFailure as e:
                n_gate += 1
                print(f"    GATE_FAIL {capture}__{cam_id}: {e}")
            except Exception as e:                       # noqa: BLE001 - one bad seq must not kill the run
                n_fail += 1
                print(f"    SEQ_FAIL {capture}__{cam_id}: {type(e).__name__}: {e}")
        if args.limit and (n_ok + n_skip) >= args.limit:
            print(f"--limit {args.limit} reached, stopping")
            break

    if args.validate:
        return
    print(f"\nREINTERHAND_TO_OURS_DONE wrote {n_ok} seqs -> {args.out_root} "
          f"(already complete {n_skip}, gate-rejected {n_gate}, errored {n_fail})")
    if done:
        zs = [v for st in done for k, v in st.items() if k.startswith("median_wrist_z")]
        if zs:
            print(f"median wrist depth across sequences: {float(np.median(zs)):.3f} m "
                  f"(run reinterhand_depth_stat.py for the real distribution)")
    # A few gate rejections are source noise. A large fraction means the joint MAPPING or the unit
    # handling is wrong, not the data - the H2O-scramble signature - and must stop everything.
    total = n_ok + n_gate
    if n_gate and total and n_gate / total > GATE_ABORT_FRAC:
        raise SystemExit(f"ABORT: {n_gate}/{total} sequences failed a correctness gate "
                         f"({n_gate / total:.1%}). At that rate the joint mapping or the unit "
                         f"handling is wrong, not the source data. Do NOT train on this store.")
    print("Next: python -m scripts.verify_box_store --data_root " + args.out_root)


if __name__ == "__main__":
    main()
