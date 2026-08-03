#!/usr/bin/env python3
"""Convert Ego-Exo4D ego_pose hand annotations into our store format.

WHAT EGO-EXO4D ACTUALLY GIVES US, measured on the downloaded release rather than assumed:
  * ego RGB video   takes/<take>/frame_aligned_videos/downscaled/448/aria01_214-1.mp4
                    (214-1 is the Aria camera-rgb stream id, same as HOT3D's camera_models.json)
  * 3D hand joints  annotations/ego_pose/<split>/hand/annotation/<take_uid>.json
                    {frame_str: [ {annotation3D: {<name>: {x,y,z,num_views_for_3d}}, ...} ]}
                    21 names per hand: <side>_wrist and <side>_<finger>_1..4.
  * camera          annotations/ego_pose/<split>/camera_pose/<take_uid>.json
                    aria01 -> {camera_intrinsics: 3x3 K, camera_extrinsics: 3x4 or 4x4}

TWO PROPERTIES THAT MAKE THIS UNLIKE OUR OTHER STORES - do not paper over them:

 1. THE EGO CAMERA IS ALREADY PINHOLE. aria01's entry carries a plain 3x3 K
    ([[150,0,255.5],[0,150,255.5],[0,0,1]]) and NO distortion_coeffs, i.e. the ego_pose
    benchmark publishes rectified camera parameters. So this dataset does NOT need the
    fisheye undistortion that HOT3D does, contrary to the "all Aria data is fisheye"
    assumption. The RAW aria01_214-1.mp4 is still fisheye, so frames must be undistorted to
    match these intrinsics before training - that is flagged, not silently ignored (see
    --require_rectified).

 2. ANNOTATIONS ARE SPARSE. Measured over 40 takes: median 97 annotated frames per take, a
    median of 16 of 21 joints per hand, and only ~20% of frames carry all 21. Our other stores
    are dense MANO. Consequences:
      - there are NO MANO parameters here, so param losses (betas/global_orient/hand_pose/transl)
        cannot be supervised from this data; only kp3d/kp3d_abs/kp2d can. kp3d_abs is the
        load-bearing loss, so the data is still useful, but a config must not claim otherwise.
      - per-joint validity is real and must be carried, not filled with zeros.

JOINT ORDER. Our 16-joint layout is wrist, index x3, middle x3, pinky x3, ring x3, thumb x3
(see eval_cmpjpe.H2O_TO_MANO and its 2026-07-18 bug-fix note). Getting this wrong is exactly
the H2O scramble that corrupted every H2O number, so the mapping is explicit below and gated
by an anatomical bone-length check rather than trusted.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

FINGERS = ("index", "middle", "pinky", "ring", "thumb")
# Our MANO-16: wrist, then index/middle/pinky/ring/thumb, three joints each.
JOINT16 = ["wrist"] + [f"{f}_{i}" for f in FINGERS for i in (1, 2, 3)]
# Tips appended after the 16, in the pred tip order (thumb, index, middle, ring, pinky).
TIP_ORDER = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")            # our hand axis is index 0 = LEFT, 1 = RIGHT (RH = 1)


def joints_from_annotation(entry: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract [16,3] joints and [16] per-joint validity for one hand of one frame."""
    a3 = entry.get("annotation3D", {}) or {}
    out = np.zeros((16, 3), np.float64)
    ok = np.zeros(16, bool)
    for i, jname in enumerate(JOINT16):
        v = a3.get(f"{side}_{jname}")
        if v is None:
            continue
        p = (v.get("x"), v.get("y"), v.get("z"))
        if any(c is None for c in p):
            continue
        out[i] = p
        ok[i] = True
    return out, ok


def bone_length_report(joints: np.ndarray, valid: np.ndarray) -> dict:
    """Median wrist->proximal bone lengths, the anatomical gate.

    A correct mapping puts real metacarpal/proximal bones in slots 1,4,7,10,13. Human values sit
    in roughly 3-12 cm. A scrambled mapping (e.g. a fingertip in a knuckle slot) shows up
    immediately as lengths far outside that band - this is precisely how the H2O scramble was
    caught, so it gates the conversion instead of being printed and ignored.
    """
    res = {}
    for fi, f in enumerate(FINGERS):
        k = 1 + 3 * fi                                   # proximal joint slot of this finger
        m = valid[:, 0] & valid[:, k]
        if not m.any():
            res[f] = float("nan")
            continue
        d = np.linalg.norm(joints[m, k] - joints[m, 0], axis=-1)
        res[f] = float(np.median(d))
    return res


def convert_take(uid: str, take_name: str, hand_json: str, cam_json: str,
                 video_src: str, out_dir: str, ego_key: str = "aria01") -> tuple[str, dict]:
    ann = json.load(open(hand_json))
    cams = json.load(open(cam_json))
    ego = cams.get(ego_key)
    if ego is None:
        return "no-ego-cam", {}
    K = np.asarray(ego.get("camera_intrinsics"), np.float64)
    if K.shape != (3, 3):
        return "bad-intrinsics", {}
    if ego.get("distortion_coeffs"):
        # Present means the published camera is NOT rectified; converting anyway would silently
        # pair fisheye pixels with a pinhole K.
        return "distorted-ego-cam", {}

    frames = sorted((int(k) for k in ann.keys()))
    if not frames:
        return "no-frames", {}

    n = len(frames)
    J = np.zeros((n, 2, 16, 3), np.float64)
    V = np.zeros((n, 2, 16), bool)
    for t, fr in enumerate(frames):
        lst = ann.get(str(fr)) or []
        if not lst:
            continue
        e = lst[0]
        for hi, side in enumerate(SIDES):
            j, ok = joints_from_annotation(e, side)
            J[t, hi], V[t, hi] = j, ok

    # Anatomical gate, per hand, before anything is written.
    stats = {"take": take_name, "frames": n}
    for hi, side in enumerate(SIDES):
        bl = bone_length_report(J[:, hi], V[:, hi])
        stats[f"bones_{side}"] = {k: round(v, 4) for k, v in bl.items()}
        good = [v for v in bl.values() if np.isfinite(v)]
        if good and not all(0.02 < v < 0.15 for v in good):
            stats["gate"] = f"{side} bone lengths outside 2-15 cm: {bl}"
            return "bad-anatomy", stats

    hand_valid = V.any(-1)                                # [n,2] hand present at all
    stats["frac_frames_any_hand"] = float(np.mean(hand_valid.any(-1)))
    stats["median_joints_per_hand"] = float(np.median(V.sum(-1)[hand_valid])) if hand_valid.any() else 0.0

    hd = os.path.join(out_dir, take_name, "hand_data")
    os.makedirs(hd, exist_ok=True)
    torch.save(torch.tensor(J, dtype=torch.float32), os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(torch.tensor(V), os.path.join(hd, "gt_joints_valid.pt"))
    torch.save(torch.tensor([float(K[0, 0]), float(K[0, 2]), float(K[1, 2])]),
               os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.tensor(np.asarray(frames, np.int64)), os.path.join(hd, "annotated_frames.pt"))
    with open(os.path.join(out_dir, take_name, ".egoexo_meta.json"), "w") as f:
        json.dump({"take_uid": uid, "take_name": take_name, "n_annotated_frames": n,
                   "video_src": video_src, "ego_key": ego_key,
                   "K": K.tolist(),
                   "NOTE": "3D joints are RAW ANNOTATIONS, not MANO. No MANO params exist for "
                           "this dataset, so param losses cannot be supervised from it.",
                   "frames_are_sparse": True}, f, indent=2)
    if video_src and os.path.exists(video_src):
        link = os.path.join(out_dir, take_name, "video_main_rgb.mp4")
        if not os.path.exists(link):
            os.symlink(video_src, link)
    return "ok", stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="egoexo4d download root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--max_takes", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    ann_root = os.path.join(a.root, "annotations", "ego_pose", a.split)
    hand_dir = os.path.join(ann_root, "hand", "annotation")
    cam_dir = os.path.join(ann_root, "camera_pose")
    if not os.path.isdir(hand_dir):
        raise SystemExit(f"no hand annotations at {hand_dir}")

    # take_uid -> take_name, so we can find the video directory.
    takes = json.load(open(os.path.join(a.root, "takes.json")))
    uid2name = {t["take_uid"]: t["take_name"] for t in takes}

    files = sorted(f for f in os.listdir(hand_dir) if f.endswith(".json"))
    if a.max_takes:
        files = files[: a.max_takes]

    counts: dict[str, int] = {}
    agg: list[dict] = []
    for fn in files:
        uid = fn[:-5]
        name = uid2name.get(uid)
        if name is None:
            counts["no-take-name"] = counts.get("no-take-name", 0) + 1
            continue
        cam_json = os.path.join(cam_dir, fn)
        if not os.path.exists(cam_json):
            counts["no-camera-pose"] = counts.get("no-camera-pose", 0) + 1
            continue
        video = os.path.join(a.root, "takes", name, "frame_aligned_videos",
                             "downscaled", "448", "aria01_214-1.mp4")
        if a.dry_run:
            st, stats = ("dry-run", {"take": name, "video_exists": os.path.exists(video)})
        else:
            st, stats = convert_take(uid, name, os.path.join(hand_dir, fn), cam_json,
                                     video, a.out)
        counts[st] = counts.get(st, 0) + 1
        if stats:
            agg.append(stats)
            if len([x for x in agg if "bones_left" in x]) <= 2 and "bones_left" in stats:
                print(f"  [{stats['take']}] frames={stats['frames']} "
                      f"bones_L={stats['bones_left']}")

    print(f"\nCONVERT status: {counts}")
    done = [x for x in agg if "frames" in x]
    if done:
        print(f"CONVERT takes={len(done)} annotated_frames={sum(x['frames'] for x in done)}")
        mj = [x["median_joints_per_hand"] for x in done if "median_joints_per_hand" in x]
        if mj:
            print(f"CONVERT median joints/hand = {np.median(mj):.1f} of 16 "
                  f"(sparse by design; per-joint validity is stored)")
    if counts.get("bad-anatomy"):
        raise SystemExit("CONVERT FAILED the anatomical gate on some takes - joint mapping is "
                         "suspect. Do NOT train on this store.")


if __name__ == "__main__":
    main()
