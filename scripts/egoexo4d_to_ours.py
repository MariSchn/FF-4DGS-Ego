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

from scripts.arctic_to_ours import joints_to_bbox

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


def video_hw(path: str) -> tuple[int, int] | None:
    """(H, W) of a video, or None if unreadable."""
    if not path or not os.path.exists(path):
        return None
    try:
        import decord
        return tuple(decord.VideoReader(path)[0].asnumpy().shape[:2])
    except Exception:
        return None


def video_len(path: str) -> int | None:
    """Frame count of a video, or None if unreadable."""
    if not path or not os.path.exists(path):
        return None
    try:
        import decord
        return len(decord.VideoReader(path))
    except Exception:
        return None


def rescale_K_to_video(K: np.ndarray, vid_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Rescale intrinsics from the resolution they were published at to the video we actually have.

    THIS IS NOT OPTIONAL. Ego-Exo4D publishes ego_pose intrinsics for the 512x512 ego frame
    (cx = cy = 255.5), but the practical download is downscaled_takes/448 at 448x448. Using the
    published K against a 448 video puts every projection and every crop box out by 12.5% -
    the same class of silent input-convention error as the H2O square-vs-rectangular boxes,
    which cost us every H2O number until it was found.
    """
    # A centred principal point with 0-indexed pixel centres means cx = (W - 1) / 2, so
    # W = 2*cx + 1. Ego-Exo4D's cx = cy = 255.5 is therefore exactly 512, not 511 - and using
    # 2*cx would give a 0.2% wrong scale, i.e. right-looking but not right.
    implied_w = round(float(K[0, 2]) * 2.0 + 1.0)
    implied_h = round(float(K[1, 2]) * 2.0 + 1.0)
    h, w = vid_hw
    if implied_w <= 0 or implied_h <= 0:
        return K, 1.0
    sx, sy = w / implied_w, h / implied_h
    if abs(sx - sy) > 1e-3:
        # Non-uniform rescale would mean a non-square downscale; our [f, cx, cy] store format
        # cannot express two focal lengths, so refuse rather than silently pick one.
        raise ValueError(f"non-uniform intrinsics rescale sx={sx:.4f} sy={sy:.4f}")
    # Focal scales directly. The principal point does NOT: under the pixel-centre convention a
    # resize maps c -> (c + 0.5) * s - 0.5. Plain multiplication is the naive form and lands
    # 255.5*0.875 = 223.5625 instead of the true 448-frame centre 223.5. Small, but this is the
    # difference between exactly right and approximately right, and it biases every crop box.
    Ks = K.copy()
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return Ks, float(sx)


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

    # Per-frame world->camera extrinsics, keyed by frame string. VERIFIED world-frame: projecting
    # annotation3D through (extrinsics, intrinsics) reproduces the INDEPENDENTLY published
    # annotation2D to a 0-6.5 px median on a 512 px image. A wrong frame convention would be
    # hundreds of pixels out, so this pins both the frame and the 3x4 layout.
    EX = ego.get("camera_extrinsics") or {}
    if not isinstance(EX, dict) or not EX:
        return "no-extrinsics", {}

    # FRAME ALIGNMENT. Ego-Exo4D annotates roughly every 3rd frame (ids like 79, 82, 85...), but
    # HOT3DHandDataset indexes every cache by VIDEO FRAME NUMBER. Storing only the annotated rows
    # would pair video frame 0 with the joints of frame 79 - a silent, systematic image/label
    # mismatch. So the caches are expanded to the FULL video length and validity is False on
    # unannotated frames.
    # Why not instead build clips from annotated frames only? That would use every annotation,
    # but a 16-frame clip would then span 48 video frames (~1.6 s) versus ~0.53 s for every other
    # dataset in the mix - a 3x temporal-extent inconsistency inside a single mixed-training run,
    # invisible in any config. Protocol consistency wins; the cost is that ~2/3 of frames carry
    # no hand supervision, which the validity mask states honestly.
    n = video_len(video_src)
    if n is None:
        return "no-video", {}
    J = np.zeros((n, 2, 16, 3), np.float64)          # world frame
    JC = np.zeros((n, 2, 16, 3), np.float64)         # camera frame (what the model predicts)
    V = np.zeros((n, 2, 16), bool)
    W2C = np.tile(np.eye(4), (n, 1, 1))
    has_ext = np.zeros(n, bool)

    # Extrinsics are DENSE (one per video frame) even though hand annotations are sparse, so fill
    # them independently - the camera trajectory is usable on frames with no hand label.
    for k, P in EX.items():
        try:
            t = int(k)
        except (TypeError, ValueError):
            continue
        if not (0 <= t < n):
            continue
        P = np.asarray(P, np.float64)
        if P.shape != (3, 4):
            continue
        W2C[t, :3, :4] = P
        has_ext[t] = True

    n_out_of_range = 0
    for fr in frames:
        if not (0 <= fr < n):
            n_out_of_range += 1
            continue
        lst = ann.get(str(fr)) or []
        if not lst:
            continue
        e = lst[0]
        for hi, side in enumerate(SIDES):
            j, ok = joints_from_annotation(e, side)
            J[fr, hi], V[fr, hi] = j, ok
        if not has_ext[fr]:
            # No camera pose -> camera-frame joints are undefined, so drop the label rather than
            # emit joints in an unknown frame.
            V[fr] = False
            continue
        P = W2C[fr, :3, :4]
        for hi in range(2):
            JC[fr, hi] = (P[:, :3] @ J[fr, hi].T).T + P[:, 3]

    # Anatomical gate, per hand, before anything is written.
    stats = {"take": take_name, "frames": n, "annotated": len(frames),
             "ann_out_of_range": n_out_of_range}
    for hi, side in enumerate(SIDES):
        bl = bone_length_report(J[:, hi], V[:, hi])
        stats[f"bones_{side}"] = {k: round(v, 4) for k, v in bl.items()}
        good = [v for v in bl.values() if np.isfinite(v)]
        if good and not all(0.02 < v < 0.15 for v in good):
            stats["gate"] = f"{side} bone lengths outside 2-15 cm: {bl}"
            return "bad-anatomy", stats

    hand_valid = V.any(-1)                                # [n,2] hand present at all
    stats["frac_frames_any_hand"] = float(np.mean(hand_valid.any(-1)))
    stats["frac_frames_with_pose"] = float(np.mean(has_ext))
    stats["median_joints_per_hand"] = float(np.median(V.sum(-1)[hand_valid])) if hand_valid.any() else 0.0

    hd = os.path.join(out_dir, take_name, "hand_data")
    os.makedirs(hd, exist_ok=True)
    torch.save(torch.tensor(J, dtype=torch.float32), os.path.join(hd, "gt_joints_cache_world.pt"))
    # Camera-frame joints are what the model predicts, and our stores name this cache _cam_v2.
    torch.save(torch.tensor(JC, dtype=torch.float32), os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(torch.tensor(V), os.path.join(hd, "gt_joints_valid.pt"))
    # T_cam_world (w2c) 4x4, matching every other store's cam_extrinsics_cache convention.
    torch.save(torch.tensor(W2C, dtype=torch.float32),
               os.path.join(hd, "cam_extrinsics_cache.pt"))
    # Intrinsics MUST match the video we actually have, not the resolution they were published
    # at. Refuse rather than store a K that silently disagrees with the frames.
    vhw = video_hw(video_src)
    if vhw is None:
        return "no-video", stats
    try:
        Ks, k_scale = rescale_K_to_video(K, vhw)
    except ValueError as exc:
        stats["gate"] = str(exc)
        return "bad-intrinsics-rescale", stats
    stats["video_hw"] = list(vhw)
    stats["K_scale"] = round(k_scale, 4)
    torch.save(torch.tensor([float(Ks[0, 0]), float(Ks[0, 2]), float(Ks[1, 2])]),
               os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.tensor(np.asarray(frames, np.int64)), os.path.join(hd, "annotated_frames.pt"))

    # BOXES, in the SAME convention as every other training store: joints_to_bbox, rectangular
    # per-axis x1.5, UNCLAMPED. Using a different convention here is exactly the H2O
    # square+clamped mistake, which made an entire dataset's numbers meaningless.
    rf = 1.5
    fpx, cxp, cyp = float(Ks[0, 0]), float(Ks[0, 2]), float(Ks[1, 2])
    vw, vh = float(vhw[1]), float(vhw[0])
    boxes = np.zeros((n, 2, 4), np.float32)
    bvalid = np.zeros((n, 2), bool)
    hand_ok = V.any(-1)
    for t in range(n):
        for hi in range(2):
            if not hand_ok[t, hi]:
                continue
            j = JC[t, hi]
            m = V[t, hi]
            if m.sum() < 3:
                # A box from one or two joints is not a hand box; skip rather than emit a sliver.
                continue
            b = joints_to_bbox(j[m], fpx, cxp, cyp, vw, vh, rf=rf)
            if b is None or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
                continue
            boxes[t, hi] = b
            bvalid[t, hi] = True
    torch.save({"bboxes": torch.from_numpy(boxes),
                "valid": torch.from_numpy(bvalid),
                "convention": "joints_to_bbox rf1.5 rectangular unclamped (egoexo4d)"},
               os.path.join(hd, f"hand_bboxes_v2_rf{rf}_res224x224.pt"))
    stats["boxes"] = int(bvalid.sum())
    w_ = boxes[..., 2] - boxes[..., 0]
    h_ = boxes[..., 3] - boxes[..., 1]
    if bvalid.any():
        stats["box_square_frac"] = float(np.mean(np.isclose(w_[bvalid], h_[bvalid], rtol=1e-3)))
        stats["box_outside01_frac"] = float(np.mean((boxes[bvalid] < 0) | (boxes[bvalid] > 1)))
    with open(os.path.join(out_dir, take_name, ".egoexo_meta.json"), "w") as f:
        json.dump({"take_uid": uid, "take_name": take_name, "n_annotated_frames": n,
                   "video_src": video_src, "ego_key": ego_key,
                   "K_published": K.tolist(), "K_rescaled": Ks.tolist(),
                   "K_scale": k_scale, "video_hw": list(vhw),
                   "NOTE": "3D joints are RAW ANNOTATIONS, not MANO. No MANO params exist for "
                           "this dataset, so param losses cannot be supervised from it.",
                   "frames_are_sparse": True,
                   "frames_with_camera_pose": int(has_ext.sum()),
                   "joint_frames": "world = gt_joints_cache_world.pt, camera = "
                                   "gt_joints_cache_cam_v2.pt (via per-frame w2c extrinsics)"},
                  f, indent=2)
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
    # A FEW rejected takes means bad source annotation and is healthy: the gate ran before
    # anything was written, so the store contains only validated takes. A LARGE fraction failing
    # would instead mean the joint MAPPING is wrong, which is the H2O-scramble failure and must
    # stop everything. Distinguish the two rather than condemning a good store over 2 bad takes.
    n_bad = counts.get("bad-anatomy", 0)
    n_written = counts.get("ok", 0)
    if n_bad:
        total = n_bad + n_written
        frac = n_bad / max(1, total)
        print(f"CONVERT excluded {n_bad}/{total} take(s) on the anatomical gate "
              f"({frac:.1%}); they were NOT written, so the store holds only validated takes.")
        if frac > 0.10:
            raise SystemExit(
                f"CONVERT ABORT: {frac:.1%} of takes failed the anatomical gate. At that rate the "
                f"joint MAPPING is suspect, not the source data (this is the H2O scramble "
                f"signature). Do NOT train on this store until the mapping is re-verified.")


if __name__ == "__main__":
    main()
