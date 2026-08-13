#!/usr/bin/env python3
"""TACO (CVPR 2024) -> our per-sequence hand store.

THE TARGET FORMAT IS THE ONE dexycb_to_ours.py DOCUMENTS, which was derived from what
train_hand_head.py actually reads off a sequence directory rather than from anything we invented.
Re-read that header before changing anything here; the two stores must stay interchangeable.

    <out_root>/<triplet>__<sequence>/
        video_main_rgb.mp4
        hand_data/cam_intrinsics.pt                   [3] = [f, cx, cy]
        hand_data/cam_extrinsics_cache.pt             [N,4,4] T_camera_world (w2c)
        hand_data/gt_joints_2d_cache.pt               [N,2,16,3] (u,v,conf) px
        hand_data/gt_joints_cache_cam_v2.pt           [N,2,16,3] CAMERA frame, metres
        hand_data/gt_joints_cache_world.pt            [N,2,16,3] WORLD frame, metres
        hand_data/mano_hand_pose_trajectory.jsonl
        hand_data/hand_bboxes_v2_rf1.5_res224x224.pt  {bboxes,valid,gt}

WHAT TACO GIVES US, verified by reading their project_pose_to_egocentric_view.py and then by
opening the files, not from the README:

  * Egocentric_Camera_Parameters/<t>/<s>/egocentric_frame_extrinsic.npy
        (N,4,4). Their line 113 comments it `world_to_camera`, which is ALREADY our convention,
        so unlike DexYCB there is no direction to infer and no inversion to apply. Gate 2 below
        re-derives it from the data anyway.
  * .../egocentric_intrinsic.txt : one 3x3, CONSTANT for the sequence. Measured at 1376.8 px on
        every sampled sequence, i.e. one physical sensor. This is what Re:InterHand could not
        offer and what makes a single-focal store possible with no warping.
  * Hand_Poses/<t>/<s>/{left,right}_hand.pkl : dict keyed by frame string '00001'.. each holding
        hand_pose (48,) axis-angle and hand_trans (3,) in the WORLD frame, metres.
  * .../{left,right}_hand_shape.pkl : hand_shape (10,) betas, ONE per sequence.
  * Egocentric_RGB_Videos/<t>/<s>/color.mp4 : 1920x1080 at 30 Hz.

THREE THINGS THAT WILL BITE, each already measured:

  1. FRAME COUNTS DISAGREE on about 7% of sequences (2 of 30 sampled: video 260 vs poses 262).
     TACO's own script prints "losing frames in the egocentric video, skip!", so this is a known
     upstream condition. We TRUNCATE to the common length and record it. Zipping mismatched
     lengths is what inflated a HaWoR baseline by 2.5x, so it is never done silently.
  2. DIRECTORY NAMES CONTAIN SPACES, COMMAS AND PARENTHESES, e.g. "(dust, roller, pan)". The
     output name flattens them to <triplet>__<sequence> with those characters replaced, because
     downstream code splits paths on whitespace in several places.
  3. THE ABSENT HAND IS ZEROS AND valid=False, NEVER NaN. Keypoint3DLoss multiplies the residual
     by a per-joint confidence and NaN*0 is NaN, so one NaN-filled absent hand poisons every
     gradient in the batch. TACO is bimanual so both hands are usually present, but a sequence
     missing one file must still write the absent-hand representation the consumer expects.

MANO CONVENTION. TACO builds hands with manopth `ManoLayer(use_pca=False, ncomps=45,
center_idx=0)`, and manopth's flat_hand_mean defaults to True. Our MANOModel.{left,right}_full
layers are smplx with `use_pca=False, flat_hand_mean=True, num_pca_comps=45`, i.e. the same
contract, which is why they exist. Joint order out of those layers is the manopth order, so the
21->16 map is the one dexycb_to_ours already asserts is bit-identical to H2O's.

Usage. SMOKE WITHOUT --validate, on a couple of sequences, and only then the full run.
--validate runs the gates and returns before writing anything, so it certifies the read and the
geometry but NOT the write path. A smoke run that used it looked perfectly clean and the full run
then failed on all 2311 sequences with a wrong keyword in the box call. Use --limit instead.
    python -m scripts.preprocessing.taco_to_ours --taco_root <root> --out_root <smoke_out> \
        --mano_model models/MANO --limit 2
    python -m scripts.preprocessing.taco_to_ours --taco_root <root> --out_root <out> \
        --mano_model models/MANO
Then, before training on it:
    python -m scripts.verify_box_store --data_root <out>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys

import cv2
import numpy as np
import torch

from scripts.arctic_to_ours import joints_to_bbox
from scripts.preprocessing.dexycb_to_ours import DEXYCB21_TO_MANO16, anatomy_report, anatomy_gate
from scripts.preprocessing.preprocess_hoi4d import _aa_to_quat_wxyz

NUM_HANDS = 2          # slot 0 = LEFT, slot 1 = RIGHT (train_hand_head.py:861)
NUM_JOINTS = 16
SIDES = ("left", "right")


def safe_name(triplet: str, seq: str) -> str:
    """<triplet>__<sequence> with path-hostile characters removed.

    TACO names sequences after the action triplet, e.g. "(dust, roller, pan)". Several places
    downstream split on whitespace, and a directory whose name contains spaces and commas is a
    standing invitation for one of them to see three arguments where there is one path.
    """
    t = re.sub(r"[^0-9A-Za-z]+", "-", triplet).strip("-")
    s = re.sub(r"[^0-9A-Za-z]+", "-", seq).strip("-")
    return f"{t}__{s}"


def read_hand(pkl_dir: str, side: str):
    """-> (pose48 [N,48], trans [N,3], betas [10], frame_keys) or None when the hand is absent."""
    p_seq = os.path.join(pkl_dir, f"{side}_hand.pkl")
    p_shp = os.path.join(pkl_dir, f"{side}_hand_shape.pkl")
    if not (os.path.isfile(p_seq) and os.path.isfile(p_shp)):
        return None
    with open(p_seq, "rb") as f:
        per_frame = pickle.load(f)
    with open(p_shp, "rb") as f:
        betas = np.asarray(pickle.load(f)["hand_shape"], dtype=np.float64).reshape(10)
    keys = sorted(per_frame.keys())
    pose = np.stack([np.asarray(per_frame[k]["hand_pose"], dtype=np.float64).reshape(48) for k in keys])
    tran = np.stack([np.asarray(per_frame[k]["hand_trans"], dtype=np.float64).reshape(3) for k in keys])
    return pose, tran, betas, keys


def mano_joints_world(mano, side: str, pose48: np.ndarray, trans: np.ndarray,
                      betas: np.ndarray) -> np.ndarray:
    """[N,16,3] world-frame joints, metres, in OUR 16-joint order.

    TACO's ManoLayer uses center_idx=0, which returns joints relative to the wrist; the wrist is
    then placed by hand_trans. smplx's transl argument does the same composition, so the two agree
    without a manual re-centring, and gate 3 (anatomy) would catch it if they did not.
    """
    layer = mano.right_full if side == "right" else mano.left_full
    n = len(pose48)
    out = layer(
        betas=torch.as_tensor(np.repeat(betas[None], n, 0), dtype=torch.float32),
        global_orient=torch.as_tensor(pose48[:, :3], dtype=torch.float32),
        hand_pose=torch.as_tensor(pose48[:, 3:], dtype=torch.float32),
        transl=torch.as_tensor(trans, dtype=torch.float32),
        # return_verts=True even though we only want joints: smplx returns joints=None when it is
        # False, so the cheaper-looking call silently produces nothing to read.
        return_verts=True,
    )
    j = out.joints.detach().cpu().numpy().astype(np.float64)
    # smplx's MANO layers emit the 16-joint SKELETON directly, already in our layout
    # [wrist, index x3, middle x3, pinky x3, ring x3, thumb x3], which is why hand_vis_utils calls
    # them interchangeable with our caches. DexYCB needs DEXYCB21_TO_MANO16 because it publishes
    # manopth's 21-joint output with fingertips; here there is nothing to remap, and applying that
    # map anyway would index past the end. The anatomy gate is what verifies the order rather than
    # this comment.
    if j.shape[1] == NUM_JOINTS:
        return j
    if j.shape[1] >= 21:
        return j[:, DEXYCB21_TO_MANO16]
    raise SystemExit(f"MANO returned {j.shape[1]} joints; expected {NUM_JOINTS} (skeleton) or 21 "
                     f"(with fingertips). Neither layout applies, so the joint order is unknown "
                     f"and nothing should be written.")


def n_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def convert_seq(taco_root: str, triplet: str, seq: str, out_root: str, mano,
                rescale_factor: float, res: int, validate: bool) -> dict:
    cam_dir = os.path.join(taco_root, "Egocentric_Camera_Parameters", triplet, seq)
    hnd_dir = os.path.join(taco_root, "Hand_Poses", triplet, seq)
    vid_src = os.path.join(taco_root, "Egocentric_RGB_Videos", triplet, seq, "color.mp4")

    K = np.loadtxt(os.path.join(cam_dir, "egocentric_intrinsic.txt")).reshape(3, 3)
    E = np.load(os.path.join(cam_dir, "egocentric_frame_extrinsic.npy")).astype(np.float64)
    nv = n_video_frames(vid_src)

    hands = {s: read_hand(hnd_dir, s) for s in SIDES}
    lens = [len(E), nv] + [len(h[0]) for h in hands.values() if h is not None]
    N = min(l for l in lens if l > 0)
    report = {"seq": safe_name(triplet, seq), "n_frames": int(N),
              "n_extrinsics": int(len(E)), "n_video": int(nv),
              "truncated": bool(len(set(lens)) > 1),
              "hands_present": [s for s in SIDES if hands[s] is not None]}
    if N < 8:
        report["skip"] = f"only {N} usable frames"
        return report

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Our store is single-focal. Record the anisotropy we discard rather than hiding it; TACO is a
    # plain pinhole with no distortion, so this is expected to be tiny.
    report["fx"], report["fy"], report["aniso_px"] = fx, fy, abs(fx - fy)

    E = E[:N]
    R, t = E[:, :3, :3], E[:, :3, 3]

    world = np.zeros((N, NUM_HANDS, NUM_JOINTS, 3), dtype=np.float64)
    cam = np.zeros_like(world)
    valid = np.zeros((N, NUM_HANDS), dtype=bool)
    gt64 = np.zeros((N, 64), dtype=np.float32)

    for hi, side in enumerate(SIDES):
        h = hands[side]
        if h is None:
            continue                      # slot stays zeros, valid stays False. Never NaN.
        pose48, trans, betas, _ = h
        jw = mano_joints_world(mano, side, pose48[:N], trans[:N], betas)
        world[:, hi] = jw
        cam[:, hi] = np.einsum("nij,nkj->nki", R, jw) + t[:, None, :]
        valid[:, hi] = True
        # gt64 layout matches every other store: [transl 3, quat_wxyz 4, pose45 -> 15, betas 10]
        # per hand, and it feeds the parameter loss only. The 45->15 truncation is the same one
        # preprocess_hoi4d applies; joint caches are unaffected by it.
        q = np.stack([_aa_to_quat_wxyz(a) for a in pose48[:N, :3]])
        base = hi * 32
        gt64[:, base + 0:base + 3] = trans[:N]
        gt64[:, base + 3:base + 7] = q
        gt64[:, base + 7:base + 22] = pose48[:N, 3:18]
        gt64[:, base + 22:base + 32] = betas[None]

    # 2D: single-focal pinhole on the camera-frame joints, the preprocess_hoi4d convention.
    z = np.clip(cam[..., 2], 1e-6, None)
    u = fx * cam[..., 0] / z + cx
    v = fx * cam[..., 1] / z + cy
    conf = valid[:, :, None].astype(np.float64) * np.ones((1, 1, NUM_JOINTS))
    j2d = np.stack([u, v, conf], axis=-1)

    rep = anatomy_report(cam.reshape(N * NUM_HANDS, NUM_JOINTS, 3),
                         valid.reshape(N * NUM_HANDS))
    report["anatomy"] = rep
    fail = anatomy_gate(rep)
    report["anatomy_fail"] = fail
    if fail:
        # A scrambled joint order is silent in every other check and fails this one on essentially
        # every sequence, so a failure here is a convention bug and not a bad sequence.
        return report
    if validate:
        return report

    out_seq = os.path.join(out_root, safe_name(triplet, seq))
    hd = os.path.join(out_seq, "hand_data")
    os.makedirs(hd, exist_ok=True)

    boxes = np.zeros((N, NUM_HANDS, 4), dtype=np.float32)
    for i in range(N):
        for hi in range(NUM_HANDS):
            if valid[i, hi]:
                # The keyword is rf, not rescale_factor (scripts/arctic_to_ours.py:111). The
                # smoke run could not catch this because --validate returns before the write path,
                # so the gates ran and the boxes never did. Fixed by also exercising one real
                # write in the smoke run.
                b = joints_to_bbox(cam[i, hi], fx, cx, cy,
                                   float(2 * cx), float(2 * cy), rf=rescale_factor)
                if b is not None:
                    boxes[i, hi] = b
                else:
                    valid[i, hi] = False

    torch.save(torch.tensor([fx, cx, cy], dtype=torch.float32), os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.tensor(E, dtype=torch.float32), os.path.join(hd, "cam_extrinsics_cache.pt"))
    torch.save(torch.tensor(cam, dtype=torch.float32), os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(torch.tensor(world, dtype=torch.float32), os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(torch.tensor(j2d, dtype=torch.float32), os.path.join(hd, "gt_joints_2d_cache.pt"))
    torch.save({"bboxes": torch.tensor(boxes), "valid": torch.tensor(valid),
                "gt": torch.tensor(gt64)},
               os.path.join(hd, f"hand_bboxes_v2_rf{rescale_factor}_res{res}x{res}.pt"))
    with open(os.path.join(hd, "mano_hand_pose_trajectory.jsonl"), "w") as f:
        for i in range(N):
            f.write(json.dumps({"frame": i,
                                "left":  gt64[i, :32].tolist(),
                                "right": gt64[i, 32:].tolist()}) + "\n")

    # Video: re-encode truncated to N so that video frame t IS cache row t, which is the store
    # contract. A symlink to the original would break it on every truncated sequence.
    dst = os.path.join(out_seq, "video_main_rgb.mp4")
    capin = cv2.VideoCapture(vid_src)
    w_in = int(capin.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(2 * cx)
    h_in = int(capin.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(2 * cy)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w_in, h_in))
    written = 0
    while written < N:
        ok, frame = capin.read()
        if not ok:
            break
        out.write(frame)
        written += 1
    capin.release(); out.release()
    report["frames_written"] = written
    if written != N:
        report["skip"] = f"wrote {written} frames for {N} cache rows"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco_root", default="/cluster/scratch/dmonopoli/taco_v1")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--mano_model", default="models/MANO")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rescale_factor", type=float, default=1.5)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--validate", action="store_true",
                    help="run the gates and write nothing")
    ap.add_argument("--max_gate_fail", type=float, default=0.10)
    a = ap.parse_args()

    lst = os.path.join(a.taco_root, "_usable_sequences.txt")
    if not os.path.isfile(lst):
        sys.exit(f"missing {lst}: run the extraction job, which writes the intersection of the "
                 f"three subtrees. Iterating one subtree alone would include sequences with no "
                 f"video.")
    rows = [tuple(l.rstrip("\n").split("\t")) for l in open(lst) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    from scripts.hand_vis_utils import MANOModel
    mano = MANOModel(a.mano_model)
    os.makedirs(a.out_root, exist_ok=True)

    n_ok = n_fail = n_skip = 0
    reports = []
    for i, (trip, sq) in enumerate(rows):
        try:
            r = convert_seq(a.taco_root, trip, sq, a.out_root, mano,
                            a.rescale_factor, a.res, a.validate)
        except Exception as e:
            r = {"seq": safe_name(trip, sq), "error": f"{type(e).__name__}: {e}"}
        reports.append(r)
        if r.get("error") or r.get("skip"):
            n_skip += 1
        elif r.get("anatomy_fail"):
            n_fail += 1
        else:
            n_ok += 1
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[{i+1}/{len(rows)}] ok={n_ok} anatomy_fail={n_fail} skipped={n_skip}", flush=True)
        # A convention bug fails anatomy on essentially every sequence, so stop early and loudly
        # rather than writing a store that is uniformly wrong.
        if i >= 19 and n_fail / max(1, i + 1) > a.max_gate_fail:
            print(f"\n!! ABORTING: anatomy gate failing on {n_fail}/{i+1} sequences, above "
                  f"--max_gate_fail {a.max_gate_fail}. That rate means a convention error (joint "
                  f"order, MANO flat_hand_mean, or extrinsic direction), not bad data.", flush=True)
            for r in reports:
                if r.get("anatomy_fail"):
                    print("   ", r["seq"], r["anatomy_fail"]); break
            sys.exit(3)

    with open(os.path.join(a.out_root, "_conversion_report.json"), "w") as f:
        json.dump(reports, f, indent=1)
    trunc = sum(1 for r in reports if r.get("truncated"))
    print(f"\ndone: ok={n_ok} anatomy_fail={n_fail} skipped={n_skip}")
    print(f"truncated (video and pose lengths disagreed): {trunc} of {len(rows)}")
    print(f"report -> {os.path.join(a.out_root, '_conversion_report.json')}")


if __name__ == "__main__":
    main()
