"""HaPTIC output -> eval_worldspace_baseline prediction contract.

HaPTIC's demo writes one pickle per frame per sequence with (at least) the keys
``cJoints`` / ``wJoints``: (1, 21, 3) hand joints in metres, OpenPose-21 order
(wrist = 0), camera frame and world frame respectively (world is only meaningful
because we injected HOI4D GT extrinsics as cTw via hoi4d_to_haptic.py).

This script collects those per-frame pickles into one ``<seq>.pt`` per sequence:
    "cam_joints"   : [N, 2, 16, 3]  metres, camera frame, smplx-16 order
    "world_joints" : [N, 2, 16, 3]  metres, world frame,  smplx-16 order
    "valid"        : [N, 2] bool
Left-hand slot (0) is all-NaN/invalid: we run HaPTIC on the right hand only.
Frames without a pickle stay NaN/invalid. N is taken from the GT cache length so
the scorer's frame alignment is exact.

Usage:
    python -m scripts.haptic_to_worldeval --haptic_out <demo_output_dir> \
        --data_root <hoi4d_test> --pred_dir <out_pred_dir> [--dump_keys]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re

import numpy as np
import torch

RH = 1
J = 16
# OpenPose-21 (wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4) ->
# smplx-16 kinematic order (wrist, index x3, middle x3, pinky x3, ring x3, thumb x3).
# Identical to _KPS2D_FOR_SMPLX16 in scripts/preprocessing/preprocess_hoi4d.py.
OP2SMPLX16 = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]


def _joints_from_pkl(d, key):
    """Pull a (21,3) float array out of one HaPTIC frame dict, tolerating wrappers."""
    v = d.get(key)
    if v is None:
        return None
    v = np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, np.float32)
    v = v.reshape(-1, 3)
    return v[:21] if v.shape[0] >= 21 else None


def _frame_index(path: str) -> int | None:
    """Frame index = last integer run in the basename (00042.pkl, img_00042_out.pkl...)."""
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else None


def convert_seq(seq: str, frame_pkls: list[str], n_frames: int, dump_keys: bool = False):
    cam = torch.full((n_frames, 2, J, 3), float("nan"))
    world = torch.full((n_frames, 2, J, 3), float("nan"))
    valid = torch.zeros(n_frames, 2, dtype=torch.bool)
    n_hit = 0
    for p in frame_pkls:
        t = _frame_index(p)
        if t is None or not (0 <= t < n_frames):
            continue
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        if dump_keys and n_hit == 0:
            print(f"  [{seq}] pkl keys: {sorted(d.keys())}")
        cj = _joints_from_pkl(d, "cJoints")
        wj = _joints_from_pkl(d, "wJoints")
        if cj is None or wj is None:
            continue
        cam[t, RH] = torch.from_numpy(cj[OP2SMPLX16])
        world[t, RH] = torch.from_numpy(wj[OP2SMPLX16])
        valid[t, RH] = True
        n_hit += 1
    return {"cam_joints": cam, "world_joints": world, "valid": valid}, n_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haptic_out", required=True,
                    help="HaPTIC demo output root (contains per-seq dirs of frame pkls)")
    ap.add_argument("--data_root", required=True, help="HOI4D test dir (GT caches, for N)")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--dump_keys", action="store_true", help="print pkl keys per seq (debug)")
    args = ap.parse_args()
    os.makedirs(args.pred_dir, exist_ok=True)

    n_seq = 0
    for seq in sorted(os.listdir(args.data_root)):
        hd = os.path.join(args.data_root, seq, "hand_data")
        gtb = os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt")
        if not os.path.exists(gtb):
            continue
        n_frames = int(torch.load(gtb, map_location="cpu")["valid"].shape[0])
        # HaPTIC names the run "<seq>_right" (set by hoi4d_to_haptic.py); check both.
        pkls = []
        for cand in (f"{seq}_right", seq):
            for pat in (os.path.join(args.haptic_out, cand, "*.pkl"),
                        os.path.join(args.haptic_out, cand, "**", "*.pkl")):
                pkls = sorted(glob.glob(pat, recursive=True))
                if pkls:
                    break
            if pkls:
                break
        if not pkls:
            continue
        pred, n_hit = convert_seq(seq, pkls, n_frames, args.dump_keys)
        torch.save(pred, os.path.join(args.pred_dir, seq + ".pt"))
        n_seq += 1
        print(f"[{n_seq}] {seq}: {n_hit}/{n_frames} frames converted", flush=True)
    # THE world_joints HERE ARE AN ORACLE, and that must be recorded as data, not left implied by
    # a docstring. hoi4d_to_haptic.py injects HOI4D GT extrinsics as cTw, so wJoints is a
    # GT-camera lift, not a tracked trajectory. The resulting dir scores BETTER on W/WA than the
    # honest SLAM composition of the same hands (measured 2026-08-06: 27.1/23.3 here vs 42.5/35.7
    # composed), and it was already wired into a seg100 table under a "+SLAM" label because
    # nothing in the artifact said otherwise (task #67).
    from scripts.pred_provenance import TRAJ_GT_ORACLE, write_provenance
    write_provenance(
        args.pred_dir,
        box_source=f"SET_BY_hoi4d_to_haptic.py_--box_dir (not visible here); haptic_out={args.haptic_out}",
        trajectory_source=TRAJ_GT_ORACLE,
        produced_by="scripts/haptic_to_worldeval.py",
        n_seqs=n_seq,
        warning=("world_joints use HOI4D GROUND-TRUTH extrinsics injected by hoi4d_to_haptic.py. "
                 "Camera-frame metrics (C_MPJPE, C_MPJPE_abs) are honest; W-MPJPE and WA-MPJPE "
                 "from this directory are ORACLE numbers. To get a comparable world row, compose "
                 "these cam preds with a real trajectory via scripts/build_slam_baseline.py."),
    )
    print(f"HAPTIC_CONVERT_DONE seqs={n_seq} -> {args.pred_dir}")
    if n_seq == 0:
        print("WARN: no sequences matched; check --haptic_out layout (use --dump_keys)")


if __name__ == "__main__":
    main()
