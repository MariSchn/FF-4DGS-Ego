#!/usr/bin/env python3
"""Write our stored predictions as the NPZ pairs Hand3R's reference scorer reads.

This is what makes the exchange with the Hand3R authors work in both directions. They offered to
run their model under our protocol, so they need to know the format we expect back; and the format
we should ask for is theirs, because then their own scorer can be run over our predictions
unmodified. A parity claim that rests on our reimplementation of their metrics is weaker than one
where their file, unedited, scores our output.

Their README specifies, per clip, a ground-truth NPZ and a prediction NPZ holding:

    joints_cam    [T, 21, 3]   absolute camera-frame joints
    joints_world  [T, 21, 3]   world-frame joints, optional for camera-only scoring
    valid_mask    [T]          bool, optional, combined across the two files

plus a CSV manifest of ``clip_id,gt_path,pred_path``.

TWENTY-ONE JOINTS, NOT SIXTEEN. Our stores hold the 16-joint SMPL-X skeleton with no fingertips,
and Hand3R evaluates the 21 manopth joints including the five vertex-derived tips. Padding the
missing five with anything, zeros or duplicates of the DIP joints, would silently produce a number
that is neither convention. So this refuses to write 16 joints as 21 unless ``--allow_16`` is
passed, in which case it writes 16 and records that in the manifest's own name, because a 16-joint
file scored by a 21-joint scorer is exactly the mistake the whole exchange exists to avoid.

    python -m scripts.hand3r_protocol.export_npz --pred_dir <dir> --data_root <store> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

RH = 1               # right hand slot, matching every other scorer in this repo
HAND3R_JOINTS = 21


def _load_gt(seq_dir: str):
    hd = os.path.join(seq_dir, "hand_data")
    cam = os.path.join(hd, "gt_joints_cache_cam_v2.pt")
    world = os.path.join(hd, "gt_joints_cache_world.pt")
    if not os.path.isfile(cam):
        return None
    g_cam = torch.load(cam, map_location="cpu")
    g_world = torch.load(world, map_location="cpu") if os.path.isfile(world) else None
    return g_cam, g_world


def _select_hand(joints: torch.Tensor) -> np.ndarray:
    """[T, 2, J, 3] -> [T, J, 3] for the right hand.

    Hand3R's protocol is right-preferred with a LEFT FALLBACK when no right hand is annotated. Ours
    is right-only. We keep right-only here and disclose it rather than inventing a fallback our
    stored predictions cannot honour: every baseline's file fills the right slot and leaves the
    left invalid, so a fallback would change the GT frame set without changing any prediction.
    """
    return joints[:, RH].numpy().astype(np.float64)


def export_clip(seq_dir: str, pred_path: str, out_dir: str, allow_16: bool) -> dict | None:
    gt = _load_gt(seq_dir)
    if gt is None:
        return None
    g_cam, g_world = gt
    pred = torch.load(pred_path, map_location="cpu")
    p_cam, p_world = pred["cam_joints"], pred.get("world_joints")
    p_valid = pred["valid"].bool() if "valid" in pred else None

    n = min(len(g_cam), len(p_cam))
    gc, pc = _select_hand(g_cam[:n]), _select_hand(p_cam[:n])
    j = gc.shape[1]
    if j != HAND3R_JOINTS and not allow_16:
        raise SystemExit(
            f"{os.path.basename(seq_dir)} has {j} joints, Hand3R's protocol has {HAND3R_JOINTS}. "
            f"Pad nothing: run the 21-joint export path, or pass --allow_16 and label the result "
            f"as a {j}-joint comparison everywhere it appears.")

    valid = np.isfinite(gc).all((1, 2))
    if p_valid is not None:
        valid &= p_valid[:n, RH].numpy()

    name = os.path.basename(seq_dir)
    gt_p = os.path.join(out_dir, "gt", f"{name}.npz")
    pr_p = os.path.join(out_dir, "pred", f"{name}.npz")
    os.makedirs(os.path.dirname(gt_p), exist_ok=True)
    os.makedirs(os.path.dirname(pr_p), exist_ok=True)

    gt_arrays = {"joints_cam": gc, "valid_mask": valid}
    pr_arrays = {"joints_cam": pc}
    if g_world is not None and p_world is not None:
        gt_arrays["joints_world"] = _select_hand(g_world[:n])
        pr_arrays["joints_world"] = _select_hand(p_world[:n])
    np.savez_compressed(gt_p, **gt_arrays)
    np.savez_compressed(pr_p, **pr_arrays)
    return {"clip_id": name, "gt_path": gt_p, "pred_path": pr_p,
            "frames": int(n), "valid": int(valid.sum()), "joints": int(j)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="our store, for the ground truth")
    ap.add_argument("--pred_dir", required=True, help="stored per-sequence predictions (.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip_list", default=None, help="restrict to these clip ids, one per line")
    ap.add_argument("--allow_16", action="store_true",
                    help="write our 16-joint skeleton instead of refusing. The output is then NOT "
                         "Hand3R-comparable and every table using it must say so.")
    a = ap.parse_args()

    wanted = None
    if a.clip_list:
        with open(a.clip_list) as fh:
            wanted = {ln.strip() for ln in fh if ln.strip()}

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(a.data_root)):
        if wanted is not None and name not in wanted:
            continue
        seq = os.path.join(a.data_root, name)
        pred = os.path.join(a.pred_dir, f"{name}.pt")
        if not os.path.isdir(seq) or not os.path.isfile(pred):
            continue
        r = export_clip(seq, pred, a.out, a.allow_16)
        if r:
            rows.append(r)

    manifest = os.path.join(a.out, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "gt_path", "pred_path"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("clip_id", "gt_path", "pred_path")})

    joints = sorted({r["joints"] for r in rows})
    with open(os.path.join(a.out, "_export_report.json"), "w") as fh:
        json.dump({"n_clips": len(rows), "joint_counts": joints,
                   "hand_set": "right only, no left fallback",
                   "data_root": a.data_root, "pred_dir": a.pred_dir,
                   "clips": rows}, fh, indent=1)

    print(f"{len(rows)} clips -> {a.out}")
    print(f"joint counts present: {joints}"
          + ("" if joints == [HAND3R_JOINTS] else "   NOT Hand3R-comparable"))
    print(f"score it with their own file:\n"
          f"  python scripts/hand3r_protocol/reference_scorer.py {manifest} "
          f"--chunk-length 100 --unit m")


if __name__ == "__main__":
    main()
