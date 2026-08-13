#!/usr/bin/env python3
"""Add a 21-joint ground-truth cache to a store that only has the 16-joint skeleton.

Hand3R evaluates the 21 manopth joints: the wrist, three per finger, and five vertex-derived
fingertips. Our stores carry the 16-joint SMPL-X skeleton, which has no tips. The five missing
joints are the highest-error ones on any hand, so a 16-joint MPJPE is systematically lower than a
21-joint one on identical predictions, and the two cannot be put in the same table.

The tips are recoverable rather than lost: every store already holds the MANO parameters in its box
file's ``gt`` tensor, and MANOModel.get_joints21_batched runs the layer and reads the tip vertices.
So this recomputes the 21-joint set from the parameters that produced the 16, and writes

    hand_data/gt_joints_cache_cam21_v2.pt     [N, 2, 21, 3]
    hand_data/gt_joints_cache_world21.pt      [N, 2, 21, 3]   when extrinsics exist

THE GATE. A joint permutation is invisible in an MPJPE and has cost this project two corrupted
result sets already (the H2O 21->16 scramble, the WiLoR+SLAM row). So every sequence is checked
anatomically: each fingertip must be further from the wrist than its own DIP joint. That is true of
every real hand and false of essentially every wrong ordering. A store failing it is not written.

    python -m scripts.preprocessing.build_joints21_cache --data_root <store> --mano_model models/MANO
"""
from __future__ import annotations

import argparse
import os

import torch

from scripts.hand_vis_utils import MANOModel

# In the 21-joint layout the first 16 are our skeleton and positions 16..20 are the tips, ordered
# thumb, index, middle, ring, pinky. The DIP of each finger is the last of its three-joint run in
# the 16-joint block, which is what the gate compares each tip against.
TIP_IDX = [16, 17, 18, 19, 20]
DIP_FOR_TIP = {16: 15, 17: 3, 18: 6, 19: 12, 20: 9}   # thumb, index, middle, ring, pinky


def _params_of(seq_dir: str) -> torch.Tensor | None:
    hd = os.path.join(seq_dir, "hand_data")
    import glob
    box = glob.glob(os.path.join(hd, "hand_bboxes_v2_rf*_res*.pt"))
    if not box:
        return None
    blob = torch.load(box[0], map_location="cpu")
    return blob.get("gt")            # [N, 64] = two hands x 32 params


def anatomy_ok(j21: torch.Tensor) -> tuple[bool, float]:
    """Every tip must sit further from the wrist than its own DIP. Returns (ok, worst ratio)."""
    d = (j21 - j21[..., :1, :]).norm(dim=-1)          # [..., 21] distance from the wrist
    worst = 1e9
    for tip, dip in DIP_FOR_TIP.items():
        finite = torch.isfinite(d[..., tip]) & torch.isfinite(d[..., dip]) & (d[..., dip] > 1e-6)
        if not bool(finite.any()):
            continue
        ratio = (d[..., tip][finite] / d[..., dip][finite]).median().item()
        worst = min(worst, ratio)
    return worst > 1.0, worst


def build(seq_dir: str, mano: MANOModel, device: str, write: bool):
    params = _params_of(seq_dir)
    if params is None or params.numel() == 0:
        return "nobox", 0.0
    n = params.shape[0]
    flat = params.view(n, 2, 32)
    left = mano.get_joints21_batched(flat[:, 0], is_right=False, device=device)
    right = mano.get_joints21_batched(flat[:, 1], is_right=True, device=device)
    j21 = torch.stack([left, right], dim=1).cpu()      # [N, 2, 21, 3]

    ok, worst = anatomy_ok(j21)
    if not ok:
        return "anatomy_fail", worst

    hd = os.path.join(seq_dir, "hand_data")
    if write:
        torch.save(j21, os.path.join(hd, "gt_joints_cache_cam21_v2.pt"))
        ext = os.path.join(hd, "cam_extrinsics_cache.pt")
        if os.path.isfile(ext):
            w2c = torch.load(ext, map_location="cpu").double()
            c2w = torch.linalg.inv(w2c)                # the store holds T_camera_world
            m = min(len(c2w), n)
            h = j21[:m].double()
            world = (c2w[:m, None, None, :3, :3] @ h[..., None]).squeeze(-1) \
                + c2w[:m, None, None, :3, 3]
            torch.save(world.float(), os.path.join(hd, "gt_joints_cache_world21.pt"))
    return "ok", worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--mano_model", default="models/MANO")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--max_fail_frac", type=float, default=0.02,
                    help="abort if more than this fraction fails the anatomy gate; a permutation "
                         "fails on essentially every sequence, so a high rate means the joint "
                         "order is wrong rather than the data")
    a = ap.parse_args()

    mano = MANOModel(a.mano_model)
    seqs = sorted(d for d in os.listdir(a.data_root)
                  if os.path.isdir(os.path.join(a.data_root, d, "hand_data")))
    counts, worst_seen = {}, 1e9
    for s in seqs:
        st, worst = build(os.path.join(a.data_root, s), mano, a.device, a.write)
        counts[st] = counts.get(st, 0) + 1
        if st in ("ok", "anatomy_fail"):
            worst_seen = min(worst_seen, worst)

    n = max(len(seqs), 1)
    fails = counts.get("anatomy_fail", 0)
    print(f"{len(seqs)} sequences: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"worst tip/DIP distance ratio seen: {worst_seen:.3f} (must exceed 1.0)")
    if fails / n > a.max_fail_frac:
        raise SystemExit(f"ABORT: {fails}/{n} failed the anatomy gate. The 16->21 joint order is "
                         f"wrong, not the data. Do NOT write this cache.")
    if not a.write:
        print("DRY RUN. Pass --write to save the caches.")


if __name__ == "__main__":
    main()
