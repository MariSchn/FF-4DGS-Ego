#!/usr/bin/env python3
"""Rewrite TACO's hand-pose JSONL in the schema the dataset actually reads.

WHY THIS EXISTS RATHER THAN A RECONVERSION. taco_to_ours wrote
``{"frame", "left", "right"}`` where every other converter writes
``{"timestamp_ns", "hand_poses"}``, and HOT3DHandDataset reads only the latter. The store is
otherwise correct: the tensors, the boxes, the videos and the extrinsics are all fine, and the
same 32-float-per-slot vector that the JSONL should describe is already saved inside the box
store's ``gt`` tensor. So the fix is a rewrite of one small text file per sequence, seconds for
all 2,311, against hours to re-encode 23 GB of video for nothing.

The bug survived because ``verify_box_store`` verifies the BOX store and never opens the JSONL.
It surfaced only when the feature-cache build reached the dataset and died on
``KeyError: 'hand_poses'``, which is the second time on this dataset that a gate certified the
half of the artefact that was already right.

    python -m scripts.preprocessing.repair_taco_jsonl --data_root <store>          # dry run
    python -m scripts.preprocessing.repair_taco_jsonl --data_root <store> --write
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

HAND_PARAM_DIM = 32          # [transl 3, quat_wxyz 4, pose15 PCA, betas 10]


def rewrite(seq_dir: str, write: bool) -> str:
    """Return a one-word status: ok, already, nobox or empty."""
    hd = os.path.join(seq_dir, "hand_data")
    jsonl = os.path.join(hd, "mano_hand_pose_trajectory.jsonl")
    box = glob.glob(os.path.join(hd, "hand_bboxes_v2_rf*_res*.pt"))
    if not box:
        return "nobox"

    if os.path.isfile(jsonl):
        with open(jsonl) as fh:
            first = fh.readline()
        if first and "hand_poses" in json.loads(first):
            return "already"

    blob = torch.load(box[0], map_location="cpu")
    gt, valid = blob["gt"], blob["valid"]           # [N, 64], [N, 2]
    if gt.numel() == 0:
        return "empty"

    lines = []
    for i in range(gt.shape[0]):
        hp = {}
        for slot in (0, 1):
            if not bool(valid[i, slot]):
                continue
            v = gt[i, slot * HAND_PARAM_DIM:(slot + 1) * HAND_PARAM_DIM].tolist()
            hp[str(slot)] = {
                "wrist_xform": {"t_xyz": v[0:3], "q_wxyz": v[3:7]},
                "pose": v[7:22],
                "betas": v[22:32],
            }
        lines.append(json.dumps({"timestamp_ns": i, "hand_poses": hp}))

    if write:
        # Write beside and rename, so an interrupted run never leaves a half-written trajectory
        # that would parse as a short sequence rather than as a failure.
        with open(jsonl + ".part", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(jsonl + ".part", jsonl)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--write", action="store_true", help="without this, only report")
    a = ap.parse_args()

    seqs = sorted(d for d in os.listdir(a.data_root)
                  if os.path.isdir(os.path.join(a.data_root, d, "hand_data")))
    counts: dict[str, int] = {}
    for s in seqs:
        st = rewrite(os.path.join(a.data_root, s), a.write)
        counts[st] = counts.get(st, 0) + 1
    print(f"{len(seqs)} sequences: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not a.write:
        print("DRY RUN. Pass --write to rewrite.")


if __name__ == "__main__":
    main()
