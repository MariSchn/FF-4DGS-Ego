#!/usr/bin/env python3
"""Verify a store's hand-box file is BOTH the right convention and structurally complete.

Two independent failure modes have bitten us, and each is invisible to the other's check:

 1. WRONG CONVENTION. Square+clamped boxes where the training convention is rectangular and
    unclamped. Measured cost: C-abs ~290-380 mm, which reads as "the model does not generalise"
    when it is purely an input artefact.

 2. MISSING KEYS. A regeneration that writes a fresh {bboxes, valid} dict silently drops "gt" -
    the GT MANO params rewritten from world into camera frame, which HOT3DHandDataset reads as
    cached["gt"]. Every eval on the store then dies with KeyError: 'gt'. This actually happened
    to the H2O store.

Recomputing boxes from joints proves nothing about either: it measures what a regeneration WOULD
produce, not what is on disk. This reads the stored tensors back.

Convention gating uses square_frac ONLY. outside01_frac (do boxes leave [0,1]) is a property of
the FOOTAGE, not the convention: a sample where hands never exit frame legitimately scores 0.
On H2O the 3-sequence sample gave 0.0000 while the full 177-sequence store gave 0.0234, so
gating on it would reject a correct store.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

# Keys HOT3DHandDataset reads out of the box cache. Dropping any of these breaks every eval.
REQUIRED_KEYS = ("bboxes", "valid", "gt")
SQUARE_FRAC_MAX = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--name", default="hand_bboxes_v2_rf1.5_res224x224.pt")
    ap.add_argument("--require_keys", default=",".join(REQUIRED_KEYS),
                    help="comma-separated keys that must exist in every box file")
    args = ap.parse_args()

    required = tuple(k for k in args.require_keys.split(",") if k)
    sq_all: list[float] = []
    out_all: list[float] = []
    missing_keys: dict[str, list[str]] = {}
    n_seq = 0

    for s in sorted(os.listdir(args.data_root)):
        p = os.path.join(args.data_root, s, "hand_data", args.name)
        if not os.path.exists(p):
            continue
        d = torch.load(p, map_location="cpu", weights_only=True)

        absent = [k for k in required if k not in d]
        if absent:
            missing_keys.setdefault(",".join(absent), []).append(s)

        if "bboxes" not in d or "valid" not in d:
            continue
        b = np.asarray(d["bboxes"])
        v = np.asarray(d["valid"]).astype(bool)
        w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
        m = v & (w > 0) & (h > 0)
        if not m.any():
            continue
        sq_all.append(float(np.mean(np.isclose(w[m], h[m], rtol=1e-3))))
        out_all.append(float(np.mean((b[m] < 0) | (b[m] > 1))))
        n_seq += 1

    if n_seq == 0:
        raise SystemExit("VERIFY FAILED: no readable box files under " + args.data_root)

    sq, out = float(np.mean(sq_all)), float(np.mean(out_all))
    print(f"VERIFY over {n_seq} seqs: square_frac={sq:.4f} outside01_frac={out:.4f}")

    problems = []
    if sq > SQUARE_FRAC_MAX:
        problems.append(f"square_frac={sq:.4f} > {SQUARE_FRAC_MAX} - the store is still SQUARE, "
                        f"so it is NOT in the joints_to_bbox training convention")
    for absent, seqs in sorted(missing_keys.items()):
        problems.append(f"{len(seqs)} sequences are MISSING key(s) [{absent}] "
                        f"(e.g. {seqs[0]}) - every eval on them will raise KeyError")

    if not (out > 0.0):
        print("  note: outside01_frac=0 means no hand leaves the frame in this store. That is a "
              "property of the footage, not the convention, and is NOT a failure.")

    for p_ in problems:
        print(f"  !! {p_}")
    if problems:
        raise SystemExit("VERIFY FAILED: do not evaluate on this store; any number would be an "
                         "input artefact rather than a result.")
    print("VERIFY PASSED: convention is rectangular/unclamped and all required keys are present.")


if __name__ == "__main__":
    main()
