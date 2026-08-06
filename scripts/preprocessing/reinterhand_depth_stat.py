#!/usr/bin/env python3
"""Median camera-frame WRIST DEPTH of a converted store, with 10/90 percentiles.

WHY THIS NUMBER DECIDES WHETHER Re:InterHand IS WORTH ADDING. Our measured transfer law is
C_abs ~= 60 mm + 0.50 * |depth shift| (report/open-lines-tracker.md, 2026-08-04): a training
mixture whose hand depths BRACKET the evaluation set's collapses the shift term, and one that
sits far from it pays ~0.5 mm per mm of offset. HOI4D-only -> H2O zero-shot scored 184.8; mix3,
whose depths bracket H2O, scored 66.2 on the identical split. So a new store earns its place by
WHERE its hands sit in camera depth, not by how many frames it ships.

Current pool (report/open-lines-tracker.md:24-25):
    TRAIN  HOT3D 0.339 | OakInk2 0.386 | ARCTIC 0.474 m
    EVAL   H2O   0.503 | HOI4D   0.677 m     (never trained on; every number is zero-shot)

Reads only our store format, so it works on ANY converted store, not just Re:InterHand:
    <data_root>/<seq>/hand_data/gt_joints_cache_cam_v2.pt   [N, 2, 16, 3] float32, CAMERA frame,
                                                            METRES; hand axis 0 = left, 1 = right
    <data_root>/<seq>/hand_data/hand_bboxes_v2_*.pt         optional; its "valid" [N,2] masks
                                                            frames where the hand is absent
Wrist is joint 0 of the smplx-16 layout, its depth is channel 2 (scripts/fit_ref_scale.py:33-34).

OVERLAP TO CONSOLIDATE: scripts/preprocessing/dexycb_depth_stat.py --store computes the SAME
statistic (median + 10/90 of joint-0 z) over the SAME (0.05, 3.0) m window on the same cache, and
also accepts raw DexYCB as a source. The two must never disagree on a shared store, so if one is
changed, change both or merge them. The only behavioural difference is that this script also masks
on the box cache's per-hand "valid" flag; that is a no-op wherever absent hands are stored as zeros
(0 m is already outside the window), and a real difference only on a store that writes NaN or a
plausible-looking placeholder for an absent hand.

BEFORE TRUSTING THE COMPARISON: the pooling method behind the 0.339/0.386/0.474 reference values
is not recorded anywhere in the repo. Run this script against an EXISTING store first and check
it reproduces that store's published number; only then compare a new store against the table.
The script prints both poolings (over frames and over sequence medians) so the discrepancy, if
any, is visible rather than hidden.

Usage:
    python -m scripts.preprocessing.reinterhand_depth_stat --data_root $S/reinterhand_ours
    python -m scripts.preprocessing.reinterhand_depth_stat --data_root $S/arctic_ours   # calibrate
    python -m scripts.preprocessing.reinterhand_depth_stat --data_root $S/x --per_seq --json out.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

WRIST_J = 0            # smplx-16 joint 0 = wrist (scripts/root_depth_anchor.py WRIST_J)
Z_AXIS = 2             # camera-frame depth
HAND_IDX = {"left": 0, "right": 1}          # train_hand_head.py:712-713

# Non-physical rejection band, metres. Anything outside is a cache defect (a zero-filled row, a
# behind-camera hand, a unit slip), not a hand, and must not move the median.
Z_MIN_M, Z_MAX_M = 0.05, 3.0

# Published pool depths, metres - report/open-lines-tracker.md:24-25 (2026-08-04 lock).
REFERENCE_TRAIN = {"HOT3D": 0.339, "OakInk2": 0.386, "ARCTIC": 0.474}
REFERENCE_EVAL = {"H2O": 0.503, "HOI4D": 0.677}


def sequence_wrist_depths(seq_dir: str, hands: str) -> np.ndarray:
    """Every accepted wrist depth of one sequence, metres, flattened over (frame, hand).

    Validity comes from three independent sources, all of which must hold:
      * the joint cache entry is finite (older stores - arctic_to_ours, oakink2_to_ours - write
        NaN for an unfitted hand; newer ones write zeros),
      * the box cache's per-hand "valid" flag when the file exists (the store-wide notion of
        "this hand is really there"), and
      * the physical band applied by the caller, which also removes the zero-filled absent hands.
    Skipping the box-cache mask would pull in hands the store itself considers absent.
    """
    cache = os.path.join(seq_dir, "hand_data", "gt_joints_cache_cam_v2.pt")
    if not os.path.exists(cache):
        return np.empty(0, np.float64)
    j = torch.load(cache, map_location="cpu", weights_only=True).float().numpy()   # [N,2,16,3]
    if j.ndim != 4 or j.shape[1] < 2 or j.shape[2] < 1:
        raise ValueError(f"{cache}: expected [N,2,16,3], got {tuple(j.shape)}")
    z = j[:, :, WRIST_J, Z_AXIS].astype(np.float64)                                # [N,2]

    ok = np.isfinite(z)
    hv = _hand_valid_mask(seq_dir, z.shape)
    if hv is not None:
        ok &= hv

    keep = np.zeros_like(ok)
    for name, hi in HAND_IDX.items():
        if hands in ("both", name):
            keep[:, hi] = True
    ok &= keep
    return z[ok]


def _hand_valid_mask(seq_dir: str, shape: tuple[int, int]) -> np.ndarray | None:
    """[N,2] bool from whichever hand_bboxes_v2_* cache the store carries, or None."""
    hd = os.path.join(seq_dir, "hand_data")
    if not os.path.isdir(hd):
        return None
    names = sorted(n for n in os.listdir(hd) if n.startswith("hand_bboxes_v2") and n.endswith(".pt"))
    for n in names:
        try:
            d = torch.load(os.path.join(hd, n), map_location="cpu", weights_only=True)
        except Exception:                       # noqa: BLE001 - a bad box file must not stop the stat
            continue
        v = d.get("valid") if isinstance(d, dict) else None
        if v is None:
            continue
        v = np.asarray(v).astype(bool)
        if v.shape == shape:
            return v
    return None


def summarise(z: np.ndarray) -> dict:
    """median + 10/90 percentiles of an already-filtered depth sample, metres."""
    if z.size == 0:
        return {"n": 0, "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {"n": int(z.size),
            "median": float(np.median(z)),
            "p10": float(np.percentile(z, 10)),
            "p90": float(np.percentile(z, 90))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", required=True, help="converted store root (dirs of sequences)")
    ap.add_argument("--hands", default="both", choices=["both", "left", "right"],
                    help="which hand slots to pool; index 0 = left, 1 = right")
    ap.add_argument("--per_seq", action="store_true", help="print a line per sequence")
    ap.add_argument("--label", default="", help="name for this store in the output")
    ap.add_argument("--json", default="", help="also write the numbers to this path")
    args = ap.parse_args()

    seqs = sorted(d for d in os.listdir(args.data_root)
                  if os.path.isdir(os.path.join(args.data_root, d)))
    if not seqs:
        raise SystemExit(f"no sequence dirs under {args.data_root}")

    pooled, per_seq_median, n_raw, n_empty = [], [], 0, 0
    rows = []
    for s in seqs:
        try:
            z = sequence_wrist_depths(os.path.join(args.data_root, s), args.hands)
        except (ValueError, RuntimeError) as e:
            print(f"  !! {s}: {e}")
            continue
        n_raw += int(z.size)
        z = z[(z > Z_MIN_M) & (z < Z_MAX_M)]          # the non-physical rejection
        if z.size == 0:
            n_empty += 1
            continue
        pooled.append(z)
        st = summarise(z)
        per_seq_median.append(st["median"])
        rows.append({"seq": s, **st})
        if args.per_seq:
            print(f"  {s:<40s} n={st['n']:<7d} median {st['median']:.3f}  "
                  f"p10 {st['p10']:.3f}  p90 {st['p90']:.3f}")

    if not pooled:
        raise SystemExit(f"no usable wrist depths under {args.data_root} - either the store has no "
                         f"gt_joints_cache_cam_v2.pt, or every depth fell outside "
                         f"({Z_MIN_M}, {Z_MAX_M}) m, which would itself be the finding")

    allz = np.concatenate(pooled)
    n_kept = int(allz.size)
    st = summarise(allz)
    label = args.label or os.path.basename(args.data_root.rstrip("/"))

    print(f"\nSTORE {label}  ({args.data_root})")
    print(f"  sequences {len(rows)}/{len(seqs)} usable"
          + (f", {n_empty} with no accepted depth" if n_empty else ""))
    print(f"  samples   {n_kept} kept of {n_raw} ({n_raw - n_kept} rejected outside "
          f"({Z_MIN_M}, {Z_MAX_M}) m), hands={args.hands}")
    print(f"  WRIST DEPTH (pooled over frames)   median {st['median']:.3f} m   "
          f"p10 {st['p10']:.3f}   p90 {st['p90']:.3f}")
    # Pooling over frames weights long sequences more heavily. The sequence-median distribution is
    # the robustness check: a big gap between the two means one long capture is driving the number.
    sm = summarise(np.asarray(per_seq_median))
    print(f"  same, pooled over SEQUENCE medians median {sm['median']:.3f} m   "
          f"p10 {sm['p10']:.3f}   p90 {sm['p90']:.3f}")
    if np.isfinite(st["median"]) and np.isfinite(sm["median"]) and abs(st["median"] - sm["median"]) > 0.05:
        print("  !! the two poolings differ by >5 cm: one long sequence dominates the frame-pooled "
              "number. Quote the sequence-median one, or rebalance.")

    print("\nREFERENCE (report/open-lines-tracker.md:24-25)")
    print("  TRAIN " + " | ".join(f"{k} {v:.3f}" for k, v in REFERENCE_TRAIN.items()))
    print("  EVAL  " + " | ".join(f"{k} {v:.3f}" for k, v in REFERENCE_EVAL.items()))

    # The actionable read: does adding this store BRACKET the two held-out eval depths? That is the
    # mechanism mix3 demonstrated (66.2 vs 184.8 on H2O), so it is the criterion, not "is it deep".
    train_now = list(REFERENCE_TRAIN.values())
    lo_before, hi_before = min(train_now), max(train_now)
    lo_after, hi_after = min(train_now + [st["median"]]), max(train_now + [st["median"]])
    print(f"\nTRAIN-POOL COVERAGE  before [{lo_before:.3f}, {hi_before:.3f}] m  ->  "
          f"after [{lo_after:.3f}, {hi_after:.3f}] m")
    for name, ev in REFERENCE_EVAL.items():
        was = lo_before <= ev <= hi_before
        now = lo_after <= ev <= hi_after
        verdict = "already bracketed" if was and now else ("NEWLY BRACKETED" if now else "still outside")
        print(f"  {name} at {ev:.3f} m: {verdict}")
    gained = (hi_after - lo_after) - (hi_before - lo_before)
    print(f"  depth span gained: {gained * 1000:.0f} mm"
          + ("  (adds no new coverage on its own)" if gained <= 0.001 else ""))
    print("\nCAVEAT: the pooling method behind the reference values is not recorded. Reproduce one "
          "of them with this same script before trusting the comparison.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"label": label, "data_root": args.data_root, "hands": args.hands,
                       "z_band_m": [Z_MIN_M, Z_MAX_M],
                       "pooled_over_frames": st, "pooled_over_sequence_medians": sm,
                       "n_sequences": len(rows), "n_samples_raw": n_raw, "n_samples_kept": n_kept,
                       "reference_train": REFERENCE_TRAIN, "reference_eval": REFERENCE_EVAL,
                       "per_sequence": rows}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
