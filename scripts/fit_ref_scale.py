#!/usr/bin/env python
"""Refit the root-anchor ref_scale on TRAIN-split HOI4D sequences only (D2-4).

The anchor's fixed DA3 bias correction (ref_scale=1.121, from a median DA3/GT
wrist-depth ratio of 0.892) was originally measured by da3_hand_probe.py (job
101840) on 8 sequences of the old 11-seq set, which overlaps the current
157-seq TEST split. That makes every anchor-based number test-set-tuned. This
script recomputes the SAME statistic (pooled median of per-frame per-hand
DA3_depth / GT_wrist_depth ratios) on train-split sequences only, and HARD
FAILS if any sequence it would use is in the test split.

Inputs per sequence (same caches the anchor evals read):
  <data_root>/<seq>/hand_data/gt_joints_cache_cam_v2.pt   [N,2,16,3] camera-frame GT (m)
  <da3_wrist_cache_dir>/<seq>_da3_wrist.pt                [N,2] DA3 metric depth at wrist (m, NaN where unavailable)

Usage (any CPU node; no GPU needed):
    python -u -m scripts.fit_ref_scale \
      --data_root /work/scratch/dmonopoli/hoi4d_pp \
      --da3_wrist_cache_dir /home/dmonopoli/da3_wrist_cache \
      --split_json /home/dmonopoli/hoi4d_split.json \
      --out /home/dmonopoli/ref_scale_refit.json

Then rerun the anchor eval with ANCHOR_REF_SCALE=<ref_scale> in the environment
(read by scripts/root_depth_anchor.py, overrides the module constant).
"""
import argparse
import json
import os
import random

import torch

WRIST_J = 0   # MANO joint 0 = wrist (matches scripts.root_depth_anchor.WRIST_J)
RH = 1        # HOI4D releases the right hand -> hand index 1
GT_Z_MIN_M = 0.05   # discard garbage cache entries outside a sane wrist-depth range
GT_Z_MAX_M = 5.0
N_BOOTSTRAP = 1000  # over-sequences bootstrap for the 95% CI


def _has_cam_cache(seq_dir):
    return os.path.exists(os.path.join(seq_dir, "hand_data", "gt_joints_cache_cam_v2.pt"))


def load_split(split_json):
    """Return (train_set, test_set) of seq basenames from the split json."""
    with open(split_json) as f:
        split = json.load(f)
    if not isinstance(split, dict) or "train" not in split or "test" not in split:
        raise SystemExit(f"[fit_ref_scale] {split_json} must be a dict with 'train' and "
                         f"'test' keys; got keys={sorted(split) if isinstance(split, dict) else type(split)}")
    return set(split["train"]), set(split["test"])


def seq_ratios(seq_dir, da3_wrist_cache_dir, hands):
    """Per-frame per-hand DA3/GT wrist-depth ratios for one sequence, or None."""
    da3_path = os.path.join(da3_wrist_cache_dir, f"{os.path.basename(seq_dir)}_da3_wrist.pt")
    if not os.path.exists(da3_path):
        return None
    da3 = torch.load(da3_path, map_location="cpu", weights_only=True).float()          # [N,2]
    gt_cam = torch.load(os.path.join(seq_dir, "hand_data", "gt_joints_cache_cam_v2.pt"),
                        map_location="cpu").float()                                    # [N,2,16,3]
    n = min(da3.shape[0], gt_cam.shape[0])
    if da3.shape[0] != gt_cam.shape[0]:
        print(f"[warn] {os.path.basename(seq_dir)}: da3 N={da3.shape[0]} vs gt N={gt_cam.shape[0]}, "
              f"using first {n}", flush=True)
    da3, gt_z = da3[:n], gt_cam[:n, :, WRIST_J, 2]                                     # [N,2] each
    valid = (torch.isfinite(da3) & (da3 > 0.01)
             & torch.isfinite(gt_z) & (gt_z > GT_Z_MIN_M) & (gt_z < GT_Z_MAX_M))
    if hands == "right":
        keep = torch.zeros_like(valid)
        keep[:, RH] = True
        valid = valid & keep
    ratios = (da3[valid] / gt_z[valid])
    return ratios if ratios.numel() > 0 else None


def bootstrap_ci(per_seq_ratios, n_iter=N_BOOTSTRAP, seed=0):
    """95% CI of the pooled median, resampling SEQUENCES (frames within a seq are
    correlated, so a frame-level bootstrap would be overconfident)."""
    rng = random.Random(seed)
    n = len(per_seq_ratios)
    meds = []
    for _ in range(n_iter):
        pick = [per_seq_ratios[rng.randrange(n)] for _ in range(n)]
        meds.append(float(torch.cat(pick).median()))
    meds.sort()
    return meds[int(0.025 * n_iter)], meds[int(0.975 * n_iter)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="preprocessed HOI4D root; one dir per seq with hand_data/ caches")
    ap.add_argument("--da3_wrist_cache_dir", required=True,
                    help="dir of per-seq DA3 metric wrist-depth caches (<seq>_da3_wrist.pt)")
    ap.add_argument("--split_json", required=True,
                    help="train/test split json (e.g. /home/dmonopoli/hoi4d_split.json); the "
                         "fit uses TRAIN seqs only and hard-fails on any test-split seq")
    ap.add_argument("--seqs", default=None,
                    help="comma-separated seq basenames to restrict the fit to (must all be "
                         "train-split). Default: every train seq with both caches present.")
    ap.add_argument("--hands", choices=["right", "both"], default="right",
                    help="which hand columns to use; HOI4D GT is right-hand only")
    ap.add_argument("--out", default="/tmp/ref_scale_refit.json")
    args = ap.parse_args()

    train_set, test_set = load_split(args.split_json)
    all_seqs = sorted(os.path.join(args.data_root, d) for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d)))
    seqs = [s for s in all_seqs if _has_cam_cache(s)]
    if args.seqs:
        want = {x.strip() for x in args.seqs.split(",") if x.strip()}
        seqs = [s for s in seqs if os.path.basename(s) in want]
        missing = want - {os.path.basename(s) for s in seqs}
        if missing:
            raise SystemExit(f"[fit_ref_scale] requested seqs not found under {args.data_root}: "
                             f"{sorted(missing)}")

    # D2-4 guard: the calibration set must be disjoint from the test split. Fail loudly
    # on ANY test-split or split-unknown sequence rather than silently dropping it.
    names = [os.path.basename(s) for s in seqs]
    contaminated = sorted(n for n in names if n in test_set)
    if contaminated:
        raise SystemExit(f"[fit_ref_scale] REFUSING to fit: {len(contaminated)} test-split "
                         f"seq(s) in the fit set: {contaminated}")
    unknown = sorted(n for n in names if n not in train_set)
    if unknown:
        raise SystemExit(f"[fit_ref_scale] REFUSING to fit: {len(unknown)} seq(s) not in the "
                         f"train split (not in {args.split_json}): {unknown}")

    per_seq_ratios, per_seq = [], {}
    for s in seqs:
        r = seq_ratios(s, args.da3_wrist_cache_dir, args.hands)
        if r is None:
            continue
        per_seq_ratios.append(r)
        per_seq[os.path.basename(s)] = {"n_frames": int(r.numel()),
                                        "median_ratio": float(r.median())}
    if not per_seq_ratios:
        raise SystemExit(f"[fit_ref_scale] no train-split seq had both a cam cache and a DA3 "
                         f"wrist cache ({args.da3_wrist_cache_dir}); build the DA3 caches for "
                         f"the train split first.")

    pooled = torch.cat(per_seq_ratios)
    median_ratio = float(pooled.median())
    ref_scale = 1.0 / median_ratio
    q1, q3 = float(pooled.quantile(0.25)), float(pooled.quantile(0.75))
    lo, hi = bootstrap_ci(per_seq_ratios)
    seq_meds = sorted(v["median_ratio"] for v in per_seq.values())

    result = {
        "ref_scale": ref_scale,
        "median_ratio": median_ratio,          # DA3 / GT wrist depth (0.892 was the test-tainted value)
        "mean_ratio": float(pooled.mean()),
        "ratio_iqr": [q1, q3],
        "median_ratio_ci95_seq_bootstrap": [lo, hi],
        "ref_scale_ci95": [1.0 / hi, 1.0 / lo],
        "per_seq_median_range": [seq_meds[0], seq_meds[-1]],
        "n_seqs": len(per_seq),
        "n_frames": int(pooled.numel()),
        "hands": args.hands,
        "split_json": os.path.abspath(args.split_json),
        "estimator": "pooled median of per-frame DA3/GT wrist-depth ratios "
                     "(replicates da3_hand_probe v2, job 101840); ref_scale = 1/median",
        "per_seq": per_seq,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nTRAIN-split ref_scale refit (n={result['n_seqs']} seqs, {result['n_frames']} frames):")
    print(f"  median DA3/GT ratio = {median_ratio:.4f}  (IQR {q1:.4f}-{q3:.4f}; "
          f"seq-bootstrap 95% CI {lo:.4f}-{hi:.4f})")
    print(f"  ref_scale = 1/median = {ref_scale:.4f}  (95% CI {1.0 / hi:.4f}-{1.0 / lo:.4f})")
    print(f"  -> {args.out}")
    print(f"  rerun anchor evals with: ANCHOR_REF_SCALE={ref_scale:.4f}")


if __name__ == "__main__":
    main()
