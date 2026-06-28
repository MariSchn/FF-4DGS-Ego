"""Offline root-smoothing diagnostic (CPU-only, no model / no cluster).

Loads a trajectory dump written by ``eval_world_space --dump_traj`` and sweeps temporal smoothing
of the chained hand-root track, reporting W-MPJPE (and WA-MPJPE-long) vs smoothing window. The
read:

    * W-MPJPE drops steadily with the window and saturates well below the baseline  => the world
      error is high-frequency placement DRIFT/jitter; a smooth feedforward trajectory head will
      recover most of it. Build the head.
    * W-MPJPE is essentially flat                                                    => the error
      is low-frequency per-frame depth BIAS (e.g. HOT3D->HOI4D domain gap); a temporal head alone
      will not fix W. Need an absolute anchor (hand-object contact) or in-domain training.

Because it replays the *same* first-window rigid gauge as the eval, the ``window=1`` row reproduces
the eval's pooled W-MPJPE bit-for-bit (a built-in sanity check).

Usage:
    python -m scripts.analyze_smoothing_diagnostic --dump world_traj.pt \
        --windows 1,3,5,9,15,21,31 --mode gaussian
"""
from __future__ import annotations

import argparse

import torch

from scripts.world_space_metrics import (
    smooth_root_trajectory,
    w_mpjpe_first_window_aligned,
    wa_mpjpe,
)


def _mean(xs):
    xs = [x for x in xs if x == x]                       # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def sweep(segs, windows, mode):
    rows = []
    for w in windows:
        ws, wal = [], []
        for s in segs:
            pred, gt, val = s["pred_world"], s["gt_world"], s["valid"]
            t = min(pred.shape[0], gt.shape[0])
            p, g, v = pred[:t], gt[:t], val[:t]
            sm = smooth_root_trajectory(p, window=int(w), mode=mode)
            ws.append(w_mpjpe_first_window_aligned(sm, g, v, int(s["wa_short"])))
            wal.append(wa_mpjpe(sm, g, window=t, valid=v))
        rows.append((w, _mean(ws), _mean(wal)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="trajectory dump from eval_world_space --dump_traj")
    ap.add_argument("--windows", default="1,3,5,9,15,21,31")
    ap.add_argument("--mode", default="gaussian", choices=["gaussian", "moving"])
    args = ap.parse_args()

    segs = torch.load(args.dump, map_location="cpu")
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    rows = sweep(segs, windows, args.mode)

    base = rows[0][1] if rows and rows[0][0] == 1 else float("nan")
    print(f"{len(segs)} segments | mode={args.mode} | baseline (w=1) W-MPJPE={base:.1f} mm\n")
    print(f"{'window':>7} {'W_MPJPE':>10} {'dW%':>7} {'WA_long':>9}")
    for w, mw, ml in rows:
        d = 100.0 * (mw - base) / base if base == base and base else float("nan")
        print(f"{w:>7} {mw:>10.1f} {d:>6.0f}% {ml:>9.1f}")

    best = min((r for r in rows if r[1] == r[1]), key=lambda r: r[1], default=None)
    if best and base == base:
        drop = 100.0 * (base - best[1]) / base
        verdict = ("DRIFT-dominated: smoothing recovers a large fraction -> a feedforward temporal "
                   "trajectory head should move W. Proceed to build it."
                   if drop >= 15.0 else
                   "BIAS-dominated: smoothing barely helps -> per-frame absolute depth is "
                   "systematically off. A temporal head alone will not fix W; add a contact/scene "
                   "anchor or train in-domain.")
        print(f"\nBest window w={best[0]}: {best[1]:.1f} mm ({drop:.0f}% below baseline). {verdict}")


if __name__ == "__main__":
    main()
