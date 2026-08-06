"""Figure A3: absolute error against how far a test sequence sits from the training depth prior.

This is the paper's transfer claim, and until now it existed only as prose and one fitted line
(`C_abs ~= 60 + 0.50 * |depth shift|`, r = +0.76). The claim is that zero-shot absolute error is
governed by the DEPTH COVERAGE of the training mixture rather than its size, so it deserves the
scatter it was fitted from.

Per test sequence:
    x = |z_prior - z_seq|   metres, how far that sequence's hands sit from the training prior
    y = C-MPJPE absolute    mm, averaged over that sequence's scored segments

Runs CPU-only off the store's GT cache and an existing result JSON, so it needs no GPU and no
re-run. Emits a CSV; plotting is separate (scripts/plot_depth_transfer.py) so the figure can be
restyled without touching the cluster.

    python -m scripts.build_depth_transfer_scatter \
        --result results/h2o_zeroshot_jitterrob.json \
        --data_root /home/dmonopoli/h2o_currentproto \
        --z_prior 0.687 --out a3_depth_transfer.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch

RIGHT = 1     # hand axis: RH=1, matching the convention the C-MPJPE scorer uses
WRIST = 0     # MANO joint 0
DEPTH_MIN = 0.05   # metres; drop unlabelled/degenerate frames rather than letting zeros pull the median


def seq_wrist_depth(data_root: str, seq: str) -> float | None:
    """Median right-wrist camera-frame depth for one sequence, or None if unusable."""
    p = os.path.join(data_root, seq, "hand_data", "gt_joints_cache_cam_v2.pt")
    if not os.path.exists(p):
        return None
    j = torch.load(p, map_location="cpu", weights_only=False)   # [T, 2, 16, 3]
    if j.dim() != 4 or j.shape[1] < 2:
        return None
    z = j[:, RIGHT, WRIST, 2].float()
    z = z[torch.isfinite(z) & (z > DEPTH_MIN)]
    return float(z.median()) if z.numel() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="world-eval JSON with per_segment[seq, C_MPJPE_abs]")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--z_prior", type=float, required=True,
                    help="median right-wrist depth of the TRAINING mixture, metres")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_seg = json.load(open(args.result))["per_segment"]
    by_seq: dict[str, list[float]] = {}
    for s in per_seg:
        v = s.get("C_MPJPE_abs")
        if isinstance(v, (int, float)) and v == v:
            by_seq.setdefault(s["seq"], []).append(float(v))

    rows, skipped = [], 0
    for seq, vals in sorted(by_seq.items()):
        z = seq_wrist_depth(args.data_root, seq)
        if z is None:
            skipped += 1
            continue
        rows.append({
            "seq": seq,
            "z_seq_m": round(z, 5),
            "shift_mm": round(abs(args.z_prior - z) * 1000.0, 3),
            "c_abs_mm": round(sum(vals) / len(vals), 4),
            "n_seg": len(vals),
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seq", "z_seq_m", "shift_mm", "c_abs_mm", "n_seg"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    if n >= 2:
        xs = [r["shift_mm"] for r in rows]
        ys = [r["c_abs_mm"] for r in rows]
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        r = sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")
        slope = sxy / sxx if sxx else float("nan")
        print(f"n={n} sequences (skipped {skipped} with no usable GT)")
        print(f"pearson r = {r:+.3f}   slope = {slope:+.4f} mm per mm   "
              f"intercept = {my - slope * mx:.1f} mm")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
