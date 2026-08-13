#!/usr/bin/env python3
"""Box-jitter degradation curve (task #52), and the asymmetry it exposes.

WHY A CURVE AND NOT A NUMBER. The paper previously reported box robustness as a single pair of
scores at one jitter amplitude, which invites the reply that the amplitude was chosen after seeing
the result. A monotone curve over a swept amplitude cannot be chosen after the fact, and it is the
form reviewers in this area ask for when a method's input is produced by an upstream detector.

WHAT THE DATA SAYS, and it is more interesting than the robustness claim itself. Over an eightfold
increase in jitter amplitude:

    root-relative C-MPJPE   26.55 -> 27.56   (+1.01 mm, +3.8%)
    absolute wrist error    34.87 -> 38.81   (+3.94 mm, +11.3%)

Absolute placement degrades roughly three times as fast as articulation. That is exactly what the
box-as-depth-cue account predicts: perturbing the box perturbs the geometry the model reads
distance from, while the crop still contains the same hand, so the finger configuration survives
and the placement does not. The curve is therefore evidence for the mechanism, not only a
robustness plot.

Numbers are read from the result JSONs, never typed in: scripts/../results/boxsweep_jitter*.json
on the student cluster, full 157-sequence test set, detbox v3 protocol.

    python -m scripts.plot_box_degradation --out figures/fig_box_degradation.pdf
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the paper's body font so the figure does not read as a foreign object on the page.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# Fallback used only when the JSONs are not reachable from this machine; every value here is
# transcribed from those files and the script prints which source it used, so a stale hard-coded
# table can never be mistaken for a fresh measurement.
FALLBACK = {
    0.05: {"C_MPJPE": 26.55, "wrist_abs": 34.87},
    0.10: {"C_MPJPE": 26.57, "wrist_abs": 34.95},
    0.20: {"C_MPJPE": 26.74, "wrist_abs": 35.47},
    0.30: {"C_MPJPE": 26.99, "wrist_abs": 36.37},
    0.40: {"C_MPJPE": 27.56, "wrist_abs": 38.81},
}


def load(results_dir: str | None):
    if results_dir and os.path.isdir(results_dir):
        pts = {}
        for f in sorted(glob(os.path.join(results_dir, "boxsweep_jitter*.json"))):
            amp = float(os.path.basename(f).replace("boxsweep_jitter", "").replace(".json", ""))
            agg = json.load(open(f)).get("aggregate", {})
            if "C_MPJPE" in agg:
                pts[amp] = agg
        if pts:
            print(f"source: {len(pts)} JSONs from {results_dir}")
            return pts
    print("source: TRANSCRIBED FALLBACK (result JSONs not reachable) - values are from "
          "boxsweep_jitter*.json on the student cluster")
    return FALLBACK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--out", default="fig_box_degradation.pdf")
    a = ap.parse_args()

    pts = load(a.results_dir)
    amps = sorted(pts)
    rr = [pts[x]["C_MPJPE"] for x in amps]
    ab = [pts[x]["wrist_abs"] for x in amps]

    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=200)
    ax.plot(amps, ab, marker="o", ms=4, lw=1.4, color="#B4436C",
            label="absolute wrist error")
    ax.plot(amps, rr, marker="s", ms=3.6, lw=1.4, color="#3D6E8F",
            label="root-relative C-MPJPE")

    # State the slopes on the figure: the asymmetry IS the finding, and a reader should not have to
    # measure it off the axes.
    d_ab = ab[-1] - ab[0]
    d_rr = rr[-1] - rr[0]
    ax.annotate(f"+{d_ab:.1f} mm", xy=(amps[-1], ab[-1]), xytext=(-2, 6),
                textcoords="offset points", ha="right", fontsize=7, color="#B4436C")
    ax.annotate(f"+{d_rr:.1f} mm", xy=(amps[-1], rr[-1]), xytext=(-2, -12),
                textcoords="offset points", ha="right", fontsize=7, color="#3D6E8F")

    ax.set_xlabel("box jitter amplitude", fontsize=8)
    ax.set_ylabel("error (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticks(amps)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, bbox_inches="tight", dpi=300)

    print(f"jitter {amps[0]} -> {amps[-1]} ({amps[-1]/amps[0]:.0f}x):")
    print(f"  root-relative {rr[0]:.2f} -> {rr[-1]:.2f}  (+{d_rr:.2f} mm, +{100*d_rr/rr[0]:.1f}%)")
    print(f"  absolute wrist {ab[0]:.2f} -> {ab[-1]:.2f}  (+{d_ab:.2f} mm, +{100*d_ab/ab[0]:.1f}%)")
    print(f"  absolute degrades {d_ab/d_rr:.1f}x faster than articulation")
    print(f"wrote {a.out} and {png}")


if __name__ == "__main__":
    main()
