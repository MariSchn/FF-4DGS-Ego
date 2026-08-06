"""Figure A3: plot the depth-transfer relation from build_depth_transfer_scatter.py's CSV.

The paper claims cross-dataset transfer is governed by the DEPTH COVERAGE of the training
mixture. This is the scatter that claim was fitted from, plus the two between-store points that
make it a prediction rather than a description.

    python -m scripts.plot_depth_transfer --csv a3.csv --out fig_depth_transfer.pdf
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The two between-store points, from report/open-lines-tracker.md (2026-08-04). Both are
# full-coverage on the identical 757-segment H2O split at the matched 16/8 protocol, so they sit
# on the same axes as the per-sequence cloud.
#   name, shift_mm, C_abs_mm
BETWEEN_STORE = [
    ("trained on HOI4D only",      184.0, 184.8),
    ("trained on a mixture whose\ndepth support contains H2O", 0.0, 66.2),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    x = np.array([float(r["shift_mm"]) for r in rows])
    y = np.array([float(r["c_abs_mm"]) for r in rows])
    n = x.size

    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(5.0, 3.1))

    ax.scatter(x, y, s=11, facecolor="#7FA8BC", edgecolor="none", alpha=0.75,
               label=f"H2O test sequences ($n={n}$)")

    xf = np.linspace(0, max(x.max(), 200) * 1.03, 100)
    ax.plot(xf, slope * xf + intercept, color="#1b1b1b", lw=1.1,
            label=(rf"fit: $C_{{\mathrm{{abs}}}} = {intercept:.0f} + {slope:.2f}\,|\Delta z|$"
                   f"   ($r={r:+.2f}$)"))

    # Label placement is explicit per point: the cloud is dense in the upper right, so the
    # callouts live in the empty regions and use leaders rather than sitting on the data.
    label_xy = {0: (18, 232), 1: (62, 42)}
    for i, (name, bx, by) in enumerate(BETWEEN_STORE):
        ax.scatter([bx], [by], s=52, marker="D", facecolor="#B5453A",
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(name, xy=(bx, by), xytext=label_xy[i], textcoords="data",
                    fontsize=7.2, color="#8d2f27", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#8d2f27", lw=0.6,
                                    shrinkA=2, shrinkB=6))

    ax.set_xlabel(r"$|\Delta z|$: distance from the training depth prior (mm)")
    ax.set_ylabel(r"$C$-MPJPE, absolute (mm)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=300, bbox_inches="tight")

    print(f"n={n}  r={r:+.3f}  slope={slope:+.4f}  intercept={intercept:.1f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
