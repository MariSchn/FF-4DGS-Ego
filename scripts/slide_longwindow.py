#!/usr/bin/env python3
"""The long-window world-space table, rendered as one image, with the camera-convention fix marked.

WHY THE OLD VALUES STAY ON THE PAGE. Two of our six cells moved by a lot and two did not move at
all, and the second fact is the one that makes the first believable. C-abs and C-rr are computed by
the same scorer over the same 471 segments before and after, and neither reads a camera pose, so
neither may change. Printing the new numbers alone would hide the control.

Every value is transcribed from a scored artefact, none is typed from memory:
    ours, corrected   $HOME/results/ours_caminv_seg100_both.json   (job 105495, detbox v3, n=157)
    ours, before      the row this table carried until 2026-08-10, from ours_shared_seg100_both
    baselines         unchanged, scored by eval_worldspace_baseline on the same protocol

    python -m scripts.slide_longwindow --out report/slide_longwindow.pdf
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

INK, MUTED, FAINT = "#1b1b1b", "#5c5c5c", "#9a9a9a"
WIN = "#B4436C"
BAND = "#f2e7ec"

HEAD = ["Method", "Regime", "C-abs", "C-rr", "WA-MPJPE", "W-MPJPE", "seg."]
ROWS = [
    ["Ours (feedforward)", "online, no SLAM", "35.4", "26.5", "27.0", "70.7", "471"],
    ["WiLoR + SLAM",       "offline + SLAM",  "83.4", "27.2", "35.2", "129.0", "468"],
    ["HaWoR",              "offline + SLAM",  "87.7", "32.6", "40.3", "133.6", "471"],
    ["HaMeR + SLAM",       "offline + SLAM",  "87.9", "30.3", "36.4", "135.8", "468"],
    ["HaPTIC + SLAM",      "offline + SLAM",  "(157.1)", "29.6", "36.5", "(138.1)", "468"],
    ["Dyn-HaMR",           "offline optim.",  "(1336.7)", "59.9", "49.0", "(195.4)", "468"],
]
# What our row read before the convention fix, per column index. None means the cell did not move.
BEFORE = {2: "35.4", 3: "26.5", 4: "34.2", 5: "113.2"}
BOLD_COL = {2, 3, 4, 5}     # our row leads every metric column


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="slide_longwindow.pdf")
    a = ap.parse_args()

    fig = plt.figure(figsize=(13.33, 7.5), dpi=200)
    fig.patch.set_facecolor("white")

    fig.text(0.055, 0.925, "World-space accuracy, 100-frame segments", fontsize=25, color=INK)
    fig.text(0.055, 0.872,
             "HOI4D, 157 sequences, detector boxes shared across every row, one scorer. "
             "Lower is better.",
             fontsize=12.5, color=MUTED)

    xs = [0.055, 0.225, 0.415, 0.505, 0.635, 0.775, 0.885]
    aligns = ["left", "left", "right", "right", "right", "right", "right"]
    y0, dy = 0.755, 0.058
    EXTRA = 0.030          # our row carries a second line, so it is taller than the rest

    # Band our row so the eye lands on it before reading any number.
    fig.patches.append(plt.Rectangle((0.045, y0 - dy - 0.048), 0.90, 0.076,
                                     transform=fig.transFigure, facecolor=BAND,
                                     edgecolor="none", zorder=0))

    for j, h in enumerate(HEAD):
        fig.text(xs[j], y0, h, fontsize=13, color=INK, ha=aligns[j], weight="bold")
    fig.lines.append(plt.Line2D([0.045, 0.945], [y0 - 0.018, y0 - 0.018],
                                transform=fig.transFigure, color=INK, lw=0.9))

    for i, row in enumerate(ROWS):
        y = y0 - (i + 1) * dy - (EXTRA if i > 0 else 0.0)
        ours = i == 0
        for j, cell in enumerate(row):
            bold = ours and j in BOLD_COL
            fig.text(xs[j], y, cell, fontsize=13.5 if bold else 12.5,
                     color=INK if ours else MUTED, ha=aligns[j],
                     weight="bold" if bold else "normal", zorder=2)
            # The struck-through previous value sits just under the cell it replaced, so the change
            # is legible per cell instead of needing a second table.
            # "was X" rather than a drawn strike-through: at 10 pt the rule lands mid-digit and
            # the number stops being readable, which defeats the point of showing it.
            if ours and j in BEFORE and BEFORE[j] != cell:
                fig.text(xs[j], y - 0.034, f"was {BEFORE[j]}", fontsize=10.5, color=FAINT,
                         ha=aligns[j], zorder=2, style="italic")

    fig.lines.append(plt.Line2D([0.045, 0.945], [y0 - 6 * dy - EXTRA - 0.026, y0 - 6 * dy - EXTRA - 0.026],
                                transform=fig.transFigure, color=INK, lw=0.9))

    fig.text(0.055, 0.275, "What changed, and the control", fontsize=15, color=INK)
    fig.text(0.055, 0.232,
             "The predicted camera trajectory was being applied backwards: the tensor is "
             "world-to-camera, the code that emits it\n"
             "labels it camera-to-world. Over 1570 clips the rotation error is 2.1$^\\circ$ as "
             "emitted against 0.36$^\\circ$ inverted, and the\n"
             "best-fit translation scale is $-$0.944. Correcting it moves W-MPJPE 113.2 to 70.7 "
             "and WA 34.2 to 27.0.",
             fontsize=12.2, color=MUTED, va="top", linespacing=1.6)
    fig.text(0.055, 0.108,
             "C-abs and C-rr do not move by a decimal. Neither reads a camera pose, so neither "
             "may change, and neither does. No\n"
             "weight was retrained and no baseline cell moves: none of them consumes this tensor.",
             fontsize=12.2, color=WIN, va="top", linespacing=1.6)

    fig.text(0.055, 0.035,
             "Parenthesised cells are measured but not attributable: neither method recovers "
             "metric depth under input matching.",
             fontsize=9.5, color=FAINT)

    fig.savefig(a.out, facecolor="white")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, facecolor="white", dpi=160)
    print(f"wrote {a.out} and {png}")


if __name__ == "__main__":
    main()
