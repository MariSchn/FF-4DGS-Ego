"""Registration panel R2: the scale solve, as a population rather than a scalar.

The paper states a single number for the scene scale s. This plots what it is actually the
median OF, over every scored segment, which is the honest form of the claim and is what the
supervisor asked to see (2026-08-06, "visual intermediate results for the registration steps").

Reads the per-segment `s med/pool=...` lines that eval_world_space prints, so it needs no GPU
and no re-run.

    python -m scripts.plot_scale_distribution --s_file s_new.txt --out fig_scale.pdf
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The reference scale: the similarity that maps our predicted (up-to-scale) camera centres onto
# the GT metric ones. Measured at 1.0230 on the sequences that carry GT camera centres
# (ours_shortwindow_16_8.json, s_gt_med, n=12). A correct solve should sit here.
S_GT = 1.0230
CLAMP_LO, CLAMP_HI = 0.1, 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_file", required=True, help="lines of 's_med/s_pool'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rows = [l.strip().split("/") for l in open(args.s_file) if "/" in l]
    s_med = np.array([float(a) for a, _ in rows])
    s_pool = np.array([float(b) for _, b in rows])
    n = s_med.size
    at_floor = int((s_med <= CLAMP_LO + 1e-4).sum())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(5.0, 2.5))

    bins = np.linspace(0.0, 1.6, 65)
    ax.hist(s_med, bins=bins, color="#CFE0E6", edgecolor="#4a6b78", linewidth=0.5,
            label=f"per-segment $s$  ($n={n}$)")

    med = float(np.median(s_med))
    mean = float(s_med.mean())
    ax.axvline(med, color="#1b1b1b", lw=1.2, label=f"median $s={med:.3f}$")
    ax.axvline(mean, color="#1b1b1b", lw=1.0, ls=(0, (4, 2)), label=f"mean $s={mean:.3f}$")
    ax.axvline(S_GT, color="#B5453A", lw=1.2, label=f"true scale $={S_GT:.3f}$")

    # The clamp floor is not a tail, it is a failure mode: mark it as one.
    ax.axvspan(0.0, CLAMP_LO, color="#B5453A", alpha=0.12, lw=0)
    ax.annotate(f"{at_floor} of {n} segments ({100*at_floor/n:.0f}%)\npinned at the clamp floor $s={CLAMP_LO}$",
                xy=(CLAMP_LO, ax.get_ylim()[1] * 0.62),
                xytext=(0.30, ax.get_ylim()[1] * 0.80),
                fontsize=7.5, color="#8d2f27", ha="left",
                arrowprops=dict(arrowstyle="->", color="#8d2f27", lw=0.7))

    ax.set_xlabel(r"solved scene scale $s=\mathrm{med}\,(z_{\mathrm{hand}}/d_{\mathrm{scene}})$")
    ax.set_ylabel("segments")
    ax.set_xlim(0.0, 1.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    if args.title:
        ax.set_title(args.title, fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=300, bbox_inches="tight")

    print(f"n={n}  median={med:.4f}  mean={mean:.4f}  at_floor={at_floor} ({100*at_floor/n:.1f}%)")
    print(f"pool: median={np.median(s_pool):.4f}  mean={s_pool.mean():.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
