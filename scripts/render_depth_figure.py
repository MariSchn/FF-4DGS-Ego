"""Render the CVPR-ready predicted-metric-depth figure from the off-node npz (runs on the Mac,
where matplotlib works; the cluster's is broken). Poster style: Latin Modern serif body.

Layout: one row per scene, columns RGB | Predicted depth | GT depth | |error|. Predicted and GT
share an identical colormap and range per row, so "they look the same" reads instantly; the error
column is near-black where the prediction is good. A title carries the headline metrics.

Usage:
    python scripts/render_depth_figure.py --npz /tmp/depth_fig.npz --out report/depth_vs_gt
"""
from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

DEPTH_CMAP = "turbo"     # standard for depth maps; pred/GT share it so equality is obvious
ERR_CMAP = "magma"
ERR_VMAX = 0.30          # metres


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="report/depth_vs_gt", help="output path stem (.pdf and .png)")
    ap.add_argument("--err_vmax", type=float, default=ERR_VMAX)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    rgb, pred, gt, mask = d["rgb"], d["pred"], d["gt"], d["mask"].astype(bool)
    n = rgb.shape[0]
    absrel = float(d["absrel"]) if "absrel" in d else float("nan")
    mae_cm = float(d["mae_cm"]) if "mae_cm" in d else float("nan")
    d1 = float(d["d1"]) if "d1" in d else float("nan")

    cols = ["RGB", "Predicted depth", "GT depth (sensor)", "$|$error$|$"]
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.3 * n))
    if n == 1:
        axes = axes[None, :]

    for r in range(n):
        gt_r = gt[r].copy()
        m = mask[r]
        # shared depth range from the valid GT pixels of this row (robust to sensor dropouts)
        vmax = float(np.percentile(gt_r[m], 95)) if m.any() else 3.0
        vmin = float(np.percentile(gt_r[m], 5)) if m.any() else 0.0
        gt_disp = np.where(m, gt_r, np.nan)
        err = np.where(m, np.abs(pred[r] - gt_r), np.nan)

        for c, (ax, title) in enumerate(zip(axes[r], cols)):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.imshow(np.clip(rgb[r], 0, 1))
            elif c == 1:
                im = ax.imshow(pred[r], cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax)
            elif c == 2:
                imd = matplotlib.colormaps[DEPTH_CMAP].copy(); imd.set_bad("white")
                im = ax.imshow(gt_disp, cmap=imd, vmin=vmin, vmax=vmax)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("m", fontsize=11)
            else:
                ime = matplotlib.colormaps[ERR_CMAP].copy(); ime.set_bad("white")
                im = ax.imshow(err, cmap=ime, vmin=0, vmax=args.err_vmax)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("m", fontsize=11)
            if r == 0:
                ax.set_title(title, fontsize=16, pad=6)

    fig.suptitle(
        f"Predicted metric depth vs ground truth (HOI4D):  "
        f"AbsRel {absrel:.3f},  MAE {mae_cm:.0f} cm,  "
        r"$\delta\!<\!1.25$ " + f"{d1*100:.1f}%",
        fontsize=19)
    # leave a clear band at the top for the suptitle so it never collides with the column headers
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.5 / (3.3 * n)], h_pad=1.2, w_pad=0.4)

    fig.savefig(f"{args.out}.pdf", transparent=True, bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}.pdf and {args.out}.png  ({n} scenes)")


if __name__ == "__main__":
    main()
