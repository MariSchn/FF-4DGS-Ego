"""Generate the Gaussian-head ablation bar chart (Table 1) for the poster.

Real numbers from the FF-4DGS-Ego evaluation on Hot3D (GS reconstruction quality).
Matches make_hand_metrics_chart.py: LaTeX serif font, ETH palette, full frame, and
a shared legend on top (no per-bar labels) so it stays a short, compact band.

Two styles are produced in one run:
  * "poster" - ETH brand colours (default; matches the tables and the hand chart).
  * "drawio" - the pastel palette of the Figure-1 architecture diagram.

    python poster_assets/make_bar_chart.py

Each style writes a vector PDF (for the A0 poster) and a 300-dpi PNG preview.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ------------------------------------------------------------------
# Data (Table 1), keyed by method. Edit numbers here only.
# ------------------------------------------------------------------
PSNR  = {"frozen": 35.748, "trained": 36.963, "ours": 37.922}   # higher is better
SSIM  = {"frozen": 0.970,  "trained": 0.977,  "ours": 0.977}    # higher is better
LPIPS = {"frozen": 0.008,  "trained": 0.005,  "ours": 0.004}    # lower  is better

LABELS = {
    "frozen":  "NeoVerse (frozen)",
    "trained": "NeoVerse (trained)",
    "ours":    "Hand Inj. (Ours)",
}
ORDER = ["frozen", "trained", "ours"]
BEST_KEY = "ours"   # highlighted method

# ------------------------------------------------------------------
# Palettes. (fill, edge) per method; poster puts "ours" in ETH blue,
# drawio mirrors the architecture figure (green = our new modules).
# ------------------------------------------------------------------
ETH_GRAY   = (111 / 255, 111 / 255, 111 / 255)
BLUE_LITE  = (155 / 255, 182 / 255, 219 / 255)   # light blue (mirrors the hand chart's light green)
ETH_BLUE   = (33 / 255,  92 / 255,  175 / 255)
BLUE_DARK  = (18 / 255,  55 / 255,  110 / 255)

POSTER_FILL = {"frozen": ETH_GRAY, "trained": BLUE_LITE, "ours": ETH_BLUE}
DRAWIO_FILL = {"frozen": "#f5f5f5", "trained": "#dae8fc", "ours": "#d5e8d4"}


def _edges(fill, style):
    if style == "drawio":
        return {"frozen": "#666666", "trained": "#6c8ebf", "ours": "#82b366"}
    return {k: tuple(c * 0.65 for c in v) for k, v in fill.items()}


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.linewidth": 1.1,
    "svg.fonttype": "none",
})

FS_TITLE = 23   # per-metric panel title
FS_VALUE = 18   # value labels on top of the bars
FS_LEGEND = 20
X_GAP = 1.0


def _panel(ax, data, title, ylim, fmt, fill, edge, style):
    x = np.arange(len(ORDER)) * X_GAP
    values = [data[k] for k in ORDER]
    bars = ax.bar(x, values, width=0.74,
                  color=[fill[k] for k in ORDER], edgecolor=[edge[k] for k in ORDER],
                  linewidth=1.6 if style == "drawio" else 1.2, zorder=3)
    bi = ORDER.index(BEST_KEY)
    bars[bi].set_linewidth(2.6 if style == "drawio" else 2.2)
    bars[bi].set_edgecolor("#3d6b2e" if style == "drawio" else BLUE_DARK)

    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.set_xticks([])                       # methods are in the shared legend
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FS_VALUE - 3)
    box_color = "#999999" if style == "drawio" else "#333333"
    for s in ax.spines.values():
        s.set_color(box_color)
        s.set_linewidth(1.1)
    ax.grid(axis="both" if style == "drawio" else "y",
            color="#dddddd" if style == "drawio" else "0.85", linewidth=0.85, zorder=0)

    span = ylim[1] - ylim[0]
    for k, rect, v in zip(ORDER, bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + span * 0.025, fmt.format(v),
                ha="center", va="bottom", fontsize=FS_VALUE,
                fontweight="bold" if k == BEST_KEY else "normal")


def render(style):
    fill = DRAWIO_FILL if style == "drawio" else POSTER_FILL
    edge = _edges(fill, style)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _panel(axes[0], PSNR,  r"PSNR (dB) $\uparrow$", (35.0, 38.6),  "{:.2f}", fill, edge, style)
    _panel(axes[1], SSIM,  r"SSIM $\uparrow$",      (0.960, 0.982), "{:.3f}", fill, edge, style)
    _panel(axes[2], LPIPS, r"LPIPS $\downarrow$",   (0.0, 0.0098),  "{:.3f}", fill, edge, style)

    fig.tight_layout(w_pad=2.5, rect=(0, 0.11, 1, 1))
    # One legend entry per panel, each centred under its own panel (3 panels = 3 methods).
    for ax, k in zip(axes, ORDER):
        h = Patch(facecolor=fill[k], edgecolor=edge[k], linewidth=1.2, label=LABELS[k])
        ax.legend(handles=[h], loc="upper center", bbox_to_anchor=(0.5, -0.03),
                  frameon=False, fontsize=FS_LEGEND, handlelength=1.4, handletextpad=0.6)

    suffix = "_drawio" if style == "drawio" else ""
    out_pdf = f"poster_assets/gs_ablation_barchart{suffix}.pdf"
    out_png = f"poster_assets/gs_ablation_barchart{suffix}.png"
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


for _style in ("poster", "drawio"):
    render(_style)
