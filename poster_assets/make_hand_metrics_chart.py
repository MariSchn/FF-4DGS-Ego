"""Generate the hand-head ablation bar chart (Table 2) for the poster.

Real numbers from the FF-4DGS-Ego hand-mesh evaluation on Hot3D, in the same
poster style as make_bar_chart.py (LaTeX serif font, ETH palette, full frame).

Compact 1x4 layout (one row, four metric panels) with a shared legend instead of
per-bar labels, so it stays short on the poster. "Local-Global Fusion"
(= cross-attention) is our best config and is highlighted.

    python poster_assets/make_hand_metrics_chart.py

Set INCLUDE_DECOUPLE = True to add the (dominated) "+ Decouple" variant.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ------------------------------------------------------------------
# Label for the best config. Switch to "Cross Attention" if you prefer the
# precise name (keep §2 / conclusion wording consistent).
# ------------------------------------------------------------------
BEST_LABEL = "Local-Global Fusion"
INCLUDE_DECOUPLE = False

# ------------------------------------------------------------------
# Data (Table 2), keyed by method. Edit numbers here only.
# ------------------------------------------------------------------
PA_MPVPE = {"pretrained": 12.115, "ours": 7.887, "ca": 7.558, "decouple": 8.376}  # down
PA_MPJPE = {"pretrained": 9.028,  "ours": 5.927, "ca": 5.713, "decouple": 6.255}  # down
AUC_V    = {"pretrained": 0.761,  "ours": 0.843, "ca": 0.849, "decouple": 0.833}  # up
AUC_J    = {"pretrained": 0.820,  "ours": 0.881, "ca": 0.886, "decouple": 0.875}  # up

LABELS = {
    "pretrained": "HaMeR (pre-trained)",
    "ours":       "Vanilla HaMeR head (w/o fusion)",   # our retrained head, no local-global fusion
    "ca":         f"{BEST_LABEL} (Ours)",               # full method = best config
    "decouple":   "+ Decouple",
}
BEST_KEY = "ca"   # highlighted method (best across all four metrics)

# Green family -> ties to the hand modules (green) in the Figure-1 architecture
# diagram, distinguishing this hand-head ablation from the blue GS/scene chart.
ETH_GRAY    = (111 / 255, 111 / 255, 111 / 255)
GREEN_BASE  = (140 / 255, 196 / 255, 148 / 255)  # our base head (light green)
GREEN_BEST  = (47 / 255,  133 / 255, 62 / 255)   # best config (deep green)
GREEN_EDGE  = (24 / 255,  80 / 255,  30 / 255)   # dark green outline for the best bar
ETH_BRONZE  = (142 / 255, 103 / 255, 19 / 255)   # optional "+ Decouple"
FILL = {"pretrained": ETH_GRAY, "ours": GREEN_BASE, "ca": GREEN_BEST, "decouple": ETH_BRONZE}
EDGE = {k: tuple(c * 0.6 for c in v) for k, v in FILL.items()}
BEST_EDGE = GREEN_EDGE

ORDER = ["pretrained", "ours", "ca"] + (["decouple"] if INCLUDE_DECOUPLE else [])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.linewidth": 1.1,
    "svg.fonttype": "none",
})

FS_TITLE = 23
FS_VALUE = 18
X_GAP = 1.0


def _panel(ax, data, title, ylim, fmt):
    x = np.arange(len(ORDER)) * X_GAP
    values = [data[k] for k in ORDER]
    fills = [FILL[k] for k in ORDER]
    edges = [EDGE[k] for k in ORDER]

    bars = ax.bar(x, values, width=0.74, color=fills, edgecolor=edges, linewidth=1.2, zorder=3)
    bi = ORDER.index(BEST_KEY)
    bars[bi].set_linewidth(2.2)
    bars[bi].set_edgecolor(BEST_EDGE)

    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.set_xticks([])                       # methods are in the shared legend
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FS_VALUE - 3)
    for s in ax.spines.values():
        s.set_color("#333333")
        s.set_linewidth(1.1)
    ax.grid(axis="y", color="0.85", linewidth=0.9, zorder=0)

    span = ylim[1] - ylim[0]
    for k, rect, v in zip(ORDER, bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + span * 0.025, fmt.format(v),
                ha="center", va="bottom", fontsize=FS_VALUE,
                fontweight="bold" if k == BEST_KEY else "normal")


fig, axes = plt.subplots(1, 4, figsize=(16, 4.7))
_panel(axes[0], PA_MPVPE, r"PA-MPVPE (mm) $\downarrow$", (0, 13.5), "{:.2f}")
_panel(axes[1], PA_MPJPE, r"PA-MPJPE (mm) $\downarrow$", (0, 10.0), "{:.2f}")
_panel(axes[2], AUC_V,    r"AUC-V $\uparrow$",           (0.70, 0.885), "{:.3f}")
_panel(axes[3], AUC_J,    r"AUC-J $\uparrow$",           (0.74, 0.915), "{:.3f}")

fig.tight_layout(w_pad=2.5, rect=(0, 0.13, 1, 1))
fig.canvas.draw()
# 3 methods, 4 panels: spread the labels evenly (panel 1 centre, figure middle, panel 4
# centre) so they read as a distributed legend rather than bunched in the centre.
centers = [ax.get_position().x0 + ax.get_position().width / 2 for ax in axes]
label_x = [centers[0], (centers[1] + centers[2]) / 2, centers[-1]]
for xc, k in zip(label_x, ORDER):
    h = Patch(facecolor=FILL[k], edgecolor=EDGE[k], linewidth=1.2, label=LABELS[k])
    fig.legend(handles=[h], loc="center", bbox_to_anchor=(xc, 0.06),
               frameon=False, fontsize=20, handlelength=1.4, handletextpad=0.6)

out_pdf = "poster_assets/hand_metrics_barchart.pdf"
out_png = "poster_assets/hand_metrics_barchart.png"
fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
plt.close(fig)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
