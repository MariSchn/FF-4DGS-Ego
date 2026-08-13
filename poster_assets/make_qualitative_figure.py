"""Compose the qualitative results panel for the poster.

This does NOT invent results — it tiles YOUR real frames (exported from the
W&B `media/` panels or rendered with scripts/eval_gs_head.py / hand_vis_utils.py)
into a labeled, publication-quality grid and writes a vector PDF + PNG preview.

Layout: rows = example sequences, columns = methods.
Drop your frames into poster_assets/qual/ named:

    <method_key>_<example_idx>.png

e.g.  gt_0.png  baseline_0.png  ours_0.png   (example 0, three methods)
      gt_1.png  baseline_1.png  ours_1.png   (example 1) ...

Any file that is missing is rendered as a labeled placeholder so you can lay the
poster out before the final frames are ready.

    python poster_assets/make_qualitative_figure.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

QUAL_DIR = "poster_assets/qual"

# Columns = methods.  (key used in the filename, label shown on the poster)
COLUMNS = [
    ("gt",       "Reference (GT)"),
    ("baseline", "NeoVerse (trained)"),
    ("ours",     "Ours (+ Hand Inj.)"),
]

# Rows = example sequences. Left-hand label per row (set to "" to hide).
ROWS = [
    "Pour",
    "Assemble",
    "Hand-off",
]

# Optional: draw a red dashed box on the hand region of each cell to direct the
# eye. Normalized [x1, y1, x2, y2] in [0,1]; set to None to disable per row.
HAND_BOXES = [None, None, None]

ETH_BLUE = (33 / 255, 92 / 255, 175 / 255)
ETH_RED  = (183 / 255, 53 / 255, 45 / 255)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})


def load_or_placeholder(path, label):
    """Return an HxWx3 uint8 array; a labeled gray tile if the file is absent."""
    if os.path.exists(path):
        return np.asarray(Image.open(path).convert("RGB")), True
    tile = np.full((224, 224, 3), 235, dtype=np.uint8)
    return tile, False


n_rows, n_cols = len(ROWS), len(COLUMNS)
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.0 * n_cols, 3.0 * n_rows),
    gridspec_kw={"wspace": 0.04, "hspace": 0.06},
)
axes = np.atleast_2d(axes)

n_missing = 0
for r in range(n_rows):
    for c, (key, col_label) in enumerate(COLUMNS):
        ax = axes[r, c]
        path = os.path.join(QUAL_DIR, f"{key}_{r}.png")
        img, found = load_or_placeholder(path, col_label)
        n_missing += (not found)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("0.7"); s.set_linewidth(1.0)

        if not found:
            ax.text(0.5, 0.5, f"{key}_{r}.png\n(drop frame here)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="0.55")

        # Highlight the "ours" column border in ETH blue.
        if key == "ours":
            for s in ax.spines.values():
                s.set_edgecolor(ETH_BLUE); s.set_linewidth(2.4)

        # Optional hand-region box.
        box = HAND_BOXES[r] if r < len(HAND_BOXES) else None
        if box is not None:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
                                    fill=False, edgecolor=ETH_RED, lw=1.8, ls="--"))

        if r == 0:
            ax.set_title(col_label, fontsize=14, fontweight="bold", pad=8,
                         color=ETH_BLUE if key == "ours" else "black")
        if c == 0 and ROWS[r]:
            ax.set_ylabel(ROWS[r], fontsize=13, fontweight="bold", rotation=90,
                          labelpad=10)

fig.savefig("poster_assets/qualitative_results.pdf", bbox_inches="tight")
fig.savefig("poster_assets/qualitative_results.png", dpi=200, bbox_inches="tight")
print("wrote poster_assets/qualitative_results.pdf")
print("wrote poster_assets/qualitative_results.png")
if n_missing:
    print(f"\n[!] {n_missing}/{n_rows * n_cols} frames are placeholders.")
    print(f"    Add real PNGs to {QUAL_DIR}/ (see COLUMNS/ROWS keys) and re-run.")
