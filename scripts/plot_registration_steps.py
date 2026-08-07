"""Cyrus's request (2026-08-06): "visual intermediate results for the registration steps".

Fig. 1's registration block names three stages and shows none of them, so a reader takes the scale
solve on faith. This renders each stage from a REAL forward pass, using the arrays
scripts/dump_registration_steps.py pulls out of eval_world_space's own predict_clip via steps_out.

The three steps are the ones Fig. 1's caption already names, so the panel labels match the paper
rather than inventing a second vocabulary. Step 1 does two things and therefore gets two panels:

  R1a  project the metric hand joints     where the joints land in the frame
  R1b  read d_scene at those pixels       the predicted scene depth, same pixels
  R2   solve one scale                    the z_hand / d_scene population and the median taken
  R3   scale the translation              the up-to-scale camera track vs the same track times s

WHY IT IS FED BY THE EVAL. An earlier standalone dumper re-implemented the load-and-forward path,
silently diverged (no bfloat16 autocast, no cond_flags), and emitted a CONSTANT hand depth of
-0.0205 m on every sequence, i.e. every hand behind the camera. A figure whose job is to show what
the eval does has to come from the eval.

COORDINATE NOTE, easy to get wrong. gs_depth is stored 90 degrees rotated relative to the image:
project_joints_to_norm_pixels emits [u, v] = [(W-1)-row, col] / W, so a joint sits at (x=u, y=v) in
the depth STORE and at (x=v, y=(W-1)-u) in the frame.

We therefore un-rotate the depth map for display with rot90(k=1) and draw the joints at the SAME
pixel coordinates in R1a and R1b. This is not cosmetic: a reader comparing two panels whose markers
sit in different places cannot see that it is one projection feeding one lookup. It is also exactly
lossless -- verified numerically, rot90(depth, k=1)[py, px] reproduces the eval's own store-frame
sample depth[y_d, x_d] with max abs difference 0.0 over the valid joints.

    python -m scripts.plot_registration_steps --dump regsteps_seq1.pt --out fig_registration.pdf
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Match the poster / paper: Latin Modern serif, no bold-by-default.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
})

INK = "#1a1a1a"
ACCENT = "#c0392b"      # the solved scale / selected quantity
COOL = "#2c6fbb"        # scene-side quantities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--frame", type=int, default=None, help="which frame to show in R1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    p = d["panel"]
    rgb = p["rgb"].float() / 255.0            # [S,3,H,W]
    depth = p["gs_depth"].float()             # [S,Hd,Wd]
    grid = p["grid_xy"].float()               # [S,2,J,2] normalised (x=u width, y=v height)
    hz = p["hand_z"].float()                  # [S,2,J]
    sc = p["scene_at_hand"].float()           # [S,2,J]
    val = p["valid"].bool()
    c2w = p["c2w"].float()                    # [S,4,4] UP TO SCALE
    s = float(p["s"])
    S, _, H, W = rgb.shape

    # Pick the frame with the most valid joints so the illustration is not a near-empty one.
    f = args.frame if args.frame is not None else int(val.sum(dim=(1, 2)).argmax())

    fig = plt.figure(figsize=(11.0, 3.15))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.15, 1.15], wspace=0.34,
                          left=0.035, right=0.985, top=0.84, bottom=0.17)

    # One set of pixel coordinates, used by BOTH R1 panels. The frame is plain image layout
    # (col = v*W, row = (W-1) - u*W); the depth store is un-rotated below to match it.
    #
    # Split by HAND SLOT, because that is what the panel turns out to be evidence of (task #70).
    # The detector fills slot 1 on this sequence and never fills slot 0 (the box store has
    # valid[:, 0].sum() == 0 over all 300 frames), yet the model emits a default MANO into the
    # empty slot and the scale solve accepts its joints: 96 per clip at median ratio -0.019.
    v_ = val[f]
    hand_of = torch.arange(val.shape[1]).view(-1, 1).expand_as(val[f])[v_].numpy()
    g = grid[f][v_]                                        # [n,2] = (u, v) normalised
    px, py = g[:, 1].numpy() * W, (W - 1) - g[:, 0].numpy() * W
    slot0 = hand_of == 0

    def draw_joints(ax, legend=False):
        if not v_.any():
            return
        ax.scatter(px[~slot0], py[~slot0], s=7, c=ACCENT,
                   edgecolors="white", linewidths=0.3, zorder=3, label="detected hand")
        if slot0.any():
            ax.scatter(px[slot0], py[slot0], s=26, marker="x", c="#f0b429",
                       linewidths=1.1, zorder=4, label="undetected slot")
        if legend:
            # A frameless white legend sits on a bright wall in this scene and disappears. Give it
            # a translucent plate and dark text so it survives whatever frame is chosen.
            lg = ax.legend(fontsize=6, loc="upper right", frameon=True, framealpha=0.78,
                           facecolor="white", edgecolor="none", labelcolor=INK,
                           handletextpad=0.4, borderpad=0.3)
            lg.get_frame().set_linewidth(0.0)

    # ---------------- R1a: the frame, with the projected joints ----------------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb[f].permute(1, 2, 0).numpy())
    draw_joints(ax, legend=True)
    ax.set_title(r"$\mathbf{R1a}$  project joints", pad=6)
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------- R1b: the predicted scene depth, SAME pixels ----------------
    ax = fig.add_subplot(gs[0, 1])
    # rot90(k=1) maps the store into image layout: store[y_d, x_d] -> disp[(W-1)-x_d, y_d].
    # Verified lossless against the eval's own samples (see the coordinate note above).
    im = ax.imshow(np.rot90(depth[f].numpy(), k=1), cmap="viridis")
    draw_joints(ax)
    ax.set_title(r"$\mathbf{R1b}$  read $d_{\mathrm{scene}}$", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    # ---------------- R2: the ratio population and the median that is taken -------------
    ax = fig.add_subplot(gs[0, 2])
    r_all = (hz[val] / sc[val]).numpy()
    hand_all = torch.arange(val.shape[1]).view(1, -1, 1).expand_as(val)[val].numpy()
    fin = np.isfinite(r_all)
    r_all, hand_all = r_all[fin], hand_all[fin]
    r_det, r_und = r_all[hand_all != 0], r_all[hand_all == 0]

    lo, hi = np.percentile(r_all, 1), np.percentile(r_all, 99)
    bins = np.linspace(min(lo, 0.0), hi, 49)
    # Stacked, so the reader sees WHICH population the left lobe is rather than being told.
    ax.hist([r_det, r_und], bins=bins, stacked=True, color=[COOL, "#f0b429"],
            edgecolor="none", label=["detected hand", "undetected slot"])
    ax.axvline(s, color=ACCENT, lw=1.6, label=rf"$s=\mathrm{{med}}={s:.3f}$")
    if r_det.size:
        ax.axvline(float(np.median(r_det)), color=ACCENT, lw=1.4, ls="--",
                   label=rf"detected only $={np.median(r_det):.3f}$")
    ax.axvline(0.0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel(r"$z_{\mathrm{hand}} / d_{\mathrm{scene}}$")
    ax.set_ylabel("joint samples")
    ax.set_title(r"$\mathbf{R2}$  solve one scale", pad=6)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    frac_neg = float((r_all <= 0).mean())
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------- R3: the camera track, up-to-scale vs metric ----------------
    ax = fig.add_subplot(gs[0, 3])
    t = c2w[:, :3, 3].numpy()
    t = t - t[0]
    ts = t * s
    ax.plot(t[:, 0], t[:, 2], "-o", ms=2.6, lw=1.0, color=COOL, label="up to scale")
    ax.plot(ts[:, 0], ts[:, 2], "-o", ms=2.6, lw=1.0, color=ACCENT,
            label=rf"$\times\,s={s:.3f}$ (metric)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(r"$\mathbf{R3}$  scale the translation", pad=6)
    ax.legend(fontsize=7, frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.savefig(args.out, bbox_inches="tight", dpi=220)
    print(f"seq {d['seq'].split('/')[-1]}  frame {f}  s={s:.4f}  "
          f"valid={int(val.sum())}  frac z<=0 {frac_neg:.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
