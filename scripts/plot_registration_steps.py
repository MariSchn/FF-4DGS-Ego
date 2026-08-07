"""Cyrus's request (2026-08-06): "visual intermediate results for the registration steps".

Fig. 1's registration block names three stages and shows none of them, so a reader takes the scale
solve on faith. This renders each stage from a REAL forward pass, using the arrays
scripts/dump_registration_steps.py pulls out of eval_world_space's own predict_clip via steps_out.

  R1  sample scene depth      the frame, the predicted scene depth, and where the joints land
  R2  solve one scale         the full z_hand / d_scene population and the median that is taken
  R3  scale the translation   the up-to-scale camera track vs the same track scaled by s

WHY IT IS FED BY THE EVAL. An earlier standalone dumper re-implemented the load-and-forward path,
silently diverged (no bfloat16 autocast, no cond_flags), and emitted a CONSTANT hand depth of
-0.0205 m on every sequence, i.e. every hand behind the camera. A figure whose job is to show what
the eval does has to come from the eval.

COORDINATE NOTE, easy to get wrong. gs_depth is stored 90 degrees rotated relative to the image:
project_joints_to_norm_pixels emits [u, v] = [(W-1)-row, col] / W. So the SAME joint is drawn at
(x=u, y=v) on the depth map and at (x=v, y=(W-1)-u) on the RGB frame. Plotting both with one
convention puts the markers in the wrong place on one of them.

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

    # ---------------- R1a: the frame, with the projected joints ----------------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb[f].permute(1, 2, 0).numpy())
    v_ = val[f]
    if v_.any():
        g = grid[f][v_]                                    # [n,2] = (u, v) normalised
        behind = (hz[f][v_] <= 0).numpy()                  # task #63: joints behind the camera
        # RGB is in plain image layout: col = v*W, row = (W-1) - u*W
        px, py = g[:, 1].numpy() * W, (W - 1) - g[:, 0].numpy() * W
        ax.scatter(px[~behind], py[~behind], s=7, c=ACCENT,
                   edgecolors="white", linewidths=0.3, zorder=3, label="in front")
        if behind.any():
            ax.scatter(px[behind], py[behind], s=9, facecolors="none", edgecolors="#f0b429",
                       linewidths=0.9, zorder=4, label=r"$z\leq 0$")
        ax.legend(fontsize=6, frameon=False, loc="upper right", labelcolor="white")
    ax.set_title(r"$\mathbf{R1}$  project joints", pad=6)
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------- R1b: the predicted scene depth, same joints ----------------
    ax = fig.add_subplot(gs[0, 1])
    dm = depth[f].numpy()
    im = ax.imshow(dm, cmap="viridis")
    if v_.any():
        g = grid[f][v_]
        # depth map is the ROTATED store: x = u, y = v
        ax.scatter(g[:, 0] * dm.shape[1], g[:, 1] * dm.shape[0], s=7, c=ACCENT,
                   edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_title(r"$\mathbf{R1}$  read $d_{\mathrm{scene}}$", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    # ---------------- R2: the ratio population and the median that is taken -------------
    ax = fig.add_subplot(gs[0, 2])
    r = (hz[val] / sc[val]).numpy()
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    ax.hist(r, bins=48, range=(min(lo, 0.0), hi), color=COOL, alpha=0.85, edgecolor="none")
    ax.axvline(s, color=ACCENT, lw=1.6, label=rf"$s=\mathrm{{med}}={s:.3f}$")
    frac_neg = float((r <= 0).mean())
    ax.axvline(0.0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel(r"$z_{\mathrm{hand}} / d_{\mathrm{scene}}$")
    ax.set_ylabel("joint samples")
    ax.set_title(r"$\mathbf{R2}$  solve one scale", pad=6)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    # The behind-camera tail is a real, measured property, not a rendering artefact: state it.
    # NOT r"\%": matplotlib renders the backslash literally outside mathtext, which is how "16\%"
    # once shipped on the poster. Keep the percent sign outside the math span.
    ax.text(0.02, 0.95, f"$z\\leq 0$: {100*frac_neg:.0f}%", transform=ax.transAxes,
            fontsize=7, va="top", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------- R3: the camera track, up-to-scale vs metric ----------------
    ax = fig.add_subplot(gs[0, 3])
    t = c2w[:, :3, 3].numpy()
    t = t - t[0]
    ts = t * s
    ax.plot(t[:, 0], t[:, 2], "-o", ms=2.6, lw=1.0, color=COOL, label="up to scale")
    ax.plot(ts[:, 0], ts[:, 2], "-o", ms=2.6, lw=1.0, color=ACCENT, label=rf"$\times\,s$ (metric)")
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
