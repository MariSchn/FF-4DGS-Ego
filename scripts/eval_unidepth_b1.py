"""B1 stage 2 (x86 GPU / venv_unidepth): UniDepth-vs-hand-anchor depth error.

Reads the bundle produced by ``scripts/extract_b1_frames.py`` and answers
publication-plan E1: "why not just use a metric-depth foundation model?"

For every stored hand-joint location we sample UniDepth's *metric* depth map and
compare it to the trusted metric hand depth (``hand_z``). The same locations carry
our anchored scene depth (``gs_at_hand``), so the two error columns are exactly
comparable — both answer "how far off is this depth source AT the hand?"

E1 target: UniDepth hand-depth error > ~10 cm (>= 2x our ~4.5 cm) => a generic
metric-depth FM cannot replace the in-scene hand anchor.

Usage (x86 GPU node, inside venv_unidepth, repo on PYTHONPATH not required):
    python scripts/eval_unidepth_b1.py \
        --bundle outputs/b1_bundle.pt \
        --weights lpiccinelli/unidepth-v2-vitl14
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def sample_depth(depth_hw: torch.Tensor, grid_xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a [H, W] depth map at normalised [0,1] (x,y) locations.

    Args:
        depth_hw: [H, W] metric depth.
        grid_xy:  [..., 2] in [0, 1], x=width, y=height.
    Returns:
        [...] sampled depth (same leading shape as grid_xy minus the last dim).
    """
    lead = grid_xy.shape[:-1]
    g = (grid_xy.reshape(1, -1, 1, 2) * 2.0 - 1.0).to(depth_hw.dtype)
    d = F.grid_sample(depth_hw[None, None], g, mode="bilinear",
                      padding_mode="border", align_corners=False)
    return d.reshape(lead)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="b1_bundle.pt from extract_b1_frames.py")
    ap.add_argument("--weights", default="lpiccinelli/unidepth-v2-vitl14",
                    help="HF id or local dir for UniDepthV2.from_pretrained.")
    ap.add_argument("--out", default="outputs/b1_unidepth.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — UniDepth on CPU is slow but will still run.")

    from unidepth.models import UniDepthV2  # noqa: E402 (env-specific import)
    print(f"Loading UniDepthV2 from {args.weights} ...")
    model = UniDepthV2.from_pretrained(args.weights).to(device).eval()

    blob = torch.load(args.bundle, map_location="cpu")
    clips, meta = blob["clips"], blob.get("meta", {})
    print(f"Bundle: {len(clips)} clips, meta={meta}\n")

    fm_err, ours_err = [], []   # per-joint absolute depth error (metres)
    with torch.no_grad():
        for ci, clip in enumerate(clips):
            rgb = clip["rgb"]            # [S, 3, H, W] in [0, 1]
            K = clip["K"].to(device)    # [3, 3]
            grid = clip["grid_xy"]      # [S, 2, J, 2]
            hand_z = clip["hand_z"]     # [S, 2, J]
            gs = clip["gs_at_hand"]     # [S, 2, J]
            valid = clip["valid"]       # [S, 2, J] bool
            S = rgb.shape[0]
            for s in range(S):
                if not bool(valid[s].any()):
                    continue
                img255 = (rgb[s].clamp(0, 1) * 255.0).to(device)   # [3, H, W]
                pred = model.infer(img255, K)
                depth = pred["depth"].squeeze().to("cpu").float()   # [H, W]
                fm_s = sample_depth(depth, grid[s])                 # [2, J]
                m = valid[s]
                fm_err.append((fm_s[m] - hand_z[s][m]).abs())
                ours_err.append((gs[s][m] - hand_z[s][m]).abs())
            if (ci + 1) % 10 == 0:
                print(f"  ...{ci + 1}/{len(clips)} clips", flush=True)

    fm = torch.cat(fm_err).numpy()
    ours = torch.cat(ours_err).numpy()

    def stats(x: np.ndarray) -> dict:
        return {"mean_cm": float(x.mean() * 100), "median_cm": float(np.median(x) * 100),
                "p90_cm": float(np.percentile(x, 90) * 100), "n": int(x.size)}

    fm_s, ours_s = stats(fm), stats(ours)
    print("\n================ B1: UniDepth vs hand-anchor (depth @ hand) ================")
    print(f"joints evaluated      : {fm_s['n']}")
    print(f"UniDepth hand error   : mean={fm_s['mean_cm']:.2f}cm  median={fm_s['median_cm']:.2f}cm  p90={fm_s['p90_cm']:.2f}cm")
    print(f"OUR anchor hand error : mean={ours_s['mean_cm']:.2f}cm  median={ours_s['median_cm']:.2f}cm  p90={ours_s['p90_cm']:.2f}cm")
    ratio = fm_s["mean_cm"] / max(ours_s["mean_cm"], 1e-6)
    print(f"ratio (FM / ours)     : {ratio:.2f}x")
    verdict = ("PASS: FM >= 2x our error -> the hand anchor is the better metric source"
               if ratio >= 2.0 else
               "WEAK: FM competitive at the hand -> reframe as complementary (see E1 fallback)")
    print(verdict)

    import json
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"unidepth": fm_s, "ours": ours_s, "ratio": ratio,
                   "weights": args.weights, "meta": meta}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
