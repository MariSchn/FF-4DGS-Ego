"""HOI4D backbone-compatibility smoke (the highest-risk, cheapest check).

Before investing in a full HOI4D port, answer ONE question: does our frozen
NeoVerse/WorldMirror backbone produce sane Gaussians / scene depth on HOI4D
imagery (a different camera + scenes than HOT3D Aria)? If gs_depth comes out
finite, positive, and in a plausible metric range (HOI4D depth is ~0.01-10 m),
the port is worth it. If it's degenerate, we rethink before porting.

This needs NO HOI4D annotations — just RGB frames. Point --rgb at a HOI4D
``align_rgb/image.mp4`` (or a folder of decoded ``%05d.jpg`` frames).

Usage (gb10 / venv_gb10):
    python -m scripts.hoi4d_backbone_smoke \
        --config configs/exp_p3_gtdepth_unfreeze.yaml \
        --rgb /work/scratch/dmonopoli/hoi4d/<seq>/align_rgb/image.mp4 \
        --frames 16 --out /work/scratch/dmonopoli/hoi4d_smoke
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from scripts.train_hand_head import build_views


# ------------------------------------------------------------------ RGB I/O
def _read_frames(rgb_path: str, n: int, res: int = 224) -> torch.Tensor:
    """Read ``n`` evenly-spaced RGB frames -> [1, n, 3, res, res] in [0,1].

    Accepts a video file (cv2.VideoCapture) or a directory of image frames.
    Center-crops to square then resizes, matching the model's square-pinhole
    expectation (HOI4D RGB is 1920x1080 -> crop to 1080x1080 -> res).
    """
    import cv2

    frames_bgr = []
    if os.path.isdir(rgb_path):
        exts = (".jpg", ".jpeg", ".png")
        files = sorted(f for f in os.listdir(rgb_path) if f.lower().endswith(exts))
        if not files:
            raise SystemExit(f"No image frames in {rgb_path}")
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
        for i in idx:
            img = cv2.imread(os.path.join(rgb_path, files[i]))
            if img is None:
                raise SystemExit(f"Failed to read {files[i]}")
            frames_bgr.append(img)
    else:
        cap = cv2.VideoCapture(rgb_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise SystemExit(f"Could not read frames from {rgb_path}")
        idx = np.linspace(0, total - 1, n).round().astype(int)
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"Failed to read frame {i} of {rgb_path}")
            frames_bgr.append(img)
        cap.release()

    out = []
    for img in frames_bgr:
        h, w = img.shape[:2]
        s = min(h, w)                                  # center square-crop
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
        img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(torch.from_numpy(img).permute(2, 0, 1))
    return torch.stack(out).unsqueeze(0)               # [1, n, 3, res, res]


# ------------------------------------------------------------------ report
def _report(name: str, t: torch.Tensor) -> None:
    t = t.float().reshape(-1)
    finite = torch.isfinite(t)
    f = t[finite]
    print(f"  {name:14s} shape={tuple(t.shape)} finite={100*finite.float().mean():.1f}% "
          f"min={f.min().item() if f.numel() else float('nan'):.4f} "
          f"max={f.max().item() if f.numel() else float('nan'):.4f} "
          f"mean={f.mean().item() if f.numel() else float('nan'):.4f} "
          f"median={f.median().item() if f.numel() else float('nan'):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_p3_gtdepth_unfreeze.yaml")
    ap.add_argument("--rgb", required=True, help="HOI4D align_rgb/image.mp4 or a frame dir")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", default=None, help="dir to save a depth-map PNG for eyeballing")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Isolate the backbone + GS head (we load the BASE reconstructor, no trained
    # hand model): disable the hand head + hand->GS injection so this is a clean
    # "does the frozen backbone give sane gs_depth on HOI4D?" test, and so the
    # forward never needs hand bboxes we don't have.
    model_cfg = dict(cfg["model"])
    model_cfg["enable_hand"] = False
    model_cfg["use_hand_crop"] = False
    model_cfg["hand_to_gs_injection"] = {"enabled": False}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the model exactly as eval_scene_metric_gt does (gs enabled, frozen).
    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    sd = base.get("state_dict", base.get("reconstructor", base)) if isinstance(base, dict) else base
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded base ckpt (missing={len(missing)}, unexpected={len(unexpected)}).")
    model.to(device).eval()
    if not model_cfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true to test scene depth.")

    res = int(cfg["data"]["resolution"][0])
    imgs = _read_frames(args.rgb, args.frames, res).to(device)
    print(f"HOI4D RGB: {tuple(imgs.shape)} from {args.rgb}")

    with torch.no_grad():
        views = build_views(imgs, args.frames, device, None, None)
        preds = model(views, is_inference=False, use_motion=False)

    print("\n========== HOI4D backbone-compat smoke ==========")
    gsd = preds.get("gs_depth")
    if gsd is None:
        raise SystemExit("Model returned no gs_depth — backbone/gs head misconfigured.")
    _report("gs_depth", gsd)
    for k in ("gs_depth_conf", "camera_params", "camera_intrs"):
        if preds.get(k) is not None:
            _report(k, preds[k])

    g = gsd.float()
    g = g[torch.isfinite(g)]
    in_range = ((g > 0.01) & (g < 10.0)).float().mean().item() if g.numel() else 0.0
    print(f"\n  gs_depth in HOI4D metric range (0.01-10 m): {100*in_range:.1f}%")
    print("  READ: finite, positive, and a large fraction in 0.01-10 m -> backbone is "
          "sane on HOI4D; the port is worth it. Degenerate/NaN/flat -> rethink first.")

    if args.out:
        try:
            os.makedirs(args.out, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            d = gsd[0].reshape(args.frames, *gsd.shape[-2:]) if gsd.dim() == 5 else gsd[0]
            mid = d[args.frames // 2].cpu().numpy()
            plt.figure(figsize=(5, 5)); plt.imshow(mid, cmap="turbo"); plt.colorbar()
            plt.title("HOI4D gs_depth (mid frame, m)"); plt.tight_layout()
            p = os.path.join(args.out, "hoi4d_gs_depth_mid.png")
            plt.savefig(p, dpi=120); print(f"\nSaved depth viz -> {p}")
        except Exception as e:
            print(f"\n[warn] could not save viz: {e}")


if __name__ == "__main__":
    main()
