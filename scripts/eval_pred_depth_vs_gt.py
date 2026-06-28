"""(a) Metric-depth quality of the predicted point cloud vs dense GT depth (Cyrus check a).

We have only ever scored the predicted scene depth AT the hand (hand_depth_anchor_loss) or on
HOT3D object meshes (object_depth_loss / B2). Neither answers "how good is our predicted METRIC
depth as a depth map vs GT depth." HOI4D ships a dense, RGB-aligned 16-bit depth sensor map
(``raw_depth/``), so it is the right place to measure standard monocular-metric-depth numbers.

Per clip: forward the model -> ``gs_depth``; sample it at the valid GT depth pixels with the same
resolution-independent normalized grid as ``object_depth_loss``; then report, over all valid pixels:

  * ABSOLUTE (no scale): AbsRel, RMSE (m), delta<1.25/1.25^2/1.25^3, mean |err| (m). This is the
    real metric-depth quality if the model's gs_depth is already metric (HOI4D-depth-trained ckpt).
  * SCALE-INVARIANT: re-fit one median scale s* = median(gt / gs) per frame, then AbsRel. Decouples
    depth SHAPE from absolute scale.
  * The per-frame s* distribution (median, CV). A tight s* means the depth is metric up to a single
    stable scale (corroborates the scale-vs-GT check b); a wild s* means the scale itself is unstable.

Usage (gb10 / venv_gb10):
    python -m scripts.eval_pred_depth_vs_gt \
        --config configs/exp_hoi4d_depth.yaml \
        --checkpoint /work/scratch/dmonopoli/checkpoints/hoi4d_depth/best.pt \
        --data_root /work/scratch/dmonopoli/hoi4d --max_seqs 6 --num_clips 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    sample_depth_at_joints,
)
from scripts.hoi4d_depth_dataset import HOI4DDepthDataset, discover_hoi4d_seqs
from scripts.train_hand_head import build_views


def _load_trained(model, path, device):
    """Load a trained checkpoint over the base (full-model dict or raw state_dict)."""
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded trained ckpt '{path}' (missing={len(missing)}, unexpected={len(unexpected)}).")


def _save_panel(path, rgb, pred_d, gt_d, gt_m) -> None:
    """RGB | pred depth | GT depth | abs-error panel for one frame (all [res,res]). cv2 only
    (matplotlib's C-extension is broken in the aarch64 venv)."""
    import cv2

    def _label(img, txt):
        cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _colorize(d, vmax, cmap, mask=None):
        x = (np.clip(d / max(vmax, 1e-6), 0, 1) * 255).astype(np.uint8)
        c = cv2.applyColorMap(x, cmap)
        if mask is not None:
            c[~mask] = 0
        return c

    vmax = float(np.percentile(gt_d[gt_m], 95)) if bool(gt_m.any()) else 3.0
    rgb_u = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])  # RGB->BGR
    pred_c = _colorize(pred_d, vmax, cv2.COLORMAP_JET)
    gt_c = _colorize(np.where(gt_m, gt_d, 0.0), vmax, cv2.COLORMAP_JET, mask=gt_m)
    err = np.where(gt_m, np.abs(pred_d - gt_d), 0.0)
    err_c = _colorize(err, 0.3, cv2.COLORMAP_HOT, mask=gt_m)
    panel = cv2.hconcat([_label(rgb_u, "RGB"), _label(pred_c, "pred (m)"),
                         _label(gt_c, "GT (m)"), _label(err_c, "|err| 0-0.3m")])
    cv2.imwrite(path, panel)


def _depth_stats(gs: torch.Tensor, gt: torch.Tensor) -> dict:
    """Per-pixel depth metrics for one frame's valid pixels. gs/gt: [N] metres (positive)."""
    err = (gs - gt).abs()
    abs_rel = (err / gt).mean().item()
    rmse = torch.sqrt(((gs - gt) ** 2).mean()).item()
    ratio = torch.maximum(gs / gt, gt / gs)
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()
    s_star = float(torch.median(gt / gs))                       # per-frame optimal scale
    gs_si = gs * s_star
    abs_rel_si = ((gs_si - gt).abs() / gt).mean().item()
    return {"abs_rel": abs_rel, "rmse": rmse, "d1": d1, "d2": d2, "d3": d3,
            "mae_m": err.mean().item(), "s_star": s_star, "abs_rel_si": abs_rel_si}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="trained ckpt over the base (else base only)")
    ap.add_argument("--data_root", required=True, help="HOI4D root with <seq>/images + <seq>/raw_depth")
    ap.add_argument("--max_seqs", type=int, default=6)
    ap.add_argument("--num_clips", type=int, default=40)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=10.0)
    ap.add_argument("--viz_dir", default="", help="if set, save quick cv2 RGB|pred|GT|err panels here")
    ap.add_argument("--viz_n", type=int, default=4, help="number of quick panels to save")
    ap.add_argument("--fig_npz", default="", help="if set, save raw arrays for N diverse scenes here "
                    "(for a polished figure rendered off-node)")
    ap.add_argument("--fig_n", type=int, default=4, help="number of diverse scenes for --fig_npz")
    args = ap.parse_args()
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone + GS head, no hand path (we only need gs_depth; mirrors hoi4d_backbone_smoke).
    mcfg = dict(cfg["model"])
    mcfg["enable_hand"] = False
    mcfg["use_hand_crop"] = False
    mcfg["hand_to_gs_injection"] = {"enabled": False}
    if not mcfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true (need gs_depth).")
    model = WorldMirror(**{k: v for k, v in mcfg.items() if k != "checkpoint"})
    base = torch.load(mcfg["checkpoint"], map_location=device)
    sd = base.get("state_dict", base.get("reconstructor", base)) if isinstance(base, dict) else base
    model.load_state_dict(sd, strict=False)
    if args.checkpoint:
        _load_trained(model, args.checkpoint, device)
    model.to(device).eval()

    res = int(cfg["data"]["resolution"][0])
    seqs = discover_hoi4d_seqs(args.data_root)[: args.max_seqs]
    if not seqs:
        raise SystemExit(f"No HOI4D seqs (images/+raw_depth/) under {args.data_root}")
    ds = HOI4DDepthDataset(seqs, num_frames=args.num_frames, res=res, depth_res=res,
                           depth_min=args.depth_min, depth_max=args.depth_max)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Eval over {len(seqs)} seq(s), up to {args.num_clips} clips ({len(ds)} available).")

    rows = []
    saved = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(rows) >= args.num_clips:
                break
            imgs = batch["img"].to(device)                          # [1,S,3,res,res]
            gt_d = batch["gt_obj_depth"][0].to(device)              # [S,res,res] m
            gt_m = batch["gt_obj_mask"][0].to(device)               # [S,res,res] bool
            views = build_views(imgs, args.num_frames, device, None, None)
            preds = model(views, is_inference=False, use_motion=False)
            gsd = preds.get("gs_depth")
            if gsd is None:
                raise SystemExit("Model returned no gs_depth.")
            for s in range(gt_d.shape[0]):
                m = gt_m[s]
                if not bool(m.any()):
                    continue
                ys, xs = torch.where(m)
                R = gt_d.shape[-1]
                grid = torch.stack([xs.float() / R, ys.float() / R], -1).view(1, 1, 1, -1, 2).to(device)
                gs_at, in_fr = sample_depth_at_joints(gsd[0:1, s:s + 1], grid)
                gs_at, in_fr = gs_at.reshape(-1), in_fr.reshape(-1)
                gt_at = gt_d[s][ys, xs]
                keep = in_fr & (gs_at > args.depth_min) & (gs_at < args.depth_max) & (gt_at > args.depth_min)
                if int(keep.sum()) < 50:
                    continue
                rows.append(_depth_stats(gs_at[keep].float(), gt_at[keep].float()))
                if args.viz_dir and saved < args.viz_n:
                    try:
                        import cv2
                        pm = gsd[0, s].squeeze().float().cpu().numpy()
                        if pm.shape != (R, R):
                            pm = cv2.resize(pm, (R, R), interpolation=cv2.INTER_LINEAR)
                        rgb = imgs[0, s].permute(1, 2, 0).cpu().numpy()
                        _save_panel(os.path.join(args.viz_dir, f"depth_panel_{saved:02d}.png"),
                                    rgb, pm, gt_d[s].cpu().numpy(), gt_m[s].cpu().numpy().astype(bool))
                        saved += 1
                    except Exception as e:                       # viz must never kill the metrics
                        print(f"[viz warn] panel {saved} failed: {type(e).__name__}: {e}", flush=True)
                        args.viz_dir = ""                        # stop trying after the first failure
            if len(rows) and len(rows) % 10 == 0:
                print(f"  ...{len(rows)} frames scored", flush=True)

    if not rows:
        raise SystemExit("No valid depth frames scored.")

    def _m(k):
        return float(np.mean([r[k] for r in rows]))

    s_star = np.array([r["s_star"] for r in rows])
    cv = float(s_star.std() / max(abs(np.median(s_star)), 1e-6))
    print("\n================ Predicted metric depth vs dense GT (HOI4D raw_depth) ================")
    print(f"frames scored        : {len(rows)}")
    print(f"ABSOLUTE (model scale): AbsRel={_m('abs_rel'):.3f}  RMSE={_m('rmse'):.3f} m  "
          f"MAE={_m('mae_m')*100:.1f} cm  d<1.25={_m('d1'):.3f}  d<1.25^2={_m('d2'):.3f}  d<1.25^3={_m('d3'):.3f}")
    print(f"SCALE-INVARIANT       : AbsRel={_m('abs_rel_si'):.3f}  (shape only, per-frame median scale)")
    print(f"per-frame scale s*    : median={np.median(s_star):.3f}  mean={s_star.mean():.3f}  "
          f"std={s_star.std():.3f}  CV={cv:.1%}")
    print("\nReading the result:")
    print(" - low AbsRel + high d<1.25 + MAE ~ a few cm => predicted metric depth is genuinely good.")
    print(" - SCALE-INVARIANT AbsRel << ABSOLUTE AbsRel => depth SHAPE is fine, the SCALE is off (b).")
    print(" - CV(s*) large => the per-frame scale itself is unstable (drives world drift).")

    # Figure data: raw arrays for a few DIVERSE scenes (one mid-frame from clips spread across the
    # whole dataset, i.e. different sequences / object classes), saved compressed for an off-node
    # polished render (the cluster matplotlib is broken; we render on the Mac in the poster style).
    if args.fig_npz:
        import cv2
        targets = sorted(set(int(x) for x in np.linspace(0, len(ds) - 1, args.fig_n)))
        rgbs, preds, gts, masks = [], [], [], []
        with torch.no_grad():
            for ti in targets:
                b = ds[ti]
                im = b["img"].unsqueeze(0).to(device)
                gsd = model(build_views(im, args.num_frames, device, None, None),
                            is_inference=False, use_motion=False).get("gs_depth")
                sidx = args.num_frames // 2
                R = b["gt_obj_depth"].shape[-1]
                pm = gsd[0, sidx].squeeze().float().cpu().numpy()
                if pm.shape != (R, R):
                    pm = cv2.resize(pm, (R, R), interpolation=cv2.INTER_LINEAR)
                rgbs.append(im[0, sidx].permute(1, 2, 0).cpu().numpy().astype(np.float32))
                preds.append(pm.astype(np.float32))
                gts.append(b["gt_obj_depth"][sidx].numpy().astype(np.float32))
                masks.append(b["gt_obj_mask"][sidx].numpy().astype(bool))
        np.savez_compressed(args.fig_npz, rgb=np.stack(rgbs), pred=np.stack(preds),
                            gt=np.stack(gts), mask=np.stack(masks),
                            absrel=float(_m("abs_rel")), mae_cm=float(_m("mae_m") * 100),
                            d1=float(_m("d1")))
        print(f"[fig] wrote {len(targets)} diverse scenes -> {args.fig_npz}", flush=True)


if __name__ == "__main__":
    main()
