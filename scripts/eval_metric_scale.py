"""Verify hand<->scene metric alignment via the L2 closed-form scale solve.

No training. For a handful of validation clips we run the model once to get the
scene depth (``gs_depth``) and the metric MANO hand joints, then solve the single
global scale ``s = median(hand_z / gs_depth_at_hand)`` (MetricScaleHead / L2) and
measure how well that one scale aligns the hand to the scene.

What the numbers mean
---------------------
* ``s`` should be a *stable* factor across clips (low coefficient of variation).
  If it swings wildly, the hand/scene mismatch is not a clean global scale.
* The hand-depth residual should drop from ``pre`` (raw gs_depth vs hand) to a
  few centimetres ``post`` (after scaling gs_depth by ``s``). Small + consistent
  ``post`` == the hand lands at the scene's metric depth.

Usage (on a gb10 node, inside venv_gb10):
    python -m scripts.eval_metric_scale \
        --config configs/ablation_hand_to_gs_injection.yaml \
        --checkpoint checkpoints/default/best_val_loss.pt \
        --num_clips 30

The checkpoint may be either a full-model dict (new ``save_checkpoint`` format,
``{"model_state_dict": ...}``) or a raw ``hand_head`` state_dict (older runs);
both are detected. The scene depth comes from the (pretrained) GS head, so a
well-trained hand head + the base reconstructor is a valid test.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.auxiliary_models.worldmirror.models.heads.metric_scale_head import (
    solve_metric_scale,
)
from scripts.hand_depth_anchor_loss import hand_depth_anchor_loss
from scripts.hand_vis_utils import MANOModel
from scripts.train_hand_head import (
    HOT3DHandDataset,
    build_views,
    compute_joints_from_batch,
    discover_sequences,
)


def _load_trained_weights(model: WorldMirror, path: str, device: str) -> None:
    """Load a trained checkpoint, auto-detecting full-model vs hand-head-only."""
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    is_full = any(k.startswith(("hand_head.", "gs_head.")) for k in sd.keys())
    if is_full:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded full-model checkpoint '{path}' "
              f"(missing={len(missing)}, unexpected={len(unexpected)}).")
    else:
        missing, unexpected = model.hand_head.load_state_dict(sd, strict=False)
        print(f"Loaded hand-head-only checkpoint '{path}' into model.hand_head "
              f"(missing={len(missing)}, unexpected={len(unexpected)}). "
              f"Scene depth comes from the pretrained GS head.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablation_hand_to_gs_injection.yaml")
    ap.add_argument("--checkpoint", required=True,
                    help="Trained checkpoint (best_val_loss.pt / latest.pt / a raw hand-head .pt)")
    ap.add_argument("--num_clips", type=int, default=30)
    ap.add_argument("--max_seqs", type=int, default=15,
                    help="Only build the dataset from this many sequences (faster startup).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg, model_cfg, vis_cfg = cfg["data"], cfg["model"], cfg.get("visualization", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_cfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true in the config (we need gs_depth).")

    # --- Model: same build as training, base ckpt, then the trained weights ---
    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    base_sd = base.get("state_dict", base.get("reconstructor", base))
    model.load_state_dict(base_sd, strict=False)
    _load_trained_weights(model, args.checkpoint, device)
    model.to(device).eval()
    model.gs_anchor_only = True  # fast path: produce gs_depth, skip prune/render

    mano_model = MANOModel(vis_cfg["mano_model_folder"])

    # --- Data: a handful of clips, reusing the training dataset ---
    num_frames = data_cfg["num_frames"]
    res = tuple(data_cfg["resolution"])
    seqs = discover_sequences(data_cfg["data_root"])
    if not seqs:
        raise SystemExit(f"No sequences under {data_cfg['data_root']}")
    random.seed(args.seed)
    random.shuffle(seqs)
    seqs = seqs[:args.max_seqs]
    print(f"Building dataset from {len(seqs)} sequence(s) (first run caches bboxes/joints).")
    dataset = HOT3DHandDataset(
        seqs, mano_model,
        num_frames=num_frames, res=res,
        clip_stride=data_cfg.get("clip_stride", num_frames),
        use_hand_crop=model_cfg.get("use_hand_crop", False),
        rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 2.0),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    rows = []
    print(f"\nEvaluating up to {args.num_clips} clips "
          f"({len(dataset.clips)} available) ...\n")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(rows) >= args.num_clips:
                break
            if "cam_intrinsics" not in batch:
                continue
            imgs = batch["img"].to(device)
            hb = batch["hand_bboxes"].to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].to(device) if "hand_valid" in batch else None
            views = build_views(imgs, num_frames, device, hb, hv)

            preds = model(views, is_inference=False, use_motion=False)
            gs_depth = preds.get("gs_depth")
            if gs_depth is None:
                raise SystemExit("Model produced no gs_depth — check enable_gs / the GS head.")
            pred_joints = compute_joints_from_batch(preds["hand_joints"], mano_model, device)
            has_hand = (
                hv.float() if hv is not None
                else torch.ones(pred_joints.shape[:3], device=device)
            )
            cam_intr = batch["cam_intrinsics"].to(device)

            s = solve_metric_scale(pred_joints, gs_depth, has_hand, cam_intr)
            _, info_pre = hand_depth_anchor_loss(pred_joints, gs_depth, has_hand, cam_intr)
            if info_pre["n_valid"] == 0:
                continue
            _, info_post = hand_depth_anchor_loss(pred_joints, gs_depth * s, has_hand, cam_intr)

            sf = float(s)
            rows.append((i, sf, info_pre["n_valid"],
                         info_pre["hand_depth_residual_m"],
                         info_post["hand_depth_residual_m"]))
            print(f"clip {i:4d} | s={sf:6.3f} | n_valid={info_pre['n_valid']:4d} | "
                  f"residual pre={info_pre['hand_depth_residual_m']*100:6.2f}cm "
                  f"-> post={info_post['hand_depth_residual_m']*100:6.2f}cm", flush=True)

    if not rows:
        raise SystemExit("No valid clips (no visible hands / no intrinsics).")

    s_vals = np.array([r[1] for r in rows])
    pre = np.array([r[3] for r in rows])
    post = np.array([r[4] for r in rows])
    med = float(np.median(s_vals))
    cv = float(s_vals.std() / max(abs(med), 1e-6))

    print("\n================ Metric-scale alignment summary ================")
    print(f"clips evaluated      : {len(rows)}")
    print(f"global scale s       : median={med:.3f}  mean={s_vals.mean():.3f}  "
          f"std={s_vals.std():.3f}  CV={cv:.1%}")
    print(f"residual @ hand (cm) : pre={pre.mean()*100:.2f}  ->  "
          f"post={post.mean()*100:.2f}  "
          f"(reduction {100*(1-post.mean()/max(pre.mean(),1e-9)):.0f}%)")
    print("\nReading the result:")
    print(" - CV(s) < ~15%        => the mismatch is a STABLE global scale (one s aligns them).")
    print(" - post << pre, ~few cm => applying s lands the hand at the scene's metric depth.")
    print(" - if s is railed at 0.1 or 10, the clamp hit — scene depth is far off-scale.")


if __name__ == "__main__":
    main()
