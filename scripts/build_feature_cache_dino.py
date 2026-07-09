"""Precompute frozen DINOv2 patch tokens for the backbone-swap ablation.

Same keying and consumption path as scripts/build_feature_cache.py (per-clip
``<seq>_<frame_offset>.pt``, bf16 ``[S, P, C]``, cached training via
``forward_hand_cached`` with ``patch_start_idx=0``), but the tokens come from a
frozen DINOv2 ViT-L/14 instead of the reconstruction backbone. At 224x224 both
produce the same 16x16=256 patch grid; only the channel width differs
(DINOv2-L: C=1024 vs WorldMirror: C=2048), so the ablation config must set
``hamer_head_kwargs.context_dim: 1024``.

Purpose: if the head trained on DINOv2 tokens matches the head trained on
reconstruction-backbone tokens, the "feedforward reconstruction features encode
metric hand depth" claim dies; if it doesn't, the ablation is the paper's
novelty defense. The random-init arm lives in build_feature_cache.py
(--random_init).

The dataset emits raw [0,1] RGB (TVF.to_tensor); DINOv2 expects ImageNet
normalization, applied here.

Usage (gb10, venv_gb10; compute nodes have internet for the torch.hub pull):
    python -m scripts.build_feature_cache_dino --config <winner cfg> \
        --data_root /home/dmonopoli/hoi4d_train --out /tmp/featcache_dino/train \
        --clip_stride 16
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

from scripts.train_hand_head import HOT3DHandDataset, discover_sequences

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip_stride", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_seqs", type=int, default=None)
    ap.add_argument("--dino_model", default="dinov2_vitl14")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    device = "cuda"

    model = torch.hub.load("facebookresearch/dinov2", args.dino_model)
    model.to(device).eval()

    from scripts.hand_vis_utils import MANOModel
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])

    seqs = discover_sequences(args.data_root)
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    ds = HOT3DHandDataset(
        seqs, mano,
        num_frames=data_cfg.get("num_frames", 16),
        res=tuple(data_cfg.get("resolution", [224, 224])),
        clip_stride=args.clip_stride,
        use_hand_crop=model_cfg.get("use_hand_crop", False),
        rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5),
        emit_cache_key=True,
    )
    os.makedirs(args.out, exist_ok=True)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    n_saved, n_skipped = 0, 0
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        p = os.path.join(args.out, item["cache_key"] + ".pt")
        if os.path.exists(p):
            n_skipped += 1
            continue
        imgs = item["img"].to(device)                       # [S, 3, H, W] in [0,1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.forward_features((imgs - mean) / std)
            tokens = feats["x_norm_patchtokens"]            # [S, P, C]
        torch.save(tokens.to(torch.bfloat16).cpu().clone(), p)
        n_saved += 1
        if n_saved % 200 == 0:
            print(f"  {n_saved} clips cached ({n_skipped} pre-existing)", flush=True)
    s, p_, c = tokens.shape if n_saved else (0, 0, 0)
    print(f"DINO_CACHE_DONE saved={n_saved} skipped={n_skipped} "
          f"shape=[{s},{p_},{c}] model={args.dino_model} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
