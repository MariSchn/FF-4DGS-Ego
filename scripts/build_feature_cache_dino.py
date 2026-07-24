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
    ap.add_argument("--batch_size", type=int, default=2)
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

    # Same corrupt-clip tolerance + batched DataLoader as build_feature_cache.py:
    # decord can hard-fail on damaged mp4 bitstreams; retry once, then skip loudly.
    from torch.utils.data import DataLoader, default_collate

    class _SafeDS(torch.utils.data.Dataset):
        def __init__(self, inner): self.inner = inner
        def __len__(self): return len(self.inner)
        def __getitem__(self, i):
            err = None
            for _ in range(2):
                try:
                    return self.inner[i]
                except Exception as e:  # noqa: BLE001 — worker must never die
                    err = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"SKIP_CLIP idx={i} {err}", flush=True)
            return None

    def _collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        return default_collate(batch) if batch else None

    dl = DataLoader(_SafeDS(ds), batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=False, collate_fn=_collate_skip_none)

    n_saved, n_skipped = 0, 0
    shape = (0, 0, 0)
    for batch in dl:
        if batch is None:
            continue
        keys = batch["cache_key"]                           # list[str]
        paths = [os.path.join(args.out, k + ".pt") for k in keys]
        if all(os.path.exists(p) for p in paths):
            n_skipped += len(keys)
            continue
        imgs = batch["img"].to(device)                      # [B, S, 3, H, W] in [0,1]
        B, S = imgs.shape[:2]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.forward_features(
                ((imgs - mean.unsqueeze(0)) / std.unsqueeze(0)).flatten(0, 1))
            tokens = feats["x_norm_patchtokens"]            # [B*S, P, C]
        tokens = tokens.reshape(B, S, *tokens.shape[1:]).to(torch.bfloat16).cpu()
        shape = tuple(tokens.shape[1:])
        for b, p in enumerate(paths):
            if not os.path.exists(p):
                torch.save(tokens[b].clone(), p)
                n_saved += 1
        if (n_saved + n_skipped) % 200 < args.batch_size:
            print(f"  cache progress: done={n_saved} skip={n_skipped}", flush=True)
    print(f"DINO_CACHE_DONE saved={n_saved} skipped={n_skipped} "
          f"shape={list(shape)} model={args.dino_model} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
