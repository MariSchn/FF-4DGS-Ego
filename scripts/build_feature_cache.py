"""Precompute frozen-backbone patch tokens for head-only (cached) training.

Saves, per clip, ``token_list[-1][:, :, patch_start_idx:]`` from
``WorldMirror.visual_geometry_transformer(imgs, use_motion=False)`` — exactly the
tensor ``HamerManoHead`` consumes on the trainer's ``use_cond=False`` path — as
bf16 ``[S, P, C]`` to ``<out>/<seq>_<frame_offset>.pt``. Cached training then
feeds it back via ``forward_hand_cached`` (``patch_start_idx=0``), skipping the
backbone: ~10x faster steps, and ablations become ~1h instead of ~22h.

Sizing: S=16, P=256 (224/14), C=2048 -> 16.8 MB/clip bf16; the 367-seq HOI4D
train split at clip_stride 16 is ~117 GB. That fits ONLY node-local /tmp
(~860 GB) — pin the build job and every cached-training job to the SAME node
(e.g. ``#SBATCH -w studgpu-spark03``), and remember /tmp does not survive a
node reboot (rebuild is ~2-3 h, resumable: existing files are skipped).

IMPORTANT: cached training must use the SAME clip_stride as the build
(data.clip_stride=16), or its <seq>_<offset> keys miss the cache (loud failure).

Usage (gb10, venv_gb10):
    python -m scripts.build_feature_cache --config configs/exp_p4_full.yaml \
        --data_root /home/dmonopoli/hoi4d_train --out /tmp/featcache/train \
        --clip_stride 16
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from scripts.train_hand_head import HOT3DHandDataset, discover_sequences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip_stride", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_seqs", type=int, default=None)
    ap.add_argument("--validate", type=int, default=2,
                    help="after building, re-run N clips through the backbone in train() "
                         "mode (as the trainer runs it) and print max|diff| vs the cache")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    device = "cuda"

    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    ckpt = torch.load(model_cfg["checkpoint"], map_location=device)
    sd = ckpt.get("state_dict", ckpt.get("reconstructor", ckpt))
    model.load_state_dict(sd, strict=False)
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
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    done = skip = 0
    for batch in dl:
        keys = batch["cache_key"]                       # list[str]
        paths = [os.path.join(args.out, k + ".pt") for k in keys]
        if all(os.path.exists(p) for p in paths):
            skip += len(keys)
            continue
        imgs = batch["img"].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            token_list, patch_start_idx, _, _ = model.visual_geometry_transformer(
                imgs, use_motion=False)
            tokens = token_list[-1][:, :, patch_start_idx:]     # [B, S, P, C]
        tokens = tokens.to(torch.bfloat16).cpu()
        for b, p in enumerate(paths):
            if not os.path.exists(p):
                torch.save(tokens[b].clone(), p)
                done += 1
        if (done + skip) % 200 < args.batch_size:
            print(f"cache progress: done={done} skip={skip}", flush=True)
    print(f"CACHE_BUILD_DONE done={done} skip={skip} out={args.out}", flush=True)

    # Fidelity check: the trainer keeps the frozen backbone in train() mode (under
    # no_grad+autocast); the cache above was built in eval() mode. If the backbone
    # carries any train-mode stochasticity (dropout/droppath) this diff exposes it.
    if args.validate:
        model.train()
        for i, batch in enumerate(DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)):
            if i >= args.validate:
                break
            imgs = batch["img"].to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tl, psi, _, _ = model.visual_geometry_transformer(imgs, use_motion=False)
                live = tl[-1][:, :, psi:][0].to(torch.bfloat16).cpu()
            saved = torch.load(os.path.join(args.out, batch["cache_key"][0] + ".pt"),
                               weights_only=True)
            diff = (live.float() - saved.float()).abs().max().item()
            print(f"CACHE_VALIDATE {batch['cache_key'][0]} max|diff|={diff:.6f}", flush=True)


if __name__ == "__main__":
    main()
