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
    ap.add_argument("--random_init", action="store_true",
                    help="backbone-swap ablation arm: skip the checkpoint and cache tokens "
                         "from a RANDOMLY initialized backbone (fixed seed 0). If these "
                         "tokens train a head as well as the pretrained ones, the "
                         "'reconstruction features encode metric depth' claim is dead.")
    ap.add_argument("--cam_token_out", default=None,
                    help="also dump the per-frame CAMERA token (token_list[-1][:,:,0]) here, one "
                         "[S,C] bf16 tensor per clip under the same <seq>_<offset>.pt key. This is "
                         "what CameraHead consumes; the patch-token cache strips it, so camera-head "
                         "training is impossible without it. ~64 KB/clip vs ~16.8 MB, so it is "
                         "nearly free to build in the same pass.")
    ap.add_argument("--min_labelled_frames", type=int, default=None,
                    help="keep only clips with at least this many GT-labelled frames, exactly as "
                         "the trainer does. MUST match data.min_labelled_frames in the training "
                         "config: the cache is keyed <seq>_<offset>, so a build that filters MORE "
                         "than the trainer leaves it looking up absent keys. Defaults to the "
                         "config value. Sparse stores make this load-bearing rather than a "
                         "tuning knob - Ego-Exo4D labels ~2.3% of frames, so an unfiltered build "
                         "writes ~16.8 MB/clip for clips that carry no supervision at all.")
    ap.add_argument("--save_model", default=None,
                    help="save the constructed model's state_dict here (random-init arm: the "
                         "eval points model.checkpoint at this file so the eval-time backbone "
                         "is bit-identical to the cache-time backbone)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    device = "cuda"

    if args.random_init:
        torch.manual_seed(0)
    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    if args.random_init:
        print("RANDOM_INIT backbone (seed 0) — no checkpoint loaded", flush=True)
    else:
        ckpt = torch.load(model_cfg["checkpoint"], map_location=device)
        sd = ckpt.get("state_dict", ckpt.get("reconstructor", ckpt))
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"MODEL_SAVED {args.save_model}", flush=True)

    from scripts.hand_vis_utils import MANOModel
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])

    seqs = discover_sequences(args.data_root)
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    min_lab = (args.min_labelled_frames if args.min_labelled_frames is not None
               else data_cfg.get("min_labelled_frames", 0))
    print(f"min_labelled_frames={min_lab} "
          f"({'from --min_labelled_frames' if args.min_labelled_frames is not None else 'from config'})",
          flush=True)
    ds = HOT3DHandDataset(
        seqs, mano,
        num_frames=data_cfg.get("num_frames", 16),
        res=tuple(data_cfg.get("resolution", [224, 224])),
        clip_stride=args.clip_stride,
        use_hand_crop=model_cfg.get("use_hand_crop", False),
        rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5),
        emit_cache_key=True,
        min_labelled_frames=min_lab,
    )
    if len(ds) == 0:
        raise SystemExit(
            f"EMPTY DATASET after min_labelled_frames={min_lab} filtering over {len(seqs)} "
            f"sequences. Building a cache here would write nothing and the training run would "
            f"then fail on missing keys. Lower --min_labelled_frames or check the store's GT.")
    os.makedirs(args.out, exist_ok=True)
    if args.cam_token_out:
        os.makedirs(args.cam_token_out, exist_ok=True)
        print(f"also dumping camera tokens -> {args.cam_token_out}", flush=True)

    # Corrupt-clip tolerance: decord's threaded decoder can hard-fail on damaged
    # mp4 bitstreams (observed: "Error at MB" -> DECORDError kills the worker).
    # Retry once, then skip the clip loudly instead of aborting a multi-hour build.
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

    from torch.utils.data import default_collate

    def _collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        return default_collate(batch) if batch else None

    # Drop already-cached clips HERE, from the clip index, not after the loader has built them.
    # The skip below still works and stays as a backstop, but reaching it costs a full video
    # decode per clip with the GPU idle. Resuming a 17.5k-clip build that way ran for half an
    # hour without writing anything. The key is the same expression the training loader uses.
    if hasattr(ds, "clips"):
        n0 = len(ds.clips)
        ds.clips = [c for c in ds.clips if not os.path.exists(os.path.join(
            args.out, f"{os.path.basename(c['seq_path'])}_{c['frame_offset']}.pt"))]
        print(f"[resume] {n0 - len(ds.clips)} clips already cached, {len(ds.clips)} to build",
              flush=True)
        if not ds.clips:
            print(f"CACHE_BUILD_DONE done=0 skip={n0} out={args.out}", flush=True)
            return

    dl = DataLoader(_SafeDS(ds), batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=False, collate_fn=_collate_skip_none)

    done = skip = 0
    for batch in dl:
        if batch is None:
            continue
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
            # CameraHead reads latest_feat[:, :, 0] - the camera token, which sits BEFORE
            # patch_start_idx and is therefore stripped by the slice above. Without it the
            # camera head cannot be trained from cache at all. It is one token per frame:
            # [S, C] bf16 is ~64 KB/clip against ~16.8 MB for the patch tokens, so dumping it
            # alongside costs ~0.4% more disk and saves a second full backbone pass.
            cam_tokens = token_list[-1][:, :, 0] if args.cam_token_out else None
        tokens = tokens.to(torch.bfloat16).cpu()
        if cam_tokens is not None:
            cam_tokens = cam_tokens.to(torch.bfloat16).cpu()    # [B, S, C]
        for b, p in enumerate(paths):
            if not os.path.exists(p):
                torch.save(tokens[b].clone(), p)
                done += 1
            if cam_tokens is not None:
                cp = os.path.join(args.cam_token_out, os.path.basename(p))
                if not os.path.exists(cp):
                    torch.save(cam_tokens[b].clone(), cp)
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
