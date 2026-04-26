"""
Evaluate the WorldMirror Gaussian-Splatting head on a fixed Hot3D val split.

Reports image-quality metrics by re-rendering the input views from the
predicted Gaussians and comparing against the ground-truth frames:
PSNR, SSIM, LPIPS.

Usage (run as a module from the repo root, with the `neoverse` conda env active):

    # Base reconstructor only (no hand-head ckpt — hand head doesn't affect GS)
    python -m scripts.eval_gs_head --config configs/train_hand_head.yaml

    # With a specific hand-head checkpoint loaded (results should be identical)
    python -m scripts.eval_gs_head \
        --config configs/train_hand_head.yaml \
        --ckpt   checkpoints/default/best_val_loss.pt

    # Sweep all .pt files in a directory
    python -m scripts.eval_gs_head \
        --config configs/train_hand_head.yaml \
        --sweep --ckpt-dir checkpoints/default \
        --out outputs/eval_gs_sweep.csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.train_hand_head import (
    HOT3DHandDataset,
    build_views,
)
from scripts.hand_vis_utils import MANOModel
from scripts.eval_hand_head import (
    resolve_val_split,
    build_model,
    load_hand_head,
    list_sweep_checkpoints,
    _STEP_RE,
)
from scripts.gs_metrics import (
    LPIPSScorer,
    render_views_from_predictions,
    metric_chunks_from_batch,
    metrics_from_chunks,
)


# ------------------------------------------------------------------
# Inference loop
# ------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, val_loader, lpips_scorer, device, num_frames, sanity=False):
    """Forward over the val loader; return per-batch GS metric chunks (CPU tensors).

    When `sanity=True`, the rendered output is replaced with the GT images.
    All metrics should then be ideal (PSNR=inf, SSIM=1, LPIPS=0); this verifies
    pred/GT plumbing without trusting the trained weights.
    """
    if model is not None:
        model.eval()

    chunks = []
    for batch in tqdm(val_loader, desc="eval-gs"):
        imgs = batch["img"].to(device, non_blocking=True)
        hb = batch.get("hand_bboxes", None)
        if hb is not None:
            hb = hb.to(device, non_blocking=True)
        hv = batch["hand_valid"].to(device, non_blocking=True) if "hand_valid" in batch else None

        H_img, W_img = imgs.shape[-2:]
        views = build_views(imgs, num_frames, device, hb, hv)

        if sanity:
            # [B, S, 3, H, W] -> [B, S, H, W, 3] (range [0, 1] preserved)
            rendered = imgs.permute(0, 1, 3, 4, 2).contiguous()
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(views, is_inference=False, use_motion=False)
                rendered = render_views_from_predictions(
                    model, preds, views, height=H_img, width=W_img,
                )

        chunks.append(metric_chunks_from_batch(
            rendered, imgs, None, lpips_scorer, device,
        ))
    return chunks


def evaluate_checkpoint(model, ckpt_path, val_loader, lpips_scorer, device, num_frames,
                        sanity=False):
    if not sanity and ckpt_path is not None:
        load_hand_head(model, ckpt_path, device)
    chunks = run_inference(
        model, val_loader, lpips_scorer, device, num_frames, sanity=sanity,
    )
    return metrics_from_chunks(chunks)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def print_summary(metrics):
    if metrics is None or metrics.get("num_valid_frames", 0) == 0:
        print("  <no valid frames>")
        return
    print(f"  PSNR={metrics['PSNR']:.2f}dB  "
          f"SSIM={metrics['SSIM']:.4f}  "
          f"LPIPS={metrics['LPIPS']:.4f}  "
          f"(N={metrics['num_valid_frames']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/train_hand_head.yaml")
    parser.add_argument("--ckpt",       default=None,
                        help="Optional hand-head checkpoint. Does NOT affect GS metrics; "
                             "only useful for sweep parity with eval_hand_head.")
    parser.add_argument("--sweep",      action="store_true")
    parser.add_argument("--ckpt-dir",   default="checkpoints/default")
    parser.add_argument("--val-list",   default="outputs/eval_val_split.json")
    parser.add_argument("--out",        default=None,
                        help="Output path. Defaults: outputs/eval_gs_results.json (single) "
                             "or outputs/eval_gs_sweep.csv (sweep).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips-net",  choices=["alex", "vgg"], default="alex")
    parser.add_argument("--sanity",     action="store_true",
                        help="Replace rendered output with GT inside the loop. "
                             "PSNR≈inf, SSIM≈1, LPIPS≈0 — verifies metric plumbing.")
    parser.add_argument("--limit-clips", type=int, default=None,
                        help="If set, evaluate only the first N clips (for quick smoke tests).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("model", {}).get("enable_gs", False):
        raise RuntimeError(
            "GS evaluation requires `model.enable_gs: true` in the config; "
            "otherwise the model never builds gs_head/gs_renderer and no splats "
            "are produced."
        )

    data_cfg = cfg["data"]
    vis_cfg  = cfg.get("visualization", {})
    seed     = cfg.get("training", {}).get("seed", 42)

    device = args.device
    out_path = Path(args.out) if args.out else Path(
        "outputs/eval_gs_sweep.csv" if args.sweep else "outputs/eval_gs_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Val split (locked; shares the file with eval_hand_head) ---
    val_seqs = resolve_val_split(
        data_root=data_cfg["data_root"],
        val_split=data_cfg.get("val_split", 0.01),
        seed=seed,
        persist_path=Path(args.val_list),
    )
    print(f"[eval-gs] Val sequences: {len(val_seqs)}")

    # --- MANO model (only used by the dataset to build GT joints) ---
    mano_folder = vis_cfg.get("mano_model_folder")
    if not mano_folder:
        raise RuntimeError("visualization.mano_model_folder must be set in config")
    mano_model = MANOModel(mano_folder)

    # --- Dataset / loader ---
    num_frames    = data_cfg["num_frames"]
    res           = tuple(data_cfg["resolution"])
    clip_stride   = data_cfg.get("clip_stride", num_frames)
    use_hand_crop = cfg["model"].get("use_hand_crop", False) or cfg["model"].get("hand_head_type") == "hand_crop"
    rescale_factor = cfg.get("hand_crop", {}).get("rescale_factor", 2.0)

    val_set = HOT3DHandDataset(
        val_seqs, mano_model,
        num_frames=num_frames, res=res, clip_stride=clip_stride,
        use_hand_crop=use_hand_crop, rescale_factor=rescale_factor,
    )
    if args.limit_clips is not None:
        val_set.clips = val_set.clips[: args.limit_clips]
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    num_clips = len(val_set)
    print(f"[eval-gs] Val clips: {num_clips}")

    # --- LPIPS scorer (built once, reused across all checkpoints) ---
    lpips_scorer = LPIPSScorer(net=args.lpips_net, device=device)

    # --- Model (built once, hand head reloaded per ckpt) ---
    model = None if args.sanity else build_model(cfg, device)

    if args.sanity:
        print("[eval-gs] SANITY mode: substituting GT for rendered; "
              "PSNR≈inf, SSIM≈1, LPIPS≈0 expected.")
        result = evaluate_checkpoint(model, None, val_loader, lpips_scorer, device,
                                     num_frames, sanity=True)
        print_summary(result)
        return

    if not args.sweep:
        ckpt_path = Path(args.ckpt) if args.ckpt else None
        if ckpt_path is not None and not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        if ckpt_path is not None:
            print(f"[eval-gs] Hand-head checkpoint: {ckpt_path}")
        else:
            print("[eval-gs] No --ckpt given; evaluating base reconstructor.")
        result = evaluate_checkpoint(model, str(ckpt_path) if ckpt_path else None,
                                     val_loader, lpips_scorer, device, num_frames)

        payload = {
            "ckpt": str(ckpt_path) if ckpt_path else None,
            "config": str(args.config),
            "val_split": str(args.val_list),
            "num_clips": num_clips,
            "num_valid_frames": result["num_valid_frames"],
            "metrics": {k: result[k] for k in ("PSNR", "SSIM", "LPIPS")},
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[eval-gs] Wrote {out_path}")
        print_summary(result)
        return

    # --- Sweep ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = list_sweep_checkpoints(ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"No .pt files in {ckpt_dir}")
    print(f"[eval-gs] Sweeping {len(ckpts)} checkpoints in {ckpt_dir}")

    fieldnames = ["ckpt", "step", "num_valid_frames", "PSNR", "SSIM", "LPIPS"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ckpt_path in ckpts:
            print(f"\n[eval-gs] -- {ckpt_path.name} --")
            result = evaluate_checkpoint(model, str(ckpt_path), val_loader,
                                         lpips_scorer, device, num_frames)
            m = _STEP_RE.match(ckpt_path.name)
            step = int(m.group(1)) if m else (
                -1 if ckpt_path.name == "best_val_loss.pt"
                else (10**9 if ckpt_path.name == "hand_head_final.pt" else 0)
            )
            row = {"ckpt": ckpt_path.name, "step": step,
                   "num_valid_frames": result["num_valid_frames"]}
            for k in ("PSNR", "SSIM", "LPIPS"):
                row[k] = f"{result[k]:.4f}" if result.get(k) is not None else ""
            writer.writerow(row)
            f.flush()
            print_summary(result)
    print(f"\n[eval-gs] Wrote {out_path}")


if __name__ == "__main__":
    main()
