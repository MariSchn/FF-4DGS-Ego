"""
Evaluate a trained hand-head checkpoint on a fixed Hot3D val split.

Reports HaMeR-style metrics: MPJPE, PA-MPJPE, MPVPE, PA-MPVPE, AUC_J, AUC_V.

Usage (run as a module from the repo root, with the `neoverse` conda env active):

    # Single checkpoint
    python -m scripts.eval_hand_head \
        --config configs/train_hand_head.yaml \
        --ckpt   checkpoints/default/best_val_loss.pt

    # Sweep all .pt files in a directory
    python -m scripts.eval_hand_head \
        --config configs/train_hand_head.yaml \
        --sweep --ckpt-dir checkpoints/default \
        --out outputs/eval_sweep.csv
"""

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror

from scripts.train_hand_head import (
    HOT3DHandDataset,
    build_views,
    discover_sequences,
)
from scripts.hand_vis_utils import MANOModel
from scripts.hand_metrics import (
    NUM_HANDS,
    metric_chunks_from_batch,
    metrics_from_chunks,
)


# ------------------------------------------------------------------
# Val-split locking
# ------------------------------------------------------------------

def resolve_val_split(data_root, val_split, seed, persist_path: Path):
    """Return the val sequence paths, persisting the chosen list on first run."""
    all_seqs = discover_sequences(data_root)
    if not all_seqs:
        raise RuntimeError(f"No sequences found in {data_root}")

    if persist_path.exists():
        with open(persist_path) as f:
            names = json.load(f)
        by_name = {os.path.basename(p): p for p in all_seqs}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise RuntimeError(
                f"Locked val sequences missing from {data_root}: {missing}. "
                f"Delete {persist_path} to re-lock against the current data root."
            )
        return [by_name[n] for n in names]

    # First run: reproduce the training shuffle exactly (uses the global `random`
    # module with the same seed, like train_hand_head.py:1012-1017), take the
    # first n_val, then persist sorted basenames for a stable file format.
    seqs = list(all_seqs)
    random.seed(seed)
    random.shuffle(seqs)
    n_val = max(1, int(len(seqs) * float(val_split)))
    val_seqs = seqs[:n_val]
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(persist_path, "w") as f:
        json.dump(sorted(os.path.basename(p) for p in val_seqs), f, indent=2)
    print(f"[eval] Locked val split written to {persist_path} ({len(val_seqs)} sequences)")
    return val_seqs


# ------------------------------------------------------------------
# Inference loop
# ------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, val_loader, mano_model, device, num_frames, sanity=False):
    """Forward over the val loader; return per-batch metric chunks (CPU tensors).

    When `sanity=True`, the model's prediction is replaced with the GT params.
    All metrics should then be ~0 (only floating-point noise from the MANO layer);
    this verifies pred/GT plumbing without trusting the trained weights.
    """
    if model is not None:
        model.eval()

    chunks = []
    for batch in tqdm(val_loader, desc="eval"):
        imgs = batch["img"].to(device, non_blocking=True)
        hb = batch.get("hand_bboxes", None)
        if hb is not None:
            hb = hb.to(device, non_blocking=True)
        hv = batch["hand_valid"].to(device, non_blocking=True) if "hand_valid" in batch else None
        gt_params = batch["gt"].to(device, non_blocking=True)

        if sanity:
            pred_params = gt_params
        else:
            views = build_views(imgs, num_frames, device, hb, hv)
            preds = model(views, is_inference=False, use_motion=False)
            pred_params = preds["hand_joints"]  # [B,S,64]

        chunks.append(metric_chunks_from_batch(
            pred_params, gt_params, hv, mano_model, device,
        ))
    return chunks


# ------------------------------------------------------------------
# Model setup
# ------------------------------------------------------------------

def build_model(cfg, device):
    model_cfg = cfg["model"]
    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    base_ckpt = torch.load(model_cfg["checkpoint"], map_location=device)
    state_dict = base_ckpt.get("state_dict", base_ckpt.get("reconstructor", base_ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def load_hand_head(model, ckpt_path, device):
    # train_hand_head.py saves model.hand_head.state_dict() directly (a flat dict).
    sd = torch.load(ckpt_path, map_location=device)
    model.hand_head.load_state_dict(sd, strict=True)


def evaluate_checkpoint(model, ckpt_path, val_loader, mano_model, device, num_frames,
                        sanity=False):
    if not sanity:
        load_hand_head(model, ckpt_path, device)
    chunks = run_inference(
        model, val_loader, mano_model, device, num_frames, sanity=sanity,
    )
    return metrics_from_chunks(chunks)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

_STEP_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def _ckpt_sort_key(path: Path):
    name = path.name
    m = _STEP_RE.match(name)
    if m:
        return (1, int(m.group(1)), name)
    if name == "best_val_loss.pt":
        return (0, -1, name)
    if name == "hand_head_final.pt":
        return (2, 0, name)
    return (3, 0, name)


def list_sweep_checkpoints(ckpt_dir: Path):
    return sorted([p for p in ckpt_dir.glob("*.pt")], key=_ckpt_sort_key)


def print_summary(label, metrics):
    if metrics is None:
        print(f"  {label}: <no valid hands>")
        return
    print(f"  {label}: "
          f"MPJPE={metrics['MPJPE']:.2f}mm  PA={metrics['PA_MPJPE']:.2f}mm  "
          f"MPVPE={metrics['MPVPE']:.2f}mm  PA={metrics['PA_MPVPE']:.2f}mm  "
          f"AUC_J={metrics['AUC_J']:.3f}  AUC_V={metrics['AUC_V']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/train_hand_head.yaml")
    parser.add_argument("--ckpt",       default="checkpoints/default/best_val_loss.pt")
    parser.add_argument("--sweep",      action="store_true")
    parser.add_argument("--ckpt-dir",   default="checkpoints/default")
    parser.add_argument("--val-list",   default="outputs/eval_val_split.json")
    parser.add_argument("--out",        default=None,
                        help="Output path. Defaults: outputs/eval_results.json (single) "
                             "or outputs/eval_sweep.csv (sweep).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sanity",     action="store_true",
                        help="Replace prediction with GT inside the loop. All metrics "
                             "should be ~0 — verifies metric/MANO plumbing.")
    parser.add_argument("--limit-clips", type=int, default=None,
                        help="If set, evaluate only the first N clips (for quick smoke tests).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    vis_cfg  = cfg.get("visualization", {})
    seed     = cfg.get("training", {}).get("seed", 42)

    device = args.device
    out_path = Path(args.out) if args.out else Path(
        "outputs/eval_sweep.csv" if args.sweep else "outputs/eval_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Val split (locked) ---
    val_seqs = resolve_val_split(
        data_root=data_cfg["data_root"],
        val_split=data_cfg.get("val_split", 0.01),
        seed=seed,
        persist_path=Path(args.val_list),
    )
    print(f"[eval] Val sequences: {len(val_seqs)}")

    # --- MANO model ---
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
    print(f"[eval] Val clips: {num_clips}")

    # --- Model (built once, hand head reloaded per ckpt) ---
    # In --sanity mode we skip building the model entirely (we never call it).
    model = None if args.sanity else build_model(cfg, device)

    if args.sanity:
        print("[eval] SANITY mode: substituting GT for prediction; metrics should be ~0.")
        result = evaluate_checkpoint(model, None, val_loader, mano_model, device,
                                     num_frames, sanity=True)
        print(f"[eval] Valid hands: {result['num_valid_hands']}")
        for label in ("left", "right", "all"):
            print_summary(label, result[label])
        return

    if not args.sweep:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[eval] Checkpoint: {ckpt_path}")
        result = evaluate_checkpoint(model, str(ckpt_path), val_loader, mano_model, device, num_frames)

        payload = {
            "ckpt": str(ckpt_path),
            "config": str(args.config),
            "val_split": str(args.val_list),
            "num_clips": num_clips,
            "num_valid_hands": result["num_valid_hands"],
            "metrics": {k: result[k] for k in ("left", "right", "all")},
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[eval] Wrote {out_path}")
        print(f"[eval] Valid hands: {result['num_valid_hands']}")
        for label in ("left", "right", "all"):
            print_summary(label, result[label])
        return

    # --- Sweep ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = list_sweep_checkpoints(ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"No .pt files in {ckpt_dir}")
    print(f"[eval] Sweeping {len(ckpts)} checkpoints in {ckpt_dir}")

    fieldnames = ["ckpt", "step", "num_valid_hands",
                  "MPJPE", "PA_MPJPE", "MPVPE", "PA_MPVPE", "AUC_J", "AUC_V"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ckpt_path in ckpts:
            print(f"\n[eval] -- {ckpt_path.name} --")
            result = evaluate_checkpoint(model, str(ckpt_path), val_loader, mano_model, device, num_frames)
            m = _STEP_RE.match(ckpt_path.name)
            step = int(m.group(1)) if m else (
                -1 if ckpt_path.name == "best_val_loss.pt"
                else (10**9 if ckpt_path.name == "hand_head_final.pt" else 0)
            )
            row = {"ckpt": ckpt_path.name, "step": step,
                   "num_valid_hands": result["num_valid_hands"]}
            all_m = result["all"] or {}
            for k in ("MPJPE", "PA_MPJPE", "MPVPE", "PA_MPVPE", "AUC_J", "AUC_V"):
                row[k] = f"{all_m[k]:.4f}" if all_m.get(k) is not None else ""
            writer.writerow(row)
            f.flush()
            print_summary("all", result["all"])
    print(f"\n[eval] Wrote {out_path}")


if __name__ == "__main__":
    main()
