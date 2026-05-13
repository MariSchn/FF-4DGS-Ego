"""Evaluate original pre-trained HaMeR as a baseline on the HOT3D val split.

Comparison target: per-frame, per-hand-crop HaMeR (no cross-attention, no
temporal context) vs. our trained HamerManoHead.  The Gaussian head is not
involved — we measure MANO parameter quality only.

Metrics are the same six reported by eval_hand_head.py (MPJPE, PA-MPJPE,
MPVPE, PA-MPVPE, AUC_J, AUC_V), computed on the same locked val sequences.
Joints and vertices are made wrist-relative before computing MPJPE/MPVPE so
that global translation differences between HaMeR's weak-perspective output
and the camera-space GT do not inflate the raw error.  PA-MPJPE / PA-MPVPE
are unaffected (Procrustes removes global pose anyway).

Installation
------------
    pip install git+https://github.com/geopavlakos/hamer

Checkpoint
----------
    Download hamer_demo.tar.gz from the HaMeR project page and extract
    hamer.ckpt into models/hamer/hamer.ckpt.

Usage
-----
    python -m scripts.eval_hamer_baseline \\
        --config  configs/train_hand_head.yaml \\
        --hamer-ckpt models/hamer/hamer.ckpt \\
        --out     outputs/hamer_baseline.json

    # Quick smoke-test on 20 clips:
    python -m scripts.eval_hamer_baseline \\
        --config configs/train_hand_head.yaml \\
        --hamer-ckpt models/hamer/hamer.ckpt \\
        --limit-clips 20
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.eval_hand_head import resolve_val_split
from scripts.train_hand_head import HOT3DHandDataset, discover_sequences
from scripts.hand_vis_utils import MANOModel, quat_wxyz_to_axis_angle_torch
from scripts.hand_metrics import (
    NUM_HANDS,
    HAND_PARAM_DIM,
    metrics_from_chunks,
    aggregate,
    _layer_joints_and_vertices,
)


# HaMeR crop constants: input must be 256x256; backbone slices x[:,:,:,32:-32]
# internally to get the 256x192 region that matches the positional embeddings.
_HAMER_H = 256
_HAMER_W = 256
_HAMER_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_HAMER_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ------------------------------------------------------------------
# Image pre-processing
# ------------------------------------------------------------------

def _crop_hand(img_chw: torch.Tensor, bbox_cxcywh: torch.Tensor) -> torch.Tensor:
    """Crop a single hand region and resize to HaMeR input size.

    Args:
        img_chw:    (3, H, W) float tensor in [0, 1].
        bbox_cxcywh: (4,) tensor – cx, cy, w, h in pixel coords.

    Returns:
        (3, _HAMER_H, _HAMER_W) normalised tensor ready for HaMeR.
    """
    _, H, W = img_chw.shape
    cx, cy, bw, bh = bbox_cxcywh.tolist()

    # Square crop (take max side) for a stable aspect ratio
    side = max(bw, bh)
    if side < 4:  # degenerate bbox — skip
        return None
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2))
    y1 = int(round(cy + side / 2))

    # Pad if the bbox extends beyond image borders
    pad_left   = max(0, -x0)
    pad_top    = max(0, -y0)
    pad_right  = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)
    if pad_left or pad_top or pad_right or pad_bottom:
        img_chw = F.pad(img_chw, (pad_left, pad_right, pad_top, pad_bottom))
        x0 += pad_left;  x1 += pad_left
        y0 += pad_top;   y1 += pad_top

    crop = img_chw[:, y0:y1, x0:x1]
    crop = F.interpolate(
        crop.unsqueeze(0), size=(_HAMER_H, _HAMER_W), mode="bilinear", align_corners=False
    ).squeeze(0)

    mean = _HAMER_MEAN.to(crop.device)
    std  = _HAMER_STD.to(crop.device)
    return (crop - mean) / std


def prepare_hamer_batch(imgs_bschw, hand_bboxes_bs24, hand_valid_bs2, device):
    """Extract all valid hand crops from a batch and build a HaMeR input dict.

    Args:
        imgs_bschw:      (B, S, 3, H, W) float [0,1].
        hand_bboxes_bs24:(B, S, 2, 4)   cx/cy/w/h pixel coords.
        hand_valid_bs2:  (B, S, 2)      bool.

    Returns:
        crops:  (N, 3, _HAMER_H, _HAMER_W) on `device`.
        rights: (N,) bool  — True if right hand.
        index:  list of (b, s, h) tuples, one per crop (N entries).
    """
    crops, rights, index = [], [], []
    B, S = imgs_bschw.shape[:2]
    for b in range(B):
        for s in range(S):
            for h in range(NUM_HANDS):
                if not hand_valid_bs2[b, s, h]:
                    continue
                crop = _crop_hand(
                    imgs_bschw[b, s].cpu(),
                    hand_bboxes_bs24[b, s, h].cpu(),
                )
                if crop is None:
                    continue
                is_right = (h == 1)
                # HaMeR expects right hands; mirror left-hand crops
                if not is_right:
                    crop = torch.flip(crop, dims=[-1])
                crops.append(crop)
                rights.append(is_right)
                index.append((b, s, h))
    if not crops:
        return None, None, []
    return (
        torch.stack(crops).to(device),
        torch.tensor(rights, dtype=torch.bool, device=device),
        index,
    )


# ------------------------------------------------------------------
# GT joints / vertices
# ------------------------------------------------------------------

def _gt_joints_verts(gt_params_bs64, mano_model, device):
    """(B, S, 64) GT params -> joints (B,S,2,16,3), verts (B,S,2,778,3)."""
    from scripts.hand_metrics import joints_and_vertices_from_params
    return joints_and_vertices_from_params(gt_params_bs64, mano_model, device)


# ------------------------------------------------------------------
# Wrist-relative normalisation
# ------------------------------------------------------------------

def _wrist_relative(joints_n163, verts_n7783):
    """Subtract wrist (joint 0) from joints and vertices."""
    wrist = joints_n163[:, :1, :]        # (N, 1, 3)
    return joints_n163 - wrist, verts_n7783 - wrist.unsqueeze(1)


# ------------------------------------------------------------------
# Main inference loop
# ------------------------------------------------------------------

@torch.no_grad()
def run_hamer_inference(hamer_model, val_loader, mano_model, device, pelvis_ind=0):
    """Run original HaMeR on every valid hand crop in the val set.

    Returns a list of per-batch chunk dicts compatible with metrics_from_chunks.
    """
    NUM_MANO_JOINTS = 16
    assert 0 <= pelvis_ind < NUM_MANO_JOINTS, (
        f"pelvis_ind={pelvis_ind} is out of range for {NUM_MANO_JOINTS} MANO joints. "
        f"Check model_cfg.EXTRA.PELVIS_IND."
    )

    hamer_model.eval()
    chunks = []
    _root_check_done = False   # verify root-zeroing on first valid batch
    total_valid = 0
    total_hands = 0

    for batch in tqdm(val_loader, desc="hamer-baseline"):
        imgs       = batch["img"].to(device)          # (B, S, 3, H, W)
        gt_params  = batch["gt"].to(device)           # (B, S, 64)
        hv         = batch.get("hand_valid")
        hb         = batch.get("hand_bboxes")

        if hv is None or hb is None:
            # Dataset was built without crop support — skip batch
            continue

        hv = hv.to(device)
        hb = hb.to(device)

        B, S, _ = gt_params.shape

        # --- GT joints + vertices ---
        gt_j, gt_v = _gt_joints_verts(gt_params, mano_model, device)
        # (B, S, 2, 16, 3) / (B, S, 2, 778, 3)

        # --- HaMeR predictions ---
        crops, rights, index = prepare_hamer_batch(imgs, hb, hv, device)

        # pred_j_map[b][s][h] = (16, 3) joints tensor; None if invalid
        pred_j_map = [[[None] * NUM_HANDS for _ in range(S)] for _ in range(B)]
        pred_v_map = [[[None] * NUM_HANDS for _ in range(S)] for _ in range(B)]

        if crops is not None and len(crops) > 0:
            hamer_out = hamer_model({"img": crops, "right": rights})

            # HaMeR output keys (adjust if the installed version differs):
            #   pred_keypoints_3d : (N, 21, 3)  – MANO joints in local frame
            #   pred_vertices     : (N, 778, 3) – mesh vertices
            # The first 16 rows of pred_keypoints_3d are the MANO joints.
            raw_joints = hamer_out["pred_keypoints_3d"][:, :16, :]  # (N, 16, 3)
            raw_verts  = hamer_out["pred_vertices"]                  # (N, 778, 3)

            for crop_idx, (b, s, h) in enumerate(index):
                j = raw_joints[crop_idx]   # (16, 3)
                v = raw_verts[crop_idx]    # (778, 3)
                # HaMeR was run on mirrored crops for left hands;
                # flip the x-axis of the output back.
                if h == 0:
                    j = j.clone(); j[:, 0] *= -1
                    v = v.clone(); v[:, 0] *= -1
                pred_j_map[b][s][h] = j
                pred_v_map[b][s][h] = v

        # --- Assemble flat tensors (one entry per (b,s,h) triplet) ---
        pred_j_list, gt_j_list = [], []
        pred_v_list, gt_v_list = [], []
        side_list, valid_list   = [], []

        for b in range(B):
            for s in range(S):
                for h in range(NUM_HANDS):
                    valid = bool(hv[b, s, h].item()) and pred_j_map[b][s][h] is not None
                    valid_list.append(valid)
                    side_list.append(h)

                    gj = gt_j[b, s, h].cpu()     # (16, 3)
                    gv = gt_v[b, s, h].cpu()     # (778, 3)

                    if valid:
                        pj = pred_j_map[b][s][h].cpu()
                        pv = pred_v_map[b][s][h].cpu()
                    else:
                        pj = torch.zeros_like(gj)
                        pv = torch.zeros_like(gv)

                    # Make pelvis-relative (pelvis_ind from model_cfg.EXTRA.PELVIS_IND)
                    # so MPJPE is not inflated by the global translation difference
                    # between weak-perspective HaMeR output and camera-space GT.
                    root_p = pj[pelvis_ind:pelvis_ind + 1]
                    root_g = gj[pelvis_ind:pelvis_ind + 1]
                    pj = pj - root_p
                    pv = pv - root_p
                    gj = gj - root_g
                    gv = gv - root_g

                    pred_j_list.append(pj)
                    pred_v_list.append(pv)
                    gt_j_list.append(gj)
                    gt_v_list.append(gv)

                    if valid:
                        total_valid += 1
                    total_hands += 1

        # Verify root joint is at the origin after subtraction (first valid batch only)
        if not _root_check_done and any(valid_list):
            stacked_pj = torch.stack(pred_j_list)   # (N, 16, 3)
            stacked_gj = torch.stack(gt_j_list)
            valid_mask = torch.tensor(valid_list)
            if valid_mask.any():
                max_pred_root = stacked_pj[valid_mask, pelvis_ind, :].abs().max().item()
                max_gt_root   = stacked_gj[valid_mask, pelvis_ind, :].abs().max().item()
                assert max_pred_root < 1e-5, (
                    f"Pred root joint not zeroed after subtraction (max={max_pred_root:.2e}). "
                    f"pelvis_ind={pelvis_ind} may be wrong."
                )
                assert max_gt_root < 1e-5, (
                    f"GT root joint not zeroed after subtraction (max={max_gt_root:.2e}). "
                    f"pelvis_ind={pelvis_ind} may be wrong."
                )
                print(f"[hamer-baseline] Root-zeroing check passed (pelvis_ind={pelvis_ind}, "
                      f"max residual pred={max_pred_root:.2e}, gt={max_gt_root:.2e})")
                _root_check_done = True

        chunks.append({
            "pred_j": torch.stack(pred_j_list),
            "gt_j":   torch.stack(gt_j_list),
            "pred_v": torch.stack(pred_v_list),
            "gt_v":   torch.stack(gt_v_list),
            "side":   torch.tensor(side_list, dtype=torch.long),
            "valid":  torch.tensor(valid_list, dtype=torch.bool),
        })

    valid_pct = 100.0 * total_valid / total_hands if total_hands > 0 else 0.0
    print(f"[hamer-baseline] Valid hands: {total_valid}/{total_hands} ({valid_pct:.1f}%)")

    return chunks


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate original HaMeR checkpoint as baseline on HOT3D val split."
    )
    parser.add_argument("--config",      default="configs/train_hand_head.yaml")
    parser.add_argument("--hamer-ckpt",  default="models/hamer/hamer.ckpt",
                        help="Path to the original HaMeR Lightning checkpoint.")
    parser.add_argument("--val-list",    default="outputs/eval_val_split.json",
                        help="Same locked val-split JSON used by eval_hand_head.py.")
    parser.add_argument("--out",         default="outputs/hamer_baseline.json")
    parser.add_argument("--batch-size",  type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-clips", type=int, default=None,
                        help="Evaluate only the first N clips (quick smoke-test).")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # --- Load HaMeR ---
    try:
        from hamer.models import load_hamer
    except ImportError:
        raise ImportError(
            "HaMeR is not installed.\n"
            "  pip install git+https://github.com/geopavlakos/hamer\n"
            "Then download hamer.ckpt into models/hamer/ from the HaMeR project page."
        )

    ckpt_path = Path(args.hamer_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"HaMeR checkpoint not found: {ckpt_path}\n"
            "Download hamer_demo.tar.gz from the HaMeR project page and extract "
            "hamer.ckpt into models/hamer/."
        )

    print(f"[hamer-baseline] Loading HaMeR from {ckpt_path}")
    hamer_model, model_cfg = load_hamer(str(ckpt_path))
    pelvis_ind = model_cfg.EXTRA.PELVIS_IND
    print(f"[hamer-baseline] Using pelvis_ind={pelvis_ind} for alignment")
    hamer_model = hamer_model.to(args.device)
    hamer_model.eval()

    # --- Config + data ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    vis_cfg  = cfg.get("visualization", {})
    seed     = cfg.get("training", {}).get("seed", 42)

    val_seqs = resolve_val_split(
        data_root=data_cfg["data_root"],
        val_split=data_cfg.get("val_split", 0.01),
        seed=seed,
        persist_path=Path(args.val_list),
    )
    print(f"[hamer-baseline] Val sequences: {len(val_seqs)}")

    mano_folder = vis_cfg.get("mano_model_folder")
    if not mano_folder:
        raise RuntimeError("visualization.mano_model_folder must be set in config")
    mano_model = MANOModel(mano_folder)

    num_frames     = data_cfg["num_frames"]
    res            = tuple(data_cfg["resolution"])
    clip_stride    = data_cfg.get("clip_stride", num_frames)
    rescale_factor = cfg.get("hand_crop", {}).get("rescale_factor", 2.0)

    # Always request crops — we need bboxes to feed HaMeR
    val_set = HOT3DHandDataset(
        val_seqs, mano_model,
        num_frames=num_frames, res=res, clip_stride=clip_stride,
        use_hand_crop=True, rescale_factor=rescale_factor,
    )
    if args.limit_clips is not None:
        val_set.clips = val_set.clips[: args.limit_clips]

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    print(f"[hamer-baseline] Val clips: {len(val_set)}")

    # --- Inference ---
    chunks = run_hamer_inference(hamer_model, val_loader, mano_model, args.device, pelvis_ind=pelvis_ind)
    result = metrics_from_chunks(chunks)

    # --- Print ---
    print(f"\n[hamer-baseline] Valid hands: {result['num_valid_hands']}")
    for label in ("left", "right", "all"):
        m = result[label]
        if m is None:
            print(f"  {label}: <no valid hands>")
        else:
            print(
                f"  {label}: "
                f"MPJPE={m['MPJPE']:.2f}mm  PA={m['PA_MPJPE']:.2f}mm  "
                f"MPVPE={m['MPVPE']:.2f}mm  PA={m['PA_MPVPE']:.2f}mm  "
                f"AUC_J={m['AUC_J']:.3f}  AUC_V={m['AUC_V']:.3f}"
            )

    # --- Save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hamer_ckpt":       str(ckpt_path),
        "config":           str(args.config),
        "val_split":        str(args.val_list),
        "num_clips":        len(val_set),
        "num_valid_hands":  result["num_valid_hands"],
        "note":             "wrist-relative MPJPE/MPVPE; PA metrics are Procrustes-aligned",
        "metrics":          {k: result[k] for k in ("left", "right", "all")},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[hamer-baseline] Results written to {out_path}")


if __name__ == "__main__":
    main()
