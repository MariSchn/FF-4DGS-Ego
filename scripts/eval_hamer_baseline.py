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
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from skimage.filters import gaussian as skimage_gaussian
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import HaMeR dataset utils directly from the source file to avoid the
# package __init__.py, which pulls in `webdataset` (not installed here).
_hamer_utils_path = Path(__file__).parent.parent / "models/hamer/hamer/datasets/utils.py"
_spec = importlib.util.spec_from_file_location("hamer_dataset_utils", _hamer_utils_path)
_hamer_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hamer_utils)
_generate_image_patch_cv2 = _hamer_utils.generate_image_patch_cv2
_expand_to_aspect_ratio   = _hamer_utils.expand_to_aspect_ratio

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


# ------------------------------------------------------------------
# Image pre-processing — mirrors ViTDetDataset.__getitem__ exactly
# ------------------------------------------------------------------

def _crop_hand(img_chw: torch.Tensor, bbox_cxcywh: torch.Tensor,
               img_size: int, bbox_shape, mean_255: np.ndarray,
               std_255: np.ndarray) -> torch.Tensor:
    """Crop a single hand region using HaMeR's ViTDetDataset pipeline.

    Matches ViTDetDataset.__getitem__ (models/hamer/hamer/datasets/vitdet_dataset.py):
      - expand_to_aspect_ratio with BBOX_SHAPE from model cfg
      - skimage Gaussian anti-aliasing before downsampling
      - cv2.warpAffine via generate_image_patch_cv2 (no integer rounding artefacts)
      - uint8-scale ImageNet normalisation (mean/std in [0, 255] range)

    Args:
        img_chw:      (3, H, W) float tensor in [0, 1].
        bbox_cxcywh:  (4,) tensor — cx, cy, w, h in pixel coords.
        img_size:     output crop side length (model_cfg.MODEL.IMAGE_SIZE, usually 256).
        bbox_shape:   model_cfg.MODEL.BBOX_SHAPE or None — passed to expand_to_aspect_ratio.
        mean_255:     (3,) numpy array — 255 * IMAGE_MEAN, e.g. 255*[0.485,0.456,0.406].
        std_255:      (3,) numpy array — 255 * IMAGE_STD,  e.g. 255*[0.229,0.224,0.225].

    Returns:
        (3, img_size, img_size) float tensor normalised for HaMeR, or None for degenerate boxes.
    """
    cx, cy, bw, bh = bbox_cxcywh.tolist()
    if max(bw, bh) < 4:
        return None

    # DEBUG — remove after confirming the fix
    if not getattr(_crop_hand, "_logged", False):
        print(f"[_crop_hand DEBUG] cx={cx:.1f} cy={cy:.1f} bw={bw:.1f} bh={bh:.1f}")
        _crop_hand._logged = True

    # Float [0,1] CHW -> uint8 HWC RGB numpy (ViTDetDataset receives uint8 RGB)
    img_np = (img_chw.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # Compute bbox_size the same way ViTDetDataset does.
    # ViTDetDataset takes [x0,y0,x1,y1] boxes and sets:
    #   scale = rescale_factor * (br - tl) / 200   (shape [2])
    #   bbox_size = expand_to_aspect_ratio(scale*200, BBOX_SHAPE).max()
    # Our bboxes are already rescaled by the dataset, so rescale_factor=1 here.
    scale_xy = np.array([bw, bh], dtype=np.float32) / 200.0
    bbox_size = float(_expand_to_aspect_ratio(scale_xy * 200.0, target_aspect_ratio=bbox_shape).max())

    assert bbox_size >= 4, f"bbox_size={bbox_size:.1f} is unexpectedly small after expand_to_aspect_ratio"

    # Gaussian anti-aliasing (identical formula to ViTDetDataset)
    downsampling_factor = (bbox_size / img_size) / 2.0
    if downsampling_factor > 1.1:
        sigma = (downsampling_factor - 1) / 2
        # Uncomment to debug: print(f"[crop] Gaussian blur sigma={sigma:.2f} (bbox={bbox_size:.0f}px -> {img_size}px)")
        img_np = skimage_gaussian(
            img_np, sigma=sigma, channel_axis=2, preserve_range=True,
        ).astype(np.uint8)

    # Affine warp crop — no flip (handled in prepare_hamer_batch), no augmentation
    img_patch_cv, _ = _generate_image_patch_cv2(
        img_np, cx, cy, bbox_size, bbox_size,
        img_size, img_size,
        do_flip=False, scale=1.0, rot=0,
        border_mode=cv2.BORDER_CONSTANT,
    )

    # HWC uint8 -> CHW float32, normalise with uint8-scale mean/std
    img_patch = img_patch_cv.astype(np.float32).transpose(2, 0, 1)  # CHW
    for c in range(3):
        img_patch[c] = (img_patch[c] - mean_255[c]) / std_255[c]

    result = torch.from_numpy(img_patch)
    assert result.shape == (3, img_size, img_size), (
        f"_crop_hand output shape {tuple(result.shape)} != expected (3, {img_size}, {img_size})"
    )
    assert not torch.isnan(result).any(), "_crop_hand produced NaN values — check mean/std or image input"

    # DEBUG — remove after confirming the fix
    if not getattr(_crop_hand, "_logged_out", False):
        print(f"[_crop_hand DEBUG] crop succeeded → shape={tuple(result.shape)} "
              f"min={result.min():.2f} max={result.max():.2f}")
        _crop_hand._logged_out = True

    return result


def prepare_hamer_batch(imgs_bschw, hand_bboxes_bs24, hand_valid_bs2, device,
                        img_size: int, bbox_shape, mean_255: np.ndarray, std_255: np.ndarray):
    """Extract all valid hand crops from a batch and build a HaMeR input dict.

    Args:
        imgs_bschw:      (B, S, 3, H, W) float [0,1].
        hand_bboxes_bs24:(B, S, 2, 4)   cx/cy/w/h pixel coords.
        hand_valid_bs2:  (B, S, 2)      bool.
        img_size:        HaMeR input resolution (from model_cfg.MODEL.IMAGE_SIZE).
        bbox_shape:      model_cfg.MODEL.BBOX_SHAPE or None.
        mean_255:        (3,) uint8-scale ImageNet mean.
        std_255:         (3,) uint8-scale ImageNet std.

    Returns:
        crops:  (N, 3, img_size, img_size) on `device`.
        rights: (N,) bool  — True if right hand.
        index:  list of (b, s, h) tuples, one per crop (N entries).
    """
    crops, rights, index = [], [], []
    B, S = imgs_bschw.shape[:2]
    H_img, W_img = imgs_bschw.shape[-2], imgs_bschw.shape[-1]
    _logged_first_bbox = False
    for b in range(B):
        for s in range(S):
            for h in range(NUM_HANDS):
                if not hand_valid_bs2[b, s, h]:
                    continue
                # Dataset stores bboxes as normalised [x1, y1, x2, y2]; convert
                # to pixel [cx, cy, w, h] that _crop_hand expects.
                bbox_norm = hand_bboxes_bs24[b, s, h].cpu()
                x1, y1, x2, y2 = bbox_norm.tolist()
                cx = (x1 + x2) / 2 * W_img
                cy = (y1 + y2) / 2 * H_img
                bw = (x2 - x1) * W_img
                bh = (y2 - y1) * H_img
                bbox_pixel = torch.tensor([cx, cy, bw, bh])

                if not _logged_first_bbox:
                    print(
                        f"[prepare_hamer_batch] First valid bbox — "
                        f"norm x1y1x2y2=[{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] → "
                        f"pixel cx/cy/w/h=[{cx:.1f},{cy:.1f},{bw:.1f},{bh:.1f}] "
                        f"(img {W_img}×{H_img})"
                    )
                    # Check format, not size — _crop_hand handles small/degenerate
                    # boxes by returning None. What we want to catch is the bug where
                    # raw pixel coords are passed instead of normalised [0,1] values.
                    assert max(x1, y1, x2, y2) <= 1.0 + 1e-3, (
                        f"Raw bbox values [{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}] look like "
                        f"pixel coords, not normalised [0,1]. The dataset must return "
                        f"normalised bboxes; check _compute_projected_bboxes."
                    )
                    _logged_first_bbox = True

                crop = _crop_hand(
                    imgs_bschw[b, s].cpu(),
                    bbox_pixel,
                    img_size=img_size, bbox_shape=bbox_shape,
                    mean_255=mean_255, std_255=std_255,
                )
                if crop is None:
                    continue
                is_right = (h == 1)
                # HaMeR expects right hands; mirror left-hand crops horizontally
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
def run_hamer_inference(hamer_model, model_cfg, val_loader, mano_model, device, pelvis_ind=0):
    """Run original HaMeR on every valid hand crop in the val set.

    Returns a list of per-batch chunk dicts compatible with metrics_from_chunks.
    """
    # Crop params derived from model_cfg — mirrors ViTDetDataset.__init__
    img_size   = model_cfg.MODEL.IMAGE_SIZE
    bbox_shape = model_cfg.MODEL.get("BBOX_SHAPE", None)
    mean_255   = np.array(model_cfg.MODEL.IMAGE_MEAN, dtype=np.float32) * 255.0
    std_255    = np.array(model_cfg.MODEL.IMAGE_STD,  dtype=np.float32) * 255.0
    print(f"[hamer-baseline] Crop pipeline: img_size={img_size}, bbox_shape={bbox_shape}, "
          f"mean={mean_255.tolist()}, std={std_255.tolist()}")

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
    _batch_idx = 0
    _first_hv_batch_done = False  # have we seen at least one batch with hand_valid=True?
    n_batches = len(val_loader)

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
        crops, rights, index = prepare_hamer_batch(
            imgs, hb, hv, device,
            img_size=img_size, bbox_shape=bbox_shape,
            mean_255=mean_255, std_255=std_255,
        )

        # pred_j_map[b][s][h] = (16, 3) joints tensor; None if invalid
        pred_j_map = [[[None] * NUM_HANDS for _ in range(S)] for _ in range(B)]
        pred_v_map = [[[None] * NUM_HANDS for _ in range(S)] for _ in range(B)]

        if crops is not None and len(crops) > 0:
            hamer_out = hamer_model({"img": crops, "right": rights})

            # HaMeR's MANO.forward re-orders joints into OpenPose convention
            # (see models/hamer/hamer/models/mano_wrapper.py:21,35) via:
            #   mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6,
            #                       18, 10, 11, 12, 19, 7, 8, 9, 20]
            # So pred_keypoints_3d[:, :16] is NOT MANO-order — it's the first
            # 16 entries of OpenPose-order, which contains fingertip vertices
            # at positions 4, 8, 12 and misorders the finger joints. Our GT
            # (smplx MANO via _layer_joints_and_vertices) is in plain MANO
            # order, so taking [:, :16] gives a permutation mismatch that
            # blows up PA-MPJPE / AUC_J (PA-MPVPE / AUC_V stay correct because
            # vertices share the same template). Recover the original MANO
            # order with the inverse permutation:
            _openpose_to_mano = torch.tensor(
                [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3],
                device=hamer_out["pred_keypoints_3d"].device,
            )
            raw_joints = hamer_out["pred_keypoints_3d"][:, _openpose_to_mano, :]  # (N, 16, 3)
            raw_verts  = hamer_out["pred_vertices"]                                # (N, 778, 3)

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

                    # NOTE: we deliberately do NOT subtract the root joint here.
                    # train_hand_head.py's validation logs absolute MPJPE/MPVPE
                    # under hand_metrics/* (no root subtraction — see
                    # scripts/hand_metrics.py:99-120). To make this baseline
                    # comparable on wandb under the same key family, we use the
                    # same absolute convention. HaMeR's pred_keypoints_3d is in
                    # MANO local frame (wrist near origin), so absolute MPJPE
                    # will reflect both the missing global position AND the
                    # frame-mismatch rotation — typically several hundred mm.
                    # PA-MPJPE / PA-MPVPE are translation+rotation invariant
                    # and remain the meaningful pose-quality readout.
                    pred_j_list.append(pj)
                    pred_v_list.append(pv)
                    gt_j_list.append(gj)
                    gt_v_list.append(gv)

                    if valid:
                        total_valid += 1
                    total_hands += 1

        _batch_idx += 1
        batch_hv_count = int(hv.sum().item())

        # After the first batch that contains any hand_valid=True entries,
        # assert that at least one crop succeeded.  If this fires, the bbox
        # conversion pipeline is broken (e.g. normalised coords passed where
        # pixel coords are expected — the bug that caused 0/172704 valid hands).
        if not _first_hv_batch_done and batch_hv_count > 0:
            batch_valid = sum(valid_list)
            assert batch_valid > 0, (
                f"First batch with hand_valid=True ({batch_hv_count} flags set) "
                f"produced 0 successful crops.\n"
                f"  This almost certainly means the bbox format is wrong.\n"
                f"  hand_bboxes must be normalised [x1,y1,x2,y2] in [0,1]; "
                f"prepare_hamer_batch converts them to pixel cx/cy/w/h.\n"
                f"  Check that the conversion W={imgs.shape[-1]}, H={imgs.shape[-2]} "
                f"is applied before _crop_hand."
            )
            print(
                f"[hamer-baseline] Sanity check passed on batch 0 with hand_valid: "
                f"{batch_valid}/{batch_hv_count} crops succeeded."
            )
            _first_hv_batch_done = True

        # Mid-run sanity: after ~5% of batches, warn loudly if still 0 valid.
        if _batch_idx == max(1, n_batches // 20) and total_valid == 0:
            raise RuntimeError(
                f"[hamer-baseline] After {_batch_idx}/{n_batches} batches, "
                f"total_valid=0 (total_hands={total_hands}).\n"
                f"  Aborting early — re-check bbox format and hand_valid flags.\n"
                f"  See prepare_hamer_batch for the normalised→pixel conversion."
            )

        # Sanity-print the frame-mismatch magnitude on the first valid batch.
        # We do NOT subtract the root here (intentional — see note in the loop
        # body), so pred and GT live in different frames. Logging their
        # respective wrist magnitudes makes the frame mismatch visible and
        # confirms that we're getting the expected huge absolute MPJPE.
        if not _root_check_done and any(valid_list):
            stacked_pj = torch.stack(pred_j_list)   # (N, 16, 3)
            stacked_gj = torch.stack(gt_j_list)
            valid_mask = torch.tensor(valid_list)
            if valid_mask.any():
                pred_wrist_norm = stacked_pj[valid_mask, pelvis_ind, :].norm(dim=-1).mean().item()
                gt_wrist_norm   = stacked_gj[valid_mask, pelvis_ind, :].norm(dim=-1).mean().item()
                print(f"[hamer-baseline] Frame-mismatch check (no root subtraction): "
                      f"pred wrist |.| mean={pred_wrist_norm*1000:.1f}mm, "
                      f"gt wrist |.| mean={gt_wrist_norm*1000:.1f}mm "
                      f"(pelvis_ind={pelvis_ind}). "
                      f"Large GT wrist norm + small pred wrist norm = expected "
                      f"frame mismatch → inflated absolute MPJPE.")
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
    parser.add_argument("--hamer-ckpt",  default="models/hamer/hamer/hamer.ckpt",
                        help="Path to the original HaMeR Lightning checkpoint.")
    parser.add_argument("--val-list",    default="outputs/eval_val_split.json",
                        help="Same locked val-split JSON used by eval_hand_head.py.")
    parser.add_argument("--out",         default="outputs/hamer_baseline.json")
    parser.add_argument("--batch-size",  type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-clips", type=int, default=None,
                        help="Evaluate only the first N clips (quick smoke-test).")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="hand-head-training",
                        help="W&B project name. Use --no-wandb to disable logging.")
    parser.add_argument("--wandb-entity",  default="3DV-Project",
                        help="W&B entity / team.")
    parser.add_argument("--wandb-run-name", default="Baseline: HaMeR pretrained",
                        help="W&B run name shown in the runs table.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (writes JSON only).")
    parser.add_argument("--data-root", default=None,
                        help="Override data.data_root from the config — useful when "
                             "running the same config on machines with different paths.")
    args = parser.parse_args()

    # --- Load HaMeR ---
    try:
        # HaMeR's __init__ instantiates pyrender-based visualisation renderers
        # which require an OpenGL/EGL context. We only need inference here, so
        # neuter both renderers before importing load_hamer to avoid the
        # init-time crash on headless cluster nodes.
        from hamer.utils import renderer as _hamer_renderer_mod
        from hamer.utils import mesh_renderer as _hamer_mesh_mod
        def _noop_init(self, *args, **kwargs):
            self.cfg = args[0] if args else kwargs.get("cfg")
        _hamer_renderer_mod.Renderer.__init__   = _noop_init  # SkeletonRenderer wraps this
        _hamer_mesh_mod.MeshRenderer.__init__   = _noop_init
        # SkeletonRenderer (in renderer.py? actually skeleton_renderer.py) — patch too
        try:
            from hamer.utils.skeleton_renderer import SkeletonRenderer as _SkelR
            _SkelR.__init__ = _noop_init
        except ImportError:
            pass
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

    if args.data_root is not None:
        print(f"[hamer-baseline] Overriding data_root: {data_cfg['data_root']} -> {args.data_root}")
        data_cfg["data_root"] = args.data_root

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

    # --- W&B init (early, so a crash mid-run still logs config) ---
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name,
            tags=["baseline", "hamer-pretrained"],
            notes=(
                "Pretrained HaMeR evaluated on the locked Hot3D val split. "
                "MPJPE/MPVPE are absolute camera-frame (no root subtraction), "
                "matching train_hand_head.py's hand_metrics/* convention so this "
                "run is directly comparable with our trained-head runs in wandb. "
                "PA-MPJPE / PA-MPVPE are Procrustes-aligned and meaningful as "
                "pose-quality metrics."
            ),
            config={
                "hamer_ckpt":      str(ckpt_path),
                "config":          str(args.config),
                "val_list":        str(args.val_list),
                "batch_size":      args.batch_size,
                "rescale_factor":  rescale_factor,
                "num_frames":      num_frames,
                "resolution":      res,
                "pelvis_ind":      pelvis_ind,
                "limit_clips":     args.limit_clips,
                "data_cfg":        data_cfg,
                "hand_crop_cfg":   cfg.get("hand_crop", {}),
            },
        )

    # --- Inference ---
    chunks = run_hamer_inference(hamer_model, model_cfg, val_loader, mano_model, args.device, pelvis_ind=pelvis_ind)
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

    # --- Save JSON ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hamer_ckpt":       str(ckpt_path),
        "config":           str(args.config),
        "val_split":        str(args.val_list),
        "num_clips":        len(val_set),
        "num_valid_hands":  result["num_valid_hands"],
        "note":             "absolute camera-frame MPJPE/MPVPE (no root subtraction); PA metrics are Procrustes-aligned",
        "metrics":          {k: result[k] for k in ("left", "right", "all")},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[hamer-baseline] Results written to {out_path}")

    # --- W&B log (single step, mirrors train_hand_head.py validation schema) ---
    if use_wandb:
        log_dict = {"hand_metrics/num_valid_hands": result["num_valid_hands"]}
        for side_label in ("left", "right", "all"):
            side_metrics = result.get(side_label)
            if side_metrics is None:
                continue
            for k, v in side_metrics.items():
                log_dict[f"hand_metrics/{side_label}/{k}"] = v
        wandb.log(log_dict)
        wandb.finish()
        print(f"[hamer-baseline] Logged {len(log_dict)} metrics to W&B "
              f"(run: {args.wandb_run_name}).")


if __name__ == "__main__":
    main()
