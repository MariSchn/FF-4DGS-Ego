"""C-MPJPE eval on H2O: camera-frame absolute hand-joint error (the SOTA metric).

C-MPJPE (Hand3R / HaWoR on HOI4D) = absolute 3D joint error in the CAMERA frame,
NO alignment, in mm. This runs our hand head on packed H2O (scripts.pack_h2o) and
compares to H2O's high-quality MANO-fit joints, reporting C-MPJPE alongside the
root-relative (RR), Procrustes (PA) and wrist-only errors.

H2O hand_pose is 21 joints/hand in H2O order; our smplx hand head outputs 16
(MANO, no fingertips). We remap H2O->MANO order and take the 16 shared kinematic
joints. H2O depth/joints/translations are in mm (-> /1000 = m).

!!! VERIFY against the first packed sequence (project GT joints onto the RGB and
eyeball them on the hand) before trusting the absolute number:
  (1) the H2O->MANO joint remap + the 16-joint subset,
  (2) the K crop/resize adjustment (pack center-square-cropped 1280x720->720->224),
  (3) that our head's predicted joints are in the same camera frame as H2O's.

Usage (gb10 / venv_gb10):
    python -m scripts.eval_cmpjpe --config configs/exp_p3_gtdepth_unfreeze.yaml \
        --ckpt /work/courses/3dv/team25/checkpoints/default/hand_head_final.pt \
        --h2o /work/scratch/dmonopoli/h2o_packed --limit 200
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from scripts.hand_vis_utils import MANOModel
from scripts.h2o_dataset import H2ODepthDataset
from scripts.train_hand_head import build_views, compute_joints_from_batch

# H2O 21-joint order -> MANO 21-joint order (taeinkwon/h2o + ap229997/hands loader).
H2O_TO_MANO = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
# MANO 21 -> our 16 kinematic joints (drop the 5 fingertips 4,8,12,16,20).
MANO21_TO_16 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
PACK_RES = 224
H2O_FULL_W, H2O_FULL_H = 1280, 720


def h2o_gt_joints16(j128: torch.Tensor):
    """[...,128] H2O hand_pose -> (joints[...,2,16,3] m, valid[...,2]). Camera frame."""
    left_valid, right_valid = j128[..., 0], j128[..., 64]
    left = j128[..., 1:64].reshape(*j128.shape[:-1], 21, 3)
    right = j128[..., 65:128].reshape(*j128.shape[:-1], 21, 3)
    idx = [H2O_TO_MANO[k] for k in MANO21_TO_16]
    # H2O hand_pose joints are already in METRES (wrist ~0.5 m), NOT mm — verified
    # by debug_h2o_joints (project lands 21/21 on the hand). Do NOT divide by 1000.
    left16 = left[..., idx, :]
    right16 = right[..., idx, :]
    joints = torch.stack([left16, right16], dim=-3)              # [...,2,16,3]
    valid = torch.stack([left_valid, right_valid], dim=-1)       # [...,2]
    return joints, valid


# GT fingertip H2O-indices for the manopth-21 tip positions (4,8,12,16,20), i.e. the
# 5 fingertips appended AFTER the 16 base joints. Order MUST match the pred tip order
# (thumb,index,middle,ring,pinky from MANOModel.get_joints21_batched) — verified by
# projecting both onto the RGB (--viz) before trusting the 21-joint number.
GT_TIP_IDX = [H2O_TO_MANO[k] for k in (4, 8, 12, 16, 20)]   # -> [9, 18, 15, 4, 20]


def h2o_gt_joints21(j128: torch.Tensor):
    """[...,128] H2O hand_pose -> (joints[...,2,21,3] m, valid[...,2]). 16 base + 5 tips."""
    left = j128[..., 1:64].reshape(*j128.shape[:-1], 21, 3)
    right = j128[..., 65:128].reshape(*j128.shape[:-1], 21, 3)
    idx = [H2O_TO_MANO[k] for k in MANO21_TO_16] + GT_TIP_IDX     # 21 H2O indices
    joints = torch.stack([left[..., idx, :], right[..., idx, :]], dim=-3)
    valid = torch.stack([j128[..., 0], j128[..., 64]], dim=-1)
    return joints, valid


def bboxes_from_joints(gt_joints, K, res=PACK_RES):
    """Project GT 3D joints (cam frame, m) -> normalized [x1,y1,x2,y2] bbox per hand.

    K is the ORIGINAL H2O intrinsics [fx,fy,cx,cy,...]; adjust for the pack's
    center-square-crop (1280x720->720) + resize (720->res). Used only to crop —
    a stand-in for a hand detector (here, derived from GT).
    """
    B, S, H, J, _ = gt_joints.shape
    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    x0 = (H2O_FULL_W - H2O_FULL_H) // 2
    s = res / float(H2O_FULL_H)
    fx, fy, cx, cy = fx * s, fy * s, (cx - x0) * s, cy * s
    z = gt_joints[..., 2].clamp(min=1e-3)
    u = fx * gt_joints[..., 0] / z + cx
    v = fy * gt_joints[..., 1] / z + cy
    out = torch.zeros(B, S, H, 4)
    for b in range(B):
        for sf in range(S):
            for h in range(H):
                uu, vv = u[b, sf, h], v[b, sf, h]
                x1, x2 = uu.min().item(), uu.max().item()
                y1, y2 = vv.min().item(), vv.max().item()
                cxb, cyb, w, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.6, (y2 - y1) * 1.6
                out[b, sf, h] = torch.tensor([(cxb - w / 2) / res, (cyb - hh / 2) / res,
                                              (cxb + w / 2) / res, (cyb + hh / 2) / res])
    return out


def _per_joint_mm(pred, gt):
    return 1000.0 * torch.sqrt(((pred - gt) ** 2).sum(-1))


def _pa_mpjpe_mm(pred, gt):
    """PA-MPJPE: per-sample similarity-align (scale+R+t) pred->gt, then joint error.

    pred, gt: [n,J,3] (m). Returns [n] per-sample mean joint error in mm. This is the
    Procrustes-aligned metric H2O SOTA reports (e.g. EgoGrasp ~47 mm) — it removes the
    global similarity transform, so it isolates pose/shape quality from absolute scale.
    Umeyama solution (same as HMR/SPIN compute_similarity_transform), batched in f64.
    Runs on CPU: the solve is tiny ([n,3,3]) and the gb10 CUDA build fails to JIT the
    fused sign(det(...)) kernel.
    """
    p, g = pred.double().cpu(), gt.double().cpu()
    mu_p, mu_g = p.mean(1, keepdim=True), g.mean(1, keepdim=True)
    x1, x2 = p - mu_p, g - mu_g
    var1 = (x1 ** 2).sum((1, 2)).clamp(min=1e-12)                 # [n]
    k = x1.transpose(1, 2) @ x2                                   # [n,3,3]
    u, s, vh = torch.linalg.svd(k)
    v = vh.transpose(1, 2)
    z = torch.eye(3, dtype=p.dtype, device=p.device).repeat(len(p), 1, 1)
    z[:, 2, 2] = torch.sign(torch.linalg.det(u @ v.transpose(1, 2)))
    r = v @ z @ u.transpose(1, 2)                                 # [n,3,3]
    scale = torch.diagonal(r @ k, dim1=1, dim2=2).sum(1) / var1   # [n]
    aligned = scale.view(-1, 1, 1) * (x1 @ r.transpose(1, 2)) + mu_g
    return (1000.0 * torch.sqrt(((aligned - g) ** 2).sum(-1))).mean(-1)


def compute_joints21_from_batch(params, mano_model, device):
    """[B,S,64] MANO params -> [B,S,2,21,3] joints (16 base + 5 fingertips)."""
    B, S, _ = params.shape
    flat = params.view(B * S, 2, 32)
    left = mano_model.get_joints21_batched(flat[:, 0], is_right=False, device=device)
    right = mano_model.get_joints21_batched(flat[:, 1], is_right=True, device=device)
    return torch.stack([left, right], dim=1).view(B, S, 2, 21, 3)


def _save_viz(img_chw, K, pred_h, gt_h, path, res=PACK_RES):
    """Project pred (red) + GT (green) 21 joints of ONE hand onto the RGB; link them.

    Tips (idx 16-20) are ringed blue. A correct pairing keeps each red dot next to its
    green dot; a mis-ordered fingertip shows as a long yellow line crossing to another
    finger — the visual check that the pred/GT tip order actually corresponds.
    """
    from PIL import Image, ImageDraw
    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    x0 = (H2O_FULL_W - H2O_FULL_H) // 2
    s = res / float(H2O_FULL_H)
    fx, fy, cx, cy = fx * s, fy * s, (cx - x0) * s, cy * s
    arr = img_chw.detach().cpu().float().numpy().transpose(1, 2, 0)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    im = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)

    def proj(j):
        z = max(float(j[2]), 1e-3)
        return fx * float(j[0]) / z + cx, fy * float(j[1]) / z + cy

    for k in range(21):
        pu, pv = proj(pred_h[k])
        gu, gv = proj(gt_h[k])
        d.line([pu, pv, gu, gv], fill=(255, 255, 0), width=1)
        d.ellipse([gu - 2, gv - 2, gu + 2, gv + 2], fill=(0, 255, 0))
        d.ellipse([pu - 2, pv - 2, pu + 2, pv + 2], fill=(255, 0, 0))
        if k >= 16:
            d.ellipse([gu - 4, gv - 4, gu + 4, gv + 4], outline=(0, 0, 255), width=2)
    im.save(path)
    print(f"  [viz] saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_p3_gtdepth_unfreeze.yaml")
    ap.add_argument("--ckpt", required=True, help="hand-head checkpoint")
    ap.add_argument("--h2o", required=True, help="packed H2O .npz dir")
    ap.add_argument("--subject", default=None,
                    help="filter to one subject substring (e.g. subject4) for the held-out split")
    ap.add_argument("--joints21", action="store_true",
                    help="eval on 21 joints (16 base + 5 fingertips), the H2O/SOTA convention")
    ap.add_argument("--hires", action="store_true",
                    help="feed native hand_crops to the hires-hand branch (set if the ckpt "
                         "was trained with model.hires_hand=true and the npz has hand_crop_L/R)")
    ap.add_argument("--viz", default=None,
                    help="save a pred/GT projection PNG of the first valid hand here (verify tip pairing)")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mc = dict(cfg["model"])
    mc["enable_gs"] = False                      # hand-only eval -> skip GS head (much faster)
    mc["hand_to_gs_injection"] = {"enabled": False}
    model = WorldMirror(**{k: v for k, v in mc.items() if k != "checkpoint"})
    base = torch.load(mc["checkpoint"], map_location=device)
    sd = base.get("state_dict", base.get("reconstructor", base)) if isinstance(base, dict) else base
    model.load_state_dict(sd, strict=False)
    ck = torch.load(args.ckpt, map_location=device)
    hsd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    tgt = model if any(k.startswith(("hand_head.", "gs_head.")) for k in hsd) else model.hand_head
    tgt.load_state_dict(hsd, strict=False)
    model.to(device).eval()

    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])
    nf = cfg["data"]["num_frames"]
    npz = sorted(glob.glob(os.path.join(args.h2o, "*.npz")))
    if args.subject:
        npz = [f for f in npz if args.subject in os.path.basename(f)]
        print(f"Filtered to {args.subject}: {len(npz)} seqs (held-out split)")
    hc_px = int(mc.get("hires_hand_kwargs", {}).get("crop_px", 256))
    ds = H2ODepthDataset(npz, num_frames=nf, res=PACK_RES, clip_stride=nf, with_hand=True,
                         hires_hand=args.hires, hand_crop_px=hc_px)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"H2O clips: {len(ds)}")

    nj = 21 if args.joints21 else 16
    gt_fn = h2o_gt_joints21 if args.joints21 else h2o_gt_joints16
    pred_fn = compute_joints21_from_batch if args.joints21 else compute_joints_from_batch
    print(f"Evaluating on {nj} joints" + (" (16 base + 5 fingertips)" if args.joints21 else ""))

    cmp, rr, wrist, pa, base_e, tip_e = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(cmp) >= args.limit or "joints" not in batch:
                break
            gt_j, gt_valid = gt_fn(batch["joints"])                # [1,S,2,nj,3] m
            # K: read from the npz of this clip (the dataset doesn't carry it; reload)
            # fall back to full-frame bbox if K unavailable.
            f = ds.clips[i][0]
            K = np.load(f)["K"]
            imgs = batch["img"].to(device)
            hb = bboxes_from_joints(gt_j, K).to(device) if K.size >= 4 else None
            hv = gt_valid.to(device)
            hc = batch["hand_crops"].to(device) if "hand_crops" in batch else None
            views = build_views(imgs, nf, device, hb, hv, hand_crops=hc)
            preds = model(views, is_inference=False, use_motion=False)
            pred_j = pred_fn(preds["hand_joints"], mano_model, device)  # [1,S,2,nj,3] m
            gt_j = gt_j.to(device)
            sel = (gt_valid.to(device) > 0.5)              # [1,S,2]
            if sel.sum() == 0:
                continue
            if args.viz and not pa:        # first valid clip only
                bs, ss, hh = torch.nonzero(sel, as_tuple=True)
                _save_viz(imgs[bs[0], ss[0]], K, pred_j[bs[0], ss[0], hh[0]],
                          gt_j[bs[0], ss[0], hh[0]], args.viz)
            pj = pred_j[sel]              # [n,nj,3]
            gj = gt_j[sel]
            cmp.append(_per_joint_mm(pj, gj).mean().item())
            rr.append(_per_joint_mm(pj - pj[:, :1], gj - gj[:, :1]).mean().item())
            wrist.append(_per_joint_mm(pj[:, :1], gj[:, :1]).mean().item())
            pa.append(_pa_mpjpe_mm(pj, gj).mean().item())
            if args.joints21:                    # tip-pairing sanity: base(0:16) vs tips(16:21)
                pjr, gjr = pj - pj[:, :1], gj - gj[:, :1]   # root-relative isolates shape/pairing
                base_e.append(_per_joint_mm(pjr[:, :16], gjr[:, :16]).mean().item())
                tip_e.append(_per_joint_mm(pjr[:, 16:], gjr[:, 16:]).mean().item())

    if not cmp:
        raise SystemExit("No valid hands evaluated (check joint remap / valid flags).")
    print(f"\n========== H2O hand metrics (camera frame, {nj} joints) ==========")
    print(f"  clips:        {len(cmp)}")
    print(f"  C-MPJPE:      {np.mean(cmp):.2f} mm   (absolute, camera frame -- vs HaWoR/Hand3R)")
    print(f"  PA-MPJPE:     {np.mean(pa):.2f} mm   (Procrustes -- vs H2O SOTA / EgoGrasp ~47)")
    print(f"  RR-MPJPE:     {np.mean(rr):.2f} mm   (root-relative, shape)")
    print(f"  WRIST:        {np.mean(wrist):.2f} mm   (absolute root)")
    if args.joints21 and tip_e:
        print(f"  [pairing] base(0:16) RR={np.mean(base_e):.2f} mm  vs  tips(16:21) RR={np.mean(tip_e):.2f} mm")
        print(f"            tips/base ratio={np.mean(tip_e)/max(np.mean(base_e),1e-6):.2f}x "
              f"(~1.5-2x = correct pairing; >4x = mis-permuted tips)")
    print("\nVERIFY the joint remap + K adjustment before quoting the absolute number.")


if __name__ == "__main__":
    main()
