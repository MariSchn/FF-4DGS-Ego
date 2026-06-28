"""Fine-tune our hand head on H2O -> the competitive C-MPJPE (the Tier-1 gate).

Zero-shot, our HOT3D head gave C-MPJPE 257 mm on H2O (scale doesn't transfer). This
fine-tunes the hand head (+ a few backbone blocks) on H2O's clean MANO-fit joints,
supervising the predicted camera-frame joints directly with an absolute + root-
relative L2 loss (the exact thing C-MPJPE measures). Reports C-MPJPE / RR / wrist
on a held-out H2O split each eval. GS head off (hand-only, fast).

The hand crop bbox is derived from the GT joints (a detector stand-in, the standard
hand-pose protocol). H2O joints are in METRES (no /1000).

Usage (gb10):
    python -m scripts.train_h2o_hand --config configs/exp_h2o_hand.yaml
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from scripts.hand_vis_utils import MANOModel
from scripts.h2o_dataset import H2ODepthDataset
from scripts.train_hand_head import build_views, compute_joints_from_batch
from scripts.eval_cmpjpe import h2o_gt_joints16, bboxes_from_joints


def build_model(model_cfg, device):
    mc = dict(model_cfg)
    mc["enable_gs"] = False
    mc["hand_to_gs_injection"] = {"enabled": False}
    mc["enable_hand"] = True
    model = WorldMirror(**{k: v for k, v in mc.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    sd = base.get("state_dict", base.get("reconstructor", base)) if isinstance(base, dict) else base
    model.load_state_dict(sd, strict=False)
    ws = model_cfg.get("warm_start_hand_head")
    if ws and os.path.exists(ws):
        wsd = torch.load(ws, map_location=device)
        wsd = wsd.get("model_state_dict", wsd) if isinstance(wsd, dict) else wsd
        tgt = model if any(k.startswith(("hand_head.", "gs_head.")) for k in wsd) else model.hand_head
        miss, unexp = tgt.load_state_dict(wsd, strict=False)
        print(f"Warm-started hand head ({len(miss)} missing, {len(unexp)} unexpected).")
    return model.to(device)


def partial_unfreeze(model, n):
    for p in model.visual_geometry_transformer.parameters():
        p.requires_grad = False
    unfrozen = []
    if n > 0:
        vgt = model.visual_geometry_transformer
        for blocks in (getattr(vgt, "frame_blocks", []), getattr(vgt, "global_blocks", [])):
            for blk in list(blocks)[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
                    unfrozen.append(p)
        model._backbone_trainable = True
    return unfrozen


def _metrics(pj, gj):
    """pj,gj [n,16,3] m -> (cmpjpe, rr, wrist) in mm."""
    cmp = (1000 * torch.sqrt(((pj - gj) ** 2).sum(-1)).mean()).item()
    rr = (1000 * torch.sqrt((((pj - pj[:, :1]) - (gj - gj[:, :1])) ** 2).sum(-1)).mean()).item()
    wr = (1000 * torch.sqrt(((pj[:, :1] - gj[:, :1]) ** 2).sum(-1)).mean()).item()
    return cmp, rr, wr


@torch.no_grad()
def evaluate(model, loader, mano, device, nf, max_batches):
    model.eval()
    C, R, W = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        gt_j, valid = h2o_gt_joints16(batch["joints"])
        hb = bboxes_from_joints(gt_j, batch["K"][0]).to(device)
        hc = batch["hand_crops"].to(device) if "hand_crops" in batch else None
        views = build_views(batch["img"].to(device), nf, device, hb, valid.to(device),
                            hand_crops=hc)
        preds = model(views, is_inference=False, use_motion=False)
        pj = compute_joints_from_batch(preds["hand_joints"], mano, device)
        gt_j = gt_j.to(device)
        sel = valid.to(device) > 0.5
        if sel.sum() == 0:
            continue
        c, r, w = _metrics(pj[sel], gt_j[sel])
        C.append(c); R.append(r); W.append(w)
    model.train()
    f = lambda x: float(np.mean(x)) if x else 999.0
    return f(C), f(R), f(W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_h2o_hand.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    data_cfg, model_cfg, tr = cfg["data"], cfg["model"], cfg["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wb = None
    if os.environ.get("WANDB_PROJECT"):
        try:
            import wandb
            wb = wandb.init(project=os.environ["WANDB_PROJECT"],
                            name=os.environ.get("WANDB_NAME", "h2o-hand"), config=cfg)
            print(f"[wandb] logging to project {os.environ['WANDB_PROJECT']}", flush=True)
        except Exception as e:
            print(f"[wandb] disabled ({type(e).__name__}: {e}); stdout-only", flush=True)
            wb = None

    model = build_model(model_cfg, device)
    unfrozen = partial_unfreeze(model, int(tr.get("unfreeze_last_n_blocks", 4)))
    hand_params = list(model.hand_head.parameters())
    trainable = hand_params + unfrozen
    print(f"Trainable: hand={sum(p.numel() for p in hand_params):,} "
          f"backbone={sum(p.numel() for p in unfrozen):,}")
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])

    npz = sorted(glob.glob(os.path.join(data_cfg["data_root"], "*.npz")))
    # H2O standard protocol: cross-subject split (test = one held-out subject).
    # Falls back to a within-subject split if only one subject is packed.
    test_subj = data_cfg.get("test_subject", "subject4")
    val_npz = [f for f in npz if test_subj in os.path.basename(f)]
    train_npz = [f for f in npz if test_subj not in os.path.basename(f)]
    if not val_npz or not train_npz:
        print(f"[warn] cross-subject split unavailable (test={test_subj}); within-subject fallback")
        n_val = max(1, int(len(npz) * float(data_cfg.get("val_split", 0.2))))
        val_npz, train_npz = npz[:n_val], npz[n_val:] or npz[:1]
    nf = data_cfg["num_frames"]
    hires = bool(model_cfg.get("hires_hand", False))
    hc_px = int(model_cfg.get("hires_hand_kwargs", {}).get("crop_px", 256))
    ds_kw = dict(num_frames=nf, res=224, clip_stride=data_cfg.get("clip_stride", 8),
                 with_hand=True, hires_hand=hires, hand_crop_px=hc_px)
    train_set, val_set = H2ODepthDataset(train_npz, **ds_kw), H2ODepthDataset(val_npz, **ds_kw)
    print(f"H2O seqs: {len(train_npz)} train / {len(val_npz)} val | "
          f"clips: {len(train_set)} / {len(val_set)}")
    bs = tr.get("batch_size", 2)
    nw = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=nw)

    w_abs, w_rr = float(tr.get("w_abs", 1.0)), float(tr.get("w_rr", 1.0))
    use_amp = tr.get("use_amp", True)
    ga = tr.get("grad_accum_steps", 4)
    gc = float(tr.get("grad_clip_norm", 5.0))
    opt = Adam(trainable, lr=float(tr["lr"]))
    max_steps = int(tr.get("max_steps", 0))
    sched = None
    if max_steps > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=max_steps, eta_min=float(tr.get("lr_min", 1e-6)))
        print(f"Cosine LR: {float(tr['lr']):.1e} -> {float(tr.get('lr_min', 1e-6)):.1e} over {max_steps} steps")
    out_dir = tr["output_dir"]; os.makedirs(out_dir, exist_ok=True)

    c0, r0, w0 = evaluate(model, val_loader, mano, device, nf, tr.get("val_max_batches", 30))
    print(f"[step 0] BASELINE  C-MPJPE={c0:.1f}  RR={r0:.1f}  WRIST={w0:.1f} mm", flush=True)
    best = c0
    gstep = 0
    model.train()
    for epoch in range(tr.get("epochs", 5)):
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            gt_j, valid = h2o_gt_joints16(batch["joints"])
            hb = bboxes_from_joints(gt_j, batch["K"][0]).to(device)
            hc = batch["hand_crops"].to(device) if "hand_crops" in batch else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                views = build_views(batch["img"].to(device), nf, device, hb, valid.to(device),
                                    hand_crops=hc)
                preds = model(views, is_inference=False, use_motion=False)
                pj = compute_joints_from_batch(preds["hand_joints"], mano, device)
                gt_j = gt_j.to(device)
                sel = valid.to(device) > 0.5
                if sel.sum() == 0:
                    continue
                p, g = pj[sel], gt_j[sel]
                # eps inside sqrt: the root-relative wrist is always 0 -> sqrt(0)
                # has an inf backward -> NaN grad. eps makes it finite everywhere.
                abs_e = torch.sqrt(((p - g) ** 2).sum(-1) + 1e-6).mean()
                rr_e = torch.sqrt((((p - p[:, :1]) - (g - g[:, :1])) ** 2).sum(-1) + 1e-6).mean()
                loss = w_abs * abs_e + w_rr * rr_e
            (loss / ga).backward()
            if (bi + 1) % ga == 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable, gc)
                if torch.isfinite(gn):
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if sched is not None:
                    sched.step()
                gstep += 1
                if gstep % tr.get("log_every", 10) == 0:
                    print(f"[step {gstep}] loss={loss.item():.4f} "
                          f"abs={abs_e.item()*1000:.1f} rr={rr_e.item()*1000:.1f}mm "
                          f"grad={gn:.1f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
                    if wb is not None:
                        wb.log({"train/loss": loss.item(), "train/abs_mm": abs_e.item() * 1000,
                                "train/rr_mm": rr_e.item() * 1000, "train/grad": float(gn),
                                "train/lr": opt.param_groups[0]["lr"]}, step=gstep)
                if gstep % tr.get("val_every", 50) == 0:
                    c, r, w = evaluate(model, val_loader, mano, device, nf, tr.get("val_max_batches", 30))
                    tag = " <-- best" if c < best else ""
                    print(f"[step {gstep}] VAL  C-MPJPE={c:.1f}  RR={r:.1f}  WRIST={w:.1f} mm "
                          f"(base {c0:.1f}){tag}", flush=True)
                    if wb is not None:
                        wb.log({"val/C-MPJPE": c, "val/RR": r, "val/WRIST": w}, step=gstep)
                    if c < best:
                        best = c
                        torch.save({"model_state_dict": model.state_dict(), "global_step": gstep,
                                    "cmpjpe": c}, os.path.join(out_dir, "best_cmpjpe.pt"))
                if max_steps and gstep >= max_steps:
                    break
        if max_steps and gstep >= max_steps:
            break
    print(f"\n=== DONE. best C-MPJPE {best:.1f} mm vs baseline {c0:.1f} mm "
          f"(HaWoR 51.8 / Hand3R 42.6) ===", flush=True)


if __name__ == "__main__":
    main()
