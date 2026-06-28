"""Train gs_head (+ a few backbone blocks) on HOI4D DENSE metric depth.

The core CVPR-pivot experiment: instead of HOT3D's sparse, masked object-depth
supervision, supervise the predicted scene depth on a DENSE GT metric depth map
(HOI4D's sensor depth). Question: does the predicted depth become metric/accurate
(object_depth_residual drops) when supervised densely? No hand head here — this
isolates "does dense supervision make the scene metric". The hand branch + the
H2O metric-hand story layer on after this validates.

Usage (gb10 / venv_gb10):
    python -m scripts.train_hoi4d_depth --config configs/exp_hoi4d_depth.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from scripts.train_hand_head import build_views
from scripts.object_depth_loss import object_depth_loss
from scripts.hoi4d_depth_dataset import HOI4DDepthDataset, discover_hoi4d_seqs


def build_model(model_cfg: dict, device: str) -> WorldMirror:
    """Backbone + GS head only (hand head / injection disabled), base ckpt loaded."""
    mc = dict(model_cfg)
    mc["enable_hand"] = False
    mc["use_hand_crop"] = False
    mc["hand_to_gs_injection"] = {"enabled": False}
    model = WorldMirror(**{k: v for k, v in mc.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    sd = base.get("state_dict", base.get("reconstructor", base)) if isinstance(base, dict) else base
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded base ckpt (missing={len(missing)}, unexpected={len(unexpected)}).")
    return model.to(device)


def partial_unfreeze(model: WorldMirror, n: int) -> list:
    """Unfreeze the last n frame+global blocks; return the unfrozen params."""
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


@torch.no_grad()
def eval_depth(model, loader, device, num_frames, max_batches, obj_kw):
    """Mean object/scene depth residual (metres) over a few val clips."""
    model.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        imgs = batch["img"].to(device)
        views = build_views(imgs, num_frames, device, None, None)
        preds = model(views, is_inference=False, use_motion=False)
        gsd = preds.get("gs_depth")
        if gsd is None:
            continue
        _, info = object_depth_loss(gsd, batch["gt_obj_depth"].to(device),
                                    batch["gt_obj_mask"].to(device), **obj_kw)
        if info["n_valid"] > 0:
            tot += info["obj_depth_residual_m"]
            n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_hoi4d_depth.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg, model_cfg, tr = cfg["data"], cfg["model"], cfg["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(model_cfg, device)
    if not model_cfg.get("enable_gs", False):
        raise SystemExit("enable_gs must be true.")
    unfrozen = partial_unfreeze(model, int(tr.get("unfreeze_last_n_blocks", 4)))
    gs_params = list(model.gs_head.parameters())
    trainable = gs_params + unfrozen
    print(f"Trainable: gs_head={sum(p.numel() for p in gs_params):,} "
          f"backbone(unfrozen)={sum(p.numel() for p in unfrozen):,}")

    # data
    seqs = discover_hoi4d_seqs(data_cfg["data_root"])
    if not seqs:
        raise SystemExit(f"No HOI4D seqs (images/+raw_depth/) under {data_cfg['data_root']}")
    n_val = max(1, int(len(seqs) * float(data_cfg.get("val_split", 0.2))))
    val_seqs, train_seqs = seqs[:n_val], seqs[n_val:] or seqs[:1]
    num_frames = data_cfg["num_frames"]
    res = data_cfg["resolution"][0]
    depth_res = int(cfg.get("obj_depth", {}).get("render_res", res))
    ds_kw = dict(num_frames=num_frames, res=res, clip_stride=data_cfg.get("clip_stride", 8),
                 depth_res=depth_res)
    train_set = HOI4DDepthDataset(train_seqs, **ds_kw)
    val_set = HOI4DDepthDataset(val_seqs, **ds_kw)
    print(f"Seqs: {len(train_seqs)} train / {len(val_seqs)} val | "
          f"clips: {len(train_set)} train / {len(val_set)} val")
    nw = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=tr.get("batch_size", 2), shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=tr.get("batch_size", 2), shuffle=False,
                            num_workers=nw, pin_memory=True)

    obj_cfg = cfg.get("obj_depth", {})
    obj_kw = dict(margin=float(obj_cfg.get("margin", 0.05)),
                  depth_min=float(obj_cfg.get("depth_min", 0.05)),
                  depth_max=float(obj_cfg.get("depth_max", 10.0)))

    use_amp = tr.get("use_amp", True)
    grad_accum = tr.get("grad_accum_steps", 8)
    grad_clip = float(tr.get("grad_clip_norm", 5.0))
    optimizer = Adam(trainable, lr=float(tr["lr"]))
    out_dir = tr["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    base_err = eval_depth(model, val_loader, device, num_frames, tr.get("val_max_batches", 12), obj_kw)
    print(f"[step 0] BASELINE depth residual (frozen backbone): {base_err*100:.2f} cm", flush=True)
    best = base_err
    global_step = 0
    model.train()
    for epoch in range(tr.get("epochs", 1)):
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            imgs = batch["img"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                views = build_views(imgs, num_frames, device, None, None)
                preds = model(views, is_inference=False, use_motion=False)
                loss, info = object_depth_loss(preds["gs_depth"],
                                               batch["gt_obj_depth"].to(device),
                                               batch["gt_obj_mask"].to(device), **obj_kw)
            (loss / grad_accum).backward()
            if (bi + 1) % grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                if torch.isfinite(gn):
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % tr.get("log_every", 10) == 0:
                    print(f"[step {global_step}] loss={loss.item():.4f} "
                          f"resid={info['obj_depth_residual_m']*100:.2f}cm "
                          f"n_valid={info['n_valid']} grad={gn:.2f}", flush=True)
                if global_step % tr.get("val_every", 50) == 0:
                    err = eval_depth(model, val_loader, device, num_frames,
                                     tr.get("val_max_batches", 12), obj_kw)
                    tag = "  <-- new best" if err < best else ""
                    print(f"[step {global_step}] VAL depth residual: {err*100:.2f} cm "
                          f"(baseline {base_err*100:.2f}){tag}", flush=True)
                    if err < best:
                        best = err
                        torch.save({"model_state_dict": model.state_dict(),
                                    "global_step": global_step, "val_residual_cm": err * 100},
                                   os.path.join(out_dir, "best_depth.pt"))
    print(f"\n=== DONE. best depth residual {best*100:.2f} cm "
          f"vs baseline {base_err*100:.2f} cm ({100*(base_err-best)/base_err:.1f}% better) ===")


if __name__ == "__main__":
    main()
