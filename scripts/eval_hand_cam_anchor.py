#!/usr/bin/env python
"""Camera-frame C-MPJPE / C-abs eval WITH the contact anchor, no extrinsics needed.

The world-space (W) eval needs per-frame camera extrinsics, which the HOI4D mirror
lacks. But the contact anchor corrects CAMERA-frame root depth, scored by C-abs,
which needs only the camera-frame GT cache (gt_joints_cache_cam_v2.pt). This script
runs the model per clip, applies the SAME apply_root_anchor as the world eval (via
predict_clip), and reports C-MPJPE (root-rel) + C-abs + the anchor gate / |dz| / Δgs
diagnostics. Run it twice (model.enable_root_anchor true vs false) for the anchor A/B.

Usage (gb10 node):
    python -u -m scripts.eval_hand_cam_anchor \
      --config /tmp/eval_contact_hoi4d.yaml \
      --data_root /work/scratch/dmonopoli/hoi4d \
      --max_seqs 11 --clip_len 16 --stride 8 --out /tmp/hoi4d_cam_eval.json
"""
import argparse
import json
import os
import traceback

import torch
import yaml

from scripts.eval_world_space import build_model, predict_clip
from scripts.world_space_metrics import c_mpjpe

RH = 1  # HOI4D releases the right hand -> hand index 1


def _has_cam_cache(seq_dir):
    return os.path.exists(os.path.join(seq_dir, "hand_data", "gt_joints_cache_cam_v2.pt"))


def eval_seq_cam(model, mano_model, device, seq_dir, cfg, clip_len, stride, max_clips=0,
                 contact_gate="proxy", da3_wrist_cache_dir=None, contact_cache_dir=None,
                 bbox_perturb=None, dino=None):
    """Run all clips of one sequence, apply the anchor, return camera-frame metrics.
    contact_gate: 'oracle' gates the anchor by the cached GT contact mask (mechanism
    ceiling); 'proxy' uses the module's |disagree|<band_m gate; 'off' handled by the
    caller disabling enable_root_anchor."""
    from scripts.train_hand_head import HOT3DHandDataset, build_views

    mcfg = cfg["model"]
    rescale = cfg.get("hand_crop", {}).get("rescale_factor", 1.5)
    res = int(cfg.get("data", {}).get("resolution", [224, 224])[0])
    ds = HOT3DHandDataset([seq_dir], mano_model, num_frames=clip_len, clip_stride=stride,
                          use_hand_crop=mcfg.get("use_hand_crop", False), rescale_factor=rescale,
                          bbox_perturb=bbox_perturb)
    if len(ds) == 0:
        return None
    hd = os.path.join(seq_dir, "hand_data")
    gt_cam = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), map_location="cpu").float()  # [N,2,16,3]
    cam_intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)
    bb = torch.load(os.path.join(hd, f"hand_bboxes_v2_rf{rescale}_res{res}x{res}.pt"), map_location="cpu")
    gt_valid = bb["valid"].bool()                          # [N,2]
    # C1: DA3 metric wrist-depth reference for this seq (external anchor target,
    # replaces gs_depth). Keyed by seq basename, from the off-cluster cache dir.
    seq_da3 = None
    if da3_wrist_cache_dir is not None:
        _da3p = os.path.join(da3_wrist_cache_dir, f"{os.path.basename(seq_dir)}_da3_wrist.pt")
        if os.path.exists(_da3p):
            seq_da3 = torch.load(_da3p, map_location="cpu", weights_only=True).float()  # [N,2] m
    # oracle contact gate: per-frame, per-hand GT contact mask (scripts.build_contact_cache).
    contact_seq = None
    if contact_gate == "oracle":
        if contact_cache_dir is not None:
            cpath = os.path.join(contact_cache_dir, f"{os.path.basename(seq_dir)}_contact.pt")
        else:
            cpath = os.path.join(hd, "contact_cache.pt")
        contact_seq = torch.load(cpath, map_location="cpu").bool() if os.path.exists(cpath) else None

    n_clips = len(ds) if max_clips <= 0 else min(len(ds), max_clips)
    anchor_log = []
    pcf = {}                                               # dedupe overlapping clip frames
    for j in range(n_clips):
        batch = ds[j]
        imgs = batch["img"].unsqueeze(0).to(device)
        hb = batch["hand_bboxes"].unsqueeze(0).to(device) if "hand_bboxes" in batch else None
        hv = batch["hand_valid"].unsqueeze(0).to(device) if "hand_valid" in batch else None
        if dino is not None:
            # Backbone-swap ablation (DINO arm): the head was trained on frozen
            # DINOv2 patch tokens (build_feature_cache_dino), so the eval must feed
            # it the same tokens — head-only forward, identical normalization.
            from scripts.train_hand_head import forward_hand_cached
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feats = dino.forward_features((imgs[0] - mean) / std)
                tokens = feats["x_norm_patchtokens"].unsqueeze(0)   # [1, S, P, 1024]
            preds = forward_hand_cached(model, tokens, imgs, hb, hv)
        else:
            views = build_views(imgs, clip_len, device, hb, hv)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(views, is_inference=True, use_motion=False)
        # predict_clip lifts joints and applies the anchor (when model.enable_root_anchor)
        # exactly as the world eval does; we use only its camera-frame joints here.
        start = j * stride
        contact_clip = None
        if contact_seq is not None:
            sl = contact_seq[start:start + clip_len]
            if sl.shape[0] == clip_len:
                contact_clip = sl.unsqueeze(0).to(device)   # [1, S, 2]
        ref_clip = None
        if seq_da3 is not None:
            dl = seq_da3[start:start + clip_len]
            if dl.shape[0] == clip_len:
                ref_clip = dl.unsqueeze(0).to(device)       # [1, S, 2] DA3 metric wrist depth
        pj = predict_clip(preds, mano_model, device, cam_intr, model=model,
                          anchor_log=anchor_log, contact_mask=contact_clip,
                          ref_d_scene=ref_clip)[0]
        for kk in range(pj.shape[0]):
            f = start + kk
            if f not in pcf and f < gt_cam.shape[0]:
                pcf[f] = pj[kk, RH]
    fr = sorted(pcf)
    if not fr:
        return None
    pc_cam = torch.stack([pcf[f] for f in fr])             # [T,16,3]
    gc_cam = gt_cam[fr, RH]                                 # [T,16,3]
    vc = gt_valid[fr, RH].unsqueeze(-1).expand(-1, 16)     # [T,16]
    werr = (pc_cam[:, 0] - gc_cam[:, 0]).norm(dim=-1)      # [T] wrist/root placement (m)
    wsel = werr[vc[:, 0].bool() & torch.isfinite(werr)]
    row = {
        "seq": os.path.basename(seq_dir),
        "frames": len(fr),
        "C_MPJPE": float(c_mpjpe(pc_cam, gc_cam, valid=vc, root_relative=True)),
        "C_abs": float(c_mpjpe(pc_cam, gc_cam, valid=vc, root_relative=False)),
        "wrist_abs": float(wsel.mean().item() * 1000.0) if wsel.numel() else float("nan"),
    }
    if anchor_log:
        n = len(anchor_log)
        row["anchor_gate_rate"] = sum(a["gate_rate"] for a in anchor_log) / n
        row["anchor_dz_gated_mm"] = 1000.0 * sum(a["dz_gated_m"] for a in anchor_log) / n
        row["anchor_disagree_mm"] = 1000.0 * sum(a.get("disagree_gated_m", 0.0) for a in anchor_log) / n
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--seq_start", type=int, default=0)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--out", default="/tmp/hoi4d_cam_eval.json")
    ap.add_argument("--bbox_perturb", default=None,
                    help="eval-time bbox ablation: jitter[:amp] or fixed[:size]")
    ap.add_argument("--contact_gate", choices=["proxy", "oracle", "off"], default="proxy",
                    help="oracle = gate the anchor by the cached GT contact mask (mechanism "
                         "ceiling); proxy = the deployable |disagree|<band_m gate; off = no anchor.")
    ap.add_argument("--da3_wrist_cache_dir", default=None,
                    help="C1: dir of per-seq DA3 metric wrist-depth caches (<seq>_da3_wrist.pt); "
                         "when set, the anchor uses DA3 depth as ref_d_scene instead of gs_depth.")
    ap.add_argument("--contact_cache_dir", default=None,
                    help="dir of per-seq contact caches (<seq>_contact.pt) for --contact_gate oracle, "
                         "when the in-tree hand_data/contact_cache.pt path is read-only/quota-locked.")
    ap.add_argument("--seqs", default=None,
                    help="comma-separated seq basenames to restrict the eval to (e.g. the held-out "
                         "val split). Default: all seqs with a cam cache under --data_root.")
    ap.add_argument("--backbone", choices=["recon", "dino"], default="recon",
                    help="dino = backbone-swap ablation arm: feed the head frozen DINOv2 patch "
                         "tokens (head-only forward) instead of the reconstruction backbone. "
                         "The random-init arm needs no flag: point the eval config's "
                         "model.checkpoint at the state_dict saved by build_feature_cache "
                         "--random_init --save_model.")
    ap.add_argument("--dino_model", default="dinov2_vitl14")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, device)
    if args.contact_gate == "off":
        model.enable_root_anchor = False        # control arm: no anchor applied
    dino = None
    if args.backbone == "dino":
        dino = torch.hub.load("facebookresearch/dinov2", args.dino_model)
        dino.to(device).eval()
        print(f"Backbone-swap eval: frozen {args.dino_model} tokens -> head-only forward")

    from scripts.hand_vis_utils import MANOModel
    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])

    all_seqs = sorted(os.path.join(args.data_root, d) for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d)))
    seqs = [s for s in all_seqs if _has_cam_cache(s)]
    if args.seqs:
        want = {x.strip() for x in args.seqs.split(",") if x.strip()}
        seqs = [s for s in seqs if os.path.basename(s) in want]
    if args.max_seqs > 0:
        seqs = seqs[args.seq_start:args.seq_start + args.max_seqs]
    print(f"Found {len(seqs)} sequences with camera-frame caches under {args.data_root}"
          + (f" (filtered to {sorted(os.path.basename(s) for s in seqs)})" if args.seqs else ""),
          flush=True)

    anchor_on = bool(getattr(model, "enable_root_anchor", False))
    results = []
    for sq in seqs:
        try:
            r = eval_seq_cam(model, mano_model, device, sq, cfg, args.clip_len, args.stride,
                             args.max_clips, contact_gate=args.contact_gate,
                             da3_wrist_cache_dir=args.da3_wrist_cache_dir,
                             contact_cache_dir=args.contact_cache_dir,
                             bbox_perturb=args.bbox_perturb, dino=dino)
            if r is None:
                continue
            results.append(r)
            astr = ""
            if "anchor_gate_rate" in r:
                astr = (f" | anchor gate={r['anchor_gate_rate'] * 100:.0f}% "
                        f"|dz|={r['anchor_dz_gated_mm']:.1f}mm Δgs={r['anchor_disagree_mm']:.1f}mm")
            print(f"[{r['seq']}] C-abs={r['C_abs']:.1f} C-rr={r['C_MPJPE']:.1f} "
                  f"({r['frames']}f){astr}", flush=True)
        except Exception as e:
            print(f"[skip {os.path.basename(sq)}] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    if not results:
        print("No sequences evaluated.")
        return

    def _mean(key):
        vals = [r[key] for r in results if key in r and r[key] == r[key]]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    agg = {k: _mean(k) for k in ("C_abs", "C_MPJPE", "wrist_abs", "anchor_gate_rate",
                                 "anchor_dz_gated_mm", "anchor_disagree_mm")}
    agg["n_seqs"] = len(results)
    with open(args.out, "w") as f:
        json.dump({"aggregate": agg, "per_seq": results}, f, indent=2)
    tag = "ANCHOR ON" if anchor_on else "anchor OFF"
    print(f"\nOURS HOI4D camera-frame ({tag}):  C-abs={agg['C_abs']:.1f}  "
          f"C-MPJPE(root-rel)={agg['C_MPJPE']:.1f}  "
          f"(n={agg['n_seqs']} seqs) -> {args.out}")
    if anchor_on and agg.get("anchor_gate_rate") == agg.get("anchor_gate_rate"):
        print(f"  anchor: gate={agg['anchor_gate_rate'] * 100:.0f}%  "
              f"|dz|={agg['anchor_dz_gated_mm']:.1f}mm  Δgs={agg['anchor_disagree_mm']:.1f}mm")
    print("  -> Compare C-abs anchor-ON vs anchor-OFF: if ON < OFF, the anchor improves "
          "absolute placement when the gs_depth reference is good (the HOT3D test could not show this).")


if __name__ == "__main__":
    main()
