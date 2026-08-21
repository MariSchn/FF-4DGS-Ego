"""Held-out-view Gaussian evaluation, on frames the model never saw.

The primary metric is PSNR on the TARGET frames. Every previous injection A/B in this project
scored the source views, which the Gaussians were unprojected from, so the hand could only ever
add noise there. Held-out frames are the regime where extra information can pay.

Fixed before any result exists, so the choice cannot follow the numbers:
  primary    PSNR on target frames
  secondary  SSIM and LPIPS on target frames
  validity   at_clamp == 0, depth close to the pretrained initialisation, splat scales not
             collapsed. An arm that fails these has not earned a comparison.
  reference  the same metrics on the context frames, which is the easy case and only tells us
             whether a model is broken outright.

Run it on the pretrained initialisation too. "Injection on beats injection off" means nothing if
both sit below the checkpoint they started from, which is exactly how the gsinj2 pair read.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader

from scripts.eval_hand_head import build_model, load_hand_head
from scripts.gs_metrics import (
    LPIPSScorer,
    render_views_from_predictions,
    metric_chunks_from_batch,
    metrics_from_chunks,
)
from scripts.metric_views import build_views_metric
from scripts.train_hand_head import (
    HOT3DHandDataset,
    discover_sequences,
    mixed_collate,
)
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    set_default_frame_width,
)


def _percentiles(t: torch.Tensor, qs=(0.05, 0.5, 0.95)) -> list[float]:
    f = t.flatten().float()
    return [float(f.quantile(q)) for q in qs]


def _splat_stats(splats) -> tuple[torch.Tensor, torch.Tensor]:
    """Scales and opacities over a batch of `Gaussians`, which carries attributes, not keys."""
    sc, op = [], []
    stack = list(splats) if isinstance(splats, (list, tuple)) else [splats]
    while stack:
        g = stack.pop()
        if isinstance(g, (list, tuple)):
            stack.extend(g)
        else:
            sc.append(g.scales.detach().flatten().float().cpu())
            op.append(g.opacities.detach().flatten().float().cpu())
    if not sc:
        raise RuntimeError("no Gaussians in preds['splats']")
    return torch.cat(sc), torch.cat(op)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="omit to score the pretrained initialisation")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n_clips", type=int, default=64)
    ap.add_argument("--n_seqs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    d, t = cfg["data"], cfg["training"]
    res = tuple(d["resolution"])
    set_default_frame_width(float(res[0]))
    n_targets = int(t.get("gs_n_targets", 2))

    seqs = discover_sequences(d["data_root"])
    seqs = [q for q in seqs
            if os.path.exists(os.path.join(q, "hand_data/cam_extrinsics_cache.pt"))]
    # The LAST sequences, so the scored clips sit outside the head of the list the arms train on,
    # and identical across every label because the order is sorted and deterministic.
    seqs = seqs[-a.n_seqs:]
    if not seqs:
        raise SystemExit(f"no sequence with camera extrinsics under {d['data_root']}")

    from scripts.hand_vis_utils import MANOModel
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])
    ds = HOT3DHandDataset(seqs, mano, num_frames=d["num_frames"], res=res,
                          clip_stride=d["clip_stride"],
                          use_hand_crop=cfg["model"].get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 2.0))
    idx = list(range(0, len(ds), max(1, len(ds) // a.n_clips)))[: a.n_clips]
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=a.batch_size,
                        shuffle=False, num_workers=2, collate_fn=mixed_collate)
    print(f"[eval] {a.label}: {len(seqs)} sequences, {len(idx)} clips, {n_targets} held out of "
          f"{d['num_frames']}", flush=True)

    model = build_model(cfg, a.device)
    loaded = {"gs_head": 0, "injection": 0, "hand_head": 0}
    if a.ckpt:
        # Not load_hand_head: it discards the strict=False report, and a checkpoint whose keys do
        # not match would then load nothing and leave three identical rows in the table.
        ck = torch.load(a.ckpt, map_location=a.device)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        own = set(model.state_dict().keys())
        hit = [k for k in sd if k in own]
        for k in hit:
            for grp, pre in (("gs_head", "gs_head"), ("injection", "hand_to_gs_injection"),
                             ("hand_head", "hand_head")):
                if k.startswith(pre):
                    loaded[grp] += 1
        model.load_state_dict(sd, strict=False)
        print(f"[ckpt] {len(hit)}/{len(sd)} keys matched the model; "
              f"gs_head={loaded['gs_head']} injection={loaded['injection']} "
              f"hand_head={loaded['hand_head']}", flush=True)
        if loaded["gs_head"] == 0:
            raise SystemExit(
                f"{a.ckpt} contributed no gs_head weights, so this row would score the pretrained "
                "Gaussian branch and silently duplicate the initialisation row")
    # The GT-camera render branch is gated on `self.training` (rasterization.py:543). Scoring a
    # held-out view against the camera it was actually taken from therefore needs train mode, and
    # no_grad plus dropout 0.0 in this config keeps it deterministic.
    model.train()
    lpips = LPIPSScorer(device=a.device)

    tgt_chunks, ctx_chunks = [], []
    cov_tgt, cov_ctx = [], []
    at_clamp, depth_med, scale_p, opacity_p = [], [], [], []
    per_clip = []
    clip_meta = [(os.path.basename(ds.clips[j]["seq_path"].rstrip("/")),
                  int(ds.clips[j]["frame_offset"])) for j in idx]
    clip_ptr = 0
    for batch in loader:
        imgs = batch["img"].to(a.device)
        B, S = imgs.shape[:2]
        views = build_views_metric(
            imgs, S, a.device,
            batch["cam_extrinsics"].to(a.device), batch["cam_intrinsics"].to(a.device),
            int(res[0]),
            hand_bboxes=batch["hand_bboxes"].to(a.device),
            hand_valid=batch["hand_valid"].bool().to(a.device),
            n_targets=n_targets,
            frame_index=batch.get("frame_index"),
        )
        with torch.no_grad():
            preds = model(views, is_inference=False, use_motion=False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                rendered = render_views_from_predictions(
                    model, preds, views, height=imgs.shape[-2], width=imgs.shape[-1])

        is_tgt = views["is_target"][:, : rendered.shape[1]]
        ch_t = metric_chunks_from_batch(rendered, imgs, is_tgt, lpips, a.device)
        tgt_chunks.append(ch_t)
        ctx_chunks.append(metric_chunks_from_batch(rendered, imgs, ~is_tgt, lpips, a.device))

        # Per-clip rows, so two checkpoints can be compared PAIRED and the bootstrap clustered
        # by sequence: 96 target frames are not 96 independent samples.
        from scripts.gs_metrics import region_metric_chunks_from_batch as _region_chunks
        from scripts.train_hand_head import _hand_region_mask
        hmask = _hand_region_mask(batch["hand_bboxes"].to(a.device),
                                  batch["hand_valid"].bool().to(a.device),
                                  rendered.shape[2], rendered.shape[3])
        ch_h = _region_chunks(rendered, imgs, hmask, lpips, a.device)
        ch_b = _region_chunks(rendered, imgs, ~hmask, lpips, a.device)
        Bv, Sv = rendered.shape[:2]
        for b in range(Bv):
            seq, off = clip_meta[clip_ptr]; clip_ptr += 1
            m = is_tgt[b].cpu()
            row = {"seq": seq, "offset": off}
            for name, ch in (("", ch_t), ("hand_", ch_h), ("bg_", ch_b)):
                ok = m & ch["valid"].view(Bv, Sv)[b]
                pv = ch["psnr"].view(Bv, Sv)[b][ok]
                lv = ch["lpips"].view(Bv, Sv)[b][ok]
                row[f"{name}psnr"] = float(pv.mean()) if pv.numel() else None
                row[f"{name}lpips"] = float(lv.mean()) if lv.numel() else None
            per_clip.append(row)

        # Is a target render dark, or merely wrong? A camera-convention error cancels on the
        # views the Gaussians were unprojected from and does not cancel on a novel one, which
        # looks the same in PSNR as genuine novel-view difficulty. Coverage separates them:
        # an empty frustum renders near zero everywhere.
        _r = rendered.float()
        for _m, _acc in ((is_tgt, cov_tgt), (~is_tgt, cov_ctx)):
            _sel = _r[_m]
            if _sel.numel():
                _acc.append((float(_sel.mean()), float(_sel.std()),
                             float((_sel.amax(dim=-1) < 0.02).float().mean())))

        gd = preds["gs_depth"].float()
        at_clamp.append(float((gd >= 4.0e8).float().mean()))
        depth_med.append(float(gd.median()))
        sc, op = _splat_stats(preds["splats"])
        scale_p.append(_percentiles(sc))
        opacity_p.append(_percentiles(op))

    out = {
        "label": a.label,
        "ckpt": a.ckpt or "pretrained initialisation",
        "n_clips": len(idx),
        "n_targets": n_targets,
        "keys_loaded": loaded,
        "per_clip": per_clip,
        "held_out": metrics_from_chunks(tgt_chunks),
        "context": metrics_from_chunks(ctx_chunks),
        "coverage": {
            "target_mean_std_darkfrac": [sum(c[i] for c in cov_tgt) / max(len(cov_tgt), 1)
                                         for i in range(3)],
            "context_mean_std_darkfrac": [sum(c[i] for c in cov_ctx) / max(len(cov_ctx), 1)
                                          for i in range(3)],
        },
        "validity": {
            "frac_depth_at_clamp": sum(at_clamp) / max(len(at_clamp), 1),
            "gs_depth_median_m": sorted(depth_med)[len(depth_med) // 2] if depth_med else None,
            "splat_scale_p05_p50_p95": [sum(c[i] for c in scale_p) / len(scale_p) for i in range(3)],
            "opacity_p05_p50_p95": [sum(c[i] for c in opacity_p) / len(opacity_p) for i in range(3)],
        },
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"HELDOUT_EVAL_OK -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
