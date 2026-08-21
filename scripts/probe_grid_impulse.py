"""Where does the Gaussian param grid live, relative to the image?

A bump is added to one quadrant of the GS head's per-layer projected features, the same tensors
the hand-to-GS injection writes into, and the render is diffed against a clean forward. If the
bump surfaces in a different image quadrant, then every consumer that indexes the grid with image
coordinates, the injection's box-guided writes, `hand_depth_anchor`'s pixel sampling, and the
ownership probe's opacity mask, has been addressing the wrong region, and by exactly this map.

The gs_depth diff is reported in its own storage layout as the positive control: the bump must
appear in the injected quadrant there, or the instrument itself is broken.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from scripts.eval_hand_head import build_model, load_hand_head
from diffsynth.utils.auxiliary import homo_matrix_inverse
from scripts.metric_views import build_views_metric
from scripts.train_hand_head import (HOT3DHandDataset, discover_sequences, mixed_collate)
from scripts.hand_vis_utils import MANOModel
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    set_default_frame_width,
)

QUADS = ("TL", "TR", "BL", "BR")


def quad_energy(m: torch.Tensor) -> dict:
    """Fraction of |m|'s mass per quadrant; m is [..., H, W]."""
    H, W = m.shape[-2:]
    h, w = H // 2, W // 2
    tot = float(m.abs().sum()) or 1.0
    return {"TL": float(m[..., :h, :w].abs().sum()) / tot,
            "TR": float(m[..., :h, w:].abs().sum()) / tot,
            "BL": float(m[..., h:, :w].abs().sum()) / tot,
            "BR": float(m[..., h:, w:].abs().sum()) / tot}


def render(model, preds, h, w):
    splats = preds["splats"]
    w2c = homo_matrix_inverse(preds["rendered_extrinsics"])
    intrs = preds["rendered_intrinsics"]
    ts = preds["rendered_timestamps"]
    B = len(splats)
    colors, _d, _a = model.gs_renderer.rasterizer.forward(
        splats, render_viewmats=[w2c[b] for b in range(B)],
        render_Ks=[intrs[b] for b in range(B)],
        render_timestamps=[ts[b] for b in range(B)],
        sh_degree=0, width=w, height=h)
    return colors.clamp(0, 1).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--delta", type=float, default=5.0)
    ap.add_argument("--n_targets", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(a.config))
    d = cfg["data"]
    res = int(d["resolution"][0])
    set_default_frame_width(float(res))
    S_all = int(d["num_frames"])

    seqs = [q for q in discover_sequences(d["data_root"])
            if os.path.exists(os.path.join(q, "hand_data/cam_extrinsics_cache.pt"))][-1:]
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])
    ds = HOT3DHandDataset(seqs, mano, num_frames=S_all, res=(res, res),
                          clip_stride=d["clip_stride"],
                          use_hand_crop=cfg["model"].get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 2.0))
    loader = DataLoader(torch.utils.data.Subset(ds, [0]), batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=mixed_collate)
    model = build_model(cfg, a.device)
    if a.ckpt:
        load_hand_head(model, a.ckpt, a.device)
    model.train()   # the GT-camera render branch is training-gated; no_grad keeps it inference

    batch = next(iter(loader))
    imgs = batch["img"].to(a.device)
    extr = batch["cam_extrinsics"].to(a.device)
    intr = batch["cam_intrinsics"].to(a.device).view(-1, 3)
    hb = batch["hand_bboxes"].to(a.device)
    hv = batch["hand_valid"].bool().to(a.device)
    views = build_views_metric(imgs, S_all, a.device, extr, intr, res,
                               hand_bboxes=hb, hand_valid=hv, n_targets=a.n_targets)

    def fwd():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(views, is_inference=False, use_motion=False)

    p_clean = fwd()
    rgb_clean = render(model, p_clean, res, res)
    depth_clean = p_clean["gs_depth"].float().squeeze(-1)

    def impulse_hook(feats, fs, fe):
        out = []
        for f in feats:
            H, W = f.shape[-2:]
            g = f.clone()
            g[..., : H // 2, : W // 2] += a.delta
            out.append(g)
        return out

    orig_forward = model.gs_head.forward

    def patched(token_list, images=None, patch_start_idx=None, frames_chunk_size=8,
                feature_hook=None):
        return orig_forward(token_list, images=images, patch_start_idx=patch_start_idx,
                            frames_chunk_size=frames_chunk_size, feature_hook=impulse_hook)

    model.gs_head.forward = patched
    p_pert = fwd()
    model.gs_head.forward = orig_forward
    rgb_pert = render(model, p_pert, res, res)
    depth_pert = p_pert["gs_depth"].float().squeeze(-1)

    S_ctx = depth_clean.shape[1]
    d_depth = (depth_pert - depth_clean).abs()[0]                 # storage layout, [S, H, W]
    d_rgb = (rgb_pert - rgb_clean).abs().amax(dim=-1)[0]          # image layout, [S_all, H, W]

    depth_q = quad_energy(d_depth)
    rgb_ctx_q = quad_energy(d_rgb[:S_ctx])
    rgb_tgt_q = quad_energy(d_rgb[S_ctx:]) if d_rgb.shape[0] > S_ctx else None
    top_depth = max(depth_q, key=depth_q.get)
    top_rgb = max(rgb_ctx_q, key=rgb_ctx_q.get)

    out = {"delta": a.delta, "seq": os.path.basename(seqs[0].rstrip("/")),
           "gs_depth_diff_quadrants_storage_layout": depth_q,
           "render_diff_quadrants_image_layout_ctx": rgb_ctx_q,
           "render_diff_quadrants_image_layout_tgt": rgb_tgt_q,
           "gs_depth_diff_mean": float(d_depth.mean()),
           "render_diff_mean": float(d_rgb.mean()),
           "injected_quadrant": "TL (in each feature map's own storage layout)",
           "conclusion": (
               f"positive control {'PASSES' if top_depth == 'TL' else 'FAILS'} "
               f"(depth diff peaks in {top_depth}); a TL feature impulse renders into image "
               f"quadrant {top_rgb}, so grid->image is "
               f"{'identity' if top_rgb == 'TL' else 'NOT identity: ' + top_rgb}")}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print("GRID_IMPULSE_OK", flush=True)


if __name__ == "__main__":
    main()
