"""Ownership probe: does removing hand-source scene Gaussians improve held-out reconstruction?

Three renders of the same clip, no training anywhere:
    A  the scene as-is;
    B  scene Gaussians whose source pixel lies under the predicted hand silhouette suppressed;
    G  ONLY those suppressed Gaussians, whose alpha footprint in a target view marks where
       hand-contaminated geometry lands there.

Scored on R_revealed = (ghost footprint in the target) minus (the target's own hand silhouette):
the pixels where the target actually observes scene that the context saw covered by hand. Full-
frame PSNR is not reported on purpose; fourteen easy frames hide two interesting ones.

Two layouts, deliberately distinct. Image-space masks (target silhouette, R_revealed, the hand
region) are row/col in the rendered image. The opacity mask instead indexes the PARAM GRID, whose
layout relative to the image is measured, not assumed: the first clip renders the suppressed
subset alone under every candidate layout and keeps the one whose footprint lands in the hand
boxes (2026-08-20: transpose-family 0.77 against image-layout 0.066 on mix5-hand).

What the RGB verdict can and cannot say: better RGB on R_revealed with stable alpha demonstrates
DECONTAMINATION and better scene-only rendering. A geometry claim additionally needs depth error
or symmetric Chamfer with completeness, and HOT3D ships no GT depth, so this probe does not make
one. Two controls close the deletion loophole: the same MAE restricted to pixels both A and B
cover, and a repeat under a white background; a gain that survives both is revealed scene, not
background showing through. A C-vs-B difference on R_revealed is a leakage detector for G_hand,
and the hand-only diagnostic render plus the magenta sentinel decide whether an inert C is
invisible parameters or a broken insertion path.

Deltas are paired per clip and the bootstrap resamples SEQUENCES, as in `paired_clip_stats`.

Targets are INTERIOR frames carried at the tensor's tail: the model slices context positionally,
so the frames are permuted while `frame_index` keeps their true timestamps. Interior targets are
what makes the later MANO interpolation valid (gap-1 error 1.26 mm against 45.44 at gap 8).

Provenance caveat: the silhouette comes from context-frame hand predictions of a joint forward
whose INPUT includes the target frames, the same forward that builds the scene Gaussians. The
leak, if any, applies identically to A and B, so the paired delta stays fair.
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
from scripts.paired_clip_stats import cluster_bootstrap
from scripts.train_hand_head import (HOT3DHandDataset, discover_sequences, mixed_collate,
                                     compute_vertices_from_batch)
from scripts.hand_vis_utils import MANOModel
from scripts.hand_gaussians import build_hand_gaussians
from scripts.mano_interp import interp_hand_params
from diffsynth.auxiliary_models.worldmirror.models.utils.sh_utils import RGB2SH
from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    set_default_frame_width,
)

GRID_LAYOUTS = ("image", "transpose", "rot90cw", "rot90ccw", "rot180")


def hand_silhouette(verts_cam, hand_valid, intr, res, dilate_px=4):
    """[B, S, H, W] image-layout mask of projected MANO vertices, dilated.

    verts_cam: [B, S, 2, V, 3] metric camera-frame vertices.
    intr:      [B, 3] (f, cx, cy) already on the render frame.
    """
    B, S, Hn, V, _ = verts_cam.shape
    f = intr[:, 0].view(B, 1, 1, 1)
    cx = intr[:, 1].view(B, 1, 1, 1)
    cy = intr[:, 2].view(B, 1, 1, 1)
    z = verts_cam[..., 2].clamp_min(1e-3)
    col = f * verts_cam[..., 0] / z + cx
    row = f * verts_cam[..., 1] / z + cy
    u = col.round().long().clamp(0, res - 1)
    v = row.round().long().clamp(0, res - 1)
    mask = torch.zeros(B, S, res, res, device=verts_cam.device)
    ok = hand_valid.bool().unsqueeze(-1).expand(B, S, Hn, V)
    b_i = torch.arange(B, device=u.device).view(B, 1, 1, 1).expand_as(u)[ok]
    s_i = torch.arange(S, device=u.device).view(1, S, 1, 1).expand_as(u)[ok]
    mask[b_i, s_i, v[ok], u[ok]] = 1.0
    k = 2 * dilate_px + 1
    return (torch.nn.functional.max_pool2d(mask.view(B * S, 1, res, res), k, 1, dilate_px)
            .view(B, S, res, res))


def to_grid(mask: torch.Tensor, layout: str) -> torch.Tensor:
    """Carry an image-layout [..., H, W] mask into the param grid's layout."""
    if layout == "image":
        return mask
    if layout == "transpose":
        return mask.transpose(-1, -2)
    if layout == "rot90cw":
        return torch.rot90(mask, k=-1, dims=(-2, -1))
    if layout == "rot90ccw":
        return torch.rot90(mask, k=1, dims=(-2, -1))
    if layout == "rot180":
        return torch.rot90(mask, k=2, dims=(-2, -1))
    raise ValueError(layout)


def render_with_alpha(model, preds, views, h, w, background="black"):
    """(colors [B,S,H,W,3], alpha [B,S,H,W]) from the already-built splats."""
    splats = preds["splats"]
    c2w = preds["rendered_extrinsics"]
    intrs = preds["rendered_intrinsics"]
    ts = preds["rendered_timestamps"]
    w2c = homo_matrix_inverse(c2w)
    B = len(splats)
    rast = model.gs_renderer.rasterizer
    prev_bg = rast.backgrounds
    rast.backgrounds = background
    try:
        colors, _depth, alpha = rast.forward(
            splats, render_viewmats=[w2c[b] for b in range(B)],
            render_Ks=[intrs[b] for b in range(B)],
            render_timestamps=[ts[b] for b in range(B)],
            sh_degree=0, width=w, height=h)
    finally:
        rast.backgrounds = prev_bg
    return colors.clamp(0, 1).float(), alpha.float().squeeze(-1)


def gaussian_set_stats(hand_sets, w2c_targets, ts_targets):
    """Numbers that separate 'invisible parameters' from 'never rendered'."""
    if not hand_sets:
        return {"n_sets": 0}
    n = [int(g.means.shape[0]) for g in hand_sets]
    op = torch.cat([g.opacities.reshape(-1) for g in hand_sets]).float()
    sc = torch.cat([g.scales.reshape(-1) for g in hand_sets]).float()
    z_stats = []
    for k, (w2c, t) in enumerate(zip(w2c_targets, ts_targets)):
        match = [g for g in hand_sets if int(g.timestamp) == int(t)]
        if not match:
            z_stats.append(None)
            continue
        m = torch.cat([g.means for g in match]).float()
        cam = m @ w2c[:3, :3].T + w2c[:3, 3]
        z_stats.append({"n": int(m.shape[0]), "z_min": float(cam[:, 2].min()),
                        "z_med": float(cam[:, 2].median()), "z_max": float(cam[:, 2].max())})
    return {"n_sets": len(hand_sets), "n_gaussians": n,
            "timestamps": [int(g.timestamp) for g in hand_sets],
            "opacity": [float(op.min()), float(op.median()), float(op.max())],
            "scale": [float(sc.min()), float(sc.median()), float(sc.max())],
            "target_cam_depth": z_stats}


def alpha_footprint(alpha_t):
    """Per-target-frame alpha evidence for a hand-only render."""
    out = []
    for s in range(alpha_t.shape[0]):
        a = alpha_t[s]
        out.append({"alpha_sum": float(a.sum()), "alpha_max": float(a.max()),
                    "px_over_0.01": int((a > 0.01).sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_clips", type=int, default=12)
    ap.add_argument("--n_seqs", type=int, default=4)
    ap.add_argument("--targets", type=int, nargs="+", default=[6, 10],
                    help="interior frame positions to hold out")
    ap.add_argument("--ghost_thresh", type=float, default=0.3)
    ap.add_argument("--grid_layout", default="auto", choices=("auto",) + GRID_LAYOUTS,
                    help="layout of the param grid the opacity mask indexes; auto measures it "
                         "on the first clip via the instrument check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(a.config))
    d = cfg["data"]
    res = int(d["resolution"][0])
    set_default_frame_width(float(res))
    S_all = int(d["num_frames"])
    tgt = sorted(a.targets)
    assert 0 < tgt[0] and tgt[-1] < S_all - 1, "targets must be interior frames"
    for q in tgt:
        assert q - 1 not in tgt and q + 1 not in tgt, (
            "adjacent targets would interpolate a hand from another target's own "
            "prediction, which defeats the held-out claim")
    order = [i for i in range(S_all) if i not in tgt] + tgt
    order_t = torch.tensor(order)

    seqs = [q for q in discover_sequences(d["data_root"])
            if os.path.exists(os.path.join(q, "hand_data/cam_extrinsics_cache.pt"))][-a.n_seqs:]
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])
    ds = HOT3DHandDataset(seqs, mano, num_frames=S_all, res=(res, res),
                          clip_stride=d["clip_stride"],
                          use_hand_crop=cfg["model"].get("use_hand_crop", False),
                          rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 2.0))
    idx = list(range(0, len(ds), max(1, len(ds) // a.n_clips)))[: a.n_clips]
    clip_meta = [(os.path.basename(ds.clips[j]["seq_path"].rstrip("/")),
                  int(ds.clips[j]["frame_offset"])) for j in idx]
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=1, shuffle=False,
                        num_workers=1, collate_fn=mixed_collate)
    model = build_model(cfg, a.device)
    if a.ckpt:
        load_hand_head(model, a.ckpt, a.device)
    model.train()   # the GT-camera render branch is training-gated; no_grad keeps it inference
    print(f"[probe] {len(idx)} clips, targets at {tgt} of {S_all}, permuted to the tail", flush=True)

    def fwd(views):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(views, is_inference=False, use_motion=False)

    n_ctx = S_all - len(tgt)
    layout = None if a.grid_layout == "auto" else a.grid_layout
    layout_sweep = {}
    per_clip = []
    diagnostics = {"hand_only": [], "sentinel": []}
    dumped = False
    for ci, batch in enumerate(loader):
        seq_name, offset = clip_meta[ci]
        imgs = batch["img"][:, order_t].to(a.device)
        extr = batch["cam_extrinsics"][:, order_t].to(a.device)
        intr = batch["cam_intrinsics"].to(a.device).view(-1, 3)
        hb = batch["hand_bboxes"][:, order_t].to(a.device)
        hv = batch["hand_valid"][:, order_t].bool().to(a.device)
        fi = order_t.view(1, -1).to(a.device)
        views = build_views_metric(imgs, S_all, a.device, extr, intr, res,
                                   hand_bboxes=hb, hand_valid=hv,
                                   n_targets=len(tgt), frame_index=fi)
        preds_a = fwd(views)
        col_a, alpha_a = render_with_alpha(model, preds_a, views, res, res)

        verts = compute_vertices_from_batch(preds_a["hand_joints"], mano, a.device)
        sil = hand_silhouette(verts.float(), hv, intr.float(), res)      # [B, S_all, H, W]
        sil_ctx, sil_tgt = sil[:, :n_ctx], sil[:, n_ctx:]

        bb = hb[0, :n_ctx].clamp(0, 1) * res
        box_mask = torch.zeros(n_ctx, res, res, device=a.device)
        for s_ in range(n_ctx):
            for h_ in range(2):
                if hv[0, s_, h_]:
                    x1, y1, x2, y2 = [int(q) for q in bb[s_, h_]]
                    box_mask[s_, max(y1, 0):y2, max(x1, 0):x2] = 1

        def ghost_overlap(lay):
            """Render the suppressed subset alone under `lay`; how much lands in the boxes?"""
            vg = dict(views)
            vg["scene_opacity_mask"] = torch.cat(
                [to_grid(sil_ctx, lay), torch.ones_like(sil_tgt)], dim=1)
            pg = fwd(vg)
            cg, ag = render_with_alpha(model, pg, vg, res, res)
            g_ctx = ag[0, :n_ctx] > a.ghost_thresh
            ov = float((g_ctx & box_mask.bool()).sum()) / max(float(g_ctx.sum()), 1.0)
            return ov, ag

        if layout is None:
            for lay in GRID_LAYOUTS:
                layout_sweep[lay], _ = ghost_overlap(lay)
            layout = max(layout_sweep, key=layout_sweep.get)
            print(f"[layout] instrument sweep on {seq_name}@{offset}: "
                  + "  ".join(f"{k}={v:.3f}" for k, v in layout_sweep.items())
                  + f"  -> using {layout}", flush=True)
        overlap, alpha_g = ghost_overlap(layout)

        if not dumped:
            dumped = True
            import numpy as _np
            from PIL import Image as _Im
            s0 = 0
            im = (imgs[0, s0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8").copy()
            im[sil_ctx[0, s0].cpu().numpy() > 0.5] = [255, 0, 0]          # silhouette in red
            gm = (alpha_g[0, s0] > a.ghost_thresh).cpu().numpy()
            im[gm] = (0.5 * im[gm] + [0, 127, 0]).astype("uint8")          # ghost footprint green
            bmask = box_mask[s0].cpu().numpy() > 0.5                       # box filled blue-ish
            im[bmask] = (0.7 * im[bmask] + [0, 0, 76]).astype("uint8")
            _out = os.path.dirname(a.out) or "."
            _Im.fromarray(im).save(os.path.join(_out, "ownab_overlay.png"))
            print(f"[overlay] written to {_out}/ownab_overlay.png "
                  f"(red silhouette, green suppressed footprint, blue box)", flush=True)

        views_b = dict(views)
        views_b["scene_opacity_mask"] = torch.cat(
            [1.0 - to_grid(sil_ctx, layout), torch.ones_like(sil_tgt)], dim=1)
        preds_b = fwd(views_b)
        col_b, alpha_b = render_with_alpha(model, preds_b, views_b, res, res)

        # C: B plus G_hand. Context hands come from their own predictions; each target's hand is
        # interpolated at gap 1 from its temporal neighbours (1.26 mm measured error), never from
        # the target's own prediction, so the composite stays honest about what a held-out frame
        # may know.
        verts_c = verts.clone()
        hv_c = hv.clone()
        c2w_all = views["camera_poses"][0]
        for j, orig in enumerate(tgt):
            k = n_ctx + j
            p_lo, p_hi = order.index(orig - 1), order.index(orig + 1)
            pj = preds_a["hand_joints"][0]
            pt = interp_hand_params(pj[p_lo].view(2, 32).float(), pj[p_hi].view(2, 32).float(),
                                    c2w_all[p_lo].float(), c2w_all[p_hi].float(),
                                    c2w_all[k].float(), 0.5)
            vt = compute_vertices_from_batch(pt.reshape(1, 1, 64), mano, a.device)
            verts_c[0, k] = vt[0, 0]
            hv_c[0, k] = hv[0, p_lo] & hv[0, p_hi]
        hand_sets = build_hand_gaussians(
            verts_c[0].float(), hv_c[0], c2w_all.float(), imgs[0].float(),
            intr[0].float(), views["timestamp"][0])
        preds_c = dict(preds_b)
        preds_c["splats"] = [list(preds_b["splats"][0]) + hand_sets]
        col_c, alpha_c = render_with_alpha(model, preds_c, views_b, res, res)

        # The C-unblocking evidence, first three clips: G_hand alone, then a loud sentinel whose
        # absence from the render convicts the insertion path rather than the parameters.
        if ci < 3:
            w2c_all = homo_matrix_inverse(preds_b["rendered_extrinsics"])[0]
            ts_all = preds_b["rendered_timestamps"][0]
            stats = gaussian_set_stats(hand_sets,
                                       [w2c_all[n_ctx + j] for j in range(len(tgt))],
                                       [ts_all[n_ctx + j] for j in range(len(tgt))])
            fp = None
            if hand_sets:
                preds_h = dict(preds_b)
                preds_h["splats"] = [list(hand_sets)]
                col_h, alpha_h = render_with_alpha(model, preds_h, views_b, res, res)
                fp = {"targets": alpha_footprint(alpha_h[0, n_ctx:]),
                      "contexts_alpha_sum": float(alpha_h[0, :n_ctx].sum())}
            diagnostics["hand_only"].append({"seq": seq_name, "offset": offset,
                                             "stats": stats, "footprint": fp})

            sent = []
            for g in hand_sets:
                s = type(g)(means=g.means, harmonics=RGB2SH(
                                torch.tensor([[1.0, 0.0, 1.0]], device=g.means.device)
                                .expand(g.means.shape[0], 3)).unsqueeze(-2).to(g.means.dtype),
                            opacities=torch.full_like(g.opacities, 0.99),
                            scales=torch.full_like(g.scales, 0.02),
                            rotations=g.rotations, timestamp=g.timestamp)
                sent.append(s)
            if sent:
                preds_s = dict(preds_b)
                preds_s["splats"] = [list(preds_b["splats"][0]) + sent]
                col_s, alpha_s = render_with_alpha(model, preds_s, views_b, res, res)
                dif = (col_s - col_b).abs().amax(dim=-1)[0]
                diagnostics["sentinel"].append({
                    "seq": seq_name, "offset": offset,
                    "target_px_changed": int((dif[n_ctx:] > 0.05).sum()),
                    "context_px_changed": int((dif[:n_ctx] > 0.05).sum()),
                    "max_delta": float(dif.max())})

        ghost = alpha_g[:, n_ctx:] > a.ghost_thresh
        revealed = ghost & (sil_tgt < 0.5)
        gt = imgs[:, n_ctx:].permute(0, 1, 3, 4, 2).float()
        hand_region = sil_tgt > 0.5
        both_covered = revealed & (alpha_a[:, n_ctx:] > 0.5) & (alpha_b[:, n_ctx:] > 0.5)

        row = {"seq": seq_name, "offset": offset, "overlap": overlap,
               "revealed_px": float(revealed.float().mean()),
               "covered_px_frac": (float(both_covered.float().sum() / revealed.float().sum())
                                   if revealed.any() else None)}
        for nm, col, al in (("A", col_a, alpha_a), ("B", col_b, alpha_b), ("C", col_c, alpha_c)):
            if revealed.any():
                row[f"{nm}_mae"] = float((col[:, n_ctx:][revealed] - gt[revealed]).abs().mean())
                row[f"{nm}_alpha"] = float(al[:, n_ctx:][revealed].mean())
            if both_covered.any():
                row[f"{nm}_mae_covered"] = float(
                    (col[:, n_ctx:][both_covered] - gt[both_covered]).abs().mean())
            if hand_region.any():
                row[f"{nm}_hand_mae"] = float(
                    (col[:, n_ctx:][hand_region] - gt[hand_region]).abs().mean())

        # White-background control: a B advantage that flips sign here was background paint, not
        # revealed scene. Splats are already built; only the two renders repeat.
        if revealed.any():
            col_aw, _ = render_with_alpha(model, preds_a, views, res, res, background="white")
            col_bw, _ = render_with_alpha(model, preds_b, views_b, res, res, background="white")
            row["A_mae_whitebg"] = float((col_aw[:, n_ctx:][revealed] - gt[revealed]).abs().mean())
            row["B_mae_whitebg"] = float((col_bw[:, n_ctx:][revealed] - gt[revealed]).abs().mean())
        per_clip.append(row)
        print(f"[clip {ci+1}/{len(idx)}] {seq_name}@{offset} overlap={overlap:.2f} "
              f"A={row.get('A_mae')} B={row.get('B_mae')}", flush=True)

    def paired(kb, ka):
        d = {(r["seq"], r["offset"]): r[kb] - r[ka]
             for r in per_clip if r.get(kb) is not None and r.get(ka) is not None}
        if not d:
            return None
        vals = sorted(d.values())
        lo, hi = cluster_bootstrap(d)
        return {"n": len(vals), "n_seq": len({s for s, _ in d}),
                "mean": sum(vals) / len(vals), "median": vals[len(vals) // 2],
                "wins_b": sum(v < 0 for v in vals), "ci95": [lo, hi],
                "significant": bool(lo > 0 or hi < 0)}

    def agg(k):
        v = [r[k] for r in per_clip if r.get(k) is not None]
        return sum(v) / len(v) if v else None

    import hashlib, subprocess
    _sem_files = ["scripts/probe_ownership_ab.py", "scripts/metric_views.py",
                  "scripts/hand_gaussians.py", "scripts/mano_interp.py",
                  "diffsynth/auxiliary_models/worldmirror/models/models/rasterization.py"]
    _h = hashlib.sha256()
    for _f in _sem_files:
        _h.update(open(_f, "rb").read())
    try:
        _commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, timeout=10).stdout.strip() or None
    except Exception:
        _commit = None
    out = {"provenance": {
               "resolved_config": cfg,
               "git_commit": _commit,
               "semantic_files_sha256": _h.hexdigest(),
               "grid_layout": layout,
               "grid_layout_sweep": layout_sweep or None,
               "r_revealed_definition": (
                   f"(alpha of hand-source-only render in target view > {a.ghost_thresh}) "
                   f"AND (target hand silhouette < 0.5); silhouette = predicted MANO vertices "
                   f"projected and max-pool dilated by 4 px at {res} px"),
               "target_hand_source": "interpolated at gap 1 from temporal neighbours, never the "
                                     "target's own prediction",
               "ownership_source": "context-frame hand predictions from the joint forward whose "
                                   "input includes target frames; identical for A and B, so the "
                                   "paired delta is fair, but not a strictly context-only signal",
               "geometry_evidence": "none: HOT3D ships no GT depth, so RGB verdicts here speak to "
                                    "decontamination and rendering only, not geometry",
           },
           "targets": tgt, "n_clips": len(idx),
           "silhouette_bbox_overlap": agg("overlap"),
           "revealed_frac_of_target_pixels": agg("revealed_px"),
           "covered_px_frac": agg("covered_px_frac"),
           "A": {"rgb_mae_revealed": agg("A_mae"), "alpha_revealed": agg("A_alpha"),
                 "rgb_mae_hand_region": agg("A_hand_mae"),
                 "rgb_mae_covered": agg("A_mae_covered"), "rgb_mae_whitebg": agg("A_mae_whitebg")},
           "B": {"rgb_mae_revealed": agg("B_mae"), "alpha_revealed": agg("B_alpha"),
                 "rgb_mae_hand_region": agg("B_hand_mae"),
                 "rgb_mae_covered": agg("B_mae_covered"), "rgb_mae_whitebg": agg("B_mae_whitebg")},
           "C": {"rgb_mae_revealed": agg("C_mae"), "alpha_revealed": agg("C_alpha"),
                 "rgb_mae_hand_region": agg("C_hand_mae"),
                 "rgb_mae_covered": agg("C_mae_covered")},
           "paired_B_minus_A": {
               "mae_revealed": paired("B_mae", "A_mae"),
               "mae_covered": paired("B_mae_covered", "A_mae_covered"),
               "mae_whitebg": paired("B_mae_whitebg", "A_mae_whitebg")},
           "paired_C_minus_B": {"mae_revealed": paired("C_mae", "B_mae")},
           "diagnostics": diagnostics,
           "per_clip": per_clip}
    ov = out["silhouette_bbox_overlap"]
    if ov is not None and ov < 0.5:
        out["verdict"] = (f"MASK LAYOUT SUSPECT: only {ov:.2f} of the suppressed subset's "
                          "footprint lies in the hand boxes; numbers above are not trustworthy")
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_clip", "provenance")},
                     indent=2), flush=True)
    print("OWNERSHIP_AB_OK", flush=True)


if __name__ == "__main__":
    main()
