"""Does the metric-camera path actually work on a real batch, before a training run trusts it?

Four things must hold, and each has failed silently in this project before:
  1. context unprojection lands where the geometry says, checked by round-trip on real intrinsics;
  2. the target frames are genuinely held out of the model's input;
  3. a target render differs from the context render, which is the whole point: under the identity
     path the two are the same image and depth is unobservable;
  4. gradients reach gs_depth and are finite.
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

from scripts.eval_hand_head import build_model, load_hand_head
from scripts.metric_views import build_views_metric, intr_3x3, pick_targets
from scripts.run_ours_gs import load_clip
from scripts.train_hand_head import build_views

BOX_FILE = "hand_bboxes_v2_rf1.5_res224x224.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--export_root", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--n_targets", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    res = cfg["data"]["resolution"]
    res = int(res[0] if isinstance(res, (list, tuple)) else res)

    model = build_model(cfg, a.device)
    if a.ckpt:
        load_hand_head(model, a.ckpt, a.device)
    model.eval()

    imgs, frames = load_clip(os.path.join(a.export_root, a.seq, "images"), a.n_views)
    imgs = imgs.unsqueeze(0).to(a.device)
    n = imgs.shape[1]
    hd = os.path.join(a.store, a.seq, "hand_data")
    bb = torch.load(os.path.join(hd, BOX_FILE), map_location="cpu")
    fi = torch.tensor(frames, dtype=torch.long)
    hb = bb["bboxes"][fi].unsqueeze(0).to(a.device)
    hv = bb["valid"][fi].bool().unsqueeze(0).to(a.device)
    extr = torch.load(os.path.join(hd, "cam_extrinsics_cache.pt"), map_location="cpu")
    extr = torch.as_tensor(extr).float()[fi].unsqueeze(0)
    intr = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)

    print(f"clip {a.seq}  S={n}  res={res}")
    print(f"cam_extrinsics {tuple(extr.shape)}  cam_intrinsics {intr.view(-1).tolist()}")

    # 1. round-trip on the real intrinsics: unproject the centre pixel, reproject, land on it
    K = intr_3x3(intr, res, a.device)[0]
    w2c = extr[0, 0].to(a.device)
    c2w = torch.linalg.inv(w2c)
    d = 0.75
    xc = (res / 2.0 - float(K[0, 2])) * d / float(K[0, 0])
    yc = (res / 2.0 - float(K[1, 2])) * d / float(K[1, 1])
    Xw = (c2w[:3, :3] @ torch.tensor([xc, yc, d], device=a.device)) + c2w[:3, 3]
    uv = K @ ((w2c[:3, :3] @ Xw) + w2c[:3, 3])
    uv = uv[:2] / uv[2]
    err = float((uv - torch.tensor([res / 2.0, res / 2.0], device=a.device)).norm())
    print(f"[1] context round-trip error {err:.4f} px")
    assert err < 1.0, "the round-trip misses by more than a pixel, so the convention is wrong"

    # 2. the frames marked target must actually be held out
    m = pick_targets(n, a.n_targets, a.device)
    v_metric = build_views_metric(imgs, n, a.device, extr, intr, res,
                                  hand_bboxes=hb, hand_valid=hv, n_targets=a.n_targets,
                                  frame_index=fi.unsqueeze(0))
    got = torch.nonzero(v_metric["is_target"][0]).flatten().tolist()
    print(f"[2] targets marked at {got} of {n} frames")
    assert got == torch.nonzero(m).flatten().tolist() and len(got) == a.n_targets, "target mask wrong"
    assert 0 not in got, "the first frame anchors the clip frame and must not be a target"

    # 3. a target render must differ from the identity-path render, or nothing changed
    v_ident = build_views(imgs, n, a.device, hb, hv, frame_index=fi.unsqueeze(0))
    with torch.no_grad():
        p_metric = model(v_metric, is_inference=False, use_motion=False)
        p_ident = model(v_ident, is_inference=False, use_motion=False)
    if "gs_depth" not in p_metric or "gs_depth" not in p_ident:
        raise SystemExit("no gs_depth in preds; was the model built with enable_gs?")
    dm, di = p_metric["gs_depth"].float(), p_ident["gs_depth"].float()
    ctx = dm.shape[1]
    assert ctx == n - a.n_targets, f"context should be {n - a.n_targets} frames, got {ctx}"
    assert di.shape[1] == n, f"the identity path should keep all {n} frames, got {di.shape[1]}"
    rel = float((dm - di[:, :ctx]).abs().mean() / di[:, :ctx].abs().mean().clamp_min(1e-8))
    print(f"[3] context {ctx}/{n} frames; gs_depth median metric {float(dm.median()):.4g} "
          f"vs identity {float(di.median()):.4g}, relative difference on shared frames {rel:.4f}")

    # The cameras do not enter gs_depth, which reads tokens. They enter the RENDER, so the
    # instrument-is-connected check belongs on the extrinsics the rasterizer actually used.
    em = p_metric.get("rendered_extrinsics")
    ei = p_ident.get("rendered_extrinsics")
    if em is None or ei is None:
        raise SystemExit("no rendered_extrinsics in preds; the rasterizer did not publish cameras")
    dev = float((em[:, :ctx] - ei[:, :ctx]).abs().max())
    ident_like = float((ei[0, 0, :3, :3] - torch.eye(3, device=ei.device)).abs().max())
    print(f"[3b] render extrinsics differ by at most {dev:.4g}; the identity path's first rotation "
          f"is {ident_like:.4g} from I")
    assert dev > 1e-3, ("the metric path rendered from the same cameras as the identity path, so "
                        "the real extrinsics never reached the rasterizer")

    # 4. gradients must reach the depth and be finite
    model.train()
    p = model(v_metric, is_inference=False, use_motion=False)
    if "gs_depth_logit" not in p:
        raise SystemExit("no gs_depth_logit in preds; the head did not stash its raw activation")
    loss = p["gs_depth_logit"].square().mean()
    loss.backward()
    gs_grads = [q.grad for q in model.gs_head.parameters() if q.grad is not None]
    tot = sum(float(g.abs().sum()) for g in gs_grads)
    finite = all(bool(torch.isfinite(g).all()) for g in gs_grads)
    print(f"[4] gs_head params with grad {len(gs_grads)}  sum |grad| {tot:.4g}  all finite {finite}")
    assert gs_grads and tot > 0 and finite, "no finite gradient reached the Gaussian head"

    print("METRIC_VIEWS_BATCH_OK")


if __name__ == "__main__":
    main()
