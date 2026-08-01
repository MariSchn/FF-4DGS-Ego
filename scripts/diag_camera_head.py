#!/usr/bin/env python3
"""Diagnose the FROZEN camera head against ground-truth extrinsics, per clip.

WHY THIS EXISTS. The long-window table shows our hand pose is within 2 mm of its own oracle
(C-MPJPE_abs 35.2 vs 33.0) while W-MPJPE is 200.9 vs 61.5 with a ground-truth camera track.
So essentially all long-window global error is camera-trajectory error. Worse, an earlier
diagnostic found that substituting IDENTITY camera poses scores W=140 against 202.7 for our
predicted ones: the frozen head is actively worse than assuming the camera never moves.

That is a suspicious signature. "Worse than identity" is what you get from a wrong CONVENTION
or a wrong SCALE, not merely from a weak predictor. Before paying for a camera-head fine-tune we
should know which, because the three causes have completely different fixes:

  * rotation wrong             -> genuine fine-tune needed (expensive)
  * translation DIRECTION wrong-> genuine fine-tune needed (expensive)
  * only the SCALE wrong       -> fix the scale estimator (cheap, possibly no training at all)
  * systematically inverted    -> a convention bug (free to fix, and it would explain everything)

This script separates those. It reports, per clip, against GT extrinsics:
  R_err      geodesic rotation error in degrees (scale-invariant)
  dir_err    angle between predicted and GT translation directions (scale-invariant)
  s_opt      the per-clip least-squares scale that best matches predicted to GT translation
  t_err_raw  translation error with the pipeline's own scene scale applied
  t_err_opt  translation error under s_opt: the floor a perfect scale estimator could reach
and the same three numbers for two references:
  IDENTITY   pretending the camera never moves (the bar the head must clear)
  INVERTED   the predicted trajectory with c2w/w2c swapped (catches a convention bug)

Read the output like this:
  R_err small and dir_err small but t_err_raw >> t_err_opt -> SCALE estimation is the problem.
  INVERTED beating the prediction                          -> convention bug, stop and fix it.
  R_err large                                              -> the head genuinely needs training.
"""
from __future__ import annotations

import argparse
import os
from typing import NamedTuple

import numpy as np
import torch
import yaml


class ClipStat(NamedTuple):
    """Per-clip camera diagnostics. All errors in the units noted; angles in degrees."""

    seq: str
    frame_offset: int
    rot_err_deg: float
    dir_err_deg: float
    s_opt: float
    t_err_raw_mm: float
    t_err_opt_mm: float
    ident_t_err_mm: float
    ident_rot_err_deg: float
    inv_rot_err_deg: float
    gt_motion_mm: float


def _rel_to_first(mats: torch.Tensor) -> torch.Tensor:
    """Normalise a [S,4,4] pose sequence so frame 0 is the identity.

    The camera head predicts in a canonical frame whose origin is the first frame, so GT must be
    put in the same frame before any comparison. Without this every number below is meaningless.
    """
    inv0 = torch.linalg.inv(mats[0])
    return torch.matmul(inv0.unsqueeze(0), mats)


def _geodesic_deg(r_a: torch.Tensor, r_b: torch.Tensor) -> torch.Tensor:
    """Per-frame geodesic angle between two [S,3,3] rotation stacks, in degrees."""
    rel = torch.matmul(r_a.transpose(-1, -2), r_b)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def _dir_err_deg(t_pred: torch.Tensor, t_gt: torch.Tensor) -> float:
    """Angle between the predicted and GT translation trajectories, ignoring scale.

    Uses only frames where the GT camera has actually moved; on a still camera the direction is
    undefined and would otherwise contribute arbitrary noise.
    """
    n_pred = torch.linalg.norm(t_pred, dim=-1)
    n_gt = torch.linalg.norm(t_gt, dim=-1)
    moved = (n_gt > 1e-4) & (n_pred > 1e-9)
    if not bool(moved.any()):
        return float("nan")
    cos = (t_pred[moved] * t_gt[moved]).sum(-1) / (n_pred[moved] * n_gt[moved])
    return float(torch.rad2deg(torch.acos(torch.clamp(cos, -1.0, 1.0))).mean())


def _optimal_scale(t_pred: torch.Tensor, t_gt: torch.Tensor) -> float:
    """Least-squares scale s minimising ||s * t_pred - t_gt||. Zero if the prediction is degenerate."""
    denom = float((t_pred * t_pred).sum())
    if denom < 1e-12:
        return 0.0
    return float((t_pred * t_gt).sum() / denom)


def analyse_clip(c2w_pred: torch.Tensor, gt_w2c: torch.Tensor, scene_scale: float) -> ClipStat | None:
    """Compare one clip's predicted c2w against GT extrinsics.

    Args:
        c2w_pred: [S,4,4] predicted camera-to-world (what the head emits).
        gt_w2c:   [S,4,4] ground-truth world-to-camera (what the store holds).
        scene_scale: the per-clip scale the pipeline itself applies to camera translation.
    """
    if c2w_pred.shape[0] != gt_w2c.shape[0] or c2w_pred.shape[0] < 2:
        return None
    if not (torch.isfinite(c2w_pred).all() and torch.isfinite(gt_w2c).all()):
        return None

    gt_c2w = torch.linalg.inv(gt_w2c.double())
    gt_rel = _rel_to_first(gt_c2w)
    pr_rel = _rel_to_first(c2w_pred.double())

    r_gt, r_pr = gt_rel[:, :3, :3], pr_rel[:, :3, :3]
    t_gt, t_pr = gt_rel[:, :3, 3], pr_rel[:, :3, 3]

    # Reference 1: identity, i.e. assume the camera never moves. This is the bar to clear.
    eye = torch.eye(3, dtype=r_gt.dtype).unsqueeze(0).expand_as(r_gt)
    ident_rot = float(_geodesic_deg(eye, r_gt).mean())
    ident_t = float(torch.linalg.norm(t_gt, dim=-1).mean() * 1000.0)

    # Reference 2: the same prediction with the c2w/w2c convention swapped.
    inv_rel = _rel_to_first(torch.linalg.inv(c2w_pred.double()))
    inv_rot = float(_geodesic_deg(inv_rel[:, :3, :3], r_gt).mean())

    s_opt = _optimal_scale(t_pr, t_gt)
    return ClipStat(
        seq="",
        frame_offset=-1,
        rot_err_deg=float(_geodesic_deg(r_pr, r_gt).mean()),
        dir_err_deg=_dir_err_deg(t_pr, t_gt),
        s_opt=s_opt,
        t_err_raw_mm=float(torch.linalg.norm(t_pr * scene_scale - t_gt, dim=-1).mean() * 1000.0),
        t_err_opt_mm=float(torch.linalg.norm(t_pr * s_opt - t_gt, dim=-1).mean() * 1000.0),
        ident_t_err_mm=ident_t,
        ident_rot_err_deg=ident_rot,
        inv_rot_err_deg=inv_rot,
        gt_motion_mm=ident_t,
    )


def _summarise(stats: list[ClipStat]) -> None:
    """Print the aggregate table and the verdict it implies."""
    if not stats:
        print("no usable clips - nothing to diagnose")
        return

    def med(attr: str) -> float:
        vals = [getattr(s, attr) for s in stats]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    print("\n" + "=" * 78)
    print(f"CAMERA-HEAD DIAGNOSTIC over {len(stats)} clips (medians)")
    print("=" * 78)
    print(f"  GT camera motion over a clip      {med('gt_motion_mm'):8.1f} mm")
    print("  -- rotation (scale-invariant) --")
    print(f"  predicted rotation error          {med('rot_err_deg'):8.2f} deg")
    print(f"  IDENTITY rotation error           {med('ident_rot_err_deg'):8.2f} deg   <- bar to clear")
    print(f"  INVERTED (c2w/w2c swap) rot error {med('inv_rot_err_deg'):8.2f} deg   <- convention check")
    print("  -- translation --")
    print(f"  direction error                   {med('dir_err_deg'):8.2f} deg")
    print(f"  optimal per-clip scale s_opt      {med('s_opt'):8.4f}")
    print(f"  error with pipeline scale         {med('t_err_raw_mm'):8.1f} mm")
    print(f"  error with OPTIMAL scale          {med('t_err_opt_mm'):8.1f} mm   <- floor for a scale fix")
    print(f"  IDENTITY translation error        {med('ident_t_err_mm'):8.1f} mm   <- bar to clear")

    print("\n--- VERDICT ---")
    rot, ident_rot, inv_rot = med("rot_err_deg"), med("ident_rot_err_deg"), med("inv_rot_err_deg")
    raw, opt, ident_t = med("t_err_raw_mm"), med("t_err_opt_mm"), med("ident_t_err_mm")

    if np.isfinite(inv_rot) and inv_rot < rot * 0.7:
        print("  !! CONVENTION BUG: the INVERTED trajectory fits GT far better than the predicted one.")
        print("     Fix the c2w/w2c handling before considering any training.")
    elif rot > ident_rot:
        print("  !! Rotation is WORSE than assuming a static camera. The head is not usable as-is")
        print("     on this data; a fine-tune (or a rotation source swap) is required.")
    else:
        print(f"  Rotation beats identity ({rot:.2f} vs {ident_rot:.2f} deg): the head has real signal.")

    if np.isfinite(raw) and np.isfinite(opt) and opt < raw * 0.5:
        print(f"  SCALE IS THE DOMINANT TRANSLATION ERROR: {raw:.1f} mm -> {opt:.1f} mm under a perfect")
        print("     per-clip scale. Fixing scale estimation is the cheap, high-yield lever.")
    elif np.isfinite(opt) and np.isfinite(ident_t) and opt > ident_t:
        print("  Even under a PERFECT scale the prediction loses to a static camera: the translation")
        print("     direction itself is wrong, so a fine-tune is required.")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--max_seqs", type=int, default=12)
    ap.add_argument("--max_clips_per_seq", type=int, default=3)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force_enable_cam", action="store_true",
                    help="Build the camera head even if the config disables it. The head's weights "
                         "come from the base checkpoint either way, so this diagnoses the same "
                         "frozen camera head the world eval uses, from any training config.")
    args = ap.parse_args()

    from scripts.eval_world_space import build_model
    from scripts.hand_vis_utils import MANOModel
    from scripts.train_hand_head import HOT3DHandDataset

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = cfg["model"]
    if args.force_enable_cam and not mcfg.get("enable_cam", True):
        print("[force_enable_cam] config had enable_cam: false - building the camera head anyway")
        mcfg["enable_cam"] = True

    # The camera head only exists if the model was built with it; make that explicit rather than
    # letting `camera_poses` silently go missing and reporting a meaningless all-NaN table.
    model = build_model(cfg, device)
    if not hasattr(model, "cam_head"):
        raise SystemExit("this model has no cam_head - nothing to diagnose")

    # Use the repo's MANOModel wrapper, exactly as eval_world_space does. Calling smplx.create
    # directly fails here: it appends model_type as a subdirectory and looks for
    # models/MANO/mano/MANO_RIGHT.pkl, whereas the checkout stores models/MANO/MANO_RIGHT.pkl.
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])

    seqs = sorted(d for d in os.listdir(args.data_root)
                  if os.path.isdir(os.path.join(args.data_root, d)))[: args.max_seqs]
    stats: list[ClipStat] = []

    for sq in seqs:
        seq_dir = os.path.join(args.data_root, sq)
        try:
            ds = HOT3DHandDataset([seq_dir], mano, num_frames=args.clip_len, clip_stride=args.stride,
                                  use_hand_crop=mcfg.get("use_hand_crop", False),
                                  rescale_factor=cfg.get("hand_crop", {}).get("rescale_factor", 1.5))
        except Exception as exc:                                  # noqa: BLE001 - report and continue
            print(f"[{sq}] dataset build failed: {exc}")
            continue
        if len(ds) == 0:
            continue

        for ci in range(min(len(ds), args.max_clips_per_seq)):
            item = ds[ci]
            if "cam_extrinsics" not in item:
                print(f"[{sq}] no cam_extrinsics in the store - skipping "
                      f"(this sequence cannot be diagnosed)")
                break
            gt_w2c = item["cam_extrinsics"].float()               # [S,4,4] T_cam_world

            from scripts.train_hand_head import build_views
            imgs = item["images"].unsqueeze(0).to(device)
            hb = item.get("hand_bboxes")
            hv = item.get("hand_valid")
            if hb is not None:
                hb = hb.unsqueeze(0).to(device)
            if hv is not None:
                hv = hv.unsqueeze(0).to(device)

            with torch.no_grad():
                views = build_views(imgs, args.clip_len, device, hb, hv)
                preds = model(views, is_inference=True, use_motion=False)

            if "camera_poses" not in preds:
                raise SystemExit("model produced no camera_poses - is enable_cam off in the config?")
            c2w = preds["camera_poses"][0].float().cpu()          # [S,4,4]

            # The pipeline scales camera translation by a per-clip scene scale; use the same
            # source it does so t_err_raw reflects what the eval actually experiences.
            scene_scale = float(preds.get("metric_scale", torch.tensor(1.0)).reshape(-1)[0]) \
                if "metric_scale" in preds else 1.0

            st = analyse_clip(c2w, gt_w2c, scene_scale)
            if st is None:
                continue
            st = st._replace(seq=sq, frame_offset=ci * args.stride)
            stats.append(st)
            print(f"[{sq} c{ci}] R={st.rot_err_deg:6.2f}deg (id {st.ident_rot_err_deg:6.2f}) "
                  f"dir={st.dir_err_deg:6.2f}deg s_opt={st.s_opt:7.4f} "
                  f"t_raw={st.t_err_raw_mm:7.1f} t_opt={st.t_err_opt_mm:7.1f} "
                  f"(id {st.ident_t_err_mm:7.1f}) mm", flush=True)

    _summarise(stats)

    if args.out:
        import json
        with open(args.out, "w") as fh:
            json.dump({"clips": [s._asdict() for s in stats],
                       "protocol": {"clip_len": args.clip_len, "stride": args.stride}}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
