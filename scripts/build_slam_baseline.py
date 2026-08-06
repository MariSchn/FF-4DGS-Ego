#!/usr/bin/env python3
"""Build a "<method> + SLAM" world-space baseline by composing a per-frame camera-frame hand
predictor (vanilla HaMeR / WiLoR, cached as {cam_joints,world_joints,valid}) with the camera
trajectory that our completed HaWoR run already produced. This is exactly the standard
"per-frame hand model + DROID-SLAM trajectory" construction (as in the Hand3R HOI4D table),
except the SLAM trajectory is shared verbatim with our HaWoR self-run row, so the three
offline rows differ ONLY in the per-frame predictor - the most controlled possible comparison.

The camera trajectory is recovered per frame from the HaWoR prediction itself: HaWoR places its
metric camera-frame hand into the world by a rigid SE(3) (world = R_k @ cam + t_k, with the
metric scale baked into t_k via Metric3D). So for each frame we solve that same R_k,t_k by
Kabsch on HaWoR's own (cam_joints -> world_joints) correspondences (residual ~0, asserted), then
re-place the baseline's cam_joints through the identical R_k,t_k. cam_joints pass through
unchanged (they drive the camera-frame C-MPJPE); only world_joints are (re)computed.

Output per seq: <out_dir>/<seq>.pt with the eval_worldspace_baseline contract
{cam_joints[N,2,16,3], world_joints[N,2,16,3], valid[N,2]}.
"""
import argparse
import glob
import os

import torch


def recover_se3(cam, world, valid):
    """Per-frame rigid R[k],t[k] with world[k] ~= R[k]@cam[k]+t[k], from HaWoR's own
    cam<->world joints. Returns R[N,3,3], t[N,3], ok[N] bool, and the max residual (m)."""
    n = cam.shape[0]
    R = torch.eye(3).repeat(n, 1, 1)
    t = torch.zeros(n, 3)
    ok = torch.zeros(n, dtype=torch.bool)
    max_res = 0.0
    for k in range(n):
        pc, pw = [], []
        for h in range(2):
            if valid[k, h] and torch.isfinite(cam[k, h]).all() and torch.isfinite(world[k, h]).all():
                pc.append(cam[k, h].reshape(-1, 3))
                pw.append(world[k, h].reshape(-1, 3))
        if not pc:
            continue
        P = torch.cat(pc, 0).double()
        Q = torch.cat(pw, 0).double()
        if P.shape[0] < 3:
            continue
        Pc, Qc = P.mean(0), Q.mean(0)
        H = (P - Pc).T @ (Q - Qc)
        U, _, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.double))
        Rk = Vt.T @ D @ U.T
        tk = Qc - Rk @ Pc
        res = float(((Rk @ P.T).T + tk - Q).norm(dim=-1).max())
        max_res = max(max_res, res)
        R[k], t[k], ok[k] = Rk.float(), tk.float(), True
    return R, t, ok, max_res


def rescale_track_to_gt(t, ok, gt_centres):
    """ORACLE DIAGNOSTIC. Rescale the recovered camera-centre track about its own centroid so its
    Sim(3) scale to the GT track becomes 1, keeping shape and gauge.

    Motivation: audit_slam_trajectory showed the DROID/HaWoR track matches GT in SHAPE to ~15 mm
    but carries a ~5.6% Sim(3) SCALE error. W-MPJPE aligns on the first 30-frame window and never
    re-solves scale, so a few percent compounds over a metre-scale camera excursion into ~100 mm of
    world error. This isolates that term: if W collapses when the scale is fixed, trajectory SCALE
    (not drift) is what caps the +SLAM rows, and a GT-free scale estimator becomes the lever.
    Returns (t_rescaled, s) with s=1.0 when the track is degenerate or GT is unavailable."""
    m = min(t.shape[0], gt_centres.shape[0])
    sel = ok[:m]
    if int(sel.sum()) < 10:
        return t, 1.0
    P, Q = t[:m][sel].double(), gt_centres[:m][sel].double()
    Pc, Qc = P.mean(0), Q.mean(0)
    X, Y = P - Pc, Q - Qc
    if float(X.norm(dim=-1).max()) < 1e-4:
        return t, 1.0
    U, S, Vt = torch.linalg.svd(X.T @ Y)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    s = float((S * torch.tensor([1.0, 1.0, d], dtype=torch.double)).sum() / (X ** 2).sum().clamp_min(1e-12))
    if not (0.1 < s < 10.0):
        return t, 1.0
    c = t[:m][sel].mean(0)
    return c + (t - c) * s, s


def compose_world(cam_b, R, t, ok):
    """world[k,h] = R[k] @ cam_b[k,h] + t[k]; NaN where the SLAM trajectory is unavailable."""
    n = cam_b.shape[0]
    world = torch.full_like(cam_b, float("nan"))
    for k in range(n):
        if not ok[k]:
            continue
        for h in range(2):
            world[k, h] = (R[k] @ cam_b[k, h].reshape(-1, 3).T).T + t[k]
    return world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_pred_dir", required=True, help="dir of per-frame cam preds (vanilla HaMeR/WiLoR)")
    ap.add_argument("--slam_pred_dir", required=True, help="HaWoR preds dir (SLAM-trajectory source)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--res_tol", type=float, default=1e-3, help="assert HaWoR SE(3) residual below this (m)")
    ap.add_argument("--rescale_traj_to_gt", default="",
                    help="ORACLE DIAGNOSTIC: data_root with cam_extrinsics_cache.pt. Fixes the SLAM "
                         "track's Sim(3) scale to GT (shape untouched) to isolate the trajectory-"
                         "scale term of W. Never a reportable method result.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cam_seqs = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.cam_pred_dir, "*.pt"))}
    slam_seqs = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.slam_pred_dir, "*.pt"))}
    common = sorted(cam_seqs & slam_seqs)
    print(f"[build_slam_baseline] cam={len(cam_seqs)} slam={len(slam_seqs)} common={len(common)}", flush=True)

    n_ok, worst_res, scales_tr = 0, 0.0, []
    for sq in common:
        cb = torch.load(os.path.join(a.cam_pred_dir, sq + ".pt"), map_location="cpu", weights_only=False)
        hw = torch.load(os.path.join(a.slam_pred_dir, sq + ".pt"), map_location="cpu", weights_only=False)
        cam_b = cb["cam_joints"].float()
        valid_b = cb["valid"].bool()
        R, t, ok, res = recover_se3(hw["cam_joints"].float(), hw["world_joints"].float(), hw["valid"].bool())
        worst_res = max(worst_res, res)
        if a.rescale_traj_to_gt:
            ep = os.path.join(a.rescale_traj_to_gt, sq, "hand_data", "cam_extrinsics_cache.pt")
            if os.path.exists(ep):
                gt_c = torch.inverse(torch.load(ep, map_location="cpu").float())[:, :3, 3]
                t, s_tr = rescale_track_to_gt(t, ok, gt_c)
                scales_tr.append(s_tr)
        m = min(cam_b.shape[0], R.shape[0])
        world_b = compose_world(cam_b[:m], R[:m], t[:m], ok[:m])
        out = {
            "cam_joints": cam_b[:m],
            "world_joints": world_b,
            "valid": (valid_b[:m] & ok[:m].unsqueeze(1)),
        }
        torch.save(out, os.path.join(a.out_dir, sq + ".pt"))
        n_ok += 1
    assert worst_res < a.res_tol, (
        f"HaWoR SE(3) recovery residual {worst_res:.4f} m exceeds tol {a.res_tol} - "
        f"world!=R@cam+t, composition invalid")
    tr = ""
    if scales_tr:
        st = torch.tensor(scales_tr)
        tr = (f" | ORACLE traj rescale applied to {len(scales_tr)} seqs, "
              f"s median {float(st.median()):.4f} min {float(st.min()):.4f} max {float(st.max()):.4f}")
    # Record what this composition actually is. The output is a SLAM-composed world track, which
    # is what makes it comparable with the other table rows; the sibling per-frame dirs carry a
    # GT-extrinsics lift instead and are oracles. Those two were indistinguishable except by
    # directory name until 2026-08-06 (task #67). Box source is inherited from the cam preds when
    # they carry their own record, since this script never sees the boxes.
    from scripts.pred_provenance import TRAJ_SLAM, read_provenance, write_provenance
    _cam_prov = read_provenance(a.cam_pred_dir) or {}
    write_provenance(
        a.out_dir,
        box_source=_cam_prov.get("box_source", f"INHERITED_UNRECORDED_FROM:{a.cam_pred_dir}"),
        trajectory_source=TRAJ_SLAM,
        produced_by="scripts/build_slam_baseline.py",
        n_seqs=n_ok,
        cam_pred_dir=a.cam_pred_dir,
        slam_pred_dir=a.slam_pred_dir,
        max_se3_residual_mm=round(worst_res * 1000.0, 4),
        oracle_traj_rescale=bool(scales_tr),
    )
    print(f"BUILD_SLAM_BASELINE_DONE wrote {n_ok}/{len(common)} seqs -> {a.out_dir} "
          f"(max SE3 residual {worst_res*1000:.3f} mm){tr}", flush=True)


if __name__ == "__main__":
    main()
