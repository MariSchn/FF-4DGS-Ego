#!/usr/bin/env python3
"""Audit the camera trajectory implied by a world-space prediction dir against GT extrinsics.

Two different "+SLAM" prediction dirs gave wildly different world error for the SAME hand
predictions (W 40.8 vs 128.1), which can only come from the trajectory they encode. A +SLAM
pred file stores world = R_k @ cam + t_k, so the per-frame SE(3) is recoverable exactly by Kabsch
on its own cam<->world pairs (that is what build_slam_baseline does). This script recovers that
SE(3), turns it into a camera-centre track, aligns it to the GT camera centres with a single
global Sim(3) (Umeyama - the gauge is arbitrary, only the SHAPE of the track is a property of the
method), and reports the residual. A dir whose track matches GT is a good trajectory; a dir whose
track IS GT (residual ~0) is an oracle and must never be reported as a SLAM result.

Usage:
  python -m scripts.audit_slam_trajectory --data_root <hoi4d test dir> --pred_dir <dir> [--max_seqs N]
"""
import argparse
import glob
import os

import torch


def recover_se3(cam, world, valid):
    """Per-frame rigid R[k], t[k] with world[k] ~= R[k] @ cam[k] + t[k]. Mirrors build_slam_baseline."""
    n = cam.shape[0]
    R = torch.eye(3).repeat(n, 1, 1)
    t = torch.zeros(n, 3)
    ok = torch.zeros(n, dtype=torch.bool)
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
        R[k], t[k], ok[k] = Rk.float(), (Qc - Rk @ Pc).float(), True
    return R, t, ok


def umeyama_sim3(P, Q):
    """Least-squares s,R,t with Q ~= s*R@P + t (both [N,3]). Returns (s, R, t)."""
    P, Q = P.double(), Q.double()
    Pc, Qc = P.mean(0), Q.mean(0)
    X, Y = P - Pc, Q - Qc
    H = X.T @ Y
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.double))
    R = Vt.T @ D @ U.T
    var = (X ** 2).sum()
    s = float((S * torch.tensor([1.0, 1.0, d], dtype=torch.double)).sum() / var.clamp_min(1e-12))
    return s, R, (Qc - s * R @ Pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--max_seqs", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.pred_dir, "*.pt")))
    if a.max_seqs:
        files = files[: a.max_seqs]
    errs, scales, n_used = [], [], 0
    for f in files:
        sq = os.path.basename(f)[:-3]
        ext_p = os.path.join(a.data_root, sq, "hand_data", "cam_extrinsics_cache.pt")
        if not os.path.exists(ext_p):
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        if "world_joints" not in d:
            continue
        R, t, ok = recover_se3(d["cam_joints"].float(), d["world_joints"].float(), d["valid"].bool())
        # camera centre in the prediction's world frame: world = R@cam + t, and the camera centre
        # is cam-frame origin -> centre_pred = t.
        w2c = torch.load(ext_p, map_location="cpu").float()          # [N,4,4] world->cam
        c2w = torch.inverse(w2c)
        gt_c = c2w[:, :3, 3]                                          # GT camera centres (metric)
        m = min(t.shape[0], gt_c.shape[0])
        sel = ok[:m]
        if int(sel.sum()) < 10:
            continue
        P, Q = t[:m][sel], gt_c[:m][sel]
        if float((P - P.mean(0)).norm(dim=-1).max()) < 1e-4:
            continue
        s, Rg, tg = umeyama_sim3(P, Q)
        res = ((s * (Rg @ P.double().T).T + tg) - Q.double()).norm(dim=-1)
        errs.append(float(res.median()))
        scales.append(s)
        n_used += 1

    if not errs:
        print(f"[audit] {a.pred_dir}: NO usable sequences")
        return
    e = torch.tensor(errs)
    sc = torch.tensor(scales)
    print(f"[audit] {a.pred_dir}")
    print(f"  seqs={n_used}  camera-centre error vs GT after global Sim(3): "
          f"median {float(e.median())*1000:.1f} mm | mean {float(e.mean())*1000:.1f} mm | "
          f"max {float(e.max())*1000:.1f} mm")
    print(f"  Sim(3) scale to GT: median {float(sc.median()):.4f} "
          f"(1.0 => already metric; far from 1.0 => up-to-scale track)")
    if float(e.median()) < 1e-3:
        print("  !! residual ~0 -> this trajectory IS the GT camera track (ORACLE, not SLAM)")


if __name__ == "__main__":
    main()
