#!/usr/bin/env python3
"""Decompose the world error of a composed "<method> + SLAM" row into rotation vs translation.

Context: with the GT camera track our hands give W = 40.8 mm; with the real DROID/HaWoR track the
same hands give W = 128.1 mm. Only the per-frame SE(3) differs, so the whole 87 mm gap is
trajectory error - but a trajectory has two parts, and a long-context trajectory head would have
to be designed for whichever dominates. Forcing the per-seq track SCALE to GT already made things
slightly worse, so scale is not it.

Method: recover the SLAM SE(3) by Kabsch (build_slam_baseline's construction), bring it into the
GT world frame with one global Sim(3) fitted on the camera centres (the world gauge is arbitrary;
W-MPJPE re-aligns on the first window anyway), then emit four composed prediction dirs:
    both   R_slam, t_slam   (sanity: must reproduce the un-aligned +SLAM number)
    rot    R_gt,   t_slam   (rotation given away)
    trans  R_slam, t_gt     (translation given away)
    oracle R_gt,   t_gt     (sanity: must reproduce the GT-track ceiling)
Score all four with eval_worldspace_baseline and the drop tells you which term owns the gap.
"""
import argparse
import glob
import os

import torch

from scripts.build_slam_baseline import recover_se3


def global_rotation_from_rotations(R_src, R_dst):
    """Global Rg minimising sum_k || Rg @ R_src[k] - R_dst[k] ||_F, in closed form as the
    orthogonal projection of sum_k R_dst[k] @ R_src[k]^T.

    Fitting the world gauge from camera CENTRES instead is ill-posed here: egocentric HOI4D camera
    tracks are near-planar (often near-collinear), so a centre-only Umeyama admits a whole family
    of rotations at nearly the same residual, and the recovered Rg is arbitrary. That produced a
    nonsense ~156 deg median rotation error on the first pass. Orientations span SO(3) properly,
    so averaging them is well-conditioned."""
    M = torch.einsum("nij,nkj->ik", R_dst.double(), R_src.double())   # sum R_dst @ R_src^T
    U, _, Vt = torch.linalg.svd(M)
    d = torch.sign(torch.linalg.det(U @ Vt))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.double))
    return (U @ D @ Vt).float()


def geodesic_deg(Ra, Rb):
    """Per-frame geodesic angle between two rotation stacks [N,3,3], in degrees."""
    tr = torch.einsum("nij,nij->n", Ra, Rb)          # trace(Ra @ Rb^T) for orthonormal Ra,Rb
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_pred_dir", required=True, help="per-frame cam hands (ours)")
    ap.add_argument("--slam_pred_dir", required=True, help="dir whose SE(3) is the SLAM track")
    ap.add_argument("--data_root", required=True, help="GT root with cam_extrinsics_cache.pt")
    ap.add_argument("--out_root", required=True, help="parent dir; four <out_root>_<mode> dirs are written")
    a = ap.parse_args()

    modes = ("both", "rot", "trans", "oracle")
    for m in modes:
        os.makedirs(f"{a.out_root}_{m}", exist_ok=True)

    cam_seqs = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.cam_pred_dir, "*.pt"))}
    slam_seqs = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(a.slam_pred_dir, "*.pt"))}
    common = sorted(cam_seqs & slam_seqs)

    rot_errs, cen_errs, n_ok = [], [], 0
    for sq in common:
        ep = os.path.join(a.data_root, sq, "hand_data", "cam_extrinsics_cache.pt")
        if not os.path.exists(ep):
            continue
        cb = torch.load(os.path.join(a.cam_pred_dir, sq + ".pt"), map_location="cpu", weights_only=False)
        hw = torch.load(os.path.join(a.slam_pred_dir, sq + ".pt"), map_location="cpu", weights_only=False)
        R_s, t_s, ok, _ = recover_se3(hw["cam_joints"].float(), hw["world_joints"].float(),
                                      hw["valid"].bool())
        c2w = torch.inverse(torch.load(ep, map_location="cpu").float())
        R_g, t_g = c2w[:, :3, :3], c2w[:, :3, 3]

        cam_b = cb["cam_joints"].float()
        valid_b = cb["valid"].bool()
        n = min(cam_b.shape[0], R_s.shape[0], R_g.shape[0])
        sel = ok[:n]
        if int(sel.sum()) < 10:
            continue
        # Gauge: rotate the SLAM track into the GT world frame using a rotation-averaged Rg, and
        # offset it so the centres share a centroid. NO scale is applied - W-MPJPE re-aligns
        # rigidly on the first window, so a global R/t is free, but a global SCALE is not, and
        # rescaling here would silently turn the "both" sanity row into the oracle-rescale
        # experiment (which is a separate, already-negative gate).
        Rg = global_rotation_from_rotations(R_s[:n][sel], R_g[:n][sel])
        R_sa = torch.einsum("ij,njk->nik", Rg, R_s[:n])
        t_rot = torch.einsum("ij,nj->ni", Rg, t_s[:n])
        t_sa = t_rot + (t_g[:n][sel].mean(0) - t_rot[sel].mean(0))

        rot_errs.append(float(geodesic_deg(R_sa[sel], R_g[:n][sel]).median()))
        cen_errs.append(float((t_sa[sel] - t_g[:n][sel]).norm(dim=-1).median()))

        for m in modes:
            Ru = R_g[:n] if m in ("rot", "oracle") else R_sa
            tu = t_g[:n] if m in ("trans", "oracle") else t_sa
            world = torch.full_like(cam_b[:n], float("nan"))
            for k in range(n):
                if not sel[k]:
                    continue
                for h in range(2):
                    world[k, h] = (Ru[k] @ cam_b[k, h].reshape(-1, 3).T).T + tu[k]
            torch.save({"cam_joints": cam_b[:n], "world_joints": world,
                        "valid": (valid_b[:n] & sel.unsqueeze(1))},
                       os.path.join(f"{a.out_root}_{m}", sq + ".pt"))
        n_ok += 1

    if rot_errs:
        r = torch.tensor(rot_errs)
        c = torch.tensor(cen_errs)
        print(f"[decompose] seqs={n_ok}  SLAM-vs-GT (after global Sim3): "
              f"rotation median {float(r.median()):.2f} deg (max {float(r.max()):.2f}) | "
              f"centre median {float(c.median())*1000:.1f} mm (max {float(c.max())*1000:.1f})",
              flush=True)
    print(f"DECOMPOSE_DONE wrote {n_ok} seqs x {len(modes)} modes -> {a.out_root}_<mode>", flush=True)


if __name__ == "__main__":
    main()
