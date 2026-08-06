"""C-MPJPE / PA / WRIST of HaWoR camera-frame joints vs H2O subject4 GT.

HaWoR outputs joints in OpenPose-21 order (lib/models/mano_wrapper.py joint_map):
    mano_to_openpose = [0,13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20]
indexing the concat [16 MANO kinematic joints (smplx order) ++ 5 tip verts
(thumb,index,middle,ring,pinky)].

GT (from extract_h2o_clips_mp4.py / eval_cmpjpe.py) is in the MANO-21 layout used by
our model: 16 base joints in order [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19] of the
H2O->MANO-21 array, then 5 tips [thumb,index,middle,ring,pinky].

We map HaWoR OpenPose-21 -> the GT layout, then report C-MPJPE (no align), PA-MPJPE,
RR-MPJPE, WRIST. PA-MPJPE is permutation-sensitive but scale/pose invariant; we use it
as a sanity check that the joint remap is right (a wrong permutation blows PA up).

Metric math (_per_joint_mm, _pa_mpjpe_mm) copied verbatim from scripts/eval_cmpjpe.py.
"""
import argparse
import glob
import os

import numpy as np
import torch

# --- GT layout indices (verbatim from extract / eval_cmpjpe) ---
H2O_TO_MANO = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
MANO21_TO_16 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

# HaWoR OpenPose-21 joint_map (position p holds concat-index mano_to_openpose[p]).
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

# --- Build HaWoR_openpose21 -> GT_layout21 permutation ---
# GT layout (positions 0..20) corresponds to these concat-C indices:
#   base 16 = smplx MANO joints in order [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]? NO.
# eval_cmpjpe's GT 16 are the kinematic MANO joints our model emits; our model and HaWoR
# both use smplx MANO, so the 16 base in the concat are smplx joints 0..15 in tree order.
# The GT 16-order picks H2O joints remapped to MANO-21 then subset MANO21_TO_16, which is
# the standard MANO 16-kinematic order == smplx joints [0..15] WITHOUT the missing tips.
# In the concat C: C0..C15 == smplx joints 0..15 (the 16 kinematic), C16..C20 == tips
# [thumb,index,middle,ring,pinky]. The GT base order is the SAME smplx 0..15 order. So:
#   GT layout C-indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] + [16,17,18,19,20]
# i.e. GT base position j -> concat C index j (identity for the 16), tips appended.
# BUT eval_cmpjpe GT base order = MANO21_TO_16 = [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
# which are positions in the MANO-21 (openpose-like) array, NOT smplx-16. The two
# conventions can differ. We therefore try BOTH a direct and a kinematic mapping and let
# PA-MPJPE pick the consistent one.

def openpose21_to_concatC(j_op):
    """[...,21,3] OpenPose-21 -> [...,21,3] concat-C order (C0..C20)."""
    inv = np.zeros(21, dtype=int)
    for pos, c in enumerate(MANO_TO_OPENPOSE):
        inv[c] = pos                    # concat index c is at openpose position pos
    return j_op[..., inv, :]


def concatC_to_gtlayout(j_c, variant):
    """Map concat-C (C0..C15 smplx joints, C16..20 tips) -> GT 21 layout."""
    if variant == "identity":
        # GT base = C0..C15 in natural order, tips C16..20.
        idx = list(range(16)) + [16, 17, 18, 19, 20]
    elif variant == "mano21sub":
        # GT base order = MANO21_TO_16 positions of a MANO-21 array. Treat concat-C as if
        # already MANO-21-ish: pick MANO21_TO_16 then tips.
        idx = MANO21_TO_16 + [16, 17, 18, 19, 20]
    else:
        raise ValueError(variant)
    return j_c[..., idx, :]


def _per_joint_mm(pred, gt):
    return 1000.0 * torch.sqrt(((pred - gt) ** 2).sum(-1))


def _pa_mpjpe_mm(pred, gt):
    p, g = pred.double().cpu(), gt.double().cpu()
    mu_p, mu_g = p.mean(1, keepdim=True), g.mean(1, keepdim=True)
    x1, x2 = p - mu_p, g - mu_g
    var1 = (x1 ** 2).sum((1, 2)).clamp(min=1e-12)
    k = x1.transpose(1, 2) @ x2
    u, s, vh = torch.linalg.svd(k)
    v = vh.transpose(1, 2)
    z = torch.eye(3, dtype=p.dtype, device=p.device).repeat(len(p), 1, 1)
    z[:, 2, 2] = torch.sign(torch.linalg.det(u @ v.transpose(1, 2)))
    r = v @ z @ u.transpose(1, 2)
    scale = torch.diagonal(r @ k, dim1=1, dim2=2).sum(1) / var1
    aligned = scale.view(-1, 1, 1) * (x1 @ r.transpose(1, 2)) + mu_g
    return (1000.0 * torch.sqrt(((aligned - g) ** 2).sum(-1))).mean(-1)


def collect_pairs(pred_dir, gt_dir):
    """Return list of (pred_cam[2,T,21,3], gt[2,T,21,3], valid_pred[2,T], valid_gt[2,T])."""
    pairs = []
    for pf in sorted(glob.glob(os.path.join(pred_dir, "*_pred.npz"))):
        clip = os.path.basename(pf).replace("_pred.npz", "")
        gf = os.path.join(gt_dir, f"{clip}_gt.npz")
        if not os.path.exists(gf):
            print(f"  no GT for {clip}, skip")
            continue
        P = np.load(pf)
        G = np.load(gf)
        pc = P["joints_cam"]            # [2,Tp,21,3]
        gj = G["joints21"]              # [Ng,2,21,3]
        gv = G["valid"]                 # [Ng,2]
        # pred valid: [2,Tp] (may be bool). HaWoR fills all frames; restrict to detected.
        pv = P["valid"] if "valid" in P.files else np.ones(pc.shape[:2], bool)
        T = min(pc.shape[1], gj.shape[0])
        pc = pc[:, :T]                  # [2,T,21,3]
        gj = np.transpose(gj[:T], (1, 0, 2, 3))   # [2,T,21,3]
        gv = np.transpose(gv[:T], (1, 0))         # [2,T]
        pv = pv[:, :T]
        pairs.append((clip, pc, gj, pv, gv))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--variant", default="auto", choices=["auto", "identity", "mano21sub"])
    args = ap.parse_args()

    pairs = collect_pairs(args.pred_dir, args.gt_dir)
    print(f"{len(pairs)} clips matched")
    if not pairs:
        return

    variants = ["identity", "mano21sub"] if args.variant == "auto" else [args.variant]
    best = None
    for var in variants:
        cmp_all, rr_all, wr_all, pa_all = [], [], [], []
        nh = 0
        for clip, pc, gj, pv, gv in pairs:
            # pc OpenPose-21 -> concat-C -> GT layout
            pc_c = openpose21_to_concatC(pc)
            pc_g = concatC_to_gtlayout(pc_c, var)           # [2,T,21,3]
            both_valid = (pv > 0.5) & (gv > 0.5)            # [2,T]
            for h in range(2):
                sel = both_valid[h]
                if sel.sum() < 1:
                    continue
                P = torch.tensor(pc_g[h][sel], dtype=torch.float64)   # [n,21,3]
                Gt = torch.tensor(gj[h][sel], dtype=torch.float64)
                if P.shape[0] == 0:
                    continue
                cmp_all.append(_per_joint_mm(P, Gt).mean(-1))         # [n]
                wr_all.append(_per_joint_mm(P[:, :1], Gt[:, :1]).mean(-1))
                Pr = P - P[:, :1]; Gr = Gt - Gt[:, :1]
                rr_all.append(_per_joint_mm(Pr, Gr).mean(-1))
                pa_all.append(_pa_mpjpe_mm(P, Gt))
                nh += P.shape[0]
        if not cmp_all:
            print(f"[{var}] no valid hands"); continue
        cmpv = torch.cat(cmp_all); rrv = torch.cat(rr_all)
        wrv = torch.cat(wr_all); pav = torch.cat(pa_all)
        res = dict(variant=var, n_hands=nh,
                   cmpjpe=cmpv.mean().item(), cmpjpe_med=cmpv.median().item(),
                   rr=rrv.mean().item(), wrist=wrv.mean().item(), pa=pav.mean().item())
        print(f"[{var}] N={nh} C-MPJPE={res['cmpjpe']:.1f} (med {res['cmpjpe_med']:.1f}) "
              f"PA={res['pa']:.1f} RR={res['rr']:.1f} WRIST={res['wrist']:.1f} mm")
        if best is None or res["pa"] < best["pa"]:
            best = res
    print("\n=== BEST (lowest PA-MPJPE -> correct joint remap) ===")
    print(f"variant={best['variant']}  N_hands={best['n_hands']}")
    print(f"C-MPJPE = {best['cmpjpe']:.1f} mm (median {best['cmpjpe_med']:.1f})")
    print(f"PA-MPJPE= {best['pa']:.1f} mm   RR-MPJPE={best['rr']:.1f} mm   WRIST={best['wrist']:.1f} mm")


if __name__ == "__main__":
    main()
