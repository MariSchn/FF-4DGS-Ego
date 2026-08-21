"""How much error does MANO interpolation add, as a function of the temporal gap?

Frame i is reconstructed from frames i-g and i+g through `mano_interp`, decoded to joints, and
compared against the GT at i. GT is the oracle only; nothing here trains. If this curve reaches
tens of millimetres at the gaps the held-out protocol uses, occlusion measurements built on the
interpolated hand would be ambiguous and the protocol must shrink its gaps first.
"""
from __future__ import annotations

import argparse
import os

import torch

from scripts.mano_interp import interp_hand_params
from scripts.train_hand_head import compute_joints_from_batch
from scripts.hand_vis_utils import MANOModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--mano", required=True, help="MANO model folder")
    ap.add_argument("--n_seqs", type=int, default=6)
    ap.add_argument("--frames_per_seq", type=int, default=24)
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()

    mano = MANOModel(a.mano)
    seqs = sorted(d for d in os.listdir(a.store)
                  if os.path.exists(os.path.join(a.store, d, "hand_data",
                                                 "gt_joints_cache_cam_v2.pt")))[: a.n_seqs]
    if not seqs:
        raise SystemExit(f"no usable sequence under {a.store}")

    err = {g: [] for g in a.gaps}
    used = {g: 0 for g in a.gaps}
    for sq in seqs:
        hd = os.path.join(a.store, sq, "hand_data")
        bb = torch.load(os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt"), map_location="cpu")
        if "gt" not in bb:
            continue
        gt = bb["gt"].float()                     # [N, 64] both hands packed
        valid = bb["valid"].bool()                # [N, 2]
        w2c = torch.as_tensor(torch.load(os.path.join(hd, "cam_extrinsics_cache.pt"),
                                         map_location="cpu")).float()
        c2w = torch.linalg.inv(w2c.double()).float()
        N = min(len(gt), len(c2w))
        step = max(1, N // a.frames_per_seq)
        for g in a.gaps:
            for i in range(g, N - g, step):
                ok = valid[i - g] & valid[i + g] & valid[i]
                if not bool(ok.any()):
                    continue
                p0 = gt[i - g].view(2, 32)
                p1 = gt[i + g].view(2, 32)
                pt = interp_hand_params(p0, p1, c2w[i - g], c2w[i + g], c2w[i], 0.5)
                ji = compute_joints_from_batch(pt.view(1, 1, 64), mano, "cpu")   # [1,1,2,16,3]
                jg = compute_joints_from_batch(gt[i].view(1, 1, 64), mano, "cpu")
                d = (ji - jg).norm(dim=-1)[0, 0]                                 # [2, 16]
                for h in range(2):
                    if ok[h]:
                        err[g].append(float(d[h].mean()) * 1000.0)
                        used[g] += 1

    print(f"{'gap':>4s} {'frames':>7s} {'n':>5s} {'mean mm':>8s} {'median':>7s} {'p90':>7s}")
    for g in a.gaps:
        e = torch.tensor(err[g])
        if not e.numel():
            print(f"{g:4d} {2*g:7d} {0:5d}  no valid pair")
            continue
        print(f"{g:4d} {2*g:7d} {e.numel():5d} {e.mean():8.2f} {e.median():7.2f} "
              f"{e.quantile(0.9):7.2f}")
    print("MANO_INTERP_GAP_OK")


if __name__ == "__main__":
    main()
