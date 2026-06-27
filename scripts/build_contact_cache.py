"""Write hand_data/contact_cache.pt ([N,2] bool) per HOI4D seq: the GT wrist sits
on the visible GT surface (|z_wrist_GT - GT_dense_depth_at_wrist| < thresh).

Reuses the preprocessed camera caches + raw_depth, with the same center-crop +
resize as eval_scale_source so the dense depth aligns with the cached joints.
The cache is the contact gate for the Phase-2 anchor (train + eval).

Run (gb10 venv):
    python -m scripts.build_contact_cache --pp /tmp/hoi4d_pp \
        --raw /work/scratch/dmonopoli/hoi4d --res 224 --thresh_m 0.05
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from scripts.contact_mask import wrist_contact_mask

RH = 1


def _center_square(a):
    h, w = a.shape[:2]
    s = min(h, w)
    return a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def _load_depth(path, res):
    d = _center_square(cv2.imread(path, cv2.IMREAD_ANYDEPTH)).astype(np.float32) / 1000.0  # mm->m
    d = cv2.resize(d, (res, res), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", required=True, help="preprocessed root (<seq>/hand_data/...)")
    ap.add_argument("--raw", required=True, help="raw HOI4D root (<seq>/raw_depth/)")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--thresh_m", type=float, default=0.05)
    args = ap.parse_args()

    seqs = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(args.pp, "*")))
            if os.path.exists(os.path.join(d, "hand_data", "gt_joints_cache_cam_v2.pt"))]
    print(f"[contact] {len(seqs)} seqs with hand caches", flush=True)
    for sq in seqs:
        hd = os.path.join(args.pp, sq, "hand_data")
        gtj = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), map_location="cpu").float()  # [N,2,16,3]
        ci = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)
        deps = sorted(glob.glob(os.path.join(args.raw, sq, "raw_depth", "*.png")))
        n = min(len(deps), gtj.shape[0])
        out = torch.zeros(gtj.shape[0], 2, dtype=torch.bool)
        for t in range(n):
            wrist = gtj[t:t + 1, :, 0, :].unsqueeze(0)                        # [1,1,2,3]
            d = _load_depth(deps[t], args.res).reshape(1, 1, 1, args.res, args.res)
            out[t] = wrist_contact_mask(wrist, d, ci, args.thresh_m)[0, 0]
        torch.save(out, os.path.join(hd, "contact_cache.pt"))
        print(f"[{sq}] contact frames RH: {int(out[:n, RH].sum())}/{n}", flush=True)


if __name__ == "__main__":
    main()
