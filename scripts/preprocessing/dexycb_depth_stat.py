#!/usr/bin/env python3
"""Median camera-frame WRIST DEPTH of DexYCB, as a go/no-go gate on the dataset itself.

WHY THIS EXISTS. Our mixture is depth-limited, not data-limited: the transfer law fitted on the
single-dataset case is C_abs ~= 60 mm + 0.50 * |median hand-depth shift|, so a training set whose
hands sit at the wrong distance costs ~0.5 mm of absolute error per mm of depth mismatch
(report/open-lines-tracker.md, "depth-diverse mixing confirmed"). DexYCB was added to the pool to
deepen it - the rig-geometry proxy measured off calibration.tar.gz gave a camera-to-ring-centroid
median of 0.877 m, but that is the CAMERA ring, not the HAND. This script measures the hand.

DECISION RULE: median wrist depth >= --min_median (default 0.70 m, i.e. at least as deep as HOI4D's
0.677). Below that, DexYCB does not buy the depth coverage it was chosen for and the dataset choice
should be revisited rather than the numbers explained away. Non-zero exit on failure.

Non-physical z is dropped BEFORE the median (default window 0.05-3.0 m): a -1 sentinel, an
unannotated frame stored as zeros, or a blown-up fit would otherwise drag the median silently.

THREE SOURCES, in decreasing order of directness:
  --store         a converted store (scripts/preprocessing/dexycb_to_ours.py). Reads
                  hand_data/gt_joints_cache_cam_v2.pt [N,2,16,3] and takes joint 0's z. This is
                  exactly the number the model will be trained against, so it is the one that
                  matters. Also works on ANY store in our format (HOI4D, H2O, ARCTIC, ...), which
                  is how you compare pools.
  --dexycb_root   raw DexYCB. Reads labels_%06d.npz['joint_3d'][hand, 0, 2] - camera-frame metres
                  by construction (dex_ycb_toolkit/hpe_eval.py:70-74 uses this array directly and
                  multiplies by 1000 to reach mm). EXACT, but one npz per frame per camera.
  --pose_npz_root raw DexYCB, fast scan: one pose.npz per capture (world-frame MANO) plus the
                  calibration extrinsics. APPROXIMATE by a bounded amount, see below.

THE pose.npz APPROXIMATION, stated rather than hidden. pose_m[..., 48:51] is the MANO `trans`, and
manopth leaves the root joint at its shaped-template position before adding trans, so
    wrist_world = root_j(betas) + trans
with root_j a CONSTANT vector (~0.10 m for MANO's right hand) that does not depend on the pose. In
camera frame that is a constant per-camera offset of at most ~0.10 m on every wrist, so the median
from --pose_npz_root is biased by <= ~0.10 m and the 10/90 spread is unaffected. Use it to triage a
partial download; confirm with --dexycb_root or --store before acting on a borderline result.

Usage:
    python -m scripts.preprocessing.dexycb_depth_stat --store $S/dexycb_ours
    python -m scripts.preprocessing.dexycb_depth_stat --dexycb_root $S/dexycb --stride 10
    python -m scripts.preprocessing.dexycb_depth_stat --pose_npz_root $S/dexycb
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

Z_MIN_M, Z_MAX_M = 0.05, 3.0        # physical window; anything outside is a sentinel or a bad fit
WRIST = 0                           # joint 0 in both the DexYCB 21 and our 16 layouts


def _report(name: str, z: np.ndarray, n_raw: int, args) -> bool:
    """Print the distribution for one source and return whether it clears the gate."""
    if z.size == 0:
        print(f"{name}: NO usable wrist depths out of {n_raw} candidates - "
              f"nothing was measured, so this is not a passing result", flush=True)
        return False
    med = float(np.median(z))
    p10, p90 = (float(x) for x in np.percentile(z, [10, 90]))
    dropped = n_raw - z.size
    print(f"{name}: median {med:.3f} m   10/90 pct {p10:.3f}/{p90:.3f} m   "
          f"n={z.size} (dropped {dropped} outside {Z_MIN_M}-{Z_MAX_M} m)", flush=True)
    ok = med >= args.min_median
    print(f"  GATE {'PASS' if ok else 'FAIL'}: median {med:.3f} m "
          f"{'>=' if ok else '<'} {args.min_median:.3f} m", flush=True)
    return ok


def _filter(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64).ravel()
    z = z[np.isfinite(z)]
    return z[(z > Z_MIN_M) & (z < Z_MAX_M)]


# ------------------------------------------------------------------ sources
def from_store(root: str, limit: int) -> tuple[np.ndarray, int]:
    """Converted store -> every valid wrist z in gt_joints_cache_cam_v2.pt.

    A hand that is absent for a frame is ZEROS in our stores (the representation
    HOT3DHandDataset expects), so an exactly-zero joint is skipped before the physical window
    is applied - otherwise every unannotated hand would count as a 0 m reading."""
    import torch

    zs = []
    n_raw = 0
    seqs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if limit:
        seqs = seqs[:limit]
    n_seq = 0
    for s in seqs:
        p = os.path.join(root, s, "hand_data", "gt_joints_cache_cam_v2.pt")
        if not os.path.exists(p):
            continue
        j = torch.load(p, map_location="cpu", weights_only=True).numpy()   # [N,2,16,3]
        present = np.abs(j).sum(axis=(-1, -2)) > 1e-8                      # [N,2]
        z = j[..., WRIST, 2][present]
        n_raw += int(z.size)
        zs.append(z)
        n_seq += 1
    print(f"  read {n_seq} sequences under {root}", flush=True)
    return (_filter(np.concatenate(zs)) if zs else np.array([])), n_raw


def from_labels(root: str, cameras: list, stride: int, limit: int) -> tuple[np.ndarray, int]:
    """Raw DexYCB -> labels_%06d.npz['joint_3d'][:, 0, 2], camera-frame metres.

    The all -1 sentinel marks an unannotated hand and is dropped by the physical window."""
    import yaml

    zs = []
    n_raw = 0
    n_cap = 0
    for sub in sorted(os.listdir(root)):
        if "subject" not in sub or not os.path.isdir(os.path.join(root, sub)):
            continue
        for cap in sorted(glob.glob(os.path.join(root, sub, "*"))):
            meta_p = os.path.join(cap, "meta.yml")
            if not os.path.exists(meta_p):
                continue
            if limit and n_cap >= limit:
                break
            with open(meta_p) as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            serials = [str(s) for s in meta["serials"] if not cameras or str(s) in cameras]
            for serial in serials:
                for i in range(0, int(meta["num_frames"]), max(1, stride)):
                    p = os.path.join(cap, serial, f"labels_{i:06d}.npz")
                    if not os.path.exists(p):
                        continue
                    j3 = np.asarray(np.load(p)["joint_3d"], np.float64).reshape(-1, 21, 3)
                    z = j3[:, WRIST, 2]
                    n_raw += int(z.size)
                    zs.append(z)
            n_cap += 1
    print(f"  read {n_cap} captures under {root} (stride {stride})", flush=True)
    return (_filter(np.concatenate(zs)) if zs else np.array([])), n_raw


def from_pose_npz(root: str, cameras: list, limit: int) -> tuple[np.ndarray, int]:
    """Raw DexYCB fast path -> pose.npz world-frame MANO trans, pushed into each camera frame.

    extrinsics.yml is CAMERA->WORLD (dex_ycb_toolkit/sequence_loader.py deprojects camera points
    with p_world = R p_cam + t), so the depth we want is the z of R^T (p_world - t). See the module
    docstring for the constant <= ~0.1 m bias this path carries."""
    import yaml

    zs = []
    n_raw = 0
    n_cap = 0
    calib = os.path.join(root, "calibration")
    extr_cache: dict = {}
    for sub in sorted(os.listdir(root)):
        if "subject" not in sub or not os.path.isdir(os.path.join(root, sub)):
            continue
        for cap in sorted(glob.glob(os.path.join(root, sub, "*"))):
            meta_p = os.path.join(cap, "meta.yml")
            pose_p = os.path.join(cap, "pose.npz")
            if not (os.path.exists(meta_p) and os.path.exists(pose_p)):
                continue
            if limit and n_cap >= limit:
                break
            with open(meta_p) as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            eid = str(meta["extrinsics"])
            if eid not in extr_cache:
                with open(os.path.join(calib, f"extrinsics_{eid}", "extrinsics.yml")) as f:
                    d = yaml.load(f, Loader=yaml.FullLoader)["extrinsics"]
                extr_cache[eid] = {k: np.asarray(v, np.float64).reshape(3, 4) for k, v in d.items()}
            T = extr_cache[eid]
            pose_m = np.asarray(np.load(pose_p)["pose_m"], np.float64).reshape(-1, 51)
            pose_m = pose_m[np.any(pose_m != 0.0, axis=1)]          # all-zero row = hand absent
            if pose_m.size == 0:
                continue
            trans_w = pose_m[:, 48:51]
            for serial in [str(s) for s in meta["serials"] if not cameras or str(s) in cameras]:
                if serial not in T:
                    continue
                R, t = T[serial][:, :3], T[serial][:, 3]
                z = ((np.linalg.inv(R) @ (trans_w - t).T).T)[:, 2]
                n_raw += int(z.size)
                zs.append(z)
            n_cap += 1
    print(f"  read {n_cap} captures under {root} (pose.npz fast path, "
          f"biased by a constant <= ~0.1 m)")
    return (_filter(np.concatenate(zs)) if zs else np.array([])), n_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="", help="converted store root (most direct source)")
    ap.add_argument("--dexycb_root", default="", help="raw DexYCB root; reads labels_*.npz")
    ap.add_argument("--pose_npz_root", default="", help="raw DexYCB root; reads pose.npz only")
    ap.add_argument("--cameras", default="", help="comma-separated serials (default: all)")
    ap.add_argument("--stride", type=int, default=10, help="frame stride for the labels_*.npz scan")
    ap.add_argument("--limit", type=int, default=0, help="cap sequences/captures scanned (smoke)")
    ap.add_argument("--min_median", type=float, default=0.70,
                    help="gate: DexYCB must hold hands at least this deep to be worth adding")
    a = ap.parse_args()

    cameras = [c for c in a.cameras.split(",") if c]
    if not (a.store or a.dexycb_root or a.pose_npz_root):
        raise SystemExit("give one of --store / --dexycb_root / --pose_npz_root")

    results = []
    if a.store:
        z, n = from_store(a.store, a.limit)
        results.append(_report(f"store {a.store}", z, n, a))
    if a.dexycb_root:
        z, n = from_labels(a.dexycb_root, cameras, a.stride, a.limit)
        results.append(_report(f"labels {a.dexycb_root}", z, n, a))
    if a.pose_npz_root:
        z, n = from_pose_npz(a.pose_npz_root, cameras, a.limit)
        results.append(_report(f"pose.npz {a.pose_npz_root}", z, n, a))

    if not all(results):
        raise SystemExit(
            "DEPTH GATE FAILED. DexYCB was added to DEEPEN the training pool (HOI4D 0.677, H2O "
            "0.503, ARCTIC 0.474, OakInk2 0.386, HOT3D 0.339 m). A shallower-than-expected DexYCB "
            "does not buy the coverage it was chosen for - revisit the dataset choice before "
            "spending the download and the conversion.")
    print("DEPTH GATE PASSED", flush=True)


if __name__ == "__main__":
    main()
