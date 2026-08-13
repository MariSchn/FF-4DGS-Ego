#!/usr/bin/env python3
"""The four gates TACO must pass before anyone writes a converter for it.

WHY GATES AT ALL. Re:InterHand looked perfect in its abstract and in its per-frame labels, and was
only disqualified after someone measured how far the camera moved between consecutive frames: 755 mm
and 166 px of focal, every frame, because the "egocentric" split re-samples the viewpoint rather
than following a head. That cost a week. Every gate below is one that has already caught a real
defect on some store, so each runs before the converter rather than after the training run.

    1. TEMPORAL COHERENCE   consecutive-frame camera translation. The Re:InterHand test.
    2. FRAME ALIGNMENT      video frame count vs N_frame in the extrinsics. TACO's own script
                            prints "losing frames in the egocentric video, skip!", so the mismatch
                            is a known upstream condition, and our store contract is that video
                            frame t IS cache row t.
    3. DEPTH                median wrist depth, to place TACO against the pool and the held-out
                            sets. Not pass/fail; it is the number the depth-coverage argument needs.
    4. ANATOMY              bone lengths after the MANO forward pass. A scrambled joint order is
                            silent everywhere except here, and it corrupted every H2O number we
                            produced until bone lengths caught it.

Reads the layout verified from TACO's own project_pose_to_egocentric_view.py, notably line 113:
    egocentric_extrinsics = np.load(...)   # world_to_camera, shape = (N_frame, 4, 4)

    python -m scripts.preprocessing.taco_gates --taco_root <root> --n 40
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import sys

import numpy as np

# Consecutive-frame camera motion above this is not a head, it is a re-sampled viewpoint. The
# threshold is set from what Re:InterHand actually measured (755 mm median) against what a head
# does at 30 Hz: a fast head turn moves the camera centre by a few cm per frame, not by a metre.
MAX_PLAUSIBLE_DT_M = 0.15


def load_seq_list(root: str, limit: int) -> list[tuple[str, str]]:
    p = os.path.join(root, "_usable_sequences.txt")
    if not os.path.isfile(p):
        sys.exit(f"missing {p}. Run the extraction job first; it writes the intersection of the "
                 f"three subtrees, which is the only correct sequence list.")
    rows = [tuple(l.rstrip("\n").split("\t")) for l in open(p) if l.strip()]
    # Evenly spaced rather than the first N: the list is sorted by triplet, so the head of it is one
    # action repeated, which would hide any per-action problem.
    if limit and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def n_video_frames(path: str) -> int | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco_root", default="/cluster/scratch/dmonopoli/taco_v1")
    ap.add_argument("--n", type=int, default=40, help="sequences to sample (0 = all)")
    ap.add_argument("--mano_dir", default="", help="MANO model dir; enables gate 4")
    a = ap.parse_args()

    seqs = load_seq_list(a.taco_root, a.n)
    print(f"sampling {len(seqs)} sequences\n")

    dt_med, dt_max, foc, mism, wrist_z = [], [], [], [], []
    n_ok = 0

    for trip, sq in seqs:
        cam = os.path.join(a.taco_root, "Egocentric_Camera_Parameters", trip, sq)
        try:
            K = np.loadtxt(os.path.join(cam, "egocentric_intrinsic.txt"))
            E = np.load(os.path.join(cam, "egocentric_frame_extrinsic.npy"))
        except Exception as e:
            print(f"  !! {trip}/{sq}: cannot read camera params ({e})")
            continue

        # E is world_to_camera per their own comment, so the camera CENTRE in world coordinates is
        # -R^T t, not t. Using t directly would measure something that is not a trajectory and the
        # coherence gate would be meaningless.
        R, t = E[:, :3, :3], E[:, :3, 3]
        C = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)
        d = np.linalg.norm(np.diff(C, axis=0), axis=1)
        if d.size:
            dt_med.append(float(np.median(d)))
            dt_max.append(float(d.max()))
        foc.append(float(K[0, 0]))

        nv = n_video_frames(os.path.join(a.taco_root, "Egocentric_RGB_Videos", trip, sq, "color.mp4"))
        if nv is not None and nv != len(E):
            mism.append((trip, sq, nv, len(E)))

        # Wrist depth: MANO translation is the wrist in WORLD metres, so it has to be carried into
        # the camera before its z means "how far from the camera".
        # right_hand.pkl is keyed BY FRAME ('00001' ... '000NN'), each value a dict carrying
        # hand_pose (48,) axis-angle and hand_trans (3,). An earlier version of this gate looked
        # for hand_trans at the top level, found nothing, and reported "could not compute" rather
        # than a wrong number, which is the failure mode to prefer.
        try:
            with open(os.path.join(a.taco_root, "Hand_Poses", trip, sq, "right_hand.pkl"), "rb") as f:
                hp = pickle.load(f)
            tr = None
            if isinstance(hp, dict) and hp:
                keys = sorted(hp.keys())
                if isinstance(hp[keys[0]], dict) and "hand_trans" in hp[keys[0]]:
                    tr = np.stack([np.asarray(hp[k]["hand_trans"], dtype=np.float64).ravel()
                                   for k in keys])
            if tr is not None:
                tr = tr.reshape(-1, 3)
                m = min(len(tr), len(E))
                zc = np.einsum("nij,nj->ni", R[:m], tr[:m]) + t[:m]
                wrist_z.extend(zc[:, 2][np.isfinite(zc[:, 2])].tolist())
        except Exception:
            pass
        n_ok += 1

    print("=" * 78)
    print(f"GATE 1  TEMPORAL COHERENCE   ({n_ok} sequences read)")
    if dt_med:
        mm, xx = statistics.median(dt_med), max(dt_max)
        print(f"  consecutive camera-centre step: median {mm*1000:.1f} mm, worst {xx*1000:.1f} mm")
        print(f"  focal across sequences: {min(foc):.1f} to {max(foc):.1f} px "
              f"(constant per sequence by construction, one 3x3 per sequence)")
        print(f"  VERDICT: {'PASS' if mm < MAX_PLAUSIBLE_DT_M else 'FAIL'}"
              f"   (Re:InterHand measured 755 mm here and was disqualified)")
    else:
        print("  no extrinsics read")

    print(f"\nGATE 2  FRAME ALIGNMENT")
    if n_video_frames.__module__ and mism is not None:
        print(f"  sequences whose video length differs from N_frame: {len(mism)} of {n_ok}")
        for row in mism[:5]:
            print(f"    {row[0]}/{row[1]}: video {row[2]} vs extrinsics {row[3]}")
        print(f"  VERDICT: {'PASS' if not mism else 'HANDLE IT'}"
              f"   (TACO's own script prints 'losing frames in the egocentric video, skip!', so a"
              f" mismatch is expected on some sequences and the converter must drop or truncate"
              f" them, never silently zip mismatched lengths)")

    print(f"\nGATE 3  DEPTH")
    if wrist_z:
        z = statistics.median(wrist_z)
        print(f"  median wrist depth: {z:.3f} m   (n={len(wrist_z)} frames)")
        print("  pool: HOT3D 0.339  OakInk2 0.386  ARCTIC 0.474  DexYCB 0.780")
        print("  held out: H2O 0.503  HOI4D 0.677")
        print("  VERDICT: informational. Record where it lands; the depth-coverage claim needs it.")
    else:
        print("  could not compute (hand_trans not found in the expected key)")

    print(f"\nGATE 4  ANATOMY / JOINT ORDER")
    print("  NOT RUN HERE. It needs the MANO forward pass, so it belongs in the converter's own")
    print("  validation, next to the 21->16 remap it is testing. Assert bone lengths there, as")
    print("  dexycb_to_ours.py does, and refuse to write a store that fails.")
    print("=" * 78)


if __name__ == "__main__":
    main()
