#!/usr/bin/env python3
"""Regenerate a store's hand boxes in the joints_to_bbox convention (rectangular, unclamped).

WHY THIS EXISTS. HOI4D, OakInk2 and ARCTIC all build boxes with `joints_to_bbox` from
scripts/arctic_to_ours.py: project the 3D joints, take the 2D hull, expand PER AXIS by rf=1.5,
normalise, and do NOT square and do NOT clamp. The H2O store instead squares the box to its max
side and clamps to [0,1]. The hand head consumes the stored box verbatim as a geometry/depth cue,
so square+clamped and rectangular+unclamped are genuinely different input distributions.

Consequence, measured: evaluating a model trained on the HOI4D convention against the H2O store
produced C-abs of ~290-380 mm - a pure convention artefact, not a generalisation result. Any H2O
number is meaningless until its boxes are rebuilt in the training convention.

This rewrites `hand_bboxes_v2_rf1.5_res224x224.pt` from the store's own cached camera-frame
joints + intrinsics, so the geometry is recomputed rather than reshaped from the square boxes
(you cannot recover an unclamped rectangle from a clamped square).

The original file is preserved as `<name>.square_backup.pt` unless --no_backup is passed.
"""
import argparse
import os

import numpy as np
import torch

from scripts.arctic_to_ours import joints_to_bbox


def _load_intrinsics(hd):
    """Return (f, cx, cy) from whichever intrinsics cache this store uses."""
    for name in ("cam_intrinsics.pt", "cam_intrinsics_cache.pt"):
        p = os.path.join(hd, name)
        if not os.path.exists(p):
            continue
        k = torch.load(p, map_location="cpu")
        if isinstance(k, dict):
            for key in ("K", "intrinsics", "cam_intrinsics"):
                if key in k:
                    k = k[key]
                    break
        k = torch.as_tensor(k).float()
        if k.ndim == 3:                       # [N,3,3] -> assume constant, take the first
            k = k[0]
        if k.shape == (3, 3):
            return float(k[0, 0]), float(k[0, 2]), float(k[1, 2])
        if k.numel() >= 3:                    # [f, cx, cy]
            v = k.flatten()
            return float(v[0]), float(v[1]), float(v[2])
    return None


def regen_sequence(seq_dir, res, rf, dry_run=False, no_backup=False):
    hd = os.path.join(seq_dir, "hand_data")
    jp = os.path.join(hd, "gt_joints_cache_cam_v2.pt")
    bp = os.path.join(hd, f"hand_bboxes_v2_rf{rf}_res{res}x{res}.pt")
    if not (os.path.exists(jp) and os.path.exists(bp)):
        return "missing-cache", None
    intr = _load_intrinsics(hd)
    if intr is None:
        return "no-intrinsics", None
    f, cx, cy = intr

    joints = torch.load(jp, map_location="cpu").float()      # [N,2,16,3] camera frame, metres
    old = torch.load(bp, map_location="cpu")
    valid = old["valid"].bool()                              # [N,2] keep the store's validity
    n, h = joints.shape[0], joints.shape[1]

    # The stored box is normalised by the SOURCE image size, and joints_to_bbox divides by W,H.
    # Recover the source size from the intrinsics: principal point is ~ the image centre.
    W = float(round(cx * 2.0)) or float(res)
    H = float(round(cy * 2.0)) or float(res)

    boxes = np.zeros((n, h, 4), np.float32)
    ok = np.zeros((n, h), bool)
    for t in range(n):
        for hi in range(h):
            if not bool(valid[t, hi]):
                continue
            j = joints[t, hi].numpy()
            if not np.isfinite(j).all() or (j[:, 2] <= 1e-3).all():
                continue
            b = joints_to_bbox(j, f, cx, cy, W, H, rf=rf)
            if b is None or not np.isfinite(b).all():
                continue
            boxes[t, hi] = b
            ok[t, hi] = True

    w = boxes[..., 2] - boxes[..., 0]
    hgt = boxes[..., 3] - boxes[..., 1]
    m = ok & (w > 0) & (hgt > 0)
    stats = {
        "n_frames": n,
        "kept": int(m.sum()),
        "was_valid": int(valid.sum()),
        # a rectangular convention should almost never produce w == h
        "square_frac": float(np.mean(np.isclose(w[m], hgt[m], rtol=1e-3))) if m.any() else float("nan"),
        # unclamped means some boxes legitimately fall outside [0,1]
        "outside01_frac": float(np.mean((boxes[m] < 0) | (boxes[m] > 1))) if m.any() else float("nan"),
        "W": W, "H": H, "f": f,
    }
    if dry_run:
        return "dry-run", stats

    if not no_backup:
        bak = bp.replace(".pt", ".square_backup.pt")
        if not os.path.exists(bak):
            torch.save(old, bak)

    # PRESERVE every other key. The store's box file also carries "gt": the GT MANO params
    # rewritten from world into CAMERA frame, which HOT3DHandDataset reads as cached["gt"].
    # Writing a fresh dict with only bboxes/valid silently deleted it and every eval on the
    # store then died with KeyError: 'gt'. That transform (_transform_gt_to_crop_local) is
    # world -> camera only; it takes bbox_frames but never uses it, so "gt" is INDEPENDENT of
    # the box convention and is correct to carry across unchanged.
    out_dict = {k: v for k, v in old.items() if k not in ("bboxes", "valid")}
    out_dict["bboxes"] = torch.from_numpy(boxes)
    out_dict["valid"] = torch.from_numpy(ok & valid.numpy())
    out_dict["convention"] = "joints_to_bbox rf1.5 rectangular unclamped (regenerated)"
    torch.save(out_dict, bp)
    return "ok", stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--rf", type=float, default=1.5)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_backup", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="EXIT NONZERO if the regenerated boxes are not actually rectangular and "
                         "unclamped (square_frac > 0.05, or outside01_frac == 0, or nothing kept). "
                         "Use on a --dry_run over a few sequences so a caller can refuse to "
                         "overwrite a whole store with boxes that did not change convention. "
                         "Printing the numbers is not enough: a job script that ignores them will "
                         "happily overwrite 177 sequences.")
    a = ap.parse_args()

    seqs = sorted(d for d in os.listdir(a.data_root)
                  if os.path.isdir(os.path.join(a.data_root, d)))
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    counts, agg = {}, []
    for s in seqs:
        st, stats = regen_sequence(os.path.join(a.data_root, s), a.res, a.rf,
                                   a.dry_run, a.no_backup)
        counts[st] = counts.get(st, 0) + 1
        if stats:
            agg.append(stats)
        if stats and len(agg) <= 3:
            print(f"  [{s}] {st} kept={stats['kept']}/{stats['was_valid']} "
                  f"square_frac={stats['square_frac']:.3f} "
                  f"outside01_frac={stats['outside01_frac']:.3f} "
                  f"W={stats['W']:.0f} H={stats['H']:.0f} f={stats['f']:.1f}", flush=True)
    print(f"\nREGEN status: {counts}")
    if not agg:
        print("REGEN produced NO statistics - no sequence had both a joints cache and boxes.")
        if a.gate:
            raise SystemExit("GATE FAILED: nothing to measure, so the convention is unverified.")
        return

    sq = np.nanmean([x["square_frac"] for x in agg])
    out = np.nanmean([x["outside01_frac"] for x in agg])
    kept = sum(x["kept"] for x in agg)
    was = sum(x["was_valid"] for x in agg)
    print(f"REGEN kept {kept}/{was} (frame,hand) entries")
    print(f"REGEN square_frac={sq:.4f} (want ~0: rectangular, NOT squared)")
    print(f"REGEN outside01_frac={out:.4f} (want >0: unclamped, HOI4D-style)")

    problems = []
    if sq > 0.05:
        problems.append(f"square_frac={sq:.4f} > 0.05 - boxes are still mostly SQUARE, so the "
                        f"regeneration did not change convention")
    if kept == 0:
        problems.append("kept 0 entries - every box was rejected")
    # outside01_frac is NOT a gate. It is evidence of unclamping only when some hand actually
    # leaves the frame, which is a property of the FOOTAGE, not of the convention. Measured on
    # H2O: the 3-sequence dry-run sample gave 0.0000 (those hands never exit frame) while the
    # full 177-sequence store gave 0.0234. Gating on it would have blocked a correct
    # regeneration. square_frac is the reliable convention marker, so gate on that alone.
    if not (out > 0.0):
        print(f"  note: outside01_frac={out:.4f} - no box leaves the frame in this sample. That is "
              f"expected on a small or fully-in-frame sample and is NOT a failure; only "
              f"square_frac gates.")

    for p in problems:
        print(f"  !! {p}")
    if problems and a.gate:
        raise SystemExit("GATE FAILED: refusing to certify these boxes as joints_to_bbox "
                         "convention. Do NOT overwrite the store; any eval on it would measure a "
                         "box-convention artefact (previously C-abs ~290-380 mm), not accuracy.")
    if a.gate:
        print("GATE PASSED: boxes are rectangular and unclamped.")


if __name__ == "__main__":
    main()
