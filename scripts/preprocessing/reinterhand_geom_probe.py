"""Decide Re:InterHand's ego CAMERA CONVENTIONS by measurement, not by reading magnitudes.

WHY THIS EXISTS
---------------
`reinterhand_to_ours.py` cannot convert a single frame until two binary questions are settled,
and getting either wrong corrupts every one of the ~57k ego frames silently - the store would
still build, the gates would still pass, and training would just quietly learn nothing:

  Q1  DISTORTION MODEL.  Each `cam_params/<frame>.json` carries `D` with FOUR coefficients, e.g.
      [0.135, 0.480, 0.577, 0.385]. Under OpenCV's fisheye/equidistant convention that is
      [k1,k2,k3,k4] on theta and those magnitudes are ordinary for a wide-FOV lens. Under the
      4-parameter radtan convention it is [k1,k2,p1,p2] where p1,p2 are TANGENTIAL and are
      normally 1e-3 or smaller, so 0.577 would be absurd. That argument is strong but it is still
      an argument, and the two models place a joint tens of pixels apart at the image edge.

  Q2  EXTRINSIC DIRECTION.  `R` and `t` could be a world->camera rigid transform, or `t` could be
      the camera CENTRE in world coordinates - which is what InterHand2.6M itself publishes
      (`cam_coord = (world_coord - campos) @ camrot.T`). Re:InterHand is rendered FROM
      InterHand2.6M captures, so both are live possibilities, and they differ by up to ~1.4 m.

THE MEASUREMENT
---------------
Re:InterHand ships a per-frame HAND SEGMENTATION MASK next to every ego image. A correct
(convention, distortion) pair projects the MANO joints INSIDE that mask; a wrong one does not.
So the probe forward-kinematics the published MANO fits, projects them under every candidate
combination, and reports the fraction of joints landing on mask support. This turns two guesses
into one number each, using only data already on disk.

The 16 smplx-kinematic joints are used, never the fingertips: tips sit within a few px of the
silhouette boundary, so including them would blur the very signal being measured.

Run this BEFORE the first conversion, and paste its verdict into the converter.

    python -m scripts.preprocessing.reinterhand_geom_probe \
        --capture_dir $S/reinterhand/m--20221007--1215--HIR112--...--two-hands \
        --mano_dir models/MANO --n_frames 16
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile

import cv2
import numpy as np

from scripts.arctic_to_ours import _to_smplx16, apply_se3, build_mano, fk_world_joints
from scripts.preprocessing.reinterhand_to_ours import SIDE_TO_HAND_IDX, split_pose48

EGO_SUBDIR = os.path.join("Ego_cameras", "envmap_per_frame")
# A hand joint may not sit further than this in front of the camera. Egocentric hands live at
# 0.15-0.80 m; anything outside says the extrinsic convention or the metre/millimetre handling is
# wrong, independently of whether the projection happens to land on the mask.
PLAUSIBLE_DEPTH_M = (0.05, 3.0)


def load_cam(path: str) -> dict:
    """One per-frame ego camera record, verified schema: R, t, focal, princpt, D."""
    with open(path) as f:
        raw = json.load(f)
    missing = {"R", "t", "focal", "princpt", "D"} - set(raw)
    if missing:
        raise SystemExit(f"{path}: cam_params is missing {sorted(missing)}; this probe (and the "
                         f"converter) only knows the R/t/focal/princpt/D schema")
    fx, fy = np.asarray(raw["focal"], np.float64)
    px, py = np.asarray(raw["princpt"], np.float64)
    return {"R": np.asarray(raw["R"], np.float64).reshape(3, 3),
            "t_mm": np.asarray(raw["t"], np.float64).reshape(3),
            "K": np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]], np.float64),
            "D": np.asarray(raw["D"], np.float64).reshape(4)}


def w2c_candidates(R: np.ndarray, t_m: np.ndarray) -> dict[str, np.ndarray]:
    """Every sane reading of a published (R, t) pair, as world->camera 4x4 matrices.

    `t` is scaled to metres by the caller. All four are cheap to test, so none is assumed away.
    """
    def se3(rot, tr):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = rot, tr
        return m

    return {
        "w2c_direct":  se3(R, t_m),                 # (R, t) IS world->camera
        "w2c_campos":  se3(R, -R @ t_m),            # t is the camera CENTRE (InterHand2.6M's own)
        "c2w_direct":  se3(R.T, -R.T @ t_m),        # (R, t) is camera->world
        "R_transpose": se3(R.T, t_m),               # R stored transposed, t already world->camera
    }


def project(j_cam: np.ndarray, K: np.ndarray, D: np.ndarray, model: str) -> np.ndarray:
    """[J,3] camera-frame metres -> [J,2] pixels under `model`. OpenCV is the reference for both,
    so neither distortion polynomial is re-implemented here."""
    pts = np.ascontiguousarray(j_cam.reshape(-1, 1, 3), np.float64)
    zero = np.zeros(3)
    if model == "fisheye":
        uv, _ = cv2.fisheye.projectPoints(pts, zero, zero, K, D.reshape(4, 1))
    elif model == "radtan":
        uv, _ = cv2.projectPoints(pts.reshape(-1, 3), zero, zero, K, D.reshape(1, 4))
    else:
        raise ValueError(model)
    return uv.reshape(-1, 2)


def inside_mask(uv: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """(n_on_mask, n_in_image, n_total) for projected pixels against a hand mask."""
    h, w = mask.shape
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inimg = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    on = int((mask[v[inimg], u[inimg]] > 0).sum()) if inimg.any() else 0
    return on, int(inimg.sum()), int(len(uv))


def index_fits(params_tar: str) -> dict[str, dict[int, str]]:
    """{frame_id: {hand_idx: member_name}} from params.tar, WITHOUT parsing any JSON.

    Read through tarfile rather than extracting: this archive is one JSON per frame per hand
    (78,088 of them for one capture) and unpacking it wholesale is what exhausted the 1.5M inode
    allocation on 2026-08-04. Indexing names first also means only the sampled frames are ever
    parsed, instead of paying 78k json.loads to look at sixteen of them.
    """
    idx: dict[str, dict[int, str]] = {}
    with tarfile.open(params_tar) as tf:
        for name in tf.getnames():
            base = os.path.basename(name)
            if not base.endswith(".json") or "_" not in base:
                continue
            fid, side = base[:-len(".json")].rsplit("_", 1)
            if side in SIDE_TO_HAND_IDX:
                idx.setdefault(fid, {})[SIDE_TO_HAND_IDX[side]] = name
    return idx


def read_fits(params_tar: str, want: dict[str, dict[int, str]]) -> dict[str, dict[int, dict]]:
    """Parse only the (frame, hand) records named in `want`."""
    fits: dict[str, dict[int, dict]] = {}
    with tarfile.open(params_tar) as tf:
        for fid, hands in want.items():
            for hi, member in hands.items():
                rec = json.load(tf.extractfile(member))
                fits.setdefault(fid, {})[hi] = {
                    "pose": np.asarray(rec["pose"], np.float64).reshape(-1),
                    "shape": np.asarray(rec["shape"], np.float64).reshape(-1),
                    "trans": np.asarray(rec["trans"], np.float64).reshape(3)}
    return fits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--mano_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trans_units", choices=["m", "mm", "auto"], default="auto",
                    help="units of the MANO `trans` field. InterHand2.6M publishes METRE MANO fits "
                         "next to MILLIMETRE camera positions, so this is detected and printed "
                         "rather than assumed.")
    a = ap.parse_args()

    ego = os.path.join(a.capture_dir, EGO_SUBDIR)
    cam_dir, msk_dir = os.path.join(ego, "cam_params"), os.path.join(ego, "masks")
    tar = os.path.join(a.capture_dir, "mano_fits", "params.tar")
    for p in (cam_dir, msk_dir, tar):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    print(f"indexing fits in {tar} ...", flush=True)
    fit_index = index_fits(tar)
    have_cam = {f[:-len(".json")] for f in os.listdir(cam_dir) if f.endswith(".json")}
    have_msk = {f[:-len(".png")] for f in os.listdir(msk_dir) if f.endswith(".png")}
    usable = sorted(have_cam & have_msk & set(fit_index),
                    key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
    print(f"frames: cam={len(have_cam)} mask={len(have_msk)} fits={len(fit_index)} "
          f"usable={len(usable)}")
    if not usable:
        raise SystemExit("no frame has a camera, a mask and a MANO fit at once")

    # Spread the sample across the whole capture: a contiguous block would all share one hand pose
    # and one camera position, which is exactly the situation in which a wrong convention can
    # still score well by accident.
    idx = np.linspace(0, len(usable) - 1, min(a.n_frames, len(usable))).round().astype(int)
    frames = [usable[i] for i in dict.fromkeys(idx)]
    fits = read_fits(tar, {f: fit_index[f] for f in frames})

    # MANO trans units, over the sampled frames. A metre/millimetre mix here is a 1000x error that
    # every other check downstream would inherit.
    tz = np.abs([r["trans"] for f in frames for r in fits[f].values()])
    mano_scale = 1.0 if a.trans_units == "m" else 1e-3 if a.trans_units == "mm" else (
        1e-3 if np.median(tz) > 10.0 else 1.0)
    print(f"MANO |trans| median={np.median(tz):.4f} -> scale={mano_scale:g} "
          f"({'mm->m' if mano_scale != 1.0 else 'already metres'})")

    mano = build_mano(a.mano_dir, a.device)
    combos = {}

    for fid in frames:
        cam = load_cam(os.path.join(cam_dir, fid + ".json"))
        mask = cv2.imread(os.path.join(msk_dir, fid + ".png"), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]

        for hi, rec in sorted(fits[fid].items()):
            side = "right" if hi == SIDE_TO_HAND_IDX["right"] else "left"
            g, hp = split_pose48(rec["pose"])
            jw = fk_world_joints(mano[side], g[None].astype(np.float32), hp[None].astype(np.float32),
                                 (rec["trans"] * mano_scale)[None].astype(np.float32),
                                 rec["shape"][None, :10].astype(np.float32), a.device)
            jw16 = _to_smplx16(jw)[0]                                     # [16,3] world metres

            for cname, w2c in w2c_candidates(cam["R"], cam["t_mm"] * 1e-3).items():
                jc = apply_se3(w2c[None].astype(np.float32), jw16[None].astype(np.float32))[0]
                for model in ("fisheye", "radtan"):
                    key = (cname, model)
                    acc = combos.setdefault(key, {"on": 0, "inimg": 0, "tot": 0, "depth": []})
                    acc["depth"].extend(jc[:, 2].tolist())
                    if (jc[:, 2] <= 1e-3).any():        # behind the camera: unprojectable, not a
                        acc["tot"] += len(jc)           # scoring opportunity, so count it as missed
                        continue
                    on, inimg, tot = inside_mask(project(jc, cam["K"], cam["D"], model), mask)
                    acc["on"] += on
                    acc["inimg"] += inimg
                    acc["tot"] += tot

    print(f"\nscored {len(frames)} frames, "
          f"{sum(len(fits[f]) for f in frames)} hand instances, 16 joints each\n")
    hdr = f"{'extrinsic':<12} {'distortion':<10} {'on-mask':>8} {'in-image':>9} {'median z (m)':>13} {'plausible':>10}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for (cname, model), acc in combos.items():
        on = acc["on"] / max(1, acc["tot"])
        inimg = acc["inimg"] / max(1, acc["tot"])
        med = float(np.median(acc["depth"]))
        ok = PLAUSIBLE_DEPTH_M[0] <= med <= PLAUSIBLE_DEPTH_M[1]
        rows.append((on, inimg, med, ok, cname, model))
    for on, inimg, med, ok, cname, model in sorted(rows, reverse=True):
        print(f"{cname:<12} {model:<10} {on:>7.1%} {inimg:>8.1%} {med:>13.3f} {'yes' if ok else 'NO':>10}")

    best = max(rows)
    runner = sorted(rows, reverse=True)[1] if len(rows) > 1 else None
    print(f"\nVERDICT  extrinsic={best[4]}  distortion={best[5]}  "
          f"on-mask={best[0]:.1%}  median wrist-region depth={best[2]:.3f} m")
    if best[0] < 0.60:
        print("  *** INCONCLUSIVE: the best combination still misses the mask most of the time.\n"
              "      Do NOT convert. Something outside these two questions is wrong - a different\n"
              "      world frame for the fits, a per-frame id offset between fits and cameras, or a\n"
              "      mask that is not the hand. Inspect an overlay before touching the converter.")
    elif runner and best[0] - runner[0] < 0.10:
        print(f"  *** AMBIGUOUS: runner-up {runner[4]}/{runner[5]} is within "
              f"{best[0] - runner[0]:.1%}. Raise --n_frames and re-run before committing.")
    else:
        print("  Decisive. Wire exactly this pair into reinterhand_to_ours.py.")


if __name__ == "__main__":
    main()
