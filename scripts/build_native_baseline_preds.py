"""Per-frame WiLoR / HaMeR hand baseline on HOI4D video -> camera-frame joints in the
``{cam_joints, world_joints, valid}`` contract consumed by ``scripts.build_slam_baseline``.

This reconstructs the ``*_native_truefocal_preds`` producer (which was lost from the repo -
it ran from a node-local staged bundle whose scratch was wiped), from:
  * the VERIFIED WiLoR API + Kfix "true-focal" placement in ``run_wilor_h2o.py`` (the H2O
    baseline run), copied here so the number is methodologically identical, and
  * the HOI4D focal/box convention in ``scripts.hoi4d_to_haptic`` (real intrinsics at the
    store resolution, rescaled to the decoded frame, principal point = image centre).

The ONLY change vs the original is that the detection box is parameterised: ``--box_dir``
points at our predictive detection boxes (detbox v3), giving the *input-matched* baseline
row. With ``--box_dir`` omitted it reads the GT box cache (the own-regime control used to
VALIDATE this reconstruction reproduces the known own-box +SLAM numbers before we trust the
detbox numbers).

Per frame (right hand, HOI4D RH=1): crop by the box, run the model -> root-relative MANO
kp3d (21) + weak-perspective ``pred_cam`` [s,tx,ty]; re-place the root with the REAL HOI4D
focal via ``cam_crop_to_full`` (Kfix) -> absolute camera-frame joints; remap MANO-21 -> the
16 smplx kinematic joints the eval uses. Left hand is left NaN/invalid (HOI4D is RH-primary,
matching the cached native preds). Output per seq ``<out_dir>/<seq>.pt``:
    cam_joints   [N,2,16,3] float, metres, CAMERA frame  (RH populated, LH NaN)
    world_joints [N,2,16,3] float, ALL NaN  (build_slam_baseline recomputes from the SLAM traj)
    valid        [N,2] bool  (only [:,1]=RH can be True)

Usage (Euler, a venv with wilor-mini; e.g. venv_haptic or hawor_env):
    python -m scripts.build_native_baseline_preds --method wilor \
        --data_root $S/hoi4d/hoi4d_test157 --box_dir $S/hoi4d_detboxes_v3 \
        --out_dir   $S/wilor_detbox_truefocal_preds
Validation (own box, must reproduce the known own-regime +SLAM number after composing):
    python -m scripts.build_native_baseline_preds --method wilor \
        --data_root $S/hoi4d/hoi4d_test157 --out_dir $S/wilor_ownbox_truefocal_preds --validate
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch

RH = 1   # right hand slot (HOI4D convention, matches hand_data caches + eval_worldspace_baseline)
J = 16   # smplx kinematic joints per hand

# WiLoR/HaMeR emit MANO-21 (native order, tips interleaved 4,8,12,16,20). Tips at 4/8/12/16/20
# means five 4-slot finger blocks in source order thumb, index, middle, ring, pinky. Our GT and
# every other row are smplx-16: wrist, index, middle, pinky, ring, thumb. The reorder is therefore
# NOT the identity on the non-tip slots.
#
# BUG FIX 2026-08-06. This was [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], i.e. the
# non-tip slots in SOURCE order, while the comment claimed it was "copied VERBATIM from
# eval_cmpjpe.py". Same index SET, different order, so no shape or range check could catch it: a
# perfect prediction scored ~53 mm root-relative through it, and the repo's own anatomical
# bone-length gate flagged 2 of 16 slots. Every <seq>.pt written by this script BEFORE this date
# has permuted finger blocks and must be rebuilt before its numbers are used.
MANO21_TO_16 = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]

# Tripwire against the canonical definition, which consumes the SAME OpenPose-21 layout. A local
# copy of a constant cannot detect drift in the original, so import the original.
try:
    from scripts.haptic_to_worldeval import OP2SMPLX16 as _CANON_OP2SMPLX16
except ImportError:
    _CANON_OP2SMPLX16 = None
if _CANON_OP2SMPLX16 is not None and list(MANO21_TO_16) != list(_CANON_OP2SMPLX16):
    raise RuntimeError(
        "native-baseline joint remap drifted from the canonical smplx-16 order:\n"
        f"  build_native_baseline_preds.MANO21_TO_16 = {list(MANO21_TO_16)}\n"
        f"  haptic_to_worldeval.OP2SMPLX16           = {list(_CANON_OP2SMPLX16)}")


def cam_t_with_focal(pred_cam, box_center, box_size, img_w, img_h, focal):
    """cam_crop_to_full with an explicit focal + principal point at the image centre.

    VERBATIM from run_wilor_h2o.py (the verified H2O baseline). pred_cam = [s, tx, ty] in
    crop space; returns the full-image camera translation [TX, TY, tz] (m). Re-derives the
    weak-perspective root under the REAL focal (the "Kfix" true-focal placement)."""
    s, tx, ty = float(pred_cam[0]), float(pred_cam[1]), float(pred_cam[2])
    cx, cy = box_center
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = box_size * s + 1e-9
    tz = 2 * focal / bs
    TX = 2 * (cx - w_2) / bs + tx
    TY = 2 * (cy - h_2) / bs + ty
    return np.array([TX, TY, tz], dtype=np.float64)


def load_boxes(seq_dir: str, seq: str, box_dir: str | None):
    """Return (boxes_norm [N,4] xyxy normalized for RH, valid [N] bool) or None.

    box_dir set -> external predicted store <box_dir>/<seq>.pt (detbox v3); else the GT cache
    in hand_data. Both are {"bboxes":[N,2,4] normalized, "valid":[N,2]} (schema-identical)."""
    if box_dir is not None:
        bp = os.path.join(box_dir, seq + ".pt")
        if not os.path.exists(bp):
            print(f"SEQ_SKIP {seq}: no predicted box in {box_dir}", flush=True)
            return None
    else:
        bp = os.path.join(seq_dir, "hand_data", "hand_bboxes_v2_rf1.5_res224x224.pt")
        if not os.path.exists(bp):
            return None
    bb = torch.load(bp, map_location="cpu")
    boxes = np.asarray(bb["bboxes"], np.float32)[:, RH]          # [N,4] normalized xyxy
    valid = np.asarray(bb["valid"], bool)[:, RH]                 # [N]
    return boxes, valid


def load_focal(seq_dir: str):
    """Real HOI4D focal at the store resolution: cam_intrinsics.pt = [f, cx, cy] with
    cx ~ store_W/2. Returns (f, cx, cy) floats; the decoded-frame focal is f*W/(2*cx),
    the SAME rescale scripts.hoi4d_to_haptic uses."""
    K = torch.load(os.path.join(seq_dir, "hand_data", "cam_intrinsics.pt"),
                   map_location="cpu").float().flatten()
    return float(K[0]), float(K[1]), float(K[2])


class WiLoRRunner:
    """Thin wrapper over wilor-mini's predict_with_bboxes (detector bypassed -> our box)."""

    def __init__(self, device, dtype=torch.float32):
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
        self.pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    def predict(self, img_rgb, box_px, is_right):
        """img_rgb HWC uint8; box_px [x1,y1,x2,y2]. Returns (kp3d[21,3] root-rel m,
        pred_cam[3]) or None if the model returns nothing."""
        out = self.pipe.predict_with_bboxes(
            img_rgb, np.array([box_px], np.float32), np.array([is_right], np.float32))
        if not out:
            return None
        wp = out[0]["wilor_preds"]
        return np.asarray(wp["pred_keypoints_3d"])[0], np.asarray(wp["pred_cam"])[0]


class HaMeRRunner:
    """HaMeR behind the same predict contract WiLoRRunner exposes.

    HaMeR ships no bbox-conditioned entry point, so the crop is built here the way its own
    ViTDetDataset builds it: a square box scaled by the model's BBOX_SHAPE rescale factor, resized
    to the model's input resolution, then normalised with the ImageNet statistics its config
    carries. Reading those from the checkpoint's config rather than hardcoding them is what keeps
    this comparable with the released model instead of with a crop we invented.

    Left hands are mirrored in, and the returned keypoints mirrored back, which is what HaMeR does
    internally, since the model is right-hand only.
    """

    def __init__(self, device, ckpt_dir=None):
        from pathlib import Path
        from hamer.models import load_hamer

        ckpt_dir = Path(ckpt_dir or os.environ.get("HAMER_CKPT_DIR", ""))
        ckpt = ckpt_dir / "hamer.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt}: set HAMER_CKPT_DIR or pass ckpt_dir. The checkpoint is not bundled.")
        self.model, self.cfg = load_hamer(str(ckpt))
        self.model = self.model.to(device).eval()
        self.device = device
        self.res = int(self.cfg.MODEL.IMAGE_SIZE)
        self.mean = np.array(self.cfg.MODEL.IMAGE_MEAN, np.float32) * 255.0
        self.std = np.array(self.cfg.MODEL.IMAGE_STD, np.float32) * 255.0
        # 2.0 is HaMeR's own demo default for --rescale_factor. It is NOT a config field: the
        # released model_config.yaml has no BBOX_RESCALE, so a getattr with a default would read
        # as if the model supplied the value while silently supplying our own. Verified against
        # the checkpoint on 2026-08-18. The crop scale moves every depth HaMeR predicts, so this
        # constant is stated rather than looked up.
        self.rescale = 2.0

    def _crop(self, img_rgb, box_px, flip):
        x1, y1, x2, y2 = [float(v) for v in box_px]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * self.rescale
        half = side / 2.0
        H, W = img_rgb.shape[:2]
        # Pad rather than clamp: clamping a box that leaves the frame changes its centre, and the
        # weak-perspective root is solved from that centre downstream.
        pad = int(max(0, half - min(cx, cy, W - cx, H - cy)) + 1)
        padded = np.pad(img_rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        px, py = cx + pad, cy + pad
        a, b = int(round(px - half)), int(round(py - half))
        crop = padded[b:b + int(round(side)), a:a + int(round(side))]
        crop = cv2.resize(crop, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        if flip:
            crop = crop[:, ::-1]
        x = (crop.astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

    def predict(self, img_rgb, box_px, is_right):
        """img_rgb HWC uint8; box_px [x1,y1,x2,y2]. Returns (kp3d[21,3] root-rel m, pred_cam[3])."""
        flip = not bool(is_right)
        x = self._crop(img_rgb, box_px, flip).to(self.device)
        with torch.no_grad():
            out = self.model({"img": x,
                              "right": torch.ones(1, device=self.device) if not flip
                                       else torch.zeros(1, device=self.device)})
        kp = out["pred_keypoints_3d"][0].float().cpu().numpy()
        cam = out["pred_cam"][0].float().cpu().numpy()
        if flip:
            # Undo the mirror on both the joints and the crop-space x translation.
            kp = kp.copy(); kp[:, 0] *= -1.0
            cam = cam.copy(); cam[1] *= -1.0
        if kp.shape != (21, 3):
            raise RuntimeError(
                f"HaMeR returned {kp.shape} keypoints, not (21,3); MANO21_TO_16 assumes the "
                f"OpenPose-21 layout WiLoR also returns, so the remap would be silently wrong.")
        return kp - kp[:1], cam


def build_runner(method: str, device: str):
    if method == "wilor":
        return WiLoRRunner(device)
    if method == "hamer":
        return HaMeRRunner(device)
    raise NotImplementedError(f"method={method}: implemented are 'wilor' and 'hamer'.")


def build_seq(seq_dir, seq, runner, box_dir, focal_axis="x"):
    b = load_boxes(seq_dir, seq, box_dir)
    if b is None:
        return None
    boxes_norm, box_valid = b
    f, cx, cy = load_focal(seq_dir)

    cap = cv2.VideoCapture(os.path.join(seq_dir, "video_main_rgb.mp4"))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    Himg, Wimg = frames[0].shape[:2]
    N = min(len(frames), boxes_norm.shape[0])
    # decoded-frame focal (rescale store-res f), matching hoi4d_to_haptic (sx=W/(2cx))
    focal = f * (Wimg / (2.0 * cx)) if focal_axis == "x" else f * (Himg / (2.0 * cy))

    cam_joints = torch.full((N, 2, J, 3), float("nan"))
    world_joints = torch.full((N, 2, J, 3), float("nan"))   # build_slam_baseline fills world
    valid = torch.zeros((N, 2), dtype=torch.bool)
    n_pred = 0
    for t in range(N):
        if not box_valid[t]:
            continue
        bn = boxes_norm[t]
        box_px = np.array([bn[0] * Wimg, bn[1] * Himg, bn[2] * Wimg, bn[3] * Himg], np.float32)
        if (box_px[2] - box_px[0]) < 2 or (box_px[3] - box_px[1]) < 2:
            continue
        img_rgb = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
        try:
            r = runner.predict(img_rgb, box_px, 1.0)      # is_right=1
        except Exception as e:
            print(f"  {seq} f{t}: predict failed: {e}", flush=True)
            continue
        if r is None:
            continue
        kp3d, pred_cam = r                                # [21,3] root-rel, [s,tx,ty]
        box_center = ((box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2)
        box_size = max(box_px[2] - box_px[0], box_px[3] - box_px[1])
        cam_t = cam_t_with_focal(pred_cam, box_center, box_size, Wimg, Himg, focal)
        j_abs = kp3d + cam_t[None, :]                     # [21,3] absolute cam frame
        cam_joints[t, RH] = torch.from_numpy(np.asarray(j_abs[MANO21_TO_16], np.float32))
        valid[t, RH] = True
        n_pred += 1
    return {"cam_joints": cam_joints, "world_joints": world_joints, "valid": valid,
            "n_frames": N, "n_pred": n_pred, "res": (Wimg, Himg), "focal": focal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="wilor", choices=["wilor", "hamer"])
    ap.add_argument("--data_root", required=True, help="HOI4D test dir (seqs with hand_data + mp4)")
    ap.add_argument("--box_dir", default=None,
                    help="predicted-box store (detbox v3): <box_dir>/<seq>.pt. Default: GT box.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--validate", action="store_true",
                    help="print RR-MPJPE vs GT (should be ~63mm for WiLoR) + det rate as a "
                         "geometry/remap sanity check (needs GT joint caches present)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = build_runner(args.method, device)
    print(f"[producer] method={args.method} device={device} box_dir={args.box_dir}", flush=True)

    seqs = sorted(d for d in os.listdir(args.data_root)
                  if os.path.isdir(os.path.join(args.data_root, d)))
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    n_ok = 0
    rr_all = []
    for sq in seqs:
        sd = os.path.join(args.data_root, sq)
        out = build_seq(sd, sq, runner, args.box_dir)
        if out is None:
            continue
        torch.save({"cam_joints": out["cam_joints"], "world_joints": out["world_joints"],
                    "valid": out["valid"]}, os.path.join(args.out_dir, sq + ".pt"))
        n_ok += 1
        rate = out["n_pred"] / max(out["n_frames"], 1)
        print(f"[{n_ok}] {sq} N={out['n_frames']} pred={out['n_pred']} ({rate*100:.0f}%) "
              f"res={out['res']} focal={out['focal']:.1f}", flush=True)
        if args.validate:
            rr = _rr_vs_gt(sd, out["cam_joints"], out["valid"])
            if rr == rr:
                rr_all.append(rr)
    if args.validate and rr_all:
        print(f"VALIDATE method={args.method} box={'detbox' if args.box_dir else 'GT'} "
              f"mean RR-MPJPE={np.mean(rr_all):.1f}mm over {len(rr_all)} seqs "
              f"(WiLoR pose quality ~63mm expected; gross deviation => geometry/remap bug)",
              flush=True)
    print(f"NATIVE_BASELINE_DONE method={args.method} wrote {n_ok} seqs -> {args.out_dir}",
          flush=True)


def _rr_vs_gt(seq_dir, cam_joints, valid):
    """Root-relative MPJPE (mm) of RH cam_joints vs the cached GT cam joints. Translation-
    invariant, so it isolates pose/remap correctness from the absolute placement."""
    fc = os.path.join(seq_dir, "hand_data", "gt_joints_cache_cam_v2.pt")
    if not os.path.exists(fc):
        return float("nan")
    gt = torch.load(fc, map_location="cpu").float()          # [N,2,16,3]
    n = min(gt.shape[0], cam_joints.shape[0])
    p, g, v = cam_joints[:n, RH], gt[:n, RH], valid[:n, RH]
    fin = torch.isfinite(p).all(-1).all(-1) & v
    if int(fin.sum()) == 0:
        return float("nan")
    p, g = p[fin], g[fin]
    p = p - p[:, :1]
    g = g - g[:, :1]
    return float(1000.0 * torch.sqrt(((p - g) ** 2).sum(-1)).mean())


if __name__ == "__main__":
    main()
