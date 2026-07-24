"""PHALP-free Dyn-HaMR track exporter: run the single-frame hand estimator on OUR detbox v3
crops and emit the two per-frame JSONs Dyn-HaMR's optimizer reads, in its EXACT format. This
bypasses Dyn-HaMR's PHALP+detectron2 tracking AND its DROID-SLAM (camera comes from
ours_slam_to_dynhamr_cam.py) -> Dyn-HaMR runs input-matched on our boxes + our SLAM.

Dyn-HaMR reads, per track <tid> (right hand only -> tid "001", is_right must == int(tid)):
  track_preds/<seq>/001/<frame>_mano.json      = {betas[10], body_pose[15,3] aa, global_orient[3] aa,
                                                  cam_trans[3], is_right}   (data/tools.py:read_mano_preds)
  track_preds/<seq>/001/<frame>_keypoints.json = {"people":[{"pose_keypoints_2d":[x,y,c ...21 joints]}]}
Frame name = zero-padded decode index (matches images/ + shots/ + the box rows), consistent with
scripts.hoi4d_to_dynhamr.

Geometry is IDENTICAL to scripts.build_native_baseline_preds (the verified Kfix true-focal producer):
we crop by the detbox, run wilor-mini, re-place the root under the REAL HOI4D focal (cam_t_with_focal),
and PROJECT the absolute 3D joints to pixels for the 2D keypoints (robust; no dependency on the
estimator's own 2D key). The MANO axis-angle params come from the estimator's pred_mano_params
(rotmat -> axis-angle via cv2.Rodrigues, exactly as Dyn-HaMR's own export_hamer.unpack_frame does).
On the first predicted frame the estimator output keys are printed so the MANO-key mapping is verified
empirically, never assumed.

NOTE: we use WiLoR as the single-frame estimator (the verified pipeline in our infra). Dyn-HaMR is
HaMeR-based; WiLoR/HaMeR share the MANO(15 hand joints + global) convention, so this is a faithful
single-frame init for Dyn-HaMR's dynamics optimizer. Swap in a HaMeRRunner with the same
predict_full() contract if a HaMeR init is required for the camera-ready row.

Usage (Euler/student, wilor-mini env):
    python -m scripts.dynhamr_export_from_boxes --data_root <hoi4d test> --box_dir $S/hoi4d_detboxes_v3 \
        --out_root <dynhamr root> --tid 001
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from scripts.build_native_baseline_preds import (RH, cam_t_with_focal,
                                                 load_boxes, load_focal)


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """[...,3,3] rotation matrices -> [...,3] axis-angle (cv2.Rodrigues per matrix)."""
    R = np.asarray(R, np.float64).reshape(-1, 3, 3)
    aa = np.stack([cv2.Rodrigues(r)[0].reshape(3) for r in R], axis=0)
    return aa.astype(np.float32)


class WiLoRFull:
    """wilor-mini wrapper returning the full MANO params + weak-persp cam needed for Dyn-HaMR init."""

    def __init__(self, device, dtype=torch.float32):
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
        self.pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
        self._logged = False

    def predict_full(self, img_rgb, box_px, is_right):
        out = self.pipe.predict_with_bboxes(
            img_rgb, np.array([box_px], np.float32), np.array([is_right], np.float32))
        if not out:
            return None
        wp = out[0]["wilor_preds"]
        if not self._logged:
            print(f"[wilor_preds keys] {sorted(wp.keys())}", flush=True)
            mp = wp.get("pred_mano_params") or wp.get("mano_params")
            if isinstance(mp, dict):
                print(f"[mano_params keys] {sorted(mp.keys())} shapes="
                      f"{ {k: np.asarray(v).shape for k, v in mp.items()} }", flush=True)
            self._logged = True
        # wilor_mini exposes global_orient/hand_pose as TOP-LEVEL keys of wilor_preds (not nested
        # under pred_mano_params); fall back to wp itself. betas may be absent -> mean shape (zeros).
        mp = wp.get("pred_mano_params") or wp.get("mano_params") or wp
        b = mp.get("betas", wp.get("betas"))
        betas = np.asarray(b).reshape(-1)[:10] if b is not None else np.zeros(10, np.float64)
        if betas.shape[0] < 10:
            betas = np.pad(betas, (0, 10 - betas.shape[0]))
        return {
            "kp3d": np.asarray(wp["pred_keypoints_3d"])[0],   # [21,3] root-rel (m)
            "pred_cam": np.asarray(wp["pred_cam"])[0],        # [s,tx,ty]
            "global_orient": np.asarray(mp["global_orient"]).reshape(-1, 3, 3),  # [1,3,3]
            "hand_pose": np.asarray(mp["hand_pose"]).reshape(-1, 3, 3),          # [15,3,3]
            "betas": betas,                                                      # [10]
        }


def project_px(joints_cam, focal, cx, cy):
    """[J,3] camera-frame joints (m) -> [J,3] (x_px, y_px, conf=1)."""
    z = np.clip(joints_cam[:, 2], 1e-6, None)
    x = focal * joints_cam[:, 0] / z + cx
    y = focal * joints_cam[:, 1] / z + cy
    return np.stack([x, y, np.ones_like(x)], axis=1)


def export_seq(seq_dir, seq, runner, box_dir, out_root, tid):
    b = load_boxes(seq_dir, seq, box_dir)
    if b is None:
        return None
    boxes_norm, box_valid = b
    f, cx0, cy0 = load_focal(seq_dir)

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
    focal = f * (Wimg / (2.0 * cx0))
    cx, cy = Wimg / 2.0, Himg / 2.0

    track_dir = os.path.join(out_root, "dynhamr", "track_preds", seq, tid)
    os.makedirs(track_dir, exist_ok=True)
    is_right = int(tid)                       # Dyn-HaMR asserts is_right == int(tid)
    n_pred = 0
    for t in range(N):
        name = f"{t:06d}"
        if not box_valid[t]:
            continue
        bn = boxes_norm[t]
        box_px = np.array([bn[0] * Wimg, bn[1] * Himg, bn[2] * Wimg, bn[3] * Himg], np.float32)
        if (box_px[2] - box_px[0]) < 2 or (box_px[3] - box_px[1]) < 2:
            continue
        img_rgb = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
        try:
            r = runner.predict_full(img_rgb, box_px, float(is_right))
        except Exception as e:
            print(f"  {seq} f{t}: predict failed: {e}", flush=True)
            continue
        if r is None:
            continue
        box_center = ((box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2)
        box_size = max(box_px[2] - box_px[0], box_px[3] - box_px[1])
        cam_t = cam_t_with_focal(r["pred_cam"], box_center, box_size, Wimg, Himg, focal)  # [3] metric
        j_abs = r["kp3d"] + cam_t[None, :]                # [21,3] absolute cam frame (m)

        # _mano.json (axis-angle MANO init + metric cam translation)
        mano_json = {
            "betas": r["betas"][:10].astype(np.float64).tolist(),
            "body_pose": rotmat_to_aa(r["hand_pose"]).astype(np.float64).tolist(),        # [15,3]
            "global_orient": rotmat_to_aa(r["global_orient"])[0].astype(np.float64).tolist(),  # [3]
            "cam_trans": cam_t.astype(np.float64).tolist(),
            "is_right": is_right,
        }
        with open(os.path.join(track_dir, f"{name}_mano.json"), "w") as fp:
            json.dump(mano_json, fp)

        # _keypoints.json (projected 21 joints; OpenPose-style flat list)
        kp = project_px(j_abs, focal, cx, cy).reshape(-1).astype(np.float64).tolist()
        with open(os.path.join(track_dir, f"{name}_keypoints.json"), "w") as fp:
            json.dump({"people": [{"pose_keypoints_2d": kp}]}, fp)
        n_pred += 1
    return {"seq": seq, "N": N, "n_pred": n_pred, "res": (Wimg, Himg), "focal": focal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--box_dir", default=None, help="detbox v3 store; default GT box cache")
    ap.add_argument("--out_root", required=True, help="Dyn-HaMR data root")
    ap.add_argument("--tid", default="001", help="track id == is_right (right hand -> 001)")
    ap.add_argument("--max_seqs", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = WiLoRFull(device)
    print(f"[dynhamr export] device={device} box_dir={a.box_dir} tid={a.tid}", flush=True)

    seqs = sorted(d for d in os.listdir(a.data_root) if os.path.isdir(os.path.join(a.data_root, d)))
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    n_ok = 0
    for sq in seqs:
        r = export_seq(os.path.join(a.data_root, sq), sq, runner, a.box_dir, a.out_root, a.tid)
        if r:
            n_ok += 1
            rate = r["n_pred"] / max(r["N"], 1)
            print(f"[{n_ok}] {sq} N={r['N']} pred={r['n_pred']} ({rate*100:.0f}%) res={r['res']}", flush=True)
    print(f"DYNHAMR_EXPORT_DONE wrote {n_ok} seqs -> {a.out_root}/dynhamr/track_preds", flush=True)


if __name__ == "__main__":
    main()
