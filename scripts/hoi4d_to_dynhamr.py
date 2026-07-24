"""HOI4D -> Dyn-HaMR input tree (estimator-independent parts).

Dyn-HaMR (dyn-hamr/data/dataset.py) reads a per-sequence input tree:
    <root>/images/<seq>/<frame>.jpg                     -- decoded RGB frames
    <root>/dynhamr/track_preds/<seq>/<tid>/<frame>_keypoints.json  -- OpenPose-style 2D (est.)
    <root>/dynhamr/track_preds/<seq>/<tid>/<frame>_mano.json       -- single-frame MANO init (est.)
    <root>/dynhamr/shot_idcs/<seq>.json                 -- {<frame>: shot_int}
    <root>/dynhamr/cameras/<seq>/shot-<idx>/cameras.npz -- camera (from ours_slam_to_dynhamr_cam)

This script produces the two estimator-INDEPENDENT pieces:
  * images/<seq>/<frame>.jpg  (decoded from video_main_rgb.mp4; frame name = zero-padded index)
  * shot_idcs/<seq>.json      (single shot: every frame -> 0, HOI4D clips are single-shot)
It also copies our detbox v3 right-hand boxes to <root>/dynhamr/boxes/<seq>.npy ([N,4] px xyxy +
[N] valid), which the box-seeded HaMeR exporter (dynhamr_export_from_boxes, run in the HaMeR env)
consumes to crop and emit the two per-track JSONs -- so the SAME detbox v3 seeds Dyn-HaMR as every
other row. The camera comes from ours_slam_to_dynhamr_cam.py (our SLAM trajectory; skips DROID).

Frame naming MUST match across images/, tracks/, shots/, and load_cameras_npz indexing: we use
the decode order i -> f"{i:06d}", identical to the box row order in the caches.

Usage:
    python -m scripts.hoi4d_to_dynhamr --test_root <hoi4d test> --box_dir $S/hoi4d_detboxes_v3 \
        --out_root <root>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

RH = 1  # right-hand slot (HOI4D convention)


def load_boxes(seq_dir: str, seq: str, box_dir: str | None):
    """Right-hand boxes as (boxes_norm [N,4] xyxy normalized, valid [N]) from detbox v3 (box_dir)
    or the GT cache. Schema-identical to build_native_baseline_preds.load_boxes."""
    if box_dir is not None:
        bp = os.path.join(box_dir, seq + ".pt")
    else:
        bp = os.path.join(seq_dir, "hand_data", "hand_bboxes_v2_rf1.5_res224x224.pt")
    if not os.path.exists(bp):
        return None
    bb = torch.load(bp, map_location="cpu")
    boxes = np.asarray(bb["bboxes"], np.float32)[:, RH]   # [N,4] normalized xyxy
    valid = np.asarray(bb["valid"], bool)[:, RH]          # [N]
    return boxes, valid


def process_seq(seq_dir: str, seq: str, box_dir: str | None, out_root: str) -> dict | None:
    b = load_boxes(seq_dir, seq, box_dir)
    if b is None:
        print(f"SEQ_SKIP {seq}: no boxes", flush=True)
        return None
    boxes_norm, box_valid = b

    img_dir = os.path.join(out_root, "images", seq)
    box_dir_out = os.path.join(out_root, "dynhamr", "boxes")
    shot_dir = os.path.join(out_root, "dynhamr", "shot_idcs")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(box_dir_out, exist_ok=True)
    os.makedirs(shot_dir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(seq_dir, "video_main_rgb.mp4"))
    frame_names, i, Wimg, Himg = [], 0, None, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if Wimg is None:
            Himg, Wimg = fr.shape[:2]
        name = f"{i:06d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), fr)
        frame_names.append(name)
        i += 1
    cap.release()
    if not frame_names:
        print(f"SEQ_SKIP {seq}: no decoded frames", flush=True)
        return None

    N = min(len(frame_names), boxes_norm.shape[0])
    frame_names = frame_names[:N]

    # detbox v3 boxes -> pixel xyxy, aligned to the decoded frame indices (est. exporter reads this)
    boxes_px = np.zeros((N, 4), np.float32)
    valid = np.zeros(N, bool)
    for t in range(N):
        bn = boxes_norm[t]
        boxes_px[t] = [bn[0] * Wimg, bn[1] * Himg, bn[2] * Wimg, bn[3] * Himg]
        valid[t] = bool(box_valid[t]) and (boxes_px[t, 2] - boxes_px[t, 0]) >= 2 and (boxes_px[t, 3] - boxes_px[t, 1]) >= 2
    np.save(os.path.join(box_dir_out, seq + ".npy"),
            {"boxes_px": boxes_px, "valid": valid, "frame_names": frame_names,
             "W": Wimg, "H": Himg}, allow_pickle=True)

    # single-shot json: every frame -> shot 0. Keys MUST be the actual image FILENAMES (with
    # .jpg): Dyn-HaMR's get_shot_img_files uses the shot-json keys directly as image paths
    # (data/dataset.py: img_paths = join(img_dir, key)); bare indices resolve to a missing file.
    with open(os.path.join(shot_dir, seq + ".json"), "w") as f:
        json.dump({nm + ".jpg": 0 for nm in frame_names}, f, indent=1)

    return {"seq": seq, "N": N, "res": (Wimg, Himg), "valid": int(valid.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--box_dir", default=None, help="detbox v3 store; default GT box cache")
    ap.add_argument("--out_root", required=True, help="Dyn-HaMR data root")
    ap.add_argument("--max_seqs", type=int, default=0)
    args = ap.parse_args()

    seqs = sorted(d for d in os.listdir(args.test_root)
                  if os.path.isdir(os.path.join(args.test_root, d)))
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    print(f"Formatting {len(seqs)} HOI4D seqs for Dyn-HaMR -> {args.out_root} "
          f"(box_dir={args.box_dir})", flush=True)
    count = 0
    for s in seqs:
        r = process_seq(os.path.join(args.test_root, s), s, args.box_dir, args.out_root)
        if r:
            count += 1
            print(f"[{count}] {r['seq']} N={r['N']} res={r['res']} valid_boxes={r['valid']}", flush=True)
    print(f"HOI4D_TO_DYNHAMR_DONE formatted {count}/{len(seqs)} seqs -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
