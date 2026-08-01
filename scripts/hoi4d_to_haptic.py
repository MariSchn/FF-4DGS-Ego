"""HOI4D -> HaPTIC input adapter: build per-sequence frames/ + det.pkl + video_list.yaml.

HaPTIC's demo short-circuits its detector stack when <seq>/det.pkl exists
(det_utils.parse_det_seq), which lets us (a) skip the mmcv/mmpose/detectron2/ViTPose
install, (b) inject the REAL HOI4D intrinsics instead of its fabricated
focal=sqrt(W^2+H^2), and (c) inject HOI4D GT world->cam extrinsics as cTw so its
"world" output is a true world frame (HaPTIC does no SLAM; default cTw=identity).

Sources (per seq dir, our preprocess layout):
    video_main_rgb.mp4                          frames (any resolution; 224 store or HD re-extract)
    hand_data/cam_intrinsics.pt                 [3] = [f, cx, cy] AT THE STORE RESOLUTION
    hand_data/cam_extrinsics_cache.pt           [N,4,4] world->cam (= cTw)
    hand_data/hand_bboxes_v2_rf1.5_res224x224.pt  {"bboxes":[N,2,4] normalized xyxy TIGHT, "valid":[N,2]}

Right hand only (HOI4D hand index 1). Output per seq under <out_root>/<seq>/:
    00000.jpg ... frames  +  det.pkl (list of one seq_info dict)

Usage (CPU, no GPU):
    python -m scripts.hoi4d_to_haptic --data_root <hoi4d_test> --out_root <haptic_in> \
        --seq_list seqA seqB ... --max_frames 256
"""
from __future__ import annotations

import argparse
import os
import pickle

import cv2
import numpy as np
import torch

RH = 1  # right hand slot in our caches


def build_seq(seq_dir: str, out_dir: str, max_frames: int = 0,
              box_dir: str | None = None) -> dict | None:
    hd = os.path.join(seq_dir, "hand_data")
    K = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().flatten()
    f, cx, cy = float(K[0]), float(K[1]), float(K[2])
    ext = os.path.join(hd, "cam_extrinsics_cache.pt")
    cTw = torch.load(ext, map_location="cpu").float().numpy() if os.path.exists(ext) else None
    # box source: default GT cache, or an external predicted-box store (detbox v3), which is
    # schema-identical ({"bboxes":[N,2,4] normalized xyxy, "valid":[N,2]}) and consumed the
    # SAME way as the GT box here (same normalized convention, HaPTIC's own x1.5 crop scale on
    # top). box_dir keeps hoi4d_test157 untouched so GT-box runs stay valid. If box_dir is given
    # but this seq has no predicted box, SKIP the seq (never silently mix GT into a detbox run).
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    if box_dir is not None:
        bp = os.path.join(box_dir, seq_name + ".pt")
        if not os.path.exists(bp):
            print(f"SEQ_SKIP {seq_name}: no predicted box in {box_dir}", flush=True)
            return None
    else:
        bp = os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt")
    bb = torch.load(bp, map_location="cpu")
    boxes = np.asarray(bb["bboxes"], np.float32)[:, RH]        # [N,4] normalized xyxy (tight)
    valid = np.asarray(bb["valid"], bool)[:, RH]               # [N]

    cap = cv2.VideoCapture(os.path.join(seq_dir, "video_main_rgb.mp4"))
    os.makedirs(out_dir, exist_ok=True)
    imgs, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        imgs.append(frame.shape)
        i += 1
    cap.release()
    if i == 0:
        return None
    H, W = imgs[0][0], imgs[0][1]
    T = min(i, boxes.shape[0])

    # tight normalized box -> pixel center/size at THIS frame resolution; HaPTIC crop
    # convention (det_utils.py:195-196): scale = concat([w,h])/100 * 1.5
    b = boxes[:T] * np.array([W, H, W, H], np.float32)
    center = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)   # [T,2]
    wh = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).clip(min=8.0)     # [T,2]
    scale = wh / 100.0 * 1.5
    # carry-forward fill for invalid frames (HaPTIC expects a box every frame)
    lastc, lasts = None, None
    for t in range(T):
        if valid[t]:
            lastc, lasts = center[t].copy(), scale[t].copy()
        elif lastc is not None:
            center[t], scale[t] = lastc, lasts

    # intrinsics cache is at the 224 store resolution; rescale to the frames we wrote
    sx, sy = W / (2.0 * cx), H / (2.0 * cy)
    intr = np.array([[f * sx, 0, W / 2.0], [0, f * sy, H / 2.0], [0, 0, 1]], np.float32)

    seq = os.path.basename(seq_dir.rstrip("/"))
    seq_info = {
        "imgname": [f"{seq}/{t:05d}.jpg" for t in range(T)],
        "img_dir": os.path.dirname(out_dir.rstrip("/")),
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "focal": np.tile(intr, [T, 1, 1])[:, None],           # (T,1,3,3)
        "is_right": np.ones(T, int),
        "cTw": (cTw[:T] if cTw is not None
                else np.tile(np.eye(4, dtype=np.float32), [T, 1, 1])),
        "hand_pose": np.zeros([T, 45], np.float32),
        "hand_tsl": np.zeros([T, 3], np.float32),
        "valid": valid[:T].astype(int),
        "seq": f"{seq}_right",
    }
    with open(os.path.join(out_dir, "det.pkl"), "wb") as fh:
        pickle.dump([seq_info], fh)
    return {"seq": seq, "frames": T, "res": (W, H), "has_ext": cTw is not None,
            "valid_rate": float(valid[:T].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seq_list", nargs="*", default=None, help="default: all seqs in data_root")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames per seq (0 = all)")
    ap.add_argument("--box_dir", default=None,
                    help="external predicted-box store (e.g. detbox v3): <box_dir>/<seq>.pt with "
                         "{bboxes[N,2,4] normalized, valid[N,2]}. Default: the GT box in hand_data. "
                         "Non-destructive - hoi4d_test157 is never modified.")
    args = ap.parse_args()

    seqs = args.seq_list or sorted(d for d in os.listdir(args.data_root)
                                   if os.path.isdir(os.path.join(args.data_root, d)))
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    os.makedirs(args.out_root, exist_ok=True)
    done = []
    for sq in seqs:
        try:
            r = build_seq(os.path.join(args.data_root, sq),
                          os.path.join(args.out_root, sq), args.max_frames,
                          box_dir=args.box_dir)
        except Exception as e:
            print(f"SEQ_FAIL {sq}: {e}", flush=True)
            continue
        if r:
            done.append(r)
            print(f"[{len(done)}/{len(seqs)}] {sq} T={r['frames']} res={r['res']} "
                  f"ext={r['has_ext']} valid={r['valid_rate']:.2f}", flush=True)
    with open(os.path.join(args.out_root, "video_list.yaml"), "w") as fh:
        for r in done:
            fh.write(f"- {r['seq']}\n")
    n_ext = sum(1 for r in done if r["has_ext"])
    print(f"HAPTIC_ADAPT_DONE seqs={len(done)} with_extrinsics={n_ext} -> {args.out_root}")
    if n_ext < len(done):
        print("WARN: seqs without extrinsics get identity cTw -> world==camera frame there")


if __name__ == "__main__":
    main()
