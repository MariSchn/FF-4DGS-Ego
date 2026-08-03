#!/usr/bin/env python3
"""Feed OUR detector boxes into HaWoR, without modifying the HaWoR repo.

CYRUS DIRECTIVE: every method in the world-space table must run on OUR predictive detector boxes
(detbox v3), not on each method's own detector. Otherwise the comparison is confounded by box
source rather than by method - the D2-10 attack. HaWoR currently runs its own YOLO hand detector
inside detect_track_video.

HOW THIS WORKS - no patching required. detect_track_video short-circuits on a cache:

    if os.path.exists(f'{seq_folder}/tracks_{start}_{end}/model_boxes.npy'):
        return start_idx, end_idx, seq_folder, imgfiles      # detector never runs

So writing that cache ourselves, in HaWoR's own format, makes HaWoR consume our boxes while its
source stays untouched. Patching a third-party pipeline would be far easier to get subtly wrong
and much harder to keep in sync.

THE FORMAT, read off lib/pipeline/tools.py:detect_track and hawor_video.py:
  model_tracks.npy : np.array(dict, dtype=object), loaded with .item(). Keys are track ids;
                     values are lists of per-frame dicts:
                        {'frame': int, 'det': True,
                         'det_box': float array (1,5) = [x1,y1,x2,y2,conf] in PIXELS,
                         'det_handedness': array (1,) with >0 right, ==0 left}
  model_boxes.npy  : detect_track builds boxes_ = [] and never appends, so it is an EMPTY object
                     array. Its only role is to exist, because it is the cache sentinel.
  Track ids when the tracker has no id: right = 10000, left = 5000. We use those constants so a
  downstream consumer that special-cases them behaves identically.

CONVENTIONS BRIDGED HERE, each a chance to silently corrupt the comparison:
  * our boxes are NORMALISED to [0,1] by the source frame size; HaWoR wants PIXELS.
  * our hand axis is index 0 = LEFT, index 1 = RIGHT (RH = 1, see eval_worldspace_baseline).
  * our boxes are deliberately UNCLAMPED (they may fall outside the frame). We keep them
    unclamped by default because clamping would change the very box statistic under comparison;
    --clamp is available if a downstream stage cannot cope.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

LEFT, RIGHT = 0, 1
TRACK_ID = {LEFT: 5000, RIGHT: 10000}
HANDEDNESS = {LEFT: 0.0, RIGHT: 1.0}


def frame_size_from_video(seq_dir: str) -> tuple[float, float] | None:
    """Read the TRUE (W, H) from the sequence video. Preferred over any inference from intrinsics.

    The intrinsics route assumes the principal point is exactly centred. On the HOI4D store it is
    not: [f, cx, cy] = [219.92, 114.28, 108.52] implies 229x217 while the video is really
    224x224 - a ~2% error that would bias EVERY exported box in a comparison whose entire point
    is input matching.
    """
    p = os.path.join(seq_dir, "video_main_rgb.mp4")
    if not os.path.exists(p):
        return None
    try:
        import decord
        vr = decord.VideoReader(p)
        h, w = vr[0].asnumpy().shape[:2]
        return (float(w), float(h))
    except Exception:
        return None


def frame_size_from_intrinsics(hand_data_dir: str) -> tuple[float, float] | None:
    """FALLBACK only: infer (W, H) from cam_intrinsics [f, cx, cy] assuming a centred principal
    point. Use frame_size_from_video when a video exists; this is an approximation."""
    p = os.path.join(hand_data_dir, "cam_intrinsics.pt")
    if not os.path.exists(p):
        return None
    k = torch.load(p, map_location="cpu")
    v = np.asarray(k, dtype=np.float64).ravel()
    if v.size < 3:
        return None
    _, cx, cy = float(v[0]), float(v[1]), float(v[2])
    return (round(cx * 2.0), round(cy * 2.0))


def resolve_frame_size(seq_dir: str) -> tuple[tuple[float, float] | None, str]:
    """True video size if readable, else the intrinsics approximation. Returns (size, source)."""
    s = frame_size_from_video(seq_dir)
    if s is not None:
        return s, "video"
    return frame_size_from_intrinsics(os.path.join(seq_dir, "hand_data")), "intrinsics(approx)"


def build_tracks(boxes: np.ndarray, valid: np.ndarray, w: float, h: float,
                 conf: float, clamp: bool) -> dict:
    """Convert our [N,2,4] normalised boxes into HaWoR's track dict."""
    tracks: dict[int, list] = {}
    n_frames = boxes.shape[0]
    for hand in (LEFT, RIGHT):
        entries = []
        for t in range(n_frames):
            if not bool(valid[t, hand]):
                continue
            b = boxes[t, hand].astype(np.float64)
            if not np.isfinite(b).all():
                continue
            x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
            if clamp:
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)
            if x2 <= x1 or y2 <= y1:
                continue
            entries.append({
                "frame": int(t),
                "det": True,
                "det_box": np.array([[x1, y1, x2, y2, conf]], dtype=np.float32),
                "det_handedness": np.array([HANDEDNESS[hand]], dtype=np.float32),
            })
        if entries:
            tracks[TRACK_ID[hand]] = entries
    return tracks


def export_sequence(seq_dir: str, out_root: str, seq_name: str, box_name: str,
                    conf: float, clamp: bool, start_idx: int = 0,
                    end_idx: int | None = None) -> tuple[str, dict]:
    hd = os.path.join(seq_dir, "hand_data")
    bp = os.path.join(hd, box_name)
    if not os.path.exists(bp):
        return "no-boxes", {}
    size, src = resolve_frame_size(seq_dir)
    if size is None:
        return "no-frame-size", {}
    w, h = size

    d = torch.load(bp, map_location="cpu")
    boxes = np.asarray(d["bboxes"], dtype=np.float64)          # [N,2,4] normalised
    valid = np.asarray(d["valid"]).astype(bool)                # [N,2]
    n = boxes.shape[0]
    end = (n - 1) if end_idx is None else end_idx

    tracks = build_tracks(boxes, valid, w, h, conf, clamp)
    if not tracks:
        return "empty-tracks", {}

    tdir = os.path.join(out_root, seq_name, f"tracks_{start_idx}_{end}")
    os.makedirs(tdir, exist_ok=True)
    # dtype=object 0-d arrays, exactly as detect_track_video writes them (hawor_video does .item()).
    np.save(os.path.join(tdir, "model_tracks.npy"), np.array(tracks, dtype=object))
    np.save(os.path.join(tdir, "model_boxes.npy"), np.array([], dtype=object))

    stats = {
        "frames": n, "W": w, "H": h, "size_src": src,
        "left_dets": len(tracks.get(TRACK_ID[LEFT], [])),
        "right_dets": len(tracks.get(TRACK_ID[RIGHT], [])),
        "outside_frame_frac": float(np.mean((boxes[valid] < 0) | (boxes[valid] > 1)))
        if valid.any() else float("nan"),
    }
    return "ok", stats


def export_from_flat_file(box_pt: str, seq_name: str, intr_root: str, out_root: str,
                          conf: float, clamp: bool) -> tuple[str, dict]:
    """Export from the FLAT detbox layout: <root>/<seq>.pt, one file per sequence.

    detbox v3 ships as flat per-sequence .pt files ({bboxes,valid,gt,det_hit}) rather than as a
    store, because it is designed to be swapped into a store's hand_bboxes_v2 slot. Those files
    carry no intrinsics, so the frame size comes from the matching GT store.
    """
    seq_dir = os.path.join(intr_root, seq_name)
    size, src = resolve_frame_size(seq_dir)
    if size is None:
        return "no-frame-size", {}
    w, h = size
    d = torch.load(box_pt, map_location="cpu")
    if "bboxes" not in d or "valid" not in d:
        return "bad-box-file", {}
    boxes = np.asarray(d["bboxes"], dtype=np.float64)
    valid = np.asarray(d["valid"]).astype(bool)
    n = boxes.shape[0]

    tracks = build_tracks(boxes, valid, w, h, conf, clamp)
    if not tracks:
        return "empty-tracks", {}
    tdir = os.path.join(out_root, seq_name, f"tracks_0_{n - 1}")
    os.makedirs(tdir, exist_ok=True)
    np.save(os.path.join(tdir, "model_tracks.npy"), np.array(tracks, dtype=object))
    np.save(os.path.join(tdir, "model_boxes.npy"), np.array([], dtype=object))
    return "ok", {
        "frames": n, "W": w, "H": h, "size_src": src,
        "left_dets": len(tracks.get(TRACK_ID[LEFT], [])),
        "right_dets": len(tracks.get(TRACK_ID[RIGHT], [])),
        "outside_frame_frac": float(np.mean((boxes[valid] < 0) | (boxes[valid] > 1)))
        if valid.any() else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="store with <seq>/hand_data/<box file>")
    ap.add_argument("--flat_box_dir", default=None,
                    help="FLAT detbox layout instead: a directory of <seq>.pt files (detbox v3). "
                         "--data_root is then used only to read each sequence's intrinsics.")
    ap.add_argument("--out_root", required=True,
                    help="HaWoR seq root; writes <seq>/tracks_<s>_<e>/model_{tracks,boxes}.npy")
    ap.add_argument("--box_name", default="hand_bboxes_v2_rf1.5_res224x224.pt")
    ap.add_argument("--conf", type=float, default=0.99,
                    help="confidence written into det_box; ours are predictions, not detections, "
                         "so a single high constant keeps HaWoR's thresholds from dropping frames")
    ap.add_argument("--clamp", action="store_true",
                    help="clip boxes into the frame. OFF by default: our convention is "
                         "deliberately unclamped and clamping would alter the quantity under test")
    ap.add_argument("--max_seqs", type=int, default=0)
    a = ap.parse_args()

    if a.flat_box_dir:
        seqs = sorted(f[:-3] for f in os.listdir(a.flat_box_dir)
                      if f.endswith(".pt") and not f.startswith("_"))
    else:
        seqs = sorted(d for d in os.listdir(a.data_root)
                      if os.path.isdir(os.path.join(a.data_root, d)))
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]

    counts: dict[str, int] = {}
    agg: list[dict] = []
    for s in seqs:
        if a.flat_box_dir:
            st, stats = export_from_flat_file(os.path.join(a.flat_box_dir, s + ".pt"), s,
                                              a.data_root, a.out_root, a.conf, a.clamp)
        else:
            st, stats = export_sequence(os.path.join(a.data_root, s), a.out_root, s,
                                        a.box_name, a.conf, a.clamp)
        counts[st] = counts.get(st, 0) + 1
        if stats:
            agg.append(stats)
            if len(agg) <= 3:
                print(f"  [{s}] frames={stats['frames']} {stats['W']}x{stats['H']} "
                      f"L={stats['left_dets']} R={stats['right_dets']} "
                      f"outside_frame={stats['outside_frame_frac']:.4f}")

    print(f"\nEXPORT status: {counts}")
    if not agg:
        raise SystemExit("EXPORT FAILED: no sequence produced tracks; HaWoR would silently fall "
                         "back to its OWN detector, which is exactly what this avoids.")
    print(f"EXPORT wrote {len(agg)} sequences to {a.out_root}")
    print(f"EXPORT left dets  {sum(x['left_dets'] for x in agg)}")
    print(f"EXPORT right dets {sum(x['right_dets'] for x in agg)}")
    print("NOTE: HaWoR now short-circuits its detector because model_boxes.npy exists. Confirm in "
          "its log that it does NOT print detection progress.")


if __name__ == "__main__":
    main()
