"""Pack H2O subject4 clips into mp4 videos for HaWoR, + per-clip GT joints.

HaWoR consumes a video clip (contiguous frames; runs detector+tracker+SLAM+infiller).
For each selected subject4 packed npz we:
  - write the [N,224,224,3] RGB frames to <out>/<clip>.mp4 at 30 fps (matches HaWoR's
    detect_track_video ffmpeg fps=30, so frame index i in the mp4 == row i here);
  - save GT joints21 (camera frame, METRES, MANO order), valid flags, and the
    pack-adjusted intrinsics to <out>/<clip>_gt.npz for the C-MPJPE metric.

Joint conventions copied verbatim from scripts/eval_cmpjpe.py (same as the WiLoR run).
"""
import argparse
import glob
import os
import subprocess

import numpy as np

H2O_TO_MANO = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
# BUG FIX 2026-08-06. IDX16 used to be [H2O_TO_MANO[k] for k in MANO21_TO_16], i.e. the remap
# applied TWICE, which put fingertips 8/12/16 into kinematic base slots. The result was still a
# permutation of 0..20, so the paired PA check could not detect it. H2O_TO_MANO is already the
# H2O -> smplx ordering: its first 16 entries are the kinematic joints and its last 5 the tips.
# Canonical definition: h2o_to_currentproto.py:125 (H2O16_IDX = H2O_TO_MANO[:16]).
IDX16 = H2O_TO_MANO[:16]
IDX21 = list(H2O_TO_MANO)
PACK_RES = 224
H2O_FULL_W, H2O_FULL_H = 1280, 720


def gt_joints(j128):
    left = j128[:, 1:64].reshape(-1, 21, 3)
    right = j128[:, 65:128].reshape(-1, 21, 3)
    j21 = np.stack([left[:, IDX21, :], right[:, IDX21, :]], axis=1)  # [N,2,21,3]
    valid = np.stack([j128[:, 0], j128[:, 64]], axis=1)              # [N,2]
    return j21.astype(np.float32), valid.astype(np.float32)


def adjusted_K(K):
    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    x0 = (H2O_FULL_W - H2O_FULL_H) // 2
    s = PACK_RES / float(H2O_FULL_H)
    return np.array([fx * s, fy * s, (cx - x0) * s, cy * s], dtype=np.float32)


def write_mp4(rgb, path, fps=30):
    """rgb [N,224,224,3] uint8 -> mp4 via ffmpeg rawvideo pipe (lossless-ish yuv)."""
    n, h, w, _ = rgb.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "12", "-r", str(fps), path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(rgb).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed for {path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2o", default="/work/scratch/dmonopoli/h2o_packed")
    ap.add_argument("--subject", default="subject4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=6)
    ap.add_argument("--min-frames", type=int, default=40,
                    help="skip clips shorter than this (SLAM needs enough motion)")
    ap.add_argument("--max-frames", type=int, default=300,
                    help="cap frames per clip to bound runtime")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.h2o, f"*{args.subject}*.npz")))
    print(f"{args.subject}: {len(files)} clips available")
    done = 0
    for f in files:
        if done >= args.n_clips:
            break
        d = np.load(f)
        rgb = d["rgb"]                          # [N,224,224,3] uint8
        j128 = d["joints"].astype(np.float32)   # [N,128]
        K = d["K"].astype(np.float32)
        N = rgb.shape[0]
        if N < args.min_frames:
            print(f"  skip {os.path.basename(f)} (only {N} frames)")
            continue
        if N > args.max_frames:
            rgb = rgb[:args.max_frames]
            j128 = j128[:args.max_frames]
            N = args.max_frames
        j21, valid = gt_joints(j128)            # [N,2,21,3], [N,2]
        Kadj = adjusted_K(K)
        clip = os.path.splitext(os.path.basename(f))[0]
        mp4 = os.path.join(args.out, f"{clip}.mp4")
        write_mp4(rgb, mp4)
        np.savez_compressed(
            os.path.join(args.out, f"{clip}_gt.npz"),
            joints21=j21,       # [N,2,21,3] m, MANO order, cam frame
            valid=valid,        # [N,2]
            Kadj=Kadj,          # [4] fx,fy,cx,cy for 224 frame
            n_frames=N,
            clip=clip,
        )
        print(f"  wrote {mp4} ({N} frames), GT npz")
        done += 1
    print(f"packed {done} clips into {args.out}")


if __name__ == "__main__":
    main()
