"""Dump a store's frames as the PNG layout the visualisation scripts read.

`vis_mano_on_renders` and `run_ours_gs` both take an `--export_root` of `<seq>/images/%06d.png`,
which so far exists only for HOI4D. H2O is the out-of-domain set and has no such export, so nothing
qualitative has ever been produced on it.

Frames go through the same cover-and-centre-crop the dataloader applies (`auxiliary.py:126-142`),
so a frame written here is the frame the model sees.
"""
from __future__ import annotations

import argparse
import os

from decord import VideoReader
from PIL import Image

from diffsynth.utils.auxiliary import center_crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", nargs="+", default=None, help="default: every sequence in the store")
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--res", type=int, default=224)
    a = ap.parse_args()

    seqs = a.seq or sorted(
        d for d in os.listdir(a.store)
        if os.path.exists(os.path.join(a.store, d, "video_main_rgb.mp4")))
    if not seqs:
        raise SystemExit(f"no sequence with video_main_rgb.mp4 under {a.store}")

    for s in seqs:
        vr = VideoReader(os.path.join(a.store, s, "video_main_rgb.mp4"))
        n = min(a.n_frames, len(vr))
        d = os.path.join(a.out, s, "images")
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            img = Image.fromarray(vr[i].asnumpy())
            center_crop(img, (a.res, a.res)).save(os.path.join(d, f"{i:06d}.png"))
        print(f"{s}: {n} frames -> {d}", flush=True)
    print(f"EXPORT_OK {len(seqs)} sequences", flush=True)


if __name__ == "__main__":
    main()
