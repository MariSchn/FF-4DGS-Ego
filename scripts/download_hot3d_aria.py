#!/usr/bin/env python3
"""Download the minimal HOT3D-Aria payload needed for egocentric hand training.

The full release is ~526 GB per the manifest, but almost all of that is data we never touch:
main_vrs (329 GB, the raw multi-stream recording - video_main_rgb already carries the RGB),
mps_artifacts (89 GB) and mps_slam_points (84 GB, semi-dense point cloud). The three groups we
actually need come to ~22 GB for all 198 sequences:

  video_main_rgb  (~98 MB/seq, mp4)  ego RGB stream
  hand_data       (~7 MB/seq,  zip)  mano_hand_pose_trajectory.jsonl + umetrack + user profile
  ground_truth    (~4.8 MB/seq, zip) headset_trajectory.csv (CAMERA TRAJECTORY),
                                     camera_models.json (intrinsics), box2d_hands.csv,
                                     masks/mask_hand_pose_available.csv, mask_hand_visible.csv

ground_truth is the one people forget: without headset_trajectory.csv there is no camera
trajectory, and world-space metrics are simply not computable. (The copy of this script that
shipped with the manifest downloads only video_main_rgb + hand_data.)

Resumable: a sequence's file is skipped when it already exists with the manifest's expected size,
so the job can be requeued after a wall-clock timeout without redoing work.

Euler note: COMPUTE nodes have no direct internet. Run under `module load eth_proxy`.
"""
import argparse
import json
import os
import zipfile

import requests

GROUPS = ("video_main_rgb", "hand_data", "ground_truth")
ZIP_GROUPS = {"hand_data", "ground_truth"}


def _get(url, path, expect_bytes=None, chunk=1 << 20):
    """Stream url -> path. Returns bytes written, or -1 if skipped as already complete."""
    if expect_bytes and os.path.exists(path) and abs(os.path.getsize(path) - expect_bytes) < 1024:
        return -1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        n = 0
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
                    n += len(c)
    os.replace(tmp, path)
    return n


def fetch_sequence(sid, entry, out_root, groups):
    """Download the requested groups for one sequence. Returns (bytes, n_skipped)."""
    seq_dir = os.path.join(out_root, sid)
    total, skipped = 0, 0
    for g in groups:
        if g not in entry:
            print(f"  [{sid}] group {g} absent from manifest", flush=True)
            continue
        url = entry[g]["download_url"]
        size = entry[g].get("file_size_bytes")
        if g in ZIP_GROUPS:
            marker = os.path.join(seq_dir, g, ".complete")
            if os.path.exists(marker):
                skipped += 1
                continue
            zpath = os.path.join(seq_dir, f"{g}.zip")
            n = _get(url, zpath, size)
            total += max(n, 0)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(os.path.join(seq_dir, g))
            os.remove(zpath)                       # keep the inode footprint down
            open(marker, "w").close()
        else:
            n = _get(url, os.path.join(seq_dir, f"{g}.mp4"), size)
            if n < 0:
                skipped += 1
            else:
                total += n
    return total, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Hot3DAria_download_urls.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--only", default="", help="comma-separated sequence ids, or a file of them")
    ap.add_argument("--exclude", default="", help="comma-separated sequence ids, or a file of them")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    def _idset(v):
        if not v:
            return set()
        if os.path.exists(v):
            return {ln.strip() for ln in open(v) if ln.strip() and not ln.startswith("#")}
        return {x.strip() for x in v.split(",") if x.strip()}

    seqs = json.load(open(a.manifest))["sequences"]
    only, excl = _idset(a.only), _idset(a.exclude)
    ids = sorted(seqs)
    if only:
        ids = [i for i in ids if i in only]
    if excl:
        ids = [i for i in ids if i not in excl]
    if a.limit:
        ids = ids[: a.limit]
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]

    plan = sum(seqs[i][g]["file_size_bytes"] for i in ids for g in groups if g in seqs[i])
    print(f"[hot3d] {len(ids)}/{len(seqs)} sequences x {groups}", flush=True)
    print(f"[hot3d] planned download: {plan / 1e9:.1f} GB -> {a.out}", flush=True)
    if a.dry_run:
        return

    done_b, done_n, skipped_n = 0, 0, 0
    for k, sid in enumerate(ids, 1):
        try:
            b, s = fetch_sequence(sid, seqs[sid], a.out, groups)
        except Exception as e:                                    # noqa: BLE001 keep going
            print(f"  [{sid}] FAILED {type(e).__name__}: {e}", flush=True)
            continue
        done_b += b
        skipped_n += s
        done_n += 1
        if k % 5 == 0 or k == len(ids):
            print(f"[hot3d] {k}/{len(ids)} seqs | {done_b / 1e9:.1f} GB new | "
                  f"{skipped_n} files already present", flush=True)
    print(f"HOT3D_DOWNLOAD_DONE seqs={done_n}/{len(ids)} new_bytes={done_b / 1e9:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
