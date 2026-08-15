#!/usr/bin/env python3
"""Verify every feature-cache clip is a readable zip, without reading 1.8 TB.

A truncated `.pt` is not caught by a size threshold. The clip that killed mix5 after eight and a
half hours of training was 5.26 MB against a typical 31 MB, and the sbatch guard looked for files
under 1 MB. Worse, a file truncated at 90% has a plausible size and no readable archive at all.

A torch `.pt` is a zip, and a zip's end-of-central-directory record sits in the last bytes of the
file. Seeking to the tail and looking for its signature costs one small read per clip instead of a
full deserialization, so the whole cache is checkable in minutes rather than hours.

    python -m scripts.check_feature_cache /cluster/scratch/dmonopoli/featcache32 [--delete]
"""
import argparse
import os
import sys

EOCD = b"PK\x05\x06"
TAIL = 1 << 16


def _is_readable_zip(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        if size < 22:
            return False
        with open(path, "rb") as f:
            f.seek(max(0, size - TAIL))
            return EOCD in f.read()
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--delete", action="store_true",
                    help="remove the damaged clips; the loader then drops them instead of crashing")
    a = ap.parse_args()

    bad, n = [], 0
    for dirpath, _, files in os.walk(a.root):
        for fn in files:
            if not fn.endswith(".pt"):
                continue
            n += 1
            p = os.path.join(dirpath, fn)
            if not _is_readable_zip(p):
                bad.append((p, os.path.getsize(p)))
            if n % 10000 == 0:
                print(f"[check] {n} clips, {len(bad)} damaged", flush=True)

    print(f"[check] {n} clips scanned, {len(bad)} damaged")
    for p, s in bad:
        print(f"  DAMAGED {s/1e6:8.2f} MB  {p}")
        if a.delete:
            os.remove(p)
            print(f"  removed {p}")
    print(f"CACHE_CHECK_DONE scanned={n} damaged={len(bad)}")
    return 1 if bad and not a.delete else 0


if __name__ == "__main__":
    sys.exit(main())
