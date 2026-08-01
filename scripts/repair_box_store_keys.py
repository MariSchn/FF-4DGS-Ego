#!/usr/bin/env python3
"""Restore keys that a box regeneration dropped, from the .square_backup.pt written beside it.

WHY. scripts/regen_boxes_joints_to_bbox.py used to write a fresh {bboxes, valid, convention}
dict, which deleted "gt" - the GT MANO params rewritten from world into camera frame that
HOT3DHandDataset reads as cached["gt"]. The H2O store was regenerated with that version and now
raises KeyError: 'gt' on every eval. The regenerated BOXES are correct (square_frac 0.0011,
outside01_frac 0.0234 over 177 seqs); only the carried-over keys are missing.

SAFE because "gt" is box-INDEPENDENT: _transform_gt_to_crop_local performs a world -> camera
transform of transl/global_orient and, although it accepts bbox_frames, never reads it (verified:
bbox_frames and valid_frames appear only in that function's signature). So the backup's "gt" is
exactly what a correct regeneration would have carried across.

This restores ONLY the missing keys and never touches bboxes/valid, so the corrected convention
is preserved.
"""
from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--name", default="hand_bboxes_v2_rf1.5_res224x224.pt")
    ap.add_argument("--keys", default="gt", help="comma-separated keys to restore if missing")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    keys = tuple(k for k in args.keys.split(",") if k)
    n_ok = n_fixed = n_nobackup = n_skip = 0

    for s in sorted(os.listdir(args.data_root)):
        p = os.path.join(args.data_root, s, "hand_data", args.name)
        if not os.path.exists(p):
            continue
        cur = torch.load(p, map_location="cpu", weights_only=True)
        absent = [k for k in keys if k not in cur]
        if not absent:
            n_ok += 1
            continue

        bak_p = p.replace(".pt", ".square_backup.pt")
        if not os.path.exists(bak_p):
            print(f"  !! {s}: missing {absent} and NO backup at {os.path.basename(bak_p)}")
            n_nobackup += 1
            continue
        bak = torch.load(bak_p, map_location="cpu", weights_only=True)
        recoverable = [k for k in absent if k in bak]
        if len(recoverable) != len(absent):
            print(f"  !! {s}: backup lacks {[k for k in absent if k not in bak]}")
            n_skip += 1
            continue

        # Sanity: the backup must describe the SAME number of frames, or it belongs to another
        # sequence and pasting its GT in would silently corrupt the store.
        if "bboxes" in bak and len(bak["bboxes"]) != len(cur["bboxes"]):
            print(f"  !! {s}: backup has {len(bak['bboxes'])} frames, current has "
                  f"{len(cur['bboxes'])} - refusing to merge mismatched files")
            n_skip += 1
            continue

        for k in recoverable:
            cur[k] = bak[k]
        if not args.dry_run:
            torch.save(cur, p)
        n_fixed += 1
        if n_fixed <= 3:
            print(f"  [{s}] restored {recoverable} (frames={len(cur['bboxes'])})")

    verb = "would restore" if args.dry_run else "restored"
    print(f"\nREPAIR: {verb} keys in {n_fixed} seqs | already-ok {n_ok} | "
          f"no-backup {n_nobackup} | skipped {n_skip}")
    if n_nobackup or n_skip:
        raise SystemExit("REPAIR INCOMPLETE: some sequences could not be restored.")


if __name__ == "__main__":
    main()
