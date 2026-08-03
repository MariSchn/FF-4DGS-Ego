#!/usr/bin/env python3
"""Sweep long-window evaluation over chunk sizes, and report which comparisons are LEGITIMATE.

WHY A DEDICATED RUNNER. "Try different chunk sizes" splits into two very different experiments
and conflating them is exactly the mistake that already cost us a round of numbers:

  MATCHED   clip_len == the model's training num_frames. Train and inference agree, so the
            number is paper-reportable. Getting a 32-frame point therefore requires a model
            TRAINED at 32 (configs/train_hoi4d_32f.yaml), not just a different eval flag.
  MISMATCHED clip_len != training num_frames. This measures out-of-distribution degradation and
            is a legitimate DIAGNOSTIC, but must never be quoted as a window-length result. Our
            own ctxgate sweep APPEARED to show C-abs degrading monotonically
            (29.7/30.2/30.7/32.2) - but that is RETRACTED: it scored 40 segments, not 157,
            so a 0.5 mm spread is noise. The mismatch penalty is UNMEASURED.

So this runner takes an explicit (config, train_frames) per model arm, derives which clip lengths
are matched for it, and stamps every output accordingly. It emits one JSON per cell plus a
summary table that keeps the two classes visually separate.

Stride is always clip_len // 2 (the locked 16/8 ratio generalised), so the overlap fraction is
constant across the sweep and window length is the only thing that varies.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def cells(clip_lens: list[int], train_frames: int) -> list[dict]:
    """Build the sweep cells for one model arm, tagging each as matched or mismatched."""
    out = []
    for cl in clip_lens:
        out.append({
            "clip_len": cl,
            "stride": max(1, cl // 2),
            "matched": cl == train_frames,
            "train_frames": train_frames,
        })
    return out


def run_cell(cell: dict, cfg: str, data_root: str, out_dir: str, segment_len: int,
             wa_short: int, python: str, extra: list[str], dry: bool) -> str | None:
    tag = f"clip{cell['clip_len']}_str{cell['stride']}"
    out_json = os.path.join(out_dir, f"chunksweep_{tag}.json")
    if os.path.exists(out_json):
        print(f"[skip] {tag} already has {out_json}")
        return out_json

    declare = (f"chunk-size sweep cell clip_len={cell['clip_len']}: model trained at "
               f"{cell['train_frames']} frames, so this cell is "
               + ("MATCHED (paper-reportable)" if cell["matched"]
                  else "MISMATCHED (out-of-distribution DIAGNOSTIC ONLY, not a window-length result)"))
    cmd = [python, "-u", "-m", "scripts.eval_world_space",
           "--config", cfg, "--data_root", data_root,
           "--clip_len", str(cell["clip_len"]), "--stride", str(cell["stride"]),
           "--segment_len", str(segment_len), "--wa_short", str(wa_short),
           "--declare_protocol", declare,
           "--out", out_json] + extra
    print(f"\n===== {tag} ({'MATCHED' if cell['matched'] else 'mismatched'}) =====", flush=True)
    print("  " + " ".join(cmd), flush=True)
    if dry:
        return None
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"  !! cell {tag} FAILED rc={rc} (continuing; the summary will show it missing)")
        return None
    return out_json


def summarise(out_dir: str, arm: str) -> None:
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if not (f.startswith("chunksweep_") and f.endswith(".json")):
            continue
        d = json.load(open(os.path.join(out_dir, f)))
        p, a = d.get("protocol", {}), d.get("aggregate", {})
        rows.append((p.get("clip_len"), p.get("stride"), p.get("train_inference_match"),
                     a.get("C_MPJPE_abs"), a.get("C_MPJPE"), a.get("WA_MPJPE_short"),
                     a.get("W_MPJPE"), a.get("n_segments")))
    if not rows:
        print("no sweep cells produced output")
        return
    rows.sort(key=lambda r: (r[0] or 0))
    print("\n" + "=" * 88)
    print(f"CHUNK-SIZE SWEEP  arm={arm}")
    print("=" * 88)
    print(f"{'clip/stride':>12} {'matched':>8} {'C_abs':>8} {'C_rr':>8} {'WA30':>8} "
          f"{'W':>9} {'nseg':>6}")
    for cl, st, m, cabs, crr, wa, w, n in rows:
        f = lambda v: f"{v:8.1f}" if isinstance(v, (int, float)) else f"{'--':>8}"
        print(f"{str(cl)+'/'+str(st):>12} {str(bool(m)):>8} {f(cabs)} {f(crr)} {f(wa)} "
              f"{f(w)[:9]:>9} {str(n):>6}")
    print("\nOnly rows with matched=True are paper-reportable. The others vary window length AND")
    print("train/inference agreement at once, so they cannot attribute a change to window length.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_frames", type=int, required=True,
                    help="data.num_frames the checkpoint was TRAINED at; decides matched cells")
    ap.add_argument("--clip_lens", default="16,32,64")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arm", default="ours")
    ap.add_argument("--segment_len", type=int, default=128)
    ap.add_argument("--wa_short", type=int, default=30)
    ap.add_argument("--max_seqs", type=int, default=200,
                    help="eval_world_space slices [seq_start : seq_start+max_seqs], so 0 yields "
                         "NOTHING rather than everything. Default 200 covers HOI4D-157 and H2O-177.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--matched_only", action="store_true",
                    help="run only cells where clip_len == train_frames")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    clip_lens = [int(x) for x in a.clip_lens.split(",") if x]
    todo = cells(clip_lens, a.train_frames)
    if a.matched_only:
        todo = [c for c in todo if c["matched"]]
    if not todo:
        raise SystemExit(f"no cells to run: train_frames={a.train_frames} is not in {clip_lens}")

    n_matched = sum(c["matched"] for c in todo)
    print(f"sweep arm={a.arm} train_frames={a.train_frames} cells={len(todo)} "
          f"({n_matched} matched, {len(todo)-n_matched} diagnostic)")
    extra = ["--max_seqs", str(a.max_seqs)]
    for c in todo:
        run_cell(c, a.config, a.data_root, a.out_dir, a.segment_len, a.wa_short,
                 a.python, extra, a.dry_run)
    if not a.dry_run:
        summarise(a.out_dir, a.arm)


if __name__ == "__main__":
    main()
