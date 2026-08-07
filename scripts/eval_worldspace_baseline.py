"""Shared world-space eval for external baselines (HaWoR, HaPTIC, ...) on HOI4D.

Our own ``eval_world_space.py`` is bound to the WorldMirror forward pass (it runs the
model, solves the scene scale, and chains per-clip predictions). External baselines
instead emit their own per-frame camera- and world-frame joints, so they need a
decoupled scorer that reuses the SAME metric definitions, GT caches, segment chunking
(segment_len frames, ``wa_short`` window) and right-hand convention as our eval — so a
baseline number here is directly comparable to the numbers printed by
``eval_world_space.py``.

Prediction contract — each baseline adapter writes one file per sequence,
``<pred_dir>/<seq>.pt``, a dict of CPU tensors:
    "cam_joints"   : [N, 2, 16, 3] float, metres, CAMERA frame,  smplx-16 order
    "world_joints" : [N, 2, 16, 3] float, metres, WORLD  frame,  smplx-16 order
    "valid"        : [N, 2] bool  — (frame, hand) the baseline actually predicted
Hand index 0 = left, 1 = right (matches the GT caches). Joints the baseline did not
predict must be NaN (and/or marked invalid); they are intersected with the GT valid
mask and dropped. Units are metres in, millimetres out.

Metrics (per 128-frame segment, then averaged over all valid segments):
    C_MPJPE / C_MPJPE_abs : right-hand (RH=1) camera-frame MPJPE, root-rel / absolute
    W_MPJPE               : world MPJPE under the shared first-window RIGID gauge
    WA_MPJPE_short/long   : world MPJPE, per-window / full-segment similarity-aligned

Run:
    python -m scripts.eval_worldspace_baseline --data_root <hoi4d_test> \
        --pred_dir <baseline_preds> --segment_len 128 --wa_short 16 --out bl_world.json
Self-test (no data needed):
    python -m scripts.eval_worldspace_baseline --selftest
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from scripts.world_space_metrics import (
    c_mpjpe,
    w_mpjpe_first_window_aligned,
    wa_mpjpe,
)

RH = 1          # right hand (matches eval_world_space.py)
J = 16          # smplx joints per hand


def load_gt(seq_dir: str):
    """Return (gt_world [N,2,16,3], gt_cam [N,2,16,3], gt_valid [N,2]) or None if uncached."""
    hd = os.path.join(seq_dir, "hand_data")
    fw = os.path.join(hd, "gt_joints_cache_world.pt")
    fc = os.path.join(hd, "gt_joints_cache_cam_v2.pt")
    fb = os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt")
    if not (os.path.exists(fw) and os.path.exists(fc) and os.path.exists(fb)):
        return None
    gt_world = torch.load(fw, map_location="cpu").float()
    gt_cam = torch.load(fc, map_location="cpu").float()
    gt_valid = torch.load(fb, map_location="cpu")["valid"].bool()
    return gt_world, gt_cam, gt_valid


def eval_sequence(seq_dir: str, pred_path: str, segment_len: int, wa_short: int,
                  drop_partial_tail: bool = False, hands: str = "both"):
    """Score one sequence; return (per-segment metric dicts, per-sequence C metrics).

    C-MPJPE is computed once over the WHOLE sequence so aggregation is per-sequence,
    matching eval_hand_cam_anchor.py — segments exist only for the world-gauge metrics,
    which need windowed alignment. Averaging C over segments instead would weight long
    sequences by their segment count and make the table incomparable to ours.

    hands selects which hands enter the WORLD metrics: "both" (default, historical behaviour)
    or "right" (RH only). This matters: C-MPJPE has always been right-hand-only (RH=1) while the
    world metrics summed over BOTH hands, so the two metric families were scored on different
    hand sets. That asymmetry is the leading explanation for our WA-MPJPE sitting at 34.7 against
    Hand3R's 22.54 while W matches to 2.5% - W is dominated by global drift common to both hands,
    whereas WA removes global placement and is therefore dominated by whichever hand is tracked
    worst. Papers split into a right-hand-only cluster (HaWoR self-report 11.27, Hand3R 22.54)
    and a both-hands cluster (StableHand 30.20, ours 34.7).

    drop_partial_tail keeps only whole segment_len windows. eval_world_space (our own online
    row) enumerates floor(n_clips / clips_per_seg) segments and never predicts the ragged tail,
    while this scorer would otherwise score a final partial window for the +SLAM rows. Setting
    the flag on EVERY row makes the segment sets identical, so a table comparison differs only
    in the predictor, not in how many windows each row was scored on."""
    gt = load_gt(seq_dir)
    if gt is None:
        return [], None
    gt_world, gt_cam, gt_valid = gt
    pred = torch.load(pred_path, map_location="cpu")
    pcam = pred["cam_joints"].float()          # [N,2,16,3]
    pworld = pred["world_joints"].float()      # [N,2,16,3]
    pvalid = pred["valid"].bool()              # [N,2]

    n = min(pcam.shape[0], gt_world.shape[0])
    if drop_partial_tail:
        n = (n // segment_len) * segment_len
        if n == 0:
            return [], None
    # (frame, hand) usable only where GT is valid AND the baseline predicted a finite joint set.
    fin_c = torch.isfinite(pcam[:n]).all(-1).all(-1)      # [n,2]
    fin_w = torch.isfinite(pworld[:n]).all(-1).all(-1)    # [n,2]
    valid = gt_valid[:n] & pvalid[:n]                     # [n,2]

    out = []
    n_seg = (n + segment_len - 1) // segment_len
    for seg in range(n_seg):
        s0 = seg * segment_len
        t = min(segment_len, n - s0)
        if t < wa_short:
            continue
        sl = slice(s0, s0 + t)

        # ---- world metrics; hand set is explicit (see `hands`) ----
        if hands == "right":
            gw = gt_world[sl, RH]                                   # [t,16,3]
            pw = pworld[sl, RH]
            vw = (valid[sl, RH] & fin_w[sl, RH]).unsqueeze(-1).expand(-1, J)
        else:
            gw = gt_world[sl].reshape(t, 2 * J, 3)                  # [t,32,3]
            pw = pworld[sl].reshape(t, 2 * J, 3)
            vw = (valid[sl] & fin_w[sl]).repeat_interleave(J, dim=1)
        if int(vw.sum()) >= 3:
            w = w_mpjpe_first_window_aligned(pw, gw, vw, wa_short)
            wa_s = wa_mpjpe(pw, gw, window=wa_short, valid=vw)
            wa_l = wa_mpjpe(pw, gw, window=t, valid=vw)
        else:
            w = wa_s = wa_l = float("nan")

        # ---- camera-frame C-MPJPE, right hand only (RH=1) ----
        gc = gt_cam[sl, RH]                                # [t,16,3]
        pc = pcam[sl, RH]                                  # [t,16,3]
        vc = (valid[sl, RH] & fin_c[sl, RH]).unsqueeze(-1).expand(-1, J)   # [t,16]
        if int(vc.sum()) >= 1:
            c_rr = c_mpjpe(pc, gc, valid=vc, root_relative=True)
            c_ab = c_mpjpe(pc, gc, valid=vc, root_relative=False)
        else:
            c_rr = c_ab = float("nan")

        out.append({"seq": os.path.basename(seq_dir), "seg": seg, "frames": t,
                    "W_MPJPE": w, "WA_MPJPE_short": wa_s, "WA_MPJPE_long": wa_l,
                    "C_MPJPE": c_rr, "C_MPJPE_abs": c_ab})

    # ---- sequence-level C metrics (the canonical aggregation unit) ----
    gc = gt_cam[:n, RH]
    pc = pcam[:n, RH]
    vc = (valid[:n, RH] & fin_c[:n, RH]).unsqueeze(-1).expand(-1, J)
    if int(vc.sum()) >= 1:
        seq_c = {"seq": os.path.basename(seq_dir), "frames": n,
                 "C_MPJPE": c_mpjpe(pc, gc, valid=vc, root_relative=True),
                 "C_MPJPE_abs": c_mpjpe(pc, gc, valid=vc, root_relative=False)}
    else:
        seq_c = None
    return out, seq_c


def aggregate(results, seq_c_rows=None):
    """W metrics: mean over segments with a finite W_MPJPE (the world gauge needs windows).
    C metrics: mean over SEQUENCES (one value per sequence), matching how ours is scored in
    eval_hand_cam_anchor.py. Falls back to per-segment C only if no seq rows were passed."""
    valid = [r for r in results if r["W_MPJPE"] == r["W_MPJPE"]]
    keys = ("W_MPJPE", "WA_MPJPE_short", "WA_MPJPE_long")

    def _mean(k, rows):
        vals = [r[k] for r in rows if k in r and r[k] == r[k]]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    agg = {k: _mean(k, valid) for k in keys}
    c_rows = [r for r in (seq_c_rows or []) if r is not None] or results
    agg["C_MPJPE"] = _mean("C_MPJPE", c_rows)
    agg["C_MPJPE_abs"] = _mean("C_MPJPE_abs", c_rows)
    agg["n_segments"] = len(valid)
    agg["n_seqs_c"] = len(c_rows) if c_rows is not results else 0
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", help="HOI4D test dir (sequences with hand_data caches)")
    ap.add_argument("--pred_dir", help="dir of <seq>.pt baseline predictions")
    ap.add_argument("--segment_len", type=int, default=128)
    ap.add_argument("--wa_short", type=int, default=16)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--out", default="baseline_world_eval.json")
    ap.add_argument("--hands", choices=["both", "right"], default="both",
                    help="which hands enter the WORLD metrics. C-MPJPE is always right-hand-only, "
                         "so --hands right makes both metric families use the same hand set and "
                         "matches the Hand3R / HaWoR convention.")
    ap.add_argument("--drop_partial_tail", action="store_true",
                    help="score only whole segment_len windows, so this row's segment set matches "
                         "eval_world_space's (which never predicts the ragged tail)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not (args.data_root and args.pred_dir):
        ap.error("--data_root and --pred_dir are required (or use --selftest)")

    seqs = sorted(d for d in os.listdir(args.data_root)
                  if os.path.isdir(os.path.join(args.data_root, d)))
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    results, seq_c_rows, n_scored = [], [], 0
    for sq in seqs:
        pp = os.path.join(args.pred_dir, sq + ".pt")
        if not os.path.exists(pp):
            continue
        rows, seq_c = eval_sequence(os.path.join(args.data_root, sq), pp,
                                    args.segment_len, args.wa_short,
                                    drop_partial_tail=args.drop_partial_tail,
                                    hands=args.hands)
        results += rows
        if seq_c is not None:
            seq_c_rows.append(seq_c)
        if rows:
            n_scored += 1
    # SCORING NOTHING IS A FAILURE, NOT AN EMPTY RESULT. The per-sequence loop skips silently when
    # <pred_dir>/<seq>.pt is absent, so a wrong --pred_dir scores ZERO sequences, writes a
    # well-formed JSON full of NaN, and exits 0. That happened on 2026-08-07 (job 104440): four of
    # five baselines were pointed at /work/scratch when their predictions live in /home, and the
    # job reported LW100_ALL_DONE with a table of nan. A non-empty output FILE is not evidence of a
    # non-empty RESULT, which is why the caller's `[ -s out.json ]` check did not catch it either.
    if not os.path.isdir(args.pred_dir):
        raise SystemExit(f"\n!! --pred_dir does not exist: {args.pred_dir}\n"
                         f"   Nothing would be scored and the output would be all-NaN. Exiting "
                         f"non-zero so a batch script cannot report success.")
    if n_scored == 0:
        n_avail = len([f for f in os.listdir(args.pred_dir) if f.endswith(".pt")])
        raise SystemExit(
            f"\n!! SCORED ZERO SEQUENCES from {args.pred_dir}\n"
            f"   {len(seqs)} sequences in {args.data_root}, {n_avail} .pt files in the pred dir, "
            f"and NOT ONE matched.\n"
            f"   Expected <pred_dir>/<seq>.pt with <seq> exactly a data_root subdirectory name.\n"
            f"   This is a FAILURE, not an empty result: continuing would write all-NaN metrics "
            f"that a downstream table renders as real numbers.")
    agg = aggregate(results, seq_c_rows)
    # Stamp the protocol so the artifact records its own hand set and window. The hand set in
    # particular was previously implicit (world = both hands, C = right only) and that asymmetry
    # went unnoticed for a long time.
    # pred_dir alone is only a NAME. It cannot say which box store the crops came from or which
    # trajectory was composed in, and those are exactly the two axes that produced three separate
    # defects on 2026-08-06 (tasks #65/#67/#68). Copy the producer's own record in, and shout when
    # it is absent or marks a GT-oracle trajectory.
    from scripts.pred_provenance import describe_or_warn
    protocol = {"segment_len": args.segment_len, "wa_short": args.wa_short,
                "hands": args.hands, "drop_partial_tail": bool(args.drop_partial_tail),
                "pred_dir": args.pred_dir, "data_root": args.data_root,
                "pred_provenance": describe_or_warn(args.pred_dir)}
    json.dump({"protocol": protocol, "aggregate": agg, "per_segment": results,
               "per_seq_c": seq_c_rows},
              open(args.out, "w"), indent=2)
    print(f"BASELINE_WORLD_EVAL n_seqs={n_scored} n_segs={agg['n_segments']}")
    print(f"  W_MPJPE={agg['W_MPJPE']:.1f}  WA_short={agg['WA_MPJPE_short']:.1f}  "
          f"WA_long={agg['WA_MPJPE_long']:.1f}  C_rr={agg['C_MPJPE']:.1f}  "
          f"C_abs={agg['C_MPJPE_abs']:.1f}  -> {args.out}")


def _selftest():
    """No data needed. GT-as-prediction must score ~0; a fixed world shift must leave WA~0
    (similarity absorbs it) but inflate W; a camera-frame shift must inflate C_abs but not C_rr."""
    torch.manual_seed(0)
    n = 160
    gt_world = torch.cumsum(torch.randn(n, 2, J, 3) * 0.01, dim=0)
    gt_cam = torch.cumsum(torch.randn(n, 2, J, 3) * 0.01, dim=0)
    valid = torch.ones(n, 2, dtype=torch.bool)
    tmpdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "wsbl_selftest")
    os.makedirs(os.path.join(tmpdir, "seqA", "hand_data"), exist_ok=True)
    hd = os.path.join(tmpdir, "seqA", "hand_data")
    torch.save(gt_world, os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(gt_cam, os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save({"valid": valid}, os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt"))

    # 1) perfect prediction -> all metrics ~0
    torch.save({"cam_joints": gt_cam.clone(), "world_joints": gt_world.clone(),
                "valid": valid.clone()}, os.path.join(tmpdir, "seqA.pt"))
    r, sc = eval_sequence(os.path.join(tmpdir, "seqA"), os.path.join(tmpdir, "seqA.pt"), 128, 16)
    a = aggregate(r, [sc])
    assert a["W_MPJPE"] < 1e-2 and a["C_MPJPE_abs"] < 1e-2, f"perfect pred not ~0: {a}"
    assert a["n_seqs_c"] == 1, f"seq-level C row missing: {a}"

    # 2) constant WORLD shift: WA (similarity) absorbs it, W (rigid, absolute) sees it
    shifted = gt_world + torch.tensor([0.10, 0.0, 0.0])
    torch.save({"cam_joints": gt_cam.clone(), "world_joints": shifted,
                "valid": valid.clone()}, os.path.join(tmpdir, "seqA.pt"))
    r2, sc2 = eval_sequence(os.path.join(tmpdir, "seqA"), os.path.join(tmpdir, "seqA.pt"), 128, 16)
    a2 = aggregate(r2, [sc2])
    assert a2["WA_MPJPE_long"] < 1.0, f"WA should absorb a rigid world shift: {a2['WA_MPJPE_long']}"

    # 3) constant CAMERA shift: C_abs sees it, C_rr cancels it
    cshift = gt_cam + torch.tensor([0.0, 0.0, 0.08])
    torch.save({"cam_joints": cshift, "world_joints": gt_world.clone(),
                "valid": valid.clone()}, os.path.join(tmpdir, "seqA.pt"))
    r3, sc3 = eval_sequence(os.path.join(tmpdir, "seqA"), os.path.join(tmpdir, "seqA.pt"), 128, 16)
    a3 = aggregate(r3, [sc3])
    assert a3["C_MPJPE"] < 1e-2 and a3["C_MPJPE_abs"] > 60.0, \
        f"C_rr must cancel / C_abs must see a camera shift: {a3}"
    print("eval_worldspace_baseline self-test: OK "
          f"(perfect W={a['W_MPJPE']:.3f} C_abs={a['C_MPJPE_abs']:.3f}; "
          f"world-shift WA_long={a2['WA_MPJPE_long']:.3f} W={a2['W_MPJPE']:.1f}; "
          f"cam-shift C_rr={a3['C_MPJPE']:.3f} C_abs={a3['C_MPJPE_abs']:.1f})")


if __name__ == "__main__":
    main()
