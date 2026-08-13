#!/usr/bin/env python3
"""Emit the world-space comparison table in Hand3R's Table II layout, from result JSONs only.

Hand3R reports one table: C-MPJPE, then Short Video and Long Video side by side, each carrying
WA-MPJPE and W-MPJPE. We had two separate tables at two segment lengths, which made the pair hard
to read against theirs and hid the fact that the two used different camera conventions for months.
This script builds the single table from the artefacts, so a number can only appear in the paper if
a scored JSON contains it.

WHICH CELL COMES FROM WHERE, because the names collide and the wrong pick is invisible:

  C-MPJPE     C_MPJPE_abs, absolute camera frame, no alignment of any kind. Aggregated per
              SEQUENCE, so it is identical in the 30- and 100-frame runs and we read it from the
              long one.
  Short WA    WA_MPJPE_long from the segment_len=30 run. "long" here means the alignment spans the
              whole segment, which at segment_len=30 is a 30-frame chunk. That is Hand3R's short
              WA. It is NOT WA_MPJPE_short from the 100-frame run, which is a 30-frame window
              nested inside a 100-frame segment and is a different population of windows.
  Short W     W_MPJPE_h3r from the segment_len=30 run, never W_MPJPE. At 30 frames our own gauge
              solves its transform on all 30, so the segment is rigidly fitted and the metric stops
              measuring drift. Hand3R's two-frame fit does not degenerate.
  Long WA     WA_MPJPE_long from the segment_len=100 run.
  Long W      W_MPJPE_h3r from the segment_len=100 run.

The script refuses to emit a table whose rows were scored under different protocols. Every
regression this project has had in its comparison tables was of that shape: a shared scorer with
one row on different flags, a camera convention fixed in one table and not the other, a detector
swapped under one method. A table assembled by hand cannot catch those; this one stops.

    python -m scripts.make_world_table --results_dir <dir> --out tab_world.tex
"""
from __future__ import annotations

import argparse
import json
import os

# (key, display label, regime, pipeline, metric-depth?). The order is the table's order.
# metric_depth=False means the row cannot recover absolute placement under input matching, so its
# absolute and global cells report the scale of the inputs we supply rather than the method's own.
ROWS = [
    ("hamer",   r"HaMeR $+$ SLAM",  "offline", "multi-stage", True),
    ("wilor",   r"WiLoR $+$ SLAM",  "offline", "multi-stage", True),
    ("hawor",   r"HaWoR",           "offline", "multi-stage", True),
    ("haptic",  r"HaPTIC $+$ SLAM", "offline", "multi-stage", False),
    ("dynhamr", r"Dyn-HaMR",        "offline", "optimisation", False),
    ("ours",    r"Ours",            "online",  "one-stage",   True),
]

# Flags that must agree across every row, or the table is not a comparison. pred_dir and
# data_root are deliberately absent: those differ per row by design.
PROTOCOL_KEYS = ("segment_len", "wa_short", "hands", "drop_partial_tail",
                 "w_h3r_align_frames", "joints_per_hand")


def load(results_dir: str, key: str, seg: int):
    path = os.path.join(results_dir, f"h3r_{key}_seg{seg}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def check_protocols(loaded: dict) -> None:
    """Every row at a given segment length must share PROTOCOL_KEYS. Raise loudly otherwise."""
    for seg in (30, 100):
        ref_key, ref = None, None
        for key in [r[0] for r in ROWS]:
            d = loaded.get((key, seg))
            if d is None:
                continue
            proto = {k: d["protocol"].get(k) for k in PROTOCOL_KEYS}
            if ref is None:
                ref_key, ref = key, proto
                continue
            if proto != ref:
                diff = {k: (ref[k], proto[k]) for k in PROTOCOL_KEYS if ref[k] != proto[k]}
                raise SystemExit(
                    f"REFUSING TO EMIT: at segment_len={seg}, row '{key}' was scored under a "
                    f"different protocol than '{ref_key}'. Differences (ref, this): {diff}. "
                    f"Re-score the odd row rather than printing the table.")
        if ref is not None and ref["segment_len"] != seg:
            raise SystemExit(f"REFUSING TO EMIT: files named seg{seg} carry "
                             f"segment_len={ref['segment_len']}.")


def cell(value, ok: bool, best: bool) -> str:
    if value is None or value != value:
        return r"\na"
    s = f"{value:.1f}"
    if not ok:
        return f"({s})$^{{\\ddagger}}$"
    return rf"\best{{{s}}}" if best else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default="tab_world_hand3r_format.tex")
    ap.add_argument("--our_gauge", action="store_true",
                    help="append two columns carrying OUR W gauge next to Hand3R's, so the cost of "
                         "the gauge is visible per row instead of asserted in prose")
    a = ap.parse_args()

    loaded = {}
    for key, *_ in ROWS:
        for seg in (30, 100):
            d = load(a.results_dir, key, seg)
            if d is not None:
                loaded[(key, seg)] = d
    if not loaded:
        raise SystemExit(f"no h3r_*_seg*.json under {a.results_dir}")
    check_protocols(loaded)

    # Gather, then decide the winners per column, so \best marks a measured lead rather than a
    # remembered one. Rows without metric depth are excluded from the absolute and global columns:
    # they would otherwise "win" a column they cannot compete in.
    data = {}
    for key, label, regime, pipeline, metric in ROWS:
        d30, d100 = loaded.get((key, 30)), loaded.get((key, 100))
        a30 = (d30 or {}).get("aggregate", {})
        a100 = (d100 or {}).get("aggregate", {})
        data[key] = {
            "label": label, "regime": regime, "pipeline": pipeline, "metric": metric,
            "c_abs": a100.get("C_MPJPE_abs", a30.get("C_MPJPE_abs")),
            "s_wa": a30.get("WA_MPJPE_long"), "s_w": a30.get("W_MPJPE_h3r"),
            "l_wa": a100.get("WA_MPJPE_long"), "l_w": a100.get("W_MPJPE_h3r"),
            "s_w_ours": a30.get("W_MPJPE"), "l_w_ours": a100.get("W_MPJPE"),
            "n30": a30.get("n_segments"), "n100": a100.get("n_segments"),
        }

    def winner(col: str, needs_metric: bool) -> str | None:
        cands = {k: v[col] for k, v in data.items()
                 if v[col] is not None and v[col] == v[col] and (v["metric"] or not needs_metric)}
        return min(cands, key=cands.get) if cands else None

    wins = {c: winner(c, c in ("c_abs", "s_w", "l_w")) for c in ("c_abs", "s_wa", "s_w", "l_wa", "l_w")}

    extra_h = r" & \multicolumn{2}{c}{W-MPJPE, our gauge $\downarrow$}" if a.our_gauge else ""
    ncol = 8 + (2 if a.our_gauge else 0)
    lines = [
        r"\begin{tabular}{@{}lllccccc" + ("cc" if a.our_gauge else "") + r"@{}}",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{Short video (30)} & \multicolumn{2}{c}{Long video (100)}"
        + extra_h + r" \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
        + (r"\cmidrule(l){9-10}" if a.our_gauge else ""),
        r"Method & Type & Pipeline & C-MPJPE $\downarrow$ & WA $\downarrow$ & W $\downarrow$ "
        r"& WA $\downarrow$ & W $\downarrow$"
        + (r" & short & long" if a.our_gauge else "") + r" \\",
        r"\midrule",
    ]
    for key, *_ in ROWS:
        v = data[key]
        row = [v["label"], v["regime"], v["pipeline"],
               cell(v["c_abs"], v["metric"], wins["c_abs"] == key),
               cell(v["s_wa"], True, wins["s_wa"] == key),
               cell(v["s_w"], v["metric"], wins["s_w"] == key),
               cell(v["l_wa"], True, wins["l_wa"] == key),
               cell(v["l_w"], v["metric"], wins["l_w"] == key)]
        if a.our_gauge:
            row += [cell(v["s_w_ours"], v["metric"], False), cell(v["l_w_ours"], v["metric"], False)]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {a.out}")
    print("\nsegment counts (they differ by method because valid frames do, not because the "
          "protocol does):")
    for key, *_ in ROWS:
        v = data[key]
        print(f"  {v['label']:20s} n30={v['n30']}  n100={v['n100']}")
    if a.our_gauge:
        print("\ngauge cost, Hand3R's two-frame fit minus ours, in mm:")
        for key, *_ in ROWS:
            v = data[key]
            for span, h, o in (("short", v["s_w"], v["s_w_ours"]), ("long", v["l_w"], v["l_w_ours"])):
                if h is not None and o is not None and h == h and o == o:
                    print(f"  {v['label']:20s} {span:5s} {h - o:+7.1f}")


if __name__ == "__main__":
    main()
