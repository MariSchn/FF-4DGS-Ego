"""Scoring zero sequences is a FAILURE, not an empty result.

WHAT THIS CAUGHT (2026-08-07, job 104440). The long-window rebuild at 100 frames pointed four of
five baselines at /work/scratch when their predictions actually live in /home. The per-sequence
loop skips silently when `<pred_dir>/<seq>.pt` is absent:

    pp = os.path.join(args.pred_dir, sq + ".pt")
    if not os.path.exists(pp):
        continue

so every sequence was skipped, `aggregate` was handed an empty list, and the script wrote a
well-formed JSON full of NaN and exited 0. The job printed `LW100_ALL_DONE` above a table reading

    Ours (feedforward)   0   nan   nan   nan   nan
    HaWoR                0   nan   nan   nan   nan

The calling sbatch checked `[ -s out.json ]`, which passed: a JSON with zero segments is still a
non-empty FILE. A non-empty file is not evidence of a non-empty result.

This is the fourth instance today of the same shape - a run that reports success while producing
nothing (the others: zguard's missing checkpoint, the hv arm's UnboundLocalError, and this). The
common fix is the same: make the empty case raise rather than serialise.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "scripts" / "eval_worldspace_baseline.py").read_text()


def test_missing_pred_dir_exits_nonzero():
    assert "if not os.path.isdir(args.pred_dir):" in SRC, (
        "a --pred_dir that does not exist must be caught before scoring, not silently produce "
        "an all-NaN result")
    m = re.search(r"if not os\.path\.isdir\(args\.pred_dir\):(.{0,400}?)\n\s*if n_scored == 0:",
                  SRC, re.S)
    assert m and "SystemExit" in m.group(1), "the missing-dir branch must raise SystemExit"


def test_zero_scored_sequences_exits_nonzero():
    m = re.search(r"if n_scored == 0:(.{0,700}?)\n\s*agg = aggregate", SRC, re.S)
    assert m, "expected a zero-scored guard immediately before aggregation"
    body = m.group(1)
    assert "SystemExit" in body, "scoring nothing must exit non-zero"
    assert "NOT ONE matched" in body or "not one matched" in body.lower(), (
        "the message must say that sequences and predictions both existed but did not match, "
        "because that is the actual failure mode (a path or naming mismatch), not 'no data'")


def test_guard_runs_before_aggregate():
    """Order matters: aggregate() on an empty list is what manufactures the NaNs."""
    guard = SRC.index("if n_scored == 0:")
    agg = SRC.index("agg = aggregate(results, seq_c_rows)")
    assert guard < agg, (
        "the zero-scored guard must precede aggregate(); after it, the NaNs already exist and "
        "the JSON is written regardless")


def test_message_explains_the_expected_naming():
    """A path bug is only actionable if the message says what the expected layout is."""
    m = re.search(r"if n_scored == 0:(.{0,700}?)\n\s*agg = aggregate", SRC, re.S)
    assert "<pred_dir>/<seq>.pt" in m.group(1), (
        "the message must state the expected filename convention so a mismatch is diagnosable "
        "without reading the source")
