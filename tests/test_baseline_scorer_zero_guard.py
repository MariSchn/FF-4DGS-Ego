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
    m = re.search(r"if n_scored == 0:(.{0,900}?)\n\s*# \.\.\.and zero SEGMENTS", SRC, re.S)
    assert m, "expected a zero-scored guard before the zero-segment guard"
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
    m = re.search(r"if n_scored == 0:(.{0,900}?)\n\s*# \.\.\.and zero SEGMENTS", SRC, re.S)
    assert "<pred_dir>/<seq>.pt" in m.group(1), (
        "the message must state the expected filename convention so a mismatch is diagnosable "
        "without reading the source")


def test_zero_segments_also_fails_even_when_sequences_matched():
    """The hole the first guard left, found the same day it was written.

    Dyn-HaMR (2026-08-08): the converter was pointed at an optimisation working directory that
    contained no final results. It wrote 157 well-formed .pt files whose tensors were 100% NaN and
    whose `valid` mask was all False. The scorer then reported

        BASELINE_WORLD_EVAL n_seqs=157 n_segs=0
          W_MPJPE=nan  WA_short=nan  ...

    and exited 0. The `n_scored == 0` guard did not fire because 157 sequences WERE matched and
    read; they simply produced no scorable window. Matching files is not the same as producing
    results, so both have to be checked.
    """
    m = re.search(r"if not results:(.{0,900}?)\n\s*agg = aggregate", SRC, re.S)
    assert m, "expected a zero-segment guard before aggregate()"
    body = m.group(1)
    assert "SystemExit" in body, "zero segments must exit non-zero"
    assert "ZERO SEGMENTS" in body, "the message must distinguish this from the zero-sequence case"
    assert "valid" in body, (
        "the message must name the concrete check (valid.sum() > 0), because the failure looks "
        "identical to a path bug from the outside")


def test_both_guards_precede_aggregate():
    """Order again: after aggregate() the NaNs exist and the JSON is written regardless."""
    for guard in ("if n_scored == 0:", "if not results:"):
        assert SRC.index(guard) < SRC.index("agg = aggregate(results, seq_c_rows)"), (
            f"{guard} must run before aggregate()")
