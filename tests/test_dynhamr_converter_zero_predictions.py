"""Writing a file per sequence is not the same as having a prediction per sequence.

WHAT THIS CAUGHT (2026-07-22, rediscovered 2026-08-09). `dynhamr_eval_run/dynhamr_output/` was
EMPTY because Dyn-HaMR's demo.py had failed, yet the pipeline reported success end to end:

    [Run] python .../demo.py --data_dir ... --out_dir ... 2>/dev/null || true
    Converting Dyn-HaMR predictions across 157 sequences...
    Converted 157 sequence predictions -> .../eval_preds
    BASELINE_WORLD_EVAL n_seqs=157 n_segs=0
      W_MPJPE=nan  WA_short=nan  WA_long=nan  C_rr=nan  C_abs=nan
    === Dyn-HaMR Evaluation Complete ===

Three separate defects lined up to make an empty run look finished:

  1. `2>/dev/null || true` on the demo.py call discarded stderr AND forced exit 0, so the
     pipeline's own `run_cmd` failure check could never fire.
  2. `convert_seq` returned True whenever the GROUND-TRUTH cache existed, regardless of whether a
     prediction was found, so 157 all-NaN files were written and counted as conversions.
  3. The scorer then aggregated an empty segment list into well-formed NaNs.

Defect 3 was fixed by test_baseline_scorer_zero_guard.py. This file covers 1 and 2. The pattern is
the recurring one in this codebase: a stage that cannot distinguish "produced nothing" from
"produced nothing wrong".
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONV = (ROOT / "scripts" / "dynhamr_to_worldeval.py").read_text()
DRIVER = (ROOT / "scripts" / "eval_dynhamr_baseline.py").read_text()


def test_convert_seq_reports_whether_a_prediction_existed():
    """The return value has to carry the distinction, or the caller cannot count it."""
    assert "return False, False" in CONV and "return True, had_prediction" in CONV, (
        "convert_seq must return (wrote_file, had_prediction); a single bool cannot distinguish "
        "'wrote an all-NaN placeholder' from 'converted a real prediction'")
    assert "had_prediction = os.path.exists(pred_path)" in CONV


def test_zero_predictions_exits_nonzero():
    m = re.search(r"if with_pred == 0:(.{0,900}?)\n\s*print\(", CONV, re.S)
    assert m, "expected a zero-prediction guard before the success print"
    body = m.group(1)
    assert "SystemExit" in body, "finding no predictions at all must exit non-zero"
    assert "n_segs=0" in body or "all-NaN" in body, (
        "the message must name the downstream symptom, because from the outside the failure looks "
        "like a scorer bug rather than a missing input")


def test_missing_output_dir_is_caught_before_writing_anything():
    assert "if not os.path.isdir(args.dynhamr_out):" in CONV, (
        "a --dynhamr_out that does not exist must fail immediately, not after writing 157 "
        "placeholder files")


def test_partial_coverage_warns():
    """Some predictions is not all predictions, and the gap changes the segment count."""
    assert "if with_pred < count:" in CONV and "WARNING" in CONV, (
        "sequences without a prediction are silently unscorable, so their count must be printed")


def test_driver_does_not_swallow_dynhamr_failure():
    """Check the COMMAND, not the file: the offending string legitimately appears in comments
    explaining why it was removed, so a whole-file substring search would fail on its own fix."""
    cmds = [ln for ln in DRIVER.splitlines()
            if "run_cmd(" in ln and "demo.py" in ln]
    assert cmds, "could not find the demo.py invocation"
    for ln in cmds:
        assert "|| true" not in ln, (
            "`|| true` forces exit 0, defeating run_cmd's failure check; this is what let an "
            f"empty Dyn-HaMR run be reported as COMPLETED. Offending line: {ln.strip()}")
        assert "2>/dev/null" not in ln, (
            f"discarding stderr hides why Dyn-HaMR produced nothing. Offending line: {ln.strip()}")


def test_driver_checks_the_output_directory_is_nonempty():
    m = re.search(r"n_out = len\(\[(.{0,200}?)\]\)(.{0,400}?)sys\.exit", DRIVER, re.S)
    assert m, "the driver must count Dyn-HaMR's output files and exit if there are none"
    assert "endswith" in m.group(1), "count actual prediction files, not directory entries"
