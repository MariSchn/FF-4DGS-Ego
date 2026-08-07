"""A missing warm-start checkpoint must stop the run, not produce an empty "COMPLETED" job.

WHAT THIS CAUGHT (2026-08-07). zguard job 104380 ran all four arms of the task #63 2x2. Every arm
died with `FileNotFoundError: /home/dmonopoli/ckpt_backup/jitterrob10ep_best.pt`, because the local
copy had been deleted after the HuggingFace upload verified. sacct nonetheless reported the job
**COMPLETED**: the per-arm failures were swallowed by the arm wrapper, the readout script ran
afterwards, printed "MISSING" on all four rows, and exited 0.

That is the same shape as two defects already in this repo's history: the enable_gs eval trap
(exit 0, silently non-metric W/WA) and the C-abs-725 untrained checkpoint. The pattern is a run
that reports success while producing nothing, and it is expensive precisely because nothing draws
attention to it.

Two properties are asserted here:
  1. the failure is a SystemExit (non-zero) rather than a bare FileNotFoundError, so a batch script
     cannot continue past it into a readout that prints MISSING and exits 0;
  2. the message names the recovery path, because "file not found" reads as "the checkpoint is
     lost" when in fact it is one command away in the private HF repo.
"""
from __future__ import annotations

import pytest

from scripts.eval_world_space import _CKPT_BACKUP_REPO, _require_checkpoint_present


def test_missing_checkpoint_raises_systemexit(tmp_path):
    missing = str(tmp_path / "jitterrob10ep_best.pt")
    with pytest.raises(SystemExit) as ei:
        _require_checkpoint_present(missing)
    # SystemExit with a string message exits non-zero, which is the property that matters: the
    # calling sbatch arm must not be able to fall through to its readout.
    assert ei.value.code != 0
    assert not isinstance(ei.value.code, int) or ei.value.code != 0


def test_message_names_the_file_and_the_recovery_route(tmp_path):
    missing = str(tmp_path / "jitterrob10ep_best.pt")
    with pytest.raises(SystemExit) as ei:
        _require_checkpoint_present(missing)
    msg = str(ei.value.code)
    assert missing in msg, "the message must name the exact path that is absent"
    assert _CKPT_BACKUP_REPO in msg, "the message must name where the file can be restored from"
    assert "RECOVERABLE" in msg
    assert "Do NOT retrain" in msg, (
        "the headline head cannot be reproduced bit-for-bit; the message must say so, or the "
        "obvious response to a missing checkpoint is to retrain and quietly change the numbers")


def test_present_checkpoint_is_a_no_op(tmp_path):
    """NEGATIVE CONTROL: a guard that always raised would satisfy the tests above."""
    p = tmp_path / "present.pt"
    p.write_bytes(b"not a real checkpoint, but it exists")
    _require_checkpoint_present(str(p))          # must not raise
