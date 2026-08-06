"""Provenance must be machine-checkable, because the distinction it encodes scores BETTER.

The failure this guards against (2026-08-06, task #67): a prediction directory whose world track
was lifted with GROUND-TRUTH extrinsics is indistinguishable, by name, from one composed with
DROID-SLAM. The oracle scores better (27.1/23.3 vs 42.5/35.7 for the same hands), so quoting the
wrong directory silently flatters a baseline, and one had already been wired into a seg100 table
under a "+SLAM" label.

Free-text trajectory labels would not fix that: a scorer has to be able to TEST for the oracle.
"""
from __future__ import annotations

import json

import pytest

from scripts.pred_provenance import (
    PROVENANCE_FILENAME,
    TRAJ_GT_ORACLE,
    TRAJ_SLAM,
    describe_or_warn,
    read_provenance,
    write_provenance,
)


def test_round_trip_records_box_and_trajectory(tmp_path):
    d = tmp_path / "some_preds"
    write_provenance(str(d), box_source="/scratch/hoi4d_detboxes_v3",
                     trajectory_source=TRAJ_SLAM, produced_by="test", n_seqs=157)

    rec = read_provenance(str(d))
    assert rec["box_source"] == "/scratch/hoi4d_detboxes_v3"
    assert rec["trajectory_source"] == TRAJ_SLAM
    assert rec["n_seqs"] == 157
    assert (d / PROVENANCE_FILENAME).exists()


def test_free_text_trajectory_is_rejected(tmp_path):
    """The oracle-vs-SLAM distinction must be a closed set, not prose.

    If arbitrary strings were allowed, "slam-ish" or "gt traj" would pass and no scorer could
    reliably detect the oracle case, which is the whole point of the record.
    """
    with pytest.raises(ValueError, match="TRAJ_"):
        write_provenance(str(tmp_path / "p"), box_source="x",
                         trajectory_source="composed with slam probably",
                         produced_by="test")


def test_missing_provenance_reports_unknown_not_silence(tmp_path, capsys):
    """An unrecorded directory must produce an explicit UNKNOWN, never an omitted key.

    A downstream table must not be able to mistake "we did not record it" for "we checked it".
    """
    out = describe_or_warn(str(tmp_path / "never_written"))
    assert out["box_source"] == "UNKNOWN"
    assert out["trajectory_source"] == "UNKNOWN"
    assert "no _provenance.json" in out["note"]
    assert "UNKNOWN" in capsys.readouterr().out


def test_oracle_directory_shouts_when_described(tmp_path, capsys):
    """Describing a GT-oracle dir must print a warning a human will notice in a job log."""
    d = tmp_path / "haptic_detbox_preds"
    write_provenance(str(d), box_source="/scratch/hoi4d_detboxes_v3",
                     trajectory_source=TRAJ_GT_ORACLE, produced_by="test", n_seqs=157)
    capsys.readouterr()                      # drop the write-time line

    out = describe_or_warn(str(d))
    assert out["trajectory_source"] == TRAJ_GT_ORACLE
    printed = capsys.readouterr().out
    assert "GROUND-TRUTH" in printed and "ORACLE" in printed


def test_corrupt_provenance_is_not_fatal(tmp_path):
    """A truncated record must degrade to UNKNOWN rather than crash a scoring run."""
    d = tmp_path / "p"
    d.mkdir()
    (d / PROVENANCE_FILENAME).write_text("{not json")
    assert read_provenance(str(d)) is None
    assert describe_or_warn(str(d))["trajectory_source"] == "UNKNOWN"


def test_record_is_json_serialisable_for_embedding(tmp_path):
    """The scorer embeds this dict straight into its output JSON, so it must serialise."""
    d = tmp_path / "p"
    write_provenance(str(d), box_source="/scratch/boxes_v3", trajectory_source=TRAJ_SLAM,
                     produced_by="test", n_seqs=3, cam_pred_dir="/a", slam_pred_dir="/b")
    json.dumps(describe_or_warn(str(d)))     # must not raise
