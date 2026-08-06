"""Provenance that travels WITH a prediction directory.

WHY THIS EXISTS. On 2026-08-06 three separate box-source defects were found in one day, and all
three had the same root cause: the box source and the trajectory source of a prediction directory
were encoded only in its NAME.

  1. The world table's WA column mixed native-detector and shared-box numbers, because
     `wilor_slam_preds` (native) and `wilor_slam_detbox_preds` (shared) differ by one infix and
     WA is alignment-invariant, so nothing looked wrong (task #65).
  2. Every `<method>_detbox_preds` dir turned out to carry a world track lifted with GROUND-TRUTH
     extrinsics, while `<tag>_slam_detbox_preds` carries the real DROID-SLAM composition. The
     oracle dirs score BETTER, so quoting the wrong one silently flatters a baseline. One of them
     had already been wired into a seg100 table under a SLAM label (task #67).
  3. The same fine-tune scored C_abs 23.41 under `hamer_fj2_v3box_preds` and 38.55 under
     `hamer_fj2_detbox_preds`, and by the time anyone asked which boxes each used, Euler's scratch
     purge had reduced the first to 1 file of 157, making the number in the abstract
     unreproducible from disk (task #68).

A directory name is not provenance. It cannot record which box store was used, which trajectory
was composed in, or which script wrote it, and it does not survive being copied or renamed. This
module writes that as data, next to the predictions, so a scorer can copy it into its own output
and a human can answer "what is this?" without archaeology.

Contract: `<pred_dir>/_provenance.json`. Absent is not fatal (many dirs predate this), but a
scorer should say so loudly rather than report a number as if its inputs were known.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

PROVENANCE_FILENAME = "_provenance.json"

# Trajectory sources, spelled out because the distinction is the one that bit us. A row built with
# GT_EXTRINSICS is an ORACLE and must never sit in a table next to SLAM-composed rows.
TRAJ_GT_ORACLE = "GT_EXTRINSICS_ORACLE"
TRAJ_SLAM = "DROID_SLAM_COMPOSED"
TRAJ_PREDICTED = "PREDICTED_BY_MODEL"
TRAJ_NONE = "NONE_CAMERA_FRAME_ONLY"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def write_provenance(pred_dir: str, *, box_source: str, trajectory_source: str,
                     produced_by: str, n_seqs: int | None = None, **extra) -> str:
    """Record what made the predictions in ``pred_dir``.

    Args:
        box_source: the detection boxes the crops came from. Use the STORE PATH when there is one
            (e.g. ``/cluster/scratch/dmonopoli/hoi4d_detboxes_v3``), not a nickname like "detbox":
            "detbox" is exactly the ambiguous label that made task #68 unresolvable.
        trajectory_source: one of the TRAJ_* constants. Required even for camera-frame-only dirs,
            because "this dir has no trajectory" is itself the fact a scorer needs.
        produced_by: the script that wrote the directory.
        n_seqs: how many sequences were written, so a later partial purge is detectable.
    """
    if trajectory_source not in (TRAJ_GT_ORACLE, TRAJ_SLAM, TRAJ_PREDICTED, TRAJ_NONE):
        raise ValueError(
            f"trajectory_source must be one of the TRAJ_* constants, got {trajectory_source!r}. "
            "Free text defeats the point: the GT-oracle vs SLAM distinction has to be machine "
            "checkable, since the oracle scores better and reads as a normal row."
        )
    rec = {
        "box_source": box_source,
        "trajectory_source": trajectory_source,
        "produced_by": produced_by,
        "n_seqs": n_seqs,
        "argv": " ".join(sys.argv),
        "git_commit": _git_commit(),
        **extra,
    }
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, PROVENANCE_FILENAME)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[provenance] {path}: boxes={box_source} traj={trajectory_source}", flush=True)
    return path


def read_provenance(pred_dir: str) -> dict | None:
    """Return the provenance record for ``pred_dir``, or None if it predates this module."""
    path = os.path.join(pred_dir, PROVENANCE_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def describe_or_warn(pred_dir: str) -> dict:
    """Provenance for a scorer to embed in its output, warning loudly when it is unknown.

    Returns a dict that is always safe to serialise. When the record is missing it says so
    explicitly rather than omitting the key, so a downstream table cannot mistake "not recorded"
    for "recorded as fine".
    """
    rec = read_provenance(pred_dir)
    if rec is not None:
        if rec.get("trajectory_source") == TRAJ_GT_ORACLE:
            print(f"  !! {pred_dir} carries a GROUND-TRUTH camera trajectory. Any world-space "
                  f"number from it is an ORACLE and must NOT be tabulated beside SLAM-composed "
                  f"or feedforward rows.", flush=True)
        return rec
    print(f"  !! {pred_dir} has no {PROVENANCE_FILENAME}: the box source and trajectory source of "
          f"these predictions are UNKNOWN. Numbers from it cannot be shown to be input-matched. "
          f"Re-generate with a producer that stamps provenance before quoting this in a table.",
          flush=True)
    return {"box_source": "UNKNOWN", "trajectory_source": "UNKNOWN",
            "note": f"no {PROVENANCE_FILENAME} in {pred_dir}"}
