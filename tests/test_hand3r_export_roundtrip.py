"""Round-trip: export our arrays, then score them with Hand3R's own file, unedited.

This is the claim the exchange rests on. If their vendored reference_scorer.py can be pointed at
what export_npz writes and produce the numbers our torch port produces, then the protocol is shared
in fact and not by assertion, and the same files can carry Hand3R's predictions back to us.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest
import torch

from scripts.hand3r_protocol import reference_scorer as ref
from scripts.hand3r_protocol.export_npz import export_clip
from scripts.hand3r_protocol.hand3r_metrics import score_clip_hand3r


def _store(tmp: str, name: str, t: int = 120, j: int = 21):
    """A minimal store and prediction pair in the layouts export_npz reads."""
    g = torch.Generator().manual_seed(4)
    frame = torch.arange(t, dtype=torch.float64)[:, None]
    shape = torch.randn(j, 3, generator=g).double() * 0.03
    travel = frame * 0.008 * torch.tensor([1.0, 0.2, 0.5], dtype=torch.float64)
    gt = shape[None] + travel[:, None, :] + torch.randn(t, j, 3, generator=g).double() * 0.002
    drift = frame * 0.0007 * torch.tensor([0.3, 1.0, -0.4], dtype=torch.float64)
    pred = gt + drift[:, None, :] + torch.randn(t, j, 3, generator=g).double() * 0.003

    # Stores hold [T, 2, J, 3] with the right hand in slot 1; the left slot is never filled by any
    # of our predictors, which is why the exporter takes the right hand and says so.
    def two(x):
        out = torch.zeros(t, 2, j, 3, dtype=torch.float64)
        out[:, 1] = x
        return out

    seq = os.path.join(tmp, "store", name, "hand_data")
    os.makedirs(seq, exist_ok=True)
    torch.save(two(gt), os.path.join(seq, "gt_joints_cache_cam_v2.pt"))
    torch.save(two(gt), os.path.join(seq, "gt_joints_cache_world.pt"))

    pdir = os.path.join(tmp, "preds")
    os.makedirs(pdir, exist_ok=True)
    torch.save({"cam_joints": two(pred), "world_joints": two(pred),
                "valid": torch.ones(t, 2, dtype=torch.bool)},
               os.path.join(pdir, f"{name}.pt"))
    return os.path.join(tmp, "store", name), os.path.join(pdir, f"{name}.pt"), pred, gt


def test_their_scorer_reads_our_export(tmp_path):
    tmp = str(tmp_path)
    seq, pred_path, pred, gt = _store(tmp, "clip_a")
    out = os.path.join(tmp, "npz")
    rec = export_clip(seq, pred_path, out, allow_16=False)
    assert rec is not None and rec["joints"] == 21

    manifest = os.path.join(out, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "gt_path", "pred_path"])
        w.writeheader()
        w.writerow({k: rec[k] for k in ("clip_id", "gt_path", "pred_path")})

    scored = ref.evaluate_manifest(__import__("pathlib").Path(manifest), 100, "m")
    got = scored["summary"]
    ours = score_clip_hand3r(pred, gt, chunk_len=100)

    for their_key, our_key in (("C-MPJPE", "C_MPJPE_abs"), ("W-MPJPE", "W_MPJPE"),
                               ("WA-MPJPE", "WA_MPJPE")):
        assert got[their_key]["count"] == 1
        assert np.isclose(got[their_key]["mean"], ours[our_key], rtol=1e-6, atol=1e-6), (
            f"{their_key}: their scorer on our export {got[their_key]['mean']:.6f} "
            f"vs our port {ours[our_key]:.6f}")


def test_sixteen_joints_are_refused(tmp_path):
    """A 16-joint file scored by a 21-joint scorer is the exact mistake this exchange exists to
    prevent, so the exporter has to stop rather than pad."""
    tmp = str(tmp_path)
    seq, pred_path, _, _ = _store(tmp, "clip_b", j=16)
    with pytest.raises(SystemExit, match="21"):
        export_clip(seq, pred_path, os.path.join(tmp, "npz16"), allow_16=False)

    rec = export_clip(seq, pred_path, os.path.join(tmp, "npz16"), allow_16=True)
    assert rec["joints"] == 16
