"""CPU unit tests for the contact anchor Phase-2 gate.

Pure-logic, deterministic, tiny synthetic tensors. Covers:
  * the per-hand GT contact predicate (scripts.contact_mask.is_contact),
  * RootDepthRefine's optional explicit contact gate (overrides the band proxy),
  * apply_root_anchor forwarding the contact mask into the gate.

Imports go through the leaf package paths; tests/conftest.py installs namespace
shims so diffsynth/__init__.py (heavy optional deps) is never executed on a bare
CPU machine.
"""
import torch

from scripts.contact_mask import is_contact
from diffsynth.auxiliary_models.worldmirror.models.heads.root_depth_refine import (
    RootDepthRefine,
)
from scripts.root_depth_anchor import apply_root_anchor


def test_is_contact_true_when_within_threshold():
    wrist_z = torch.tensor([[[0.40, 0.40]]])   # [B=1,S=1,2]
    dense_at = torch.tensor([[[0.42, 0.99]]])  # hand0 2cm off surface, hand1 59cm off
    in_frame = torch.tensor([[[True, True]]])
    m = is_contact(wrist_z, dense_at, in_frame, thresh_m=0.05)
    assert m.tolist() == [[[True, False]]]


def test_is_contact_false_when_out_of_frame_or_no_depth():
    wrist_z = torch.tensor([[[0.40, 0.40]]])
    dense_at = torch.tensor([[[0.40, 0.00]]])  # hand1 has no valid depth (0)
    in_frame = torch.tensor([[[False, True]]])  # hand0 out of frame
    m = is_contact(wrist_z, dense_at, in_frame, thresh_m=0.05)
    assert m.tolist() == [[[False, False]]]


def test_external_contact_gate_overrides_band():
    m = RootDepthRefine(hidden=4, conf_thresh=0.1, band_m=0.05)
    with torch.no_grad():                       # force a non-zero delta so gating is observable
        m.net[-1].weight.fill_(1.0)
        m.net[-1].bias.fill_(0.5)
    wrist_z = torch.tensor([[[0.40, 0.40]]])
    d_scene = torch.tensor([[[0.90, 0.90]]])    # disagree 0.50 >> band_m
    conf = torch.ones(1, 1, 2)
    in_frame = torch.ones(1, 1, 2, dtype=torch.bool)
    contact = torch.tensor([[[True, False]]])
    delta, gate = m(wrist_z, d_scene, conf, in_frame, contact=contact)
    assert gate.tolist() == [[[True, False]]]   # contact fires hand0 despite 50cm disagreement
    assert delta[0, 0, 1].item() == 0.0          # hand1 gated off


def test_falls_back_to_band_when_no_contact():
    m = RootDepthRefine(hidden=4, conf_thresh=0.1, band_m=0.05)
    wrist_z = torch.tensor([[[0.40, 0.40]]])
    d_scene = torch.tensor([[[0.42, 0.90]]])    # hand0 2cm (in band), hand1 50cm (out)
    conf = torch.ones(1, 1, 2)
    in_frame = torch.ones(1, 1, 2, dtype=torch.bool)
    _, gate = m(wrist_z, d_scene, conf, in_frame)   # no contact arg -> band proxy
    assert gate.tolist() == [[[True, False]]]


def test_apply_root_anchor_passes_contact_to_gate():
    # band so tight only an explicit contact can fire (disagree is 0.40m >> band).
    m = RootDepthRefine(hidden=4, conf_thresh=0.0, band_m=0.001)
    with torch.no_grad():
        m.net[-1].weight.fill_(1.0)
        m.net[-1].bias.fill_(0.3)
    B, S = 1, 1
    pred = torch.zeros(B, S, 2, 1, 3)
    pred[..., 2] = 0.5                                  # both wrists at depth 0.5m, single joint
    gs = torch.full((B, S, 1, 8, 8), 0.9)              # uniform scene depth 0.9m -> 40cm disagree
    cam_intr = torch.tensor([[600.0, 704.0, 704.0]])   # locked 1408-square intrinsics (center => in-frame)
    contact = torch.tensor([[[True, False]]])
    _, _delta, info = apply_root_anchor(m, pred, gs, None, cam_intr, contact_mask=contact)
    assert bool(info["gate"][0, 0, 0]) is True         # contact fires hand0 despite 40cm disagreement
    assert bool(info["gate"][0, 0, 1]) is False        # hand1 gated off
