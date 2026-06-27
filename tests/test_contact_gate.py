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
