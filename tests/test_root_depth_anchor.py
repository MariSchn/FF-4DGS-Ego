"""CPU unit tests for the root-depth anchor orchestration + loss (Mac-runnable).

Modules are loaded directly via importlib to bypass diffsynth/__init__ (modelscope).
Run: python tests/test_root_depth_anchor.py
"""
import importlib.util
import os

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RootDepthRefine = _load(
    "root_depth_refine",
    "diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py",
).RootDepthRefine
_rda = _load("root_depth_anchor", "scripts/root_depth_anchor.py")
apply_root_anchor = _rda.apply_root_anchor
root_anchor_loss = _rda.root_anchor_loss


def _scene(B=1, S=2, Hd=64, Wd=64, depth=0.5):
    return torch.full((B, S, 1, Hd, Wd), depth), torch.full((B, S, 1, Hd, Wd), 0.9)


def test_apply_is_noop_at_init():
    B, S = 1, 2
    pred_joints = torch.randn(B, S, 2, 16, 3).abs() + 0.3
    pred_joints[..., 2] = 0.5
    gs_depth, gs_conf = _scene(B, S, depth=0.5)
    cam_intr = torch.tensor([[600.0, 704.0, 704.0]])
    m = RootDepthRefine()
    corrected, dz, info = apply_root_anchor(m, pred_joints, gs_depth, gs_conf, cam_intr)
    assert torch.allclose(dz, torch.zeros_like(dz)), "zero-init -> no shift"
    assert torch.allclose(corrected, pred_joints), "zero-init -> joints unchanged"
    assert "d_scene" in info and "gate" in info


def test_apply_shifts_only_z():
    B, S = 1, 1
    pred_joints = torch.zeros(B, S, 2, 16, 3)
    pred_joints[..., 2] = 0.5
    gs_depth, gs_conf = _scene(B, S, depth=0.5)
    cam_intr = torch.tensor([[600.0, 704.0, 704.0]])
    m = RootDepthRefine()
    with torch.no_grad():  # force a constant +0.1 shift through the gate
        m.net[0].weight.zero_(); m.net[0].bias.zero_()
        m.net[-1].weight.zero_(); m.net[-1].bias.fill_(0.1)
    corrected, dz, info = apply_root_anchor(m, pred_joints, gs_depth, gs_conf, cam_intr)
    assert torch.allclose(corrected[..., :2], pred_joints[..., :2]), "x,y unchanged"
    assert torch.allclose(corrected[..., 2], pred_joints[..., 2] + dz[..., None]), "z shifted by per-hand dz"


def test_consistency_loss_zero_when_matched():
    wrist_z = torch.full((1, 2, 2), 0.5)
    d_scene = torch.full((1, 2, 2), 0.5)
    gate = torch.ones(1, 2, 2, dtype=torch.bool)
    has_hand = torch.ones(1, 2, 2)
    loss = root_anchor_loss(wrist_z, d_scene, gate, has_hand)
    assert float(loss) == 0.0, "matched depth -> zero anchor loss"


def test_consistency_loss_ignores_gated_off():
    wrist_z = torch.zeros(1, 1, 2)
    d_scene = torch.ones(1, 1, 2)
    gate = torch.zeros(1, 1, 2, dtype=torch.bool)
    has_hand = torch.ones(1, 1, 2)
    loss = root_anchor_loss(wrist_z, d_scene, gate, has_hand)
    assert float(loss) == 0.0, "all gated-off -> zero loss (no NaN from empty mean)"


if __name__ == "__main__":
    test_apply_is_noop_at_init()
    test_apply_shifts_only_z()
    test_consistency_loss_zero_when_matched()
    test_consistency_loss_ignores_gated_off()
    print("PASS: root_depth_anchor unit tests")
