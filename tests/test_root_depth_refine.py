"""CPU unit tests for RootDepthRefine (Mac-runnable).

The module is loaded directly via importlib so the test does NOT trigger
diffsynth/__init__ (which imports modelscope, absent on dev machines). Run:
    python tests/test_root_depth_refine.py
"""
import importlib.util
import os

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(
    _ROOT, "diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py"
)
_spec = importlib.util.spec_from_file_location("root_depth_refine", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RootDepthRefine = _mod.RootDepthRefine


def _inputs(B=2, S=3):
    wrist_z = torch.full((B, S, 2), 0.40)
    d_scene = torch.full((B, S, 2), 0.45)
    conf = torch.full((B, S, 2), 0.9)
    in_frame = torch.ones(B, S, 2, dtype=torch.bool)
    return wrist_z, d_scene, conf, in_frame


def test_zero_init_is_noop():
    m = RootDepthRefine()
    dz, gate = m(*_inputs())
    assert torch.allclose(dz, torch.zeros_like(dz)), "zero-init must emit dz=0 (warm-start preserved)"
    assert gate.all(), "high conf, in frame, small disagreement -> gated ON"


def test_gate_off_when_conf_low():
    m = RootDepthRefine(conf_thresh=0.5)
    wrist_z, d_scene, _, in_frame = _inputs()
    conf = torch.full_like(wrist_z, 0.1)
    dz, gate = m(wrist_z, d_scene, conf, in_frame)
    assert not gate.any(), "conf below threshold must gate OFF"
    assert torch.allclose(dz, torch.zeros_like(dz)), "gated-off dz must be 0"


def test_gate_off_when_disagreement_exceeds_band():
    m = RootDepthRefine(band_m=0.2)
    wrist_z, _, conf, in_frame = _inputs()
    d_scene = wrist_z + 1.0  # 1 m disagreement -> free-space background
    dz, gate = m(wrist_z, d_scene, conf, in_frame)
    assert not gate.any(), "disagreement beyond band must gate OFF"


def test_nonzero_weights_move_toward_scene():
    m = RootDepthRefine()
    # force a known positive response on the (d_scene - wrist_z) feature (index 2)
    with torch.no_grad():
        m.net[0].weight.zero_(); m.net[0].bias.zero_()
        m.net[0].weight[0, 2] = 1.0          # hidden unit 0 = disagreement
        m.net[-1].weight.zero_(); m.net[-1].bias.zero_()
        m.net[-1].weight[0, 0] = 1.0          # pass hidden unit 0 to output
    wrist_z, d_scene, conf, in_frame = _inputs()  # d_scene - wrist_z = +0.05
    dz, gate = m(wrist_z, d_scene, conf, in_frame)
    assert (dz[gate] > 0).all(), "positive disagreement should yield a positive shift toward scene"


if __name__ == "__main__":
    test_zero_init_is_noop()
    test_gate_off_when_conf_low()
    test_gate_off_when_disagreement_exceeds_band()
    test_nonzero_weights_move_toward_scene()
    print("PASS: RootDepthRefine unit tests")
