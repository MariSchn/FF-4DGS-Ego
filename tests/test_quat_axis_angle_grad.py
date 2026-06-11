"""Regression tests for quat_wxyz_to_axis_angle_torch gradient stability.

Root cause of the P1a "frozen zombie" run (job 97944): converting a degenerate
(near-zero-norm) quaternion -- an absent-hand filler or a collapsed head
prediction -- to axis-angle has a Jacobian of ~pi/eps^2 (~3e16). The forward is
finite (it maps to ~0), so the loss looked healthy, but that enormous gradient
became NaN as it propagated and combined under bf16. grad_norm went NaN, the
NaN-guard skipped every optimizer step, and training silently froze (loss
finite, weights unchanged, identical validation at steps 200/250/300).

Note: a NaN check alone does NOT catch this -- 3e16 is finite. The invariant we
pin is that the gradient stays BOUNDED for degenerate inputs (the fix replaces
them with identity, giving a zero gradient), while real rotations are unchanged.
The model trains in fp32/bf16, so those are the dtypes under test.
"""
import math
import sys
import types

import torch

# hand_vis_utils imports heavy optional deps (decord, cv2) at module load that
# are unrelated to the pure-torch function under test. Stub any that are absent
# so the import succeeds in a minimal (CPU/CI) environment.
for _name in ("decord", "cv2"):
    try:
        __import__(_name)
    except Exception:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("decord", types.ModuleType("decord")), "VideoReader"):
    sys.modules.setdefault("decord", types.ModuleType("decord")).VideoReader = object

from scripts.hand_vis_utils import quat_wxyz_to_axis_angle_torch

# Training dtypes: fp32 master weights + bf16 autocast (amp_bf16=True).
TRAIN_DTYPES = (torch.float32, torch.bfloat16)
# Real-rotation gradients are O(1); the bug produced ~3e16. Anything under this
# bound means the singular Jacobian has been tamed.
GRAD_BOUND = 1e3


def _max_abs_grad(values, dtype):
    q = torch.tensor([values], dtype=dtype).detach().requires_grad_(True)
    out = quat_wxyz_to_axis_angle_torch(q)
    out.sum().backward()
    g = q.grad
    assert torch.isfinite(g).all(), f"non-finite gradient for {values} in {dtype}: {g}"
    return g.abs().max().item()


def test_zero_quaternion_gradient_is_bounded():
    # Absent-hand filler. The pre-fix Jacobian here was ~3e16.
    for dt in TRAIN_DTYPES:
        gmax = _max_abs_grad([0.0, 0.0, 0.0, 0.0], dt)
        assert gmax < GRAD_BOUND, (dt, gmax)


def test_near_zero_norm_quaternion_gradient_is_bounded():
    # A collapsed head prediction: all four components driven toward zero.
    for dt in TRAIN_DTYPES:
        gmax = _max_abs_grad([1e-9, 1e-9, 1e-9, 1e-9], dt)
        assert gmax < GRAD_BOUND, (dt, gmax)


def test_identity_rotation_gradient_is_bounded():
    for dt in TRAIN_DTYPES:
        gmax = _max_abs_grad([1.0, 0.0, 0.0, 0.0], dt)
        assert gmax < GRAD_BOUND, (dt, gmax)


def test_present_near_identity_rotation_gradient_is_bounded():
    # A real hand barely rotated from identity (norm ~1, tiny vector part). This
    # is NOT degenerate and must still produce a finite, well-scaled gradient.
    for dt in TRAIN_DTYPES:
        gmax = _max_abs_grad([1.0, 1e-9, -1e-9, 5e-10], dt)
        assert gmax < GRAD_BOUND, (dt, gmax)


def test_normal_rotations_have_bounded_gradient():
    for values in ([0.7071, 0.7071, 0.0, 0.0], [0.9659, 0.0, 0.2588, 0.0]):
        for dt in TRAIN_DTYPES:
            gmax = _max_abs_grad(values, dt)
            assert gmax < GRAD_BOUND, (values, dt, gmax)


def test_forward_correct_for_known_rotation():
    # 90 deg about +x: w = cos(45), x = sin(45) -> axis-angle = (pi/2, 0, 0).
    q = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
    aa = quat_wxyz_to_axis_angle_torch(q)
    expected = torch.tensor([[math.pi / 2, 0.0, 0.0]])
    assert torch.allclose(aa, expected, atol=1e-4), (aa, expected)


def test_forward_identity_is_near_zero_rotvec():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    aa = quat_wxyz_to_axis_angle_torch(q)
    assert torch.allclose(aa, torch.zeros_like(aa), atol=1e-6), aa


def test_forward_degenerate_maps_to_zero_rotvec():
    # Replaced with identity -> zero rotation vector (and zero gradient).
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    aa = quat_wxyz_to_axis_angle_torch(q)
    assert torch.allclose(aa, torch.zeros_like(aa), atol=1e-6), aa
