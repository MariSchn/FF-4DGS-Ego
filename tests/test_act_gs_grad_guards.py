"""Regression tests for backward-NaN guards in act_gs (P1a NaN hunt, round 3).

Job 98564 verdict: with the depth-head exp clamp in place, gs_l1 and gs_lpips
still produced NaN grads at step 181 while every non-render term stayed
finite. The shared node is the rasterized RGB image; its Gaussian parameters
come from act_gs activations whose call site is
``reg_dense_scales(...).clamp_max(0.3)`` (rasterization.py:696) -- the exact
masked-inf pattern: exp(raw>88)=inf is masked in forward by the clamp, but
backward computes 0 * inf = NaN. reg_dense_rotation additionally had the
norm()-backward-at-zero trap (0/0) from the same bug class as the MANO
quat->axis-angle fix.
"""
import torch

from diffsynth.auxiliary_models.worldmirror.models.utils import act_gs

TRAIN_DTYPES = (torch.float32, torch.bfloat16)
OVERFLOW_RAW = 100.0  # > ~88 -> exp() = inf in fp32 and bf16
GRAD_BOUND = 1e6      # all guarded gradients must stay well below inf/NaN


def _grad_through(fn, values, dtype, post=lambda y: y):
    x = torch.tensor([values], dtype=dtype).detach().requires_grad_(True)
    y = post(fn(x))
    y.sum().backward()
    g = x.grad
    assert torch.isfinite(g).all(), f"non-finite grad for {values} in {dtype}: {g}"
    return g.abs().max().item()


def test_scales_masked_overflow_call_site_pattern():
    # The exact call-site pattern: reg_dense_scales(raw).clamp_max(0.3).
    # Overflowed entry gets forward 0.3 and must get grad 0, not 0*inf=NaN.
    for dt in TRAIN_DTYPES:
        gmax = _grad_through(
            act_gs.reg_dense_scales,
            [0.1, OVERFLOW_RAW, -1.0],
            dt,
            post=lambda y: y.clamp_max(0.3),
        )
        assert gmax < GRAD_BOUND, (dt, gmax)


def test_scales_healthy_values_unchanged():
    x = torch.tensor([[-1.2, 0.0, -0.5]])
    assert torch.allclose(act_gs.reg_dense_scales(x), x.exp())


def test_rotation_zero_quat_grad_bounded():
    # All-zero raw rotation (collapsed prediction): replaced with identity, so
    # the gradient is zero (no 0/0, no 1/eps blowup).
    for dt in TRAIN_DTYPES:
        gmax = _grad_through(act_gs.reg_dense_rotation, [0.0, 0.0, 0.0, 0.0], dt)
        assert gmax < GRAD_BOUND, (dt, gmax)


def test_rotation_zero_quat_maps_to_identity():
    out = act_gs.reg_dense_rotation(torch.zeros(1, 4))
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-5), out


def test_rotation_healthy_quat_unchanged():
    q = torch.tensor([[1.0, 0.5, -0.25, 0.125]])
    out = act_gs.reg_dense_rotation(q)
    expected = q / q.norm(dim=-1, keepdim=True)
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.allclose(out.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_offsets_zero_and_overflow_grad_bounded():
    # At the clamp edge the legitimate slope is O(exp(_EXP_MAX)); the guard's
    # job is to keep it FINITE (pre-fix: inf/NaN), not small.
    offsets_bound = float(torch.exp(torch.tensor(act_gs._EXP_MAX)).item()) * 10
    for dt in TRAIN_DTYPES:
        for values in ([0.0, 0.0, 0.0], [OVERFLOW_RAW, 0.0, 0.0]):
            gmax = _grad_through(act_gs.reg_dense_offsets, values, dt)
            assert gmax < offsets_bound, (values, dt, gmax)


def test_offsets_healthy_values_unchanged():
    xyz = torch.tensor([[0.3, -0.2, 0.1]])
    d = xyz.norm(dim=-1, keepdim=True)
    expected = xyz / d * (torch.exp(d - 6.0) - torch.exp(torch.tensor(-6.0)))
    assert torch.allclose(act_gs.reg_dense_offsets(xyz), expected, atol=1e-6)
