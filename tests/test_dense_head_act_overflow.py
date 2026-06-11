"""Regression tests for exp-overflow NaN gradients in DPTHead.activate_head.

Root cause of the P1a round-2 NaN freeze (jobs 98128/97944, anomaly-traced in
job 98417 to ExpBackward0 at dense_head.py activate_head): the gs_head depth
activation is bare exp. When hand-injected features push a raw depth feature
past ~88, exp overflows to inf. The forward stays finite (overflowed pixels are
masked/filtered downstream), but backward then computes grad_in = grad_out *
exp(x) = 0 * inf = NaN, which spreads through the shared upstream features to
every render-path loss term (gs_l1, gs_lpips, hand_depth_anchor).

The fix clamps the exponent (_EXP_ACT_MAX); exp(20) ~ 4.9e8 is huge but finite,
so a zero downstream gradient stays zero. Healthy values are unaffected.
"""
import math

import torch

from diffsynth.auxiliary_models.worldmirror.models.heads.dense_head import (
    DPTHead,
    _EXP_ACT_MAX,
)

TRAIN_DTYPES = (torch.float32, torch.bfloat16)
OVERFLOW_RAW = 100.0  # > ~88 -> exp() = inf in both fp32 and bf16


def _head():
    # activate_head only uses self for _apply_inverse_log_transform; skip __init__.
    return object.__new__(DPTHead)


def _activated(activation, raw, dtype):
    """Run activate_head on a [B=1, C=2, H=1, W=2] map: healthy col 0, raw col 1."""
    out_head = torch.tensor(
        [[[[0.5, raw]], [[0.5, raw]]]], dtype=dtype
    ).detach().requires_grad_(True)
    h = _head()
    attr, conf = h.activate_head(out_head, activation=activation)
    return out_head, attr, conf


def test_masked_overflow_gradient_stays_finite_exp():
    # The exact mechanism: loss reads ONLY the healthy column, so the overflow
    # column receives grad 0 -- which must stay 0, not become 0 * inf = NaN.
    for dt in TRAIN_DTYPES:
        out_head, attr, conf = _activated("exp+expp1", OVERFLOW_RAW, dt)
        loss = attr[:, :, 0, :].sum() + conf[:, :, 0].sum()
        assert torch.isfinite(loss), (dt, loss)
        loss.backward()
        g = out_head.grad
        assert torch.isfinite(g).all(), f"non-finite grad in {dt}: {g}"


def test_masked_overflow_gradient_stays_finite_inv_log():
    for dt in TRAIN_DTYPES:
        out_head, attr, conf = _activated("inv_log+expp1", OVERFLOW_RAW, dt)
        loss = attr[:, :, 0, :].sum() + conf[:, :, 0].sum()
        assert torch.isfinite(loss), (dt, loss)
        loss.backward()
        g = out_head.grad
        assert torch.isfinite(g).all(), f"non-finite grad in {dt}: {g}"


def test_healthy_values_unchanged_by_clamp():
    out_head, attr, conf = _activated("exp+expp1", 1.0, torch.float32)
    assert torch.allclose(attr[0, 0, 0], torch.tensor([math.exp(0.5)]), atol=1e-5)
    assert torch.allclose(attr[0, 0, 1], torch.tensor([math.exp(1.0)]), atol=1e-5)
    assert torch.allclose(conf[0, 0, 0], torch.tensor(1 + math.exp(0.5)), atol=1e-5)


def test_overflow_forward_is_large_but_finite():
    _, attr, conf = _activated("exp+expp1", OVERFLOW_RAW, torch.float32)
    assert torch.isfinite(attr).all(), attr
    assert torch.isfinite(conf).all(), conf
    assert attr[0, 0, 1].item() == torch.exp(torch.tensor(_EXP_ACT_MAX)).item()


def test_mechanism_control_unclamped_exp_does_produce_nan():
    # Documents the failure mode the fix removes: a zero downstream gradient
    # through an overflowed bare exp turns NaN. If torch ever changes this
    # semantics, the clamp becomes optional -- this test will flag it.
    x = torch.tensor([0.5, OVERFLOW_RAW], requires_grad=True)
    y = torch.exp(x)            # y[1] = inf
    y[0].backward()             # grad for x[1] = 0 * inf = NaN
    assert not torch.isfinite(x.grad).all(), x.grad
