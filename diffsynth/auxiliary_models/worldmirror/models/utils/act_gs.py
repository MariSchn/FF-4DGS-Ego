import math

import torch
from einops import rearrange

# Backward-NaN guards (P1a NaN-grad freeze, round 3). Two failure modes, both
# triggered once hand-injected features grow large in the hand bbox region:
#
# 1. Masked-inf exp: exp(raw > ~88) = inf. Downstream clamps/masks keep the
#    FORWARD finite (e.g. reg_dense_scales(...).clamp_max(0.3) at the call
#    site), but backward then computes grad * exp(raw) = 0 * inf = NaN, which
#    poisons every gradient sharing the upstream features (job 98564 verdict:
#    gs_l1/gs_lpips NaN while every non-render term stayed finite). Clamp the
#    exponent: inert for healthy values, removes the inf.
# 2. norm() backward at zero: d/dx ||x|| = x/||x|| = 0/0 = NaN for an all-zero
#    vector, even when the forward is guarded with +eps (same bug class as the
#    MANO quat->axis-angle fix). Use sqrt(sum(x^2) + eps^2), whose gradient is
#    finite everywhere.
_EXP_MAX = 20.0


def _safe_norm(x, eps=1e-8):
    return torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps * eps)


def reg_dense_offsets(xyz, shift=6.0):
    d = _safe_norm(xyz)
    return xyz / d.clamp(min=1e-8) * (torch.exp((d - shift).clamp(max=_EXP_MAX)) - math.exp(-shift))

def reg_dense_scales(scales):
    return scales.clamp(max=_EXP_MAX).exp()

def reg_dense_rotation(rotations, eps=1e-8):
    # Replace degenerate (near-zero-norm) quaternions with identity BEFORE
    # normalizing: torch.where zeroes their gradient, and the rasterizer gets a
    # valid unit rotation instead of a zero quat. Same pattern as the MANO
    # quat->axis-angle guard.
    identity = torch.zeros_like(rotations)
    identity[..., 0] = 1.0
    rotations = torch.where(_safe_norm(rotations, eps) < 1e-6, identity, rotations)
    return rotations / (_safe_norm(rotations, eps) + eps)

def reg_dense_sh(sh):
    return rearrange(sh, '... (d_sh xyz) -> ... d_sh xyz', xyz=3)

def reg_dense_opacities(opacities):
    return opacities.sigmoid()

def reg_dense_weights(weights):
    return weights.sigmoid()

def reg_dense_life_span(life_spans):
    return life_spans.sigmoid()
