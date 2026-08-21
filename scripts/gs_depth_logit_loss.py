"""Supervise the Gaussian depth on its pre-activation logit, not on the activated depth.

`gs_depth` is `exp(clamp(z, max=20))` (`dense_head.py:355`). The clamp has zero gradient above its
maximum, so a loss on the activated depth cannot move a unit that has already saturated, and a run
whose depth reached `exp(20)` in every pixel could not be pulled back by any downstream term. This
compares `z` against `log(d_target)` instead, which keeps a gradient in exactly that regime.

The residual is a log-ratio, so a depth wrong by orders of magnitude produces a residual of that
many nats rather than saturating a metric knee. Comparing depths on their logarithm is also the
right form for a quantity that varies multiplicatively.

Target choice. Supervising against metric sensor depth puts `gs_depth` in metres and therefore in
MANO's units directly. That is a deliberate departure from upstream, which divides its targets by a
per-clip scene scale and so keeps a normalised gauge. We take the metric target because the point of
this project is that the two branches share one unit.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

D_MIN, D_MAX = 0.05, 10.0     # metres; a sensor reading outside this is not a surface
LOGIT_MAX = 20.0              # dense_head._EXP_ACT_MAX, kept here so the diagnostic can report it


def gs_depth_logit_loss(
    gs_depth_logit: torch.Tensor,
    target_depth: torch.Tensor,
    *,
    beta: float = 0.1,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Smooth-L1 between the raw depth logit and the log of a target depth.

    Args:
        gs_depth_logit: [B, S, H, W] pre-activation depth, from `preds["gs_depth_logit"]`.
        target_depth:   [B, S, H, W] positive target depth in metres, broadcastable to the logit.
        beta:           Huber knee, in log units. 0.1 is about a 10% relative error.
        valid:          optional [B, S, H, W] bool mask on top of the range gate.

    Returns:
        (loss, info). Loss is 0 when nothing is valid, and never NaN.
    """
    z = gs_depth_logit
    d = target_depth
    while d.dim() > z.dim():
        if d.shape[-1] == 1:
            d = d.squeeze(-1)
        elif d.dim() > 2 and d.shape[2] == 1:
            d = d.squeeze(2)
        else:
            raise ValueError(f"target {tuple(d.shape)} cannot be aligned to logit {tuple(z.shape)}")
    if d.shape != z.shape:
        raise ValueError(f"target {tuple(d.shape)} does not match logit {tuple(z.shape)}")

    m = torch.isfinite(d) & (d > D_MIN) & (d < D_MAX) & torch.isfinite(z)
    if valid is not None:
        m = m & valid.to(torch.bool)
    n = int(m.sum())
    if n == 0:
        return z.new_zeros(()), {"logit_residual": 0.0, "n_valid": 0, "frac_at_clamp": 0.0,
                                 "logit_median": 0.0}

    per_pix = F.smooth_l1_loss(z, d.clamp_min(D_MIN).log(), beta=beta, reduction="none")
    mf = m.to(per_pix.dtype)
    loss = (per_pix * mf).sum() / mf.sum()
    with torch.no_grad():
        res = ((z - d.clamp_min(D_MIN).log()).abs() * mf).sum() / mf.sum()
        frac = float((z >= LOGIT_MAX).to(torch.float32).mean())
        med = float(z[m].median())
    return loss, {"logit_residual": float(res), "n_valid": n,
                  "frac_at_clamp": frac, "logit_median": med}


def _selftest() -> None:
    """A logit already past the clamp must still receive a gradient, which is the whole point,
    and the residual must grow with the number of decades the depth is wrong by."""
    torch.manual_seed(0)
    B, S, H, W = 1, 2, 8, 8
    target = torch.full((B, S, H, W), 0.70)

    z_ok = target.log().clone().requires_grad_(True)
    l0, i0 = gs_depth_logit_loss(z_ok, target)
    assert float(l0) < 1e-6, f"aligned case should be ~0, got {float(l0)}"

    # the case that matters: saturated past the clamp, where a loss on the activated depth is dead
    z_sat = torch.full((B, S, H, W), 25.0, requires_grad=True)
    l1, i1 = gs_depth_logit_loss(z_sat, target)
    l1.backward()
    assert z_sat.grad is not None and float(z_sat.grad.abs().sum()) > 0, \
        "no gradient at a saturated logit, which is the regime this loss exists for"
    assert i1["frac_at_clamp"] == 1.0, "diagnostic should report full saturation"

    grew = [float(gs_depth_logit_loss(target.log() + k, target)[0]) for k in (0.0, 1.0, 5.0, 20.0)]
    assert grew == sorted(grew), f"loss must grow with the log error, got {grew}"
    print(f"gs_depth_logit_loss self-test: OK (aligned {float(l0):.2e}, saturated loss {float(l1):.2f} "
          f"with grad {float(z_sat.grad.abs().sum()):.3f}, residual {i1['logit_residual']:.2f} nats, "
          f"growth {['%.2f' % g for g in grew]})")


if __name__ == "__main__":
    _selftest()
