"""Feedforward global metric-scale head.

The frozen Gaussian backbone predicts an up-to-scale scene; the MANO hand is
metric. ``solve_metric_scale`` recovers the single global scalar ``s`` that lines
them up by a closed-form median solve at inference. This head learns to predict
that same ``s`` in one forward pass from the camera/register token, so the metric
coupling becomes part of the feedforward model (no per-clip solve, and a scale is
available even on frames where the hand is briefly out of view).

``s`` is then multiplied onto the camera translation + gs_depth (apply_metric_scale)
to unify the metric hand with the GS scene — the "add a scale head rather than do
metric-scale GS" route. The head is supervised end-to-end against the metric hand
(see scripts.scale_head_loss), whose optimum is exactly the closed-form solve.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScaleHead(nn.Module):
    """Predict one positive global scale per clip from the register token.

    Reads the same token CameraHead uses (``feat_seq[-1][:, :, 0]``), pools it
    over the frame axis, and maps it to a log-scale; ``exp`` gives a positive
    scale initialised at ~1 (identity) so an untrained head does not perturb the
    scene.
    """

    def __init__(self, dim_in: int, hidden: int = 256, clamp: tuple[float, float] = (0.1, 10.0)):
        super().__init__()
        self.clamp = clamp
        self.norm = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the last layer -> log_s = 0 -> s = 1 at start (identity scale).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat_seq: list) -> torch.Tensor:
        """feat_seq: list of [B, S, N, dim_in] token tensors. Returns s: [B]."""
        reg = feat_seq[-1][:, :, 0]          # [B, S, dim_in] register/camera token
        reg = self.norm(reg)
        pooled = reg.mean(dim=1)             # [B, dim_in] global over frames
        log_s = self.mlp(pooled).squeeze(-1)  # [B]
        s = torch.exp(log_s)
        return s.clamp(self.clamp[0], self.clamp[1])
