"""Inject hand-aware local features into the GS branch's DPT feature pyramid.

The hamer head produces, per hand, an enhanced local crop token tensor —
local roi-aligned tokens that have already cross-attended to the full-image
patch tokens. This module projects those tokens to each DPT layer's channel
space and scatters each hand's features back into the corresponding bbox
region of that layer's feature map, so the GS head's fusion sees hand-aware
features at every scale.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandToGSInjection(nn.Module):
    def __init__(
        self,
        hand_dim: int = 1024,
        gs_dims: Sequence[int] = (256, 512, 1024, 1024),
        use_hand_valid_mask: bool = True,
    ):
        super().__init__()
        self.hand_dim = hand_dim
        self.gs_dims = list(gs_dims)
        self.use_hand_valid_mask = use_hand_valid_mask
        self.projs = nn.ModuleList(
            [nn.Conv2d(hand_dim, gs_dim, kernel_size=1) for gs_dim in self.gs_dims]
        )

    def forward(
        self,
        enhanced_crop_tokens: torch.Tensor,
        hand_bboxes: torch.Tensor,
        hand_valid: Optional[torch.Tensor],
        feats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Args:
            enhanced_crop_tokens: [N*2, crop_size**2, hand_dim], N = B*S, 2 = L/R hand,
                ordered as (b*S + s) * 2 + h.
            hand_bboxes: [B, S, 2, 4] xyxy in normalized [0, 1] image coords.
            hand_valid: [B, S, 2] bool/float, or None.
            feats: list of [B*S, gs_dims[i], h_i, w_i] tensors, one per DPT layer.

        Returns:
            feats_out: list of [B*S, gs_dims[i], h_i, w_i] tensors with per-hand
            features added at each bbox region. Each entry is a clone of the
            corresponding input (no in-place modification).
        """
        assert len(feats) == len(self.projs), (
            f"feats count {len(feats)} != projs count {len(self.projs)}"
        )
        N2, K, hand_dim = enhanced_crop_tokens.shape
        assert hand_dim == self.hand_dim, f"hand_dim {hand_dim} != {self.hand_dim}"
        crop_size = int(round(K ** 0.5))
        assert crop_size * crop_size == K, f"token count {K} is not a perfect square"

        B, S, _, _ = hand_bboxes.shape
        N = B * S
        assert N2 == N * 2, f"enhanced_crop_tokens batch {N2} != B*S*2 = {N * 2}"
        assert hand_bboxes.shape == (B, S, 2, 4), (
            f"hand_bboxes shape {tuple(hand_bboxes.shape)} != {(B, S, 2, 4)}"
        )

        feats_2d = enhanced_crop_tokens.permute(0, 2, 1).reshape(
            N2, hand_dim, crop_size, crop_size
        )

        if self.use_hand_valid_mask and hand_valid is not None:
            mask = hand_valid.reshape(N2, 1, 1, 1).to(feats_2d.dtype)
            feats_2d = feats_2d * mask
            valid_flat = hand_valid.reshape(N2).bool()
        else:
            valid_flat = None

        bboxes = hand_bboxes.reshape(N, 2, 4)

        feats_out = []
        for i, (proj, feat_i) in enumerate(zip(self.projs, feats)):
            assert feat_i.shape[0] == N, (
                f"feat[{i}] batch {feat_i.shape[0]} != B*S = {N}"
            )
            assert feat_i.shape[1] == self.gs_dims[i], (
                f"feat[{i}] channels {feat_i.shape[1]} != gs_dims[{i}] = {self.gs_dims[i]}"
            )
            _, _, h_i, w_i = feat_i.shape

            projected = proj(feats_2d)  # [N*2, gs_dims[i], crop_size, crop_size]
            out_i = feat_i.clone()

            for n in range(N):
                b = n // S
                s = n % S
                for h in range(2):
                    idx = n * 2 + h

                    if valid_flat is not None and not bool(valid_flat[idx]):
                        continue

                    x1n, y1n, x2n, y2n = bboxes[n, h].detach().tolist()

                    x1 = max(0, min(w_i, int(round(x1n * w_i))))
                    y1 = max(0, min(h_i, int(round(y1n * h_i))))
                    x2 = max(0, min(w_i, int(round(x2n * w_i))))
                    y2 = max(0, min(h_i, int(round(y2n * h_i))))

                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue

                    resized = F.interpolate(
                        projected[idx:idx + 1],
                        size=(bh, bw),
                        mode="bilinear",
                        align_corners=False,
                    )  # [1, gs_dims[i], bh, bw]

                    out_i[n, :, y1:y2, x1:x2] = (
                        out_i[n, :, y1:y2, x1:x2] + resized.squeeze(0)
                    )

            feats_out.append(out_i)

        return feats_out
