"""Inject hand-aware local features into the GS branch's feature map.

The hamer head produces, per hand, an enhanced local crop token tensor —
local roi-aligned tokens that have already cross-attended to the full-image
patch tokens. This module projects those tokens to the GS feature map's
channel space and scatters each hand's features back into the corresponding
bbox region of the GS feature map, so the GS head sees hand-aware features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandToGSInjection(nn.Module):
    def __init__(
        self,
        hand_dim: int = 1024,
        gs_dim: int = 128,
        use_hand_valid_mask: bool = True,
    ):
        super().__init__()
        self.hand_dim = hand_dim
        self.gs_dim = gs_dim
        self.use_hand_valid_mask = use_hand_valid_mask
        self.proj = nn.Conv2d(hand_dim, gs_dim, kernel_size=1)

    def forward(
        self,
        enhanced_crop_tokens: torch.Tensor,
        hand_bboxes: torch.Tensor,
        hand_valid,
        gs_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            enhanced_crop_tokens: [N*2, crop_size**2, hand_dim], N = B*S, 2 = L/R hand,
                ordered as (b*S + s) * 2 + h.
            hand_bboxes: [B, S, 2, 4] xyxy in normalized [0, 1] image coords.
            hand_valid: [B, S, 2] bool/float, or None.
            gs_feat: [B, S, gs_dim, H, W]. Not modified in place; a new tensor is returned.

        Returns:
            gs_feat_out: [B, S, gs_dim, H, W] with per-hand features added at each bbox.
        """
        B, S, C, H, W = gs_feat.shape
        N = B * S
        N2, K, hand_dim = enhanced_crop_tokens.shape
        assert N2 == N * 2, f"enhanced_crop_tokens batch {N2} != B*S*2 = {N * 2}"
        assert hand_dim == self.hand_dim, f"hand_dim {hand_dim} != {self.hand_dim}"
        assert C == self.gs_dim, f"gs_feat channel {C} != {self.gs_dim}"
        crop_size = int(round(K ** 0.5))
        assert crop_size * crop_size == K, f"token count {K} is not a perfect square"
        assert hand_bboxes.shape == (B, S, 2, 4), (
            f"hand_bboxes shape {tuple(hand_bboxes.shape)} != {(B, S, 2, 4)}"
        )

        feats = enhanced_crop_tokens.permute(0, 2, 1).reshape(N2, hand_dim, crop_size, crop_size)

        if self.use_hand_valid_mask and hand_valid is not None:
            mask = hand_valid.reshape(N2, 1, 1, 1).to(feats.dtype)
            feats = feats * mask
            valid_flat = hand_valid.reshape(N2).bool()
        else:
            valid_flat = None

        feats = self.proj(feats)  # [N*2, gs_dim, crop_size, crop_size]

        bboxes = hand_bboxes.reshape(N, 2, 4)
        gs_feat_out = gs_feat.clone()

        for n in range(N):
            b = n // S
            s = n % S
            for h in range(2):
                idx = n * 2 + h

                # Skip invalid hands entirely so the conv bias doesn't leak
                # a constant feature contribution into their bbox region.
                if valid_flat is not None and not bool(valid_flat[idx]):
                    continue

                x1n, y1n, x2n, y2n = bboxes[n, h].detach().tolist()

                x1 = max(0, min(W, int(round(x1n * W))))
                y1 = max(0, min(H, int(round(y1n * H))))
                x2 = max(0, min(W, int(round(x2n * W))))
                y2 = max(0, min(H, int(round(y2n * H))))

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                resized = F.interpolate(
                    feats[idx:idx + 1],
                    size=(bh, bw),
                    mode="bilinear",
                    align_corners=False,
                )  # [1, gs_dim, bh, bw]

                gs_feat_out[b, s, :, y1:y2, x1:x2] = (
                    gs_feat_out[b, s, :, y1:y2, x1:x2] + resized.squeeze(0)
                )

        return gs_feat_out
