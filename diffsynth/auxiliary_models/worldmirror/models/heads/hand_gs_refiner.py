"""Hand↔GS cross-attention refiner.

Refines Gaussian positions in the hand region by letting per-pixel GS features
attend to the two hand tokens produced by HamerManoHead. The attended feature
is projected to a small 3D offset (Δxyz) applied only to Gaussians whose pixel
falls inside either hand bounding box.
"""

import torch
import torch.nn as nn

from .hamer_head import CrossAttention


class HandGSCrossAttnRefiner(nn.Module):
    LOCAL_DIM = 3  # Δxyz

    def __init__(
        self,
        gs_feature_dim: int,
        hand_token_dim: int,
        heads: int = 4,
        dim_head: int = 64,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        max_offset: float = 0.05,
    ):
        super().__init__()
        self.max_offset = max_offset

        self.gs_norm = nn.LayerNorm(gs_feature_dim)
        self.hand_norm = nn.LayerNorm(hand_token_dim)
        self.cross_attn = CrossAttention(
            dim=gs_feature_dim,
            context_dim=hand_token_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.post_norm = nn.LayerNorm(gs_feature_dim)
        self.offset_mlp = nn.Sequential(
            nn.Linear(gs_feature_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, self.LOCAL_DIM),
        )
        # Start with near-zero offsets so training begins from identity.
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    def forward(
        self,
        gs_feats: torch.Tensor,
        hand_tokens: torch.Tensor,
        hand_bboxes: torch.Tensor | None = None,
        hand_valid: torch.Tensor | None = None,
    ):
        """
        Args:
            gs_feats: [B, S, C_gs, H, W] spatial GS features.
            hand_tokens: [B, S, 2, C_hand] per-hand tokens from HamerManoHead.
            hand_bboxes: [B, S, 2, 4] normalized bboxes (x1, y1, x2, y2) in [0, 1].
            hand_valid: [B, S, 2] bool, True for valid hands.

        Returns:
            delta_xyz: [B, S, H, W, 3] per-pixel 3D offset.
            hand_mask: [B, S, H, W] float mask (1.0 inside hand region, 0.0 elsewhere).
        """
        B, S, C, H, W = gs_feats.shape
        N = B * S

        q = gs_feats.reshape(N, C, H * W).permute(0, 2, 1).contiguous()  # [N, H*W, C]
        q = self.gs_norm(q)

        kv = hand_tokens.reshape(N, 2, -1)  # [N, 2, C_hand]
        kv = self.hand_norm(kv)
        if hand_valid is not None:
            kv = kv * hand_valid.reshape(N, 2, 1).to(kv.dtype)

        attended = self.cross_attn(q, context=kv)  # [N, H*W, C]
        attended = self.post_norm(attended)

        offset = self.offset_mlp(attended)  # [N, H*W, 3]
        offset = torch.tanh(offset) * self.max_offset
        delta_xyz = offset.reshape(B, S, H, W, 3)

        hand_mask = self._bbox_mask(hand_bboxes, hand_valid, B, S, H, W,
                                    delta_xyz.device, delta_xyz.dtype)
        return delta_xyz, hand_mask

    @staticmethod
    def _bbox_mask(hand_bboxes, hand_valid, B, S, H, W, device, dtype):
        if hand_bboxes is None:
            return torch.zeros(B, S, H, W, device=device, dtype=dtype)

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        # hand_bboxes: [B, S, 2, 4]
        x1 = hand_bboxes[..., 0, None, None]
        y1 = hand_bboxes[..., 1, None, None]
        x2 = hand_bboxes[..., 2, None, None]
        y2 = hand_bboxes[..., 3, None, None]
        inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)  # [B, S, 2, H, W]

        if hand_valid is not None:
            inside = inside & hand_valid[..., None, None].bool()

        return inside.any(dim=2).to(dtype)  # [B, S, H, W]
