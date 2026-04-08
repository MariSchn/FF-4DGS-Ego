"""
Hand prediction head with ROI-Align feature cropping + Cross-Attention.

Instead of processing the full-image feature map and spatially averaging
(which dilutes hand signal with background), this head:

  1. Projects backbone tokens to multi-scale 2D feature maps (same as DPTHead)
  2. Adds positional embeddings *before* ROI Align (preserving absolute spatial info)
  3. Applies ROI Align at each scale using hand bounding boxes
  4. Fuses cropped multi-scale features via DPT-style refinement blocks
  5. Applies Cross-Attention: local crop features (Q) attend to the global
     backbone feature map (K, V), conditioned on crop geometry
  6. Predicts per-hand MANO parameters via a lightweight MLP head
  7. (Optional, default on) Converts crop-relative translation predictions
     back to global camera-space t_xyz via pinhole back-projection,
     following the HaMeR/CLIFF approach.

The bounding boxes are expected in normalized [0, 1] coordinates
(x1, y1, x2, y2) and can come from any source: GT projection, a hand
detector, or a learned proposal network.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .dense_head import _make_scratch, _make_fusion_block, custom_interpolate
from ..utils.grid import create_uv_grid, position_grid_to_embed


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between local cropped features (Q) and global backbone
    features (K, V), conditioned on crop geometry.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    The crop_geometry vector is projected and *added* to every Query token so
    the model knows where in the full frame the crop originated — this helps
    resolve the global wrist translation t_xyz.

    Parameters
    ----------
    local_dim : int
        Channel dimension of the local (ROI-aligned / DPT-fused) feature map.
    global_dim : int
        Channel dimension of the global backbone feature map.
    embed_dim : int
        Common embedding dimension used inside the attention block.
        Must be divisible by both 2 and ``num_heads``.
    num_heads : int
        Number of attention heads (default 8).
    geom_dim : int
        Dimension of the crop-geometry conditioning vector (default 4).
    """

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        geom_dim: int = 4,
    ) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for 2-D positional embeddings"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim

        # Pre-attention layer norms (applied over channel dim)
        self.norm_q = nn.LayerNorm(local_dim)
        self.norm_kv = nn.LayerNorm(global_dim)

        # Project local → Q, global → K and V (all in embed_dim space)
        self.q_proj = nn.Linear(local_dim, embed_dim)
        self.k_proj = nn.Linear(global_dim, embed_dim)
        self.v_proj = nn.Linear(global_dim, embed_dim)

        # Crop-geometry conditioning: injected as an additive bias on every Q token
        self.geom_proj = nn.Linear(geom_dim, embed_dim)

        # Multi-head cross-attention (batch_first for clean [B, L, C] semantics)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Output projection + post-attention residual norm
        self.out_proj = nn.Linear(embed_dim, local_dim)
        self.norm_out = nn.LayerNorm(local_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        local_feat: torch.Tensor,   # [M, C_l, H_l, W_l]
        global_feat: torch.Tensor,  # [N, C_g, H_g, W_g]
        crop_geom: torch.Tensor,    # [M, 4]
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        local_feat : Tensor ``[M, C_l, H_l, W_l]``
            Local cropped features (M = N × num_hands).
        global_feat : Tensor ``[N, C_g, H_g, W_g]``
            Full-frame global backbone features (N = B × S frames).
        crop_geom : Tensor ``[M, 4]``
            Crop geometry vector (offset_x, offset_y, crop_w, crop_h) per hand.

        Returns
        -------
        Tensor ``[M, C_l, H_l, W_l]``
            Attended local features (same shape as input).
        """
        M, C_l, H_l, W_l = local_feat.shape
        N, C_g, H_g, W_g = global_feat.shape
        num_hands = M // N

        # ── Flatten spatial dims to token sequences ──────────────────────
        # local:  [M, H_l*W_l, C_l]
        local_seq = local_feat.flatten(2).permute(0, 2, 1)
        local_seq_norm = self.norm_q(local_seq)

        # global: [N, H_g*W_g, C_g] → replicate for each hand → [M, H_g*W_g, C_g]
        global_seq = global_feat.flatten(2).permute(0, 2, 1)
        global_seq_norm = self.norm_kv(global_seq)
        global_seq_norm = global_seq_norm.repeat_interleave(num_hands, dim=0)

        # ── Project to common embed_dim ───────────────────────────────────
        Q = self.q_proj(local_seq_norm)    # [M, H_l*W_l, embed_dim]
        K = self.k_proj(global_seq_norm)   # [M, H_g*W_g, embed_dim]
        V = self.v_proj(global_seq_norm)   # [M, H_g*W_g, embed_dim]

        # ── 2-D sinusoidal positional embeddings ──────────────────────────
        Q = Q + self._pos_embed_2d(H_l, W_l, self.embed_dim, Q.device, Q.dtype)
        K = K + self._pos_embed_2d(H_g, W_g, self.embed_dim, K.device, K.dtype)

        # ── Inject crop geometry into Q (broadcast over sequence length) ──
        # geom_emb: [M, 1, embed_dim] — same offset for every Q token
        geom_emb = self.geom_proj(crop_geom).unsqueeze(1)
        Q = Q + geom_emb

        # ── Cross-attention: Attention(Q, K, V) ───────────────────────────
        attn_out, _ = self.attn(Q, K, V)   # [M, H_l*W_l, embed_dim]

        # ── Output projection + residual connection ───────────────────────
        attn_out = self.out_proj(attn_out)              # [M, H_l*W_l, C_l]
        attn_out = self.norm_out(attn_out + local_seq)  # pre-norm residual

        # Reshape back to spatial map [M, C_l, H_l, W_l]
        return attn_out.permute(0, 2, 1).reshape(M, C_l, H_l, W_l)

    @staticmethod
    def _pos_embed_2d(
        H: int,
        W: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """2-D sinusoidal positional embedding.

        Encodes Y-axis in the first ``dim//2`` channels and X-axis in the
        remaining ``dim - dim//2`` channels using alternating sin/cos pairs.

        Returns ``[1, H*W, dim]``.
        """
        dim_y = dim // 2
        dim_x = dim - dim_y
        # Both halves must be even for sin/cos interleaving
        dim_y = (dim_y // 2) * 2
        dim_x = (dim_x // 2) * 2

        ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        def _sinusoidal(grid: torch.Tensor, out_dim: int) -> torch.Tensor:
            div = torch.exp(
                torch.arange(0, out_dim, 2, device=device, dtype=dtype)
                * (-math.log(10000.0) / out_dim)
            )                                                   # [out_dim//2]
            pe = torch.zeros(H, W, out_dim, device=device, dtype=dtype)
            pe[..., 0::2] = torch.sin(grid.unsqueeze(-1) * div)
            pe[..., 1::2] = torch.cos(grid.unsqueeze(-1) * div)
            return pe

        pe = torch.cat([
            _sinusoidal(grid_y, dim_y),
            _sinusoidal(grid_x, dim_x),
        ], dim=-1)                                              # [H, W, dim_y+dim_x]

        # Pad to exactly `dim` if rounding dropped a channel
        if pe.shape[-1] < dim:
            pad = torch.zeros(H, W, dim - pe.shape[-1], device=device, dtype=dtype)
            pe = torch.cat([pe, pad], dim=-1)

        return pe.reshape(1, H * W, dim)


# ---------------------------------------------------------------------------
# Hand Crop Head
# ---------------------------------------------------------------------------

class HandCropHead(nn.Module):
    """
    Hand prediction head with ROI Align on backbone feature maps, enhanced
    with a Cross-Attention block for global spatial awareness.

    Parameters
    ----------
    dim_in : int
        Input feature dimension from backbone (typically ``2 * embed_dim``).
    patch_size : int
        Backbone patch size (default 14 for DINOv2-L).
    hand_param_dim : int
        Output dimension **per hand** (32 for MANO: 3 pos + 4 quat + 15 pose + 10 betas).
    num_hands : int
        Number of hands to predict per frame (default 2).
    crop_size : int
        ROI Align output size at the *native patch-grid* scale.
        Other DPT scales are derived proportionally (4x, 2x, 1x, 0.5x).
        With ``crop_size=8`` and ``patch_size=14``: the finest crop is 32x32,
        covering 8*14=112 pixels in the original 224x224 image.
    features : int
        Channel width of the DPT fusion blocks.
    out_channels : list[int]
        Per-layer projection channel sizes (must have 4 entries).
        ``out_channels[2]`` is used as the global feature dim for
        cross-attention.
    pos_embed : bool
        If True, add sinusoidal positional embeddings to the projected
        feature maps *before* ROI Align so that absolute image position
        is preserved in the cropped features.
    cross_attn_heads : int
        Number of attention heads in the Cross-Attention block (default 8).
    cross_attn_local_size : int
        Local feature map is spatially pooled to this size (H=W) before
        being used as Q tokens in cross-attention.  Smaller = more
        memory-efficient; 7 gives 49 Q tokens (default).
    crop_relative_translation : bool
        If True (default), the MLP predicts ``(z, tx_crop, ty_crop)``
        instead of global ``(t_x, t_y, t_z)``.  The crop-relative
        offsets are analytically converted to global camera-space
        coordinates using the bbox geometry and focal length (HaMeR /
        CLIFF style).  This constrains the 2D reprojection of the hand
        to stay anchored near the crop centre.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        hand_param_dim: int = 32,
        num_hands: int = 2,
        crop_size: int = 8,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        pos_embed: bool = True,
        cross_attn_heads: int = 8,
        cross_attn_local_size: int = 7,
        crop_relative_translation: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hand_param_dim = hand_param_dim
        self.num_hands = num_hands
        self.crop_size = crop_size
        self.pos_embed = pos_embed
        self.crop_relative_translation = crop_relative_translation
        self._global_dim = out_channels[2]   # channel dim of feats[2]

        # --- Multi-scale projection (mirrors DPTHead) -----------------------
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0],
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])

        # --- Feature fusion (mirrors DPTHead) --------------------------------
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(
            features, features // 2, kernel_size=3, stride=1, padding=1,
        )

        # --- Cross-Attention: local crop Q ↔ global backbone K/V -----------
        # Pool the DPT-fused local features down to a compact spatial size
        # before converting them to Q tokens (memory efficiency).
        self.local_pool = nn.AdaptiveAvgPool2d(cross_attn_local_size)

        embed_dim = features // 2  # common attention embedding dimension
        self.cross_attn = CrossAttentionBlock(
            local_dim=features // 2,
            global_dim=out_channels[2],
            embed_dim=embed_dim,
            num_heads=cross_attn_heads,
            geom_dim=4,
        )

        # --- Per-hand prediction head ----------------------------------------
        # After cross-attention the spatial map is pooled → 1D feature vector.
        # Crop-geometry is concatenated so the predictor can map crop-local
        # features back to global camera coordinates.
        self.hand_feat_extractor = nn.Sequential(
            nn.Conv2d(features // 2, features // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 4, features // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                          # → [N*nh, features//4]
        )
        # +4 for crop geometry: (offset_x, offset_y, crop_w, crop_h)
        self.hand_fc = nn.Linear(features // 4 + 4, hand_param_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        hand_bboxes: torch.Tensor,
        hand_valid: Optional[torch.Tensor] = None,
        focal_length: Optional[float] = None,
        crop_local_output: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        token_list : list[Tensor]
            4 tensors of shape ``[B, S, N_tokens, C]`` from backbone
            intermediate layers.
        images : Tensor
            Input images ``[B, S, 3, H, W]`` (used only for shape).
        patch_start_idx : int
            Index where patch tokens start in the token sequence.
        hand_bboxes : Tensor
            ``[B, S, num_hands, 4]`` bounding boxes in **normalized [0, 1]**
            image coordinates, format ``(x1, y1, x2, y2)``.
        hand_valid : Tensor, optional
            ``[B, S, num_hands]`` boolean mask.  Invalid hands get zeroed
            predictions.
        focal_length : float, optional
            Camera focal length in pixels.  Required when
            ``crop_relative_translation=True``.  Defaults to ``H`` (image
            height) if not provided — a rough but usable approximation for
            Aria egocentric cameras.

        Returns
        -------
        hand_params : Tensor
            ``[B, S, num_hands * hand_param_dim]`` — concatenated MANO
            parameters for all hands.
        """
        B, S, _, H, W = images.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        N = B * S  # total number of frames

        # ---- Phase 1: project tokens → multi-scale feature maps -------------
        feats = []
        global_feat = None  # feats[2]: native patch-grid scale for cross-attention
        for i, (proj, resize, tokens) in enumerate(
            zip(self.projects, self.resize_layers, token_list)
        ):
            patch_tokens = tokens[:, :, patch_start_idx:]          # [B, S, P, C]
            patch_tokens = patch_tokens.reshape(N, -1, tokens.shape[-1])
            patch_tokens = self.norm(patch_tokens)

            feat = (patch_tokens
                    .permute(0, 2, 1)
                    .reshape(N, -1, ph, pw))                       # [N, C, ph, pw]
            feat = proj(feat)

            # Positional embeddings encode *absolute* image position so that
            # the cropped features know where in the full frame they come from.
            if self.pos_embed:
                feat = self._apply_pos_embed(feat, W, H)

            feat = resize(feat)
            feats.append(feat)

            # Save native-scale feature map for cross-attention (before ROI Align)
            if i == 2:
                global_feat = feat  # [N, out_channels[2], ph, pw]

        # Shapes (224×224 input, patch=14 → ph=pw=16):
        #   feats[0]: [N, 256,  64, 64]   (4× upsampled)
        #   feats[1]: [N, 512,  32, 32]   (2× upsampled)
        #   feats[2]: [N, 1024, 16, 16]   (native patch grid)  ← global_feat
        #   feats[3]: [N, 1024,  8,  8]   (2× downsampled)

        # ---- Phase 2: ROI Align at each scale --------------------------------
        rois = self._prepare_rois(hand_bboxes, B, S, H, W)
        # rois: [N * num_hands, 5]

        cs = self.crop_size
        roi_output_sizes = [cs * 4, cs * 2, cs, max(cs // 2, 2)]

        cropped_feats = []
        for feat, out_size in zip(feats, roi_output_sizes):
            spatial_scale = feat.shape[-1] / float(W)
            cropped = roi_align(
                feat, rois,
                output_size=(out_size, out_size),
                spatial_scale=spatial_scale,
                aligned=True,
            )
            cropped_feats.append(cropped)

        # ---- Phase 3: fuse cropped features (DPT-style) ---------------------
        layer_1_rn = self.scratch.layer1_rn(cropped_feats[0])
        layer_2_rn = self.scratch.layer2_rn(cropped_feats[1])
        layer_3_rn = self.scratch.layer3_rn(cropped_feats[2])
        layer_4_rn = self.scratch.layer4_rn(cropped_feats[3])

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        out = self.scratch.refinenet1(out, layer_1_rn)
        out = self.scratch.output_conv1(out)
        # out: [N * num_hands, features//2, crop_size*8, crop_size*8]

        # ---- Phase 3b: Cross-Attention (local ↔ global) ----------------------
        # Compute crop geometry before cross-attention so it can be used both
        # as the attention conditioning signal and for the final MLP input.
        crop_geom = self._compute_crop_geometry(hand_bboxes, B, S)  # [N*nh, 4]

        # Pool local features to a compact spatial size (e.g. 7×7) to reduce
        # Q-sequence length and keep memory usage manageable.
        out = self.local_pool(out)  # [N*nh, features//2, local_size, local_size]

        # Cross-attention: Q=local (crop), K=V=global (full frame)
        # crop_geom is injected into Q so the model can resolve global position.
        out = self.cross_attn(out, global_feat, crop_geom)
        # out: [N*nh, features//2, local_size, local_size]

        # ---- Phase 4: predict per-hand MANO params --------------------------
        # Pool and flatten the attended features
        vis_feat = self.hand_feat_extractor(out)     # [N*nh, features//4]

        # Concatenate crop geometry and predict MANO parameters
        hand_params = self.hand_fc(torch.cat([vis_feat, crop_geom], dim=-1))
        # hand_params: [N * num_hands, hand_param_dim]

        # ---- Phase 4b: convert crop-relative translation → global t_xyz -----
        # When crop_local_output=True we skip the conversion and return the
        # network's raw (z, tx_crop, ty_crop) — used during training so the
        # loss supervises in the crop-local frame directly.
        if self.crop_relative_translation and not crop_local_output:
            hand_params = self._crop_relative_to_global(
                hand_params, hand_bboxes, B, S, H, W, focal_length,
            )

        hand_params = hand_params.reshape(B, S, self.num_hands, self.hand_param_dim)

        # Zero-out predictions for absent / invalid hands
        if hand_valid is not None:
            mask = hand_valid.reshape(B, S, self.num_hands, 1).float()
            hand_params = hand_params * mask

        # Flatten hands dim → [B, S, num_hands * hand_param_dim]
        hand_params = hand_params.reshape(B, S, self.num_hands * self.hand_param_dim)
        return hand_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_rois(
        self,
        hand_bboxes: torch.Tensor,
        B: int, S: int, H: int, W: int,
    ) -> torch.Tensor:
        """Convert normalised bboxes to ``roi_align`` format.

        Returns ``[N*num_hands, 5]`` tensor where each row is
        ``[batch_idx, x1, y1, x2, y2]`` in *pixel* coordinates and
        ``batch_idx`` indexes into the ``B*S`` (= N) dim of the feature map.
        """
        N = B * S
        bboxes = hand_bboxes.reshape(N, self.num_hands, 4)

        bboxes_pixel = bboxes.clone()
        bboxes_pixel[..., [0, 2]] *= W   # x coords
        bboxes_pixel[..., [1, 3]] *= H   # y coords

        batch_idx = torch.arange(N, device=bboxes.device, dtype=bboxes.dtype)
        batch_idx = batch_idx[:, None].expand(-1, self.num_hands)  # [N, nh]

        rois = torch.cat([
            batch_idx.reshape(-1, 1),
            bboxes_pixel.reshape(-1, 4),
        ], dim=1)
        return rois

    def _compute_crop_geometry(
        self,
        hand_bboxes: torch.Tensor,
        B: int, S: int,
    ) -> torch.Tensor:
        """Principal-point adjustment: encode each crop's position and size.

        Returns ``[N*num_hands, 4]`` with columns::

            offset_x = crop_centre_x − cx   (cx = 0.5 in normalised coords)
            offset_y = crop_centre_y − cy   (cy = 0.5)
            crop_w   = x2 − x1
            crop_h   = y2 − y1

        The offset tells the predictor how far the crop centre is from
        the camera principal point so it can correctly recover the
        global wrist translation  t_x = (u − cx) · Z / f.
        """
        N = B * S
        bboxes = hand_bboxes.reshape(N, self.num_hands, 4)  # normalised

        cx = (bboxes[..., 0] + bboxes[..., 2]) / 2.0  # crop centre x
        cy = (bboxes[..., 1] + bboxes[..., 3]) / 2.0  # crop centre y
        cw = bboxes[..., 2] - bboxes[..., 0]           # crop width
        ch = bboxes[..., 3] - bboxes[..., 1]           # crop height

        # Offset from principal point (image centre = 0.5 in normalised coords)
        geom = torch.stack([cx - 0.5, cy - 0.5, cw, ch], dim=-1)
        return geom.reshape(N * self.num_hands, 4)

    def _crop_relative_to_global(
        self,
        hand_params: torch.Tensor,
        hand_bboxes: torch.Tensor,
        B: int, S: int, H: int, W: int,
        focal_length: Optional[float] = None,
    ) -> torch.Tensor:
        """Convert crop-relative translation to global camera-space t_xyz.

        The network predicts ``(z, tx_crop, ty_crop)`` in the first 3 dims of
        ``hand_params``, where ``tx_crop, ty_crop ∈ ~[-1, 1]`` are offsets
        from the crop centre in normalised crop coordinates.

        The conversion to global wrist position follows the pinhole camera
        model (matching HaMeR / CLIFF)::

            u_pixel = bbox_cx_pixel + tx_crop * bbox_w_pixel / 2
            v_pixel = bbox_cy_pixel + ty_crop * bbox_h_pixel / 2

            t_x = (u_pixel - W/2) * z / f
            t_y = (v_pixel - H/2) * z / f
            t_z = z

        Parameters
        ----------
        hand_params : Tensor ``[N*nh, hand_param_dim]``
            Raw network output.  First 3 channels are ``(z, tx_crop, ty_crop)``.
        hand_bboxes : Tensor ``[B, S, num_hands, 4]``
            Normalised bounding boxes ``(x1, y1, x2, y2)`` in [0, 1].
        focal_length : float, optional
            Focal length in pixels.  Defaults to ``H``.

        Returns
        -------
        hand_params with the first 3 channels replaced by global ``(t_x, t_y, t_z)``.
        """
        if focal_length is None:
            focal_length = float(H)

        N = B * S
        bboxes = hand_bboxes.reshape(N * self.num_hands, 4)  # [N*nh, 4]

        # Predicted crop-relative values
        z = hand_params[:, 0]           # depth (direct prediction)
        tx_crop = hand_params[:, 1]     # crop-relative x offset ∈ ~[-1, 1]
        ty_crop = hand_params[:, 2]     # crop-relative y offset ∈ ~[-1, 1]

        # Bbox centre and size in pixel coordinates
        bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0 * W
        bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0 * H
        bbox_w = (bboxes[:, 2] - bboxes[:, 0]) * W
        bbox_h = (bboxes[:, 3] - bboxes[:, 1]) * H

        # Back-project from crop-relative to global pixel coordinates
        u_pixel = bbox_cx + tx_crop * bbox_w / 2.0
        v_pixel = bbox_cy + ty_crop * bbox_h / 2.0

        # Pinhole back-projection to camera-space 3D
        t_x = (u_pixel - W / 2.0) * z / focal_length
        t_y = (v_pixel - H / 2.0) * z / focal_length

        # Replace the first 3 channels with global t_xyz
        hand_params = hand_params.clone()
        hand_params[:, 0] = t_x
        hand_params[:, 1] = t_y
        hand_params[:, 2] = z

        return hand_params

    def _apply_pos_embed(
        self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1,
    ) -> torch.Tensor:
        """Sinusoidal positional embedding (identical to DPTHead)."""
        patch_w, patch_h = x.shape[-1], x.shape[-2]
        pos = create_uv_grid(
            patch_w, patch_h,
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )
        pos = position_grid_to_embed(pos, x.shape[1])
        pos = (pos * ratio).permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos
