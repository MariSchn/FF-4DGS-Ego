"""High-resolution hand-crop encoder branch (the pose-resolution fix).

The whole FF-4DGS-Ego pipeline runs at 224 px, which caps hand-pose accuracy
(PA-MPJPE 43.6 on H2O; WiLoR drops from ~6-9 mm at full res to 47.6 on our 224 px
protocol). This module adds a small image encoder that consumes a NATIVE 256 px
crop tightly around each hand and produces a feature map. Those features are then
fused into the HaMeR head's cropped-feature context right before MANO regression,
so the head sees real high-frequency hand detail instead of a few 224 px patches.

Behind a config flag (model.hires_hand). Default off -> the 224 px path is
byte-for-byte unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _build_resnet_trunk(arch: str, pretrained: bool):
    """Return (trunk, feat_channels) — a torchvision ResNet truncated before avgpool.

    The trunk outputs a [N, C, h, w] spatial feature map (h=w=crop/32). We keep the
    spatial map (not the pooled vector) so the fusion can hand the regression
    transformer a sequence of high-res tokens, mirroring the existing crop tokens.
    """
    import torchvision

    weights = None
    if pretrained:
        try:
            weights = "DEFAULT"
        except Exception:
            weights = None
    ctor = getattr(torchvision.models, arch)
    try:
        net = ctor(weights=weights)
    except Exception:
        # Older torchvision or no network access for weights -> random init.
        net = ctor(weights=None)
    feat_channels = net.fc.in_features  # 512 for resnet18/34, 2048 for resnet50
    trunk = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3, net.layer4,
    )
    return trunk, feat_channels


class HiResHandEncoder(nn.Module):
    """Encode a native-res hand crop -> a sequence of `out_dim` tokens.

    Forward takes crops shaped [M, 3, crop_px, crop_px] (M = arbitrary batch of
    hand crops, already RGB in [0,1]) and returns [M, T, out_dim] where T is the
    number of spatial locations in the trunk's final feature map. The HaMeR head
    concatenates these as extra key/value tokens for its regression cross-attention.
    """

    # ImageNet normalisation (the torchvision pretrained trunks expect it).
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, out_dim: int = 1024, arch: str = "resnet18",
                 pretrained: bool = True, crop_px: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.crop_px = crop_px
        self.trunk, feat_channels = _build_resnet_trunk(arch, pretrained)
        self.proj = nn.Linear(feat_channels, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1), persistent=False)

    def _normalize(self, crops: torch.Tensor) -> torch.Tensor:
        return (crops - self._mean.to(crops.dtype)) / self._std.to(crops.dtype)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """crops [M, 3, P, P] in [0,1] -> tokens [M, T, out_dim]."""
        x = self._normalize(crops)
        feat = self.trunk(x)                      # [M, C, h, w]
        m, c, h, w = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)  # [M, h*w, C]
        tokens = self.proj(tokens)                 # [M, h*w, out_dim]
        return self.norm(tokens)
