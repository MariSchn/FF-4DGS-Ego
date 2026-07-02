"""Post-hoc root-depth correction (contact anchor Phase 1).

Given per-hand (wrist_z, d_scene, conf, in_frame), predict a depth shift delta_z
applied rigidly to the hand root. Zero-init -> delta_z = 0 at start, so a
warm-started head is unchanged until the module learns a correction. delta_z is
masked OFF where the scene-depth reference is unreliable (low confidence, out of
frame, or disagreeing with the head's own estimate by more than ``band_m``),
which is the Phase-1 contact proxy; Phase 2 replaces the gate with an explicit
hand-object contact signal.
"""
import torch
import torch.nn as nn


class RootDepthRefine(nn.Module):
    def __init__(self, hidden: int = 32, conf_thresh: float = 0.1, band_m: float = 0.5,
                 refine_ref: bool = False):
        super().__init__()
        self.conf_thresh = float(conf_thresh)
        self.band_m = float(band_m)
        # refine_ref=False (default): correct the head's own wrist depth (add a residual).
        # refine_ref=True: take the depth straight from the reference (DA3), correct its
        # global scale with a learned factor, and use that as the wrist depth wherever the
        # reference is valid. This does not depend on the head's weak absolute depth.
        self.refine_ref = bool(refine_ref)
        # features: [wrist_z, d_scene, d_scene - wrist_z, conf]
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if self.refine_ref:
            # scale = exp(log_scale) (starts at 1), shift starts at 0, so at init the
            # refined depth equals raw DA3; training then learns DA3's ~0.89 correction.
            self.log_scale = nn.Parameter(torch.zeros(()))
            self.shift = nn.Parameter(torch.zeros(()))

    def forward(self, wrist_z, d_scene, conf, in_frame, contact=None):
        """All inputs [B, S, 2] (per hand). Returns (delta_z [B,S,2], gate bool [B,S,2]).
        When ``contact`` (bool [B,S,2]) is given it REPLACES the |disagree|<band_m
        proxy: fire where the hand actually touches the surface (the scale-source
        test showed the scene depth is reliable there, biased in free space). The
        proxy gate is kept as the fallback for inputs without a contact signal."""
        disagree = d_scene - wrist_z
        feats = torch.stack([wrist_z, d_scene, disagree, conf], dim=-1)  # [B,S,2,4]
        residual = self.net(feats).squeeze(-1)  # [B,S,2]
        # Reliability gate: trust DA3 only where it agrees with the head within band_m
        # (or at true contact when given). Blindly trusting DA3 everywhere hurt on the
        # sequence where DA3 itself was bad, so we gate on agreement.
        reliable = contact.bool() if contact is not None else (disagree.abs() < self.band_m)
        gate = in_frame & (conf > self.conf_thresh) & reliable
        if self.refine_ref:
            # Where trusted, take the depth from DA3 (scale-corrected); elsewhere keep the head.
            refined = torch.exp(self.log_scale) * d_scene + self.shift + residual
            delta = (refined - wrist_z) * gate.float()
        else:
            delta = residual * gate.float()
        return delta, gate
