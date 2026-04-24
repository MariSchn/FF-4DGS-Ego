"""3D self-supervised consistency loss between the MANO hand head and the Gaussian head.

Computes MANO vertices/joints from the hand head's predicted (betas, pose,
global orient, translation), selects hand-region Gaussian centers via the
refiner's bbox-derived mask, and pulls each selected center toward its
nearest mesh vertex / joint with an L2² loss.
"""

import os

import smplx
import torch
import torch.nn as nn

HAND_PARAM_DIM = 32  # per hand: t_xyz(3) + q_wxyz(4) + pose_pca(15) + betas(10)

# Module-level flag so the diagnostic line prints once per epoch. The training
# loop calls reset_diag_flag() at the top of each epoch.
_DIAG_PRINTED_THIS_EPOCH = False


def reset_diag_flag():
    global _DIAG_PRINTED_THIS_EPOCH
    _DIAG_PRINTED_THIS_EPOCH = False


def quat_wxyz_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Differentiable wxyz quaternion -> axis-angle rotation vector.

    q: [..., 4] ordered (w, x, y, z). Returns rotvec [..., 3].
    """
    eps = 1e-8
    # Safe norm: adding eps under sqrt avoids NaN gradients at zero quaternions.
    q_norm = (q.pow(2).sum(dim=-1, keepdim=True) + eps).sqrt()
    q = q / q_norm.clamp(min=eps)
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    # Flip to the hemisphere where w >= 0 so atan2 stays in [0, pi].
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    w = w * sign
    xyz = xyz * sign
    sin_half = (xyz.pow(2).sum(dim=-1, keepdim=True) + eps).sqrt()
    # Keep w strictly inside (-1, 1) — guards any downstream acos/sqrt from
    # hitting the boundary when the head emits a near-unit quaternion.
    w_safe = w.clamp(min=-1.0 + eps, max=1.0 - eps)
    angle = 2.0 * torch.atan2(sin_half, w_safe)
    # Small-angle series: rotvec ≈ 2 * xyz
    small = sin_half < 1e-6
    factor = torch.where(small, 2.0 * torch.ones_like(sin_half), angle / sin_half.clamp(min=eps))
    return xyz * factor


class HandGSConsistencyLoss(nn.Module):
    """L2² distance from hand-region Gaussian centers to predicted MANO surface/joints."""

    def __init__(
        self,
        mano_folder: str,
        target: str = "vertices",
        mask_threshold: float = 0.5,
        max_samples_per_frame: int = 256,
    ):
        super().__init__()
        assert target in ("vertices", "joints"), f"Unknown target: {target}"
        self.target = target
        self.mask_threshold = mask_threshold
        self.max_samples_per_frame = max_samples_per_frame

        self.mano_left = smplx.create(
            os.path.join(mano_folder, "MANO_LEFT.pkl"),
            "mano", use_pca=True, is_rhand=False, num_pca_comps=15,
        )
        self.mano_right = smplx.create(
            os.path.join(mano_folder, "MANO_RIGHT.pkl"),
            "mano", use_pca=True, is_rhand=True, num_pca_comps=15,
        )
        # smplx left-hand shapedirs fix (https://github.com/vchoutas/smplx/issues/48).
        if torch.sum(torch.abs(
            self.mano_left.shapedirs[:, 0, :] - self.mano_right.shapedirs[:, 0, :]
        )) < 1:
            self.mano_left.shapedirs[:, 0, :] *= -1

        for p in self.mano_left.parameters():
            p.requires_grad = False
        for p in self.mano_right.parameters():
            p.requires_grad = False

    def _mano_points(self, params_flat: torch.Tensor, mano_layer) -> torch.Tensor:
        """params_flat: [N, 32]. Returns points [N, P, 3] in the same frame as transl."""
        transl = params_flat[..., :3]
        q_wxyz = params_flat[..., 3:7]
        hand_pose = params_flat[..., 7:22]
        betas = params_flat[..., 22:32]
        global_orient = quat_wxyz_to_rotvec(q_wxyz)
        out = mano_layer(
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=transl,
            return_verts=True,
        )
        return out.vertices if self.target == "vertices" else out.joints

    def forward(self, preds: dict, views: dict | None = None):
        """Returns a scalar consistency loss (zero if predictions are missing)."""
        global _DIAG_PRINTED_THIS_EPOCH
        if "hand_joints" not in preds or "gs_means" not in preds:
            # Cannot compute without both heads' outputs.
            ref = preds.get("hand_joints") or preds.get("gs_means")
            return ref.new_zeros(()) if isinstance(ref, torch.Tensor) else torch.zeros(())

        hand_params = preds["hand_joints"]  # [B, S, 64]
        B, S = hand_params.shape[:2]
        hand_params = hand_params.reshape(B, S, 2, HAND_PARAM_DIM)

        gs_means = preds["gs_means"]  # [B, S, HW, 3]
        HW = gs_means.shape[2]
        BS = B * S

        # Hand-region mask: prefer the refiner's mask (post-sigmoid/clipped), else
        # derive from bboxes in views.
        if "hand_gs_mask" in preds:
            mask = preds["hand_gs_mask"].reshape(B, S, HW)
        elif views is not None and "hand_bboxes" in views:
            H = W = int(HW ** 0.5)
            mask = _bbox_mask(views["hand_bboxes"][:, :S], views.get("hand_valid", None),
                              B, S, H, W, gs_means.device, gs_means.dtype).reshape(B, S, HW)
        else:
            return gs_means.new_zeros(())

        hand_valid = None
        if views is not None:
            hand_valid = views.get("hand_valid", None)
        if hand_valid is not None:
            hand_valid = hand_valid[:, :S].reshape(BS, 2).bool()
        else:
            hand_valid = torch.ones(BS, 2, dtype=torch.bool, device=gs_means.device)

        # MANO forward per hand side.
        params_flat = hand_params.reshape(BS, 2, HAND_PARAM_DIM)
        left_pts = self._mano_points(params_flat[:, 0], self.mano_left)   # [BS, P, 3]
        right_pts = self._mano_points(params_flat[:, 1], self.mano_right)  # [BS, P, 3]

        # Mark invalid hands' points as far-away so nearest-neighbor skips them.
        INF_DIST = 1e6
        left_pts = torch.where(
            hand_valid[:, 0:1, None], left_pts, torch.full_like(left_pts, INF_DIST)
        )
        right_pts = torch.where(
            hand_valid[:, 1:2, None], right_pts, torch.full_like(right_pts, INF_DIST)
        )
        mesh_pts = torch.cat([left_pts, right_pts], dim=1)  # [BS, 2P, 3]

        # Pick the top-K hand pixels per frame to keep memory bounded.
        K = min(self.max_samples_per_frame, HW)
        mask_flat = mask.reshape(BS, HW)
        means_flat = gs_means.reshape(BS, HW, 3)

        topk = torch.topk(mask_flat, k=K, dim=-1)
        sel_idx = topk.indices                          # [BS, K]
        sel_valid = (topk.values > self.mask_threshold)  # [BS, K]
        sel_valid = sel_valid & hand_valid.any(dim=-1, keepdim=True)  # drop frames with no valid hand

        # Gather selected Gaussian centers.
        batch_idx = torch.arange(BS, device=gs_means.device)[:, None].expand(-1, K)
        sampled = means_flat[batch_idx, sel_idx]  # [BS, K, 3]

        if not _DIAG_PRINTED_THIS_EPOCH:
            s, m = sampled.detach().float(), mesh_pts.detach().float()
            s_finite = s[s.abs() < 1e5]  # drop the INF_DIST fill
            m_finite = m[m.abs() < 1e5]
            print(
                f"[DEBUG SCALE] Gaussians (sampled) - "
                f"min: {s_finite.min().item():.3f}, max: {s_finite.max().item():.3f}, "
                f"mean: {s_finite.mean().item():.3f} | "
                f"per-axis mean: "
                f"x={s[..., 0].mean().item():.3f}, y={s[..., 1].mean().item():.3f}, z={s[..., 2].mean().item():.3f}"
            )
            print(
                f"[DEBUG SCALE] MANO (mesh_pts) - "
                f"min: {m_finite.min().item():.3f}, max: {m_finite.max().item():.3f}, "
                f"mean: {m_finite.mean().item():.3f} | "
                f"per-axis mean: "
                f"x={m_finite[..., 0].mean().item():.3f}, y={m_finite[..., 1].mean().item():.3f}, z={m_finite[..., 2].mean().item():.3f}"
            )

        # Nearest mesh-point distance for each sample.
        d2 = torch.cdist(sampled, mesh_pts).pow(2)      # [BS, K, 2P]
        nn_d2, _ = d2.min(dim=-1)                       # [BS, K]

        # Mean over valid samples (K) so the scale is independent of how many
        # Gaussians we sample per frame — keeps the loss a per-sample mean
        # rather than a sum.
        w = sel_valid.to(nn_d2.dtype)
        num_samples = w.sum().clamp(min=1.0)
        loss = (nn_d2 * w).sum() / num_samples

        # Approximate RMS distance in cm — follows the user's formula literally
        # (sqrt of mean squared distance, * 100 to convert m -> cm).
        dist_cm = torch.sqrt(nn_d2.detach().mean().clamp(min=0.0)) * 100.0

        if not _DIAG_PRINTED_THIS_EPOCH:
            print(
                f"[DIAG] Raw Consistency Loss: {loss.item():.6f} | "
                f"Approx Dist: {dist_cm.item():.2f} cm"
            )
            _DIAG_PRINTED_THIS_EPOCH = True

        return loss


def _bbox_mask(hand_bboxes, hand_valid, B, S, H, W, device, dtype):
    ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
    xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x1 = hand_bboxes[..., 0, None, None]
    y1 = hand_bboxes[..., 1, None, None]
    x2 = hand_bboxes[..., 2, None, None]
    y2 = hand_bboxes[..., 3, None, None]
    inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)  # [B, S, 2, H, W]

    if hand_valid is not None:
        inside = inside & hand_valid[..., None, None].bool()

    return inside.any(dim=2).to(dtype)  # [B, S, H, W]
