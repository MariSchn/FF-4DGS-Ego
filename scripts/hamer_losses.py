"""
Losses for HandCropHead.

Adapted from HaMeR (https://github.com/geopavlakos/hamer).
"""

import torch
import torch.nn as nn


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        batch_size = pred_keypoints_2d.shape[0]
        conf = gt_keypoints_2d[:, :, :, -1].unsqueeze(-1).clone()  # [B, S, N, 1]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :, :-1])).sum(dim=(1, 2, 3))
        return loss.sum() / batch_size


class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.

        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f'Unsupported loss type: {loss_type!r}. Choose from "l1" or "l2".')

    def forward(
        self,
        pred_keypoints_3d: torch.Tensor,
        gt_keypoints_3d: torch.Tensor,
        pelvis_id: int = 0,
    ) -> torch.Tensor:
        """
        Compute 3D keypoint loss (pelvis-relative).

        Both prediction and ground truth are root-centred using the pelvis
        joint before the loss is computed, making it invariant to global
        translation.

        Args:
            pred_keypoints_3d (torch.Tensor): Shape [B, S, N, 3] — predicted
                3D keypoints from HandCropHead (B: batch, S: sequence/frames,
                N: num keypoints).
            gt_keypoints_3d (torch.Tensor): Shape [B, S, N, 4] — ground truth
                3D keypoints with per-keypoint confidence in the last channel.
            pelvis_id (int): Index of the pelvis / root joint used for centring.

        Returns:
            torch.Tensor: Scalar 3D keypoint loss.
        """
        gt_keypoints_3d = gt_keypoints_3d.clone()
        batch_size = pred_keypoints_3d.shape[0]

        # Root-centre both prediction and ground truth
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, :, pelvis_id, :].unsqueeze(2)
        gt_keypoints_3d[:, :, :, :-1] = (
            gt_keypoints_3d[:, :, :, :-1]
            - gt_keypoints_3d[:, :, pelvis_id, :-1].unsqueeze(2)
        )

        # conf: [B, S, N, 1]
        conf = gt_keypoints_3d[:, :, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :, :-1]

        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1, 2, 3))
        return loss.sum() / batch_size


HAND_PARAM_DIM = 32
NUM_HANDS = 2

# Per-hand slice offsets inside the 32-D layout
# [transl(3), quat_wxyz(4), hand_pose_pca(15), betas(10)]
_TRANSL_SLICE = slice(0, 3)
_QUAT_SLICE   = slice(3, 7)
_POSE_SLICE   = slice(7, 22)
_BETAS_SLICE  = slice(22, 32)


def _safe_quat_to_rotmat(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] quaternions (w, x, y, z) to [..., 3, 3] rotation matrices.

    Zero-quaternions (absent hands) are replaced with identity before the
    conversion so there's no NaN from dividing by zero norm. The caller is
    expected to mask the resulting loss so identity-filler doesn't contribute.
    """
    norm = q_wxyz.norm(dim=-1, keepdim=True)
    zero_mask = norm < 1e-8
    identity_q = torch.zeros_like(q_wxyz)
    identity_q[..., 0] = 1.0
    q = torch.where(zero_mask, identity_q, q_wxyz)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    r = torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - wz),     2 * (wy + xz),
        2 * (xy + wz),     w2 - x2 + y2 - z2, 2 * (yz - wx),
        2 * (xz - wy),     2 * (wx + yz),     w2 - x2 - y2 + z2,
    ], dim=-1)
    return r.view(*q.shape[:-1], 3, 3)


class ParameterLoss(nn.Module):
    def __init__(self):
        """MANO parameter loss — split into (transl, global_orient, hand_pose, betas).

        Mirrors HaMeR's convention (models/hamer/hamer/models/hamer.py:174-184):
        a separate masked-MSE is computed per MANO parameter key so that the
        caller can weight them independently. Global orient is compared in
        rotation-matrix space (via quat→rotmat) to remove the antipodal
        ambiguity of a raw quaternion MSE and to match HaMeR's aa_to_rotmat
        formulation.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    @staticmethod
    def _masked_mean(err: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(mask * err) averaged over non-batch dims, summed over batch.

        Matches HaMeR's existing ParameterLoss reduction: a sum over the
        batch of per-sample means so gradient magnitude is proportional to
        (approximately) the number of valid hands in the batch.
        """
        per_sample = (mask * err).reshape(err.shape[0], -1).mean(dim=-1)
        return per_sample.sum()

    def forward(
        self,
        pred_param: torch.Tensor,
        gt_param: torch.Tensor,
        has_param: torch.Tensor,
    ) -> dict:
        """
        Args:
            pred_param: [B, S, 64] — two hands packed as (left[32], right[32]) of
                (transl[3], quat_wxyz[4], pose_pca[15], betas[10]).
            gt_param:   [B, S, 64] in the same layout.
            has_param:  [B, S, 2] — 1.0 where the hand is present, else 0.0.

        Returns:
            dict with scalar entries 'transl', 'global_orient', 'hand_pose',
            'betas'. Summed over the batch (HaMeR convention); caller applies
            per-key weights.
        """
        B, S, D = pred_param.shape
        assert D == NUM_HANDS * HAND_PARAM_DIM
        pred = pred_param.view(B, S, NUM_HANDS, HAND_PARAM_DIM)
        gt   = gt_param.view(B, S, NUM_HANDS, HAND_PARAM_DIM)
        dtype = pred.dtype

        # [B, S, 2, 1] for per-slice broadcasting
        mask = has_param.to(dtype).view(B, S, NUM_HANDS, 1)

        # --- translation (3) ---
        transl_err = self.loss_fn(pred[..., _TRANSL_SLICE], gt[..., _TRANSL_SLICE])
        loss_transl = self._masked_mean(transl_err, mask)

        # --- global orient (4-d quaternion → 3x3 rotmat MSE) ---
        pred_R = _safe_quat_to_rotmat(pred[..., _QUAT_SLICE])  # [B, S, 2, 3, 3]
        gt_R   = _safe_quat_to_rotmat(gt[..., _QUAT_SLICE])
        orient_err = self.loss_fn(pred_R, gt_R).flatten(-2)    # [B, S, 2, 9]
        loss_orient = self._masked_mean(orient_err, mask)

        # --- hand pose (15-D PCA) ---
        pose_err = self.loss_fn(pred[..., _POSE_SLICE], gt[..., _POSE_SLICE])
        loss_pose = self._masked_mean(pose_err, mask)

        # --- betas (10) ---
        betas_err = self.loss_fn(pred[..., _BETAS_SLICE], gt[..., _BETAS_SLICE])
        loss_betas = self._masked_mean(betas_err, mask)

        return {
            "transl":        loss_transl,
            "global_orient": loss_orient,
            "hand_pose":     loss_pose,
            "betas":         loss_betas,
        }
