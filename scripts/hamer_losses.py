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


class ParameterLoss(nn.Module):
    def __init__(self):
        """
        MANO parameter loss module (pose / global orientation / betas).

        Uses MSE and a binary per-hand mask so absent hands don't contribute.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred_param: torch.Tensor,
        gt_param: torch.Tensor,
        has_param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss over predicted MANO parameters.

        Args:
            pred_param (torch.Tensor): [B, S, 64] — two hands packed as
                (left[32], right[32]) of (pos, quat, pose_pca, betas).
            gt_param (torch.Tensor):   [B, S, 64] ground truth in the same layout.
            has_param (torch.Tensor):  [B, S, 2] — 1.0 where the hand is present,
                0.0 otherwise.
        Returns:
            torch.Tensor: Scalar parameter loss.
        """
        B, S, D = pred_param.shape
        assert D == NUM_HANDS * HAND_PARAM_DIM
        pred = pred_param.view(B, S, NUM_HANDS, HAND_PARAM_DIM)
        gt   = gt_param.view(B, S, NUM_HANDS, HAND_PARAM_DIM)
        mask = has_param.to(pred.dtype).view(B, S, NUM_HANDS, 1)

        loss = (mask * self.loss_fn(pred, gt)).reshape(B, -1).mean(dim=-1)
        return loss.sum()