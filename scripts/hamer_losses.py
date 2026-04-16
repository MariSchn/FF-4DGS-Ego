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
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
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
        conf = gt_keypoints_2d[:, :, :, -1].unsqueeze(-1).clone()  # [B, S, N, 1]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :, :-1])).sum(dim=(2, 3))
        return loss.sum()

        
class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.

        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
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

        # Root-centre both prediction and ground truth
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, :, pelvis_id, :].unsqueeze(2)
        gt_keypoints_3d[:, :, :, :-1] = (
            gt_keypoints_3d[:, :, :, :-1]
            - gt_keypoints_3d[:, :, pelvis_id, :-1].unsqueeze(2)
        )

        # conf: [B, S, N, 1]
        conf = gt_keypoints_3d[:, :, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :, :-1]

        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(2, 3))
        return loss.sum()


class ParameterLoss(nn.Module):
    def __init__(self):
        """
        MANO parameter loss module (pose / global orientation / betas).

        Uses MSE and a binary mask to ignore examples where the ground truth
        annotation is unavailable.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        pred_param: torch.Tensor,
        gt_param: torch.Tensor,
        # has_param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss over MANO parameters predicted by HandCropHead.

        Args:
            pred_param (torch.Tensor): Shape [B, S, ...] — predicted MANO
                parameters (position, quaternion, pose, betas).
            gt_param (torch.Tensor): Shape [B, S, ...] — ground truth MANO
                parameters.
        Returns:
            torch.Tensor: Scalar parameter loss.
        """

        # has_param (torch.Tensor): Shape [B, S] — binary mask; 1 if the            
        # sample has a valid ground truth annotation, 0 otherwise.
        has_param = (gt_param.abs().sum(dim=-1) > 0).float() # [B, S]
        
        num_dims = len(pred_param.shape)

        # Broadcast mask over all parameter dimensions beyond [B, S]
        mask = has_param.type(pred_param.dtype).view(
            *has_param.shape, *([1] * (num_dims - len(has_param.shape)))
        )

        loss = mask * self.loss_fn(pred_param, gt_param)
        return loss.sum()