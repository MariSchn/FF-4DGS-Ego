from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_transformer import VisualGeometryTransformer
from ..heads.camera_head import CameraHead
from ..heads.dense_head import DPTHead
from ..heads.hamer_head import HamerManoHead
from ..heads.hand_to_gs_injection import HandToGSInjection
from .rasterization import GaussianSplatRenderer
from ..utils.camera_utils import vector_to_camera_matrices, extrinsics_to_vector
from ..utils.priors import normalize_depth, normalize_poses

from huggingface_hub import PyTorchModelHubMixin
from diffsynth.models.utils import hash_state_dict_keys


class WorldMirror(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=518,
                 patch_size=14,
                 embed_dim=1024,
                 gs_dim=256,
                 enable_cond=True,
                 enable_cam=True,
                 enable_pts=True,
                 enable_depth=True,
                 enable_norm=True,
                 enable_motion=True,
                 enable_gs=True,
                 enable_dynamic_gs_attr=True,
                 life_span_gamma=10.0,
                 dynamic_threshold=0.0,
                 enable_global_motion_tracking=False,
                 dynamic_threshold2=0.0,
                 occlusion_threshold=0.05,
                 bidirection=True,
                 patch_embed="dinov2_vitl14_reg",
                 fixed_patch_embed=False,
                 sampling_strategy="uniform",
                 dpt_gradient_checkpoint=False,
                 condition_strategy=["token", "pow3r", "token"],
                 **kwargs):

        super().__init__()
        # Configuration flags
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gs_dim = gs_dim
        self.enable_cam = enable_cam
        self.enable_pts = enable_pts
        self.enable_depth = enable_depth
        self.enable_cond = enable_cond
        self.enable_norm = enable_norm
        self.enable_motion = enable_motion
        self.enable_gs = enable_gs
        self.enable_dynamic_gs_attr = enable_dynamic_gs_attr
        self.enable_hand = kwargs.get("enable_hand", True)
        self.hand_head_type = kwargs.get("hand_head_type", "hamer")
        self.use_hand_crop = kwargs.get("use_hand_crop", False)
        self.hand_crop_size = kwargs.get("hand_crop_size", 8)
        self.hires_hand = kwargs.get("hires_hand", False)
        self.hires_hand_kwargs = kwargs.get("hires_hand_kwargs", {})
        self.hamer_head_kwargs = kwargs.get("hamer_head_kwargs", {})
        self.hand_to_gs_injection_cfg = kwargs.get("hand_to_gs_injection", {})
        self.enable_scale_head = kwargs.get("enable_scale_head", False)
        self.enable_root_anchor = kwargs.get("enable_root_anchor", False)
        self.root_anchor_kwargs = kwargs.get("root_anchor_kwargs", {})

        self.life_span_gamma = life_span_gamma
        self.dynamic_threshold = dynamic_threshold
        self.enable_global_motion_tracking = enable_global_motion_tracking
        self.dynamic_threshold2 = dynamic_threshold2
        self.occlusion_threshold = occlusion_threshold
        self.bidirection = bidirection
        self.patch_embed = patch_embed
        self.sampling = sampling_strategy
        self.dpt_checkpoint = dpt_gradient_checkpoint
        self.cond_methods = condition_strategy
        self.config = self._store_config()

        # Visual geometry transformer
        self.visual_geometry_transformer = VisualGeometryTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_cond=enable_cond,
            enable_motion=enable_motion,
            sampling_strategy=sampling_strategy,
            patch_embed=patch_embed,
            fixed_patch_embed=fixed_patch_embed,
            condition_strategy=condition_strategy
        )

        # freeze backbone
        self.freeze_backbone = kwargs.get("freeze_backbone", True)
        if self.freeze_backbone:
            for param in self.visual_geometry_transformer.parameters():
                param.requires_grad = False
        # Whether the backbone forward should build an autograd graph. Defaults to
        # the freeze flag, but the trainer flips this to True when it partially
        # unfreezes a few transformer blocks (freeze_backbone stays True for the
        # head-grouping logic, yet the unfrozen blocks still need gradients).
        self._backbone_trainable = not self.freeze_backbone

        # Initialize prediction heads
        self._init_heads(embed_dim, patch_size, gs_dim)

    def _store_config(self):
        """Save the model configuration"""
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "gs_dim": self.gs_dim,
            "enable_cam": self.enable_cam,
            "enable_pts": self.enable_pts,
            "enable_depth": self.enable_depth,
            "enable_norm": self.enable_norm,
            "enable_gs": self.enable_gs,
            "enable_hand": self.enable_hand,
            "enable_root_anchor": getattr(self, "enable_root_anchor", False),
            "hand_head_type": self.hand_head_type,
            "use_hand_crop": self.use_hand_crop,
            "hand_crop_size": self.hand_crop_size,
            "hires_hand": self.hires_hand,
            "patch_embed": self.patch_embed,
            "sampling_strategy": self.sampling,
            "dpt_checkpoint": self.dpt_checkpoint,
            "condition_strategy": self.cond_methods,
        }

    def _init_heads(self, dim, patch_size, gs_dim):
        """Initialize all prediction heads"""

        # Camera pose prediction head
        if self.enable_cam:
            self.cam_head = CameraHead(dim_in=2 * dim)

        # Feedforward global metric-scale head (the "scale head" route). Predicts
        # one positive scale per clip from the register token; trained to match the
        # metric hand (scripts.scale_head_loss). Optional + off by default so it
        # does not affect other experiments / older checkpoints.
        if getattr(self, "enable_scale_head", False):
            from ..heads.scale_head import ScaleHead
            self.scale_head = ScaleHead(dim_in=2 * dim)

        # 3D point prediction head
        if self.enable_pts:
            self.pts_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1"
            )

        # Depth prediction head
        if self.enable_depth:
            self.depth_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2,
                patch_size=patch_size,
                activation="exp+expp1",
            )

        # Surface normal prediction head
        if self.enable_norm:
            self.norm_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="norm+expp1",
            )

        # Velocity prediction heads
        if self.enable_motion:
            self.velocity_fwd_head = DPTHead(
                dim_in=dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1",
            )
            self.velocity_bwd_head = DPTHead(
                dim_in=dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1",
            )

        # Gaussian splatting feature head and renderer
        if self.enable_gs:
            self.gs_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2,
                patch_size=patch_size,
                features=gs_dim,
                is_gsdpt=True,
                activation="exp+expp1"
            )
            self.gs_renderer = GaussianSplatRenderer(
                sh_degree=0,
                enable_prune=True,
                voxel_size=0.002,
                is_4dgs=self.enable_dynamic_gs_attr,
                life_span_gamma=self.life_span_gamma,
                dynamic_threshold=self.dynamic_threshold,
                global_motion_tracking=self.enable_global_motion_tracking,
                dynamic_threshold2=self.dynamic_threshold2,
                occlusion_threshold=self.occlusion_threshold,
                bidirection=self.bidirection,
            )
            # Dynamic Gaussian splatting attribute heads
            if self.enable_dynamic_gs_attr:
                self.gs_fwd_attr_head = DPTHead(
                    dim_in=dim,
                    output_dim=3,
                    patch_size=patch_size,
                    activation="rotation+none", # use 'none' to disable confidence prediction
                )
                self.gs_bwd_attr_head = DPTHead(
                    dim_in=dim,
                    output_dim=3,
                    patch_size=patch_size,
                    activation="rotation+none", # use 'none' to disable confidence prediction
                )

        # hand tracking head
        if self.enable_hand:
            if self.hand_head_type == "hamer":
                # Backbone-swap ablation: hamer_head_kwargs may carry its own
                # context_dim (e.g. 1024 for DINOv2-L tokens). Pop it from a copy
                # so it doesn't collide with the explicit kwarg below.
                _hh_kwargs = dict(self.hamer_head_kwargs)
                self.hand_head = HamerManoHead(
                    context_dim=_hh_kwargs.pop("context_dim", 2 * dim),
                    use_crop=self.use_hand_crop,
                    crop_size=self.hand_crop_size,
                    patch_size=patch_size,
                    hires_hand=self.hires_hand,
                    hires_hand_kwargs=self.hires_hand_kwargs,
                    **_hh_kwargs,
                )
            elif self.hand_head_type == "dpt":
                self.hand_head = DPTHead(
                    dim_in=2 * dim,
                    output_dim=64,       # 2 hands * (3 t_xyz + 4 q_wxyz + 15 pose + 10 betas)
                    patch_size=patch_size,
                    activation="linear+none",
                )
            else:
                raise ValueError(f"Unknown hand_head_type: {self.hand_head_type}")

        # Optional: inject hamer's enhanced local crop features into the GS branch.
        # Only meaningful when both heads exist and hamer's use_crop path is on,
        # so the dict carries enhanced_crop_tokens.
        if (
            self.enable_hand
            and self.enable_gs
            and self.hand_head_type == "hamer"
            and self.hand_to_gs_injection_cfg.get("enabled", False)
        ):
            cfg = dict(self.hand_to_gs_injection_cfg)
            cfg.pop("enabled", None)
            hand_dim = cfg.pop("hand_dim", self.hamer_head_kwargs.get("dim", 1024))
            # gs_dims must match DPTHead.out_channels per layer (post-projects,
            # pre-fusion). The default mirrors DPTHead's [256, 512, 1024, 1024].
            gs_dims = cfg.pop("gs_dims", [256, 512, 1024, 1024])
            self.hand_to_gs_injection = HandToGSInjection(
                hand_dim=hand_dim,
                gs_dims=gs_dims,
                **cfg,
            )

        # Optional post-hoc root-depth anchor (contact Phase 1). Owned here so it
        # trains with the head and is saved in the model state dict. Applied
        # outside the forward (after gs_depth exists) by the train loop / eval.
        if getattr(self, "enable_root_anchor", False):
            from ..heads.root_depth_refine import RootDepthRefine
            self.root_depth_refine = RootDepthRefine(**(self.root_anchor_kwargs or {}))

    def forward(self, views: Dict[str, torch.Tensor], cond_flags: List[int]=[0, 0, 0], is_inference=True, use_motion=True):
        """
        Execute forward pass through the WorldMirror model.

        Args:
            views: Input data dictionary
            cond_flags: Conditioning flags [depth, rays, camera]

        Returns:
            dict: Prediction results dictionary
        """
        imgs = views['img']

        # Enable conditional input during training if enabled, or during inference if any cond_flags are set
        use_cond = sum(cond_flags) > 0
        if (imgs.shape[1] == 1):
            use_motion = False

        # Extract priors and process features based on conditional input
        backbone_ctx = torch.enable_grad() if getattr(self, "_backbone_trainable", not self.freeze_backbone) else torch.no_grad()
        with backbone_ctx:
            if use_cond:
                priors = self.extract_priors(views)
                token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.visual_geometry_transformer(
                    imgs, priors, cond_flags=cond_flags, use_motion=(use_motion and is_inference)
                )
            else:
                token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.visual_geometry_transformer(imgs, use_motion=(use_motion and is_inference))

        # Generate all predictions
        preds = self._gen_all_preds(
            token_list, imgs, patch_start_idx, views, cond_flags, is_inference, use_motion,
            fwd_token_list, bwd_token_list, 
        )

        for key, value in preds.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                preds[key] = value.to(torch.float32)
            elif isinstance(value, list):
                for batch_value in value:
                    for frame_value in batch_value:
                        frame_value.to(torch.float32)
        return preds

    def _gen_all_preds(self, token_list, imgs, patch_start_idx,
                        views, cond_flags, is_inference, use_motion,
                       fwd_token_list=[], bwd_token_list=[]):
        """Generate all enabled predictions"""
        preds = {}

        # Camera pose prediction
        if self.enable_cam:
            cam_seq = self.cam_head(token_list)
            cam_params = cam_seq[-1]
            preds["camera_params"] = cam_params
            c2w_mat, int_mat = self.transform_camera_vector(cam_params, imgs.shape[-2], imgs.shape[-1])
            preds["camera_poses"] = c2w_mat  # C2W pose (OpenCV) in world coordinates: [B, S, 4, 4]
            preds["camera_intrs"] = int_mat  # Camera intrinsic matrix: [B, S, 3, 3]

        # Feedforward global metric scale (the scale-head route); [B] positive scalar.
        if getattr(self, "enable_scale_head", False):
            preds["pred_scale"] = self.scale_head(token_list)

        # Depth prediction
        if self.enable_depth:
            depth, depth_conf = self.depth_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["depth"] = depth
            preds["depth_conf"] = depth_conf

        # 3D point prediction
        if self.enable_pts:
            pts, pts_conf = self.pts_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["pts3d"] = pts
            preds["pts3d_conf"] = pts_conf

        # tracking hand
        enhanced_crop_tokens = None
        if self.enable_hand:
            if self.hand_head_type == "hamer":
                hamer_kwargs = {}
                if self.use_hand_crop:
                    hamer_kwargs["hand_bboxes"] = views.get("hand_bboxes")
                    hamer_kwargs["hand_valid"] = views.get("hand_valid")
                if self.hires_hand:
                    hamer_kwargs["hand_crops"] = views.get("hand_crops")
                hand_out = self.hand_head(
                    token_list,
                    images=imgs,
                    patch_start_idx=patch_start_idx,
                    **hamer_kwargs,
                )
                preds["hand_joints"] = hand_out["params"]
                preds["hand_conf"] = hand_out["conf"]
                enhanced_crop_tokens = hand_out.get("enhanced_crop_tokens")
            elif self.hand_head_type == "hand_crop":
                hand_bboxes = views.get("hand_bboxes", None)
                hand_valid = views.get("hand_valid", None)
                hand_joints = self.hand_head(
                    token_list,
                    images=imgs,
                    patch_start_idx=patch_start_idx,
                    hand_bboxes=hand_bboxes,
                    hand_valid=hand_valid,
                    focal_length=views.get("focal_length", None),
                    crop_local_output=views.get("crop_local_output", False),
                )
                preds["hand_joints"] = hand_joints
            elif self.hand_head_type == "dpt":
                hand_joints, hand_conf = self.hand_head(
                    token_list,
                    images=imgs,
                    patch_start_idx=patch_start_idx,
                )
                hand_joints = hand_joints.mean(dim=(2, 3))
                preds["hand_joints"] = hand_joints
                preds["hand_conf"] = hand_conf
            
        # Normal prediction
        if self.enable_norm:
            normals, norm_conf = self.norm_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["normals"] = normals
            preds["normals_conf"] = norm_conf

        # Prepare context predictions for motion and GS heads
        if self.enable_motion or self.enable_gs:
            context_preds = self.prepare_contexts(views, cond_flags, is_inference, use_motion)
        else:
            context_preds = {}

        fwd_token_list = context_preds.get("fwd_token_list", fwd_token_list)
        bwd_token_list = context_preds.get("bwd_token_list", bwd_token_list)

        # Velocity prediction
        if self.enable_motion and use_motion:
            assert len(fwd_token_list) > 0 and len(bwd_token_list) > 0
            vel_fwd, vel_fwd_conf = self.velocity_fwd_head(
                fwd_token_list,
                images=context_preds.get("imgs", imgs)[:, :-1],
                patch_start_idx=patch_start_idx
            )
            vel_bwd, vel_bwd_conf = self.velocity_bwd_head(
                bwd_token_list,
                images=context_preds.get("imgs", imgs)[:, 1:],
                patch_start_idx=patch_start_idx
            )
            preds["velocity_fwd"] = vel_fwd
            preds["velocity_fwd_conf"] = vel_fwd_conf
            preds["velocity_bwd"] = vel_bwd
            preds["velocity_bwd_conf"] = vel_bwd_conf

        # 3D Gaussian Splatting
        if self.enable_gs:
            # Build a hand-injection hook that runs inside dense_head, on the
            # per-layer projected features (post-projects, pre-fusion). This
            # gives the DPT fusion blocks visibility into hand-aware features
            # at every scale, instead of only patching the final gs_feat.
            #
            # dense_head may chunk frames internally for memory; the hook is
            # called per chunk with (feats, frame_start, frame_end) over the
            # original [0, S) range, so we slice all per-frame inputs to match
            # the current chunk before invoking the injection module.
            gs_feature_hook = None
            if (
                getattr(self, "hand_to_gs_injection", None) is not None
                and self.hand_head_type == "hamer"
                and enhanced_crop_tokens is not None
                and views.get("hand_bboxes") is not None
            ):
                hand_bboxes = views.get("hand_bboxes")
                hand_valid = views.get("hand_valid")
                B_h, S_h, _, _ = hand_bboxes.shape
                _, K_h, D_h = enhanced_crop_tokens.shape
                # [B*S*2, K, D] -> [B, S, 2, K, D] for per-frame slicing.
                ect_bs = enhanced_crop_tokens.reshape(B_h, S_h, 2, K_h, D_h)

                def _gs_feature_hook(feats, fs, fe):
                    ect_chunk = ect_bs[:, fs:fe].reshape(-1, K_h, D_h)
                    bb_chunk = hand_bboxes[:, fs:fe]
                    hv_chunk = hand_valid[:, fs:fe] if hand_valid is not None else None
                    return self.hand_to_gs_injection(ect_chunk, bb_chunk, hv_chunk, feats)

                gs_feature_hook = _gs_feature_hook

            gs_feat, gs_depth, gs_depth_conf = self.gs_head(
                context_preds.get("token_list", token_list),
                images=context_preds.get("imgs", imgs),
                patch_start_idx=patch_start_idx,
                feature_hook=gs_feature_hook,
            )
            preds["gs_depth"] = gs_depth
            preds["gs_depth_conf"] = gs_depth_conf

            # Fast path for the L1 metric anchor / L2 scale eval: gs_depth (above)
            # is all they need. Skip the expensive splat build + voxel prune +
            # rasterization. Off by default; inference always renders.
            if not is_inference and getattr(self, "gs_anchor_only", False):
                return preds

            # Dynamic GS attributes
            if self.enable_dynamic_gs_attr and use_motion:
                assert len(fwd_token_list) > 0 and len(bwd_token_list) > 0
                gs_fwd_attr, _ = self.gs_fwd_attr_head(
                    fwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, :-1],
                    patch_start_idx=patch_start_idx
                )
                gs_bwd_attr, _ = self.gs_bwd_attr_head(
                    bwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, 1:],
                    patch_start_idx=patch_start_idx
                )
                preds["gs_fwd_attr"] = gs_fwd_attr
                preds["gs_bwd_attr"] = gs_bwd_attr

            preds = self.gs_renderer.render(
                gs_feats=gs_feat,
                images=imgs,
                predictions=preds,
                views=views,
                context_predictions=context_preds,
                is_inference=is_inference
            )
        return preds

    def extract_priors(self, views):
        """
        Extract and normalize geometric priors.

        Args:
            views: Input view data dictionary.

        Returns:
            tuple: (depths, rays, poses) Normalized priors.
        """
        h, w = views['img'].shape[-2:]

        # Initialize prior variables
        depths = rays = poses = None

        # Extract camera pose
        if 'camera_poses' in views:
            extrinsics = views['camera_poses'][:, :, :3]
            extrinsics = normalize_poses(extrinsics)
            cam_params = extrinsics_to_vector(extrinsics)
            poses = cam_params[:, :, :7]  # Shape: [B, S, 7]

        # Extract depth map
        if 'depthmap' in views:
            depth_h, depth_w = views['depthmap'].shape[-2:]
            depths = views['depthmap']
            if depth_h != h or depth_w != w:  # Check if depth dimensions match target resolution
                try:
                    depths = F.interpolate(depths, size=(h, w), mode='bilinear', align_corners=False)
                except:
                    import pdb; pdb.set_trace()
            depths = normalize_depth(depths)  # Shape: [B, S, H, W]

        # Extract ray directions
        if 'camera_intrs' in views:
            intrinsics = views['camera_intrs'][:, :, :3, :3]
            fx, fy = intrinsics[:, :, 0, 0] / w, intrinsics[:, :, 1, 1] / h
            cx, cy = intrinsics[:, :, 0, 2] / w, intrinsics[:, :, 1, 2] / h
            rays = torch.stack([fx, fy, cx, cy], dim=-1)  # Shape: [B, S, 4]

        return (depths, rays, poses)

    def transform_camera_vector(self, camera_params, h, w):
        ext_mat, int_mat = vector_to_camera_matrices(
            camera_params, image_hw=(h, w)
        )
        # Create homogeneous transformation matrix
        homo_row = torch.tensor([0, 0, 0, 1], device=ext_mat.device).view(1, 1, 1, 4)
        homo_row = homo_row.repeat(ext_mat.shape[0], ext_mat.shape[1], 1, 1)
        w2c_mat = torch.cat([ext_mat, homo_row], dim=2)
        c2w_mat = torch.linalg.inv(w2c_mat)
        return c2w_mat, int_mat

    def prepare_contexts(self, views, cond_flags, is_inference, use_motion):
        # Generate context views predictions
        context_preds = {}
        # only for training or evaluation
        if is_inference:
            return context_preds

        assert self.enable_cam and (self.enable_motion or self.enable_gs)
        if 'is_target' not in views:
            context_nums = views['img'].shape[1]
        else:
            context_nums = (views['is_target'][0] == False).sum().item()
        context_imgs = views['img'][:, :context_nums]

        use_cond = sum(cond_flags) > 0

        # Extract context priors and process features based on context views
        if use_cond:
            priors = self.extract_priors(views)
            context_priors = (prior[:, :context_nums] if prior is not None else None for prior in priors)
            context_token_list, _, context_fwd_token_list, context_bwd_token_list = self.visual_geometry_transformer(
                context_imgs, context_priors, cond_flags=cond_flags, use_motion=use_motion
            )
        else:
            context_token_list, _, context_fwd_token_list, context_bwd_token_list = self.visual_geometry_transformer(context_imgs, use_motion=use_motion)

        # Execute predictions
        # Context camera pose prediction
        context_cam_seq = self.cam_head(context_token_list)
        context_cam_params = context_cam_seq[-1]
        context_c2w_mat, context_int_mat = self.transform_camera_vector(context_cam_params, context_imgs.shape[-2], context_imgs.shape[-1])
        context_preds['camera_poses'] = context_c2w_mat  # C2W pose (OpenCV) in world coordinates: [B, S, 4, 4]
        context_preds['camera_intrs'] = context_int_mat  # Camera intrinsic matrix: [B, S, 3, 3]
        context_preds['token_list'] = context_token_list
        context_preds['imgs'] = context_imgs
        context_preds['fwd_token_list'] = context_fwd_token_list
        context_preds['bwd_token_list'] = context_bwd_token_list

        return context_preds

    @staticmethod
    def state_dict_converter():
        return ModelDictConverter()


class ModelDictConverter:
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == '1a1d001a35f78f3a7796a1e719ead340':
            config = {
                "enable_norm": False,
                "strict_load": True,
                # "upcast_to_float32": True,
            }
        else:
            config = {}
        return state_dict, config
