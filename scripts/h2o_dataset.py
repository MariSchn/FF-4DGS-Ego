"""H2O dataset over the packed .npz (one file per sequence; see scripts.pack_h2o).

Returns clips in the same shape our dense-depth experiment consumes
(``img`` [S,3,R,R], ``gt_obj_depth`` [S,R,R] m, ``gt_obj_mask`` [S,R,R]) PLUS the
clean H2O hand annotations (``joints`` camera-frame 3D, ``cam_pose`` cam->world)
for the metric-hand experiment. So the dense-depth run (train_hoi4d_depth) works
on H2O unchanged, and the hand head can layer on top using the high-quality MANO-
fit joints that are H2O's whole reason for being the headline dataset.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def discover_h2o_npz(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "*.npz")))


class H2ODepthDataset(Dataset):
    def __init__(self, npz_files, num_frames=16, res=224, clip_stride=8,
                 depth_min=0.05, depth_max=10.0, with_hand=True,
                 hires_hand=False, hand_crop_px=256):
        self.num_frames = num_frames
        self.res = res
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.with_hand = with_hand
        # High-res hand-crop branch: return per-hand native crops [S, 2, 3, P, P].
        # Uses the packed hand_crop_L/R arrays when present; otherwise falls back
        # to upscaling the 224 frame as a SHAPE-ONLY placeholder (for smoke tests
        # before the real hi-res repack).
        self.hires_hand = hires_hand
        self.hand_crop_px = hand_crop_px
        self.clips = []  # (npz_path, start)
        for f in npz_files:
            try:
                with np.load(f, mmap_mode="r") as d:
                    n = int(d["rgb"].shape[0])
            except Exception:
                continue
            for start in range(0, n - num_frames + 1, clip_stride):
                self.clips.append((f, start))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        f, start = self.clips[idx]
        S = self.num_frames
        d = np.load(f)
        rgb = d["rgb"][start:start + S].astype(np.float32) / 255.0      # [S,R,R,3]
        img = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()    # [S,3,R,R]
        depth = torch.from_numpy(d["depth"][start:start + S].astype(np.float32))  # [S,R,R] m
        mask = (depth > self.depth_min) & (depth < self.depth_max)
        out = {"img": img, "gt_obj_depth": depth, "gt_obj_mask": mask}
        if self.with_hand:
            # H2O hand_pose: [128]/frame = left[valid + 21x3] + right[valid + 21x3], METRES.
            out["joints"] = torch.from_numpy(d["joints"][start:start + S].astype(np.float32))
            out["cam_pose"] = torch.from_numpy(d["cam_pose"][start:start + S].astype(np.float32))
            out["K"] = torch.from_numpy(d["K"].astype(np.float32))   # [6] fx,fy,cx,cy,w,h
        if self.hires_hand:
            out["hand_crops"] = self._load_hand_crops(d, start, S, img)
        return out

    def _load_hand_crops(self, d, start, S, img):
        """Return per-hand native crops [S, 2, 3, P, P] in [0,1].

        Real path: the packed hand_crop_L/R arrays. Placeholder path (no hi-res
        in the npz): upscale the 224 frame to P px for both hands — shape-correct,
        for smoke tests only until the real hi-res repack lands.
        """
        P = self.hand_crop_px
        keys = d.files if hasattr(d, "files") else list(d.keys())
        if "hand_crop_L" in keys and "hand_crop_R" in keys:
            cl = d["hand_crop_L"][start:start + S].astype(np.float32) / 255.0  # [S,P,P,3]
            cr = d["hand_crop_R"][start:start + S].astype(np.float32) / 255.0
            left = torch.from_numpy(cl).permute(0, 3, 1, 2)   # [S,3,P,P]
            right = torch.from_numpy(cr).permute(0, 3, 1, 2)
            return torch.stack([left, right], dim=1)          # [S,2,3,P,P]
        # Placeholder: upscale the 224 frame (same image for both hands).
        up = F.interpolate(img, size=(P, P), mode="bilinear", align_corners=False)  # [S,3,P,P]
        return torch.stack([up, up], dim=1)                   # [S,2,3,P,P]
