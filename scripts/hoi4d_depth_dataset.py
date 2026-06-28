"""HOI4D dense-metric-depth dataset (the no-masking dense-supervision experiment).

Reads Livioni-mirror HOI4D sequences (``<seq>/images/*.png`` RGB + ``<seq>/raw_depth/*.png``
16-bit mm depth) and returns clips shaped for our model + ``object_depth_loss``:
``img`` [S,3,res,res], ``gt_obj_depth`` [S,Rd,Rd] (metres), ``gt_obj_mask`` [S,Rd,Rd].

Unlike the HOT3D object-render path, the depth here is DENSE GT (a real sensor map,
RGB-aligned), so the mask covers the whole valid frame — there is no "unsupervised
region to degrade", which is exactly the point of moving to a depth-sensor dataset.

RGB is 1920x1080; we center-square-crop to 1080x1080 then resize, and crop+resize the
depth IDENTICALLY (NEAREST) so depth pixels stay aligned with what the model sees.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def discover_hoi4d_seqs(root: str) -> list[str]:
    """Sequence dirs under ``root`` that have both images/ and raw_depth/."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(os.path.join(d, "images")) and os.path.isdir(os.path.join(d, "raw_depth")):
            out.append(d)
    return out


def _center_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return arr[y0:y0 + s, x0:x0 + s]


class HOI4DDepthDataset(Dataset):
    def __init__(self, seq_dirs, num_frames=16, res=224, clip_stride=8,
                 depth_res=224, depth_min=0.05, depth_max=10.0):
        self.num_frames = num_frames
        self.res = res
        self.depth_res = depth_res
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.clips = []  # (img_files[S], depth_files[S])
        for sd in seq_dirs:
            imgs = sorted(glob.glob(os.path.join(sd, "images", "*.png")))
            deps = sorted(glob.glob(os.path.join(sd, "raw_depth", "*.png")))
            n = min(len(imgs), len(deps))
            if n < num_frames:
                continue
            for start in range(0, n - num_frames + 1, clip_stride):
                self.clips.append((imgs[start:start + num_frames], deps[start:start + num_frames]))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        import cv2
        imfiles, depfiles = self.clips[idx]
        imgs, depths, masks = [], [], []
        for imf, df in zip(imfiles, depfiles):
            img = cv2.imread(imf)                                   # BGR HxWx3
            img = _center_square(img)
            img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))

            d = cv2.imread(df, cv2.IMREAD_ANYDEPTH)                 # 16-bit mm
            d = _center_square(d).astype(np.float32) / 1000.0      # -> metres
            d = cv2.resize(d, (self.depth_res, self.depth_res), interpolation=cv2.INTER_NEAREST)
            m = (d > self.depth_min) & (d < self.depth_max)
            depths.append(torch.from_numpy(d))
            masks.append(torch.from_numpy(m))
        return {
            "img": torch.stack(imgs),                              # [S,3,res,res]
            "gt_obj_depth": torch.stack(depths),                   # [S,Rd,Rd] metres
            "gt_obj_mask": torch.stack(masks),                     # [S,Rd,Rd] bool
        }
