"""Throwaway recon: print shapes/dtypes of the per-sequence GT caches so the
world-space harness can decode them correctly. Run on a gb10 node."""
import os
import torch

SD = "/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609/P0001_550ea2ac"
HD = os.path.join(SD, "hand_data")

for f in [
    "gt_joints_cache_world.pt",
    "gt_joints_cache_cam_v2.pt",
    "cam_extrinsics_cache.pt",
    "cam_intrinsics.pt",
    "hand_bboxes_v2_rf1.5_res224x224.pt",
]:
    p = os.path.join(HD, f)
    try:
        t = torch.load(p, map_location="cpu")
        if torch.is_tensor(t):
            print(f, "TENSOR", tuple(t.shape), t.dtype,
                  "range", round(float(t.float().min()), 3), round(float(t.float().max()), 3))
        elif isinstance(t, dict):
            print(f, "DICT keys", list(t.keys())[:8])
            for k, v in list(t.items())[:3]:
                print("   ", k, tuple(v.shape) if torch.is_tensor(v) else type(v))
        elif isinstance(t, (list, tuple)):
            print(f, "LIST len", len(t), "elem0",
                  tuple(t[0].shape) if torch.is_tensor(t[0]) else type(t[0]))
        else:
            print(f, type(t))
    except Exception as e:
        print(f, "ERR", repr(e))

# video frame count
try:
    import sys
    sys.path.insert(0, "/home/dmonopoli/FF-4DGS-Ego")
    from decord import VideoReader
    vr = VideoReader(os.path.join(SD, "video_main_rgb.mp4"))
    print("video frames:", len(vr))
except Exception as e:
    print("video ERR", repr(e))
