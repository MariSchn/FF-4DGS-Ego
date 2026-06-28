"""Debug H2O joint convention: print stats + overlay GT joints on the packed RGB.

If the red dots land on the hand, the joint parse + remap + K crop-adjust are right
and the C-MPJPE bug is purely a *prediction* frame mismatch. If they don't, the GT
parse/projection is the culprit. CPU only (numpy + PIL).

    python3 scripts/debug_h2o_joints.py <seq.npz> <out_dir>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

FULL_W, FULL_H = 1280, 720


def main():
    f = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
    os.makedirs(out, exist_ok=True)
    d = np.load(f)
    rgb, joints, K, cam_pose = d["rgb"], d["joints"], d["K"], d["cam_pose"]
    R = rgb.shape[1]
    print(f"rgb {rgb.shape} joints {joints.shape} K {K} cam_pose {cam_pose.shape}", flush=True)

    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    x0 = (FULL_W - FULL_H) // 2
    s = R / float(FULL_H)
    fxa, fya, cxa, cya = fx * s, fy * s, (cx - x0) * s, cy * s
    print(f"adjusted K (for {R} crop): fx={fxa:.1f} fy={fya:.1f} cx={cxa:.1f} cy={cya:.1f}", flush=True)

    for fi in [0, min(60, rgb.shape[0] - 1), min(120, rgb.shape[0] - 1)]:
        j = joints[fi]
        lvalid, rvalid = float(j[0]), float(j[64])
        left = j[1:64].reshape(21, 3)
        right = j[65:128].reshape(21, 3)
        print(f"\nframe {fi}: lvalid={lvalid} rvalid={rvalid}", flush=True)
        print(f"  left wrist(mm)={left[0]}  right wrist(mm)={right[0]}", flush=True)
        img = Image.fromarray(rgb[fi]).convert("RGB")
        dr = ImageDraw.Draw(img)
        for hand, valid, col in [(left, lvalid, (255, 0, 0)), (right, rvalid, (0, 255, 0))]:
            if valid < 0.5:
                continue
            Z = np.clip(hand[:, 2], 1e-3, None)
            u = fxa * hand[:, 0] / Z + cxa
            v = fya * hand[:, 1] / Z + cya
            print(f"  proj u=[{u.min():.0f},{u.max():.0f}] v=[{v.min():.0f},{v.max():.0f}] "
                  f"(in-frame: {int(((u>=0)&(u<R)&(v>=0)&(v<R)).sum())}/21)", flush=True)
            for uu, vv in zip(u, v):
                if 0 <= uu < R and 0 <= vv < R:
                    dr.ellipse([uu - 2, vv - 2, uu + 2, vv + 2], fill=col)
        p = os.path.join(out, f"h2o_overlay_{fi}.png")
        img.resize((448, 448)).save(p)
        print(f"  saved {p}", flush=True)


if __name__ == "__main__":
    main()
