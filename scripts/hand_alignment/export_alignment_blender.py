"""
export_alignment_blender.py

Export one frame of the alignment as assets you can open in Blender to make a
fully cinematic figure (real materials, lighting, shadows, depth of field).

Writes, into <output_dir>/blender_export/:
  gaussians_frame<F>.ply   Standard INRIA 3D-Gaussian-Splatting PLY (the format
                           the common Blender add-ons read). Activated values in
                           gaussians.pt are converted back to the raw form those
                           add-ons expect (opacity -> logit, scale -> log).
  hand_frame<F>_<side>.obj GT MANO hand mesh at the saved alignment scale.

How to use in Blender (free add-ons, pick one):
  * KIRI Engine "3D Gaussian Splatting Render" add-on, or the "3DGS Render"
    add-on  -> File > Import > the .ply
  * File > Import > Wavefront (.obj)  -> the hand mesh(es)
Both are exported in the SAME reconstruction world frame, so they line up on
import — no manual alignment needed. (Reconstruction frame is Z-up, which is
also Blender's default up axis.)

Usage:
  python scripts/hand_alignment/export_alignment_blender.py \
      --output_dir outputs/hand_alignment/<dir> --frame 64
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch


def load(output_dir):
    out = Path(output_dir)
    groups = {int(d["timestamp"]): d
              for d in torch.load(out / "gaussians.pt", map_location="cpu",
                                  weights_only=False)}
    hands_by_frame = {}
    default_scale = 1.0
    hp = out / "hand_meshes.pt"
    if hp.exists():
        for m in torch.load(hp, weights_only=False):
            hands_by_frame.setdefault(int(m["frame_offset"]), []).append(m)
            default_scale = float(m.get("default_scale", default_scale))
    al = out / "alignment.json"
    if al.exists():
        default_scale = float(json.load(open(al)).get("scale", default_scale))
    return groups, hands_by_frame, default_scale


def write_3dgs_ply(path, g):
    """Write a frame's Gaussians as an INRIA-format 3DGS PLY (binary LE).

    gaussians.pt stores *activated* opacity (sigmoid) and *activated* scale
    (exp). The standard 3DGS PLY / Blender add-ons re-apply sigmoid & exp on
    load, so we invert them here: opacity -> logit, scale -> log.
    """
    means = g["means"].numpy().astype(np.float32)
    f_dc = g["harmonics"].numpy()[:, 0, :3].astype(np.float32)   # SH DC coeff
    opac = g["opacities"].numpy().astype(np.float32).clip(1e-6, 1 - 1e-6)
    opac_logit = np.log(opac / (1 - opac)).astype(np.float32)
    scale_log = np.log(g["scales"].numpy().astype(np.float32).clip(1e-9)).astype(np.float32)
    quat = g["rotations"].numpy().astype(np.float32)             # wxyz
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-8)
    N = means.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)

    props = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"])
    data = np.concatenate(
        [means, normals, f_dc, opac_logit[:, None], scale_log, quat], axis=1
    ).astype(np.float32)

    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {N}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def write_obj(path, verts, faces, name):
    with open(path, "w") as fh:
        fh.write(f"o {name}\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:  # OBJ is 1-indexed
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def hand_vertices(m, scale):
    if "pos_pred" in m and "displacement" in m:
        return (np.asarray(m["pos_pred"], np.float64)
                + scale * np.asarray(m["displacement"], np.float64))
    return np.asarray(m["vertices"], np.float64)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--frame", type=int, default=None, help="Default: middle frame.")
    p.add_argument("--scale", type=float, default=None, help="Default: saved scale.")
    args = p.parse_args()

    groups, hands_by_frame, default_scale = load(args.output_dir)
    frame = args.frame if args.frame is not None else (max(groups) + 1) // 2
    scale = args.scale if args.scale is not None else default_scale
    if frame not in groups:
        raise SystemExit(f"Frame {frame} not in gaussians.pt.")

    out = Path(args.output_dir) / "blender_export"
    out.mkdir(parents=True, exist_ok=True)

    ply = out / f"gaussians_frame{frame:04d}.ply"
    write_3dgs_ply(ply, groups[frame])
    print(f"Wrote {ply}  ({groups[frame]['means'].shape[0]:,} Gaussians)")

    for m in hands_by_frame.get(frame, []):
        verts = hand_vertices(m, scale)
        obj = out / f"hand_frame{frame:04d}_{m['side']}.obj"
        write_obj(obj, verts, np.asarray(m["faces"], np.int32), m["side"])
        print(f"Wrote {obj}")

    print(f"\nDone (frame {frame}, scale {scale:.4f}). Import both into Blender — "
          f"they share the same world frame.")


if __name__ == "__main__":
    main()
