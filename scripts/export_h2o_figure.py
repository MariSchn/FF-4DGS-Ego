"""Projectaria-free renderable-Gaussian + PREDICTED-hand export for the figure.

Forwards the model (GS head on + the H2O-tuned hand head) on one H2O clip and writes
the artifacts that scripts/hand_alignment/render_alignment_3d.py reads:
  gaussians.pt, gaussians_frame0000.ply, camera_params.json, hand_meshes.pt

H2O scene Gaussians are up-to-scale, but the hand head outputs METRIC joints. We bring
the metric hand into the (up-to-scale) scene frame by matching the predicted metric
wrist depth to the scene's gs_depth at the wrist pixel:
    s = median_over_frames( wrist_Z_metric / gs_depth[wrist_pixel] )
then hand_in_scene = hand_metric / s, lifted cam->world by the predicted extrinsics.
This is exactly the hand<->scene metric coupling, made visible. No HOT3D /
projectaria_tools (H2O is pinhole) so it runs on gb10.

Usage (gb10):
  python -m scripts.export_h2o_figure --config configs/exp_h2o_hand.yaml \
    --ckpt /work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt \
    --h2o /work/scratch/dmonopoli/h2o_packed --subject subject4 \
    --clip_index 0 --output_dir /work/scratch/dmonopoli/fig_h2o
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import trimesh

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.auxiliary_models.worldmirror.utils.save_utils import save_gs_ply, save_camera_params
from scripts.hand_vis_utils import MANOModel
from scripts.h2o_dataset import H2ODepthDataset
from scripts.train_hand_head import build_views, compute_joints_from_batch
from scripts.eval_cmpjpe import h2o_gt_joints16, bboxes_from_joints

SH_C0 = 0.28209479177387814


def build_model(model_cfg, ckpt, device):
    mc = dict(model_cfg)
    mc["enable_gs"] = True                       # need the GS scene for the figure
    mc["enable_hand"] = True
    mc["hand_to_gs_injection"] = {"enabled": False}
    model = WorldMirror(**{k: v for k, v in mc.items() if k != "checkpoint"})
    base = torch.load(model_cfg["checkpoint"], map_location=device)
    sd = base.get("state_dict", base.get("reconstructor", base)) if isinstance(base, dict) else base
    model.load_state_dict(sd, strict=False)
    ck = torch.load(ckpt, map_location=device)
    hsd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    tgt = model if any(k.startswith(("hand_head.", "gs_head.")) for k in hsd) else model.hand_head
    tgt.load_state_dict(hsd, strict=False)
    return model.to(device).eval()


def recover_scale_wrist_depth(pred_joints, gs_depth, pred_K, valid):
    """s s.t. metric_hand / s lands at the scene's up-to-scale depth at the wrist.

    pred_joints [1,S,2,16,3] metric cam frame; gs_depth [1,S,H,W,1] up-to-scale;
    pred_K [S,3,3] (model image space); valid [1,S,2]. Returns float s (median).
    """
    _, S, Hh, Wh, _ = gs_depth.shape
    ratios = []
    for k in range(S):
        K = pred_K[k]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        for i in range(2):
            if valid[0, k, i] < 0.5:
                continue
            w = pred_joints[0, k, i, 0]          # wrist, metric cam frame
            z = float(w[2])
            if z <= 1e-3:
                continue
            u = int(round(fx * float(w[0]) / z + cx))
            v = int(round(fy * float(w[1]) / z + cy))
            if not (0 <= u < Wh and 0 <= v < Hh):
                continue
            d = float(gs_depth[0, k, v, u, 0])
            if d > 1e-4:
                ratios.append(z / d)
    if not ratios:
        print("  [scale] no valid wrist/depth samples; falling back to s=1.0")
        return 1.0
    s = float(np.median(ratios))
    print(f"  [scale] s = median(wrist_Z / gs_depth) = {s:.4f} over {len(ratios)} samples")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_h2o_hand.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h2o", required=True)
    ap.add_argument("--subject", default="subject4")
    ap.add_argument("--clip_index", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(cfg["model"], args.ckpt, device)
    mano = MANOModel(cfg["visualization"]["mano_model_folder"])
    nf = cfg["data"]["num_frames"]

    npz = sorted(glob.glob(os.path.join(args.h2o, "*.npz")))
    if args.subject:
        npz = [f for f in npz if args.subject in os.path.basename(f)]
    ds = H2ODepthDataset(npz, num_frames=nf, res=224, clip_stride=nf, with_hand=True)
    print(f"H2O clips: {len(ds)}; exporting clip {args.clip_index}")
    batch = ds[args.clip_index]
    img = batch["img"].unsqueeze(0).to(device)              # [1,S,3,R,R]
    joints = batch["joints"].unsqueeze(0)                   # [1,S,128]
    K = np.load(ds.clips[args.clip_index][0])["K"]

    gt_j, valid = h2o_gt_joints16(joints)
    hb = bboxes_from_joints(gt_j, K).to(device)
    views = build_views(img, nf, device, hb, valid.to(device))
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(views, is_inference=True, use_motion=False)
    print("Pred keys:", [k for k in preds if isinstance(preds[k], torch.Tensor)])

    gaussian_list = preds["splats"][0]
    pred_c2w = preds["rendered_extrinsics"][0].float().cpu().numpy()   # [S,4,4] cam2world
    pred_K = preds["rendered_intrinsics"][0].float().cpu().numpy()     # [S,3,3]
    gs_depth = preds["gs_depth"].float().cpu().numpy()                 # [1,S,H,W,1]

    pred_joints = compute_joints_from_batch(preds["hand_joints"], mano, device).float().cpu()
    s = recover_scale_wrist_depth(pred_joints, gs_depth, pred_K, valid)

    # Predicted hand meshes: metric cam frame -> /s (scene units) -> world via c2w.
    hand_params = preds["hand_joints"][0].float().cpu()    # [S,64]
    hand_meshes = []
    for k in range(nf):
        c2w = pred_c2w[k]
        for i, side in [(0, "left"), (1, "right")]:
            if valid[0, k, i] < 0.5:
                continue
            verts_cam, faces = mano.get_mesh_from_params(hand_params[k, i * 32:(i + 1) * 32],
                                                         is_right=(i == 1))
            verts_scene = verts_cam / s
            verts_world = (c2w[:3, :3] @ verts_scene.T).T + c2w[:3, 3]
            verts_world = verts_world.astype(np.float32)
            hand_meshes.append({
                "frame_offset": k, "frame_global": k, "side": side,
                "pos_pred": verts_world,
                "displacement": np.zeros_like(verts_world, dtype=np.float32),
                "vertices": verts_world, "default_scale": 1.0,
                "faces": faces.astype(np.int32),
            })
            trimesh.Trimesh(vertices=verts_world, faces=faces).export(
                os.path.join(args.output_dir, f"hand_meshes_frame{k:04d}_{side}.obj"))
    print(f"Exported {len(hand_meshes)} predicted hand mesh(es)")

    save_camera_params(extrinsics=pred_c2w, intrinsics=pred_K, target_dir=args.output_dir)
    torch.save(
        [{"means": gs.means.cpu(), "harmonics": gs.harmonics.cpu(), "opacities": gs.opacities.cpu(),
          "scales": gs.scales.cpu(), "rotations": gs.rotations.cpu(), "timestamp": gs.timestamp}
         for gs in gaussian_list],
        os.path.join(args.output_dir, "gaussians.pt"))
    gs0 = gaussian_list[0]
    save_gs_ply(path=os.path.join(args.output_dir, "gaussians_frame0000.ply"),
                means=gs0.means.float(), scales=gs0.scales.float(), rotations=gs0.rotations.float(),
                rgbs=(0.5 + SH_C0 * gs0.harmonics[..., 0, :]).clamp(0, 1).float(),
                opacities=gs0.opacities.float())
    torch.save(hand_meshes, os.path.join(args.output_dir, "hand_meshes.pt"))
    print(f"Wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
