"""OakInk2 (egocentric view) -> our HOI4D-style per-sequence hand dataset.

OakInk2 per-sequence annotation pickle (anno_preview/<seq>.pkl or full anno) holds, keyed by frame id:
    raw_mano:  rh__pose_coeffs [1,16,4] QUATERNIONS [w,x,y,z], rh__tsl [1,3], rh__betas [1,10]  (+ lh__)
    cam_def:   {serial: name}          -- egocentric camera identified by serial (default 104422070969)
    cam_intr:  {serial: {frame: K[3,3]}}
    cam_extr:  {serial: {frame: T[4,4]}}   world->camera
    mocap_frame_id_list: [int]
Images: <scene_dir>/<serial>/<frame:...>.jpg  (ego = the ego serial).

Convert: quat pose_coeffs -> axis-angle (global_orient = quat[0], hand_pose = quat[1:16]) -> smplx
MANO(use_pca=False) FK with tsl+betas -> WORLD joints; apply cam_extr[ego] -> ego-camera joints;
MANO21_TO_16; K_ego -> [f,cx,cy]; cam_extr[ego] -> extrinsics; ego frames -> video; project -> bbox.
Reuses the ARCTIC converter's verified helpers (apply_se3, joints_to_bbox, build_mano, fk_world_joints).

VALIDATION note (same as ARCTIC): the MANO convention (smplx vs OakInk's manotorch, flat_hand_mean,
quat->aa order) must be checked -- OakInk provides no precomputed joints, so validate via bone lengths
and by projecting joints onto the ego image (--validate prints index-bone length; expect ~30-45 mm).

Usage (smplx-capable env + MANO models):
    python -m scripts.oakink2_to_ours --anno_root <.../anno_preview> --image_root <.../image> \
        --mano_dir <mano> --out_root $S/oakink2_ours --ego_serial 104422070969 [--validate]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import cv2
import numpy as np
import torch

from scripts.arctic_to_ours import (MANO21_TO_16, apply_se3, build_mano,
                                    fk_world_joints, joints_to_bbox)

EGO_SERIAL_DEFAULT = "104422070969"


def quat_to_axis_angle(q):
    """q [...,4] wxyz -> axis-angle [...,3]. Robust near identity."""
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    w = np.clip(q[..., 0], -1.0, 1.0)
    ang = 2.0 * np.arccos(w)                       # [0, 2pi]
    ang = np.where(ang > np.pi, ang - 2 * np.pi, ang)   # wrap to [-pi,pi]
    s = np.sqrt(np.clip(1.0 - w * w, 1e-12, None))
    axis = q[..., 1:] / s[..., None]
    return axis * ang[..., None]                   # [...,3]


def mano_params_from_quat(pc, tsl, betas):
    """pc [N,16,4] wxyz, tsl [N,3], betas [N,10] -> rot[N,3], pose[N,45], trans[N,3], shape[N,10]."""
    aa = quat_to_axis_angle(np.asarray(pc, np.float64))   # [N,16,3]
    rot = aa[:, 0].astype(np.float32)                     # global orient
    pose = aa[:, 1:].reshape(aa.shape[0], 45).astype(np.float32)
    return rot, pose, np.asarray(tsl, np.float32).reshape(-1, 3), np.asarray(betas, np.float32).reshape(-1, 10)


def ego_serial_of(anno, override):
    if override:
        return override
    cam_def = anno.get("cam_def", {})
    for serial, name in cam_def.items():
        if "ego" in str(name).lower():
            return serial
    return EGO_SERIAL_DEFAULT


def convert_seq(seq, anno, mano, image_root, out_root, device, ego_serial, max_frames=0):
    ego = ego_serial_of(anno, ego_serial)
    frames = list(anno.get("mocap_frame_id_list", sorted(anno.get("raw_mano", {}).keys())))
    if max_frames:
        frames = frames[:max_frames]
    N = len(frames)
    if N == 0:
        return None
    K_by = anno.get("cam_intr", {}).get(ego, {})
    E_by = anno.get("cam_extr", {}).get(ego, {})
    if not E_by:
        print(f"SEQ_SKIP {seq}: no ego extrinsics for serial {ego}", flush=True)
        return None

    cam = np.full((N, 2, 16, 3), np.nan, np.float32)
    wld = np.full((N, 2, 16, 3), np.nan, np.float32)
    valid = np.zeros((N, 2), bool)
    world2ego = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    rawm = anno["raw_mano"]
    for hi, hk in [(0, "lh"), (1, "rh")]:
        # gather per-frame params for this hand
        pc, tsl, betas, ok = [], [], [], []
        for t, fid in enumerate(frames):
            fm = rawm.get(fid, {})
            e = E_by.get(fid)
            if e is not None:
                world2ego[t] = np.asarray(e, np.float32)
            if f"{hk}__pose_coeffs" in fm and e is not None:
                pc.append(np.asarray(fm[f"{hk}__pose_coeffs"]).reshape(16, 4))
                tsl.append(np.asarray(fm[f"{hk}__tsl"]).reshape(3))
                betas.append(np.asarray(fm[f"{hk}__betas"]).reshape(10))
                ok.append(t)
        if not ok:
            continue
        rot, pose, tr, sh = mano_params_from_quat(np.stack(pc), np.stack(tsl), np.stack(betas))
        side = "left" if hk == "lh" else "right"
        jw = fk_world_joints(mano[side], rot, pose, tr, sh, device)      # [n,21,3] world
        for k, t in enumerate(ok):
            jc = apply_se3(world2ego[t:t + 1], jw[k:k + 1])[0]           # [21,3] ego cam
            cam[t, hi] = jc[MANO21_TO_16]
            wld[t, hi] = jw[k][MANO21_TO_16]
            valid[t, hi] = np.isfinite(jc[MANO21_TO_16]).all()

    # ego frames -> video
    seq_id = seq.replace("/", "_")
    out_seq = os.path.join(out_root, seq_id)
    os.makedirs(os.path.join(out_seq, "hand_data"), exist_ok=True)
    scene_dir = os.path.join(image_root, seq)
    vw, W, H = None, None, None
    for fid in frames:
        cands = glob.glob(os.path.join(scene_dir, ego, f"*{fid}*.jpg")) or \
            [os.path.join(scene_dir, ego, f"{fid}.jpg")]
        im = cv2.imread(cands[0]) if cands and os.path.exists(cands[0]) else None
        if im is None:
            continue
        if vw is None:
            H, W = im.shape[:2]
            vw = cv2.VideoWriter(os.path.join(out_seq, "video_main_rgb.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        vw.write(im)
    if vw is not None:
        vw.release()

    # intrinsics from ego K (first available frame)
    Kego = None
    for fid in frames:
        if fid in K_by:
            Kego = np.asarray(K_by[fid], np.float32); break
    if Kego is None:
        print(f"SEQ_SKIP {seq}: no ego intrinsics", flush=True)
        return None
    f, cx, cy = float(Kego[0, 0]), float(Kego[0, 2]), float(Kego[1, 2])
    if W is None:
        W, H = int(round(cx * 2)), int(round(cy * 2))

    boxes = np.zeros((N, 2, 4), np.float32)
    for t in range(N):
        for hi in range(2):
            if valid[t, hi]:
                boxes[t, hi] = joints_to_bbox(cam[t, hi], f, cx, cy, W, H)

    hd = os.path.join(out_seq, "hand_data")
    torch.save(torch.tensor([f, W / 2.0, H / 2.0]), os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.from_numpy(world2ego), os.path.join(hd, "cam_extrinsics_cache.pt"))
    torch.save(torch.from_numpy(cam), os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(torch.from_numpy(wld), os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save({"bboxes": torch.from_numpy(boxes), "valid": torch.from_numpy(valid)},
               os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt"))
    return {"seq": seq_id, "N": N, "res": (W, H), "ego": ego, "valid_rate": float(valid.mean())}


def validate(seq, anno, mano, device, ego_serial):
    """No precomputed joints in OakInk -> sanity via index-finger bone length (~30-45 mm)."""
    frames = list(anno.get("mocap_frame_id_list", []))[:8]
    rawm = anno["raw_mano"]
    pc, tsl, betas = [], [], []
    for fid in frames:
        fm = rawm.get(fid, {})
        if "rh__pose_coeffs" in fm:
            pc.append(np.asarray(fm["rh__pose_coeffs"]).reshape(16, 4))
            tsl.append(np.asarray(fm["rh__tsl"]).reshape(3)); betas.append(np.asarray(fm["rh__betas"]).reshape(10))
    if not pc:
        print(f"VALIDATE {seq}: no rh mano"); return
    rot, pose, tr, sh = mano_params_from_quat(np.stack(pc), np.stack(tsl), np.stack(betas))
    jw = fk_world_joints(mano["right"], rot, pose, tr, sh, device)
    bl = np.linalg.norm(jw[:, 5] - jw[:, 6], axis=-1).mean() * 1000
    span = np.linalg.norm(jw[:, 12] - jw[:, 0], axis=-1).mean() * 1000
    print(f"VALIDATE {seq}: index-bone {bl:.1f} mm (expect ~30-45), wrist->mid-tip {span:.1f} mm "
          f"(expect ~90-120). Large/small => quat->aa or MANO convention wrong.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno_root", required=True, help="dir of per-seq .pkl (anno_preview or full)")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--mano_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--ego_serial", default="")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano = build_mano(a.mano_dir, device)
    pkls = sorted(glob.glob(os.path.join(a.anno_root, "*.pkl")))
    if a.max_seqs:
        pkls = pkls[: a.max_seqs]
    os.makedirs(a.out_root, exist_ok=True)
    done = 0
    for pk in pkls:
        seq = os.path.splitext(os.path.basename(pk))[0]
        try:
            anno = pickle.load(open(pk, "rb"))
            if a.validate:
                validate(seq, anno, mano, device, a.ego_serial); continue
            r = convert_seq(seq, anno, mano, a.image_root, a.out_root, device, a.ego_serial, a.max_frames)
            if r:
                done += 1
                print(f"[{done}] {r['seq']} N={r['N']} res={r['res']} ego={r['ego']} valid={r['valid_rate']:.2f}", flush=True)
        except Exception as e:
            print(f"SEQ_FAIL {seq}: {e}", flush=True)
    print(f"OAKINK2_TO_OURS_DONE wrote {done} seqs -> {a.out_root}", flush=True)


if __name__ == "__main__":
    main()
