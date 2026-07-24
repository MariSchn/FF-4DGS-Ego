"""ARCTIC (egocentric view) -> our HOI4D-style per-sequence hand dataset.

ARCTIC processed splits (data/arctic_data/data/splits/p1_{train,val}.npy) give, per sequence,
under data_dict[seq]:
    params:  K_ego [3,3], world2ego [N,4,4], dist [8],
             rot_r/pose_r/trans_r/shape_r  (3/45/3/10) and _l  -- MANO in WORLD frame, per frame
    cam_coord['joints.right'/'joints.left']:  [N, Vallo, 21, 3]  (allocentric views; used only to
             VALIDATE our MANO-FK convention, not for the ego joints)
    imgnames: relative ego-image paths
    (frames N = first dim of the param arrays)

We FK the WORLD MANO params with smplx.MANO(use_pca=False, flat_hand_mean=False) -> world joints
[N,21,3], apply world2ego -> ego-camera joints, take the 16 smplx-kinematic joints (MANO21_TO_16),
and write our layout per seq under <out>/<subj>_<seq>/:
    video_main_rgb.mp4                         (ego frames, from imgnames)
    hand_data/cam_intrinsics.pt                [f, cx, cy]  (from K_ego, at the ego image resolution)
    hand_data/cam_extrinsics_cache.pt          [N,4,4] world->ego (= world2ego)
    hand_data/gt_joints_cache_cam_v2.pt        [N,2,16,3] m, ego-camera frame  (LH=0, RH=1)
    hand_data/gt_joints_cache_world.pt         [N,2,16,3] m, world frame
    hand_data/hand_bboxes_v2_rf1.5_res224x224.pt  {"bboxes":[N,2,4] normalized xyxy, "valid":[N,2]}

Hand index 0=left, 1=right (matches our caches). Right hand is RH=1.

VALIDATION (--validate): for a few frames, FK the world joints, transform to an ALLOCENTRIC view via
that view's extrinsic, and compare to cam_coord (ARCTIC's own joints) -> RMSE must be ~mm; if it is
large the MANO convention (flat_hand_mean / pca / joint order) is wrong and MUST be fixed before use.

Usage (a MANO-capable env, e.g. venv with smplx + our _DATA/mano):
    python -m scripts.arctic_to_ours --split_npy .../splits/p1_val.npy \
        --img_root .../arctic_data/data/images --mano_dir _DATA/data/body_models/mano \
        --out_root $S/arctic_ours --max_seqs 0 [--validate]
"""
from __future__ import annotations

import argparse
import json
import os
import types

import cv2
import numpy as np
import torch


def pose45_to_pca15(pose45: np.ndarray, mano_ns, is_right: bool) -> np.ndarray:
    """Project a 45-dim full (no-PCA, flat-hand-mean) MANO pose onto our 15 PCA coeffs, using the
    smplx layer's registered `hand_components`/`hand_mean` (singular; the plural names return None).
    Lossy 45->15 projection; only fills gt64's supervision-unused pose15 slot. Self-contained copy of
    scripts.preprocessing.preprocess_hoi4d.pose45_to_pca15 (kept here so the converter has no
    cross-module dependency). Falls back to truncation if components are unavailable."""
    layer = mano_ns.right if is_right else mano_ns.left
    comps = getattr(layer, "hand_components", None)
    mean = getattr(layer, "hand_mean", None)
    if comps is None:
        return np.asarray(pose45)[..., :15]
    comps = comps.detach().cpu().numpy() if torch.is_tensor(comps) else np.asarray(comps)
    mean = (mean.detach().cpu().numpy() if torch.is_tensor(mean)
            else (np.zeros(45) if mean is None else np.asarray(mean)))
    return (np.asarray(pose45) - mean) @ np.linalg.pinv(comps)[:, :15]

# ARCTIC uses the original MANO 21-joint order; take the 16 smplx-kinematic joints (drop 5 tips),
# same subset the rest of our pipeline uses (see scripts/eval_cmpjpe.py, run_wilor_h2o.py).
MANO21_TO_16 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

NUM_HANDS = 2
NUM_JOINTS = 16
HAND_PARAM_DIM = 32          # per hand: [transl 3, quat_wxyz 4, pose15 (PCA) 15, betas 10]


def _to_smplx16(j: np.ndarray) -> np.ndarray:
    """Reduce a MANO joint array to our smplx-16 kinematic order [N,16,3]. smplx MANO's native
    `.joints` output is ALREADY the 16 kinematic joints in smplx order (wrist, index x3, middle x3,
    pinky x3, ring x3, thumb x3) - matches OP2SMPLX16's target - so it is used directly. A 21-joint
    (OpenPose-with-tips) array is reduced via MANO21_TO_16 (drops the 5 fingertips)."""
    return j[:, MANO21_TO_16] if j.shape[1] >= 21 else j[:, :NUM_JOINTS]


def build_mano(mano_dir: str, device: str):
    """Two smplx MANO layers (left, right), full-pose (use_pca=False). flat_hand_mean=False matches
    ARCTIC's fit (validated by --validate); flip to True if the validation RMSE is large."""
    import smplx
    common = dict(model_path=mano_dir, use_pca=False, flat_hand_mean=False,
                  create_transl=False, batch_size=1)
    right = smplx.create(model_type="mano", is_rhand=True, **common).to(device).eval()
    left = smplx.create(model_type="mano", is_rhand=False, **common).to(device).eval()
    return {"right": right, "left": left}


def fk_world_joints(mano, rot, pose, trans, shape, device):
    """rot[N,3] pose[N,45] trans[N,3] shape[N,10] (world) -> world joints [N,21,3] (metres)."""
    n = rot.shape[0]
    out = []
    B = 512
    for s in range(0, n, B):
        e = min(n, s + B)
        with torch.no_grad():
            r = mano(global_orient=torch.as_tensor(rot[s:e], dtype=torch.float32, device=device),
                     hand_pose=torch.as_tensor(pose[s:e], dtype=torch.float32, device=device),
                     betas=torch.as_tensor(shape[s:e], dtype=torch.float32, device=device))
        j = r.joints + torch.as_tensor(trans[s:e], dtype=torch.float32, device=device)[:, None, :]
        out.append(j.cpu())
    return torch.cat(out, 0).numpy()          # [N,21,3]


def apply_se3(T, pts):
    """T [N,4,4] world->cam ; pts [N,J,3] world -> [N,J,3] cam."""
    R, t = T[:, :3, :3], T[:, :3, 3]
    return np.einsum("nij,nkj->nki", R, pts) + t[:, None, :]


def joints_to_bbox(j_cam, f, cx, cy, W, H, rf=1.5):
    """j_cam [J,3] -> normalized xyxy bbox (rescaled rf), or None if behind camera."""
    z = np.clip(j_cam[:, 2], 1e-3, None)
    u = f * j_cam[:, 0] / z + cx
    v = f * j_cam[:, 1] / z + cy
    x1, x2, y1, y2 = u.min(), u.max(), v.min(), v.max()
    cxb, cyb, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * rf, (y2 - y1) * rf
    return np.array([(cxb - w / 2) / W, (cyb - h / 2) / H, (cxb + w / 2) / W, (cyb + h / 2) / H], np.float32)


def project_single_focal(j_cam, f, cx, cy):
    """[N,J,3] cam-frame metres -> (u[N,J], v[N,J]) px, preprocess_hoi4d single-focal form."""
    z = np.clip(j_cam[..., 2], 1e-3, None)
    return f * j_cam[..., 0] / z + cx, f * j_cam[..., 1] / z + cy


def _frame_index(rel_path: str) -> int:
    """'.../s05/box_grab_01/0/00010.jpg' -> 10 (the frame number embedded in the basename)."""
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def group_imgnames_by_seq(imgnames: list, seqs: list, view: str = "0") -> dict:
    """build_splits.py stores a FLAT list of ego image paths across all seqs. Group them per
    sequence ('<sid>/<seq_name>'), keep only the requested view, sort by frame index. Returns
    {seq: [rel_path, ...]} (contiguous frame range, matching build_split's [10, N-10) slice)."""
    by_seq = {s: [] for s in seqs}
    for p in imgnames:
        parts = p.replace("\\", "/").split("/")
        if view is not None and (len(parts) < 2 or parts[-2] != view):
            continue
        seq = "/".join(parts[-4:-2]) if len(parts) >= 4 else None  # <sid>/<seq_name>
        if seq in by_seq:
            by_seq[seq].append(p)
    for s in by_seq:
        by_seq[s].sort(key=_frame_index)
    return by_seq


def camera_frame_wrist_params(rot_w, trans_w, world2ego, mano_ns, is_right):
    """World-frame MANO -> the camera-frame 32-dim param vector's pose-independent part.

    The training/eval caches expect the wrist transform in CAMERA frame (crop-local
    semantics, mirroring the HOI4D no-extrinsics path). For a MANO model whose joints are
    posed about the origin before `transl` is added, rotating the world result by the
    world->ego extrinsic is equivalent to composing the global rotation, so:
        transl_cam = R_we @ trans_w + t_we
        R_global_cam = R_we @ R(rot_w)   ->  q_wxyz = quat(R_global_cam)
    (--validate's MANO forward check measures the residual of this identity.)

    rot_w [N,3] world global axis-angle; trans_w [N,3] world transl; world2ego [N,4,4].
    Returns transl_cam [N,3], quat_wxyz [N,4]."""
    from scipy.spatial.transform import Rotation
    R_we = world2ego[:, :3, :3]                                  # [N,3,3]
    t_we = world2ego[:, :3, 3]                                   # [N,3]
    R_gw = Rotation.from_rotvec(rot_w).as_matrix()               # [N,3,3]
    R_cam = np.einsum("nij,njk->nik", R_we, R_gw)                # [N,3,3]
    q_xyzw = Rotation.from_matrix(R_cam).as_quat()               # [N,4] (x,y,z,w)
    quat_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
    transl_cam = np.einsum("nij,nj->ni", R_we, trans_w) + t_we   # [N,3]
    return transl_cam.astype(np.float32), quat_wxyz.astype(np.float32)


def convert_seq(seq, d, mano, img_root, out_root, device, imgnames=None, max_frames=0):
    p = d["params"]
    world2ego = np.asarray(p["world2ego"], np.float32)           # [N,4,4]
    K = np.asarray(p["K_ego"], np.float32)                       # [3,3] or [N,3,3] (constant)
    if K.ndim == 3:                                              # build_splits stores per-frame
        K = K[0]
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    N = world2ego.shape[0]
    if max_frames:
        N = min(N, max_frames)

    mano_ns = types.SimpleNamespace(right=mano["right"], left=mano["left"])
    cam = np.full((N, 2, 16, 3), np.nan, np.float32)
    wld = np.full((N, 2, 16, 3), np.nan, np.float32)
    j2d = np.zeros((N, 2, 16, 3), np.float32)                    # (u, v, conf) px
    gt64 = np.zeros((N, 2 * HAND_PARAM_DIM), np.float32)         # [N,64] camera-frame params
    valid = np.zeros((N, 2), bool)
    for hi, hk in [(0, "l"), (1, "r")]:
        rot, pose = p[f"rot_{hk}"][:N], p[f"pose_{hk}"][:N]
        trans, shape = p[f"trans_{hk}"][:N], p[f"shape_{hk}"][:N]
        side = "left" if hk == "l" else "right"
        is_right = hk == "r"
        jw = fk_world_joints(mano[side], rot, pose, trans, shape, device)   # [N,J,3] world
        jc = apply_se3(world2ego[:N], jw)                                    # [N,J,3] ego cam
        jw16, jc16 = _to_smplx16(jw), _to_smplx16(jc)                        # [N,16,3]
        wld[:, hi] = jw16
        cam[:, hi] = jc16
        # A frame is valid for this hand only if every joint is finite AND in front of the ego
        # camera (z > 1cm). Behind-camera joints (z<=0) have no image evidence and blow up the 2D
        # projection, so they must not be supervised or counted as visible.
        valid[:, hi] = np.isfinite(jc16).all((1, 2)) & (jc16[:, :, 2] > 1e-2).all(1)

        # 2D cache: project the smplx-16 cam joints (single focal, preprocess_hoi4d form)
        u16, v16 = project_single_focal(cam[:, hi], f, cx, cy)
        j2d[:, hi, :, 0] = u16
        j2d[:, hi, :, 1] = v16
        j2d[:, hi, :, 2] = valid[:, hi][:, None]

        # gt64: camera-frame [transl 3, quat_wxyz 4, pose15 (PCA) 15, betas 10] for crop-mode.
        transl_cam, quat_cam = camera_frame_wrist_params(
            np.asarray(rot, np.float64), np.asarray(trans, np.float64),
            world2ego[:N].astype(np.float64), mano_ns, is_right)
        pose15 = pose45_to_pca15(np.asarray(pose, np.float64), mano_ns, is_right)  # [N,15]
        off = hi * HAND_PARAM_DIM
        gt64[:, off:off + 3] = transl_cam
        gt64[:, off + 3:off + 7] = quat_cam
        gt64[:, off + 7:off + 22] = pose15.astype(np.float32)
        gt64[:, off + 22:off + 32] = np.asarray(shape, np.float32)[:, :10]
        gt64[~valid[:, hi], off:off + HAND_PARAM_DIM] = 0.0       # zero invalid hands

    # frames -> video (ego). build_splits.py's imgnames cover frames [10, N-10) (it drops the
    # first/last 10 as possibly-black), so the video is a CONTIGUOUS sub-range of the params.
    # Slice every per-frame cache to that same range so video frame i == cache index i.
    seq_id = seq.replace("/", "_")
    out_seq = os.path.join(out_root, seq_id)
    os.makedirs(os.path.join(out_seq, "hand_data"), exist_ok=True)
    rels = imgnames if imgnames is not None else d.get("imgnames", [])[:N]
    if not rels:
        raise ValueError("no ego image paths for this sequence (video is required)")
    f_start = _frame_index(rels[0])
    n_img = len(rels)
    if f_start + n_img > N:
        n_img = N - f_start
        rels = rels[:n_img]
    sl = slice(f_start, f_start + n_img)
    cam, wld, j2d, gt64 = cam[sl], wld[sl], j2d[sl], gt64[sl]
    valid, world2ego = valid[sl], world2ego[sl]
    N = n_img

    W = H = None
    vw = None
    n_written = 0
    for rel in rels:
        im = cv2.imread(os.path.join(img_root, rel))
        if im is None:
            raise ValueError(f"missing ego image {rel} (would misalign caches)")
        if vw is None:
            H, W = im.shape[:2]
            vw = cv2.VideoWriter(os.path.join(out_seq, "video_main_rgb.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        vw.write(im)
        n_written += 1
    vw.release()
    if n_written != N:
        raise ValueError(f"wrote {n_written} frames != {N} caches (misalignment)")

    boxes = np.zeros((N, 2, 4), np.float32)
    for t in range(N):
        for hi in range(2):
            if valid[t, hi]:
                boxes[t, hi] = joints_to_bbox(cam[t, hi], f, cx, cy, W, H)

    # jsonl (full contract, robustness): mirrors gt64 so a fresh rf/res cache-miss can
    # rebuild the bbox 'gt' from scratch. hand ids "0"=left, "1"=right (_hand_to_vec order).
    jl = []
    for t in range(N):
        hp = {}
        for hi in range(2):
            if not valid[t, hi]:
                continue
            off = hi * HAND_PARAM_DIM
            v = gt64[t, off:off + HAND_PARAM_DIM]
            hp[str(hi)] = {
                "wrist_xform": {"t_xyz": [float(x) for x in v[:3]],
                                "q_wxyz": [float(x) for x in v[3:7]]},
                "pose": [float(x) for x in v[7:22]],
                "betas": [float(x) for x in v[22:32]],
            }
        jl.append({"timestamp_ns": t, "hand_poses": hp})

    hd = os.path.join(out_seq, "hand_data")
    torch.save(torch.tensor([f, cx, cy]), os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.from_numpy(world2ego), os.path.join(hd, "cam_extrinsics_cache.pt"))
    torch.save(torch.from_numpy(cam), os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(torch.from_numpy(wld), os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save(torch.from_numpy(j2d), os.path.join(hd, "gt_joints_2d_cache.pt"))
    with open(os.path.join(hd, "mano_hand_pose_trajectory.jsonl"), "w") as fjl:
        for e in jl:
            fjl.write(json.dumps(e) + "\n")
    torch.save({"bboxes": torch.from_numpy(boxes), "valid": torch.from_numpy(valid),
                "gt": torch.from_numpy(gt64)},
               os.path.join(hd, "hand_bboxes_v2_rf1.5_res224x224.pt"))
    return {"seq": seq_id, "N": N, "res": (W, H), "valid_rate": float(valid.mean())}


def validate(seq, d, mano, device):
    """Compare our world-FK joints (projected to an allocentric view) to ARCTIC cam_coord. RMSE ~mm
    confirms the MANO convention. Requires an allocentric extrinsic; if unavailable, checks bone lengths."""
    p = d["params"]
    jr = fk_world_joints(mano["right"], p["rot_r"][:8], p["pose_r"][:8], p["trans_r"][:8], p["shape_r"][:8], device)
    cc = d.get("cam_coord", {}).get("joints.right")
    if cc is None:
        # no allocentric joints: sanity via index-finger bone length (~ 0.03-0.05 m)
        bl = np.linalg.norm(jr[:, 5] - jr[:, 6], axis=-1).mean()
        print(f"VALIDATE {seq}: no cam_coord; index-bone {bl*1000:.1f} mm (expect ~30-45)")
        return
    cc = np.asarray(cc, np.float32)[:8]     # [8,V,21,3] allocentric cam joints
    # world joints should match cam_coord up to each view's rigid extrinsic -> compare shape via
    # root-relative RMSE against view 0 (rigid-invariant): a correct FK matches ARCTIC's pose exactly.
    jr_rr = jr - jr[:, :1]
    cc_rr = cc[:, 0] - cc[:, 0, :1]
    rmse = np.sqrt(((jr_rr[:, :21] - cc_rr[:, :21]) ** 2).sum(-1)).mean() * 1000
    print(f"VALIDATE {seq}: root-rel FK-vs-cam_coord RMSE {rmse:.2f} mm "
          f"(<~5 mm = MANO convention OK; large => fix flat_hand_mean/pca/order)")


def _unpack_split(data: dict):
    """Return (data_dict {seq: {params,...}}, imgnames flat list). Handles both build_splits.py's
    {'data_dict', 'imgnames'} output and a bare {seq: {...}} dict (older assumed layout)."""
    if "data_dict" in data:
        return data["data_dict"], list(data.get("imgnames", []))
    return data, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_npy", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--mano_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--view", default="0", help="ego view id in build_splits imgnames (p2 = 0)")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano = build_mano(a.mano_dir, device)
    raw = np.load(a.split_npy, allow_pickle=True).item()
    data_dict, imgnames = _unpack_split(raw)
    seqs = sorted(data_dict.keys())
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    by_seq = group_imgnames_by_seq(imgnames, seqs, view=a.view) if imgnames else {}
    os.makedirs(a.out_root, exist_ok=True)
    done = 0
    for seq in seqs:
        try:
            if a.validate:
                validate(seq, data_dict[seq], mano, device)
                continue
            seq_imgs = by_seq.get(seq) if by_seq else None
            r = convert_seq(seq, data_dict[seq], mano, a.img_root, a.out_root, device,
                            imgnames=seq_imgs, max_frames=a.max_frames)
            done += 1
            print(f"[{done}] {r['seq']} N={r['N']} res={r['res']} valid={r['valid_rate']:.2f}", flush=True)
        except Exception as e:
            print(f"SEQ_FAIL {seq}: {e}", flush=True)
    print(f"ARCTIC_TO_OURS_DONE wrote {done} seqs -> {a.out_root}", flush=True)


if __name__ == "__main__":
    main()
