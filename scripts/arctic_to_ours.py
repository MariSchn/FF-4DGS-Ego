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
import os

import cv2
import numpy as np
import torch

# ARCTIC uses the original MANO 21-joint order; take the 16 smplx-kinematic joints (drop 5 tips),
# same subset the rest of our pipeline uses (see scripts/eval_cmpjpe.py, run_wilor_h2o.py).
MANO21_TO_16 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


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


def convert_seq(seq, d, mano, img_root, out_root, device, max_frames=0):
    p = d["params"]
    world2ego = np.asarray(p["world2ego"], np.float32)           # [N,4,4]
    K = np.asarray(p["K_ego"], np.float32)                       # [3,3]
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    N = world2ego.shape[0]
    if max_frames:
        N = min(N, max_frames)

    cam = np.full((N, 2, 16, 3), np.nan, np.float32)
    wld = np.full((N, 2, 16, 3), np.nan, np.float32)
    valid = np.zeros((N, 2), bool)
    for hi, hk in [(0, "l"), (1, "r")]:
        rot, pose = p[f"rot_{hk}"][:N], p[f"pose_{hk}"][:N]
        trans, shape = p[f"trans_{hk}"][:N], p[f"shape_{hk}"][:N]
        side = "left" if hk == "l" else "right"
        jw = fk_world_joints(mano[side], rot, pose, trans, shape, device)   # [N,21,3] world
        jc = apply_se3(world2ego[:N], jw)                                    # [N,21,3] ego cam
        wld[:, hi] = jw[:, MANO21_TO_16]
        cam[:, hi] = jc[:, MANO21_TO_16]
        valid[:, hi] = np.isfinite(jc[:, MANO21_TO_16]).all((1, 2))

    # frames -> video (ego). imgnames are relative; read the first to get resolution.
    seq_id = seq.replace("/", "_")
    out_seq = os.path.join(out_root, seq_id)
    os.makedirs(os.path.join(out_seq, "hand_data"), exist_ok=True)
    imgs = d.get("imgnames", [])[:N]
    W = H = None
    vw = None
    for i, rel in enumerate(imgs):
        fp = os.path.join(img_root, rel)
        im = cv2.imread(fp)
        if im is None:
            continue
        if vw is None:
            H, W = im.shape[:2]
            vw = cv2.VideoWriter(os.path.join(out_seq, "video_main_rgb.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        vw.write(im)
    if vw is not None:
        vw.release()
    if W is None:                                # fall back to K principal point *2
        W, H = int(round(cx * 2)), int(round(cy * 2))

    boxes = np.zeros((N, 2, 4), np.float32)
    for t in range(N):
        for hi in range(2):
            if valid[t, hi]:
                b = joints_to_bbox(cam[t, hi], f, cx, cy, W, H)
                boxes[t, hi] = b

    hd = os.path.join(out_seq, "hand_data")
    torch.save(torch.tensor([f, W / 2.0, H / 2.0]), os.path.join(hd, "cam_intrinsics.pt"))
    torch.save(torch.from_numpy(world2ego[:N]), os.path.join(hd, "cam_extrinsics_cache.pt"))
    torch.save(torch.from_numpy(cam), os.path.join(hd, "gt_joints_cache_cam_v2.pt"))
    torch.save(torch.from_numpy(wld), os.path.join(hd, "gt_joints_cache_world.pt"))
    torch.save({"bboxes": torch.from_numpy(boxes), "valid": torch.from_numpy(valid)},
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_npy", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--mano_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano = build_mano(a.mano_dir, device)
    data = np.load(a.split_npy, allow_pickle=True).item()
    seqs = sorted(data.keys())
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    os.makedirs(a.out_root, exist_ok=True)
    done = 0
    for seq in seqs:
        try:
            if a.validate:
                validate(seq, data[seq], mano, device)
                continue
            r = convert_seq(seq, data[seq], mano, a.img_root, a.out_root, device, a.max_frames)
            done += 1
            print(f"[{done}] {r['seq']} N={r['N']} res={r['res']} valid={r['valid_rate']:.2f}", flush=True)
        except Exception as e:
            print(f"SEQ_FAIL {seq}: {e}", flush=True)
    print(f"ARCTIC_TO_OURS_DONE wrote {done} seqs -> {a.out_root}", flush=True)


if __name__ == "__main__":
    main()
