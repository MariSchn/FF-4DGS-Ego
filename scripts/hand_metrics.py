"""Shared HaMeR-style hand-pose metrics.

The train-time validation loop in `scripts.train_hand_head.run_validation` and
the offline benchmark in `scripts.eval_hand_head` both import from this module,
so the numbers reported during training match the eval script bit-for-bit
(given the same val sequences and the same checkpoint).

Pipeline:
    chunks = []
    for batch in loader:
        # forward pass to get pred_params [B,S,64], gt_params [B,S,64]
        chunks.append(metric_chunks_from_batch(pred_params, gt_params, hv,
                                               mano_model, device))
    result = metrics_from_chunks(chunks)
    # result["all"]["PA_MPJPE"], etc.
"""

import torch

from scripts.hand_vis_utils import quat_wxyz_to_axis_angle_torch


HAND_PARAM_DIM = 32
NUM_HANDS = 2
AUC_THR_MIN_MM = 0.0
AUC_THR_MAX_MM = 50.0
AUC_THR_STEPS  = 51


# ------------------------------------------------------------------
# Procrustes
# ------------------------------------------------------------------

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """Batched similarity-transform alignment (sR + t) of S1 onto S2.

    Adapted from `hamer/utils/pose_utils.py` (which is itself adapted from HMR).
    Inlined here to avoid pulling in hamer's package-level imports (pyrender, etc.).

    Args:
        S1, S2: (B, N, 3) point sets.
    Returns:
        S1_hat: (B, N, 3) — S1 transformed to best match S2.
    """
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.matmul(X1, X2.permute(0, 2, 1))
    U, _, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(S1.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(-1).unsqueeze(-1)
    t = mu2 - scale * torch.matmul(R, mu1)
    return (scale * torch.matmul(R, S1) + t).permute(0, 2, 1)


# ------------------------------------------------------------------
# MANO joints + vertices from packed 64-D params
# ------------------------------------------------------------------

def _layer_joints_and_vertices(mano_model, params_n32, is_right, device):
    """Run smplx layer on [N, 32] params; return (joints[N,16,3], verts[N,778,3])."""
    mano_model._ensure_device(device)
    p = params_n32.to(device)
    transl = p[:, 0:3]
    quat   = p[:, 3:7]
    pose   = p[:, 7:22]
    betas  = p[:, 22:32]
    rotvec = quat_wxyz_to_axis_angle_torch(quat)
    layer = mano_model.right if is_right else mano_model.left
    out = layer(betas=betas, global_orient=rotvec, hand_pose=pose,
                transl=transl, return_verts=True)
    return out.joints, out.vertices


def joints_and_vertices_from_params(params_bs64, mano_model, device):
    """[B,S,64] -> joints [B,S,2,16,3], vertices [B,S,2,778,3]."""
    B, S, D = params_bs64.shape
    assert D == NUM_HANDS * HAND_PARAM_DIM
    flat = params_bs64.reshape(B * S, NUM_HANDS, HAND_PARAM_DIM)
    jL, vL = _layer_joints_and_vertices(mano_model, flat[:, 0], is_right=False, device=device)
    jR, vR = _layer_joints_and_vertices(mano_model, flat[:, 1], is_right=True,  device=device)
    joints = torch.stack([jL, jR], dim=1).view(B, S, NUM_HANDS, jL.shape[-2], 3)
    verts  = torch.stack([vL, vR], dim=1).view(B, S, NUM_HANDS, vL.shape[-2], 3)
    return joints, verts


# ------------------------------------------------------------------
# Metric primitives (inputs in meters; mm-based outputs)
# ------------------------------------------------------------------

def _per_point_l2_mm(pred, gt):
    """(M, N, 3) meters -> (M, N) per-point distances in mm."""
    return 1000.0 * torch.sqrt(((pred - gt) ** 2).sum(-1))


def metrics_for(pred, gt):
    """MPJPE/MPVPE-style scalars + AUC over PA-aligned per-point error."""
    raw_err_mm = _per_point_l2_mm(pred, gt)
    pred_pa = compute_similarity_transform(pred, gt)
    pa_err_mm = _per_point_l2_mm(pred_pa, gt)

    thresholds = torch.linspace(
        AUC_THR_MIN_MM, AUC_THR_MAX_MM, AUC_THR_STEPS, device=pa_err_mm.device,
    )
    pck = (pa_err_mm.unsqueeze(-1) <= thresholds).float().mean(dim=(0, 1))
    auc = (torch.trapz(pck, thresholds) / (thresholds[-1] - thresholds[0])).item()

    return {
        "mean_mm":    raw_err_mm.mean().item(),
        "pa_mean_mm": pa_err_mm.mean().item(),
        "auc":        auc,
    }


def aggregate(pred_j, gt_j, pred_v, gt_v):
    """{MPJPE, PA_MPJPE, AUC_J, MPVPE, PA_MPVPE, AUC_V}."""
    j = metrics_for(pred_j, gt_j)
    v = metrics_for(pred_v, gt_v)
    return {
        "MPJPE":    j["mean_mm"],
        "PA_MPJPE": j["pa_mean_mm"],
        "AUC_J":    j["auc"],
        "MPVPE":    v["mean_mm"],
        "PA_MPVPE": v["pa_mean_mm"],
        "AUC_V":    v["auc"],
    }


# ------------------------------------------------------------------
# Per-batch chunk + final aggregation (the shared interface)
# ------------------------------------------------------------------

@torch.no_grad()
def metric_chunks_from_batch(pred_params, gt_params, hand_valid, mano_model, device):
    """Convert a single batch's pred/GT params into flat CPU chunks.

    Both `eval_hand_head` and `train_hand_head` call this on every batch and
    accumulate the returned dicts in a list, then pass that list to
    `metrics_from_chunks`.

    Args:
        pred_params: [B, S, 64] tensor on `device`.
        gt_params:   [B, S, 64] tensor on `device`.
        hand_valid:  [B, S, 2] bool tensor or None (None == all valid).
        mano_model:  MANOModel.
    Returns:
        dict of CPU tensors:
            pred_j:  [B*S*2, 16,  3]
            gt_j:    [B*S*2, 16,  3]
            pred_v:  [B*S*2, 778, 3]
            gt_v:    [B*S*2, 778, 3]
            side:    [B*S*2]    (0=left, 1=right)
            valid:   [B*S*2] bool
    """
    pj, pv = joints_and_vertices_from_params(pred_params, mano_model, device)
    gj, gv = joints_and_vertices_from_params(gt_params,   mano_model, device)

    B, S, _ = pred_params.shape
    if hand_valid is None:
        hand_valid = torch.ones(B, S, NUM_HANDS, dtype=torch.bool, device=device)

    side = torch.tensor([0, 1], device=device).view(1, 1, 2).expand(B, S, 2).reshape(-1)

    return {
        "pred_j": pj.reshape(-1, pj.shape[-2], 3).cpu(),
        "gt_j":   gj.reshape(-1, gj.shape[-2], 3).cpu(),
        "pred_v": pv.reshape(-1, pv.shape[-2], 3).cpu(),
        "gt_v":   gv.reshape(-1, gv.shape[-2], 3).cpu(),
        "side":   side.cpu(),
        "valid":  hand_valid.reshape(-1).cpu().bool(),
    }


def metrics_from_chunks(chunks):
    """Concatenate per-batch chunks and produce per-side and combined metrics.

    Returns:
        {"num_valid_hands": int,
         "left":  {MPJPE, PA_MPJPE, MPVPE, PA_MPVPE, AUC_J, AUC_V} or None,
         "right": ...,
         "all":   ...}
    """
    if not chunks:
        return {"num_valid_hands": 0, "left": None, "right": None, "all": None}

    cat = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    valid = cat["valid"]
    side  = cat["side"]

    out = {"num_valid_hands": int(valid.sum().item())}
    for name, sel in [("left",  side == 0),
                      ("right", side == 1),
                      ("all",   torch.ones_like(side, dtype=torch.bool))]:
        mask = valid & sel
        if mask.sum() == 0:
            out[name] = None
            continue
        out[name] = aggregate(
            cat["pred_j"][mask], cat["gt_j"][mask],
            cat["pred_v"][mask], cat["gt_v"][mask],
        )
    return out
