"""Per-clip camera-pose refinement — attack the world-placement bottleneck.

The oracle-camera diagnostic showed that replacing the predicted per-clip cameras
with GT extrinsics cuts HOT3D W-MPJPE 188 -> 115 mm (= HaWoR parity), so the
per-clip CAMERA POSE is the world-placement bottleneck, not the hand head or the
chaining. The feedforward backbone predicts a clip's Gaussians AND its per-frame
cameras jointly; within a clip the static map is pose-derived, so a frame's pose
can be sharpened toward the multi-frame photometric consensus of the clip's own
static Gaussian map (the iComMa / GS-CPR family).

This module does that gradient-based, through the differentiable gsplat rasterizer:
a per-frame 6-DoF se3 delta (zero-init), optimized on a photometric (+ optional
depth) loss against the input frame, warm-started from the feedforward pose. No
backprop through the frozen backbone, no SLAM, no feature-matcher dependency.

Design choices that make this robust on a testbed we cannot run locally:
- Render ONLY the constant (static, timestamp == -1) Gaussians. The moving hand is
  excluded automatically, so the pose is constrained by the rigid scene (what we want).
- Auto-detect the viewmat convention (c2w vs w2c) per clip by rendering at the
  feedforward pose under both and keeping whichever matches the input image. gsplat
  wants world->camera; the eval treats rendered_extrinsics as camera->world.
- A `sanity` path reports the feedforward-pose render PSNR so the plumbing can be
  validated on-GPU BEFORE any refinement is trusted. If the feedforward render does
  not match the input, refinement is meaningless and we debug the convention first.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def _skew(w: torch.Tensor) -> torch.Tensor:
    """[3] -> [3,3] skew-symmetric matrix."""
    O = w.new_zeros(())
    return torch.stack([
        torch.stack([O, -w[2], w[1]]),
        torch.stack([w[2], O, -w[0]]),
        torch.stack([-w[1], w[0], O]),
    ])


def _exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """se(3) exponential. xi = [omega(3), v(3)] -> [4,4] SE(3). Differentiable, small-angle safe."""
    omega, v = xi[:3], xi[3:]
    theta = torch.linalg.norm(omega).clamp_min(1e-12)
    W = _skew(omega)
    W2 = W @ W
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    s, c = torch.sin(theta), torch.cos(theta)
    # Rodrigues; the (1-cos)/theta^2 and (theta-sin)/theta^3 terms are finite as theta->0.
    A = s / theta
    B = (1.0 - c) / (theta * theta)
    C = (theta - s) / (theta ** 3)
    R = eye + A * W + B * W2
    V = eye + B * W + C * W2
    t = V @ v
    T = torch.eye(4, dtype=xi.dtype, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _constant_splats(splats_b: List) -> List:
    """Keep only constant (timestamp == -1) Gaussians = the rigid scene map. Falls back to
    every Gaussian if a clip has no constant ones (fully dynamic), so we never render empty."""
    const = [g for g in splats_b if getattr(g, "timestamp", -1) == -1 and g.means.shape[0] > 0]
    return const if const else [g for g in splats_b if g.means.shape[0] > 0]


def _render(rasterizer, const_splats, w2c: torch.Tensor, K: torch.Tensor,
            H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render the static map from a world->camera viewmat. Returns (rgb [H,W,3], alpha [H,W,1])."""
    rgb, _depth, alpha = rasterizer.rasterize_splats(
        const_splats, w2c[None], K[None], width=W, height=H, sh_degree=0,
    )
    return rgb[0], alpha[0]


def _photo_loss(rgb: torch.Tensor, alpha: torch.Tensor, img_hw3: torch.Tensor,
                ignore: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked L1 over rendered (alpha-covered) pixels. `ignore` [H,W] bool downweights e.g. the
    hand region. Returns a scalar; large/NaN-safe (returns 0-grad tensor if no valid pixels)."""
    m = (alpha[..., 0] > 0.05).float()
    if ignore is not None:
        m = m * (~ignore).float()
    denom = m.sum().clamp_min(1.0)
    return ((rgb - img_hw3).abs().mean(dim=-1) * m).sum() / denom


def _psnr(rgb: torch.Tensor, alpha: torch.Tensor, img_hw3: torch.Tensor) -> float:
    m = (alpha[..., 0] > 0.05).float()
    denom = m.sum().clamp_min(1.0)
    mse = (((rgb - img_hw3) ** 2).mean(dim=-1) * m).sum() / denom
    return float(-10.0 * torch.log10(mse.clamp_min(1e-8)))


def _hand_ignore_mask(hand_bboxes: Optional[torch.Tensor], frame: int, H: int, W: int,
                      device) -> Optional[torch.Tensor]:
    """[H,W] bool, True inside any valid hand bbox for this frame, so the moving hand/arm does
    not pull the camera pose. hand_bboxes: [S,2,4] xyxy at model resolution (or None)."""
    if hand_bboxes is None:
        return None
    boxes = hand_bboxes[frame]                      # [2,4]
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    for b in boxes:
        x0, y0, x1, y1 = [int(round(float(v))) for v in b[:4]]
        if x1 <= x0 or y1 <= y0:
            continue
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _detect_convention(rasterizer, const_splats, ext: torch.Tensor, K: torch.Tensor,
                       img_hw3: torch.Tensor, H: int, W: int) -> Tuple[str, float, float]:
    """Decide whether `ext` (a clip-pose row) is c2w or already w2c, by rendering the static map
    under both and keeping whichever matches the input frame. Returns (conv, psnr_c2w, psnr_w2c).
    conv == 'c2w' means the eval pose is camera->world (gsplat viewmat = inverse(ext))."""
    with torch.no_grad():
        rgb_a, al_a = _render(rasterizer, const_splats, torch.inverse(ext), K, H, W)  # ext as c2w
        rgb_b, al_b = _render(rasterizer, const_splats, ext, K, H, W)                 # ext as w2c
        p_c2w, p_w2c = _psnr(rgb_a, al_a, img_hw3), _psnr(rgb_b, al_b, img_hw3)
    return ("c2w" if p_c2w >= p_w2c else "w2c"), p_c2w, p_w2c


def refine_clip_poses(
    rasterizer,
    splats_b: List,
    ext: torch.Tensor,                  # [S,4,4] preds["rendered_extrinsics"][0] (treated as c2w by the eval)
    K: torch.Tensor,                    # [S,3,3] preds["rendered_intrinsics"][0]
    images: torch.Tensor,               # [S,3,H,W] input frames in [0,1]
    hand_bboxes: Optional[torch.Tensor] = None,  # [S,2,4] xyxy @ model res, to ignore the hand
    iters: int = 40,
    lr: float = 3e-3,
    frame_stride: int = 1,
    sanity: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Sharpen each frame's camera pose against the clip's own static Gaussian map.

    Returns (refined_ext [S,4,4] in the SAME convention as the input `ext`, info). When `sanity`
    is True it skips optimization and only reports the feedforward-pose render PSNR per refined
    frame (cheap plumbing/convention check). Refinement keeps the optimized se3 delta only if it
    LOWERS the photometric loss vs the feedforward pose (never makes a frame worse)."""
    device = ext.device
    S = ext.shape[0]
    H, W = images.shape[-2:]
    const = _constant_splats(splats_b)
    refined = ext.clone()
    info: Dict = {"conv": None, "psnr_ff": [], "psnr_ref": [], "loss_ff": [], "loss_ref": [],
                  "improved": 0, "n_frames": 0, "n_gauss": int(sum(g.means.shape[0] for g in const))}

    if len(const) == 0 or info["n_gauss"] == 0:
        info["error"] = "no_constant_gaussians"
        return refined, info

    img0 = images[0].permute(1, 2, 0).float()       # [H,W,3]
    conv, p_c2w, p_w2c = _detect_convention(rasterizer, const, ext[0].float(), K[0].float(), img0, H, W)
    info["conv"], info["psnr_detect"] = conv, {"as_c2w": p_c2w, "as_w2c": p_w2c}

    def to_w2c(pose_row: torch.Tensor) -> torch.Tensor:
        return torch.inverse(pose_row) if conv == "c2w" else pose_row

    frames = list(range(0, S, max(1, frame_stride)))
    for k in frames:
        img_k = images[k].permute(1, 2, 0).float()
        Kk = K[k].float()
        ext_k = ext[k].float()
        ignore = _hand_ignore_mask(hand_bboxes, k, H, W, device)
        with torch.no_grad():
            rgb_ff, al_ff = _render(rasterizer, const, to_w2c(ext_k), Kk, H, W)
            loss_ff = float(_photo_loss(rgb_ff, al_ff, img_k, ignore))
            psnr_ff = _psnr(rgb_ff, al_ff, img_k)
        info["psnr_ff"].append(psnr_ff)
        info["loss_ff"].append(loss_ff)
        info["n_frames"] += 1
        if sanity:
            info["psnr_ref"].append(psnr_ff)
            info["loss_ref"].append(loss_ff)
            continue

        # Optimize a zero-init se3 delta applied in the camera body frame: c2w' = c2w @ exp(delta).
        delta = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        c2w_k = ext_k if conv == "c2w" else torch.inverse(ext_k)   # normalize to c2w for the body-frame update
        best_loss, best_delta = loss_ff, delta.detach().clone()
        with torch.enable_grad():
            for _ in range(iters):
                opt.zero_grad()
                c2w_ref = c2w_k @ _exp_se3(delta)
                w2c_ref = torch.inverse(c2w_ref)
                rgb, al = _render(rasterizer, const, w2c_ref, Kk, H, W)
                loss = _photo_loss(rgb, al, img_k, ignore)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
                lv = float(loss.detach())
                if lv < best_loss:
                    best_loss, best_delta = lv, delta.detach().clone()

        # Accept only if it helped; write back in the ORIGINAL convention the eval expects.
        if best_loss < loss_ff - 1e-6:
            c2w_best = (c2w_k @ _exp_se3(best_delta)).detach()
            refined[k] = c2w_best if conv == "c2w" else torch.inverse(c2w_best)
            info["improved"] += 1
            with torch.no_grad():
                rgb_r, al_r = _render(rasterizer, const, to_w2c(refined[k].float()), Kk, H, W)
                info["psnr_ref"].append(_psnr(rgb_r, al_r, img_k))
        else:
            info["psnr_ref"].append(psnr_ff)
        info["loss_ref"].append(best_loss)

    # Summaries for one-line logging.
    if info["psnr_ff"]:
        info["psnr_ff_mean"] = float(sum(info["psnr_ff"]) / len(info["psnr_ff"]))
        info["psnr_ref_mean"] = float(sum(info["psnr_ref"]) / len(info["psnr_ref"]))
    return refined, info
