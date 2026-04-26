"""Shared image-quality metrics for the WorldMirror Gaussian-Splatting head.

The train-time validation loop in `scripts.train_hand_head.run_validation` and
the offline benchmark in `scripts.eval_gs_head` both import from this module,
so the numbers reported during training match the eval script bit-for-bit
(given the same val sequences and the same checkpoint).

Pipeline:
    scorer = LPIPSScorer(device=device)
    chunks = []
    for batch in loader:
        # forward pass to populate preds["splats"] / rendered_extrinsics / etc.
        rendered = render_views_from_predictions(model, preds, views, H, W)
        chunks.append(metric_chunks_from_batch(rendered, views["img"],
                                               valid_mask, scorer, device))
    result = metrics_from_chunks(chunks)
    # result["PSNR"], result["SSIM"], result["LPIPS"], result["num_valid_frames"]
"""

import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

from diffsynth.utils.auxiliary import homo_matrix_inverse


# ------------------------------------------------------------------
# LPIPS scorer
# ------------------------------------------------------------------

class LPIPSScorer:
    """Lazily-built LPIPS network reused across batches.

    Both the training loop and the eval script construct one of these
    once and pass it into `metric_chunks_from_batch`.
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.net = net
        self.device = device
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = lpips.LPIPS(net=self.net).to(self.device)
            self._model.eval()
        return self._model

    @torch.no_grad()
    def score(self, pred_01: torch.Tensor, gt_01: torch.Tensor) -> torch.Tensor:
        """pred_01 / gt_01: [N, 3, H, W] in [0, 1] -> [N] LPIPS in fp32 on CPU."""
        model = self._ensure()
        pred = (pred_01.to(self.device) * 2.0 - 1.0).float()
        gt   = (gt_01.to(self.device)   * 2.0 - 1.0).float()
        d = model(pred, gt).reshape(-1).float().cpu()
        return d


# ------------------------------------------------------------------
# Rendering: turn preds["splats"] + cameras into RGB views
# ------------------------------------------------------------------

def render_views_from_predictions(model, preds, views, height: int, width: int):
    """Re-render the input views from the predicted Gaussians.

    Mirrors the rasterizer call in `scripts/reconstruct_4dgs.py:208-216`:
    the model's `_gen_all_preds` already populates `splats`,
    `rendered_extrinsics`, `rendered_intrinsics`, `rendered_timestamps`
    when `enable_gs=True`, so we just rasterize.

    Args:
        model: WorldMirror with `enable_gs=True`.
        preds: dict returned by `model(views, ...)` — must contain
            `splats`, `rendered_extrinsics`, `rendered_intrinsics`,
            `rendered_timestamps`.
        views: the `views` dict that was fed into the model. Unused here
            but kept in the signature to match the natural call site.
        height, width: render resolution (typically input H/W).

    Returns:
        Tensor [B, S, H, W, 3] in [0, 1] (fp32).
    """
    if "splats" not in preds:
        raise RuntimeError(
            "preds is missing 'splats' — was the model built with "
            "enable_gs=True? Check `model.enable_gs` and the config."
        )

    splats = preds["splats"]                       # List[List[Gaussians]], len B
    cam2world = preds["rendered_extrinsics"]       # [B, S, 4, 4]
    intrinsics = preds["rendered_intrinsics"]      # [B, S, 3, 3]
    timestamps = preds["rendered_timestamps"]      # [B, S]
    world2cam = homo_matrix_inverse(cam2world)     # [B, S, 4, 4]

    B = len(splats)
    rendered, _depth, _alpha = model.gs_renderer.rasterizer.forward(
        splats,
        render_viewmats=[world2cam[b] for b in range(B)],
        render_Ks=[intrinsics[b] for b in range(B)],
        render_timestamps=[timestamps[b] for b in range(B)],
        sh_degree=0,
        width=width,
        height=height,
    )
    # rendered: [B, S, H, W, 3] in [0, 1]
    return rendered.clamp(0.0, 1.0).float()


# ------------------------------------------------------------------
# Per-batch chunk + final aggregation (the shared interface)
# ------------------------------------------------------------------

def _psnr_per_frame(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred/gt: [N, 3, H, W] in [0, 1] -> [N] PSNR (fp32, on CPU)."""
    mse = ((pred.float() - gt.float()) ** 2).mean(dim=(1, 2, 3))
    out = torch.where(
        mse > 0,
        10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)),
        torch.full_like(mse, float("inf")),
    )
    return out.cpu()


def _ssim_per_frame(pred_hwc: np.ndarray, gt_hwc: np.ndarray) -> float:
    return float(ssim_fn(pred_hwc, gt_hwc, data_range=1.0, channel_axis=2))


@torch.no_grad()
def metric_chunks_from_batch(rendered, gt_imgs, valid_mask, lpips_scorer, device):
    """Convert a single batch's rendered/GT views into flat CPU chunks.

    Args:
        rendered:    [B, S, H, W, 3] float in [0, 1] (output of
                     `render_views_from_predictions`).
        gt_imgs:     [B, S, 3, H, W] float in [0, 1] (i.e. `views["img"]`).
        valid_mask:  [B, S] bool tensor or None (None == all valid).
        lpips_scorer: LPIPSScorer.

    Returns:
        dict of CPU tensors:
            psnr:  [B*S]
            ssim:  [B*S]
            lpips: [B*S]
            valid: [B*S] bool
    """
    B, S = rendered.shape[:2]
    pred_chw = rendered.permute(0, 1, 4, 2, 3).reshape(B * S, 3, *rendered.shape[2:4])
    gt_chw   = gt_imgs.reshape(B * S, *gt_imgs.shape[2:])
    pred_chw = pred_chw.float().cpu()
    gt_chw   = gt_chw.float().cpu()

    psnr = _psnr_per_frame(pred_chw, gt_chw)

    ssim_vals = []
    for i in range(B * S):
        p = pred_chw[i].permute(1, 2, 0).numpy()
        g = gt_chw[i].permute(1, 2, 0).numpy()
        ssim_vals.append(_ssim_per_frame(p, g))
    ssim = torch.tensor(ssim_vals, dtype=torch.float32)

    lpips_vals = lpips_scorer.score(pred_chw, gt_chw)

    if valid_mask is None:
        valid = torch.ones(B * S, dtype=torch.bool)
    else:
        valid = valid_mask.reshape(-1).cpu().bool()

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_vals, "valid": valid}


def metrics_from_chunks(chunks):
    """Concatenate per-batch chunks and produce mean PSNR / SSIM / LPIPS.

    Returns:
        {"num_valid_frames": int,
         "PSNR": float, "SSIM": float, "LPIPS": float}
        or {"num_valid_frames": 0, "PSNR": None, "SSIM": None, "LPIPS": None}
        if there are no valid frames.
    """
    if not chunks:
        return {"num_valid_frames": 0, "PSNR": None, "SSIM": None, "LPIPS": None}

    cat = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    valid = cat["valid"]
    n = int(valid.sum().item())
    if n == 0:
        return {"num_valid_frames": 0, "PSNR": None, "SSIM": None, "LPIPS": None}

    psnr = cat["psnr"][valid]
    finite_psnr = psnr[torch.isfinite(psnr)]
    psnr_mean = float(finite_psnr.mean().item()) if finite_psnr.numel() > 0 else float("inf")

    return {
        "num_valid_frames": n,
        "PSNR":  psnr_mean,
        "SSIM":  float(cat["ssim"][valid].mean().item()),
        "LPIPS": float(cat["lpips"][valid].mean().item()),
    }
