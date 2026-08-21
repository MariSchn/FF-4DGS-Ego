"""Pull the Gaussians in the hand region onto the MANO surface.

The depth anchor (``hand_depth_anchor_loss``) reads ``gs_depth`` at the projected hand joints and
compares it to the hand's own depth. That read is contaminated: where the reconstructor does not
model the thin foreground hand it returns the background behind it, which is strictly farther, so
the residual is biased in one direction and the term trains toward a target that is wrong wherever
the hand is not in contact with a surface. Run 11176291 held for 100 steps under that term and then
reached 1.3e7 m.

This states the constraint the other way round, following the hand loss of Tian et al.: the
Gaussians that fall in the hand region must lie ON the MANO surface. Nothing is read from the
model's own depth, so there is nothing to contaminate, the target is defined whether or not the
hand touches anything, and a cloud that has run away in scale is penalised in proportion to how far
it ran rather than saturating.

Gradient flows to the Gaussian positions only. MANO is the reference and is detached.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

Z_MIN = 0.05          # metres; matches hand_depth_sampling.Z_MIN
N_SAMPLES = 512       # pixels sampled per hand; fixed so every hand shares one cdist


def unproject_gs_depth(gs_depth: torch.Tensor, cam_intr: torch.Tensor,
                       image_width: float) -> torch.Tensor:
    """Camera-frame points from the stored depth map, inverting the forward projection exactly.

    ``project_joints_to_norm_pixels`` maps a camera point to ``col = f x/z + cx``,
    ``row = f y/z + cy`` and then stores it at width ``u = (W-1) - row`` and height ``v = col``.
    Inverting that mapping, the depth held at index ``[v, u]`` belongs to the camera point with
    ``col = v`` and ``row = (W-1) - u``. Getting this backwards puts every point in an unrelated
    part of the frame while still returning a plausible tensor.

    Args:
        gs_depth:    [B, S, H, W] positive depth in the stored (rotated) layout.
        cam_intr:    [B, 3] = [focal, cx, cy].
        image_width: the pixel frame the intrinsics are expressed in.
    Returns:
        [B, S, H, W, 3] camera-frame points in metres.
    """
    B, S, H, W = gs_depth.shape
    f = cam_intr[:, 0].view(B, 1, 1, 1)
    cx = cam_intr[:, 1].view(B, 1, 1, 1)
    cy = cam_intr[:, 2].view(B, 1, 1, 1)
    dev, dt = gs_depth.device, gs_depth.dtype
    # index grids scaled from the map's own resolution to the intrinsics' pixel frame
    vi = torch.arange(H, device=dev, dtype=dt).view(1, 1, H, 1) * (image_width / H)
    ui = torch.arange(W, device=dev, dtype=dt).view(1, 1, 1, W) * (image_width / W)
    col, row = vi.expand(B, S, H, W), (image_width - 1.0) - ui.expand(B, S, H, W)
    z = gs_depth.clamp_min(Z_MIN)
    return torch.stack([(col - cx) * z / f, (row - cy) * z / f, z], dim=-1)


def hand_gaussian_chamfer_loss(
    gs_depth: torch.Tensor,
    pred_verts: torch.Tensor,
    hand_bboxes: torch.Tensor,
    has_hand: torch.Tensor,
    cam_intr: torch.Tensor,
    image_width: float,
    *,
    keep_frac: float = 0.8,
    n_samples: int = N_SAMPLES,
    detach_hand: bool = True,
) -> tuple[torch.Tensor, dict]:
    """One-sided Chamfer from hand-region Gaussians to the nearest MANO vertex.

    Args:
        gs_depth:    [B, S, 1, H, W] or [B, S, H, W, 1] or [B, S, H, W] positive scene depth.
        pred_verts:  [B, S, 2, V, 3] camera-frame metric MANO vertices (m).
        hand_bboxes: [B, S, 2, 4] xyxy in normalised [0, 1] image coordinates.
        has_hand:    [B, S, 2] in {0, 1}.
        cam_intr:    [B, 3] = [focal, cx, cy].
        image_width: the pixel frame of the intrinsics.
        keep_frac:   fraction of the closest hand-region pixels kept. A hand box also contains
                     the object and the surface behind it, whose Gaussians legitimately do not lie
                     on MANO, so the tail is trimmed. Trimming the SELECTION rather than clamping
                     the VALUE matters: a clamp saturates, and a saturated term cannot pull back a
                     cloud that has run away by orders of magnitude, which is how the depth anchor
                     failed. Under trimming every kept distance still grows with the error.
        detach_hand: keep MANO fixed and move only the Gaussians.

    Returns:
        (loss, info). Loss is 0 when no hand region is valid, never NaN.
    """
    d = gs_depth
    if d.dim() == 5:
        d = d[:, :, 0] if d.shape[2] == 1 else d[..., 0]
    if d.dim() != 4:
        raise ValueError(f"gs_depth must reduce to [B,S,H,W], got {tuple(gs_depth.shape)}")

    B, S, H, W = d.shape
    pts = unproject_gs_depth(d, cam_intr, image_width)          # [B,S,H,W,3], grad -> gs_depth
    verts = pred_verts.detach() if detach_hand else pred_verts

    # One batched cdist, not one per hand per frame. The first version looped over B*S*2 = 64
    # hands and paid a kernel launch for each; at B=2, S=16 that cost 10.06 s/it against 2.07
    # without this term. Sampling a fixed count per hand makes the shapes uniform so the whole
    # batch goes through torch.cdist once.
    boxes_cpu = hand_bboxes.detach().float().cpu()
    has_cpu = has_hand.detach().bool().cpu()

    idx_b, idx_s, idx_h, spans = [], [], [], []
    for b in range(B):
        for s_ in range(S):
            for h in range(2):
                if not bool(has_cpu[b, s_, h]):
                    continue
                x0, y0, x1, y1 = boxes_cpu[b, s_, h].tolist()
                # the box is normalised to the image and is NOT clamped to it upstream
                c0, c1 = int(max(0.0, x0) * W), int(min(1.0, x1) * W)
                r0, r1 = int(max(0.0, y0) * H), int(min(1.0, y1) * H)
                if c1 <= c0 or r1 <= r0:
                    continue
                idx_b.append(b); idx_s.append(s_); idx_h.append(h)
                spans.append((r0, r1, c0, c1))

    if not spans:
        return d.new_zeros(()), {"chamfer_m": 0.0, "n_pairs": 0}

    K, dev = len(spans), d.device
    span = torch.tensor(spans, device=dev, dtype=torch.long)              # [K,4] r0 r1 c0 c1
    rr = torch.rand(K, n_samples, device=dev)
    cc = torch.rand(K, n_samples, device=dev)
    rows = (span[:, 0:1] + rr * (span[:, 1:2] - span[:, 0:1])).long().clamp_(0, H - 1)
    cols = (span[:, 2:3] + cc * (span[:, 3:4] - span[:, 2:3])).long().clamp_(0, W - 1)

    bi = torch.tensor(idx_b, device=dev, dtype=torch.long).unsqueeze(1).expand(K, n_samples)
    si = torch.tensor(idx_s, device=dev, dtype=torch.long).unsqueeze(1).expand(K, n_samples)
    p = pts[bi.reshape(-1), si.reshape(-1), rows.reshape(-1), cols.reshape(-1)]
    p = p.view(K, n_samples, 3)                                          # [K,P,3], grad -> gs_depth

    hi = torch.tensor(idx_h, device=dev, dtype=torch.long)
    v = verts[bi[:, 0], si[:, 0], hi]                                    # [K,V,3]

    near = torch.cdist(p, v).min(dim=2).values                           # [K,P] metres
    keep = max(1, int(keep_frac * n_samples))
    near = near.topk(keep, dim=1, largest=False).values                  # trim the tail
    ok = torch.isfinite(near)
    if not bool(ok.any()):
        return d.new_zeros(()), {"chamfer_m": 0.0, "n_pairs": 0}
    n_pairs = int(ok.sum())
    loss = (near[ok] ** 2).sum() / n_pairs
    with torch.no_grad():
        med = float(near[ok].median())
    return loss, {"chamfer_m": med, "n_pairs": n_pairs}


def _selftest() -> None:
    """A depth map holding a plane at the vertex depth must score near zero, and the same map
    scaled away must score worse in proportion, which is what the saturating depth anchor did not."""
    torch.manual_seed(0)
    B, S, H, W = 1, 2, 64, 64
    Wpx = 224.0
    cam_intr = torch.tensor([[220.0, 112.0, 112.0]])
    depth = torch.full((B, S, H, W), 0.70)
    pts = unproject_gs_depth(depth, cam_intr, Wpx)
    # take the MANO stand-in FROM the unprojected centre so the two agree by construction
    verts = pts[:, :, H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4].reshape(B, S, 1, -1, 3)
    verts = verts.expand(B, S, 2, verts.shape[-2], 3).contiguous()
    boxes = torch.tensor([[0.45, 0.45, 0.55, 0.55]]).view(1, 1, 1, 4).expand(B, S, 2, 4).contiguous()
    has = torch.ones(B, S, 2)

    d0 = depth.clone().requires_grad_(True)
    l0, i0 = hand_gaussian_chamfer_loss(d0, verts, boxes, has, cam_intr, Wpx)
    l0.backward()
    assert i0["n_pairs"] > 0, "no hand-region pixels selected"
    assert i0["chamfer_m"] < 0.02, f"aligned case should be near zero, got {i0['chamfer_m']:.4f}"
    assert d0.grad is not None and torch.isfinite(d0.grad).all(), "no finite gradient to gs_depth"

    worse = []
    for k in (1.0, 2.0, 10.0, 1e3):
        l, i = hand_gaussian_chamfer_loss(depth * k, verts, boxes, has, cam_intr, Wpx)
        worse.append(float(l))
    assert worse == sorted(worse), f"loss must grow with the scale error, got {worse}"
    print(f"hand_gaussian_chamfer_loss self-test: OK (aligned {i0['chamfer_m']*1000:.1f} mm, "
          f"n={i0['n_pairs']}, loss vs scale {['%.4f' % w for w in worse]})")


if __name__ == "__main__":
    _selftest()
