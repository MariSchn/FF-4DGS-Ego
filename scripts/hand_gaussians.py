"""G_hand: Gaussians anchored to the MANO surface, in the world frame the scene splats live in.

The scene branch predicts positions freely; the hand branch must not. Centres are the 778 MANO
vertices carried to world through the frame's camera, colours are sampled from the input image at
each vertex's projection, exactly the convention the scene Gaussians use for their base colour.
Rotations are identity and scales isotropic in this v0: appearance detail is not the claim under
test, geometry ownership is.

One Gaussians set per frame, stamped with that frame's timestamp, so the rasterizer draws each
hand only into its own view. A held-out frame's hand comes from `mano_interp`, which is the
caller's responsibility, and is what makes the hand the clip's motion model.
"""
from __future__ import annotations

import torch

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.auxiliary_models.worldmirror.models.utils.sh_utils import RGB2SH


def build_hand_gaussians(verts_cam: torch.Tensor, hand_valid: torch.Tensor,
                         c2w: torch.Tensor, images: torch.Tensor, intr: torch.Tensor,
                         timestamps: torch.Tensor,
                         scale_m: float = 0.0025, opacity: float = 0.95) -> list:
    """Per-frame hand Gaussians for one batch element.

    Args:
        verts_cam:  [S, 2, V, 3] metric camera-frame MANO vertices.
        hand_valid: [S, 2] bool.
        c2w:        [S, 4, 4] camera-to-world.
        images:     [S, 3, H, W] in [0, 1].
        intr:       [3] (f, cx, cy) on the render frame.
        timestamps: [S] the frame timestamps the rest of the clip uses.

    Returns:
        list of `Gaussians`, one per frame that has at least one valid hand.
    """
    S, Hn, V, _ = verts_cam.shape
    H, W = images.shape[-2:]
    f, cx, cy = float(intr[0]), float(intr[1]), float(intr[2])
    out = []
    for s in range(S):
        keep = hand_valid[s].bool()
        if not bool(keep.any()):
            continue
        v = verts_cam[s][keep].reshape(-1, 3)                      # [N, 3] camera frame
        z = v[:, 2].clamp_min(1e-3)
        u_px = (f * v[:, 0] / z + cx).round().long().clamp(0, W - 1)
        v_px = (f * v[:, 1] / z + cy).round().long().clamp(0, H - 1)
        rgb = images[s, :, v_px, u_px].transpose(0, 1)             # [N, 3]

        R, t = c2w[s, :3, :3], c2w[s, :3, 3]
        means_w = v @ R.T + t

        n = means_w.shape[0]
        dev, dt = means_w.device, means_w.dtype
        quats = torch.zeros(n, 4, device=dev, dtype=dt); quats[:, 0] = 1.0
        out.append(Gaussians(
            means=means_w,
            harmonics=RGB2SH(rgb).unsqueeze(-2),                   # [N, 1, 3], sh degree 0
            opacities=torch.full((n,), opacity, device=dev, dtype=dt),
            scales=torch.full((n, 3), scale_m, device=dev, dtype=dt),
            rotations=quats,
            timestamp=int(timestamps[s]),
        ))
    return out


def _selftest() -> None:
    S, V, H, W = 2, 5, 8, 8
    verts = torch.zeros(S, 2, V, 3); verts[..., 2] = 0.5           # on-axis at 50 cm
    verts[0, 0, 1] = torch.tensor([0.1, 0.0, 0.5])                 # one vertex off to the right
    valid = torch.tensor([[True, False], [False, False]])
    c2w = torch.eye(4).expand(S, 4, 4).clone()
    c2w[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    img = torch.zeros(S, 3, H, W); img[0, 0, :, :] = 1.0           # frame 0 is pure red
    intr = torch.tensor([4.0, 4.0, 4.0])                           # f=4, centre at (4,4)
    ts = torch.tensor([7, 9])

    g = build_hand_gaussians(verts, valid, c2w, img, intr, ts)
    assert len(g) == 1 and g[0].timestamp == 7, "only frame 0 has a valid hand"
    assert g[0].means.shape == (V, 3)
    # world position = camera position + c2w translation under identity rotation
    assert torch.allclose(g[0].means[0], torch.tensor([1.0, 2.0, 3.5]), atol=1e-6)
    # the off-axis vertex projects to u = 4*0.1/0.5 + 4 = 4.8 -> pixel 5
    assert torch.allclose(g[0].means[1], torch.tensor([1.1, 2.0, 3.5]), atol=1e-6)
    # colour sampled from the red frame
    rgb0 = g[0].harmonics[:, 0, :]
    from diffsynth.auxiliary_models.worldmirror.models.utils.sh_utils import SH2RGB
    back = SH2RGB(rgb0)
    assert torch.allclose(back[:, 0], torch.ones(V), atol=1e-5), back[:, 0]
    print("hand_gaussians self-test: OK (world transform, projection pixel, colour sampling)")


if __name__ == "__main__":
    _selftest()
