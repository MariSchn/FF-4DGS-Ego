"""Build the model's views with real cameras and held-out target frames.

The default path (`build_views` in train_hand_head.py) hands the model identity camera poses and
identity intrinsics, and the photometric loss then re-renders the *input* views from Gaussians
unprojected from those same views. Under identity cameras, unprojecting pixel p at any depth d and
reprojecting through the same camera returns p, so the render is invariant to depth and the
photometric loss cannot observe geometry. That is the flat direction the Gaussian head walked into
until the exp() clamp absorbed it.

This module builds the alternative: Gaussians come from context frames only, unprojected with the
real intrinsics and the real pose, and the loss renders held-out target frames from a different
viewpoint. Parallax then makes depth observable.

CONVENTION, and it is the part that silently ruins everything if wrong. The store's
`cam_extrinsics` is WORLD-TO-CAMERA: `eval_world_space` inverts it to obtain camera centres.
`depth_to_world_coords_points` documents its `extrinsic` argument as CAMERA-TO-WORLD. So the
views must carry the inverse of what the batch holds. The same confusion in `rendered_extrinsics`
made every world-space number wrong until it was found.
"""
from __future__ import annotations

import torch


def intr_3x3(cam_intr: torch.Tensor, res: int, device) -> torch.Tensor:
    """[B,3,3] pinhole K on the model's square `res`, from (f, cx, cy) already on that frame.

    `HOT3DHandDataset` carries the cached intrinsics across the cover-and-centre-crop the loader
    applies, so no rescaling belongs here. Doing it again forced the principal point to the centre
    and moved the focal by a third on the stores whose source frame is not square.
    """
    ci = cam_intr.view(-1, 3).to(device).float()
    B = ci.shape[0]
    k = torch.zeros(B, 3, 3, device=device, dtype=torch.float32)
    k[:, 0, 0] = k[:, 1, 1] = ci[:, 0]
    k[:, 0, 2], k[:, 1, 2] = ci[:, 1], ci[:, 2]
    k[:, 2, 2] = 1.0
    return k


def pick_targets(S: int, n_targets: int, device) -> torch.Tensor:
    """[S] bool marking the TRAILING held-out frames.

    Upstream counts targets but then slices positionally: `context_nums` is the number of False
    entries, and the context is `img[:, :context_nums]` (`worldmirror.py:626-627`), while the
    rasterizer splits masks at the same boundary (`rasterization.py:545-546`). Targets spread
    through the middle would therefore be counted correctly and then taken from the wrong frames.
    """
    m = torch.zeros(S, dtype=torch.bool, device=device)
    if n_targets <= 0 or S < 3:
        return m
    m[S - min(n_targets, S - 2):] = True
    return m


def build_views_metric(imgs, num_frames, device, cam_extrinsics, cam_intrinsics, res,
                       hand_bboxes=None, hand_valid=None, n_targets=2,
                       crop_local_output=False, hand_crops=None, frame_index=None):
    """Views with real cameras and held-out targets. Mirrors `build_views`'s keys exactly.

    Args:
        imgs:            [B,S,3,H,W].
        cam_extrinsics:  [B,S,4,4] WORLD-TO-CAMERA, as the store holds it.
        cam_intrinsics:  [B,3] cached (f, cx, cy).
        res:             the square resolution the intrinsics refer to.
        n_targets:       how many frames to hold out per clip.
    """
    B, S, _, H, W = imgs.shape
    if int(num_frames) != S:
        raise RuntimeError(f"build_views_metric: num_frames={num_frames} but imgs carries S={S}")

    w2c = cam_extrinsics.to(device=device, dtype=torch.float32)
    if w2c.shape[:2] != (B, S) or w2c.shape[-2:] != (4, 4):
        raise ValueError(f"cam_extrinsics must be [B,S,4,4], got {tuple(w2c.shape)}")
    c2w = torch.linalg.inv(w2c.double()).float()          # the model wants camera-to-world

    k = intr_3x3(cam_intrinsics, int(res), device)         # [B,3,3]
    if k.shape[0] == 1 and B > 1:
        k = k.expand(B, 3, 3)
    intrs = k.unsqueeze(1).expand(B, S, 3, 3).contiguous()

    timestamp = (frame_index.to(device=device, dtype=torch.long) if frame_index is not None
                 else torch.arange(S, device=device).unsqueeze(0).expand(B, -1))

    views = {
        "img":          imgs,
        "is_target":    pick_targets(S, n_targets, device).view(1, S).expand(B, S).contiguous(),
        "timestamp":    timestamp,
        # TRUE, and it is what makes a held-out view renderable at all. The rasterizer draws a
        # Gaussian into a view only when its timestamp matches that view's, or when it is -1
        # (`rasterization.py:321`); the forward/backward windows that would relax this need the
        # motion module, which `enable_motion` leaves off. With per-frame timestamps every target
        # frame therefore renders empty: measured mean 0.0000, std 0.0000, every pixel dark, while
        # the context frames of the same clips render at 35.2 dB. Marking the clip static fuses the
        # Gaussians at timestamp -1 so they appear in every view.
        # The approximation this buys: within a 16-frame window the scene, and the hand in it, is
        # treated as motionless, so a target frame is scored against the hand where the context
        # left it. Both arms of the injection A/B carry it identically.
        "is_static":    torch.ones((B, S), dtype=torch.bool, device=device),
        "valid_mask":   torch.ones((B, S, H, W), dtype=torch.bool, device=device),
        "camera_poses": c2w,
        "camera_intrs": intrs,
        "depthmap":     torch.ones((B, S, H, W), device=device),
    }
    if hand_bboxes is not None:
        views["hand_bboxes"] = hand_bboxes
    if hand_valid is not None:
        views["hand_valid"] = hand_valid
    if hand_crops is not None:
        views["hand_crops"] = hand_crops
    if crop_local_output:
        views["crop_local_output"] = True
    return views


def _selftest() -> None:
    """A plane at known depth seen by two cameras: unprojecting from the first and reprojecting
    into the second must land where the geometry says, and must MOVE, because a render that does
    not move under a viewpoint change is the degeneracy this module exists to remove."""
    # The unprojection is reimplemented here rather than imported, both to keep the test free of
    # the package's heavy dependencies and because writing it out is what checks that the
    # convention above was read correctly.
    dev, res = "cpu", 224
    cam_intr = torch.tensor([[220.0, 112.0, 112.0]])
    K = intr_3x3(cam_intr, res, dev)[0]

    # camera 1 at the origin, camera 2 translated 10 cm along x, both looking down -z
    w2c1 = torch.eye(4)
    w2c2 = torch.eye(4); w2c2[0, 3] = -0.10          # world-to-camera: shifting the camera by +x
    d = 0.80

    def unproject(w2c, u, v, depth):
        """pixel + depth -> world, using camera-to-world as depth_to_world_coords_points does."""
        xc = (u - float(K[0, 2])) * depth / float(K[0, 0])
        yc = (v - float(K[1, 2])) * depth / float(K[1, 1])
        Xc = torch.tensor([xc, yc, depth])
        c2w = torch.linalg.inv(w2c)
        return (c2w[:3, :3] @ Xc) + c2w[:3, 3]

    world = unproject(w2c1, res / 2.0, res / 2.0, d)

    def project(w2c, X):
        Xc = (w2c[:3, :3] @ X) + w2c[:3, 3]
        uv = K @ Xc
        return uv[:2] / uv[2]

    u1, u2 = project(w2c1, world), project(w2c2, world)
    assert torch.allclose(u1, torch.tensor([res / 2.0, res / 2.0]), atol=1e-3), \
        f"centre pixel must round-trip to the centre, got {u1.tolist()}"
    shift = float((u2 - u1).norm())
    expected = float(K[0, 0]) * 0.10 / 0.80           # f * baseline / depth
    assert abs(shift - expected) < 1.0, f"parallax {shift:.2f} px, expected {expected:.2f}"
    assert shift > 5.0, "no parallax, so depth would still be unobservable"

    m = pick_targets(16, 2, dev)
    assert int(m.sum()) == 2 and not bool(m[0]), "targets must be 2 and never the first frame"
    print(f"metric_views self-test: OK (centre round-trip {u1.tolist()}, parallax {shift:.1f} px "
          f"against {expected:.1f} predicted, targets at {torch.nonzero(m).flatten().tolist()})")


if __name__ == "__main__":
    _selftest()
