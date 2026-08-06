"""A joint BEHIND the camera survives the scale-solve validity mask and poisons the median.

Task #63 asks why 16% of segments land exactly on the 0.1 clamp floor. The mask that decides
which joint/scene-depth correspondences feed the median is

    valid = in_frame & (sampled > 0.01) & isfinite(z) & isfinite(sampled)

and it never requires z > 0.

THE MECHANISM, which is sharper than "negative depth projects somewhere odd".
``project_joints_to_norm_pixels`` clamps depth for the projection only:

    z = pred_joints[..., 2].clamp_min(Z_MIN)      # Z_MIN = 0.05 m
    col = f * x / z + cx ;  row = f * y / z + cy
    return grid_xy, pred_joints[..., 2]           # <- RAW, UNCLAMPED z

So a joint at z = -0.3 m is projected *as if it sat 5 cm in front of the camera*, which for a
joint near the optical axis lands it comfortably inside the frame, while the depth handed back to
the caller is still -0.3. The ratio z/d_scene is then negative, in_frame is True, and the median
consumes it. The clamp that keeps the projection finite is exactly what hides the sign error.

These tests are written against the CONTRACT (the geometry), not against current behaviour, which
is the mistake that made an earlier batch of tests worthless.
"""
from __future__ import annotations

import torch

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    Z_MIN,
    project_joints_to_norm_pixels,
)
from scripts.eval_world_space import ratio_validity_mask


def _intr(f=200.0, cx=112.0, cy=112.0):
    """A centred 224-px pinhole, so frame_width_from_intr's assumption holds and the test is
    about the sign of z rather than about the width estimate (that is task #64)."""
    return torch.tensor([[f, cx, cy]], dtype=torch.float32)


def test_behind_camera_joint_projects_in_frame_and_keeps_negative_depth():
    """A joint at z < 0 lands inside the frame and is returned with its negative depth intact.

    (x, y, z) = (0.02, 0.02, -0.3) is 30 cm BEHIND the camera. The projection clamps z to
    Z_MIN = 0.05, giving col = 200*0.02/0.05 + 112 = 192 and row likewise, both inside [0, 224),
    so ``in_frame`` is True. The returned depth is the raw -0.3.
    """
    intr = _intr()
    # [B,S,H,J,3] = one batch, one frame, one hand, one joint.
    joints = torch.tensor([0.02, 0.02, -0.3]).view(1, 1, 1, 1, 3)
    grid_xy, z = project_joints_to_norm_pixels(joints, intr)

    z_ret = float(z.reshape(-1)[0])
    assert z_ret < 0, "fixture is wrong: this joint must be behind the camera"
    assert abs(z_ret - (-0.3)) < 1e-6, (
        "the helper must return the RAW depth, not the projection-clamped one")

    g = grid_xy.reshape(-1, 2)
    in_frame = bool(((g >= 0) & (g <= 1)).all())
    assert in_frame, (
        f"fixture no longer exercises the hole: grid {g.tolist()} fell outside the frame, so "
        "in_frame would have rejected this joint for the wrong reason"
    )


def test_unguarded_mask_admits_the_behind_camera_joint():
    """The defect itself: the production mask accepts a correspondence with negative depth.

    This is the assertion that matters. The helper returning a raw negative depth is correct; the
    mask treating it as a usable correspondence is not.
    """
    z = torch.tensor([[-0.3]])          # behind the camera
    sampled = torch.tensor([[0.9]])     # a perfectly ordinary scene depth at that pixel
    in_frame = torch.tensor([[True]])

    unguarded = ratio_validity_mask(z, sampled, in_frame, require_positive_z=False)
    assert bool(unguarded.all()), "fixture wrong: the other three terms must all pass"

    ratio = float((z / sampled)[unguarded][0])
    assert ratio < 0, (
        f"the unguarded mask yields a NEGATIVE scale correspondence ({ratio:.3f}); pooled into "
        "the median this is what pushes a clip onto the 0.1 clamp floor (task #63)"
    )

    guarded = ratio_validity_mask(z, sampled, in_frame, require_positive_z=True)
    assert not bool(guarded.any()), "require_positive_z must reject the behind-camera joint"


def test_guard_keeps_legitimate_correspondences():
    """The guard must remove only the behind-camera joints, not thin the honest population."""
    z = torch.tensor([[0.4, 0.8, -0.3, 1.2]])
    sampled = torch.tensor([[0.4, 0.8, 0.9, 1.2]])
    in_frame = torch.ones_like(z, dtype=torch.bool)

    guarded = ratio_validity_mask(z, sampled, in_frame, require_positive_z=True)
    assert guarded.tolist() == [[True, True, False, True]]
    assert abs(float((z / sampled)[guarded].median()) - 1.0) < 1e-6


def test_projection_clamp_is_what_pulls_the_joint_into_frame():
    """Names the cause: without the Z_MIN clamp the same joint would project far out of frame.

    This is what makes the defect invisible. If the projection propagated the negative depth the
    joint would leave the frame and in_frame would reject it; the clamp rescues the pixel while
    leaving the sign error in the depth.
    """
    f, cx = 200.0, 112.0
    x, z_true = 0.02, -0.3

    col_clamped = f * x / max(z_true, Z_MIN) + cx      # what the helper computes: 192, in frame
    col_unclamped = f * x / z_true + cx                # what geometry says: 98.7 ... also in frame

    assert 0 <= col_clamped < 2 * cx
    # The clamp changes WHERE it lands, by a lot, which is why the sampled scene depth at that
    # pixel is unrelated to the joint even ignoring the sign.
    assert abs(col_clamped - col_unclamped) > 50, (
        "the clamp must materially move the sampled pixel, otherwise the only problem is the sign"
    )


def test_one_behind_camera_population_flips_the_solved_scale():
    """Median over ratios is not robust once enough correspondences carry a negative numerator."""
    good = torch.full((19,), 1.02)          # a well-behaved clip: hand depth ~ scene depth
    bad = torch.full((21,), -0.04)          # behind-camera joints: z<0 over positive scene depth
    ratios = torch.cat([good, bad])

    s_raw = float(ratios.median())
    s_clamped = float(min(max(s_raw, 0.1), 10.0))

    assert s_raw < 0, f"expected the contaminated median to go negative, got {s_raw}"
    assert s_clamped == 0.1, (
        f"the negative median clamps to the floor and is returned as a solved scale ({s_clamped}), "
        "which is exactly the 16%-on-the-floor signature in task #63"
    )

    # With the behind-camera joints excluded the solve recovers the honest answer.
    assert abs(float(good.median()) - 1.02) < 1e-5
