"""Tests for the shared projection + sampling helper.

CPU-only, deterministic, tiny synthetic tensors. Verifies that:
  * a joint placed at a known normalised location samples the expected gs_depth cell,
  * sampling is resolution independent (same value at Hd=Wd=224 vs 448 for the same
    normalised coordinate),
  * out-of-frame joints are flagged by ``in_frame``.
"""
import torch

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    IMAGE_WIDTH,
    project_joints_to_norm_pixels,
    sample_depth_at_joints,
)


def _joint_for_norm(u_norm: float, v_norm: float, z: float) -> torch.Tensor:
    """Build a [B,S,H,J,3] joint that projects to (u_norm, v_norm) at depth z.

    Uses the helper's convention with cam_intr = [f=W, cx=0, cy=0]:
        col = f*x/z          -> v = col          -> v_norm = f*x/(z*W) = x/z
        row = f*y/z          -> u = (W-1) - row  -> u_norm = ((W-1) - f*y/z)/W
    Solve for x, y given target normalised coords.
    """
    W = IMAGE_WIDTH
    f = W
    x = v_norm * z  # since v_norm = x/z
    # u_norm = ((W-1) - f*y/z)/W  ->  f*y/z = (W-1) - u_norm*W  ->  y = z*((W-1) - u_norm*W)/f
    y = z * ((W - 1.0) - u_norm * W) / f
    return torch.tensor([[[[[x, y, z]]]]], dtype=torch.float64)


def _cam_intr() -> torch.Tensor:
    # f = W, principal point at origin -> simple analytic inverse above.
    return torch.tensor([[IMAGE_WIDTH, 0.0, 0.0]], dtype=torch.float64)


def test_projection_roundtrip_hits_target_norm_coords():
    z = 0.5
    u_norm, v_norm = 0.25, 0.75
    joints = _joint_for_norm(u_norm, v_norm, z)
    grid_xy, z_out = project_joints_to_norm_pixels(joints, _cam_intr())

    assert torch.allclose(grid_xy[..., 0], torch.tensor(u_norm, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(grid_xy[..., 1], torch.tensor(v_norm, dtype=torch.float64), atol=1e-9)
    # z returned is the raw metric depth, unclamped.
    assert torch.allclose(z_out, torch.tensor(z, dtype=torch.float64), atol=1e-9)


def test_samples_expected_cell_at_known_location():
    """A ramp gs_depth where each cell encodes its width index; sampling the
    center of a known column returns that column's value exactly."""
    Hd = Wd = 4
    # Value depends only on width index (x = u axis). align_corners=False:
    # cell center i is at normalised (i + 0.5) / Wd.
    ramp = torch.arange(Wd, dtype=torch.float64).view(1, 1, 1, Wd).expand(1, 1, Hd, Wd).contiguous()
    gs_depth = ramp.unsqueeze(2)  # [B,S,1,Hd,Wd]

    target_col = 2
    u_norm = (target_col + 0.5) / Wd  # center of column 2
    v_norm = (1 + 0.5) / Hd           # any in-frame row center
    joints = _joint_for_norm(u_norm, v_norm, z=0.5)
    grid_xy, _ = project_joints_to_norm_pixels(joints, _cam_intr())
    sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy)

    assert bool(in_frame.all())
    assert torch.allclose(sampled, torch.tensor(float(target_col), dtype=torch.float64), atol=1e-6)


def test_sampling_resolution_independence_224_vs_448():
    """Same constant depth at the same normalised coordinate yields the same
    sampled value at Hd=Wd=224 and Hd=Wd=448."""
    const = 1.234
    u_norm, v_norm = 0.4, 0.6
    joints = _joint_for_norm(u_norm, v_norm, z=0.7)
    grid_xy, _ = project_joints_to_norm_pixels(joints, _cam_intr())

    gs_224 = torch.full((1, 1, 1, 224, 224), const, dtype=torch.float64)
    gs_448 = torch.full((1, 1, 1, 448, 448), const, dtype=torch.float64)
    s224, f224 = sample_depth_at_joints(gs_224, grid_xy)
    s448, f448 = sample_depth_at_joints(gs_448, grid_xy)

    assert bool(f224.all()) and bool(f448.all())
    assert torch.allclose(s224, s448, atol=1e-9)
    assert torch.allclose(s224, torch.tensor(const, dtype=torch.float64), atol=1e-9)


def test_resolution_independence_on_smooth_ramp():
    """On a horizontal ramp normalised to [0,1], sampling the same normalised
    coordinate gives (nearly) the same value across resolutions."""
    u_norm, v_norm = 0.5, 0.5  # exact center avoids border effects
    joints = _joint_for_norm(u_norm, v_norm, z=0.6)
    grid_xy, _ = project_joints_to_norm_pixels(joints, _cam_intr())

    def ramp_depth(n: int) -> torch.Tensor:
        col = torch.arange(n, dtype=torch.float64) / (n - 1)  # 0..1 across width
        return col.view(1, 1, 1, 1, n).expand(1, 1, 1, n, n).contiguous()

    s224, _ = sample_depth_at_joints(ramp_depth(224), grid_xy)
    s448, _ = sample_depth_at_joints(ramp_depth(448), grid_xy)
    assert torch.allclose(s224, s448, atol=1e-3)
    # center of a 0..1 ramp is ~0.5
    assert torch.allclose(s224, torch.tensor(0.5, dtype=torch.float64), atol=1e-2)


def test_out_of_frame_joints_flagged():
    z = 0.5
    in_joint = _joint_for_norm(0.5, 0.5, z)
    out_joint = _joint_for_norm(1.5, 0.5, z)   # u_norm > 1 -> out of frame
    out_joint2 = _joint_for_norm(0.5, -0.5, z)  # v_norm < 0 -> out of frame
    joints = torch.cat([in_joint, out_joint, out_joint2], dim=3)  # J axis

    grid_xy, _ = project_joints_to_norm_pixels(joints, _cam_intr())
    gs_depth = torch.ones((1, 1, 1, 8, 8), dtype=torch.float64)
    _, in_frame = sample_depth_at_joints(gs_depth, grid_xy)

    assert bool(in_frame[0, 0, 0, 0])       # in
    assert not bool(in_frame[0, 0, 0, 1])   # out (u)
    assert not bool(in_frame[0, 0, 0, 2])   # out (v)
