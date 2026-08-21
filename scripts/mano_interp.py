"""Interpolate MANO hand parameters to a held-out frame's time, in WORLD space.

The context frames carry predicted hands; a held-out target needs one, and interpolating in the
camera frame is wrong whenever the camera moves: a hand static in the world would appear to
translate. So the hand transform is carried to world through each context camera, interpolated
there, and carried back through the target camera.

Parameter layout per hand, from `hand_vis_utils.MANOModel` (32-D):
    transl[0:3]  quat_wxyz[3:7]  pose_pca[7:22]  betas[22:32]

Articulation is 15 PCA coefficients, a linear space, so linear interpolation is the correct
geodesic there; SLERP applies to the global orientation quaternion only. Betas are a per-clip
constant and the caller decides which value to hold.

No extrapolation: a target outside the surrounding context pair is the caller's error to refuse.
"""
from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, wxyz, broadcastable [..., 4]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] rotation to wxyz quaternion, stable across all trace branches."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # Four candidate constructions; the one anchored on the largest diagonal term is stable.
    q_w = torch.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], dim=-1)
    q_x = torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], dim=-1)
    q_y = torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], dim=-1)
    q_z = torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], dim=-1)
    trace_like = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                              1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1)
    best = trace_like.argmax(dim=-1)
    q = torch.stack([q_w, q_x, q_y, q_z], dim=-2)
    q = torch.gather(q, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion to [..., 3, 3] rotation."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Geodesic interpolation, wxyz, double-cover resolved by sign flip."""
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.arccos(dot)
    sin = torch.sin(theta)
    near = sin.abs() < 1e-6
    w0 = torch.where(near, 1.0 - t, torch.sin((1 - t) * theta) / sin.clamp_min(1e-12))
    w1 = torch.where(near, torch.full_like(sin, t), torch.sin(t * theta) / sin.clamp_min(1e-12))
    out = w0 * q0 + w1 * q1
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def interp_hand_params(p0: torch.Tensor, p1: torch.Tensor,
                       c2w0: torch.Tensor, c2w1: torch.Tensor, c2w_t: torch.Tensor,
                       t: float) -> torch.Tensor:
    """Hand params at time t in the TARGET camera's frame.

    Args:
        p0, p1:  [..., 32] per-hand params in their OWN camera frames.
        c2w0, c2w1, c2w_t: [..., 4, 4] camera-to-world of the two context frames and the target.
        t: fraction of the way from frame 0 to frame 1.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t={t}: extrapolation is refused, pick the surrounding context pair")
    R0, tr0 = c2w0[..., :3, :3], c2w0[..., :3, 3]
    R1, tr1 = c2w1[..., :3, :3], c2w1[..., :3, 3]
    Rt_w2c = c2w_t[..., :3, :3].transpose(-1, -2)
    tt = c2w_t[..., :3, 3]

    # Root position through world.
    pw0 = torch.einsum("...ij,...j->...i", R0, p0[..., 0:3]) + tr0
    pw1 = torch.einsum("...ij,...j->...i", R1, p1[..., 0:3]) + tr1
    pwt = (1 - t) * pw0 + t * pw1
    transl_t = torch.einsum("...ij,...j->...i", Rt_w2c, pwt - tt)

    # Global orientation through world: R_w_hand = R_cw @ R_cam_hand.
    qw0 = quat_mul(mat_to_quat(R0), p0[..., 3:7])
    qw1 = quat_mul(mat_to_quat(R1), p1[..., 3:7])
    qwt = slerp(qw0, qw1, t)
    quat_t = quat_mul(mat_to_quat(Rt_w2c), qwt)

    pose_t = (1 - t) * p0[..., 7:22] + t * p1[..., 7:22]
    betas_t = (1 - t) * p0[..., 22:32] + t * p1[..., 22:32]
    return torch.cat([transl_t, quat_t, pose_t, betas_t], dim=-1)


def _selftest() -> None:
    torch.manual_seed(0)

    # 1. slerp endpoints and half-angle midpoint.
    ang = torch.tensor(torch.pi / 2)
    qz = torch.tensor([torch.cos(ang / 2), 0.0, 0.0, torch.sin(ang / 2)])
    qi = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mid = slerp(qi, qz, 0.5)
    half = torch.tensor([torch.cos(ang / 4), 0.0, 0.0, torch.sin(ang / 4)])
    assert torch.allclose(slerp(qi, qz, 0.0), qi, atol=1e-6)
    assert torch.allclose(mid, half, atol=1e-6), mid

    # 2. quaternion round-trip through matrices, random rotations.
    q = torch.randn(64, 4); q = q / q.norm(dim=-1, keepdim=True)
    q2 = mat_to_quat(quat_to_mat(q))
    flip = torch.where((q * q2).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    assert torch.allclose(q, q2 * flip, atol=1e-5)

    # 3. THE test this module exists for: a hand static in the world, seen by a moving camera,
    #    must come back exactly, at every t, in a third camera's frame.
    def rot_z(a):
        c, s = torch.cos(torch.tensor(a)), torch.sin(torch.tensor(a))
        m = torch.eye(4); m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        return m
    c2w0, c2w1, c2wt = rot_z(0.1), rot_z(0.5), rot_z(0.3)
    c2w0[:3, 3] = torch.tensor([0.2, 0.0, 0.1]); c2w1[:3, 3] = torch.tensor([-0.1, 0.3, 0.0])
    c2wt[:3, 3] = torch.tensor([0.05, 0.15, 0.05])
    p_world = torch.tensor([0.3, -0.2, 0.8])
    q_world = torch.randn(4); q_world = q_world / q_world.norm()

    def into(c2w):
        w2c_R = c2w[:3, :3].T
        p = torch.cat([w2c_R @ (p_world - c2w[:3, 3]),
                       quat_mul(mat_to_quat(w2c_R), q_world),
                       torch.randn(15) * 0.0 + 0.4, torch.zeros(10)])
        return p
    p0, p1 = into(c2w0), into(c2w1)
    for t in (0.0, 0.3, 1.0):
        pt = interp_hand_params(p0, p1, c2w0, c2w1, c2wt, t)
        expect = into(c2wt)
        assert torch.allclose(pt[0:3], expect[0:3], atol=1e-5), (t, pt[0:3], expect[0:3])
        f = torch.where((pt[3:7] * expect[3:7]).sum() < 0, -1.0, 1.0)
        assert torch.allclose(pt[3:7] * f, expect[3:7], atol=1e-5), (t, pt[3:7], expect[3:7])

    # 4. a hand rotating 90 degrees in world lands at 45 at the midpoint.
    qh1 = quat_mul(qz, q_world)
    p1b = p1.clone()
    p1b[3:7] = quat_mul(mat_to_quat(c2w1[:3, :3].T), qh1)
    mid = interp_hand_params(p0, p1b, c2w0, c2w1, c2wt, 0.5)
    q_mid_world = quat_mul(mat_to_quat(c2wt[:3, :3]), mid[3:7])
    expect_mid = quat_mul(half, q_world)
    f = torch.where((q_mid_world * expect_mid).sum() < 0, -1.0, 1.0)
    assert torch.allclose(q_mid_world * f, expect_mid, atol=1e-5)

    print("mano_interp self-test: OK (endpoints exact, static-world hand exact under a moving "
          "camera, world rotation halves at t=0.5)")


if __name__ == "__main__":
    _selftest()
