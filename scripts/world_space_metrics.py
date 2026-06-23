"""World-space hand-placement metrics and clip chaining (workstream C1).

Pure tensor math, no model/data dependencies, so it is unit-testable on its own
(run ``python -m scripts.world_space_metrics`` for the self-test).

Provides:
  * ``solve_similarity`` / ``apply_similarity`` — Umeyama similarity (s, R, t).
  * ``chain_trajectories_by_overlap`` — stitch clip-local world trajectories into one
    global trajectory by aligning overlapping frames (our SLAM-free analogue of the
    global camera trajectory HaWoR/Hand3R get from SLAM).
  * ``w_mpjpe``  — absolute world-space MPJPE over a segment (no alignment).
  * ``wa_mpjpe`` — per-window similarity-aligned world MPJPE (short / long windows).

Conventions: all joint tensors are ``[T, J, 3]`` in metres (T frames, J joints), and
metrics are returned in millimetres. ``valid`` is an optional ``[T]`` or ``[T, J]`` bool
mask (default: all valid).
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- similarity
def solve_similarity(src: torch.Tensor, dst: torch.Tensor,
                     weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted Umeyama similarity aligning ``src`` onto ``dst``: find (s, R, t) minimising
    ``sum_i w_i || s R src_i + t - dst_i ||^2``. ``src``/``dst`` are ``[N, 3]``; ``weights`` is an
    optional ``[N]`` non-negative weight (default: uniform, recovering plain Umeyama). Returns
    scalars/tensors (s: scalar, R: [3,3], t: [3]) in ``src``'s dtype/device.
    """
    assert src.shape == dst.shape and src.shape[-1] == 3 and src.dim() == 2
    src = src.to(torch.float64)
    dst = dst.to(torch.float64)
    n = src.shape[0]
    if weights is None:
        w = torch.ones(n, dtype=torch.float64, device=src.device)
    else:
        w = weights.to(torch.float64).clamp_min(0).to(src.device)
    wsum = w.sum().clamp_min(1e-12)
    mu_s = (w[:, None] * src).sum(0) / wsum
    mu_d = (w[:, None] * dst).sum(0) / wsum
    xs = src - mu_s
    xd = dst - mu_d
    cov = (w[:, None] * xd).T @ xs / wsum       # [3,3] weighted covariance
    u, d, vt = torch.linalg.svd(cov)
    sgn = torch.sign(torch.det(u @ vt))
    s_diag = torch.ones(3, dtype=torch.float64, device=src.device)
    s_diag[-1] = sgn
    rot = u @ torch.diag(s_diag) @ vt          # [3,3], proper rotation
    var_s = (w * (xs ** 2).sum(-1)).sum() / wsum
    scale = (d * s_diag).sum() / var_s.clamp_min(1e-12)
    trans = mu_d - scale * (rot @ mu_s)
    return scale.float(), rot.float(), trans.float()


def solve_similarity_robust(src: torch.Tensor, dst: torch.Tensor, iters: int = 5,
                            eps: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robust similarity via IRLS with Huber weights. Down-weights outlier correspondences
    (e.g. a few badly mispredicted joints in a seam), so a handful of bad joints cannot rotate
    the whole clip. Falls back to plain Umeyama on the first pass, then reweights ``iters`` times.
    """
    s, r, t = solve_similarity(src, dst)
    for _ in range(iters):
        resid = (apply_similarity(src, s, r, t) - dst).norm(dim=-1)   # [N] metres
        med = resid.median()
        mad = (resid - med).abs().median().clamp_min(eps)
        delta = (1.4826 * mad * 1.345).clamp_min(eps)                 # Huber threshold ~1.345 sigma
        w = torch.ones_like(resid)
        big = resid > delta
        w[big] = delta / resid[big].clamp_min(eps)
        s, r, t = solve_similarity(src, dst, weights=w)
    return s, r, t


def apply_similarity(pts: torch.Tensor, s: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply ``s R pts + t`` to ``pts`` of shape ``[..., 3]``."""
    shp = pts.shape
    flat = pts.reshape(-1, 3).to(r.dtype)
    out = s * (flat @ r.T) + t                 # [N,3]
    return out.reshape(shp)


# --------------------------------------------------------------------------- chaining
def chain_trajectories_by_overlap(clip_worlds: list[torch.Tensor], overlap: int) -> torch.Tensor:
    """Stitch a list of clip-local world trajectories into one global trajectory.

    ``clip_worlds[i]`` is ``[Lc, J, 3]`` in clip ``i``'s own frame, and consecutive clips
    share ``overlap`` real frames (clip i's first ``overlap`` frames are the same physical
    frames as the tail of the running global trajectory). Clip 0 defines the global frame.
    For each later clip we solve the similarity that maps its overlap onto the global
    overlap and apply it to the whole clip, then append the clip's new (non-overlapping)
    frames. Returns the global trajectory ``[T_total, J, 3]``.
    """
    assert len(clip_worlds) >= 1 and overlap >= 1
    global_traj = clip_worlds[0].clone()
    for i in range(1, len(clip_worlds)):
        clip = clip_worlds[i]
        o = min(overlap, clip.shape[0], global_traj.shape[0])
        src = clip[:o].reshape(-1, 3)              # clip-local overlap points
        dst = global_traj[-o:].reshape(-1, 3)      # global overlap points
        s, r, t = solve_similarity(src, dst)
        clip_global = apply_similarity(clip, s, r, t)   # [Lc, J, 3] in global frame
        # average the overlap (smooth the seam), append the new frames.
        global_traj[-o:] = 0.5 * (global_traj[-o:] + clip_global[:o])
        global_traj = torch.cat([global_traj, clip_global[o:]], dim=0)
    return global_traj


def chain_trajectories_global(clip_worlds: list[torch.Tensor], overlap: int,
                              iters: int = 8, robust: bool = True) -> torch.Tensor:
    """Global (SLAM-free bundle-adjustment analogue) alternative to greedy chaining.

    Greedy chaining (above) anchors each clip only to its immediate predecessor, so seam errors
    accumulate left-to-right (drift). Here we instead solve all clip-to-global similarities
    *jointly* by block-coordinate descent on the bundle objective
    ``min_{T_i, G} sum_i sum_{frames} || T_i(clip_i) - G ||^2``:

      1. fix the per-clip transforms, set each global joint ``G[g]`` to the mean of every clip's
         transformed estimate of it (consensus);
      2. fix ``G``, re-solve each clip's similarity ``T_i`` to best fit the *whole* clip to ``G``
         (robust Umeyama), so every clip is constrained by the global consensus, not just one seam.

    Clip 0's transform is pinned to identity to fix the global gauge (same reference frame as the
    greedy chainer). Returns the consensus global trajectory ``[T_total, J, 3]``.
    """
    assert len(clip_worlds) >= 1 and overlap >= 1
    n = len(clip_worlds)
    if n == 1:
        return clip_worlds[0].clone()
    lc, j = clip_worlds[0].shape[0], clip_worlds[0].shape[1]
    step = lc - overlap
    assert step >= 1, "overlap must be < clip length"
    t_total = (n - 1) * step + lc
    dtype = clip_worlds[0].dtype
    solver = solve_similarity_robust if robust else solve_similarity

    eye = torch.eye(3, dtype=clip_worlds[0].dtype, device=clip_worlds[0].device)
    zero = torch.zeros(3, dtype=clip_worlds[0].dtype, device=clip_worlds[0].device)
    one = torch.ones((), dtype=clip_worlds[0].dtype, device=clip_worlds[0].device)
    transforms = [(one, eye, zero)]  # clip 0 pinned to identity (gauge anchor)

    # initialise the later transforms by greedy chaining, then refine globally.
    greedy = chain_trajectories_by_overlap(clip_worlds, overlap)
    g = greedy.clone()
    for i in range(1, n):
        off = i * step
        transforms.append(solver(clip_worlds[i].reshape(-1, 3), g[off:off + lc].reshape(-1, 3)))

    def _consensus(tfs: list) -> torch.Tensor:
        acc = torch.zeros(t_total, j, 3, dtype=torch.float64)
        cnt = torch.zeros(t_total, 1, 1, dtype=torch.float64)
        for i in range(n):
            s, r, tt = tfs[i]
            off = i * step
            acc[off:off + lc] += apply_similarity(clip_worlds[i], s, r, tt).to(torch.float64)
            cnt[off:off + lc] += 1
        return (acc / cnt.clamp_min(1)).to(dtype)

    for _ in range(iters):
        g = _consensus(transforms)
        for i in range(1, n):            # clip 0 stays identity
            off = i * step
            transforms[i] = solver(clip_worlds[i].reshape(-1, 3), g[off:off + lc].reshape(-1, 3))
    return _consensus(transforms)


# --------------------------------------------------------------------------- metrics
def _valid_mask(valid: torch.Tensor | None, t: int, j: int, device) -> torch.Tensor:
    if valid is None:
        return torch.ones(t, j, dtype=torch.bool, device=device)
    if valid.dim() == 1:
        return valid.bool().unsqueeze(1).expand(t, j)
    return valid.bool()


def w_mpjpe(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    """Absolute world-space MPJPE (mm), no alignment. ``pred``/``gt``: ``[T, J, 3]`` (m)."""
    m = _valid_mask(valid, pred.shape[0], pred.shape[1], pred.device)
    err = (pred - gt).norm(dim=-1)             # [T, J] in metres
    sel = err[m]
    sel = sel[torch.isfinite(sel)]             # drop inf/nan (e.g. a degenerate first-window extrapolation)
    return float(sel.mean().item() * 1000.0) if sel.numel() else float("nan")


def c_mpjpe(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor | None = None,
            root_relative: bool = True, root: int = 0) -> float:
    """Camera-frame MPJPE (mm). ``pred``/``gt``: ``[T, J, 3]`` metric camera-frame joints (m).

    Scale-free and chaining-free: scores the per-frame hand prediction directly in the camera
    frame, isolating the hand head from the scene scale and world chaining. ``root_relative``
    subtracts the root joint (wrist, index ``root``) from both pred and gt per frame -> the
    standard camera-space hand-pose metric (articulation only). ``root_relative=False`` gives the
    absolute camera-frame placement (includes the hand's metric depth from the camera).
    """
    t, j = pred.shape[0], pred.shape[1]
    m = _valid_mask(valid, t, j, pred.device)
    p, g = pred, gt
    if root_relative:
        p = p - p[:, root:root + 1, :]
        g = g - g[:, root:root + 1, :]
    err = (p - g).norm(dim=-1)
    sel = err[m]
    sel = sel[torch.isfinite(sel)]
    return float(sel.mean().item() * 1000.0) if sel.numel() else float("nan")


def wa_mpjpe(pred: torch.Tensor, gt: torch.Tensor, window: int,
             valid: torch.Tensor | None = None) -> float:
    """World-aligned MPJPE (mm): split the segment into non-overlapping windows of
    ``window`` frames, solve one similarity per window (pred->gt over that window's valid
    joints), apply, then MPJPE. ``window >= T`` gives a single full-segment alignment
    (the "long" variant); a small ``window`` gives the "short" variant.
    """
    t_total, j = pred.shape[0], pred.shape[1]
    m = _valid_mask(valid, t_total, j, pred.device)
    errs = []
    for start in range(0, t_total, max(1, window)):
        end = min(start + window, t_total)
        pw, gw, mw = pred[start:end], gt[start:end], m[start:end]
        ps, gs = pw[mw], gw[mw]                 # [N,3] valid points in this window
        if ps.shape[0] < 3:
            continue
        s, r, t = solve_similarity(ps, gs)
        aligned = apply_similarity(pw, s, r, t)
        e = (aligned - gw).norm(dim=-1)[mw]
        errs.append(e)
    if not errs:
        return float("nan")
    return float(torch.cat(errs).mean().item() * 1000.0)


# --------------------------------------------------------------------------- W gauge (shared)
def first_window_rigid_align(pred: torch.Tensor, gt: torch.Tensor,
                             valid: torch.Tensor | None, wa_short: int) -> torch.Tensor | None:
    """Reproduce ``eval_world_space``'s W-MPJPE gauge exactly so the eval and the offline smoothing
    sweep score identically. Find the first ``wa_short`` window with >=3 valid+finite joints, solve
    the Umeyama ROTATION over it (scale-free), apply a RIGID (scale=1) translation, return aligned
    ``pred`` [T,J,3]. Predictions are already metric (in-scene hand anchor) so scale is never
    re-solved. Returns ``None`` when no window has enough valid+finite joints.
    """
    t, j = pred.shape[0], pred.shape[1]
    v = _valid_mask(valid, t, j, pred.device)
    w0, found = 0, False
    for st in range(0, max(1, t - wa_short + 1)):
        vm = v[st:st + wa_short]
        pw_, gw_ = pred[st:st + wa_short], gt[st:st + wa_short]
        fm = vm & torch.isfinite(pw_).all(-1) & torch.isfinite(gw_).all(-1)
        if int(fm.sum()) >= 3:
            w0, found = st, True
            break
    if not found:
        return None
    m0 = v[w0:w0 + wa_short]
    ps, gs = pred[w0:w0 + wa_short][m0], gt[w0:w0 + wa_short][m0]
    fin = torch.isfinite(ps).all(-1) & torch.isfinite(gs).all(-1)
    ps, gs = ps[fin], gs[fin]
    if ps.shape[0] < 3:
        return None
    one = torch.tensor(1.0, device=ps.device)
    _, rr, _ = solve_similarity(ps, gs)          # Umeyama rotation (scale-independent)
    if not torch.isfinite(rr).all():
        rr = torch.eye(3, device=ps.device)
    tt = gs.mean(0) - (rr @ ps.mean(0))          # rigid translation (scale=1)
    return apply_similarity(pred, one, rr, tt)


def w_mpjpe_first_window_aligned(pred: torch.Tensor, gt: torch.Tensor,
                                 valid: torch.Tensor | None, wa_short: int) -> float:
    """W-MPJPE (mm) under the shared first-window rigid gauge. ``nan`` if no usable window."""
    aligned = first_window_rigid_align(pred, gt, valid, wa_short)
    if aligned is None:
        return float("nan")
    return w_mpjpe(aligned, gt, valid)


# --------------------------------------------------------------------------- smoothing diagnostic
def _gaussian_kernel(length: int, device, dtype) -> torch.Tensor:
    """Normalised 1-D Gaussian of odd ``length`` (sigma = length/6 so +-3 sigma spans it)."""
    radius = length // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    sigma = max(length / 6.0, 1e-3)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_time_finite(x: torch.Tensor, window: int, mode: str = "gaussian") -> torch.Tensor:
    """Finite-aware temporal smoothing of ``x`` [T, ...] along axis 0. Non-finite samples are
    excluded from each window's weighted mean and the weights re-normalised; a fully non-finite
    window stays non-finite. ``window <= 1`` is the identity. ``mode``: 'gaussian' or 'moving'
    (boxcar). ``window`` is coerced to odd. T is small (~128) so a per-frame loop is cheap.
    """
    if window <= 1:
        return x.clone()
    radius = window // 2
    length = 2 * radius + 1
    t = x.shape[0]
    rest = x.shape[1:]
    flat = x.reshape(t, -1).to(torch.float64)
    if mode == "moving":
        ker = torch.ones(length, dtype=torch.float64, device=x.device)
    else:
        ker = _gaussian_kernel(length, x.device, torch.float64)
    out = torch.empty_like(flat)
    for ti in range(t):
        lo, hi = max(0, ti - radius), min(t, ti + radius + 1)
        seg = flat[lo:hi]                                   # [w, D]
        kw = ker[lo - (ti - radius): hi - (ti - radius)]    # align kernel to the (possibly clipped) seg
        fin = torch.isfinite(seg)                           # [w, D]
        w = kw[:, None] * fin                               # zero weight where non-finite
        wsum = w.sum(0)                                      # [D]
        num = (w * torch.where(fin, seg, torch.zeros_like(seg))).sum(0)
        out[ti] = torch.where(wsum > 0, num / wsum.clamp_min(1e-12),
                              torch.full_like(num, float("nan")))
    return out.reshape(t, *rest).to(x.dtype)


def smooth_root_trajectory(pred: torch.Tensor, window: int, hands: int = 2, joints: int = 16,
                           mode: str = "gaussian", root: int = 0) -> torch.Tensor:
    """Replace the per-frame absolute ROOT (wrist) track with a temporally smoothed version while
    keeping each frame's root-relative articulation. ``pred`` is [T, hands*joints, 3] world joints
    (the chained segment trajectory). This is the inference-time post-process that emulates a smooth
    temporal trajectory head: if it lowers W-MPJPE, the world error is high-frequency placement
    jitter/drift that a temporal head will absorb; if W is unchanged, the error is low-frequency
    bias (per-frame depth systematically off) that a temporal head alone will NOT fix.
    """
    if window <= 1:
        return pred.clone()
    t = pred.shape[0]
    j = pred.reshape(t, hands, joints, 3)
    root_track = j[:, :, root:root + 1, :]                  # [T, H, 1, 3]
    shape = j - root_track                                  # per-frame articulation (preserved)
    root_sm = _smooth_time_finite(root_track, window, mode)  # [T, H, 1, 3]
    return (root_sm + shape).reshape(t, hands * joints, 3)


# --------------------------------------------------------------------------- velocity-head ceilings
def replace_root_with_gt_motion(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor | None,
                                hands: int = 2, joints: int = 16, root: int = 0) -> torch.Tensor:
    """Velocity/trajectory-head CEILING (upper bound). Replace the predicted ROOT track with the GT
    root MOTION, anchored at the *predicted* root of the first valid frame (perfect inter-frame
    motion, realistic absolute anchor), keeping the predicted per-frame articulation. If scoring this
    collapses W, then most of W is bad relative root motion a trajectory head could fix; if W stays
    high, the residual is articulation/anchor a velocity head cannot fix. The W gauge re-aligns the
    first window, so the anchor offset is largely removed and this isolates motion quality.
    """
    t = pred.shape[0]
    p = pred.reshape(t, hands, joints, 3)
    g = gt.reshape(t, hands, joints, 3)
    vm = _valid_mask(valid, t, hands * joints, pred.device).reshape(t, hands, joints)
    out = p.clone()
    for h in range(hands):
        pr, gr = p[:, h, root], g[:, h, root]               # [T,3] pred / gt root
        ok = vm[:, h, root] & torch.isfinite(pr).all(-1) & torch.isfinite(gr).all(-1)
        idx = torch.nonzero(ok).flatten()
        if idx.numel() == 0:
            continue
        t0 = int(idx[0])
        root_sm = pr[t0:t0 + 1] + (gr - gr[t0:t0 + 1])       # [T,3] GT motion, pred anchor
        out[:, h] = root_sm.unsqueeze(1) + (p[:, h] - pr.unsqueeze(1))
    return out.reshape(t, hands * joints, 3)


def reanchor_to_gt(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor | None,
                   window: int) -> torch.Tensor:
    """Periodic re-anchor CEILING. Re-fit the predicted trajectory to GT every ``window`` frames with
    a RIGID (rotation + translation, NO scale) transform, then stitch. Bounds how low W goes if
    absolute position is periodically corrected using our own within-window relative motion (no
    per-window scale, unlike WA-MPJPE). Score the result with ``w_mpjpe`` (no further alignment)."""
    t, j = pred.shape[0], pred.shape[1]
    v = _valid_mask(valid, t, j, pred.device)
    out = pred.clone()
    for st in range(0, t, max(1, window)):
        en = min(st + window, t)
        a = first_window_rigid_align(pred[st:en], gt[st:en], v[st:en], en - st)
        if a is not None:
            out[st:en] = a
    return out


# --------------------------------------------------------------------------- self-test
def _selftest() -> None:
    torch.manual_seed(0)
    t_total, j = 128, 16
    base = torch.cumsum(torch.randn(t_total, j, 3) * 0.01, dim=0)  # smooth GT world traj (m)

    # 1) similarity round-trip: a known s,R,t is recovered.
    ang = torch.tensor(0.7)
    rz = torch.tensor([[torch.cos(ang), -torch.sin(ang), 0.0],
                       [torch.sin(ang), torch.cos(ang), 0.0],
                       [0.0, 0.0, 1.0]])
    s_true, t_true = torch.tensor(1.3), torch.tensor([0.2, -0.1, 0.4])
    moved = apply_similarity(base.reshape(-1, 3), s_true, rz, t_true)
    s, r, tt = solve_similarity(moved, base.reshape(-1, 3))
    back = apply_similarity(moved, s, r, tt).reshape(t_total, j, 3)
    assert (back - base).norm(dim=-1).mean() < 1e-4, "similarity round-trip failed"

    # 2) WA-MPJPE of a globally transformed copy is ~0; W-MPJPE is large.
    pred_aligned_copy = apply_similarity(base.reshape(-1, 3), s_true, rz, t_true).reshape(t_total, j, 3)
    assert wa_mpjpe(pred_aligned_copy, base, window=t_total) < 1e-2, "WA-MPJPE should ~0 for a similarity of GT"
    assert w_mpjpe(pred_aligned_copy, base) > 1.0, "W-MPJPE should be large (absolute)"

    # 3) chaining: split GT into overlapping clips, push each into a random local frame,
    #    chain back, and recover the GT global trajectory.
    clip_len, overlap = 16, 8
    clips_local = []
    starts = list(range(0, t_total - overlap, clip_len - overlap))
    for st in starts:
        seg = base[st:st + clip_len]
        if seg.shape[0] < overlap + 1:
            continue
        ang2 = torch.randn(1).item()
        rr = torch.tensor([[torch.cos(torch.tensor(ang2)), -torch.sin(torch.tensor(ang2)), 0.0],
                           [torch.sin(torch.tensor(ang2)), torch.cos(torch.tensor(ang2)), 0.0],
                           [0.0, 0.0, 1.0]])
        local = apply_similarity(seg.reshape(-1, 3), torch.tensor(0.8 + abs(ang2)),
                                 rr, torch.randn(3)).reshape(seg.shape)
        clips_local.append(local)
    chained = chain_trajectories_by_overlap(clips_local, overlap=overlap)
    # chained is in clip-0's local frame; align once to GT to compare shape/placement.
    n_cmp = min(chained.shape[0], t_total)
    err = wa_mpjpe(chained[:n_cmp], base[:n_cmp], window=n_cmp)
    assert err < 1.0, f"chaining drift too high: {err:.3f} mm"

    # 4) global vs greedy under NOISE: add per-clip Gaussian noise to the local points so each
    #    clip is individually imperfect. Greedy accumulates seam error (drift); the global
    #    bundle solve distributes it. Assert global drift <= greedy drift (absolute W-MPJPE,
    #    each aligned once to GT to remove the arbitrary global gauge).
    torch.manual_seed(1)
    clips_noisy = []
    for st in starts:
        seg = base[st:st + clip_len]
        if seg.shape[0] < overlap + 1:
            continue
        ang2 = torch.randn(1).item()
        rr = torch.tensor([[torch.cos(torch.tensor(ang2)), -torch.sin(torch.tensor(ang2)), 0.0],
                           [torch.sin(torch.tensor(ang2)), torch.cos(torch.tensor(ang2)), 0.0],
                           [0.0, 0.0, 1.0]])
        local = apply_similarity(seg.reshape(-1, 3), torch.tensor(0.9 + abs(ang2)),
                                 rr, torch.randn(3)).reshape(seg.shape)
        local = local + 0.01 * torch.randn_like(local)        # ~1cm per-joint noise in local frame
        clips_noisy.append(local)

    def _drift(traj: torch.Tensor) -> float:
        nc = min(traj.shape[0], t_total)
        s, r, t = solve_similarity(traj[:nc].reshape(-1, 3), base[:nc].reshape(-1, 3))
        aligned = apply_similarity(traj[:nc], s, r, t)
        return w_mpjpe(aligned, base[:nc])

    greedy_drift = _drift(chain_trajectories_by_overlap(clips_noisy, overlap=overlap))
    global_drift = _drift(chain_trajectories_global(clips_noisy, overlap=overlap, iters=8, robust=True))
    assert global_drift <= greedy_drift + 1e-6, \
        f"global chaining did not reduce drift: global={global_drift:.2f} greedy={greedy_drift:.2f} mm"

    # 5) C-MPJPE: a constant per-frame translation cancels under root-relative but not absolute.
    cam_gt = base[:32]                                   # [32,16,3] fake cam joints
    cam_pred = cam_gt + 0.05                             # +5cm on every coord/joint/frame
    assert c_mpjpe(cam_pred, cam_gt, root_relative=True) < 1e-3, \
        "root-rel C-MPJPE must cancel a pure per-frame translation"
    off = c_mpjpe(cam_pred, cam_gt, root_relative=False)
    assert off > 80.0, f"absolute C-MPJPE must see the translation (got {off:.1f} mm)"

    # 6) smoothing diagnostic discriminates DRIFT (fixable by a temporal head) from BIAS (not):
    #    smoothing the root track removes high-frequency placement JITTER (W drops a lot) but
    #    leaves a low-frequency linear depth RAMP almost untouched (W ~ unchanged).
    torch.manual_seed(2)
    Ts = 128
    root6 = torch.cumsum(torch.randn(Ts, 1, 1, 3) * 0.005, dim=0)       # smooth true root (m)
    shape6 = torch.randn(1, 1, 16, 3) * 0.03                            # fixed articulation
    clean6 = (root6 + shape6).expand(Ts, 1, 16, 3)[:, 0].clone()        # [T,16,3] one hand
    gt6 = clean6
    valid6 = torch.ones(Ts, 16, dtype=torch.bool)

    jitter = clean6 + torch.randn(Ts, 1, 3) * 0.02                      # 2 cm zero-mean root jitter
    w_jit = w_mpjpe_first_window_aligned(jitter, gt6, valid6, 16)
    w_jit_sm = w_mpjpe_first_window_aligned(
        smooth_root_trajectory(jitter, window=15, hands=1), gt6, valid6, 16)
    assert w_jit_sm < 0.7 * w_jit, f"smoothing should cut jitter W ({w_jit_sm:.1f} vs {w_jit:.1f})"

    ramp = torch.zeros(Ts, 1, 3)
    ramp[:, 0, 2] = torch.linspace(0, 0.2, Ts)                          # 0->20 cm depth drift
    biased = clean6 + ramp
    w_bias = w_mpjpe_first_window_aligned(biased, gt6, valid6, 16)
    w_bias_sm = w_mpjpe_first_window_aligned(
        smooth_root_trajectory(biased, window=15, hands=1), gt6, valid6, 16)
    assert w_bias_sm > 0.85 * w_bias, \
        f"smoothing must NOT remove linear drift ({w_bias_sm:.1f} vs {w_bias:.1f})"

    # window=1 is the identity (sanity for the offline sweep's baseline row).
    assert torch.allclose(smooth_root_trajectory(biased, window=1, hands=1), biased)

    # 7) velocity-head ceilings: GT root motion (perfect velocity) collapses the drift; periodic
    #    rigid re-anchoring also cuts it. Both bound what a trajectory head could recover.
    velg = replace_root_with_gt_motion(biased, gt6, valid6, hands=1)
    w_velg = w_mpjpe_first_window_aligned(velg, gt6, valid6, 16)
    assert w_velg < 0.3 * w_bias, f"GT-velocity ceiling must collapse drift ({w_velg:.1f} vs {w_bias:.1f})"
    w_rean = w_mpjpe(reanchor_to_gt(biased, gt6, valid6, 16), gt6, valid6)
    assert w_rean < 0.5 * w_bias, f"periodic re-anchor must cut drift ({w_rean:.1f} vs {w_bias:.1f})"

    print("world_space_metrics self-test: OK "
          f"(similarity round-trip, WA~0 for GT-similarity, noiseless chaining drift {err:.4f} mm; "
          f"noisy drift greedy={greedy_drift:.1f} -> global={global_drift:.1f} mm; "
          f"C-MPJPE root-rel~0, abs={off:.1f} mm; "
          f"smoothing jitter {w_jit:.1f}->{w_jit_sm:.1f} mm, drift {w_bias:.1f}->{w_bias_sm:.1f} mm; "
          f"ceilings gt-vel={w_velg:.1f} reanchor={w_rean:.1f} mm)")


if __name__ == "__main__":
    _selftest()
