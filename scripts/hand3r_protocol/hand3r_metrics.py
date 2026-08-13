"""Hand3R's world-space protocol, in torch, so our eval can emit a Hand3R-comparable number.

Our own metrics live in ``scripts/world_space_metrics.py`` and stay the paper's primary convention.
This module exists only to score the SAME predictions a second way, because on 2026-08-13 the
Hand3R authors sent their reference scorer and it does not match what our code claimed it matched.

The alignment algebra is identical in both: Umeyama on ``cov = dst_centred.T @ src_centred / N``,
with the reflection fix on the last singular value. ``tests/test_hand3r_protocol_parity.py`` proves
that numerically against their NumPy file, vendored here as ``reference_scorer.py``. So every number
this module produces differently from ours is a PROTOCOL difference, not an implementation one.
There are four, and they are what this module implements:

1. THE W GAUGE. Ours solves the rigid transform on the first ``wa_short`` frames, i.e. 30 of them.
   Theirs solves it on the FIRST TWO FRAMES. ``world_space_metrics.py`` carried a comment asserting
   ours "matches Hand3R's first-window align, no scale"; their PROTOCOL.md says "estimate a rigid
   transform from the first two frames". A 30-frame fit is a least-squares compromise over the
   first 30 frames of the chunk and a 2-frame fit is not, so the two gauges do not measure the same
   drift and ours is the more forgiving of the two.
2. CHUNKING. They cut the clip into non-overlapping chunks and DISCARD a tail shorter than 10
   frames. Our ``wa_mpjpe`` keeps any window holding at least 3 valid points, so a 5-frame tail is
   scored by us and dropped by them.
3. WITHIN-CLIP AVERAGING. They average chunk scores equally. We pool the per-joint errors across
   chunks and take one mean, which weights a chunk by how many valid joints it has.
4. VALIDITY. They drop invalid frames and THEN chunk, so a gap in the middle of a clip is
   compressed and two frames either side of it become temporal neighbours. We keep the time axis
   and mask. Their own PROTOCOL.md flags this as a compatibility wart rather than a design choice.

Joint set and hand set are protocol differences too, but they are decided by the caller: pass 21
joints and the right-preferred single hand and this scores Hand3R's Table II convention exactly.
"""
from __future__ import annotations

import torch

from scripts.world_space_metrics import apply_similarity, solve_similarity

# Their PROTOCOL.md: "A tail shorter than 10 frames is skipped." Also the floor on a whole clip.
MIN_CHUNK_FRAMES = 10


def _rigid_from_first_two(pred: torch.Tensor, gt: torch.Tensor) -> tuple:
    """Their W gauge: Umeyama over frames 0 and 1 only, with the scale pinned to 1.

    All joints of both frames enter as one point cloud (42 points at 21 joints), so the hand's own
    shape, not its motion, determines the rotation. That is the whole difference from a 30-frame
    fit: over 30 frames the wrist sweeps a real trajectory and the fit trades pose error against
    trajectory error, whereas over 2 frames there is almost no trajectory to trade against.
    """
    p = pred[:2].reshape(-1, 3)
    g = gt[:2].reshape(-1, 3)
    _, rot, _ = solve_similarity(p, g)
    one = torch.tensor(1.0, device=pred.device)
    trans = g.mean(0) - (rot @ p.mean(0))
    return one, rot, trans


def _chunks(n_frames: int, chunk_len: int):
    for start in range(0, n_frames, chunk_len):
        end = min(n_frames, start + chunk_len)
        if end - start >= MIN_CHUNK_FRAMES:
            yield start, end


def score_clip_hand3r(pred: torch.Tensor, gt: torch.Tensor,
                      valid: torch.Tensor | None = None,
                      chunk_len: int = 100) -> dict:
    """Score one clip under Hand3R's protocol. ``pred``/``gt`` are ``[T, J, 3]`` in metres, one
    hand, and ``valid`` is ``[T]``. Returns mm, or an empty dict when the clip is rejected.

    The frame filter runs BEFORE chunking, reproducing the compression their PROTOCOL.md documents.
    """
    t = pred.shape[0]
    mask = torch.ones(t, dtype=torch.bool, device=pred.device) if valid is None else valid.bool()
    mask = mask & torch.isfinite(gt).all(-1).all(-1)
    if int(mask.sum()) < MIN_CHUNK_FRAMES:
        return {}
    finite_pred = torch.isfinite(pred).all(-1).all(-1)
    # Their rule: a clip whose predictions are more than half non-finite is thrown away rather than
    # scored on the remainder, so a method cannot buy a good score by predicting only easy frames.
    if int((mask & ~finite_pred).sum()) / int(mask.sum()) > 0.5:
        return {}
    mask = mask & finite_pred
    if int(mask.sum()) < MIN_CHUNK_FRAMES:
        return {}

    p, g = pred[mask], gt[mask]
    out = {"C_MPJPE_abs": float((p - g).norm(dim=-1).mean() * 1000.0), "n_frames": int(mask.sum())}

    w, wa, mre = [], [], []
    for s, e in _chunks(p.shape[0], chunk_len):
        pc, gc = p[s:e], g[s:e]
        wa.append(float((apply_similarity(pc, *solve_similarity(pc.reshape(-1, 3), gc.reshape(-1, 3)))
                         - gc).norm(dim=-1).mean() * 1000.0))
        w.append(float((apply_similarity(pc, *_rigid_from_first_two(pc, gc)) - gc)
                       .norm(dim=-1).mean() * 1000.0))
        mre.append(float((pc[:, 0] - gc[:, 0]).norm(dim=-1).mean() * 1000.0))
    if w:
        # Chunk scores averaged equally, per their "Chunk scores are averaged equally within a clip".
        out["W_MPJPE"] = sum(w) / len(w)
        out["WA_MPJPE"] = sum(wa) / len(wa)
        out["MRE"] = sum(mre) / len(mre)
    return out


def aggregate_hand3r(clips: list[dict]) -> dict:
    """Unweighted mean over clips, per their "the final reported value is the unweighted mean of
    the clip-level scores". A clip contributes to a metric only if it produced that metric."""
    keys = ("C_MPJPE_abs", "W_MPJPE", "WA_MPJPE", "MRE")
    agg = {}
    for k in keys:
        vals = [c[k] for c in clips if k in c and c[k] == c[k]]
        agg[k] = float(sum(vals) / len(vals)) if vals else float("nan")
        agg[k + "_n"] = len(vals)
    return agg
