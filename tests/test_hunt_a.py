"""BUG HUNTER A - failing tests that pin real defects in the current tree.

Every test here FAILS against the code as it stands today. Each one asserts a contract
the repository states about itself (docstring, sibling implementation, or documented
store fact), never an invented one. CPU-only, no checkpoints, no datasets, no GPU.

tests/conftest.py already installs the lightweight `diffsynth` namespace shim used by
the existing metric-coupling tests, so the pure torch leaf modules import here.
"""
from __future__ import annotations

import importlib

import pytest
import torch


# ===========================================================================
# BUG 1 (CRITICAL) - the WiLoR / HaMeR baseline rows are built with a joint
# remap that PERMUTES the finger blocks relative to every other row in the table.
#
# scripts/build_native_baseline_preds.py:47-50 states its own premise and its own
# provenance:
#     "WiLoR/HaMeR emit MANO-21 (native order, tips interleaved 4,8,12,16,20).
#      The eval + our GT use the 16 smplx kinematic joints. This subset is copied
#      VERBATIM from run_wilor_h2o.py / eval_cmpjpe.py so the joint correspondence
#      is identical to every other row."
# For that layout the repo defines the 21 -> smplx-16 map in THREE independent
# places, all agreeing:
#     scripts/eval_cmpjpe.py:46                     H2O_TO_MANO[:16]
#     scripts/haptic_to_worldeval.py:36             OP2SMPLX16
#     scripts/preprocessing/preprocess_hoi4d.py:226 _KPS2D_FOR_SMPLX16
#         = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
# build_native_baseline_preds.py uses [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
# 17, 18, 19] instead: the same SET of source joints in a DIFFERENT order.
# ===========================================================================

# manopth / OpenPose-21 hand layout, spelled out verbatim in
# scripts/preprocessing/preprocess_hoi4d.py:219-224 ("wrist; thumb1,2,3,TIP;
# index..TIP; middle..TIP; ring..TIP; pinky..TIP").
_OP21_NAMES = (
    ["wrist"]
    + [f"thumb{i}" for i in (1, 2, 3)] + ["thumb_TIP"]
    + [f"index{i}" for i in (1, 2, 3)] + ["index_TIP"]
    + [f"middle{i}" for i in (1, 2, 3)] + ["middle_TIP"]
    + [f"ring{i}" for i in (1, 2, 3)] + ["ring_TIP"]
    + [f"pinky{i}" for i in (1, 2, 3)] + ["pinky_TIP"]
)

# The smplx-16 kinematic order our GT caches are written in, quoted at
# preprocess_hoi4d.py:224 and dexycb_to_ours.py:50-51:
# "[wrist, index x3, middle x3, pinky x3, ring x3, thumb x3]".
_SMPLX16_NAMES = (
    ["wrist"]
    + [f"index{i}" for i in (1, 2, 3)]
    + [f"middle{i}" for i in (1, 2, 3)]
    + [f"pinky{i}" for i in (1, 2, 3)]
    + [f"ring{i}" for i in (1, 2, 3)]
    + [f"thumb{i}" for i in (1, 2, 3)]
)


def test_native_baseline_joint_remap_permutes_finger_blocks():
    """WiLoR/HaMeR baseline preds land in (wrist, thumb, index, middle, ring, pinky)
    order while the GT caches they are scored against are smplx-16
    (wrist, index, middle, pinky, ring, thumb)."""
    from scripts.build_native_baseline_preds import MANO21_TO_16
    from scripts.haptic_to_worldeval import OP2SMPLX16
    from scripts.preprocessing.preprocess_hoi4d import _KPS2D_FOR_SMPLX16

    # The two in-repo references for the SAME source layout agree with each other.
    assert OP2SMPLX16 == _KPS2D_FOR_SMPLX16

    produced = [_OP21_NAMES[k] for k in MANO21_TO_16]
    reference = [_OP21_NAMES[k] for k in OP2SMPLX16]
    assert reference == _SMPLX16_NAMES  # sanity: the reference map is the smplx-16 order

    assert sorted(MANO21_TO_16) == sorted(OP2SMPLX16), "same source joints, so it is a pure permutation"

    assert produced == _SMPLX16_NAMES, (
        "build_native_baseline_preds.MANO21_TO_16 does NOT produce the smplx-16 order "
        "its own docstring claims to copy from eval_cmpjpe.py.\n"
        f"  MANO21_TO_16 -> {produced}\n"
        f"  smplx-16     -> {_SMPLX16_NAMES}\n"
        "Every WiLoR / HaMeR cam_joints row therefore has its finger blocks permuted "
        "against gt_joints_cache_cam_v2.pt."
    )


# ===========================================================================
# BUG 2 (CRITICAL) - the projection helper derives the normalisation width as
# 2*cx, and that invariant is FALSE on the HOI4D store the headline numbers use.
#
# hand_depth_sampling.frame_width_from_intr docstring:
#     "The pipeline's intrinsics are square-pinhole with the principal point at the
#      frame centre ... so the width is recoverable from the intrinsics themselves
#      and never has to be assumed."
# scripts/hawor_boxes_from_detbox.py:50-56 records the actual HOI4D store value and
# says exactly why that premise does not hold:
#     "On the HOI4D store it is not: [f, cx, cy] = [219.92, 114.28, 108.52] implies
#      229x217 while the video is really 224x224 - a ~2% error that would bias EVERY
#      exported box"
# The store resolution is also pinned by the cache filename the loaders build,
# hand_bboxes_v2_rf1.5_res224x224.pt (train_hand_head.py:436).
# ===========================================================================

# Verbatim from scripts/hawor_boxes_from_detbox.py:54-55.
HOI4D_STORE_INTR = (219.92, 114.28, 108.52)
HOI4D_STORE_RES = 224.0


def test_frame_width_from_intr_is_wrong_on_the_hoi4d_store():
    from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
        frame_width_from_intr,
    )

    f, cx, cy = HOI4D_STORE_INTR
    intr = torch.tensor([[f, cx, cy]], dtype=torch.float64)
    w = float(frame_width_from_intr(intr)[0])

    # A square frame with a centred principal point forces 2*cx == 2*cy. It does not hold,
    # so the derivation cannot be valid on this store no matter what the true width is.
    assert abs(2.0 * cx - 2.0 * cy) < 1.0, (
        f"the module's own premise fails: 2*cx={2 * cx:.2f} vs 2*cy={2 * cy:.2f}"
    )

    assert abs(w - HOI4D_STORE_RES) < 1.0, (
        f"frame_width_from_intr returned {w:.2f} px for a store whose frames are "
        f"{HOI4D_STORE_RES:.0f}x{HOI4D_STORE_RES:.0f}. Every projected joint is normalised by a "
        f"width that is {100.0 * (w - HOI4D_STORE_RES) / HOI4D_STORE_RES:.1f}% too large, so the "
        "scene depth is sampled off the joint - the same failure class as the hardcoded 1408."
    )


def test_projected_frame_corner_is_not_at_the_frame_corner_on_hoi4d():
    """A joint that projects onto the LAST real row of the 224 px frame should
    normalise to u = 0. With W = 2*cx it lands 2% of the frame away instead."""
    from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
        project_joints_to_norm_pixels,
    )

    f, cx, cy = HOI4D_STORE_INTR
    intr = torch.tensor([[f, cx, cy]], dtype=torch.float64)
    z = 0.5
    # Choose y so that row = f*y/z + cy == HOI4D_STORE_RES - 1 (bottom row of the real frame).
    y = (HOI4D_STORE_RES - 1.0 - cy) * z / f
    x = (0.5 * HOI4D_STORE_RES - cx) * z / f          # column at the real frame's horizontal centre
    joints = torch.tensor([[[[[x, y, z]]]]], dtype=torch.float64)

    grid, _ = project_joints_to_norm_pixels(joints, intr)
    u_norm = float(grid[0, 0, 0, 0, 0])

    assert abs(u_norm) < 0.005, (
        f"a joint on the bottom row of the real {HOI4D_STORE_RES:.0f} px frame normalises to "
        f"u={u_norm:.4f} instead of 0.0, i.e. the depth is sampled "
        f"{u_norm * HOI4D_STORE_RES:.1f} px away from the joint."
    )


# ===========================================================================
# BUG 3 (HIGH) - half-pixel sampling offset: the normalised grid is built as
# i/R while sample_depth_at_joints uses grid_sample(align_corners=False), whose
# pixel CENTRE for index i is (i + 0.5)/R.
#
# The repo gets this right elsewhere and says so:
#     scripts/eval_world_space.py:258-259  u01 = (ix.float() + 0.5) / grid
#     ("Inverts the EXACT projection project_joints_to_norm_pixels uses to sample
#      this depth map")
# The i/R form appears in three places:
#     scripts/object_depth_loss.py:69        (training supervision)
#     scripts/eval_scene_metric_gt.py:205    (scene-metric eval)
#     scripts/eval_pred_depth_vs_gt.py:160   (the depth-vs-GT report figure)
# ===========================================================================

def _ramp_depth(res: int, step_m: float = 0.01, base_m: float = 1.0) -> torch.Tensor:
    """A depth map that increases by `step_m` per COLUMN, [res, res]."""
    col = base_m + step_m * torch.arange(res, dtype=torch.float32)
    return col.view(1, res).expand(res, res).contiguous()


def test_object_depth_loss_has_a_half_pixel_sampling_offset():
    """Feeding the SAME depth map as prediction and as ground truth must give a
    residual of exactly 0. It does not."""
    from scripts.object_depth_loss import object_depth_loss

    R = 16
    step = 0.01                                   # 10 mm per pixel
    d = _ramp_depth(R, step_m=step)
    gs_depth = d.view(1, 1, 1, R, R)              # [B,S,1,Hd,Wd]
    gt_obj_depth = d.view(1, 1, R, R)             # [B,S,R,R]
    gt_obj_mask = torch.ones(1, 1, R, R, dtype=torch.bool)

    loss, info = object_depth_loss(gs_depth, gt_obj_depth, gt_obj_mask)

    assert info["n_valid"] == R * R
    assert info["obj_depth_residual_m"] < 1e-6, (
        f"prediction == ground truth but the reported residual is "
        f"{1000.0 * info['obj_depth_residual_m']:.2f} mm on a {1000.0 * step:.0f} mm/px ramp "
        "(~half a pixel). scripts/object_depth_loss.py:69 builds the grid as i/R while "
        "sample_depth_at_joints uses align_corners=False, where pixel i is centred at "
        "(i+0.5)/R. eval_world_space.py:258 uses the +0.5 form."
    )


# ===========================================================================
# BUG 4 (MEDIUM) - the DexYCB joint-remap drift guard compares a hand-typed
# literal to itself, so it can never fire.
#
# scripts/preprocessing/dexycb_to_ours.py:52-55 claims:
#     "...it is BIT-IDENTICAL to H2O's H2O16_IDX = H2O_TO_MANO[:16] ...; that
#      equality is asserted at import time so the 2026-07-18 H2O scramble (which
#      corrupted every H2O number until anatomical bone lengths caught it) cannot
#      silently repeat here."
# dexycb_to_ours.py:158-160 compares DEXYCB21_TO_MANO16 against a local copy of the
# same literal, never against h2o_to_currentproto.H2O16_IDX.
# ===========================================================================

def test_dexycb_remap_guard_does_not_actually_track_the_h2o_constant():
    import scripts.preprocessing.dexycb_to_ours as dexycb
    import scripts.preprocessing.h2o_to_currentproto as h2o

    original = list(h2o.H2O16_IDX)
    # Simulate exactly the drift the guard claims to catch: the H2O remap gets scrambled.
    h2o.H2O16_IDX = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    try:
        with pytest.raises(RuntimeError):
            importlib.reload(dexycb)
    finally:
        h2o.H2O16_IDX = original
        importlib.reload(dexycb)


# ===========================================================================
# BUG 5 (MEDIUM) - HandToGSInjection clamps the DESTINATION rectangle but not the
# SOURCE crop, so a partially out-of-frame hand box has its whole crop squeezed
# into the visible sub-rectangle instead of being cropped.
#
# The module contract (hand_to_gs_injection.py:5-7, 49-50) is that each hand's
# features are scattered "back into the corresponding bbox region". Boxes that leave
# the frame are the normal case and are deliberately kept unclamped
# (scripts/hawor_boxes_from_detbox.py:32-34).
# ===========================================================================

def _inject_row0(x1n: float, x2n: float) -> torch.Tensor:
    from diffsynth.auxiliary_models.worldmirror.models.heads.hand_to_gs_injection import (
        HandToGSInjection,
    )

    mod = HandToGSInjection(hand_dim=1, gs_dims=(1,), use_hand_valid_mask=False)
    with torch.no_grad():
        mod.projs[0].weight.fill_(1.0)
        mod.projs[0].bias.fill_(0.0)

    # crop_size = 2; a left(0.0) -> right(1.0) ramp across the crop's two columns.
    tokens = torch.tensor([[[0.0], [1.0], [0.0], [1.0]],      # hand 0 (the one under test)
                           [[0.0], [0.0], [0.0], [0.0]]])     # hand 1
    bboxes = torch.tensor([[[[x1n, 0.0, x2n, 1.0],
                             [0.0, 0.0, 0.0, 0.0]]]])          # hand 1 degenerate -> skipped
    feats = [torch.zeros(1, 1, 8, 8)]
    return mod(tokens, bboxes, None, feats)[0][0, 0, 0]        # row 0 of the feature map


def _dexycb_principal_point(c: float, offset: float, s: float) -> float:
    """The pixel-centre resize form dexycb_to_ours.py:400-404 implements and documents:
    'The resize form is (c + 0.5) * s - 0.5, NOT c * s: under the pixel-centre convention
     plain multiplication lands the principal point half a pixel out and biases every crop box'
    """
    return (c - offset + 0.5) * s - 0.5


def test_h2o_and_hoi4d_intrinsics_use_the_wrong_resize_convention_for_cx_cy():
    """Three converters write the same `cam_intrinsics.pt` [f, cx, cy] contract, but
    dexycb_to_ours rescales the principal point under the pixel-centre convention while
    h2o_to_currentproto and preprocess_hoi4d use plain multiplication."""
    import numpy as np

    from scripts.preprocessing.h2o_to_currentproto import square_intrinsics
    from scripts.preprocessing.preprocess_hoi4d import load_intrinsics

    # --- H2O: 1280x720 -> centre square 720 -> 224 (scripts/pack_h2o.py:33-38, PIL resize) ---
    fx, fy, cx, cy, w, h = 636.6, 636.3, 635.3, 366.9, 1280.0, 720.0
    res = 224
    _, (_fxs, _fys, cx_s, cy_s) = square_intrinsics(
        np.array([fx, fy, cx, cy, w, h], dtype=np.float64), res
    )
    x0 = (int(w) - int(h)) // 2
    s = res / h
    assert abs(cx_s - _dexycb_principal_point(cx, x0, s)) < 1e-6, (
        f"H2O square_intrinsics cx = {cx_s:.4f}, pixel-centre form gives "
        f"{_dexycb_principal_point(cx, x0, s):.4f} (off by {0.5 * (s - 1.0):+.3f} px)"
    )
    assert abs(cy_s - _dexycb_principal_point(cy, 0.0, s)) < 1e-6

    # --- HOI4D: 1920x1080 -> centre square 1080 -> 224 (the packed store resolution,
    #     pinned by hand_bboxes_v2_rf1.5_res224x224.pt) ---
    K = np.array([[1060.4, 0.0, 970.9], [0.0, 1060.0, 523.2], [0.0, 0.0, 1.0]])
    intr = load_intrinsics(K, 1920, 1080, 224)
    s_h = 224.0 / 1080.0
    x0_h = (1920 - 1080) // 2
    assert abs(float(intr[1]) - _dexycb_principal_point(K[0, 2], x0_h, s_h)) < 1e-6, (
        f"HOI4D load_intrinsics cx = {float(intr[1]):.4f}, pixel-centre form gives "
        f"{_dexycb_principal_point(K[0, 2], x0_h, s_h):.4f} (off by {0.5 * (s_h - 1.0):+.3f} px)"
    )


def test_partially_out_of_frame_box_squeezes_the_crop_instead_of_cropping_it():
    """Two boxes that occupy the SAME visible columns but different real extents must
    not inject identical content: the wider box only shows its right-hand half there."""
    inside = _inject_row0(0.0, 0.5)      # box fully inside  -> feature cols 0..3
    half_out = _inject_row0(-0.5, 0.5)   # same visible cols 0..3, but the box is 2x as wide
                                         # and its LEFT half is off-frame

    # Column 0 now sits at the MIDDLE of the box, so it must receive mid-crop content
    # (~0.5 on the ramp), not the crop's left edge (~0.0).
    assert float(half_out[0]) > float(inside[0]) + 0.1, (
        "a box whose left half is off-frame injects byte-identical content into the visible "
        f"columns as a box half its width: {half_out[:4].tolist()} vs {inside[:4].tolist()}. "
        "hand_to_gs_injection.py:105-120 clamps x1/x2 into the frame and then resizes the FULL "
        "crop_size x crop_size token grid into the clamped rectangle, so hand features land at "
        "the wrong pixels whenever the hand leaves the frame."
    )
