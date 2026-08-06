"""BUG HUNTER B - proven defects.

Every test here fails against the code as it is today. Nothing is modified; each test
imports the module under test (or loads it by path when it lives outside a package) and
demonstrates the module violating its OWN documented contract, or contradicting a
producer/sibling that reads or writes the same tensor.

CPU only, torch/numpy only, no datasets, no checkpoints, no GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_by_path(name: str, rel: str):
    """Import a standalone script that is not inside an importable package."""
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------
# A synthetic right hand in OpenPose-21 order, the layout every consumer below claims
# to receive: 0 wrist; 1-3 thumb CMC/MCP/IP + 4 tip; 5-8 index; 9-12 middle;
# 13-16 ring; 17-20 pinky.  Wrist->MCP distances are ordinary adult-hand values and
# sit inside the repo's OWN anatomical bands (h2o_to_currentproto.py:464-466
# MCP_GATES), which is what makes the gate below a fair, repo-defined oracle.
# ---------------------------------------------------------------------------------
_OPENPOSE_FINGERS = {          # name -> (openpose base slot, wrist->MCP distance in mm)
    "thumb": (1, 35.0),
    "index": (5, 90.0),
    "middle": (9, 88.0),
    "ring": (13, 84.0),
    "pinky": (17, 78.0),
}
# The repo's own slot contract for the smplx-16 layout, copied from
# scripts/preprocessing/h2o_to_currentproto.py:464-466.
_MCP_GATES = {1: ("index MCP", 55.0, 115.0), 4: ("middle MCP", 60.0, 120.0),
              7: ("pinky MCP", 45.0, 100.0), 10: ("ring MCP", 50.0, 110.0),
              13: ("thumb CMC", 15.0, 65.0)}


def _openpose21_hand() -> torch.Tensor:
    """[21, 3] metres, OpenPose-21 order, wrist at the origin."""
    j = torch.zeros(21, 3)
    dirs = {"thumb": (0.80, 0.60, 0.0), "index": (0.20, 0.98, 0.0),
            "middle": (-0.05, 1.00, 0.0), "ring": (-0.30, 0.95, 0.0),
            "pinky": (-0.55, 0.84, 0.0)}
    for name, (base, mcp_mm) in _OPENPOSE_FINGERS.items():
        d = torch.tensor(dirs[name])
        d = d / d.norm()
        for k in range(3):                       # MCP, PIP, DIP  (then the tip)
            j[base + k] = d * (mcp_mm + 25.0 * k) / 1000.0
        j[base + 3] = d * (mcp_mm + 95.0) / 1000.0
    return j


def _gate_failures(j16: torch.Tensor) -> list[str]:
    """Which of the repo's MCP_GATES bands the given smplx-16 hand violates."""
    bad = []
    for slot, (name, lo, hi) in _MCP_GATES.items():
        dist_mm = float((j16[slot] - j16[0]).norm()) * 1000.0
        if not (lo <= dist_mm <= hi):
            bad.append(f"slot {slot:2d} {name:<11s} {dist_mm:6.1f} mm (band {lo:.0f}-{hi:.0f})")
    return bad


# =================================================================================
# BUG 1 - the WiLoR/HaMeR baseline producer uses the WRONG OpenPose-21 -> smplx-16
# permutation, so every native-baseline row in the world/global comparison table is
# scored with the thumb in the index slot and the pinky in the thumb slot.
#
# scripts/build_native_baseline_preds.py:47-50 states its input layout ("MANO-21
# native order, tips interleaved 4,8,12,16,20" = OpenPose-21) and claims the subset
# is "copied VERBATIM from run_wilor_h2o.py / eval_cmpjpe.py so the joint
# correspondence is identical to every other row".  The repo's canonical map for
# exactly that input layout is [0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3]
# (haptic_to_worldeval.OP2SMPLX16:36, preprocess_hoi4d._KPS2D_FOR_SMPLX16:226,
# eval_cmpjpe.H2O_TO_MANO[:16]).  build_native_baseline_preds instead just drops the
# tips, which preserves OpenPose finger order (thumb,index,middle,ring,pinky) rather
# than producing smplx order (index,middle,pinky,ring,thumb).
# =================================================================================
def test_bug1_native_baseline_joint_remap_is_the_wrong_permutation():
    from scripts.build_native_baseline_preds import MANO21_TO_16
    from scripts.haptic_to_worldeval import OP2SMPLX16
    from scripts.preprocessing.preprocess_hoi4d import _KPS2D_FOR_SMPLX16

    # Two sibling adapters feeding the SAME scorer (eval_worldspace_baseline) from the
    # SAME OpenPose-21 input layout must use the same map.
    assert OP2SMPLX16 == _KPS2D_FOR_SMPLX16, "sanity: the repo's canonical map is consistent"

    hand_op = _openpose21_hand()
    j_canonical = hand_op[OP2SMPLX16]
    j_baseline = hand_op[MANO21_TO_16]

    err_mm = float((j_baseline - j_canonical).norm(dim=-1).mean()) * 1000.0
    bad_canonical = _gate_failures(j_canonical)
    bad_baseline = _gate_failures(j_baseline)

    assert MANO21_TO_16 == OP2SMPLX16, (
        "scripts/build_native_baseline_preds.py:50 uses a DIFFERENT OpenPose-21 -> "
        "smplx-16 permutation than every other consumer of the same layout, despite "
        "its own comment claiming it is copied verbatim from eval_cmpjpe.py.\n"
        f"  canonical (haptic_to_worldeval:36 / preprocess_hoi4d:226): {OP2SMPLX16}\n"
        f"  build_native_baseline_preds:50                          : {MANO21_TO_16}\n"
        f"  same index SET, different order -> a PERFECT prediction scores "
        f"{err_mm:.1f} mm root-relative through this map.\n"
        f"  repo's own MCP_GATES (h2o_to_currentproto.py:464) on the canonical map: "
        f"{len(bad_canonical)} failures\n"
        f"  ... on build_native_baseline_preds' map: {len(bad_baseline)} failures -> "
        + "; ".join(bad_baseline)
    )


# =================================================================================
# BUG 2 - the contact gate samples the GT dense depth map at a 90-degree-rotated
# pixel, so the "wrist sits on the visible surface" test is decided by the depth of
# an unrelated part of the frame.
#
# Producer: scripts/build_contact_cache.py:31-33 loads the raw HOI4D depth PNG,
# centre-square-crops and resizes it -> a plain [row, col] map, paired with
# preprocess_hoi4d.load_intrinsics (:85-97), the plain unrotated pinhole for exactly
# that crop+resize.  Consumer: scripts/contact_mask.py:35-36 samples it through
# project_joints_to_norm_pixels, which applies the Aria u=(W-1)-row / v=col rotation
# (hand_depth_sampling.py:19-25).  Nothing rotates the depth map to match.
# =================================================================================
def test_bug2_contact_mask_samples_gt_depth_at_a_rotated_pixel():
    from scripts.contact_mask import wrist_contact_mask
    from scripts.root_depth_anchor import (
        project_joints_to_norm_pixels, sample_depth_at_joints)

    R = 224
    f, cx, cy = 220.0, R / 2.0, R / 2.0
    cam_intr = torch.tensor([[f, cx, cy]])

    # A slanted surface whose depth varies along the image ROW, exactly the
    # [row, col] layout build_contact_cache emits.
    rows = torch.arange(R).float().view(R, 1).expand(R, R).contiguous()
    dense = (0.40 + 0.004 * rows).view(1, 1, 1, R, R)

    # Put the wrist EXACTLY on that surface: choose a pixel, read the plane depth
    # there, and back-project through the same plain pinhole the intrinsics describe.
    col, row = 170.0, 60.0
    z_true = 0.40 + 0.004 * row                      # 0.64 m
    x = (col - cx) * z_true / f
    y = (row - cy) * z_true / f
    wrist = torch.tensor([[[[x, y, z_true], [x, y, z_true]]]])     # [B,S,2,3]

    grid, _ = project_joints_to_norm_pixels(wrist.unsqueeze(3), cam_intr)
    sampled, _ = sample_depth_at_joints(dense, grid)
    sampled_m = float(sampled[0, 0, 1, 0])
    contact = wrist_contact_mask(wrist, dense, cam_intr, thresh_m=0.05)

    assert bool(contact[0, 0, 1]), (
        "wrist_contact_mask says NO CONTACT for a wrist lying EXACTLY on the GT "
        "surface (true residual 0.0 mm, threshold 50 mm).\n"
        f"  wrist projects to (col={col:.0f}, row={row:.0f}); true surface depth "
        f"{z_true:.4f} m\n"
        f"  depth actually sampled                : {sampled_m:.4f} m "
        f"(error {abs(sampled_m - z_true) * 1000:.1f} mm)\n"
        f"  normalised grid used                  : {grid.view(-1)[:2].tolist()}\n"
        f"  unrotated grid the producer implies   : [{col / R:.4f}, {row / R:.4f}]\n"
        "  contact_mask.py:35 feeds an UNROTATED [row,col] GT depth map (built by "
        "build_contact_cache.py:31-33) through project_joints_to_norm_pixels, which "
        "applies the Aria u=(W-1)-row / v=col rotation."
    )


# =================================================================================
# BUG 3 - object_depth_loss and the hand-depth-anchor / metric-scale path sample the
# SAME gs_depth tensor at DIFFERENT pixels for the same 3D point, so the object-depth
# supervision pulls gs_depth where the metric-scale solve never reads it.
#
# object_depth_loss.py:69 hand-builds grid = (col/R, row/R).  Every other consumer of
# gs_depth (hand_depth_anchor_loss.py:73, scale_head_loss.py:49, metric_scale_head.py:56,
# hand_scene_registration_loss.py:77, root_depth_anchor.py:43) goes through
# project_joints_to_norm_pixels, i.e. ((W-1)-row)/W, col/W.  Both call the same
# sample_depth_at_joints on the same tensor; they cannot both be right.
# =================================================================================
def test_bug3_object_depth_loss_samples_gs_depth_with_a_different_grid_convention():
    from scripts.object_depth_loss import object_depth_loss
    from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
        project_joints_to_norm_pixels, sample_depth_at_joints)

    R = 64
    f, cx, cy = 60.0, R / 2.0, R / 2.0           # 2*cx == R, so W_NORM == R
    cam_intr = torch.tensor([[f, cx, cy]])
    rows = torch.arange(R).float().view(R, 1).expand(R, R)
    cols = torch.arange(R).float().view(1, R).expand(R, R)
    gs_depth = (1.0 + 0.01 * rows + 0.0001 * cols).contiguous().view(1, 1, 1, R, R)

    # One 3D camera-frame point on an object surface.
    p = torch.tensor([0.10, -0.06, 0.50])
    col = float(f * p[0] / p[2] + cx)
    row = float(f * p[1] / p[2] + cy)
    ci, ri = int(round(col)), int(round(row))    # b2_render_object_depth.py:142-147

    # gt_obj_depth is written [row, col] by render_object_depth (depth[v*W+u], u=col).
    mask = torch.zeros(1, 1, R, R, dtype=torch.bool)
    mask[0, 0, ri, ci] = True
    od_val = 0.5
    gt_obj = torch.zeros(1, 1, R, R)
    gt_obj[0, 0, ri, ci] = od_val

    _, info = object_depth_loss(gs_depth, gt_obj, mask)
    # every gs value > od_val, so the |.| residual recovers the sampled value
    gs_from_objloss = od_val + info["obj_depth_residual_m"]

    grid, _ = project_joints_to_norm_pixels(p.view(1, 1, 1, 1, 3), cam_intr)
    gs_from_helper = float(sample_depth_at_joints(gs_depth, grid)[0])

    assert abs(gs_from_objloss - gs_from_helper) < 1e-3, (
        "object_depth_loss and the hand-depth-anchor / metric-scale path read the "
        "SAME gs_depth tensor at DIFFERENT pixels for the same 3D point.\n"
        f"  3D point projects to (col={col:.2f}, row={row:.2f})\n"
        f"  object_depth_loss.py:69 grid  : [{ci / R:.4f}, {ri / R:.4f}] -> gs = "
        f"{gs_from_objloss:.4f} m\n"
        f"  project_joints_to_norm_pixels : {[round(v, 4) for v in grid.view(-1).tolist()]}"
        f" -> gs = {gs_from_helper:.4f} m\n"
        f"  disagreement {abs(gs_from_objloss - gs_from_helper) * 1000:.1f} mm; the "
        "object grid is (col, row) while the helper's is ((W-1)-row, col)."
    )


# =================================================================================
# BUG 4 - the H2O ground truth used to score the HaWoR baseline is remapped TWICE,
# scrambling the joints (three fingertips land in kinematic base slots).
#
# scripts/eval_cmpjpe.py:46,56,69 is the canonical definition: base = H2O_TO_MANO[:16],
# tips = H2O_TO_MANO[16:].  h2o_to_currentproto.py:120-125 records the 2026-07-18 bug
# fix that removed exactly the "compose H2O_TO_MANO with a 21->16 selector" pattern.
# report/poster_assets/hawor/extract_h2o_clips_mp4.py:22-23 re-introduces it, while its
# docstring claims "joint conventions copied verbatim from scripts/eval_cmpjpe.py".
# =================================================================================
def _eval_cmpjpe_h2o_to_mano():
    """Read eval_cmpjpe.H2O_TO_MANO without importing it (it pulls in WorldMirror)."""
    import ast
    tree = ast.parse((ROOT / "scripts" / "eval_cmpjpe.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "H2O_TO_MANO":
            return ast.literal_eval(node.value)
    raise AssertionError("H2O_TO_MANO not found in scripts/eval_cmpjpe.py")


def test_bug4_hawor_h2o_gt_joint_remap_is_double_applied():
    extract = _load_by_path("hunt_extract_h2o",
                            "report/poster_assets/hawor/extract_h2o_clips_mp4.py")
    from scripts.preprocessing.h2o_to_currentproto import H2O_TO_MANO as H2O_REF, H2O16_IDX

    H2O_TO_MANO = _eval_cmpjpe_h2o_to_mano()
    assert H2O_TO_MANO == H2O_REF and H2O16_IDX == H2O_TO_MANO[:16], "sanity: references agree"
    GT_TIP_IDX = H2O_TO_MANO[16:]
    assert extract.H2O_TO_MANO == H2O_TO_MANO, "sanity: the base table itself matches"

    # Synthetic H2O hand_pose row: H2O joint j carries the marker vector (j, j, j),
    # so the first coordinate of each output slot names the H2O index it came from.
    j128 = np.zeros((1, 128), np.float32)
    j128[0, 0] = 1.0                       # left valid
    j128[0, 64] = 1.0                      # right valid
    marker = np.repeat(np.arange(21, dtype=np.float32)[:, None], 3, axis=1).reshape(-1)
    j128[0, 1:64] = marker
    j128[0, 65:128] = marker

    got, _ = extract.gt_joints(j128)       # [1, 2, 21, 3]
    want = marker.reshape(21, 3)[H2O_TO_MANO[:16] + GT_TIP_IDX]

    got_ids = got[0, 1, :, 0].astype(int).tolist()     # right hand
    want_ids = want[:, 0].astype(int).tolist()
    assert got_ids == want_ids, (
        "extract_h2o_clips_mp4.gt_joints emits a DIFFERENT H2O joint order than "
        "eval_cmpjpe: it composes H2O_TO_MANO with MANO21_TO_16 (the tips-interleaved "
        "selector), i.e. the remap is applied twice.\n"
        f"  eval_cmpjpe (canonical) H2O indices : {want_ids}\n"
        f"  extract_h2o_clips_mp4   H2O indices : {got_ids}\n"
        "  fingertips leaking into base slots  : "
        f"{sorted(set(got_ids[:16]) & set(H2O_TO_MANO[16:]))}"
    )


# =================================================================================
# BUG 5 - the "loss effect check" guard, which exists specifically to catch a loss
# that is weighted but never actually computed, has a hole exactly where the root/
# contact-anchor loss lives: it SKIPS any weighted term that is missing from
# avg_terms and then prints "PASSED: every weighted loss is actually firing".
#
# scripts/train_hand_head.py:1769-1770 does `if name not in avg_terms: continue`.
# avg_terms is built from accum_terms (:2879), and `loss_root_anchor` (:2562, :2635,
# summed into the loss at :2778 as `w.get("root_anchor", 0.0) * ...`) is never
# accumulated into it.  Five shipped configs set root_anchor: 1.0, and :2562
# initialises the term to an exact zero that survives the whole run whenever
# enable_root_anchor / cam_intrinsics / the depth reference is missing - the exact
# failure mode the docstring says this check catches.
# =================================================================================
def _load_check_loss_effect():
    """exec just the _check_loss_effect source (train_hand_head needs decord)."""
    import ast
    src = (ROOT / "scripts" / "train_hand_head.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_check_loss_effect":
            ns: dict = {}
            exec(compile(ast.Module([node], []), "train_hand_head.py", "exec"), ns)
            return ns["_check_loss_effect"]
    raise AssertionError("_check_loss_effect not found in scripts/train_hand_head.py")


def _tracked_loss_terms() -> set[str]:
    """The exact key set the training loop accumulates into avg_terms."""
    import re
    src = (ROOT / "scripts" / "train_hand_head.py").read_text()
    return set(re.findall(r'accum_terms\["([a-z0-9_]+)"\]', src))


def test_bug5_loss_effect_guard_silently_skips_the_root_anchor_loss():
    import re
    check = _load_check_loss_effect()
    tracked = _tracked_loss_terms()
    src = (ROOT / "scripts" / "train_hand_head.py").read_text()

    # Ground the premise in the real code, not in an invented scenario:
    assert 'w.get("root_anchor", 0.0)' in src, "root_anchor really is a weighted loss term"
    assert "root_anchor" not in tracked, (
        "premise: loss_root_anchor is never accumulated into avg_terms "
        f"(tracked terms: {sorted(tracked)})")
    assert re.search(r"loss_root_anchor = torch\.zeros", src), \
        "premise: the term is initialised to an exact zero that can survive a whole run"

    # A run declaring root_anchor: 1.0 (configs/exp_p4_contact.yaml et al.) whose
    # anchor never fires: the term contributes exactly 0.0 for every step.
    loss_weights = {"kp3d_abs": 1.0, "transl": 1.0, "root_anchor": 1.0}
    avg_terms = {k: 0.37 for k in tracked}          # every TRACKED term is healthy

    with pytest.raises(SystemExit) as excinfo:
        check(loss_weights, avg_terms, 50, strict=True)
    assert "root_anchor" in str(excinfo.value)
