# Contact anchor Phase 1 (scene-depth root correction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A learned, feedforward, post-hoc module that corrects the predicted hand-root depth toward the metric `gs_depth` sampled at the predicted wrist, so the world-space hand track stops drifting (a SLAM-free analog of the `re16` GT-reanchor oracle).

**Architecture:** After the model produces `params` and `gs_depth`, project the predicted wrist (joint 0), sample `gs_depth` (detached) there, and a small zero-init MLP emits a per-hand depth shift `Δz` (gated by depth confidence, in-frame, and a plausibility band). `Δz` rigidly shifts the hand in camera depth (all joints `z += Δz`). The module is a submodule of `WorldMirror` (trained, saved); the project/sample/apply orchestration is a shared helper called by both the train loop and the world eval. A gated consistency loss pulls the corrected wrist depth toward `gs_depth`. `gs_depth` is detached, so with a frozen backbone there is no scene<->hand circularity.

**Tech Stack:** PyTorch. Reuses `project_joints_to_norm_pixels` + `sample_depth_at_joints` (`diffsynth/.../utils/hand_depth_sampling.py`), `compute_joints_from_batch` (`scripts/train_hand_head.py`). Unit tests run on the Mac (CPU, torch 2.7); integration runs on the cluster (gb10) via the existing single-srun harness.

---

## File structure

- Create `diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py` — `RootDepthRefine` nn.Module (zero-init MLP + gating). One responsibility: map per-hand `(wrist_z, d_scene, conf, in_frame)` to a gated `Δz`.
- Create `scripts/root_depth_anchor.py` — stateless orchestration: `apply_root_anchor(module, pred_joints, gs_depth, gs_depth_conf, cam_intr)` (project wrist, sample, run module, apply `Δz`) and `root_anchor_loss(corrected_wrist_z, d_scene, gate, has_hand)`. Shared by train + eval.
- Create `tests/test_root_depth_refine.py` — CPU unit tests (Mac-runnable).
- Modify `diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py` — own `self.root_depth_refine` when `enable_root_anchor` is set; expose it in `get_config`.
- Modify `scripts/train_hand_head.py` — after `pred_joints` (line 1719): apply the anchor, use corrected joints for kp losses, add the gated consistency loss term + weight; surface the module in the "Trainable parameters" print and ensure it is not frozen.
- Modify `scripts/eval_world_space.py` — apply the same correction to `pred_joints` in the chained world path (line ~356) before world placement, behind the same flag.
- Create `configs/exp_p4_contact.yaml` — fork of `exp_p3_wabs.yaml`: `enable_gs: true` (needed for `gs_depth`), `enable_root_anchor: true`, `root_anchor` loss weight, frozen backbone.

---

### Task 1: RootDepthRefine module

**Files:**
- Create: `diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py`
- Test: `tests/test_root_depth_refine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_root_depth_refine.py
import torch
from diffsynth.auxiliary_models.worldmirror.models.heads.root_depth_refine import RootDepthRefine


def _inputs(B=2, S=3):
    wrist_z = torch.full((B, S, 2), 0.40)
    d_scene = torch.full((B, S, 2), 0.45)
    conf = torch.full((B, S, 2), 0.9)
    in_frame = torch.ones(B, S, 2, dtype=torch.bool)
    return wrist_z, d_scene, conf, in_frame


def test_zero_init_is_noop():
    m = RootDepthRefine()
    dz, gate = m(*_inputs())
    assert torch.allclose(dz, torch.zeros_like(dz)), "zero-init must emit dz=0 (warm-start preserved)"
    assert gate.all(), "high conf, in frame, small disagreement -> gated ON"


def test_gate_off_when_conf_low():
    m = RootDepthRefine(conf_thresh=0.5)
    wrist_z, d_scene, _, in_frame = _inputs()
    conf = torch.full_like(wrist_z, 0.1)
    dz, gate = m(wrist_z, d_scene, conf, in_frame)
    assert not gate.any(), "conf below threshold must gate OFF"
    assert torch.allclose(dz, torch.zeros_like(dz)), "gated-off dz must be 0"


def test_gate_off_when_disagreement_exceeds_band():
    m = RootDepthRefine(band_m=0.2)
    wrist_z, _, conf, in_frame = _inputs()
    d_scene = wrist_z + 1.0  # 1 m disagreement -> free-space background
    dz, gate = m(wrist_z, d_scene, conf, in_frame)
    assert not gate.any(), "disagreement beyond band must gate OFF"


def test_nonzero_weights_move_toward_scene():
    m = RootDepthRefine()
    torch.nn.init.constant_(m.net[-1].weight, 0.0)
    torch.nn.init.constant_(m.net[-1].bias, 0.0)
    # force a known positive response on the (d_scene - wrist_z) feature (index 2)
    with torch.no_grad():
        m.net[-1].weight[0, :] = 0.0
        m.net[0].weight.zero_(); m.net[0].bias.zero_()
        m.net[0].weight[0, 2] = 1.0          # pass disagreement through
        m.net[-1].weight[0, 0] = 1.0          # and out
    wrist_z, d_scene, conf, in_frame = _inputs()  # d_scene - wrist_z = +0.05
    dz, gate = m(wrist_z, d_scene, conf, in_frame)
    assert (dz[gate] > 0).all(), "positive disagreement should yield a positive shift toward scene"


if __name__ == "__main__":
    test_zero_init_is_noop()
    test_gate_off_when_conf_low()
    test_gate_off_when_disagreement_exceeds_band()
    test_nonzero_weights_move_toward_scene()
    print("PASS: RootDepthRefine unit tests")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_root_depth_refine.py`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py
"""Post-hoc root-depth correction (contact anchor Phase 1).

Given per-hand (wrist_z, d_scene, conf, in_frame), predict a depth shift delta_z
applied rigidly to the hand root. Zero-init -> delta_z = 0 at start, so a
warm-started head is unchanged until the module learns a correction. delta_z is
masked OFF where the scene-depth reference is unreliable (low confidence, out of
frame, or disagreeing with the head's own estimate by more than `band_m`), which
is the Phase-1 contact proxy; Phase 2 replaces the gate with an explicit
hand-object contact signal.
"""
import torch
import torch.nn as nn


class RootDepthRefine(nn.Module):
    def __init__(self, hidden: int = 32, conf_thresh: float = 0.1, band_m: float = 0.5):
        super().__init__()
        self.conf_thresh = float(conf_thresh)
        self.band_m = float(band_m)
        # features: [wrist_z, d_scene, d_scene - wrist_z, conf]
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, wrist_z, d_scene, conf, in_frame):
        """All inputs [B, S, 2] (per hand). Returns (delta_z [B,S,2], gate bool [B,S,2])."""
        disagree = d_scene - wrist_z
        feats = torch.stack([wrist_z, d_scene, disagree, conf], dim=-1)  # [B,S,2,4]
        delta = self.net(feats).squeeze(-1)  # [B,S,2]
        gate = in_frame & (conf > self.conf_thresh) & (disagree.abs() < self.band_m)
        return delta * gate.float(), gate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_root_depth_refine.py`
Expected: `PASS: RootDepthRefine unit tests`

- [ ] **Step 5: Commit**

```bash
git add diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py tests/test_root_depth_refine.py
git commit -m "feat(contact): RootDepthRefine module (zero-init gated root-depth shift)"
```

---

### Task 2: Anchor orchestration + consistency loss

**Files:**
- Create: `scripts/root_depth_anchor.py`
- Test: `tests/test_root_depth_anchor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_root_depth_anchor.py
import torch
from diffsynth.auxiliary_models.worldmirror.models.heads.root_depth_refine import RootDepthRefine
from scripts.root_depth_anchor import apply_root_anchor, root_anchor_loss


def _scene(B=1, S=2, Hd=64, Wd=64, depth=0.5):
    return torch.full((B, S, 1, Hd, Wd), depth), torch.full((B, S, 1, Hd, Wd), 0.9)


def test_apply_is_noop_at_init():
    B, S = 1, 2
    pred_joints = torch.randn(B, S, 2, 16, 3).abs() + 0.3   # positive z in frame-ish
    pred_joints[..., 2] = 0.5                                # wrist+joints at 0.5 m
    gs_depth, gs_conf = _scene(B, S, depth=0.5)
    cam_intr = torch.tensor([[600.0, 704.0, 704.0]])         # [B,3]
    m = RootDepthRefine()
    corrected, dz, info = apply_root_anchor(m, pred_joints, gs_depth, gs_conf, cam_intr)
    assert torch.allclose(dz, torch.zeros_like(dz)), "zero-init -> no shift"
    assert torch.allclose(corrected, pred_joints), "zero-init -> joints unchanged"
    assert "d_scene" in info and "gate" in info


def test_apply_shifts_only_z():
    B, S = 1, 1
    pred_joints = torch.zeros(B, S, 2, 16, 3)
    pred_joints[..., 2] = 0.5
    gs_depth, gs_conf = _scene(B, S, depth=0.5)
    cam_intr = torch.tensor([[600.0, 704.0, 704.0]])
    m = RootDepthRefine()
    with torch.no_grad():           # force a constant +0.1 shift through the gate
        m.net[0].weight.zero_(); m.net[0].bias.zero_()
        m.net[-1].weight.zero_(); m.net[-1].bias.fill_(0.1)
    corrected, dz, info = apply_root_anchor(m, pred_joints, gs_depth, gs_conf, cam_intr)
    assert torch.allclose(corrected[..., :2], pred_joints[..., :2]), "x,y unchanged"
    assert torch.allclose(corrected[..., 2], pred_joints[..., 2] + dz[..., None]), "z shifted by per-hand dz"


def test_consistency_loss_zero_when_matched():
    wrist_z = torch.full((1, 2, 2), 0.5)
    d_scene = torch.full((1, 2, 2), 0.5)
    gate = torch.ones(1, 2, 2, dtype=torch.bool)
    has_hand = torch.ones(1, 2, 2)
    loss = root_anchor_loss(wrist_z, d_scene, gate, has_hand)
    assert float(loss) == 0.0, "matched depth -> zero anchor loss"


def test_consistency_loss_ignores_gated_off():
    wrist_z = torch.zeros(1, 1, 2)
    d_scene = torch.ones(1, 1, 2)            # large disagreement
    gate = torch.zeros(1, 1, 2, dtype=torch.bool)  # all gated off
    has_hand = torch.ones(1, 1, 2)
    loss = root_anchor_loss(wrist_z, d_scene, gate, has_hand)
    assert float(loss) == 0.0, "all gated-off -> zero loss (no NaN from empty mean)"


if __name__ == "__main__":
    test_apply_is_noop_at_init()
    test_apply_shifts_only_z()
    test_consistency_loss_zero_when_matched()
    test_consistency_loss_ignores_gated_off()
    print("PASS: root_depth_anchor unit tests")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_root_depth_anchor.py`
Expected: FAIL with `ImportError` (`scripts/root_depth_anchor.py` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/root_depth_anchor.py
"""Orchestration for the Phase-1 scene-depth root anchor: project the predicted
wrist, sample the (detached) metric gs_depth there, run RootDepthRefine, and apply
the per-hand depth shift to the joints. Shared by train_hand_head and
eval_world_space so train-time and eval-time corrections are identical.
"""
import torch
import torch.nn.functional as F

from diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling import (
    project_joints_to_norm_pixels, sample_depth_at_joints,
)

WRIST_J = 0  # MANO joint 0 = wrist (pelvis_id used by the kp losses)


def apply_root_anchor(module, pred_joints, gs_depth, gs_depth_conf, cam_intr):
    """pred_joints [B,S,2,J,3] camera-frame (m). gs_depth [B,S,1,Hd,Wd] (detached
    inside). cam_intr [B,3]. Returns (corrected_joints, delta_z [B,S,2], info)."""
    wrist = pred_joints[:, :, :, WRIST_J:WRIST_J + 1, :]          # [B,S,2,1,3]
    grid_xy, z = project_joints_to_norm_pixels(wrist, cam_intr)   # [B,S,2,1,2], [B,S,2,1]
    d_scene, in_frame = sample_depth_at_joints(gs_depth.detach(), grid_xy)  # [B,S,2,1]
    if gs_depth_conf is not None:
        conf, _ = sample_depth_at_joints(gs_depth_conf.detach(), grid_xy)
    else:
        conf = torch.ones_like(d_scene)
    wrist_z = z[..., 0]
    d_scene = d_scene[..., 0]
    conf = conf[..., 0]
    in_frame = in_frame[..., 0]
    in_frame = in_frame & (d_scene > 0.01) & torch.isfinite(d_scene) & torch.isfinite(wrist_z)

    delta_z, gate = module(wrist_z, d_scene, conf, in_frame)       # [B,S,2]
    corrected = pred_joints.clone()
    corrected[..., 2] = corrected[..., 2] + delta_z.unsqueeze(-1)  # rigid depth shift per hand
    info = {"d_scene": d_scene, "wrist_z": wrist_z, "conf": conf, "gate": gate}
    return corrected, delta_z, info


def root_anchor_loss(corrected_wrist_z, d_scene, gate, has_hand, delta_m: float = 0.05):
    """Gated Huber pulling the corrected wrist depth toward the scene depth.
    All [B,S,2]. Zero (no grad) when nothing is gated, avoiding an empty-mean NaN."""
    mask = (gate & (has_hand > 0.5)).float()
    denom = mask.sum()
    if float(denom) < 1.0:
        return corrected_wrist_z.sum() * 0.0
    per = F.huber_loss(corrected_wrist_z, d_scene, reduction="none", delta=delta_m)
    return (mask * per).sum() / denom
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_root_depth_anchor.py`
Expected: `PASS: root_depth_anchor unit tests`

- [ ] **Step 5: Commit**

```bash
git add scripts/root_depth_anchor.py tests/test_root_depth_anchor.py
git commit -m "feat(contact): root anchor orchestration + gated consistency loss"
```

---

### Task 3: Own the module on WorldMirror

**Files:**
- Modify: `diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py` (init block near line 228; `get_config` near line 113)

- [ ] **Step 1: Add the submodule (gated by config)**

In `__init__`, near where the hand head is built (after the `HamerManoHead(...)` block, ~line 238), add:

```python
        self.enable_root_anchor = kwargs.get("enable_root_anchor", False)
        if self.enable_root_anchor:
            from ..heads.root_depth_refine import RootDepthRefine
            ra = kwargs.get("root_anchor_kwargs", {}) or {}
            self.root_depth_refine = RootDepthRefine(**ra)
```

In `get_config` (the dict returned ~line 113), add the key so a reloaded model rebuilds it:

```python
            "enable_root_anchor": getattr(self, "enable_root_anchor", False),
```

- [ ] **Step 2: Verify it constructs (smoke, CPU)**

Run:
```bash
python -c "
from diffsynth.auxiliary_models.worldmirror.models.heads.root_depth_refine import RootDepthRefine
import torch
m = RootDepthRefine()
print('params', sum(p.numel() for p in m.parameters()))
"
```
Expected: prints a small positive parameter count (a few thousand). Full model construction is validated on the cluster in Task 6 (needs the backbone checkpoint).

- [ ] **Step 3: Commit**

```bash
git add diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py
git commit -m "feat(contact): WorldMirror owns RootDepthRefine when enable_root_anchor"
```

---

### Task 4: Wire into training

**Files:**
- Modify: `scripts/train_hand_head.py` (after `pred_joints` at line 1719; loss assembly ~line 1839; trainable-params print ~line 1530; loss config read ~line 1357)

- [ ] **Step 1: Read the anchor weight from config**

After `max_steps = ...` (the line added earlier, ~1358), add:

```python
    w_root_anchor = float(cfg["loss_weights"].get("root_anchor", 0.0))
    root_anchor_warmup_steps = int(training_cfg.get("root_anchor_warmup_steps", 0))
```

- [ ] **Step 2: Apply the correction after pred_joints**

Immediately after `pred_joints = compute_joints_from_batch(pred_params, mano_model, device)` (line 1719), add:

```python
            loss_root_anchor = torch.zeros((), device=device)
            if getattr(model, "enable_root_anchor", False) and "cam_intrinsics" in batch:
                from scripts.root_depth_anchor import apply_root_anchor, root_anchor_loss
                gs_depth_pred = preds.get("gs_depth")
                if gs_depth_pred is not None:
                    pred_joints, _delta_z, _ra = apply_root_anchor(
                        model.root_depth_refine, pred_joints, gs_depth_pred,
                        preds.get("gs_depth_conf"), batch["cam_intrinsics"].to(device),
                    )
                    # has_hand is computed below; defer the loss until after it exists.
                    _ra_inputs = (pred_joints[:, :, :, 0, 2], _ra["d_scene"], _ra["gate"])
                else:
                    _ra_inputs = None
            else:
                _ra_inputs = None
```

Then, after `has_hand` is computed (it is derived just below, ~line 1738), add:

```python
            if _ra_inputs is not None:
                from scripts.root_depth_anchor import root_anchor_loss
                _cw, _ds, _gate = _ra_inputs
                ra_ramp = min(1.0, global_step / root_anchor_warmup_steps) if root_anchor_warmup_steps > 0 else 1.0
                loss_root_anchor = ra_ramp * root_anchor_loss(_cw, _ds, _gate, has_hand)
```

NOTE: this reassigns `pred_joints` to the corrected joints before the kp3d / kp3d_abs losses are computed, so those losses already supervise the corrected placement. Leave `pred_params` (used by `criterion_param`) unchanged; the absolute supervision now flows through `kp3d_abs` on the corrected joints, and the param transl loss stays a weak secondary signal.

- [ ] **Step 3: Add the loss term**

In the `loss = (...)` sum (~line 1839), add a line:

```python
                + w_root_anchor * loss_root_anchor
```

- [ ] **Step 4: Ensure the module is trainable and visible**

The module lives under `model` and is not the backbone, so the existing freeze logic (which freezes only the backbone) leaves it trainable. Confirm by extending the "Trainable parameters" print (~line 1530) to include it:

```python
    n_root_anchor = sum(p.numel() for p in getattr(model, "root_depth_refine", torch.nn.Module()).parameters())
```
and append `root_anchor={n_root_anchor:,}` to that print string.

- [ ] **Step 5: Compile-check**

Run: `python -m py_compile scripts/train_hand_head.py`
Expected: no output (success).

- [ ] **Step 6: Commit**

```bash
git add scripts/train_hand_head.py
git commit -m "feat(contact): apply root anchor + gated consistency loss in training"
```

---

### Task 5: Wire into world eval

**Files:**
- Modify: `scripts/eval_world_space.py` (chained world path, the `compute_joints_from_batch` at line ~356)

- [ ] **Step 1: Apply the correction before world placement**

Replace the line `pj = compute_joints_from_batch(preds["hand_joints"], mano_model, device)[0].float().cpu()` (line ~356) with:

```python
            pj_full = compute_joints_from_batch(preds["hand_joints"], mano_model, device)  # [1,S,2,16,3] cam
            if getattr(model, "enable_root_anchor", False) and preds.get("gs_depth") is not None:
                from scripts.root_depth_anchor import apply_root_anchor
                pj_full, _, _ = apply_root_anchor(
                    model.root_depth_refine, pj_full, preds["gs_depth"],
                    preds.get("gs_depth_conf"), cam_intr.to(device),
                )
            pj = pj_full[0].float().cpu()  # [S,2,16,3] cam (m)
```

(`cam_intr` is already in scope in this loop; it is loaded per sequence from the hand-data cache.)

- [ ] **Step 2: Compile-check**

Run: `python -m py_compile scripts/eval_world_space.py`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_world_space.py
git commit -m "feat(contact): apply root anchor in chained world-space eval"
```

---

### Task 6: Config + bounded train/eval run

**Files:**
- Create: `configs/exp_p4_contact.yaml`

- [ ] **Step 1: Write the config (fork of exp_p3_wabs, gs on, anchor on)**

```yaml
# P4-CONTACT: Phase-1 scene-depth root anchor. Forked from exp_p3_wabs.yaml.
# Deltas vs wabs: enable_gs true (gs_depth is the anchor reference), root anchor on,
# root_anchor loss weight. Frozen backbone (cheap probe). gs_depth is detached in the
# anchor, so with the backbone frozen there is no scene<->hand circularity.
data:
  data_root: "/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609"
  num_frames: 16
  resolution: [224, 224]
  clip_stride: 8
  val_split: 0.01
  num_workers: 4
model:
  checkpoint: "models/NeoVerse/reconstructor.ckpt"
  warm_start_hand_head: "/work/courses/3dv/team25/checkpoints/default/hand_head_final.pt"
  freeze_backbone: true
  enable_scale_head: false
  enable_hand: true
  hand_head_type: "hamer"
  use_hand_crop: true
  hand_crop_size: 8
  enable_cam: true
  enable_pts: false
  enable_depth: false
  enable_norm: false
  enable_motion: false
  enable_gs: true              # <-- needed: gs_depth is the metric anchor reference
  enable_root_anchor: true     # <-- NEW: own + apply RootDepthRefine
  root_anchor_kwargs:
    hidden: 32
    conf_thresh: 0.1
    band_m: 0.5
  hand_to_gs_injection:
    enabled: false
    use_hand_valid_mask: true
  hamer_head_kwargs:
    dim: 1024
    depth: 6
    heads: 8
    dim_head: 64
    mlp_dim: 1024
    dropout: 0.0
    crop_global_depth: 1
loss_weights:
  kp3d:          0.05
  kp3d_abs:      0.3
  kp2d:          0.05
  global_orient: 0.01
  hand_pose:     0.01
  betas:         0.01
  transl:        1.0
  root_anchor:   1.0          # <-- NEW: gated consistency, corrected wrist z -> gs_depth
  gs_l1:         0.0
  gs_lpips:      0.0
  hand_depth_anchor: 0.0
metric_scale:
  enable: true
  clamp: [0.1, 10.0]
training:
  lr: 1e-4
  min_lr: 1e-7
  epochs: 1
  batch_size: 4
  grad_accum_steps: 8
  use_amp: true
  log_every: 5
  val_every: 10
  val_max_batches: 8
  save_every: 1000000
  grad_clip_norm: 5.0
  kp3d_abs_warmup_steps: 8
  root_anchor_warmup_steps: 8
  freeze_gs_head: true
  keep_last_checkpoints: 0
  seed: 42
  output_dir: "/tmp/rt_contact"
visualization:
  mano_model_folder: "models/MANO"
  num_vis_frames: 4
hand_crop:
  rescale_factor: 1.5
debug:
  enabled: false
  max_sequences: 5
  single_frame: false
wandb:
  enabled: false
  project: "hand-head-training"
  entity: 3DV-Project
  run_name: "P4-CONTACT: scene-depth root anchor (frozen backbone)"
  tags: []
  notes: null
```

- [ ] **Step 2: Parse-check the config**

Run: `python -c "import yaml; yaml.safe_load(open('configs/exp_p4_contact.yaml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add configs/exp_p4_contact.yaml
git commit -m "feat(contact): exp_p4_contact config (gs on, root anchor on, frozen backbone)"
```

- [ ] **Step 4: Push, then run the single-srun train+eval on the cluster**

```bash
git push origin feat/hand-scene-metric-coupling
```

On the cluster login node (inside tmux), pull the new/changed files then run:

```bash
cd /home/dmonopoli/FF-4DGS-Ego && git fetch origin && \
git checkout origin/feat/hand-scene-metric-coupling -- \
  diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py \
  diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py \
  scripts/root_depth_anchor.py scripts/train_hand_head.py scripts/eval_world_space.py \
  configs/exp_p4_contact.yaml

srun --account=3dv --partition=jobs --gpus=gb10:1 --time=01:30:00 bash -c '
set -uo pipefail
export XDG_CACHE_HOME=/tmp/xc TORCH_HOME=/tmp/th HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl
mkdir -p /tmp/xc /tmp/th /tmp/hf /tmp/mpl
cd /home/dmonopoli/FF-4DGS-Ego
source /work/scratch/dmonopoli/venv_gb10/bin/activate
DR=/work/courses/3dv/team25/data/hot3d_aria/preprocessed_pinhole_f609
echo "######## TRAIN p4_contact (max_steps=20) ########"
python -u -m scripts.train_hand_head --config configs/exp_p4_contact.yaml \
  training.output_dir=/tmp/rt_contact training.max_steps=20 \
  debug.enabled=true debug.max_sequences=8 2>&1 \
  | grep --line-buffered -vE "Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
echo "######## build eval config ########"
python - <<PY
import yaml
c = yaml.safe_load(open("configs/exp_p4_contact.yaml"))
c["model"]["warm_start_hand_head"] = "/tmp/rt_contact/hand_head_final.pt"
yaml.safe_dump(c, open("/tmp/eval_contact.yaml", "w")); print("wrote /tmp/eval_contact.yaml")
PY
echo "######## W-EVAL: contact-anchored head ########"
python -u -m scripts.eval_world_space --config /tmp/eval_contact.yaml \
  --data_root "$DR" --max_seqs 4 --max_segs 1 2>&1 \
  | grep --line-buffered -vE "fwd\+lift|Loaded GT joints|Loaded 2D GT|No calibration for|\[VIS\]"
echo "######## DONE ########"
' 2>&1 | tee /tmp/contact_log.txt
```

Expected: training prints `Warm-start: ... 134/134 ...` (hand head) and a non-zero `root_anchor=` count in the Trainable-parameters line; validation prints `WRIST` falling; the eval prints `W_MPJPE` for the contact-anchored head plus `W_MPJPE_re16`. Read: contact `W_MPJPE` vs the supervision-only run and vs the `re16` ceiling.

---

## Self-review

**Spec coverage:** mechanism (c) post-hoc correction (Task 1+2+4+5), reuse of `gs_depth`/project/sample (Task 2), gating as Phase-1 contact proxy (Task 1), detached reference / no circularity (Task 2, `gs_depth.detach()` + frozen backbone in Task 6 config), consistency loss (Task 2/4), `enable_gs` requirement (Task 6 config), eval vs `re16` ceiling (Task 6). Phase 2 (explicit contact signal) is intentionally out of scope per the staged spec.

**Placeholder scan:** none — every step has runnable code/commands and expected output.

**Type consistency:** `RootDepthRefine.forward(wrist_z, d_scene, conf, in_frame) -> (delta_z, gate)` used identically in Task 1 tests and `apply_root_anchor` (Task 2). `apply_root_anchor(module, pred_joints, gs_depth, gs_depth_conf, cam_intr) -> (corrected, delta_z, info)` used identically in train (Task 4) and eval (Task 5). `root_anchor_loss(corrected_wrist_z, d_scene, gate, has_hand)` signature matches its test (Task 2) and call site (Task 4).

## Open dependency

The supervision-only W number (the `exp_p3_wabs` 16-frame run, in flight) sets the bar: if it already reaches the `re16` ceiling, this anchor has little headroom and Phase 2 is moot. Run Task 6 after that number is in, on the same `--max_seqs`/`--max_segs` so the comparison is apples-to-apples.
