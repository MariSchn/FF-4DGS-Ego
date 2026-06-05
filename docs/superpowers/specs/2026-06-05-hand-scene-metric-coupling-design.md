# Hand↔Scene Metric Coupling (L1 + L2) — Design Spec

**Date:** 2026-06-05
**Branch:** `feat/hand-scene-metric-coupling`
**Status:** approved, in implementation

## Goal

Make the up-to-scale monocular Gaussian scene agree in **metric** scale with the
metric MANO hand, in the FF-4DGS-Ego (WorldMirror/NeoVerse) pipeline. This is the
poster's stated future work: *"Native 3D alignment in model predictions."*

Two complementary mechanisms:

- **L1 — HDGLA training loss.** A geometric anchor that pulls predicted scene
  depth (`gs_depth`) toward the hand's metric depth at the projected hand joints.
  This is the constraint that actually forces metric scale (attention cannot).
- **L2 — MetricScaleHead inference solve.** A closed-form global scale that makes
  *exported* splats metric without retraining, plus a minimal hand export so the
  aligned hand+scene composite (Figure 5b) is reproducible.

## Approved design decisions

| Decision | Choice |
|---|---|
| Anchor points | **All 16 joints** per hand (over-determined, uses depth spread) |
| Gradient direction | **Scene follows hand** — detach the MANO target (hand is the trusted metric anchor) |
| L2 scale granularity | **Global scalar per sequence** |
| Hand export | **Include** minimal world-frame hand export in `reconstruct_4dgs.py` |

## Locked coordinate conventions (grounded)

- `pred_joints` (train_hand_head.py:1247) = `[B, S, H=2, J=16, 3]`, **camera-frame
  metric metres**; `[..., 2]` is the metric Z.
- Projection mirrors train_hand_head.py:1304-1311 → `project_vertices`
  (hand_vis_utils.py:337-340): `u=(W-1)-row`, `v=col`, `W=1408` (square Aria frame).
- Axis mapping into `gs_depth[..., Hd, Wd]`: **width ← u, height ← v** (derived from
  bbox pipeline train_hand_head.py:454-458 + injection `feats[..., y1:y2, x1:x2]`).
- `gs_depth` = `[B, S, 1, Hd, Wd]`, positive (DPTHead `is_gsdpt`, `output_dim=2`,
  `exp` activation), worldmirror.py:434-440. Exists **only when `enable_gs=true`**.
- `has_hand` (`hv`/`hand_valid`) = `[B, S, H]` in {0,1} (train_hand_head.py:1272-1276).

The shared projection+sampling core is already implemented and grounded in:
`diffsynth/auxiliary_models/worldmirror/models/utils/hand_depth_sampling.py`
(`project_joints_to_norm_pixels`, `sample_depth_at_joints`). **Both units import it.**

## Units & interfaces

### Unit 1 — `scripts/hand_depth_anchor_loss.py` (L1, pure)

```python
def hand_depth_anchor_loss(
    pred_joints,   # [B,S,H,J,3] camera-frame metric
    gs_depth,      # [B,S,1,Hd,Wd]
    has_hand,      # [B,S,H] in {0,1}
    cam_intr,      # [B,3] = [focal, cx, cy]
    *,
    margin: float = 0.02,        # Huber/smooth_l1 beta (metres)
    depth_min: float = 0.01,     # reject sampled depth <= this
    gs_depth_conf=None,          # optional [B,S,1,Hd,Wd]; conf gate
    conf_thresh: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    # returns (loss_scalar, {"hand_depth_residual_m": float, "n_valid": int})
```

Mechanics:
- `grid_xy, z = project_joints_to_norm_pixels(pred_joints, cam_intr)`.
- `sampled, in_frame = sample_depth_at_joints(gs_depth, grid_xy)`.
- valid = `has_hand[...,None broadcast over J] ∧ in_frame ∧ (sampled > depth_min)`
  (∧ conf gate if provided).
- `loss = smooth_l1(sampled, z.detach(), beta=margin)` masked-mean over valid.
- If `n_valid == 0` → return `0.0` loss (no NaN), `n_valid=0`.
- Metric: `mean(|sampled - z.detach()|)` over valid, reported in metres.

### Unit 2 — `diffsynth/.../models/heads/metric_scale_head.py` (L2, pure)

```python
def solve_metric_scale(
    pred_joints, gs_depth, has_hand, cam_intr,
    *, clamp=(0.1, 10.0), depth_min=0.01, gs_depth_conf=None, conf_thresh=0.0,
) -> torch.Tensor:           # scalar tensor s (one global value)

def apply_metric_scale(preds: dict, s: torch.Tensor) -> dict:
    # scales preds['gs_depth'] and preds['camera_poses'][...,:3,3] by s; returns preds
```

Mechanics:
- Project + sample as in Unit 1; `ratio = z / sampled` over valid samples.
- `s = clamp(median(ratio), *clamp)`. Global per call.
- `apply_metric_scale` multiplies `gs_depth` and camera translation by `s` so the
  rasterizer's internal scale ratio (rasterization.py:529-537) collapses to ~1.

### Unit 3 — wiring `scripts/train_hand_head.py` (surgical)

- Instantiate near criteria (~:1060); read `cfg["loss_weights"].get("hand_depth_anchor", 0.0)`.
- After `pred_joints` (:1247) compute the loss with `gs_depth = preds.get("gs_depth")`
  (only when `model.enable_gs`); **linear warm-up ramp** of the weight over
  `hand_depth_anchor.warmup_steps` using the global step.
- Add `w_ramp * loss` to the sum at :1324-1334; log residual to `accum_terms`,
  `val_terms`, and W&B; mirror into `run_validation` (~:902-910).
- Guard: if `gs_depth is None` (GS off) the term is a no-op zero.

### Unit 4 — wiring `diffsynth/.../models/worldmirror.py` (surgical)

- Between `preds["gs_depth"]=...` (:440) and `gs_renderer.render` (:460): when
  `is_inference` and `metric_scale.enable` and hands available, `s = solve_metric_scale(...)`;
  `apply_metric_scale(preds, s)`; store `preds["metric_scale"]=s`.
- Requires hand joints + `cam_intr` available at this point — **verification item**:
  confirm `preds["hand_joints"]` and intrinsics are accessible here; if not, thread
  them through or compute scale in `reconstruct_4dgs.py` instead (acceptable fallback).

### Unit 5 — wiring `scripts/reconstruct_4dgs.py` (hand export)

- Apply/log `predictions.get("metric_scale")` to `camera_params.json`.
- Decode `predictions["hand_joints"]` → MANO joints/vertices (camera frame) via the
  `mano_model` wrapper; compose with predicted `cam2world` per frame; scale by `s`;
  save world-frame hand (`hand_world.npz`: per-frame verts/joints + scale) alongside
  scene splats so an aligned hand+scene composite can be rendered.

### Config — `configs/ablation_hand_to_gs_injection.yaml` (`enable_gs: true`)

```yaml
loss_weights:
  hand_depth_anchor: 1.0
hand_depth_anchor:
  margin: 0.02
  warmup_steps: 800
  depth_min: 0.01
  conf_thresh: 0.0
metric_scale:
  enable: true
  clamp: [0.1, 10.0]
```

## Testing (CPU, synthetic tensors — runnable here)

`tests/test_hand_depth_sampling.py`, `tests/test_hand_depth_anchor_loss.py`,
`tests/test_metric_scale_head.py`:
- Sampling: a joint at a known normalised location samples the expected cell;
  **same result at Hd=224 and Hd=448** (resolution independence).
- L1: `gs_depth ≡ z` → loss ≈ 0; known offset → exact residual; `has_hand=0` → 0
  loss & `n_valid=0`; out-of-frame joints masked.
- L2: `gs_depth = z / k` → recovered `s ≈ k`; clamp respected; median rejects a
  minority of outliers; `apply_metric_scale` drives the residual → 0.
- Gradient flows to `gs_depth`, **not** to `pred_joints` (target detached).

No GPU/training run here. Training is run on the cluster with the ablation config.

## Verification items (must confirm against real code/data, not assume)

1. `imgs`/`gs_depth` are a **full-frame** resize/center-crop of the square 1408
   frame (square→square center-crop preserves the frame); if an off-center crop is
   ever used, the sampler needs the crop transform.
2. The `gs_depth` axis mapping (width←u, height←v) holds end-to-end (bbox→crop→DPT).
3. `preds["hand_joints"]` + intrinsics are reachable at the worldmirror.py:440 site
   for L2 (else fall back to applying scale in `reconstruct_4dgs.py`).
4. Joint index 0 is the metric-translation-bearing root in this MANO wrapper.

## Delegation plan

- **Phase 1 (parallel subagents):** Unit 1 (+tests), Unit 2 (+tests). New files only,
  both import the existing shared helper — no shared-file conflicts.
- **Phase 2 (single writer):** verification items + Units 3-5 wiring + config. Touches
  shared files; one agent keeps edits coherent. Parent reconciles + runs CPU tests.

## Risks (from adversarial review)

- Gauge: L1 is gauge-clean only under `gsdepth+gtcamera` (fixed GT cameras) — verify
  the training branch. At inference (`predcamera`) scale drifts → L2 is required.
- Cold start: ramp anchor weight; Huber margin softens noisy early targets.
- Photometric vs anchor: only conflicts with `enable_gs=true`; keep weight modest,
  monitor `gs_l1`/`gs_lpips` don't regress.
- Single global scale ignores per-frame trajectory drift on long clips (S≤120).
- Occlusion: wrist may sample hand surface; conf + depth-validity gating mitigates.
