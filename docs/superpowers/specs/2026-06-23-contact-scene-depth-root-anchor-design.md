# Contact anchor: scene-depth root anchor (Phase 1) + hand-object contact (Phase 2)

**Date:** 2026-06-23
**Status:** approved; mechanism resolved to (c) post-hoc correction
**Branch:** feat/hand-scene-metric-coupling

## Design update (2026-06-23): mechanism is (c) post-hoc, not in-head

Codebase investigation showed the validated metric depth is `gs_depth` (rendered by
the GS head), and the GS head runs **after** the hand head in `worldmirror.forward`.
The pre-hand-head depth head (`preds["depth"]`) is disabled and unsupervised in the
working configs. So the original "feed scene depth into `dec_trans`" (in-head) cannot
see the good depth.

Resolved (approved) to **mechanism (c): a learned, feedforward, post-hoc root-depth
correction**. After the model produces `params` and `gs_depth`, we project the
predicted wrist, sample `gs_depth` there, and a small trained module emits a depth
shift `Δz` applied to the root (rigidly shifting the hand in camera depth). It reuses
the validated 8% `gs_depth`, samples at the true wrist (not a bbox proxy), needs no new
depth objective, and stays end-to-end feedforward (gradients flow, no test-time
optimisation or temporal filtering). `gs_depth` is **detached** as a fixed reference,
which (with a frozen backbone) rules out the circularity risk below. The sections that
follow describe the original in-head framing; the authoritative mechanism is (c) as
captured here and in the implementation plan.

## Problem and goal

The W-MPJPE diagnostic localised the world-space error to the hand-root **absolute
depth**. Scale is unbiased (~0.98 vs GT), articulation is competitive (root-relative
C-MPJPE ~41.6, tied with Hand3R), scene depth is accurate (8% AbsRel). Only the
per-frame root depth-from-camera drifts, and that drift accumulates into the chained
world track (C-abs ~102 mm amplifies to W ~327 mm). The oracle that periodically
re-anchors the root to GT every 16 frames collapses W to ~65 mm (`W_MPJPE_re16`).

The contact anchor is a **feedforward, SLAM-free analog of that oracle**: instead of
GT, it re-anchors the root to the **metric scene** we already reconstruct. The root
depth stops drifting because it is tied to a metric reference at every frame where
that reference is reliable.

Target: drive W toward the `re16` ceiling (~65 mm) **without GT**, end to end, no
test-time optimisation or filtering (Cyrus's constraint).

## Phase 1: scene-depth root anchor (mechanism B, feedforward fusion)

### Core

The GS scene depth at the hand's image location is metric and accurate. Feed it to the
root predictor as an input so the root depth is grounded in that reference rather than
guessed from the crop alone. This injects the information the crop lacks, which is why
it should fix the drift (a loss-only consistency term cannot, since a head that could
infer the depth would not be drifting).

### Data flow

```
hand bbox (already available for cropping)
        │
        ├─► sample GS depth over the bbox region ─► d_scene  (metric, camera-frame z)
        └─► sample gs_depth_conf over same region ─► conf
crop/query token (per hand) ─────────────────────────────┐
d_scene, conf (embedded) ────────────────────────────────┴─► dec_trans ─► root transl residual
```

Sampling at the **bbox region** (median GS depth over the box, not a single pixel)
avoids the chicken-and-egg of needing the predicted wrist before sampling, and is
robust to holes and to the exact pixel. The bbox center is a wrist proxy; median over
the box is the robust estimate of the surface the hand occupies or rests on.

### Where it plugs in

Reuses the `dec_trans` root branch already added (`root_refine`):

- `dec_pose` still predicts the base `pos(3)` translation.
- `dec_trans` is **conditioned on `d_scene`**: input becomes `concat(token, embed(d_scene_norm, conf))`, output is the residual added to `pos(3)`. With `d_scene`
  in the input, the branch can set the root depth to `d_scene` plus a learned offset
  (the canonical-wrist offset and any contact standoff).
- Lateral axes (x, y) stay as before (2D reprojection already pins them); the anchor
  acts on the **depth axis**, the failing one.

### Gating (the Phase-1 contact proxy)

`d_scene` is a good root reference only when the hand is near a surface (in contact),
where scene depth at the hand ≈ surface ≈ hand depth. In free space it is background
and would poison the root. Phase 1 gates with two cheap proxies:

1. `gs_depth_conf` over the bbox region above a threshold.
2. A plausibility gate: only trust `d_scene` when it is within a band of the head's own
   current root-depth estimate (rejects far-background readings).

This works well on manipulation-heavy data (HOI4D, HOT3D) where hands are mostly on
objects. Phase 2 replaces these proxies with a real contact signal.

### Loss

Feeding `d_scene` to the branch is the main mechanism (B). On top, a light gated
consistency term anchors the predicted root depth to the scene:

```
L_anchor = gate * Huber( root_cam_z(pred_joints[0]) , d_scene )
```

applied on the predicted **joint-0 camera-frame z** (not raw transl, to avoid the
canonical offset), warmup-ramped, with a modest weight. This is mechanism A kept as a
supporting signal, not the only lever.

### Reuse / build size

Small. Reuses `dec_trans` (done), depth sampling (`sample_depth_at_joints` for the
pixel-sampling pattern, plus a bbox-region variant), `gs_depth` + `gs_depth_conf`
(already produced), the existing transl/abs3d losses. New work: bbox-region depth
sampler, the `d_scene` embedding into `dec_trans`, the gated consistency loss, a config.

### Requirement / cost note

The anchor needs `gs_depth`, so the contact config must **enable the depth/GS path**
(the `exp_p3_wabs` probe ran with `enable_gs: false` for speed). This adds per-step
cost versus the hand-head-only probes. Budget for it in the run plan.

## Phase 2: hand-object contact (extension, only if a gap to re16 remains)

Replace the Phase-1 `gs_depth_conf` + plausibility proxies with an explicit
**hand-object contact signal**:

- A contact estimate per frame (learned contact head, or geometric proximity between
  the predicted hand surface and the GS scene surface).
- At contact, trust the scene anchor fully; in free space, fall back to the head's own
  root prediction (no anchor).

This is the full "contact" mechanism and the novelty framing. It is a gate/weight swap
on Phase 1, not a rewrite.

## Evaluation

Same `eval_world_space` as the supervision run. On HOT3D and HOI4D:

| condition | what it measures |
|---|---|
| deployed head | baseline W (~327) |
| supervision-only (exp_p3_wabs, 16-frame) | how far heavier supervision alone gets |
| **contact anchor (this design)** | feedforward scene re-anchoring |
| `W_MPJPE_re16` | GT-oracle ceiling (~65), the target |

Success: contact closes the gap toward `re16` beyond what supervision alone does,
without GT.

## Risks and open questions

- **`d_scene` reliability when the hand is occluded or not reconstructed in the GS.**
  Gating is the mitigation; Phase 2's contact signal is the real fix.
- **Frozen-backbone GS depth quality on HOT3D.** Depth was shown good on HOI4D (dense
  GT supervision). HOT3D frozen depth may be weaker, making the anchor noisier. The
  clean Phase-1 test may want HOI4D first.
- **Circularity.** HDGLA anchors scene-depth to the hand (scene-follows-hand); this
  anchors the hand to scene-depth (hand-follows-scene). With a **frozen** backbone the
  GS depth is fixed, so there is no loop. If the backbone is later unfrozen, the two
  must not be active together without care.
- **bbox center vs true wrist.** Mitigated by region-median sampling; revisit if the
  anchor is biased.

## Build order

1. bbox-region GS-depth sampler (`d_scene`, `conf`).
2. Condition `dec_trans` on `d_scene` (embedding + concat).
3. Gated consistency loss `L_anchor`.
4. `exp_p4_contact.yaml` (fork of `exp_p3_wabs`, enable depth/GS path, anchor flag,
   loss weight), frozen backbone.
5. Bounded train + `eval_world_space` W, same single-srun harness as the supervision run.
6. (Phase 2) contact head / geometric contact gate; free-space fallback.
