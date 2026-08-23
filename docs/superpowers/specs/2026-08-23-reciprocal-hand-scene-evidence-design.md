# Reciprocal Hand-Scene Evidence Design

**Date:** 2026-08-23

**Status:** Proposed for implementation

## Goal

Establish, with target-blind and causally controlled experiments, whether a unified world-space
hand and Gaussian representation supports both of these directional claims:

1. predicted hand geometry improves held-out scene rendering;
2. predicted scene geometry and camera motion improve world-space hand placement.

Only if both gates pass may the paper summarize the result as reciprocal or mutual improvement.
The paper must otherwise state only the direction supported by the measurements.

## Non-goals

- Feature injection is not revived without new evidence. Its fixed-protocol ON/OFF comparison is
  null.
- The MANO-surface Chamfer loss is not used as evidence of hand-to-scene coupling. Its rendering
  gain is reproduced by geometrically incorrect targets and disappears on the longer schedule.
- A DINO-versus-reconstruction backbone comparison is not treated as a causal scene-to-hand
  intervention. It changes pretraining and representation together.
- Input-view, timestamp-matched rendering is not used as reconstruction evidence.
- Results with train/test sequence overlap or target-frame leakage are not reported as held out.

## Evidence contract

Every headline result must record the resolved configuration, checkpoint hash, git commit,
semantic-code hash, dataset manifest hash, target indices, camera source, box source, hand and
joint set, segment length, temporal stride, gauge, and number of sequence clusters. Evaluation
scripts must abort when required provenance fields are absent.

All confidence intervals are paired bootstraps clustered by sequence. Frame-level bootstrap is
forbidden. Oracle inputs are permitted only in explicitly named upper-bound arms and never in the
proposed method.

## Gate 0: target blindness

### Root cause to remove

`WorldMirror.forward` currently applies the global visual transformer to every image in `views`
before `_gen_all_preds` runs the hand head. Because global attention spans the sequence, hand
predictions and `enhanced_crop_tokens` for nominal context frames can depend on held-out target
RGB. `prepare_contexts` protects the Gaussian head by recomputing context tokens, but it does not
retroactively protect the earlier hand forward or the injection hook.

### Required data flow

The model forward receives context images only. It produces context MANO predictions, context hand
tokens, cameras, depth, and scene Gaussians. Target camera matrices are supplied later to the
rasterizer and are never passed through the visual transformer. Target RGB, target MANO, target
silhouettes, and target sensor depth exist only in the scorer.

Target MANO for the predicted insertion arm is interpolated from the nearest valid context MANO
predictions. Inserted color is transported by MANO vertex correspondence from a context image.
No color, opacity, feature, or geometry is sampled from the target image.

### Mandatory invariance test

For the same context clip, evaluate three target payloads: original RGB, all zeros, and seeded
random noise. The following must be bit-identical before rendering and numerically identical after
rendering within a declared rasterizer tolerance:

- context MANO parameters;
- interpolated target MANO parameters;
- context `enhanced_crop_tokens` when injection is enabled;
- scene Gaussian attributes;
- inserted predicted hand Gaussians;
- target render for arm P.

The test must fail on the current leaky all-frame implementation before the production change is
made. A unit test that mocks the backbone is insufficient; the gate requires an end-to-end model
forward on a real clip or the smallest real serialized fixture that exercises sequence attention.

## Gate 1: hand improves scene

### Population

Run on held-out HOI4D and H2O sequences excluded from all hand-head, Gaussian-head, and correction-
head training. Freeze split manifests before evaluation. Use interior target frames with valid
context neighbors and report invalid/interpolation exclusions.

### Arms

- **A:** static-fused scene Gaussians only.
- **P:** A plus `G_hand` from target-blind, context-predicted MANO.
- **G:** A plus target GT MANO, with colors still transported from context. Oracle only.
- **C:** A plus the same number and mean color of hand Gaussians collapsed to the predicted hand
  centroid. This controls placement without shape.
- **N1:** P shifted by 1 cm in a seeded isotropic direction. This measures tolerance and is not
  expected necessarily to erase the effect.
- **N3:** P shifted by 3 cm in a seeded isotropic direction.
- **Z10:** P shifted by 10 cm along camera depth.

All arms use identical scene Gaussians, target cameras, rasterizer settings, Gaussian counts where
the control definition permits, and context-derived appearance.

### Metrics

Primary metrics are target hand-silhouette RGB MAE, target hand-silhouette sensor-depth MAE, and
occlusion-boundary error. Hand-box PSNR/LPIPS and full-frame PSNR/SSIM/LPIPS are secondary. Alpha
coverage is always reported so deletion or black-render artifacts cannot masquerade as gains.

### Pass condition

P must improve over A on both unseen datasets with a sequence-bootstrap interval excluding zero on
at least one primary metric, with the other primary metrics directionally consistent. P must be
closer to G than A is. C must be null or worse than P, and N3 and Z10 must be worse than P. N1 is a
tolerance measurement, not a required failure. The result must not depend on fewer than half the
valid sequence clusters.

If only a full-frame image metric improves, or if C/Z10 reproduce P, the geometric hand-to-scene
claim fails.

## Gate 2: complementarity to AnySplat

Run the same frozen manifests, target cameras, context-only hand predictions, inserted hand sets,
and scorer on AnySplat. Convert `G_hand` into AnySplat's rendered Gaussian convention without
changing its scene prediction. Report AnySplat A, P, G, C, and Z10 at minimum.

The gate passes if predicted insertion improves a primary hand-region metric over plain AnySplat
on both datasets and the C/Z10 controls lose the improvement. The absolute effect may differ from
NeoVerse, but its sign and geometric specificity must survive. A positive result only on the weak
NeoVerse reconstruction is insufficient for a complementarity claim.

Gate 2 starts only after Gate 1 passes. If Gate 1 fails, no AnySplat compute is spent on insertion.

## Gate 3: scene improves hand

### Intervention

Add a geometry-conditioned residual head that corrects only the MANO root translation in its first
version. The baseline MANO prediction remains unchanged. The residual head receives:

- the baseline root translation and confidence;
- a stop-gradient set of reconstructed 3D scene points sampled in an annulus around each hand box;
- scene-point confidence and local surface normals where available;
- the stop-gradient reconstructed camera trajectory expressed relative to the center context
  frame;
- hand-box geometry and validity, which every parameter-matched arm also receives.

Sampling the annulus avoids treating the hand's own occluding pixels as background surface. A small
set encoder followed by a residual MLP keeps the intervention explicit: the only treatment-specific
information is reconstructed geometry and camera motion. The scene and baseline hand networks are
frozen for this causal experiment.

### Training and test split

Train the residual module only on stores that exclude HOI4D and H2O. Use the same metric 3D hand
supervision, optimizer, schedule, boxes, seeds, and parameter count in every learned control. Test
zero-shot on the same frozen HOI4D and H2O manifests as Gates 1 and 2.

### Arms

- **H0:** frozen baseline MANO head, no residual module.
- **Hcap:** parameter-matched residual module with hand prediction, boxes, and confidence, but no
  scene input. This controls extra capacity.
- **Hscene:** correct scene points and correct predicted camera trajectory.
- **Hdepth:** correct scene points with camera trajectory removed.
- **Hcamera:** correct camera trajectory with scene points removed.
- **Hshuffle:** the full scene descriptor and trajectory from another sequence in the minibatch.
- **Hconst:** constant median depth and identity relative camera motion through the same module.
- **Horacle:** sensor depth and GT camera trajectory. Oracle headroom only.

Hcap, Hscene, Hdepth, Hcamera, Hshuffle, and Hconst must have the same trainable architecture and
parameter count. Shuffling is deterministic and never pairs a sample with its own sequence.

### Metrics and mechanism checks

Primary metrics are absolute camera-frame MPJPE, root-depth error, W-MPJPE, and WA-MPJPE on a
locked world protocol. Root-relative MPJPE is a control and may remain unchanged. Contact-frame
penetration and hand-to-surface distance are reported separately from free-space frames.

The module must also report sensitivity to its inputs. Replacing scene geometry by shuffled or
constant geometry at inference must change the predicted residual. Improvement should correlate
across sequences with scene/camera accuracy. A module whose output is invariant to scene corruption
has not established scene-to-hand causality.

### Pass condition

Hscene must beat both H0 and Hcap on both unseen datasets in a world-placement primary metric.
Hscene must also beat Hshuffle and Hconst with a sequence-bootstrap interval excluding zero. At
least one of Hdepth or Hcamera must explain a reproducible portion of the full gain, and Horacle must
show non-negative remaining headroom. A gain shared by Hcap, Hshuffle, and Hscene is attributed to
capacity or regularization and fails the causal gate.

## Gate 4: reciprocal system

Evaluate a 2-by-2 decomposition using the same checkpoint family and manifests:

| Configuration | Hand-to-scene insertion | Scene-to-hand conditioner |
|---|---:|---:|
| Independent | no | no |
| Composition only | yes | no |
| Geometry-conditioned hand | no | yes |
| Reciprocal | yes | yes |

The reciprocal row must retain the Gate 1 scene gain and Gate 3 hand gain simultaneously. This
table, rather than a feature-injection ON/OFF table, is the evidence for the phrase "improve each
other."

## Gate 5: locked world-space comparability

Create one canonical protocol manifest and one scorer contract for every in-house and external
method. Each comparable row must match:

- named dataset sequences and exact frame ranges;
- detector-box or GT-box source;
- left/right/both hand set and missing-hand rule;
- joint regressor and joint count;
- 16-frame model clips, stride 8, and 128-frame evaluation segments;
- camera convention and camera source;
- W alignment: one first-window rigid transform, scale fixed, reused for later frames;
- WA alignment: per-window similarity with the same window length;
- millimeter conversion and validity masks.

The scorer rejects metadata that differs from the locked manifest. Rows that cannot satisfy the
contract are placed in a clearly separated non-comparable table and never used for a best-method
claim. Before scoring real predictions, synthetic tests must verify invariance to a constant world
gauge, sensitivity to scale drift, missing-hand handling, and segment-boundary behavior.

The existing mix5 JSONs marked `matches_locked_protocol: false` cannot support a world-space SOTA
sentence until regenerated under this gate.

## Paper update policy

The abstract, title, introduction, method diagram, and contribution list are updated only after the
experimental gates close. No placeholder result becomes declarative prose.

If Gates 1, 2, and 3 pass, the core sentence is:

> Predicted hand geometry improves target-blind novel-view reconstruction, while reconstructed
> scene geometry and camera motion improve zero-shot world-space hand placement.

If only Gates 1 and 2 pass, the paper is reframed around compositional hand-aware reconstruction
and must not claim reciprocal improvement. If only Gate 3 passes, it is a scene-conditioned
world-space hand-tracking paper. If neither direction passes, the paper reports unification and
controlled negative results without a mutual-improvement claim, or is redirected to a different
thesis.

The title must not contain "4D" unless motion is enabled and evaluated. Feature injection and
hand-guided scale calibration are removed unless later evidence independently restores them.

## Execution order and resource policy

1. Cancel obsolete pending feature-injection/Chamfer jobs.
2. Implement and pass the target-payload invariance regression.
3. Run Gate 1 on a small smoke population, then the full HOI4D and H2O manifests.
4. Run Gate 2 only if Gate 1 passes.
5. Implement the Gate 3 controls test-first, smoke them, then run matched seeds sequentially.
6. Lock and validate the world scorer before interpreting Gate 3 world metrics.
7. Run the reciprocal 2-by-2 table.
8. Update the paper and figures from the final evidence ledger.

Only one GPU job may be runnable at a time. Every chain uses `afterok` only when a child consumes a
parent artifact; independent evaluations use explicit sequencing. A failed producer must not allow
an artifact consumer to start. Scratch quota and file quota are checked before every submission,
and large per-clip caches are consolidated or streamed rather than emitted as millions of files.

## Expected code boundaries

- `scripts/probe_insertion_48.py`: becomes dataset-agnostic and strictly context-only, or delegates
  to a new focused insertion evaluator.
- `scripts/metric_views.py`: separates model context views from target render-camera views.
- `scripts/run_anysplat_heldout.py`: exposes the same insertion and scorer contract for AnySplat.
- `diffsynth/auxiliary_models/worldmirror/models/heads/`: contains the scene-geometry conditioner;
  it does not reuse the null hand-to-GS feature injection.
- `scripts/train_hand_head.py`: trains the frozen-input residual arms with explicit control modes.
- `scripts/eval_world_space.py` and `scripts/world_space_metrics.py`: enforce the locked protocol
  manifest and metadata validation.
- `tests/`: contains the real-forward target-blindness regression, scene-corruption sensitivity,
  protocol/gauge tests, and AnySplat composition parity tests.
- `report/`: receives generated tables and claim patches only after gate outcomes are known.

## Stop conditions

Stop and reassess the architecture if three attempted fixes fail to make the real-forward
target-blindness test pass, or if the scene conditioner improves equally under correct and shuffled
geometry after two verified seeds. Do not spend additional seeds to turn a mechanism-null result
into significance.
