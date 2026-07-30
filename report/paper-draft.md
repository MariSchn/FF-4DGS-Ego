# Absolute Egocentric Hand Pose from Frozen Feedforward Reconstruction Features

**Working draft, rewritten 2026-07 around the current thesis.** The previous draft
("Metric Hands, Metric Scene") argued a scene-metric claim that was experimentally
falsified; see the thesis-history note at the bottom. Target: CVPR ~Nov 2026.
Numbers in [brackets] are pending; everything else matches the canonical HOI4D results.
Citations: §References (`novelty-assessment.md`, `related-work-positioning.md`).

---

## Abstract

Estimating where a hand is in 3D, in metric camera coordinates, from a single egocentric
RGB stream is the part of hand pose estimation that crop-based methods structurally get
wrong: weak-perspective regressors recover articulation well but leave the absolute
translation under-constrained, and world-space methods buy absolute scale with SLAM plus a
metric-depth foundation model. [REFRAME IN PROGRESS, 2026-07-15: the original claim that
a frozen scene-reconstruction backbone specially encodes absolute hand depth is FALSIFIED
by our own backbone-swap ablation (frozen DINOv2 21.9mm beats recon 23.6mm; random-init
27.7mm). The paper's contribution is the RECIPE and its controlled analysis, not the
backbone.] We show that a lightweight MANO hand head trained on frozen generic features,
with an absolute-3D keypoint loss (kp3d_abs) and hand-box crop geometry, yields
camera-frame absolute hand pose with no SLAM, no depth foundation model, and no scene
supervision. On a 157-sequence HOI4D test split with dense GT depth, the head reaches
**23.6mm absolute camera-frame MPJPE** with GT-derived boxes (leakage-audited clean split
as headline), versus **83.4mm (WiLoR)** and **88.0mm (HaMeR)** run end-to-end in their
fully native regime (own detector, full-res frames, true-focal conversion) on the same
split. Hand3R reports ~42.6mm on its own, unpublished split; the comparison is cross-split
and cross-protocol and we do not claim to beat it. We further show the result degrades
gracefully under bounding-box noise (31.8mm under jitter, recovered to 27.3mm by
jitter-augmented training) and report an end-to-end detector-box protocol [detbox v2,
eval in flight; detector boxes built, recall 0.829].

---

## 1. Introduction

Egocentric hand-object interaction video needs the hand in *metric camera space*: contact
reasoning, manipulation learning, and AR overlays all break if the hand floats at the wrong
depth. Yet the dominant estimators are crop-based weak-perspective regressors (HaMeR,
WiLoR): excellent articulation (PA-MPJPE of a few mm) but absolute translation recovered
from a crop-scale heuristic, which fails badly in the egocentric near field. The standard
escape is a pipeline: SLAM for camera trajectory plus a metric-depth foundation model for
scale (HaWoR, WHOLE, EgoGrasp), or a metric scene foundation model that the hand is anchored
into (Hand3R).

We take a third route. We ask what actually delivers absolute egocentric hand depth in a
single feedforward model, and answer with controlled ablations: a HaMeR-style MANO head on
FROZEN features (any strong generic backbone suffices; reconstruction pretraining is not
required), trained with an absolute-3D keypoint loss and hand-box crop geometry, recovers
camera-frame hand pose at 21.9-23.6mm on HOI4D. Run end-to-end in their native regime, the
crop-based baselines sit at 83-88mm absolute on the same split.

**Contributions.** [REFRAMED 2026-07-15 after the backbone-swap null; final wording
pending the detbox v2 number and discussion with supervision.]
1. A controlled decomposition of absolute egocentric hand depth: absolute-3D keypoint
   supervision is causally necessary (zeroing it: 725mm), the backbone is NOT the source
   of metric depth (frozen DINOv2 21.9 beats recon 23.6; random-init 27.7), and box crop
   geometry carries a large share of the signal (motivating the honest detector-box
   protocol).
2. A simple single-model recipe: frozen generic backbone + MANO head + kp3d_abs
   supervision, no SLAM, no depth-FM, no scene supervision, evaluated end-to-end with
   detector boxes [detbox v2].
3. An evaluation on dense-GT-depth egocentric data (HOI4D; H2O [in progress]) with
   native-regime end-to-end baselines, box-convention ablations both ways, bbox-robustness
   ablations, and a leakage-audited split.

---

## 2. Related Work

**Feedforward reconstruction backbones.** DUSt3R/MASt3R, VGGT, CUT3R, WorldMirror, NeoVerse
regress geometry (pointmaps, depth, cameras, Gaussians) in a single pass. We use a frozen
NeoVerse-lineage backbone purely as a feature extractor; we do not train, evaluate, or claim
its scene output.

**Crop-based hand pose.** HaMeR-style transformers and WiLoR recover metric MANO articulation
from crops but leave absolute translation weak-perspective-constrained. Run end-to-end in
their fully native regime on our HOI4D split (own detector, full-res frames, true-focal
conversion), their absolute camera-frame error is 88.0 and 83.4mm respectively, the failure
mode we target. (Their published demo conversion uses a dummy focal length and is non-metric
by construction, ~24.5m; forcing them into our 224px crop regime inflates error to
170-240mm, which we report only as a box-convention ablation, not as the comparison.)

**World-space ego hands.** HaWoR (DROID-SLAM + Metric3D on the background, hand masked out),
WHOLE, EgoGrasp assemble absolute scale from external SLAM and depth-FMs. Multi-stage, with
heavy dependencies; we use a single frozen backbone.

**The closest neighbor: Hand3R** (concurrent, 2026) jointly predicts a hand and a dense
metric scene in one feedforward pass on CUT3R, reporting ~42.6mm absolute C-MPJPE on HOI4D.
It gets absolute scale from a metric scene foundation model; we show a hand head alone,
without any metric scene output, is sufficient. We do NOT claim to beat Hand3R: their
split is unpublished, the box conventions differ, and the joint sets may differ (theirs
possibly 21-joint with fingertips vs our smplx-16), so the numbers are in the same
neighborhood but not comparable [footnote in table; a same-protocol run is impossible
until their split/code is released].
HaPTIC is the strongest video baseline we run: root-relative C_rr 25.7 on the native-HD
rerun (28.7 at 224px) / WA 35.3; its absolute number awaits the true-focal conversion
pass [in flight].

---

## 3. Method

### 3.1 Overview
Given a monocular egocentric clip, a frozen feedforward reconstruction backbone produces
per-frame features. A trained hand head consumes hand-region crops of these features plus
global context and regresses MANO parameters and an absolute root translation; a
differentiable MANO layer yields camera-frame metric joints.

### 3.2 Hand head and supervision
A HaMeR-style transformer head predicts the MANO parameter vector and translation. Beyond
the standard 2D/3D keypoint, pose, and shape losses, the head is supervised with an
**absolute-3D keypoint loss (kp3d_abs)** on unaligned camera-frame joints. This is the term
that forces the head to read absolute depth out of the backbone features instead of leaving
translation to a weak-perspective heuristic; removing it is the primary ablation [control
run pending].

### 3.3 Box protocol
Training and the headline evaluation use GT-derived hand bounding boxes. Because this is an
evaluation advantage over detector-driven baselines, we (a) measure degradation under box
perturbation (jitter and fixed boxes), (b) retrain with jitter augmentation, and (c) report
an end-to-end protocol with detector boxes at native resolution [detbox v2, pending].

---

## 4. Experimental Setup

**Data.** HOI4D (dense GT depth), 157-sequence test split, right hand, smplx-16 joints.
Split audit: 5 warm-start-contaminated sequences excluded in the clean-152 headline; a
scene-disjoint-132 variant controls for sibling takes. H2O [pending].

**Metrics.** Absolute camera-frame MPJPE (C-abs), root-relative (C_rr), world-space
(W-MPJPE), PA-MPJPE.

**Baselines.** WiLoR and HaMeR run end-to-end in their fully native regime (own detector,
full-res frames, true-focal conversion) as the primary rows, plus a box-convention
ablation in our crop regime; HaPTIC (video, native-HD); Hand3R (paper numbers, their
split, footnoted as non-comparable); HaWoR (own-SLAM world regime) [build in progress].

---

## 5. Results

### 5.1 Headline table

| Method | C-abs (mm) | C_rr (mm) | Regime / notes |
|---|---|---|---|
| **Ours (winner10ep)** | **23.6** (full-157; clean-152 is the headline) | 17.3 | GT-derived boxes (oracle localization; honest E2E = detbox v2, below) |
| Ours (frozen DINOv2 backbone) | 21.9 | 14.8 | same recipe, generic backbone |
| WiLoR (native E2E) | 83.4 | 27.2 | own detector, full-res, true-focal; 156/157 seqs |
| HaMeR (native E2E) | 88.0 | 30.3 | WiLoR detector (own infeasible, see footnote), native 2.0 rescale |
| HaPTIC | [true-focal pass in flight] | 25.7 (native-HD) | video baseline |
| Hand3R | ~42.6 | - | paper number, their split, NOT comparable (unpublished split, box/joint conventions differ) |

Box-convention ablation (demoted from headline): forcing baselines into our 224px crop
regime gives WiLoR 206.3 det-box / 240.0 GT-box and HaMeR 176.8 det-box / 168.3 GT-box;
GT boxes do not rescue weak-perspective absolute depth.

**Long-window world space (128-frame segments).** All rows below use one scorer, `wa_short` 30,
the same 60-sequence subset and matched segment counts, so they differ only in the predictor and
the camera trajectory. Offline rows share a single DROID/HaWoR trajectory, making this the most
controlled comparison available.

**Protocol.** 16 frames at 224x224 in **both training and inference**; chunks of 16 with an
overlap of 8, predictions averaged on the overlap. W-MPJPE uses a single first-window rigid
alignment (rotation + translation, scale held fixed, never re-solved) that is reused for every
later frame, with no per-chunk re-alignment. WA-MPJPE uses a per-window Procrustes (similarity)
alignment. Long window = 128-frame segments.

> **OURS ROWS PENDING RE-MEASUREMENT AT THE MATCHED 16/8 SETTING (2026-07-30).** Two issues were
> found and fixed. (i) An identity-camera-pose bug: the `gs_anchor_only` fast path returned before
> the rasterizer, which is what publishes `rendered_extrinsics`, so `predict_clip` silently fell
> back to identity `c2w` and the world trajectory had zero camera translation *and* rotation
> (fixed in `9dd474d`). (ii) A train/inference mismatch: the head is trained on 16-frame clips but
> every world eval had been run with `--clip_len 32 --stride 16`, i.e. double the training length
> and out of distribution. The full-157 re-run at the matched 16/8 setting is in flight.
> The baseline rows below are unaffected by both: they never call `predict_clip`, and they are
> external per-frame predictions scored independently of our clip length. All camera-frame
> C-MPJPE numbers are likewise unaffected.

| Row | Regime | W-MPJPE | WA-short | WA-long | C-abs |
|---|---|---|---|---|---|
| Ours (self-chained) | online | [re-measuring at 16/8] | - | - | - |
| Ours + SLAM | offline | [re-measuring at 16/8] | - | - | - |
| HaWoR | offline | 215.6 | 42.0 | 71.7 | 86.9 |
| WiLoR + SLAM | offline | 219.0 | 36.4 | 65.1 | 84.0 |
| HaMeR + SLAM | offline | 227.6 | 37.5 | 66.5 | 89.0 |

A 20-sequence preview at the matched 16/8 setting gives ours W 166.6, WA-short 27.8, WA-long
54.2, C-abs 29.7, against a re-anchor-every-16-frames ceiling of 25.2. Clip length barely moves
W (167.9 at 32 frames) but does move the camera-frame error: C-abs degrades monotonically as
clips lengthen, 29.7 / 30.2 / 30.7 / 32.2 at 16 / 32 / 48 / 64, because the head only ever saw
16 frames. That is the concrete cost of the mismatch and the reason the matched setting is the
one reported.

The shape of the long-window problem is robust to both fixes: every method we can run sits far
from a GT-trajectory oracle, so the error is dominated by the camera trajectory rather than the
hand. What the fixes did overturn is the list of levers. Re-measured on valid data, per-clip and
per-sequence scene scale (1.5%), chunk-linking (1.3%), dense-chaining (negative) and an oracle
GT wrist depth (2%) are all neutral or harmful, and offline SLAM lands on top of our own
trajectory rather than improving it. Reducing the number of chaining seams five-fold buys only
8% of W while leaving WA-long flat, so the drift accumulates per frame rather than per seam. The
single informative comparison is that discarding our predicted camera entirely and chaining on
hand overlap alone *beats* using it, which localizes the remaining error in the frozen camera
head rather than in any post-hoc correction of its output.

### 5.2 Box robustness
winner10ep degrades to 31.8mm under 0.2 jitter and 43.5mm with fixed 0.30 boxes. The
jitter-augmented retrain (TEST60) holds at clean 26.9 / jitter 27.3 / fixed 31.2, so the
sensitivity is trainable away at small cost.

### 5.3 End-to-end with detector boxes
The naive E2E number (140.6mm) was diagnosed as a flawed-input artifact: detection at 224px,
no box squaring, decayed carry-forward fallbacks (det-vs-GT IoU 0.383, 18.8% of frames with
zero overlap). The corrected protocol (HD detection + the exact training box protocol) has
its detector boxes built (157/157 seqs, mean detection recall 0.829); the eval on winner10ep
and the jitter-robust checkpoint is [in flight]. The honest E2E number goes here and is the
number this paper leads with against the native baselines (83-88mm).

### 5.4 Ablations
- **kp3d_abs causal control** (TEST60, all else identical): zeroing only the absolute-3D
  term gives C_abs 725 / C_rr 131. Absolute supervision is causally necessary.
- **Backbone swap** (full-157, same head/recipe/data): recon 23.6/17.3, frozen DINOv2 ViT-L
  21.9/14.8, random-init frozen 27.7/18.3. The backbone is not the source of metric depth;
  the random-init result implicates box crop geometry as a major depth cue, which is why
  the detector-box E2E protocol (5.3) is the honest headline.
- **Seed variance** [3-seed run queued; required to license the backbone-swap ordering].

---

## 6. Limitations

- Headline protocol uses GT-derived boxes; the detector-box E2E number [in flight] is the
  honest headline and box geometry is a known depth cue (see random-init ablation).
- Long-window (128-frame) world accuracy is weak in absolute terms for every method we can
  run, ours included; we claim camera-frame absolute pose and short-window world placement,
  not long-window world accuracy. The comparison protocol is fair (matched trajectory, scorer,
  sequences and segments), but our own rows are being re-measured after the identity-camera-pose
  fix (`9dd474d`) and no long-window "ours" number should be quoted until that lands. The
  short-window world headline predates the offending commit (`d8faf8d`) and is unaffected.
- Hand3R comparison is cross-split, cross-box-convention, and possibly cross-joint-count;
  a same-protocol run is impossible until their split/code is released.
- Single dataset until H2O lands [training in progress, subject-disjoint split].
- HaMeR could not be run with its own detector (dependency-incompatible); it uses the
  WiLoR detector with HaMeR's native rescale.
- H2O column trains the head from scratch (no HOI4D warm start, to avoid cross-dataset
  contamination) and zeroes MANO parameter-space losses (H2O's MANO convention is 44-49mm
  off smplx forward kinematics); keypoint losses are identical to HOI4D.

---

## References (see novelty-assessment.md for the verified sweep)
- [hand3r] *Hand3R*, 2026, arXiv:2602.03200 (CUT3R).
- [hawor] *HaWoR*, CVPR 2025, arXiv:2501.02973.
- [whole] *WHOLE*, 2026, arXiv:2602.22209. [egograsp] *EgoGrasp*, 2026, arXiv:2601.01050.
- [wilor] *WiLoR*. [hamer] *HaMeR*. [haptic] *HaPTIC*.
- [vggt] *VGGT*, arXiv:2503.11651. [worldmirror] *WorldMirror*, arXiv:2510.10726.
- [neoverse] *NeoVerse*, arXiv:2601.00393. [cut3r] *CUT3R*, arXiv:2501.12387.
- [mccho] *MCC-HO*, CVPR 2024, arXiv:2404.06507.

---
**Thesis history.** The previous version of this draft ("Metric Hands, Metric Scene") claimed
that anchoring an up-to-scale feedforward Gaussian scene to the in-scene metric hand makes the
scene metric. That claim was experimentally falsified: the non-circular object-depth eval
(2026-06) showed the anchor makes object-region depth worse, and the scale-source ablation
(2026-07) showed hand-as-global-scene-scale reaches only 0.728 vs oracle 1.022. The 4DGS
backbone is frozen third-party and Gaussian rendering is off; scene reconstruction is not a
contribution lever. The old draft is preserved in git history.
