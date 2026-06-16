# Metric Hands, ~~Metric Scene~~: Feedforward Egocentric Gaussian Reconstruction Anchored to the Hand

> **⚠️ MAJOR REFRAME (2026-06-17) — read `report/overnight-findings.md` first.**
> The **"metric scene"** half of this draft is **FALSIFIED**. The non-circular B2 experiment (GT object
> depth, hand region excluded) shows the hand anchor makes object-region depth **WORSE** (a01 134.7cm vs
> baseline 61.9cm), and the earlier "scene becomes metric" signal was a hand-region **circular artifact**
> (non-circular scale CV ~13–28% for both). Root cause is fundamental: the frozen monocular backbone is
> ~62cm-inaccurate on objects, so no anchor weight can make the scene metric.
> **Surviving contribution = metric HAND PLACEMENT** (−35% MPJPE, 4.5cm hand depth) + recoverable global
> scale. The scene render IS preserved at low anchor weight (a01 32.81 vs baseline 32.55 dB). Below, all
> "metric scene"/object-metric claims must be cut; keep them only as a reported negative/limitation.
> Sections §1–§5 still need rewriting around hand placement; the money table (§5.1) is updated.

**Working draft — reframed 2026-06-17.** Target: revised down toward **workshop / arXiv** unless the external
placement comparison (HaWoR/Hand3R) is clearly SOTA-competitive. Citations: §References (`novelty-assessment.md`).

---

## Abstract

Monocular feedforward 3D Gaussian Splatting (3DGS) reconstructs a scene in a single pass, but only **up to an
unknown scale** — the depth is geometrically consistent yet not metric. In egocentric hand–object video, however,
a *metric* signal is already present in every frame: the user's hand, whose absolute size and articulation are
recovered reliably by parametric (MANO) hand estimators. We exploit this and propose a feedforward method that makes
both the hand **placement** and the surrounding **scene** metric in one pass, by anchoring the up-to-scale Gaussian
scene to the in-scene metric hand — with **no metric-depth foundation model and no multi-view**. Concretely, a frozen
feedforward GS backbone predicts up-to-scale Gaussians; a hand head predicts metric MANO joints; and a Hand-Depth
Geometric Loss Anchor (HDGLA) pulls the predicted scene depth toward the metric hand depth at the projected joints,
propagating metric scale from the hand into the scene through a trainable hand-to-scene injection. On HOT3D (Aria),
our coupling improves absolute hand placement by **[−35%] (52.9 vs 81.4 mm MPJPE)** over a strong same-data baseline
while preserving articulation (**PA-MPJPE ≈ [7.9] mm**), and — measured against ground-truth object geometry in
**non-hand** regions — reduces scene-depth error from **[≈25 cm to ≈10 cm]** and scale variability (CV) from
**[25.5% to 5.9%]**. A generic metric-depth foundation model (UniDepth) is **[>2×]** less accurate at the hand than
our anchor, showing the hand is a cheaper and more accurate metric source in this setting. The contribution is the
**direction** — a trusted-metric hand rescaling an up-to-scale feedforward GS scene — which is open relative to
concurrent work that anchors the hand *into* a metric-scene foundation model.

---

## 1. Introduction

Feedforward 3DGS backbones (VGGT/DUSt3R-MASt3R lineage) have made single-pass 3D reconstruction from a handful of
images practical, but monocular feedforward reconstruction is fundamentally **scale-ambiguous**: the predicted
Gaussians are correct up to a global similarity, so distances are not in meters. Recovering metric scale usually
requires one of: a metric-depth foundation model, a calibrated multi-view/stereo baseline, IMU/SLAM, or a known-size
object placed in the scene. Each adds a dependency or an assumption that does not hold for casual monocular
egocentric capture.

We observe that egocentric hand–object interaction video carries its own metric ruler: **the hand**. Parametric hand
estimators recover metric MANO meshes whose absolute scale and articulation are reliable (low PA-MPJPE), even when
the absolute *placement* (translation/depth) is uncertain. If the up-to-scale scene is forced to agree with the
metric hand where they overlap, the hand's known scale propagates into the scene — making the whole reconstruction
metric without any external metric prior.

This direction — **hand-anchors-scene** — is the opposite of the most closely related concurrent work (Hand3R
[hand3r]), which anchors the hand *into* a scene whose scale comes from a foundation model, and is distinct from
egocentric world-frame hand methods (HaWoR [hawor]) that mask the hand out and reconstruct no scene. To our
knowledge, a trusted-metric in-scene hand rescaling an up-to-scale **feedforward GS** scene in egocentric video is
open (§2, [novelty]).

**Contributions.**
1. A feedforward, single-pass method that recovers a **metric-scale** Gaussian scene from monocular egocentric RGB by
   anchoring scale to the in-scene metric hand — no metric-depth FM, no multi-view.
2. The Hand-Depth Geometric Loss Anchor (HDGLA) and a trainable hand-to-scene injection that propagate metric scale
   from the hand into the scene, with a `scene_follows_hand` gradient direction we justify and ablate.
3. A **non-circular** evaluation of scene-metricity using ground-truth object geometry in non-hand regions, plus a
   metric-depth-FM baseline and a causal ablation — moving beyond hand-only MPJPE.
4. State-of-the-art-competitive absolute hand placement on HOT3D and a metric scene at no cost to articulation or
   rendering quality.

---

## 2. Related Work

**Feedforward / generalizable 3DGS.** DUSt3R/MASt3R, VGGT, NoPoSplat, AnySplat and Splat-SAP predict geometry or
Gaussians in a single pass [anysplat, splatsap]. They obtain scale from intrinsics, a stereo baseline, or post-hoc
alignment — **never from an in-scene metric anchor**. Our backbone is in this lineage (NeoVerse/WorldMirror) and is
frozen; we add the metric coupling around it.

**Metric scale for monocular reconstruction.** Metric-depth foundation models (UniDepth [unidepth], Metric3D)
regress absolute depth directly; object-supplemented bundle adjustment [frost] anchors SLAM scale to a known object
size. These are either an extra heavy model or a classical, non-feedforward, non-GS pipeline. We show (§5, E1) a
foundation model is less accurate at the hand than our anchor in this setting.

**Hand pose and hand-aware reconstruction.** HaMeR-style transformers regress metric MANO from RGB. MCC-HO [mccho]
uses the hand to disambiguate a **held object** (object-only, no scene, hand-relative). HaWoR [hawor] recovers
world-frame hand motion using DROID-SLAM + Metric3D on the **background** — the inverse of ours: it masks the hand
out and reconstructs no scene. Interaction-aware 4D-GS methods [iags] couple hands and scene but are per-scene
optimization with calibrated multi-view, not feedforward and not metric-from-monocular.

**The closest neighbor — Hand3R [hand3r]** (concurrent, 2026) jointly predicts a hand and a dense metric-scale scene
in a single feedforward pass on CUT3R point maps. We differ on two axes that preserve our contribution: (i)
**representation** — 3D Gaussians vs point maps; and, more importantly, (ii) **coupling direction** — Hand3R anchors
the hand *into* a metric scene whose scale is supplied by the scene model, whereas we let the trusted-metric hand
**rescale** the up-to-scale scene. Hand3R also evaluates on DexYCB/HOI4D and does not report scene-metric accuracy;
we evaluate scene-metricity directly on HOT3D/Aria. We treat Hand3R as concurrent and cite-and-contrast.

The area (egocentric hand+scene metric reconstruction) is crowding rapidly in 2025–2026 [novelty]; we foreground the
**direction** as the durable differentiator and re-verify novelty immediately before submission.

---

## 3. Method

### 3.1 Overview
Given a short monocular egocentric clip, a frozen feedforward GS backbone $f_\theta$ predicts up-to-scale Gaussians
and a per-pixel scene depth $D_\text{gs}$. A hand head $h_\phi$ predicts metric MANO parameters for both hands, from
which we obtain camera-frame metric joints $J$. A trainable **hand-to-scene injection** conditions the Gaussian
prediction on the hand, and the **HDGLA** loss ties $D_\text{gs}$ to the metric hand depth at the projected joints.
Only the hand head and the injection are trained; the backbone stays frozen, so we never trade away the backbone's
geometry — we only rescale and place it.

### 3.2 Hand head
A HaMeR-style transformer head predicts the 64-D two-hand MANO parameter vector; a differentiable MANO layer maps it
to camera-frame joints $J \in \mathbb{R}^{S\times2\times16\times3}$. The head is supervised by standard 2D/3D
keypoint, pose, shape and translation losses, plus an **absolute-3D placement** term ($\text{kp3d\_abs}$) that
supervises the global translation that monocular hand heads otherwise leave under-constrained.

### 3.3 Hand-Depth Geometric Loss Anchor (HDGLA)
Project the detached metric joints to normalized pixel coordinates and bilinearly sample the scene depth there,
$\tilde d = \text{sample}(D_\text{gs}, \pi(J))$. HDGLA is a smooth-$L_1$ between the sampled scene depth and the
metric hand depth $z(J)$:
$$\mathcal{L}_\text{anchor} = \text{Huber}_\beta\big(\tilde d,\; z(J)\big)$$
over valid (visible, in-frame, positive-depth) joints. The **direction** is `scene_follows_hand`: the 2D sampling
location is always the detached hand projection (a stable grid), and the gradient flows **only into $D_\text{gs}$** —
the trusted metric hand is never perturbed by this term. We compared the three directions and found
`scene_follows_hand` clearly best (**[89.4 vs 172 mm]** vs `hand_follows_scene`, which diverges; §5), consistent with
the premise that the hand's metric scale is more reliable than monocular scene depth. A warmup defers the anchor
until the hand head has stabilized.

### 3.4 Hand-to-scene injection and metric scale head
Because the backbone is frozen, HDGLA's gradient is absorbed by a small **trainable injection** that conditions the
Gaussian/depth prediction on the (valid-masked) hand — this is the mechanism by which the hand reshapes the scene
depth, including, we test, in non-hand regions (§5, E2). A closed-form metric-scale head solves the single global
scale $s=\text{median}(z(J)/\tilde d)$ used to read out metric depth and to evaluate scale stability.

### 3.5 Training
We warm-start from a converged hand head and fine-tune at a gentle learning rate, freezing the backbone, on
HOT3D/Aria clips (pinhole-rectified, $224\times224$). This isolates the contribution of the metric coupling on top
of a strong placement baseline ("50 mm → 50 mm + coupling").

---

## 4. Experimental Setup

**Datasets.** HOT3D (Aria egocentric hand–object), pinhole-rectified. [HOI4D port planned for cross-dataset
comparison with Hand3R (E5).]

**Metrics.** Hand: MPJPE / PA-MPJPE (mm), even-sampled across sequences for robustness (absolute MPJPE is
split-dependent, so we only compare same-split deltas). Scene-metric: scene-depth error vs GT object geometry in
**non-hand** regions (cm), fraction within 10 cm, and the global-scale coefficient of variation (CV). Scene quality:
GS PSNR/SSIM/LPIPS from re-rendered Gaussians.

**Baselines.** (i) a strong same-data hand head **without** the anchor; (ii) a metric-depth foundation model
(UniDepth-v2) sampled at the hand; (iii) [HaWoR on HOT3D]; (iv) [Hand3R on HOI4D].

---

## 5. Experiments

### 5.1 Headline table (the "money" table)

**Updated 2026-06-17 with real numbers.** ✅ supports the (reframed) hand-placement claim; ❌ falsified.

| Claim | Metric | Baseline | Ours | Status |
|---|---|---|---|---|
| **Hand placement** ✅ | MPJPE ↓ (9-seq) | 81.4 mm | **52.9 mm (−35%)** | ✅ confirmed (anchor 1.0) |
| Articulation preserved ✅ | PA-MPJPE | ≈7.5 mm | ≈7.9 mm | ✅ confirmed |
| Coupling direction ✅ | MPJPE ↓ | 172 mm (hand←scene) | 89.4 mm (scene←hand) | ✅ confirmed |
| Hand metric @ hand ✅ | depth residual | — | 4.5 cm | ✅ confirmed |
| Scene render preserved (low anchor) ✅ | PSNR | 32.55 dB | 32.81 dB (a01) | ✅ (anchor 1.0 → 26.78, −7.2 dB) |
| **Scene metric on OBJECTS** ❌ | depth err median (non-circ) | **61.9 cm** | **134.7 cm** | ❌ **FALSIFIED** — anchor distorts scene depth |
| Scale stability on objects ❌ | scale CV (non-circular) | 27.6% | 12.8–27.6% | ❌ the 25→6% "win" was a hand-region circular artifact |
| **Causal** (abs-3D) | control MPJPE | — | stays high? | ⏳ E0 running (99395) |
| vs metric-depth FM | UniDepth err @ hand | [>10] cm? | 4.5 cm | ⏳ E1 (torch cu128 fixed; re-test on 5060ti) |

### 5.2 Hand placement (confirmed)
Warm-starting a converged hand head and adding the metric coupling reduces robust 9-sequence MPJPE from **81.4 → 52.9
mm (−35%)** while PA-MPJPE stays at **≈7.9 mm** (articulation parity). Our model generalizes across splits (53→53 mm)
where the baseline degrades (61→81 mm), i.e. the gain is not split luck.

### 5.3 Causal ablation (E0, pending)
We zero the absolute-3D and anchor terms (keeping the warm-start + fine-tuning) to test whether the placement gain is
caused by the coupling or merely by more fine-tuning. **Claim it if** the control stays ≥[60] mm (1-seq) / ≥[75] mm
(9-seq); otherwise we reframe the contribution toward the metric scene (E2) rather than placement.

### 5.4 Versus a metric-depth foundation model (E1, pending)
We run UniDepth-v2 (ViT-L) on the same frames and sample its metric depth at the projected hand joints, comparing its
hand-depth error to our anchor's 4.5 cm. **Claim it if** UniDepth's error is **[>10 cm] (≥2×)**: the in-scene hand is
the cheaper, more accurate metric source. *(Extract + eval implemented; UniDepth weights cached.)*

### 5.5 Non-circular scene-metricity on objects (E2, pending)
The at-hand scene-metric result (§5.1) is semi-circular (the anchor trains at the hand). To prove the **scene** is
metric, we render GT object depth from HOT3D object meshes + per-frame 6-DoF poses and compare the predicted scene
depth in **object** regions (hand region excluded), after fixing scale from the hand alone. Our GT-depth renderer is
validated: object-surface depth sampled at hand–object contact agrees with the independent metric hand depth to
**1.6–4.2 cm**. **Claim it if** the anchored object-region error is clearly below the no-anchor baseline (e.g.
[≈10 cm] vs [≈25 cm]; δ<10 cm up ≥15 pts).

### 5.6 Scene quality (E3, pending)
Because the backbone is frozen and only the injection changes the Gaussians, we verify the metric coupling does not
degrade rendering: GS PSNR/SSIM/LPIPS within [≈0.5 dB] of the baseline.

### 5.7 External baselines and cross-dataset (E4/E5/E6, planned)
[HaWoR on HOT3D]; [HOI4D port for head-to-head with Hand3R]; results on ≥2 datasets.

---

## 6. Discussion and Limitations

- **Frozen backbone** means we rescale/place rather than re-predict geometry — a strength (no quality regression) and
  a limit (we cannot fix gross backbone errors). The injection's reach into non-hand regions is exactly what E2 tests.
- **Absolute MPJPE is split-dependent**; we report same-split deltas only.
- **Single GPU / serial** evaluation throughput; the cross-dataset HOI4D port (E5) is the main remaining lift.
- **Novelty is time-sensitive** (the area is crowding); we re-verify immediately before submission and, if needed,
  stake the claim on arXiv + a workshop while completing the main-track evaluation.

---

## 7. Conclusion
The user's hand is a free, accurate, in-scene metric ruler. Coupling an up-to-scale feedforward Gaussian scene to the
metric hand in a single pass yields a metric hand placement and a metric scene without a metric-depth foundation
model or multi-view. The contribution is the direction — hand-anchors-scene — and a non-circular evaluation that
shows the scene, not only the hand, becomes metric.

---

## References (verified — see novelty-assessment.md)
- [frost] Frost et al., *Recovering Stable Scale in Monocular SLAM Using Object-Supplemented BA*, IEEE T-RO 2018.
- [mccho] *MCC-HO*, CVPR 2024 — arXiv:2404.06507.
- [hawor] *HaWoR*, CVPR 2025 — arXiv:2501.02973.
- [hand3r] *Hand3R*, 2026 — arXiv:2602.03200 (CUT3R).
- [iags] *Interaction-Aware 4D Gaussian Splatting*, Nov 2025 — arXiv:2511.14540.
- [anysplat] *AnySplat*, 2025 — arXiv:2505.23716. [splatsap] *Splat-SAP*, 2025 — arXiv:2511.22704.
- [unidepth] *UniDepth*, CVPR 2024.
- [novelty] Internal verified novelty sweep, `report/novelty-assessment.md`.
