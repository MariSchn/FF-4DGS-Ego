# FF-4DGS-Ego: Road to a Tier-1 Publication

**Last updated:** 2026-07 (full rewrite around the current thesis; the old scene-metric plan is in git history)
**Target:** CVPR ~Nov 2026 (fallback: workshop/arXiv to stake the claim)
**Status:** strong internal headline number; the evaluation package (fair baselines, E2E protocol, second dataset) is the gap.

---

## 0. The one-sentence contribution

> **Feedforward scene-reconstruction features encode absolute (metric) egocentric hand depth: a hand head trained with an absolute-3D keypoint loss (kp3d_abs) on a frozen recon backbone recovers camera-frame 3D hand pose from monocular egocentric RGB, with no SLAM and no metric-depth foundation model.**

Positioning (see `related-work-positioning.md`, `novelty-assessment.md`): Hand3R
(arXiv 2602.03200) is THE neighbor and the bar; it reports absolute camera-frame
C-MPJPE ~42.6mm on HOI4D. ~40mm is the competitive threshold; we are at 23.6mm on our
own split, so the fight is about protocol fairness, not raw distance.

---

## 1. What we have (canonical numbers, HOI4D, our 157-seq test split, right hand, smplx-16, mm)

| Result | Number | Caveat |
|---|---|---|
| **Ours (winner10ep, GT-derived boxes)** | **C-abs 23.6 full-157**; clean-152 is the leakage-audited headline | GT-bbox advantage acknowledged |
| World-space | W-MPJPE 195.6 | weak; chaining/abs-depth limited, not a headline claim |
| bbox robustness (winner10ep) | jitter0.2: 31.8 / fixed0.30: 43.5 | degrades under box noise |
| jitter-augmented retrain (jitterrob, TEST60) | clean 26.9 / jitter 27.3 / fixed 31.2 | mitigation works |
| E2E with WiLoR detector boxes | 140.6 | diagnosed flawed-input artifact (224px detection, no squaring, decayed carry-forward; det-vs-GT IoU 0.383, 18.8% zero-overlap frames). detbox v2 (HD detection + exact training box protocol) in flight; honest E2E pending |
| WiLoR baseline (same split, same GT) | C-abs 218.2 | root-relative re-run pending after joint-order fix |
| HaMeR baseline (same split, same GT) | C-abs 187.9 | same re-run pending |
| HaPTIC | C_rr 28.7 / WA 35.3 valid | C-abs 2.7x off from 224px weak-persp miscalibration; native-HD rerun in flight |
| HaWoR | not built | own-SLAM regime, hard to run fairly |
| Hand3R | ~42.6 C-MPJPE | paper number, their split, not re-run |

Split audit (self-reported): 5 test seqs contaminated by warm-start (excluded in clean-152);
25 sibling-take seqs (scene-disjoint-132 variant exists).

---

## 2. The gap to Tier-1 (what reviewers will demand)

1. **The GT-bbox confound.** Our headline uses GT-derived boxes; baselines do not get that
   luxury in their native use. Response: bbox-perturbation ablation (done, above) + detbox v2
   E2E number (in flight). This is the single most likely reject reason.
2. **A fair absolute baseline in its native regime.** HaPTIC at native HD, WiLoR/HaMeR
   root-relative re-runs, and a decision on HaWoR (build it or justify its exclusion).
3. **Causal ablation.** Show the gain comes from kp3d_abs + recon features, not just more
   training on HOI4D (control head, backbone-swap style ablation).
4. **Second dataset.** H2O (dense GT depth) to show it is not a HOI4D artifact.
5. **Hand3R protocol parity.** Cross-split comparison is the weakest joint; get to a
   same-protocol number or state the caveat prominently.
6. **Seed/variance reporting** for the headline model.

---

## 3. Paper story (draft)

- **Title direction:** absolute egocentric hand pose from frozen feedforward reconstruction features.
- **Abstract arc:** single-frame crop methods leave absolute translation under-constrained;
  world-space methods buy it with SLAM + depth-FMs; we show a frozen feedforward recon
  backbone already encodes absolute ego hand depth, and a small trained head extracts it.
- **Experiments:** headline C-abs vs WiLoR/HaMeR/HaPTIC (same split) and Hand3R (cited),
  bbox-robustness ablation, detbox v2 E2E, kp3d_abs causal control, H2O, leakage-audited
  split variants.
- **Limitations to state:** world-space (W-MPJPE) is weak; GT-box protocol dependence;
  Hand3R not re-run.

---

## 4. Risks & kill/pivot criteria

- **detbox v2 E2E lands far above the clean number:** the headline survives only as a
  GT-box-protocol result; frame the paper around the protocol-controlled comparison and the
  robustness retrain, or hold for a detector fix.
- **HaPTIC native-HD C-abs turns out strong:** our margin shrinks to a comparison against a
  fair multi-view/video method; the frozen-backbone finding must carry more weight.
- **Hand3R same-protocol run beats us:** the empirical claim dies; the probing finding alone
  is workshop material.
- **Niche closes (new 2026 paper):** move to arXiv/workshop immediately.

---
**Thesis history.** This plan originally targeted "Metric Hands, Metric Scene": a feedforward
4D Gaussian scene made metric by an in-scene hand anchor. The scene-metric claim was
experimentally falsified in 2026-07 (scale-source ablation: hand-as-global-scene-scale 0.728
vs oracle 1.022; the 4DGS backbone is frozen third-party and Gaussian rendering is off).
Scene reconstruction is not a contribution lever; the old plan and its E0-E7 roadmap are
preserved in git history.
