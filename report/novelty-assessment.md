# Novelty Assessment: Absolute Egocentric Hand Depth from a Frozen Feedforward Recon Backbone

**Date:** 2026-06-15 (literature sweep), reframed 2026-07 around the current thesis.
**Method (FF-4DGS-Ego, current):** frozen feedforward scene-reconstruction backbone
(VGGT/WorldMirror/NeoVerse lineage) + a trained HaMeR-style MANO hand head supervised with an
absolute-3D keypoint loss (kp3d_abs). Output: absolute (camera-frame) 3D hand pose from
monocular egocentric RGB. No SLAM, no metric-depth foundation model, no scene supervision.
Evaluated on dense-GT-depth datasets (HOI4D primary, H2O planned).
**Source of the literature list:** automated, adversarially-verified sweep (27 primary sources,
124 claims extracted, 25 verified by 3-vote refutation panel). Re-verify before submission.

---

## Verdict

The contribution is **empirical plus a probing finding**, not a new primitive: frozen
feedforward reconstruction features carry enough signal to place an egocentric hand
absolutely in the camera frame, and a small trained head extracts it to 23.6mm C-abs on our
HOI4D 157-seq split (clean-152 leakage-audited headline), where single-frame crop methods
sit at 187.9 to 218.2mm on the same split and Hand3R reports ~42.6mm on its own split.
The niche "absolute ego hand depth directly from a recon-backbone feature space, no
SLAM/depth-FM" is not occupied by a verified prior work, but the *area* (egocentric
hand+scene, camera-frame hand placement) is crowding fast in 2025-2026. Novelty is
time-sensitive and must be re-checked immediately before any submission.

---

## Closest prior work and why each differs

| Work | Venue/Date | What it does | Relation to the current thesis |
|---|---|---|---|
| **Hand3R** ⚠️ | 2026 (arXiv 2602.03200, on CUT3R) | joint hand + dense metric-scale scene, feedforward, monocular; absolute C-MPJPE ~42.6mm on HOI4D | **the direct competitor and the bar.** Same regime (feedforward, camera-frame absolute). Differs in mechanism (scene FM supplies scale, scene-aware prompting) and in that we output hand pose only. The comparison that matters is empirical, same-protocol. |
| **HaWoR** | CVPR 2025 (2501.02973) | world-space hand motion; metric scale via DROID-SLAM + Metric3D on background | opposite recipe: external SLAM + depth-FM, hand masked out. Not a single feedforward model. Own-SLAM regime; a fair re-run is hard (baseline not yet built). |
| **HaPTIC** | 2025/26 | multi-view/video hand pose | strong root-relative (C_rr 28.7 on our split) but absolute depth miscalibrated at 224px weak-perspective; native-HD rerun pending. Not recon-feature based. |
| **WiLoR / HaMeR** | 2024/25 | single-frame crop MANO regression | weak-perspective translation is structurally under-constrained; 218.2 / 187.9 C-abs on our split. The regime we improve on. |
| **EgoGrasp / WHOLE** | 2026 | world-space ego hand-object pose/trajectories | scale from depth-FM / metric SLAM, multi-stage pipelines, not a single frozen recon backbone. |
| **MCC-HO** | CVPR 2024 | hand disambiguates held-object location/scale | object-only; hand-relative, not absolute camera-frame hand pose. |
| **UniSH / MetricHMSR / SHARE** | 2024-26 | metric human (SMPL) + scene from monocular | body-side analogs of "human grounded in recon geometry"; not egocentric MANO hands; cite as principle precedent. |

Backbone lineage (DUSt3R/MASt3R, VGGT, CUT3R, WorldMirror, NeoVerse) is the correct prior-art
family; none of these train or evaluate an absolute egocentric hand head.

---

## The one to study closely: Hand3R (the most threatening neighbor)

- Joint hand + dense metric-scale scene, single feedforward pass, built on CUT3R.
- It already delivers absolute camera-frame hand pose, so "we do absolute camera-frame hand
  pose" is not novel by itself. What we bring: (1) a large same-task accuracy margin
  (23.6 vs their reported 42.6, with the honest caveat that ours is our split with
  GT-derived boxes and theirs is their split, paper number, not re-run); (2) the finding
  that a frozen, hand-agnostic recon backbone suffices, with no metric scene output needed.
- **Action:** the cross-split caveat is the paper's weakest joint. A same-protocol
  comparison (their eval regime, or their code on our split) is what makes the claim stick.

---

## Caveats (carried over from the verification panel, updated)

1. **Time-sensitivity is the dominant risk.** Hand3R (Feb'26), EgoGrasp (Jan'26), WHOLE
   (Feb'26) are months old. Re-check right before submission.
2. **The empirical bar, not idea-novelty, is the gap to Tier-1.** The GT-bbox evaluation
   advantage and the pending detector-box E2E protocol are the reviewer attack surface.
3. **Known split issues (self-audited):** 5 test seqs contaminated by warm-start (excluded
   in clean-152), 25 sibling-take seqs (scene-disjoint-132 variant exists). Report the
   audited variants.
4. Verifier process noise in the original sweep: small-model fetches hallucinated claims for
   EgoGrasp and Hand3R and were corrected by the panel. Trust verified claim text only.

---

## Tier-1 feasibility, honest read

The idea ("recon features encode absolute ego hand depth") is a clean, testable claim with a
strong internal number. What decides Tier-1 is the evaluation package: fair absolute
baselines in their native regime, a defensible E2E (detector-box) protocol, a second dataset
(H2O), and an ablation showing kp3d_abs plus the backbone features (not just more training)
cause the gain. Without those, the honest ceiling is workshop/arXiv.

---

## Key sources

- Hand3R, 2026, arXiv 2602.03200 (CUT3R: cut3r.github.io)
- HaWoR, CVPR 2025, arXiv 2501.02973
- EgoGrasp 2026, arXiv 2601.01050; WHOLE 2026, arXiv 2602.22209
- MCC-HO, CVPR 2024, arXiv 2404.06507
- MetricHMSR, arXiv 2506.09919; SHARE, arXiv 2510.15342
- Backbones: VGGT 2503.11651; WorldMirror 2510.10726; NeoVerse 2601.00393; CUT3R 2501.12387

---
**Thesis history.** This document originally assessed the niche "trusted-metric in-scene hand
rescaling an up-to-scale feedforward Gaussian scene" (hand-anchors-scene). That scene-metric
framing was experimentally falsified in 2026-07 (scale-source ablation: hand-as-global-scene-scale
0.728 vs oracle 1.022; the 4DGS backbone is frozen third-party and Gaussian rendering is off).
The old assessment is preserved in git history; do not resurrect the scene claim.
