# FF-4DGS-Ego: related work & positioning

Compiled 2026-06-18 (research agent), rewritten 2026-07 around the current thesis.
Use for the paper's related-work section.

**Current thesis:** absolute (camera-frame) 3D hand pose from monocular egocentric RGB,
via a trained hand head on a frozen feedforward backbone, supervised with an absolute-3D
keypoint loss (kp3d_abs) and evaluated on dense-GT-depth datasets (HOI4D, H2O). The
contribution is the **recipe** — absolute-3D keypoint supervision, hand-box geometry fed
to the head, and detector-box robustness — not any particular backbone. The
backbone-swap ablation (2026-07) settled this: a frozen DINOv2 backbone matches the
scene-reconstruction backbone (21.9 single-seed vs 23.6 ± 0.8 over 3 seeds, C-abs; the
1.7mm gap is within seed noise, so state DINO ≈ recon, not DINO > recon) and random-init
is clearly worse (27.7), so the features need only be a competent frozen image
representation. An
earlier claim that feedforward *reconstruction* features specifically encode absolute
egocentric hand depth is therefore NOT made; scene reconstruction is not a contribution
lever.

## 1. Feedforward 3D-from-images (backbone lineage)
- **DUSt3R** (2312.14132): pairwise pointmaps from unposed images. Origin of the unposed feedforward regression regime our backbone lives in.
- **MASt3R** (2406.09756): +matching, metric pointmaps from training priors. Evidence that recon backbones can carry metric information; we probe the same property for hand depth.
- **VGGT** (2503.11651): large feedforward transformer (cameras+depth+pointmaps+tracks), canonical backbone of the family.
- **CUT3R** (2501.12387): online/streaming pointmaps. Scene-only; the backbone Hand3R builds on.
- **NoPoSplat** (2410.24207): pose-free feedforward 3DGS, sparse static views. (Also cite MVSplat 2403.14627, Fast3R, Spann3R.)

## 2. Feedforward 4D / dynamic reconstruction from monocular video (backbone family)
- **4DGT** (2506.08015): feedforward 4D Gaussians from *posed* monocular video (Aria). Backbone-family prior; no hand.
- **WorldMirror** (2510.10726): all-in-one feedforward reconstruction with prior prompting. The template our frozen backbone follows.
- **NeoVerse** (2601.00393): 4D world model from in-the-wild monocular video (WorldMirror-based). **Our frozen backbone.** We train only a hand head on top of it.

## 3. Joint hand+scene / hand-in-camera-frame (the direct competitors)
- **Hand3R** (2602.03200): **THE closest prior.** Online 4D hand-scene: frozen hand expert + 4D scene FM (CUT3R), one pass → MANO + dense metric scene. Reports absolute camera-frame C-MPJPE ~42.6mm on HOI4D. **Gap we attack:** their absolute hand accuracy, in their evaluation regime. Cite-and-contrast; their numbers are from the paper (not re-run, cross-split caveat applies).
- **HaPTIC**: multi-view/video hand pose. Strong root-relative baseline on our split (C_rr 28.7 / WA 35.3); its absolute depth suffers from weak-perspective miscalibration at 224px (native-HD rerun pending).
- **SHARE** (2510.15342): grounds SMPL humans to scene via pointmap depth. Full-body, non-ego analog of grounding a body in recon geometry.
- **MetricHMSR** (2506.09919): metric human mesh + scene from monocular. Human analog; not egocentric hand.

## 4. World-grounded and single-frame ego hand pose
- **HaWoR** (CVPR'25, 2501.02973): world-space ego hand motion; metric scale via DROID-SLAM + Metric3D on the background (hand masked out). Opposite recipe to ours: external SLAM + depth-FM vs a single frozen recon backbone. Own-SLAM regime; not yet re-run by us.
- **WHOLE** (2602.22209), **EgoGrasp** (2601.01050): world-space ego hand-object trajectories; scale from depth-FM / metric SLAM.
- **WiLoR**, **HaMeR**: single-frame crop-based MANO regression, weak-perspective. Off-the-shelf on our HOI4D split their absolute camera-frame error is large (see `table_global_comparison`: WiLoR 240.0, HaMeR 168.3 C-abs, GT boxes), which is the structural weakness absolute supervision targets. Fine-tuned on our split with the same kp3d_abs loss, HaMeR closes most of that gap (full fine-tune 21.4 GT-box) — so the honest framing is "the recipe, applied to any competent backbone including a 632M HaMeR, produces metric absolute depth", with our light frozen-feature head competitive and more detector-box robust.
- **Metric3D v2** (2404.15506), **UniDepth V2** (2502.20110), **Depth Pro** (2410.02073), **MoGe-2** (2507.02546): metric-depth FMs; the standard external source of absolute scale that SLAM-based pipelines bolt on. We use none of them.

## 3 closest works
1. **Hand3R** (2602.03200): feedforward hand + metric scene in one pass; the absolute camera-frame bar (~42.6mm C-MPJPE, HOI4D, their split).
2. **HaWoR** (2501.02973): world-space ego hands, but via SLAM + depth-FM, hand masked out of the scene.
3. **WiLoR / HaMeR**: the single-frame crop regime; strong articulation, weak absolute placement.

## Positioning (claim)
Single-frame crop methods (WiLoR, HaMeR) leave absolute translation under-constrained; world-space methods (HaWoR, WHOLE, EgoGrasp) buy absolute scale with SLAM plus a metric-depth FM; Hand3R gets it from a metric scene foundation model. We show that a **light hand head on a frozen feedforward backbone, trained with an absolute-3D keypoint loss and given the hand box as geometry, recovers metric absolute egocentric hand depth without SLAM or a depth-FM**: it reaches 23.6mm camera-frame C-MPJPE (3 seeds) on our HOI4D 157-seq test split (GT-derived boxes; leakage-audited clean-152 headline; detector-box E2E in `table_global_comparison`), versus off-the-shelf WiLoR 240.0 / HaMeR 168.3 on the same split and ~42.6 reported by Hand3R on their split. The absolute signal is not specific to reconstruction features — a frozen DINOv2 backbone matches it (21.9mm single-seed, within the winner's ±0.8 seed noise) — so the lever is the recipe, not the backbone. The same recipe applied as a full HaMeR fine-tune reaches 21.4mm; our frozen-feature head is competitive there and degrades markedly less under detector boxes (footnote d of the table).

---
**Thesis history.** An earlier version of this document positioned the project as "feedforward 4D Gaussian scene made metric by an in-scene MANO hand anchor". That scene-metric framing was experimentally falsified in 2026-07 (scale-source ablation: hand-as-global-scene-scale 0.728 vs oracle 1.022; the 4DGS backbone is frozen third-party and Gaussian rendering is off). Scene reconstruction is not part of the claim; see git history for the old text.
