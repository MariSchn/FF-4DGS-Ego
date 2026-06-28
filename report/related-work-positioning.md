# FF-4DGS-Ego — related work & novelty positioning

Compiled 2026-06-18 (research agent). Use for the paper's related-work + novelty claim.

## 1. Feedforward 3D-from-images (pointmap / GS)
- **DUSt3R** (2312.14132) — pairwise pointmaps from unposed images. We inherit unposed feedforward regression but predict Gaussians + are dynamic.
- **MASt3R** (2406.09756) — +matching, metric pointmaps. They get scale from training priors; we recover it at inference via the hand anchor.
- **VGGT** (2503.11651) — large feedforward transformer (cameras+depth+pointmaps+tracks), canonical backbone. We extend to time-varying GS + metric hand head.
- **CUT3R** (2501.12387) — online/streaming pointmaps. Same online-ego regime; scene-only (no hand/GS/anchor).
- **NoPoSplat** (2410.24207) — pose-free feedforward 3DGS, sparse static views; leaves scale up-to-scale. (Also cite MVSplat 2403.14627, Fast3R, Spann3R.)

## 2. Feedforward 4D / dynamic GS from monocular video (backbone family)
- **4DGT** (2506.08015) — feedforward 4D Gaussians from *posed* monocular video. **Closest GS-output analog**; needs posed/metric Aria input, no hand, no anchor.
- **WorldMirror** (2510.10726) — all-in-one feedforward GS + geometry with prior prompting. Template our backbone follows; general/static, no hand/anchor.
- **NeoVerse** (2601.00393) — 4D world model GS from in-the-wild monocular video (WorldMirror-based). **Our backbone family**; we specialize to egocentric HOI + metric coupling.

## 3. Joint hand+scene / human+scene (the coupling idea)
- **Hand3R** (2602.03200) — **THE closest prior.** Online 4D hand-scene: frozen hand expert + 4D scene FM, one pass → MANO + dense metric scene. **Gap:** outputs pointmaps/meshes (not renderable Gaussians), and gets scale from a metric scene backbone — we make the *hand* the metric anchor of an up-to-scale GS scene.
- **SHARE** (2510.15342) — grounds SMPL humans to scene via pointmap depth. Same anchor-to-scene coupling; full-body, non-GS, non-ego.
- **MetricHMSR** (2506.09919) — metric human mesh + scene from monocular; weak-perspective scale. Human analog of our goal; not feedforward ego hand-driven GS.

## 4. Metric-scale recovery (depth FMs + known-object anchoring)
- **Metric3D v2** (2404.15506), **UniDepth V2** (2502.20110), **Depth Pro** (2410.02073), **MoGe-2** (2507.02546) — metric-from-depth-FM; alternative to our hand anchor (they use camera/training priors, we use an in-frame known-scale articulated object).
- **WHOLE** (2602.22209), **EgoGrasp** (2601.01050), **HaWoR** (CVPR'25) — world-grounded ego hand-object, anchoring scale to hands/first-frame object. **None produce a feedforward, renderable, metric 4D Gaussian scene anchored to the hand.**

## 3 closest works
1. **Hand3R** (2602.03200) — hand+scene one pass; not GS-renderable, scale from scene backbone.
2. **4DGT** (2506.08015) — feedforward 4D Gaussians; needs posed/metric input, no hand.
3. **WorldMirror/NeoVerse** (2510.10726 / 2601.00393) — feedforward GS world model; general, up-to-scale, no hand head.

## Novelty gap (claim)
FF-4DGS-Ego is the **first feedforward model that predicts a renderable up-to-scale 4D Gaussian scene from monocular egocentric RGB and resolves metric-scale ambiguity using an in-scene MANO hand as a geometric anchor** (VGGT/WorldMirror-lineage GS backbone + metric hand head). Hand3R shares hand+scene coupling but is non-renderable and gets scale from the scene backbone; 4DGT outputs 4D GS but needs posed/metric capture and no hand; depth-FM/known-object methods recover scale but never inside a feedforward 4D-GS pipeline. The unoccupied intersection — **feedforward + egocentric + 4D Gaussians + metric-scale-from-hand-anchor** (HOT3D + H2O) — is our claim.
