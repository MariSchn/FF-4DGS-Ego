# World-space hand baseline research (for the scale comparison)

Compiled 2026-06-18 by a research agent. Source-of-truth for `scale-evaluation.md`.

## Two-tier framing (the story)

| Tier | Methods | Output | On a SCALE benchmark |
|---|---|---|---|
| **A. World-space METRIC** (true scale competitors) | **HaWoR, Hand3R**, Dyn-HaMR, HMP-SLAM | metric world-space trajectories (scale from metric-depth+SLAM, or a 4D scene FM) | the real baselines |
| **B. Camera-frame / single-image** (fail at scale) | **HaMeR, WiLoR**, OmniHands, HaMuCo | MANO + **weak-perspective camera, dummy focal=5000** → translation is a crop artifact, not metric | catastrophic absolute error |

## The ready-made citation — Hand3R (arXiv:2602.03200) HOI4D benchmark, **C-MPJPE** (our metric)

| Method | Type | **C-MPJPE (mm)** ↓ | WA-MPJPE short | WA-MPJPE long |
|---|---|---|---|---|
| HaMeR-SLAM | offline multi-stage | **248.23** | 52.69 | 85.46 |
| WiLoR-SLAM | offline multi-stage | **252.24** | 52.91 | 87.51 |
| HaWoR | offline multi-stage | **51.77** | 22.54 | 27.40 |
| Hand3R | online one-stage | **42.6** | 38.04 | 56.71 |

This single table is the two-tier story: camera-frame methods ≈250 mm even *with SLAM bolted on*; world-space methods 42–52 mm.

## Per-method (repo · output · effort)

- **HaWoR** (CVPR'25) — https://github.com/ThunderVVV/HaWoR · code+weights public (HF `ThunderVVV/HaWoR`). **World-space metric** (DROID-SLAM trajectory scaled to **Metric3D** depth). Video in; needs intrinsics for SLAM. HOT3D: **W-MPJPE 33.20, WA-MPJPE 11.27**, PA 4.79. Effort **medium** (build DROID-SLAM CUDA ext, torch1.13/cu117). **#1 must-run competitor.**
- **Hand3R** (arXiv:2602.03200, Feb 2026) — **no code released.** Feedforward world-space metric hands + scene, **no SLAM/intrinsics**; HaMeR ViT + CUT3R. **The closest conceptual rival to us.** HOI4D C-MPJPE 42.6. **Cite, can't run.**
- **HaMeR** (CVPR'24) — https://github.com/geopavlakos/hamer · public. **Camera-frame, weak-perspective, NOT metric** (`pred_cam_t_full` via dummy focal 5000). Easy to run. The canonical "great pose, no scale" point.
- **WiLoR** — https://github.com/rolpotamias/WiLoR · public. Same weak-perspective regression as HaMeR + fast detector. **Easiest install.** Its `detector.pt` is what HaWoR uses (partial HaWoR prep).
- **Dyn-HaMR** (CVPR'25) — https://github.com/ZhengdiYu/Dyn-HaMR · world-space metric but **optimization-based** (SLAM+priors). **Hard** to run.
- OmniHands / HaMuCo — relative/camera-frame, **not** scale competitors. Skip.

## Ranked recommendation to actually run (on shared data, e.g. H2O subj4)

1. **HaWoR** — the world-space metric competitor; reports W/WA-MPJPE. (cost: DROID-SLAM build)
2. **WiLoR** — easiest camera-frame baseline; weak-perspective `cam_t` → shows scale failure.
3. **HaMeR** — second camera-frame baseline; matches Hand3R's table.
- **Cite Hand3R** (no code). Skip OmniHands/HaMuCo for scale.

All three have public weights and already appear in Hand3R's HOI4D table, so our re-run numbers are checkable. Commands are in the agent transcript / HaWoR+WiLoR+HaMeR READMEs.

## Hard constraint
HaWoR/Dyn-HaMR need DROID-SLAM (torch1.13/cu117, ≤ sm_86) → **do not run on this cluster's Blackwell GPUs**. Need one Ampere/Hopper GPU. HaMeR/WiLoR (no SLAM ext) may port to torch≥2.7/Blackwell — worth a feasibility attempt.
