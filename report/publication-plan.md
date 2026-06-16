# FF-4DGS-Ego — Road to a Tier-1 Publication

> **⚠️ 2026-06-17 REFRAME — see `report/overnight-findings.md`.** The scene-metric thesis (E2) was
> **FALSIFIED**: the non-circular B2 eval shows the anchor makes object depth *worse* (135 vs 62 cm),
> and the at-hand "scene-metric" win was a circular artifact. **Contribution narrows to metric HAND
> PLACEMENT.** Realistic target drops toward **workshop/arXiv** unless the external placement
> comparison (HaWoR/Hand3R, E4/E5) is clearly SOTA-competitive — that is now the make-or-break, not
> scene-metric. E1/E2/E3 below are superseded by the real results in overnight-findings.md.

**Last updated:** 2026-06-16
**Target:** CVPR / ICCV / ECCV main track (fallback: a top workshop to stake the claim fast)
**Status:** novel idea + confirmed thesis + strong internal result; the *evaluation* is the gap.

---

## 0. The one-sentence contribution

> **A feedforward, single-pass method that recovers a *metric-scale* 3D-Gaussian scene from monocular egocentric RGB by anchoring scale to the in-scene metric MANO hand — making both the hand placement and the scene geometry metric without a metric-depth foundation model or multi-view.**

Positioning (verified novelty sweep, `novelty-assessment.md`): the **hand-anchors-scene** direction on a **feedforward GS** scene is open. Closest = **Hand3R** (reverse direction: scene-FM supplies scale, point-maps not GS, DexYCB/HOI4D). We must cite-and-contrast it and **foreground the direction + "no metric-FM needed"**.

---

## 1. What we already have (confirmed, same-protocol)

| Result | Number | Evidence |
|---|---|---|
| Direction: scene_follows_hand wins | 89.4mm vs 172mm (hand_follows_scene) | P1 full-val |
| **Placement: warm-start + abs-3D vs baseline** | **52.93 vs 81.37mm (−35%), robust 9-seq** | eval_large_* |
| Our model generalizes; baseline degrades | ours 53→53, his 61→81 across splits | 1-seq vs 9-seq |
| **Scene-metric thesis (at hand)** | residual **13.1→4.5cm**, scale CV **25.5→5.9%** | eval_metric_scale |
| Pose preserved | PA-MPJPE ~7.9mm (≈ his 7.5) | all evals |
| Novelty | niche OPEN; Hand3R distinct on 4 axes | novelty-assessment.md |

**Honest caveats baked in:** absolute MPJPE is split-dependent (his model = 50/61/81 across splits) → only same-split deltas count; the scene-metric proof is currently *at the hand* (semi-circular).

---

## 2. The gap to Tier-1 (what reviewers will demand)

1. **Causation** — is the win from abs-3D, or just more fine-tuning? → **control ablation**.
2. **"Why not just a metric-depth FM?"** → **B1: UniDepth/Metric3D baseline**.
3. **Non-circular scene-metric** — prove the *scene* (not just the hand) is metric → **B2: GT object depth**.
4. **External comparison to published SOTA** — we currently beat only our teammate's checkpoint → **HOI4D port, head-to-head with Hand3R; run HaWoR ourselves on HOT3D**.
5. **Multi-dataset** — ≥2 datasets (HOT3D + HOI4D, ideally + DexYCB/Ego-Exo4D).
6. **Scene-quality not harmed** — GS PSNR/SSIM/LPIPS maintained while adding metric scale.

---

## 3. Experiment roadmap — with the NUMBER each must hit

Legend: ✅ done · 🔄 running · 🔨 building · ⬜ todo

| # | Experiment | Status | **Target number to CLAIM it** | If it misses |
|---|---|---|---|---|
| E0 | **Control** (warm-start, abs-3D+anchor zeroed) | 🔄 99188 running | control MPJPE **stays ≥ ~60mm (1-seq) / ≥ ~75mm (9-seq)** → abs-3D *causes* the −28mm win | if control ≈ 53 → win is just more training → **reframe or kill** |
| E1 | **B1 metric-depth-FM** (UniDepth at hand) | 🟢 staged — queued 99195 (extract); UniDepth-v2 ViT-L weights cached; x86 eval ready | UniDepth hand-depth error **> ~10cm (≥2× our 4.5cm)** → FM can't replace the anchor | if FM < ~5cm → contribution weakened → pivot to "anchor + FM complementary" |
| E2 | **B2 GT-object-depth scene-metric** (non-hand) | 🟢 staged — queued 99194; **render geometry validated** (hand-contact depth agrees 1.6–4.2cm) | anchored object-region depth error **clearly < no-anchor** (e.g. anchored ~8-12cm vs no-anchor ~20-30cm; δ<0.1m up ≥15pts) | if no gain in object regions → thesis is hand-only → weaken claim to "metric hand placement" |
| E3 | **Scene-quality** (full-frame GS PSNR/SSIM/LPIPS) | 🟢 staged — `eval_gs_quality_gb10.sh` ready (reuses eval_gs_head) | PSNR/SSIM **within noise of baseline** (we don't trade scene quality for metric scale) | if PSNR drops >1dB → note trade-off, tune anchor weight |
| E4 | **Run HaWoR on HOT3D** (monocular external baseline) | 🟡 scoped — HaWoR cloned; ships `scripts/scripts_eval/eval_hawor_hot3d.py`; HF weights downloading | our placement (52.9) **< HaWoR's** on the same HOT3D clips | if we lose → emphasize scene-metric (HaWoR has no scene) |
| E5 | **HOI4D port + Hand3R head-to-head** | ⬜ (multi-wk) | **competitive C-/W-MPJPE with Hand3R** + we report scene-metric they don't | if backbone fails on HOI4D → pivot to HOT3D-primary + HaWoR/FM baselines |
| E6 | **Multi-dataset table** | ⬜ | results on **≥2 datasets** (HOT3D + HOI4D) | single-dataset → workshop, not main track |
| E7 | **Final novelty re-check** | ⬜ | no new 2026 paper closes the exact niche | if closed → reframe / move fast |

---

## 4. The headline numbers we are trying to land (the paper's money table)

These are the targets that, if hit, make a credible main-track paper:

1. **Placement (same-protocol):** ours **≈53mm** vs baseline **81mm** (−35%), **robust across seqs**, AND **competitive with / better than HaWoR & Hand3R** on a common benchmark. *PA-MPJPE ≈ 8mm (pose parity).*
2. **Scene metric, non-circular (object regions):** anchored depth error **≈10cm** vs no-anchor **≈25cm**; scale **CV < 6%** vs **>25%**.
3. **vs metric-depth FM:** UniDepth at hand **>10cm** (FM can't do it) — so the **hand anchor is the cheaper, more accurate metric source**.
4. **Causal:** control (no abs-3D) **does not** recover the win → abs-3D is the cause.
5. **Scene quality:** PSNR within ~0.5dB of baseline.
6. **≥2 datasets** + the above hold on both.

If 1-6 hold → strong main-track story. If 1-4 hold but 5-6 partial → workshop + arXiv to stake the claim.

---

## 5. Paper story / structure (draft)

- **Title (working):** *Metric Hands, Metric Scene: Feedforward Egocentric Gaussian Reconstruction Anchored to the Hand.*
- **Abstract arc:** monocular feedforward GS is up-to-scale; the in-scene metric hand is a free, accurate metric anchor; we couple them in one pass → metric hand placement + metric scene, no metric-FM/multi-view.
- **Method:** frozen feedforward GS backbone + HaMeR head + HDGLA anchor (hand→scene metric coupling) + abs-3D placement loss.
- **Experiments:** E0 (causal ablation), E1 (vs depth-FM), E2 (non-circular scene-metric), E4/E5 (vs HaWoR/Hand3R), E6 (multi-dataset), E3 (scene quality).
- **Figures:** (1) teaser — hand-in-scene 3D alignment (render_alignment_3d, gsplat); (2) scale-CV + residual bars (anchor vs baseline vs FM); (3) object-region depth-error maps (B2); (4) qualitative scene+hand renders.
- **Cite-and-contrast:** Hand3R (reverse direction), HaWoR (masks hand out), MCC-HO (object only), metric-depth FMs.

---

## 6. Timeline & milestones (single GPU, serial — the constraint)

**Phase A — core claim (this week, on HOT3D, no port):**
- [🔄] E0 control verdict — *tonight*
- [🔨] E1 UniDepth baseline — env building; result ~2-3 days
- [🔨] E2 GT-object-depth — object library downloaded ✅; pipeline ~3-5 days
- [⬜] E3 scene-quality numbers — ~1 day (reuse region-masked GS metrics)
- [⬜] E4 HaWoR on HOT3D — ~3-4 days (install + run)

**Decision gate (end of Phase A):** do E0-E2 land their targets? If yes → commit to Phase B. If E1/E2 miss → reframe to the workshop scope.

**Phase B — external/main-track (multi-week):**
- [⬜] E5 HOI4D port + Hand3R head-to-head — ~3-4 weeks (the lift; gated on backbone working on HOI4D)
- [⬜] E6 second-dataset table consolidation
- [⬜] E7 final novelty re-check + write-up

**Submission targets:** ICCV/CVPR deadline (check the live calendar at write-up time) — and given the niche is crowding (Hand3R Feb'26), an **arXiv stake + workshop** in parallel is prudent.

---

## 7. Risks & kill/pivot criteria

- **E0 fails (control ≈ warm-start):** the win is fine-tuning, not abs-3D → either find the true cause or pivot the framing to "metric coupling" (E2) as the contribution, not placement.
- **E1 fails (FM accurate at hand):** reframe as "hand anchor is FM-free + complementary"; still publishable but weaker.
- **E5 backbone fails on HOI4D:** pivot to **HOT3D-primary**, with HaWoR + FM + a self-defined monocular metric hand+scene benchmark (riskier but viable for workshop / borderline main-track).
- **Niche closes (a concurrent paper):** move to arXiv/workshop immediately; lean on the GS+direction+benchmark differentiators.
- **Single-GPU serialization** is the throughput bottleneck — Phase A experiments queue one-at-a-time; budget ~1.5 weeks wall-clock for A.

---

## 8. Immediate next actions (live)

**Build status (2026-06-16): E1/E2/E3 fully implemented, deployed, and queued behind the control. All buildable-without-GPU work is done.**

Cluster queue (serial, MaxJobs=1): `99188` control RUNNING → `99194` B2/E2 PENDING → `99195` B1/E1-extract PENDING. UniDepth weights cached; HaWoR HF weights downloading.

1. **Read E0 control verdict** when 99188 finishes — the make-or-break for the placement claim.
2. **E2 (99194)** auto-fires after control → read object-region depth error (anchored vs no-anchor). Eval: `scripts/eval_scene_metric_gt.py` (+ `b2_render_object_depth.py`, geometry validated).
3. **E1**: after `99195` writes `outputs/b1_bundle.pt`, submit `eval_unidepth_b1_x86.sh` (x86 jobs partition) → UniDepth-vs-anchor hand-depth error.
4. **E3**: submit `eval_gs_quality_gb10.sh` (one slot) → scene-quality delta.
5. **E4**: finish HaWoR install (conda env py3.10 + torch1.13cu117, DROID-SLAM CUDA build on a GPU node, Metric3D + DROID Google-Drive weights via gdown), then run its `eval_hawor_hot3d.py` on the sequences overlapping ours. MANO already in `models/MANO`.
6. Then scope the HOI4D port (E5).

**Implementation artifacts (this build session):** `scripts/extract_b1_frames.py`, `scripts/eval_unidepth_b1.py`, `scripts/b2_render_object_depth.py`, `scripts/eval_scene_metric_gt.py`; wrappers `extract_b1_gb10.sh`, `eval_unidepth_b1_x86.sh`, `eval_scene_metric_gt_gb10.sh`, `eval_gs_quality_gb10.sh`. Env: `venv_unidepth` (x86, UniDepthV2 importable, weights cached); trimesh present in both venvs. NOTE: `venv_gb10` is **aarch64** — never import/pip it on the x86 login node.
