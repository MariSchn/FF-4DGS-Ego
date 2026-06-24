# Open limitations & next steps (status 2026-06-24)

One-line state: hand **shape/articulation is strong and transfers** (WA-MPJPE 21/47 mm on HOI4D,
root-rel C-MPJPE 93 mm); **absolute world placement (W-MPJPE) is the open gap** vs Hand3R (126 mm).
The 2026-06-22 session refuted scale and pose as W levers; the 2026-06-24 session ran the W-attack
probe and built + diagnosed the contact anchor (below).

## Update 2026-06-24: W-attack resolved + contact anchor diagnosed

Net: **`kp3d_abs` supervision is a real (partial) W lever, and the contact anchor works mechanically
but is reference-limited on the HOT3D testbed.**

**Supervision lever.** Turning on absolute-3D keypoint supervision moves **W-MPJPE 308 -> 250** and
halves per-clip absolute placement (**C-abs 115 -> 53**) on HOT3D (P0001, 8 segs, bounded probe).
Articulation holds (C root-rel ~33, beats Hand3R 42.6). So W is supervision-responsive, not purely
domain-locked. This is bankable and was reported to Cyrus.

**Contact anchor (scene-depth root re-anchor, Phase 1).** Built end to end as a feedforward post-hoc
module (`RootDepthRefine`: zero-init gated MLP emits a root depth shift toward `gs_depth` sampled at the
predicted wrist; gated consistency loss; no test-time optimisation, satisfying Cyrus's constraint).
A/B on HOT3D:

| run | W pooled | C-abs |
|---|---|---|
| deployed baseline | 308 | 115 |
| supervision only (anchor off) | 250 | 53 |
| contact (anchor on) | 246-249 | 51-55 |

The anchor adds ~4 mm, which is within the bounded probe's ±13 mm run-to-run noise. Gate/Δgs logging
showed why. Once given its own learning rate (1e-3; the head's 2e-5 starved the zero-init MLP), the
anchor trains and fires (gate ~70%, |dz| ~27 mm), but the disagreement it corrects toward is large and
noisy (**Δgs ~122 mm**, range 54-222) while the head is only ~50 mm off GT. So HOT3D's frozen
`gs_depth` is a **worse** reference than the head it corrects, and anchoring to it is a wash. Also the
re16 GT-reanchor ceiling (~49) equals C-abs (~50): per-frame placement is already at the oracle
ceiling, so the residual W is chaining/scale drift between frames, which a per-frame root correction
cannot touch.

**Verdict.** The anchor mechanism is sound; HOT3D is an unfair testbed (bad reference, exactly the
design's predicted risk). The fair and favorable test is **HOI4D** (validated 8% AbsRel `gs_depth` from
`hoi4d_depth/best_depth.pt`, plus a weaker HOT3D-trained head on HOI4D, so good reference + bad head).
**BLOCKED:** the HOI4D world-eval caches were cleared, and rebuilding them needs per-frame camera
extrinsics that are missing on the cluster (the Livioni mirror has images + depth but no poses), and all
three filesystems are at quota. HOI4D-contact is therefore a fast-follow gated on (1) re-acquiring
extrinsics from yinloonga/HOI4D, (2) a disk/inode quota bump (Cyrus pinged).

## What the 2026-06-22 session established (the diagnosis)

## What we just established (the diagnosis)

- **Sequence-level scale does NOT fix W.** 3-scale eval: per-clip 353.1, per-seq-median 350.0,
  per-seq-pooled 352.3 → W is **insensitive to global metric scale**.
- **Global camera poses do NOT fix W.** `--oracle_cam` (GT extrinsics, perfect poses, no chaining)
  = **363.7 mm ≈ chained 353.1**. Cross-checked by C-MPJPE_abs = 366.3 (uses *no* extrinsics) → not a
  convention bug.
- **Therefore W is hand-head-limited: the hand-root absolute depth-from-camera.** Shape transfers,
  absolute depth doesn't — a signature that includes a **HOT3D→HOI4D domain/scale gap** (we train on
  HOT3D, eval W on HOI4D).

## Limitations status

| # | Limitation | Status | What to do |
|---|---|---|---|
| L1 | **Loose hand–scene coupling** (tie predicted mesh to rendered hand region → exact registration) | Registration loss **implemented + self-test-validated** as `scripts/hand_scene_registration_loss.py` (`scene_follows_hand` dir); **not yet wired into training**. New finding: it targets **coupling / C-MPJPE consistency, NOT W**. | Wire it in (vertex accessor + train hook + unfreeze-GS config) and run as the **coupling rung of the scene-metric ladder** — framed for WA + the ablation story, not as a W fix. |
| L2 | **Per-frame heuristic scale → sequence-level / learned** | **Addressed + refuted as a W lever.** Pooled global solve tested → W flat. Learned ScaleHead (route b) is directional for *scene-metric* (14.07→13.08 cm). | Keep ScaleHead as the **route-b rung** of the scene-metric ladder (converge it, roadmap B1). Do **not** pitch it as a W fix. |
| L3 | **W-MPJPE absolute-placement gap vs Hand3R** (the now-central limitation) | Diagnosed: **not scale, not pose** → hand-head absolute depth + domain gap. | **ACTIVE: `configs/exp_p3_wabs.yaml`** — frozen-backbone hand-head retrain with `kp3d_abs` 0→0.3 + `transl`→1.0. Probe: moves W ⇒ supervision-limited (tune/extend); barely moves ⇒ domain-limited ⇒ **HOI4D fine-tune** (we have HOI4D right-hand pose GT). |
| L4 | **Frozen backbone** (rescale/place, can't re-predict geometry) | Partially addressed: partial-unfreeze + GT-depth rung **converged on HOI4D (20.5→13.4 cm)**, directional on HOT3D. | **Converged (a) on HOT3D** = the headline scene-metric experiment (roadmap B2). Gated on GPU/`/work` quota. |
| L5 | **HaWoR world-space external comparison** (Cyrus's explicit ask) | **Blocked** — DROID-SLAM is Blackwell-only-cluster-blocked; A100/H100 denied. | Use Hand3R published numbers (W 126 / C-MPJPE 42.6) as the comparison anchor for now; revisit HaWoR if GPU access changes. |
| L6 | **Renderable metric 4D-GS demo** (Fig 5) — the Hand3R differentiator (we render, they don't) | Local Blender Fig-5 renderer exists. | Produce the **metric Fig 5** (splat + metric hands). High value: it's the visual that distinguishes us. |
| L7 | Absolute MPJPE split-dependent; single-GPU throughput; novelty time-sensitive | Noted; same-split deltas only; novelty deep-research run. | Re-verify novelty pre-submission; keep same-split discipline. |

## Prioritized next steps

1. **[active] Run `exp_p3_wabs`** — the cheap W-attack probe (frozen backbone, `kp3d_abs` up). Decides
   supervision- vs domain-limited for W. Eval on HOI4D W/oracle after.
2. **Reframe the headline around WA + metric coupling** (the ablation ladder is the spine; **concede
   pose / absolute placement** — consistent with roadmap D1). W is reported honestly as bounded by
   monocular absolute hand depth.
3. **Wire + run the registration loss (L1)** as the coupling rung — strengthens the scene-metric ladder.
4. **Converged (a) on HOT3D (L4/B2)** — headline scene-metric drop; gated on compute.
5. **Metric Fig 5 demo (L6)** — the renderable-and-metric differentiator.
6. HaWoR (L5) — blocked; defer.

## Active experiment — `configs/exp_p3_wabs.yaml`

Forked from `exp_p3_scalehead.yaml`. Frozen backbone + GS head; **only the hand head trains**.
Changes: `kp3d_abs` 0.05→**0.3** (was 0.0 in the deployed head's training config — this **introduces**
absolute-3D keypoint supervision), `transl` 0.1→**1.0**, `enable_scale_head:false`, all GS/anchor/scale
losses 0, `kp3d_abs_warmup_steps` 50, `lr` 2e-5, `save_every` 1e6 (avoid multi-GB mid-run checkpoints —
scratch quota exhausted / `$HOME` inode-pressured), `output_dir` under team25. Eval: rerun W + oracle on
HOI4D; watch WA does not regress.

## Artifacts created/updated this session

- `scripts/hand_scene_registration_loss.py` — registration loss (L1), validated, not yet wired.
- `scripts/world_space_metrics.py` / `scripts/eval_world_space.py` — `c_mpjpe()`, 3-scale eval,
  `eval_oracle_cam()`.
- `configs/exp_p3_wabs.yaml` — the active W-attack config (L3).
- `report/hoi4d_world_eval_2026-06-22_3scale.json` — 3-scale result.
- `report/hoi4d-world-space-eval.md` — both refutations documented (scale + pose).
- This file.
