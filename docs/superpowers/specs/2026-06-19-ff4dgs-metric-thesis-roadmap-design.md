# FF-4DGS-Ego — metric-scene thesis roadmap (design)

**Date:** 2026-06-19
**Target:** CVPR ~Nov 2026 (main conference)
**Status:** design locked via brainstorming; experiments partly demonstrated (directional)

## Thesis (locked)

**Unified metric model:** the first feedforward model that turns an up-to-scale egocentric
4D Gaussian-Splatting scene **metric** by anchoring to an in-scene metric MANO hand — one
forward pass, renderable, no SLAM. The contribution is the **mechanism/capability**, not a
hand-pose leaderboard win. We explicitly concede pose (PA-MPJPE 43.6 vs reported 16.7–30.8;
different protocol).

## Compute assumption (locked)

**Hybrid: prove small now, scale later.** Demonstrate every mechanism at small scale on
the consumer gb10 cards available today; upgrade to converged magnitudes if/when an
H100/A100 lands (Daniel's pending resource request). The plan must not stall if no bigger
card arrives.

## Core result: the metric-scene ablation ladder

| Rung | Claim | Evidence | State |
|---|---|---|---|
| frozen, no metric loss | scene not metric on objects | 62 cm (B2) | done |
| + in-scene hand anchor | hand 3.5× better anchor than depth FM | B1 (4.5 vs 15.7 cm) | done |
| + feedforward scale head (b) | learnable global metric scale, one pass | 14.07 → 13.08 cm (4 steps) | directional |
| + partial-unfreeze + GT depth (a) | backbone learns metric geometry | HOT3D 23.8 → 22.9 cm (6 steps); **HOI4D 20.5 → 13.4 cm converged** | a: directional / HOI4D: converged |

This ladder is the paper's spine. Three of four rungs are demonstrated; the top rung is
converged on HOI4D and directional on HOT3D.

## Approach (chosen)

**Approach 3, HOI4D-led** (mechanism-primary, ablation ladder on both datasets). Robust
under hybrid compute: the decisive rung (HOI4D dense-depth) already works without an H100;
the scale-head and small partial-unfreeze rungs run to a verdict on consumer cards; the
converged HOT3D version only *upgrades* the result. Turns the frozen-backbone failure (B2)
into rung 1 of the story rather than a hole.

## Workstreams

### A. Now — no compute gate (this week)
- A1. Capture (a)/(b) directional verdicts; write result notes; add rows to the table. *(done/in progress)*
- A2. Send Cyrus: table + the two scale verdicts + re-land the GPU ask. **(highest-leverage ungated move)**
- A3. This spec.

### B. Critical converged runs — gated on `/work` quota clearing or a GPU
- B1. Converged (b) scale head → residual floor.
- B2. **Converged (a) on HOT3D → decisive object-metric drop. (headline experiment)**
- B3. Extend HOI4D → sub-10 cm.

### C. Credibility / CVPR bar
- C1. **HaWoR world-space comparison** (Cyrus's explicit ask) — DROID-SLAM on a 2080ti; fills the pending table rows.
- C2. Renderable metric 4D-GS demo (metric Fig 5) — visualizes "renderable + metric", the Hand3R differentiator.

### D. Framing / write-up
- D1. Narrative: mechanism contribution; concede pose; position vs Hand3R (we are renderable + metric-from-hand, they are not renderable).

## Known infrastructure constraints (carry forward)
- `/work` per-user quota and `/home` (block+inode) are both exhausted; freeing 30 GB did
  not clear `/work` (hard cap / group quota / slow reconcile). Cannot write logs or edit
  repo code on the cluster.
- Working pattern that beats this: stream via `srun`, stage repo + small data on node-local
  `/tmp` (860 G), patch the `/tmp` copy to surface metrics. See `configs/_p3_harness.sh`.
- gsplat JIT-compiles ~70 s/run; GS rasterization ~20–40 s/it; full-data runs do not reach
  a validation inside the 30-min streaming cap — hence few-sequence smoke for verdicts.
- QOS = 1 job/user (serial). gb10 nodes report 114 GB GPU memory (OOM is not the limit;
  speed is).

## Success criteria
- Tier-1 submittable when: converged (a) shows HOT3D object depth well below the frozen
  baseline (B2), the ladder is clean on both datasets, HaWoR comparison exists, and a
  renderable metric demo is in. Pose is conceded in framing throughout.
