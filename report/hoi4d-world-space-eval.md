# HOI4D world-space hand-placement eval (2026-06-22)

**Setup.** `exp_p3_scalehead` = frozen NeoVerse/WorldMirror backbone + warm-started HaMeR-style hand
head + in-scene-hand metric-scale anchor (`solve_metric_scale`, closed-form at inference). 11 HOI4D
dense-depth sequences (right hand, ZY20210800001), preprocessed at res 224. Protocol
`segment_len=128, clip_len=16, stride=8, wa_short=16`, **`--max_segs 1`** (one 128-frame segment per
sequence). Ran on a gb10 (GH200) node. Greedy and global (bundle) chaining give ~identical numbers.

## Results (per sequence, mm)

| sequence | W-MPJPE | WA-MPJPE short (16f) | WA-MPJPE long (128f) |
|---|---:|---:|---:|
| C12/N11 | 442.7 | 22.1 | 37.3 |
| C12/N26 | 558.2 | 20.3 | 35.2 |
| C13/N11 | 207.3 | 18.2 | 41.5 |
| C13/N12 | 208.5 | 15.6 | 49.2 |
| C14/N15 | 378.6 | 22.0 | 37.7 |
| C14/N17 | 352.1 | 20.0 | 35.1 |
| C1/N19  | 150.2 | 11.5 | 20.7 |
| C2/N11  | 750.0 | 30.5 | 93.6 |
| C3/N01  | 217.1 | 29.6 | 53.7 |
| C5/N15  | 174.9 | 13.7 | 24.9 |
| C7/N11  | 444.7 | 28.3 | 84.1 |
| **mean (11)** | **353.1** | **21.1** | **46.6** |

## vs Hand3R (HOI4D world-space)

Hand3R reports **100f-W-MPJPE 125.81 mm**, **C-MPJPE 42.6 mm** (camera-frame, not evaluated here).

- **W-MPJPE: ours 353 mm vs Hand3R 126 mm — we are ~2.8x WORSE on absolute world placement.**
  (Ours is a 128f single segment vs their 100f window; the window difference does not explain a 2.8x
  gap.)
- We have no apples-to-apples WA number from Hand3R, but our WA is low in absolute terms (21/47 mm).

## Reading — the result is a clear diagnosis, not a win

- **Local hand shape is excellent (WA 21 mm short / 47 mm long).** WA re-solves a Sim3 per window, so
  it measures how well the predicted hand *shape/trajectory* matches GT independent of any global
  scale/placement error. ~2-5 cm is strong.
- **Absolute world placement is poor (W 353 mm).** W aligns once (first valid window, rigid) then
  measures absolute drift over the segment. The large gap between W (353) and WA (47) means the
  **per-clip metric scale from the in-scene hand anchor has high variance**, so the chained absolute
  world trajectory drifts. WA is immune (per-window realign); W exposes it.
- **So the hand head is good; the bottleneck is absolute scale/placement consistency.** The in-scene
  hand anchor recovers good per-clip geometry but a noisy *scale*, which compounds along the
  trajectory.

## Update 2026-06-22 — sequence-level scale REFUTED; C-MPJPE added

Added a 3-scale eval (`report/hoi4d_world_eval_2026-06-22_3scale.json`, 11 seqs) testing whether a
sequence-level metric scale fixes the W drift:

| scale | W-MPJPE | WA short/long | C-MPJPE rootrel/abs |
|---|---:|---:|---:|
| per-clip (per-frame heuristic) | **353.1** | 21.1 / 46.6 | **93.0 / 366.3** |
| per-seq **median** of clip scales | 350.0 | — | — |
| per-seq **pooled** (1 median over all z/depth corrs) | 352.3 | 21.0 / 46.7 | — |

(per-clip scale std = 0.227 on mean ~0.6; per-seq scale 0.33–0.85 across sequences.) Hand3R: W 125.81,
C-MPJPE 42.6.

**The "per-clip scale variance → drift" hypothesis is refuted.** W is flat (353→350→352) across all
three scales on every sequence — **W is insensitive to the global metric scale.** The drift is **camera-
pose chaining / global-consistency**, not scale: WA (per-window realign) = 21 mm *locally* vs W (global)
= 353 mm. Hand3R's low W almost certainly comes from **SLAM-based globally-consistent camera poses
(DROID-SLAM)**; we chain per-clip backbone poses from 16-frame clips with no global solve, and greedy ≈
global(bundle) chaining (353 ≈ 353) cannot fix systematically-off per-clip poses. **C-MPJPE root-rel 93
vs WA 21** further says the predicted hand carries a real camera-frame orientation/scale error.

**Consequences:** (i) a learned feedforward ScaleHead would also not move W (W is scale-insensitive);
(ii) we hypothesized the real W lever was a **global camera trajectory** (SLAM / global BA). The
`--oracle_cam` diagnostic below **refutes that too.**

## Update 2026-06-22b — oracle (GT-pose) ceiling REFUTES the pose hypothesis; W is hand-head-limited

Ran `--oracle_cam`: place the predicted **camera-frame** hand into the **global GT world via GT
extrinsics** — perfect camera poses, **no scale, no chaining, no alignment** — and measure world error
directly (the W ceiling if poses were perfect). Result over the same 11 seqs:

| | W-MPJPE (mm) |
|---|---:|
| ours, chained (per-clip backbone poses) | 353.1 |
| **oracle (GT extrinsics, perfect poses)** | **363.7** |
| C-MPJPE_abs (camera-frame, no extrinsics at all) | 366.3 |
| Hand3R | 125.8 |

Per-seq oracle: 285–421 mm. **Perfect camera poses do NOT lower W** (363.7 ≈ chained 353.1). The
pose/chaining hypothesis is **refuted** — SLAM/global-BA would not close the gap. And the oracle
(363.7) ≈ the independently-computed **C-MPJPE_abs (366.3)**, which uses *no extrinsics whatsoever* —
two independent paths agree, ruling out an extrinsics-convention bug. **W is capped by the hand head's
absolute camera-frame metric placement**, dominated by **root depth-from-camera**: root-relative
C-MPJPE is 93 mm and WA (per-window Sim3) is 21 mm, but absolute is ~365 mm — the hand's *shape and
local trajectory are excellent; its absolute metric depth in the camera frame is what's wrong.*

**Net of both refutations:** neither sequence-level scale nor global camera poses move W. The W gap vs
Hand3R is a **monocular absolute-hand-depth** gap (a known-hard problem; Hand3R closes it with
joint SLAM+hand+scene optimization over the whole sequence). Levers that *could* move W: stronger
absolute-3D supervision / larger hand-head capacity (partial unfreeze), or multi-frame depth
consistency — **not** scale and **not** poses. The unfreeze-GS registration loss targets
hand↔scene **coupling / C-MPJPE consistency**, not W. **WA-MPJPE (21/47 mm) is where our metric
coupling is genuinely strong and is the honest headline.**

## Reproduction notes

- Eval finally runs via the `eth-cluster` alias only (direct student-cluster1/2 hostnames hang at
  auth); launcher staged by base64 + `split -b 90000` chunks over `clssh` (scp hangs); eval modules
  (`eval_world_space.py`, `world_space_metrics.py`) are not on cluster `/home`, embedded in the
  launcher. Generator `/tmp/gen_combined_launcher.py`, stage+fire `/tmp/run_eval.sh`. Raw results:
  `report/hoi4d_world_eval_2026-06-22_wfix.json`.
- **W-MPJPE metric fixes applied** (`scripts/eval_world_space.py`, `scripts/world_space_metrics.py`):
  (1) finite-safe `w_mpjpe` (drop inf/nan); (2) **rigid** (no-scale) first-window gauge — matches
  Hand3R "first-window align, no scale" and avoids a Sim3 scale blow-up; (3) **anchor on the first
  window with >=3 valid joints** (the hand enters after frame 0 in 9/11 seqs, so a fixed frames-0:16
  window was empty -> the earlier W=NaN). With these, W is finite for all 11.
