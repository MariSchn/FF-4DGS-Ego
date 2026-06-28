# Scale Evaluation — absolute hand placement vs world-space hand models

> **⚠️ NOTE (2026-06-19):** EgoGrasp's **H2O** PA-MPJPE is **18.9** (Dyn-HaMR 16.7, HaWoR 30.8) — all
> better than ours, so any "SOTA-grade / competitive" phrasing below is superseded: we are **not**
> SOTA on H2O hand pose. The correct same-clips baseline is **PA 46.76** (clean
> subject1→4 split). The real contribution is **B1** (the in-scene hand beats a depth-FM 3.5× at the
> hand) + renderable 4D-GS, NOT the hand metrics. See `report/AUTONOMOUS-SESSION-STATUS.md`,
> `report/b1-result.md`, and the honest `report/scale-table.pdf`. Treat the sections below as
> superseded where they claim competitiveness.

**Supervisor directive (2026-06-18):** *"the most important is the scale thing — build a
great evaluation with other models for the scale thing."*

This document defines the **scale** evaluation, places our numbers in it, builds the
comparison table against world-space hand baselines, and states exactly what is needed to
make each comparison airtight. Scale = **absolute / metric hand placement**, NOT articulation.

---

## 1. What "scale" means here, and why PA-MPJPE does not test it

A hand prediction can be decomposed into **(a)** articulation/shape (finger pose) and **(b)**
absolute placement in 3D (where the wrist is, in metric units). The standard metrics isolate
these differently:

| Metric | Alignment before error | What it measures | Tests scale? |
|---|---|---|---|
| **PA-MPJPE** | Procrustes (rotation+**scale**+translation) per frame | pure articulation/shape | **No** — scale divided out |
| **RR-MPJPE** | root (wrist) translation only | shape at true scale, no global placement | partial (size, not placement) |
| **C-MPJPE** | none — camera frame, absolute | absolute placement in camera frame | **Yes** |
| **WRIST** | none — absolute root position error | pure global placement of the hand | **Yes (purest)** |
| **W-MPJPE** (HaWoR) | none — **world** frame (needs camera trajectory) | absolute placement in world | **Yes** |
| **WA-MPJPE** (HaWoR) | one global similarity over the whole **trajectory** | scale-consistent world placement | **Yes (relative scale)** |

**Key identity:** for our method, `C-MPJPE − PA-MPJPE` ≈ the absolute-placement (scale) error,
and `WRIST` is its purest form. PA-MPJPE being good while C-MPJPE lags is *exactly* a scale
gap — which is the thing the supervisor wants evaluated.

---

## 2. Our scale performance (held-out H2O subject4, full eval, 1953 clips)

| metric | 16-joint | 21-joint (airtight) | reads as |
|---|---|---|---|
| **PA-MPJPE** (no scale) | 41.6 | **43.6 mm** | articulation quality (EgoGrasp H2O 18.9) |
| RR-MPJPE | 49.6 | 54.7 | hand size good |
| **C-MPJPE** (absolute) | 55.0 | **58.0 mm** | placement lags pose by ~14 mm |
| **WRIST** (absolute root) | 38.9 | **36.6 mm** | the scale bottleneck |

- Zero-shot baseline (HOT3D-trained head on H2O): C-MPJPE **254.9 mm** → our metric head is **−78%**.
- The **14 mm gap** (C-MPJPE 58 − PA 43.6) is the absolute-placement error our hand-depth
  anchor / metric head is designed to close. **This is the quantity to drive down.**

---

## 3. Comparison — the two-tier story

The field splits into two tiers on scale, and **that split is the contribution framing**:

- **Tier A — world-space METRIC** (HaWoR, Hand3R, Dyn-HaMR): recover absolute scale via metric-depth
  + SLAM, or a 4D scene foundation model. The real competitors.
- **Tier B — camera-frame / single-image** (HaMeR, WiLoR): MANO + a **weak-perspective camera with a
  dummy focal length (5000)**. Absolute translation is a crop-to-frame artifact, **not metric** — they
  are structurally incapable of scale, even with SLAM bolted on.

### 3a. The field's own benchmark (Hand3R, arXiv:2602.03200, HOI4D, **C-MPJPE = our metric**)
This published table *is* the two-tier story, and is directly citable:

| Method | Tier | **C-MPJPE (mm)** ↓ | WA-MPJPE (short / long) |
|---|---|---|---|
| HaMeR + SLAM | B (single-image) | 248.23 | 52.69 / 85.46 |
| WiLoR + SLAM | B (single-image) | 252.24 | 52.91 / 87.51 |
| HaWoR | A (world-space) | 51.77 | 22.54 / 27.40 |
| Hand3R | A (world-space, feedforward) | 42.6 | 38.04 / 56.71 |
| **Ours (FF-4DGS-Ego)** | **A (feedforward, no SLAM)** | **58.0** *(H2O, not HOI4D)* | — |

**Reading:** camera-frame methods sit at ~**250 mm** absolute error even with SLAM; world-space methods
at **42–52 mm**. **Our 58 mm puts us in the world-space-metric tier** — ~4–5× better than single-image
methods — and we get there **feedforward, without SLAM or per-sequence optimization** (HaWoR is offline
multi-stage + DROID-SLAM; only Hand3R shares our feedforward setting, and it has **no public code**).

### 3b. H2O (our home benchmark) — **same data, same metric code, run by us**
| Method | Type | PA-MPJPE | **C-MPJPE (abs)** | WRIST | Source |
|---|---|---|---|---|---|
| **Ours (FF-4DGS-Ego)** | feedforward GS + metric hand head | **43.6** | **58.0** | **36.6** | run by us (airtight, 21-jt) |
| **WiLoR** | single-image, weak-perspective | 47.6 | **615** | 612 | **run by us, H2O subj4 (400 fr / 797 hands)** |
| EgoGrasp | H2O hand SOTA | 18.9 | — | — | reported |

**This is the apples-to-apples scale result.** On the *same* H2O clips with the *same* metric code,
the strong single-image method WiLoR lands at **C-MPJPE 615 mm vs our 58 mm — a 10× gap** — while its
**pose** is fine (PA-MPJPE 47.6, even slightly worse than our 43.6). With its native dummy focal (5000)
the root is ~8.5 m off (C-MPJPE 8530 mm); even re-projected with H2O's true intrinsics it is ~615 mm.
**Single-image methods structurally cannot do metric scale; we can.** (WiLoR ran on a 2080ti node —
py3.10 + chumpy patch + torch2.3/cu118; reproducible recipe in `baseline-run-results.md`.)

**Caveat (must state):** §3a mixes datasets — ours is **H2O**, the Hand3R table is **HOI4D**. It is a
*tier-placement* argument, not a same-data head-to-head. The same-data run (§4) is the airtight version.
HaWoR also reports **HOT3D**: W-MPJPE 33.2 / WA-MPJPE 11.3 — runnable on our HOT3D head for a shared
comparison.

---

## 4. The fair-comparison protocol (how to make this airtight)

The methods differ on two axes that must be controlled:

1. **Frame:** ours predicts **camera-frame** metric hands (no SLAM). HaWoR predicts **world-frame**
   hands (DROID-SLAM camera trajectory). The **shared, fair scale metric is camera-frame C-MPJPE**
   (evaluate HaWoR's per-frame hand *before* its world conversion), or evaluate both with
   **WA-MPJPE** (one global similarity per sequence) which is frame-agnostic.
2. **Data:** run every method on the **same** held-out clips. Two viable shared benchmarks:
   - **H2O subj4** (we already eval here; need to run HaWoR/HaMeR here) — cleanest, GT MANO.
   - **HOT3D** (HaWoR ships `eval_hawor_hot3d.py`; our HOT3D head exists) — their home turf.

**Recommended minimal airtight comparison:** HaMeR + HaWoR on **H2O subj4**, report **C-MPJPE +
WRIST + PA-MPJPE** for all, same clips. That single table is the deliverable that answers
"is our scale competitive?".

---

## 5. Run status — what exists, what blocks it

| Asset | Status |
|---|---|
| HaWoR repo | **cloned** at `~/HaWoR` (incl. `scripts_eval/eval_hawor_hot3d.py`, world W-/WA-MPJPE) |
| HaWoR weights | **incomplete** — only 378 MB, `weights/hawor/checkpoints` empty (download didn't finish) |
| HaMeR weights | present: `team25/models/hamer/hamer.ckpt` |
| Our eval harness | `scripts/eval_cmpjpe.py` (C-/PA-/RR-MPJPE + WRIST, 21-joint) — needs NeoVerse base ckpt |
| **GPU (CORRECTED 2026-06-18)** | cluster is **NOT** Blackwell-only. `sinfo` shows **1080ti (Pascal sm_61, 8/node)** and **2080ti (Turing sm_75, 8/node)** nodes alongside 5060ti/gb10 (Blackwell). CUDA 11.7 supports sm_61–sm_86, so **HaWoR's torch1.13+cu117+DROID-SLAM RUNS on the 2080ti/1080ti nodes.** |

**The hardware blocker was wrong — the comparison is runnable here.** Submit baseline jobs with
`--gpus=2080ti:1` (or `1080ti:1`). No external A100 needed. Remaining friction is plain setup
(env + DROID-SLAM build for HaWoR; trivial for WiLoR/HaMeR) on the quota-constrained `/work`.

---

## 6. Action plan (priority for the scale comparison)

**Baselines to run, ranked (all have public weights; all appear in Hand3R's HOI4D table so our
re-runs are checkable):**
1. **HaWoR** — the Tier-A world-space competitor to beat (reports W/WA-MPJPE). Needs DROID-SLAM build → **Ampere/Hopper GPU**.
2. **WiLoR** — *easiest* Tier-B camera-frame baseline (no SLAM); weak-perspective `cam_t` shows scale failure. Its `detector.pt` is also HaWoR's detector.
3. **HaMeR** — second Tier-B baseline, matches Hand3R's published table.
- **Hand3R** — closest conceptual rival (feedforward+metric+scene) but **no code** → cite its table only.

**Steps:**
1. **[needs GPU access]** Run HaWoR + WiLoR + HaMeR on **H2O subj4** → C-MPJPE/WRIST/PA, same clips,
   camera-frame metric. *The single most important table for the scale claim.*
2. **[doable on Blackwell now?]** WiLoR/HaMeR have no SLAM CUDA ext → attempt a torch-2.7 port on the
   gb10 node for a real camera-frame baseline *before* the A100 lands. Expect ~250 mm C-MPJPE (their
   structural scale failure) — a strong contrast to our 58 mm.
3. **[ours]** Add **WA-MPJPE** (per-sequence global similarity) to `eval_cmpjpe.py` to report ours in
   HaWoR's metric. (Needs the restored NeoVerse ckpt — see disk blocker below.)
4. **[ours]** Drive down the 14 mm scale gap: scale-head for the wrist / higher-res re-pack — the
   C-MPJPE plateau at 55 over 680 steps is capacity/res-bound, not LR-bound.
5. **[fallback]** Until GPU access lands, §3a (Hand3R's HOI4D benchmark + our tier placement) is a
   *strong* interim: it shows we're in the world-space-metric tier, feedforward, no SLAM.

## 7. Blockers needing your decision
- **NeoVerse base checkpoint deleted** (my cleanup error — it was the symlink target, not a duplicate).
  Re-download from HF `Yuppie1204/NeoVerse` (6 GB) is staged but **needs disk space**: requires deleting
  `checkpoints/p3_gtdepth/latest.pt` (6.6 GB, redundant last-state of a *completed* experiment; the two
  `best_*` checkpoints stay). I did **not** auto-delete it. **Approve that delete and I restore the ckpt
  + re-run our eval.** (Not needed for the table above; needed to re-run our own numbers / add WA-MPJPE.)
- **GPU access** (Ampere/Hopper) — the single unblock for running HaWoR (Tier A). Highest leverage.

---

## 8. Honest one-paragraph summary (for the supervisor)

Our hand **pose** trails H2O SOTA (PA-MPJPE 43.6 vs EgoGrasp 18.9). The open question is
**scale** — absolute placement — where C-MPJPE 58 / WRIST 36.6 show a ~14 mm gap vs our own pose
quality. The world-space competitors (HaWoR 51.8, Hand3R 42.6 C-MPJPE) report only on **HOI4D** with a
**SLAM world-frame** protocol, so we cannot yet claim a head-to-head. The fix is not research but
**one Ampere/Hopper GPU**: run HaWoR + HaMeR on the **same** H2O clips with the **same** camera-frame
C-MPJPE, and report one clean table. HaWoR is already cloned; only its weights + a non-Blackwell GPU
are missing.
