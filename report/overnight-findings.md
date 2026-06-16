# Overnight findings (2026-06-17) — scene-metric thesis FALSIFIED; reframe to hand placement

**Run autonomously while the user was away.** All numbers below are from real cluster jobs
(scratch joblogs); see job IDs. This is the honest record + the strategic reframe.

---

## The complete results table

| Metric | baseline (no anchor) | anchor 0.1 (a01) | anchor 1.0 (warm-start) |
|---|---|---|---|
| Hand placement MPJPE (9-seq robust) | 81.4 mm | ~55 mm (train-val) | **52.9 mm (−35%)** |
| Hand PA-MPJPE | — | 7.8 mm | 7.9 mm |
| Hand depth @ hand (residual) | — | — | **4.5 cm** |
| Scene render PSNR (eval_gs_head, offline) | 32.55 dB | **32.81 dB (recovered)** | 26.78 dB (**−7.2 dB degraded**) |
| **Scene depth on OBJECTS (B2, non-circular) median err** | **61.9 cm** | **134.7 cm (2× worse)** | not run |
| B2 δ<10cm | 7.1% | 1.5% | — |
| Scale CV on objects (non-circular) | 27.6% | 12.8% | — |

Jobs: E3-a01=99393, B2-a01=99394, control E0=99395 (running). a01 model = anchor-0.1 retrain
(`checkpoints/p2_warmstart_a01/best_mpjpe.pt`), trained ~step 80 (4.4h) to recover scene quality.

---

## What is CONFIRMED

1. **Metric hand placement works (the surviving contribution).** Anchoring to the hand's metric
   scale improves absolute MPJPE −35% (52.9 vs 81.4mm, robust across 9 seqs) and makes the hand
   depth metric (4.5cm at the hand). PA-MPJPE preserved (~7.9mm).
2. **The anchor weight trades hand-metricity against scene render quality.** Anchor 1.0 = best hand
   (52.9mm) but degrades render −7.2dB (26.78 vs ~34dB). Anchor 0.1 = recovers render (32.81 ≈
   32.55 baseline) at a small hand cost (~55mm).

## What is FALSIFIED (the negative result)

3. **The "metric SCENE" thesis is dead.** B2 (non-circular, GT object depth, hand region excluded):
   the anchor makes object-region depth **WORSE** — a01 = 134.7cm vs baseline 61.9cm. The anchor
   distorts scene depth to satisfy the hand; it does not propagate metric scale to objects.
4. **The earlier "scene becomes metric" signal (CV 25.5→5.9%) was a circular artifact.** It was
   measured AT the hand, where the anchor trains. Non-circularly (objects), scale CV is ~13–28% for
   both anchored and baseline — no improvement.
5. **Root cause is fundamental, not tunable.** Even the baseline (frozen backbone, no anchor) has
   ~62cm object-depth error — the monocular feedforward backbone is simply not metric-accurate on
   the scene. No anchor weight can make a 62cm-accurate depth metric on objects; the anchor only
   adds distortion. So this is not a "tune the weight" fix — the scene-metric claim cannot hold with
   this backbone.

---

## Strategic reframe

**This is no longer a "Metric Hands, Metric Scene" paper.** The defensible contribution is
**feedforward monocular egocentric METRIC HAND PLACEMENT** via hand-scale anchoring, in a GS
framework — *not* a metric scene reconstruction.

Consequences:
- We lose the scene-metric differentiator vs Hand3R/HaWoR. We are now on their turf (hand pose),
  minus their scene capability. The paper's viability hinges entirely on whether our **placement is
  SOTA-competitive externally** (HaWoR/Hand3R on a common benchmark) — the E4/E5 comparisons.
- Honest tier read: drops from "potential main-track" toward **workshop**, unless the external
  placement numbers are clearly strong. The B2 negative is good science (caught a wrong claim) but
  it removes the headline.
- Report the scene-metric negative honestly as a characterized limitation/analysis.

## Next priorities (revised)

1. **Control E0 (running)** — placement causality is now the primary internal validation.
2. **External placement comparison** (HaWoR on HOT3D; Hand3R on HOI4D) — now make-or-break.
3. Offline-eval the a01 hand MPJPE (rigor; currently train-val ~55mm).
4. E1 UniDepth (vs metric-depth FM at the hand) — torch cu128 reinstalled; re-test on a 5060ti node
   after the control frees the queue. Still relevant for "hand depth without a depth FM".
5. Rewrite the paper around hand placement; keep the scene-metric negative as analysis.

---

## ⚠️ Hardware obstacle for the (now make-or-break) external comparison

The placement claim now lives or dies on an external SOTA comparison (HaWoR / Hand3R). But the cluster
is **Blackwell-only**: gb10 nodes = GB10 (Grace-Blackwell), x86 jobs = RTX 5060 Ti (sm_120). HaWoR ships
**torch 1.13 + cu117** (supports only ≤ sm_86 Ampere) and **DROID-SLAM** CUDA extensions written for that
old stack — neither runs on Blackwell without a torch→2.7/cu128 upgrade + a DROID-SLAM port, which is a
multi-day, uncertain effort. (Same root cause as the UniDepth E1 crash: cu121 torch had no sm_120 kernel;
fixed by cu128. UniDepth V2 has no custom CUDA build, so cu128 alone fixes it; DROID-SLAM does, so it's
harder.)

**Implication:** the external comparison is not a quick win on this hardware. Options for the morning
decision: (a) port HaWoR to cu128 + rebuild DROID (multi-day, risky); (b) compare on numbers reported in
the HaWoR/Hand3R papers for shared benchmarks instead of re-running them; (c) get GPU access with an
Ampere/Hopper node elsewhere; (d) scope to a workshop where the internal baseline + the honest negative
suffice. This is the key strategic call.
