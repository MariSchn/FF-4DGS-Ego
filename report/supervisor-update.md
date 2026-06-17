# Progress update since the report — and a request for a GPU

*(Draft to send to the supervisor. Honest status + concrete ask.)*

---

Hi [supervisor],

Quick update on what I've done since the report/poster, where it landed, and one thing that would
unblock the next step.

## TL;DR
The report ended by flagging **region-masked / object-region evaluation as "the next step toward a
conclusive result"** (§4.4, §6). I built exactly that, and pushed the **metric** angle hard
(absolute hand *placement* + whether the *scene* itself becomes metric). The honest outcome: **the
hand becomes metric and placement improves a lot, but the scene does NOT become metric on objects** —
and I now know precisely why, and what it would take to fix. Good news is it's a clean result and a
clear path; the main blocker to a main-track-grade comparison is **hardware**, not research.

## What I built since the report
1. **Absolute, metric hand placement** (the report only had PA-MPJPE = relative). I added an
   absolute-3D placement loss + a hand-depth geometric anchor that ties the predicted scene depth to
   the metric MANO hand. Result on HOT3D (robust 9-seq split):
   **absolute MPJPE 81.4mm → 52.9mm (−35%)**, hand depth **4.5cm** at the hand, PA-MPJPE preserved
   (~7.9mm). So the hand is now placed metrically, not just up-to-scale.
2. **A non-circular scene-metric evaluation** — the "next step" from the report. I render
   **ground-truth object depth** from the HOT3D object meshes + per-frame 6-DoF poses, exclude the
   hand region, and compare the predicted scene depth against it in *object* regions (validated: at
   hand–object contact my GT render agrees with the metric hand to 1.6–4.2cm).

## The honest key finding
- **The scene does not become metric on objects.** With the anchor, object-region depth error is
  *worse*, not better: **134.7cm vs 61.9cm** for the no-anchor baseline. The earlier "scene becomes
  metric" signal (scale CV 25→6%) was a **circular artifact** — it was measured *at the hand*, where
  the anchor trains; non-circularly (objects) it vanishes.
- **Root cause is fundamental:** even the no-anchor frozen backbone is ~62cm-inaccurate on scene
  depth, and the hand can only fix the global *scale*, not the *relative* structure. So no anchor
  weight makes the scene metric — the anchor only adds distortion.
- **Scene render quality is preserved** at a gentle anchor (PSNR within ~0.3dB of baseline),
  consistent with the report's PSNR gain; a strong anchor degrades it (−7dB).
- I also ran the **causal ablation** (control with the new losses zeroed): fine-tuning alone recovers
  most of the placement gain, so the abs-3D loss is a modest contributor (same-split confirmation
  running).

This is a *clean, characterized negative* on the scene-metric claim — the kind of thing better found
now than after submission — plus a *solid positive* on metric hand placement.

## Where this leaves us, and the path to main-track
The defensible contribution is **feedforward monocular egocentric *metric hand placement*, anchored to
the hand, no depth-foundation-model needed.** To make it main-track rather than workshop, two things:

1. **(Running now) "Unfreeze" experiment** — the real shot at a *true* metric scene: instead of a
   frozen backbone, I let the encoder train so the egocentric video's reconstruction signal + the hand
   anchor can *teach* the network metric depth. If object-region depth drops below the 62cm baseline,
   the scene-metric headline is resurrected and this becomes a strong paper.
2. **External SOTA comparison (the make-or-break)** — we currently only beat our own baseline.
   Reviewers will require comparison to **HaWoR** and **Hand3R** on a shared benchmark. HaWoR's repo
   *already ships a HOT3D eval* with sequences that overlap ours, so this is "run their eval," not
   build one.

## The ask: a GPU (this is the bottleneck)
The external comparison is blocked **only by hardware**. Our cluster is **Blackwell-only** (the gb10
nodes and the RTX 5060 Ti partition, sm_120). The SOTA stacks don't run on Blackwell:
- **HaWoR** needs torch 1.13 / cu117 (≤ Ampere) + DROID-SLAM;
- **UniDepth** (metric-depth FM baseline) needs torch ≤2.6, but Blackwell requires ≥2.7 — a hard conflict.

They all run fine on **Ampere/Hopper**. **A single H100 (or A100) for a few days would unblock the
HaWoR-on-HOT3D comparison — the single most important number for a main-track submission** — and would
also let me run the metric-depth-FM baseline. If the unfreeze experiment also lands, we'd have a real
main-track story (metric placement + a true metric-scene result + external SOTA).

Could you grant access to an H100/A100? That's the one thing standing between the current honest
results and a main-track-grade evaluation.

Thanks,
Dario

---

### Backup numbers (if he asks)
| | hand MPJPE (9-seq) | hand depth @ hand | scene PSNR | object depth err (non-circ) |
|---|---|---|---|---|
| baseline (no anchor) | 81.4mm | — | 32.55 dB | 61.9 cm |
| anchor 0.1 (scene-preserving) | ~55mm | — | 32.81 dB | 134.7 cm |
| anchor 1.0 (best hand) | 52.9mm | 4.5 cm | 26.78 dB (−7.2) | — |
| control (no abs-3D) | ~55mm | — | — | — |
- Falsified: "metric scene on objects." Confirmed: metric hand placement; scene render preserved at low anchor.
- Full detail: `report/overnight-findings.md`, `report/external-comparison-plan.md`.
