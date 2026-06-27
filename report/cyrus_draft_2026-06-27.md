# Draft message to Cyrus — 2026-06-27 (HOI4D contact-anchor results)

> DRAFT for your review — not sent. This delivers the HOI4D result Cyrus is waiting for
> ("Next is HOI4D... then I'll send the next results"). Two decisions before sending:
> (1) tone on the "global hand-scene scaling is refuted" finding — drafted candidly below;
> (2) whether to attach the H2O Figure 5 render (cyrus_figures/fig5_h2o_hero.png).

---

Hey Cyrus — HOI4D contact-anchor results, as promised. Short version: the anchor is a
necessary **safety** mechanism, but on HOI4D it doesn't *add* placement, and the reason is
instructive.

**The HOI4D numbers (camera-frame, GT-contact gated, 11 seqs):**

| arm | gate | C-abs (mm) | C-rr (mm) |
|-----|------|-----------|-----------|
| control (no anchor) | — | 105.8 | 29.7 |
| wide-band anchor (old 50 cm gate) | proxy | 150.9 | — |
| contact anchor (true-contact gate) | oracle | 111.4 | 29.8 |

Two things fall out:

1. **The wide band *hurts* (+45 mm).** The old gate fired wherever |d_scene − wrist| < 50 cm,
   which includes free space, and there it pulls the wrist toward a biased scene depth.
   Gating by *true contact* removes that harm (back to ≈control). So contact-gating is a real
   fix — but a fix that *protects* the head, not one that improves on a good head.

2. **The anchor doesn't add on HOI4D because the head is already good here** (C-abs 105.8).
   The true-contact gate only fires ~10 % of frames with ~25 mm corrections — there isn't
   much placement error left at contact for it to remove. The anchor needs a *bad* head and a
   *good* reference at the same time; HOI4D gives a good reference but the head is already
   strong, and HOT3D was the opposite (good head, bad reference). Neither dataset hits the
   sweet spot the re-anchor oracle promised.

**Why the coupling is subtler than "hand rescales the scene" — the scale-source ablation.**
I tested the headline directly: score |s·gs_depth − GT dense depth| over non-hand pixels for
different scale sources. Using the hand to set the *global* scene scale makes the scene **2×
worse** (s_hand ≈ 0.73 vs oracle 1.02; non-hand residual 30 cm vs 4 cm). The mechanism is
clean: the frozen feedforward backbone doesn't reconstruct the thin foreground hand, so the
Gaussian depth *sampled at the projected hand* reads the background behind it (~37 % too far),
so z_hand / gs_depth_at_hand is a biased scale even though the scene is globally near-metric.

But here's the part that saves the idea: when I stratify those per-joint ratios by an
*independent* GT contact signal, the hand recovers the scale **almost perfectly at contact**
(s_contact ≈ 1.005 ≈ oracle) and is biased only in free space (0.804). So the hand-scene metric
coupling is **real, but contact-local** — exactly where the anchor is allowed to fire. That's
the honest scope of the coupling: a contact-time scale signal, not a global rescaler.

**What's solidly banked across both datasets:** root-relative articulation is competitive —
C-rr **29.7 on HOI4D** and ~33 on HOT3D, both under Hand3R's 42.6. The hand *shape* is not the
problem on either set.

**On your engineering ask (root-translation scale head + larger-weight supervision):** the
larger-weight *absolute* supervision half is the kp3d_abs result I sent — it's the lever that
actually moved W (308 → 250, C-abs 115 → 53 on HOT3D). The dedicated root-translation/scale
head is the natural next build on top of that; I'll scope it next.

Net read: "metric hand placement + a contact-local metric coupling + competitive articulation,
with absolute-keypoint supervision as the lever" is the honest and defensible story; the naive
"hand rescales the whole scene" version is refuted by the ablation (and we can show *why*,
which is itself a clean result). Happy to talk through framing.
