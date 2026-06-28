Hey Chenyangguang, update on the last few days since the scale-table — and this connects directly to
your question about drawing from the previous world-space models.

## I ran the actual world-space hand eval vs Hand3R, and it turned into a clean diagnosis

On the table I only had HaWoR/Hand3R as reported rows. I built and ran our own world-space hand-placement
eval on the HOI4D dense-depth sequences (the same benchmark Hand3R reports on), so we now have real
numbers, not just theirs:

- **W-MPJPE = 353 mm** (first-window rigid align, then absolute drift over the segment) vs **Hand3R 126**.
  So on absolute world placement we are ~2.8× worse — I want to be upfront about that.
- **But WA-MPJPE = 21 mm (16-frame) / 47 mm (128-frame)** — this re-solves a similarity per window, so it
  measures how good the hand *shape and local trajectory* are independent of global placement. ~2-5 cm is
  strong, and it's the metric that reflects what we actually do well.
- **C-MPJPE = 93 mm root-relative / 366 mm absolute** (Hand3R 42.6). Same story: relative is decent,
  absolute is off.

## The useful part: I ruled out the two obvious explanations for the W gap

1. **It is not a scale problem.** I tested three sequence-level metric scales (per-clip, per-sequence
   median, and a pooled global solve over all hand/scene depth correspondences). W is essentially flat:
   353 / 350 / 352. So a better scale estimator — including the learned scale head — will not move W.

2. **It is not a camera-pose problem.** I ran an oracle: take the predicted hand in camera frame and drop
   it into the global world using the **GT camera extrinsics** (perfect poses, no chaining). W stays at
   **364 mm ≈ our 353**. So even with perfect SLAM-grade poses, W doesn't improve. I cross-checked this
   against the camera-frame C-MPJPE-absolute (366 mm, computed with no extrinsics at all) — two
   independent paths agree, so it's not a bug.

**So the bottleneck is the hand head's absolute per-frame depth — the hand-root depth-from-camera.** The
local shape transfers (WA 21 mm), the absolute depth doesn't, and part of that is a HOT3D→HOI4D domain
gap (we train on HOT3D, eval on HOI4D). That's a much sharper picture than "we're worse on W."

## On your question — yes, the previous world-space models are exactly the right place to look

I went through HaWoR, Hand3R, Dyn-HaMR (egocentric hands) and WHAM, SLAHMR, GLAMR, WHAC (world-grounded
human motion). The key realization: **most of their machinery is spent recovering two things we already
have for free** — a globally-consistent camera trajectory (they use DROID-SLAM) and absolute metric
scale (they fit it against SLAM or a depth model). We get the camera from the eval and **the metric scale
from the in-scene hand directly.** So the SLAM-heavy half of these methods — HaWoR's and SLAHMR's
DROID-SLAM, which is also exactly what I can't run on the Blackwell cards — is the half we don't need.

What they have that we don't is **how they place the subject over time**, and that maps onto our exact
failure (noisy per-frame absolute depth that drifts when chained):

- **Velocity / trajectory decoder instead of per-frame independent depth.** WHAM (egocentric-velocity
  RNN), WHAC (MotionVelocimeter), GLAMR (egocentric trajectory predictor) all predict inter-frame
  *velocity* and integrate, rather than regressing each frame's absolute depth on its own. That converts
  the high-variance quantity that's killing our W into a smooth, temporally-coupled one. This is the most
  direct fix, and it's a small head on top of our frozen backbone.
- **A learned temporal hand-motion prior** (Dyn-HaMR's HMP, trained on ARCTIC; WHAM/GLAMR use AMASS) to
  penalize implausible frame-to-frame jumps. Regularizes the drift directly; HMP is hand-specific and
  reusable.
- **Test-time temporal smoothing** (SLAHMR/Dyn-HaMR, stripped down). Because we already have metric scale
  and good local shape, we can smooth *only the root-depth track* with scale held fixed — a cheap
  inference-time post-process, no retraining, and probably the fastest measurable W win.
- **Hand-object contact as an absolute anchor.** None of the hand models use it, and HOI4D is object-rich
  — at contact frames we can pin the hand depth to the object/scene geometry and reset drift. Genuinely
  novel vs. all three hand baselines.

The clean framing this gives us: **we already solved the camera + scale problem these methods work hardest
on (via the metric hand); what we need to borrow is their temporal trajectory modelling, not their SLAM.**
That keeps us feed-forward and SLAM-free, which is also our differentiator vs HaWoR (offline + DROID-SLAM).

## Where I'm headed + the same bottleneck

Immediate: a test-time depth-track smoothing pass (cheap, no retrain) to confirm the drift diagnosis, then
a small velocity/trajectory head + temporal prior on the frozen backbone. None of it needs a big GPU.

The thing that still needs the GPU is the converged scene-metric run we discussed — the partial-unfreeze +
GT-object-depth training. On the single consumer card it OOMs before it converges, so the headline object-
depth number (does the scene drop below the 62 cm frozen baseline?) is still stuck. That remains the one
result that most moves us from workshop to main-track, and it's gated on an A100/H100. Any progress on that
front would help a lot.

Happy to walk through the eval and the world-space-model comparison whenever.
