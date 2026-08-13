# Long-window results: the camera poses were being used backwards

Notes for the meeting on 2026-08-11.

## The short version

Our world-space evaluation was applying the predicted camera trajectory in the wrong direction.
Fixing it roughly halves the long-window error. Nothing about the model changed, only how we read
its output.

## How we found it

We had a standing anomaly nobody could explain: feeding the model's *predicted* camera poses into
the world lift scored worse than feeding *identity* poses, 202.7 against 140. A trajectory that is
worse than no trajectory is not a weak trajectory, it is a wrong one.

So we compared the predicted poses against ground truth over 1570 clips:

| | as emitted | inverted |
|---|---|---|
| rotation error | 2.149 deg | **0.362 deg** |
| inverted is better in | | 1539 of 1570 clips (98.0%) |
| best-fit translation scale | | median **-0.944** |
| direction error | 142.6 deg | |

A translation scale of minus one is not a small calibration error. The tensor is world-to-camera
even though the code that produces it is labelled "C2W pose (OpenCV)".

## What it costs us end to end

We did not adopt this on the diagnostic. We ran the same checkpoint over the same 60 sequences
twice, once as-is and once inverted, in a single job so nothing else could differ.

| | as emitted | inverted |
|---|---|---|
| **W-MPJPE** | 114.78 | **59.83** |
| WA-MPJPE short | 28.96 | **21.12** |
| WA-MPJPE long | 48.16 | **30.91** |
| C-MPJPE root-relative | 24.73 | 24.73 |
| C-MPJPE absolute | 34.10 | 34.10 |
| solved scene scale | 0.673 | 0.673 |

That is a 60-sequence probe, run twice in one job. The corrected numbers over the **full
157-sequence test set**, detbox v3, 471 segments, are:

| | before | after |
|---|---|---|
| **W-MPJPE** | 113.2 | **70.5** |
| WA-MPJPE short | | 25.8 |
| WA-MPJPE long | 34.2 | 38.8 |
| C-MPJPE absolute | 35.4 | 35.8 |
| C-MPJPE root-relative | 26.5 | 26.6 |

The two camera-frame numbers landing within 0.4 mm of the values already in the paper is the check
that matters: they are computed by a different code path from the previous table and they did not
move, which is what "the fix only touches the world lift" has to mean in practice.

The WA long figure is the one to be careful with. Our previous 34.2 came from the dump-and-score
pipeline rather than from this script, so 38.8 against 34.2 is not a clean before-and-after and
should not be quoted as a regression until both are produced by the same path.

The three rows that do not move are the point. None of them reads a camera pose, so none of them
should change, and none of them does. The fix touches exactly what it should.

There is a second confirmation that is independent of the size of the improvement. With the emitted
poses, substituting the *true* metric scale made the error **worse**, 129.23 against 114.78. That
cannot happen in a correct pipeline. Inverted, the true scale helps, 52.71 against 59.83. What was
happening is that our too-small solved scale was shrinking a backwards trajectory toward identity,
which is also why identity used to win.

## What this changes

**Only our rows move.** No baseline reads this tensor. HaWoR, HaMeR, WiLoR, HaPTIC and Dyn-HaMR use
either ground-truth extrinsics or their own DROID-SLAM, so not one baseline cell changes.

**Camera-frame results are untouched.** C-MPJPE stands exactly where it was.

**The camera-head fine-tune is cancelled.** It was the last training run on the critical path, and
its premise was that the frozen camera head is the long-window bottleneck. The head predicts
rotation to 0.36 degrees. It was never the bottleneck, we were using it backwards. Those GPUs are
now free.

**Our ranking of long-window error sources has to be redone.** We had chaining at about 85 mm,
hand root depth at about 70, scale at about 45. Every one of those was measured on backwards poses,
so none of them can be quoted until it is re-measured.

## Where the remaining long-window error sits

The corrected run also prints its own ceilings, and they say the trajectory is still the story:

    W-MPJPE as measured                 70.5
    with ground-truth per-clip scale    66.1
    with ground-truth velocity          38.5
    re-anchored every 16 frames         22.8

Perfect relative motion would take 70.5 down to 38.5, and re-anchoring every 16 frames takes it to
22.8, which is below the camera-frame error. So the residual is accumulation over the window, not
per-frame hand accuracy.

Our solved scene scale is still 0.688 against a true 1.039, so about a third too small, and unlike
before the fix that now costs us rather than accidentally helping: substituting the true scale
improves W by 4.4 mm. That makes the scale solve the clearest next target, and it is worth
re-measuring the whole ranking before choosing.

## What I would like to decide with you

1. The freed GPU time. My preference is the frozen-versus-unfrozen ablation on the full pool, which
   is what you asked for. The configs are written and validated against the stores: ARCTIC,
   OakInk2, HOT3D and DexYCB, 37.6k cached clips, HOI4D and H2O held out. The two arms differ only
   in 8 unfrozen encoder blocks and their own 1e-5 learning rate.

2. Whether to stop the variable-length runs. They are on two datasets only, which we have agreed to
   move away from. The question they were asked has an answer: variable clip length wins at every
   step we measured, by a margin that fell from 15.9 mm to 4.1 as training went on. Worth keeping
   as the recipe, not worth ten epochs to prove.

3. How much of the paper's long-window discussion to rewrite now versus after the re-measurement.
   The current text blames the frozen camera head, and that sentence is now false.
