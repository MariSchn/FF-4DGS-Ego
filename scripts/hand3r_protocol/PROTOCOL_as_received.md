# Protocol specification

## Dataset split

The split unit is `(H, C, S)` = `(subject, category, scene)`. All clips/takes
from one such group stay in the same split.

- Seed: 42.
- Held-out ratio: 0.25, stratified by category.
- Training: 143 scenes, 1,271 clip directories.
- Full held-out set: 42 scenes, 411 clip directories.
- Paper subset (`val2`): 300 clips covering 39 held-out scenes.
- Hand3R aggregate: 298 of these clips pass the evaluator's validity checks.

The phrase “HOI4D test set” in Table II refers to this custom scene-disjoint
held-out subset, not the official hidden HOI4D test-server split.

## Input boxes

Table II uses GT-guided hand boxes generated from the annotated HOI4D 2D hand
keypoints. The boxes are transformed with the same resize and center crop as
the RGB input. The paper results do not use a learned hand detector.

The HaMeR-SLAM, WiLoR-SLAM, and HaWoR comparison wrappers use the same
GT-derived per-frame crop and handedness information.

## Joint set

The evaluator uses the 21 joints returned by manopth:

1. wrist;
2. index MCP, PIP, DIP, tip;
3. middle MCP, PIP, DIP, tip;
4. little MCP, PIP, DIP, tip;
5. ring MCP, PIP, DIP, tip;
6. thumb MCP, PIP, DIP, tip.

The five tips are vertex-derived MANO fingertips.

## Hand set

One 21-joint hand contributes per valid frame:

- select the annotated right hand when present;
- otherwise select the annotated left hand;
- if both are present, only the right hand contributes.

Consequently, Table II is a right-preferred single-hand-per-frame evaluation,
not an average over every visible hand instance.

## C-MPJPE

For clip `s`, after validity filtering:

```text
C-MPJPE_s = mean over frames t and joints j of
            ||J_pred_cam[t,j] - J_gt_cam[t,j]||_2
```

It is reported in millimeters. It is an **absolute** camera-frame metric:

- no wrist/root subtraction;
- no translation or rotation alignment;
- no scale alignment;
- no Procrustes alignment.

The final number is the unweighted mean of clip-level C-MPJPE values. It is not
weighted by the number of frames in each clip.

## PA-MPJPE

Each frame is independently aligned with one similarity transform (scale,
rotation, translation) estimated from all 21 joints, followed by MPJPE.

## World-space metrics

Metrics are computed independently on non-overlapping 30-frame or 100-frame
chunks. A tail shorter than 10 frames is skipped.

- `WA-MPJPE`: estimate one similarity transform from every frame and all 21
  joints in the chunk; apply it to the whole chunk; then compute MPJPE.
- `W-MPJPE`: estimate one rigid transform from the first two frames and all 21
  joints; keep scale fixed; apply it to the whole chunk; then compute MPJPE.
- `MPJPE`: direct world-coordinate MPJPE without alignment.
- `MRE`: direct mean error of joint 0 (wrist/root).
- `RTE`: after rigidly aligning the predicted wrist trajectory to the GT wrist
  trajectory, divide the sum of per-frame translation errors by the sum of GT
  inter-frame displacements and report a percentage.

Chunk scores are averaged equally within a clip. Clip scores are then averaged
equally across the evaluation set.

## Validity filtering

- Frames without a usable annotated target hand are invalid.
- Frames marked invalid by the prediction adapter are invalid.
- A clip with fewer than 10 valid frames is skipped.
- If more than half of the otherwise-valid frames contain non-finite predicted
  joints, the clip is skipped.
- Otherwise, non-finite predicted frames are removed before scoring.

These rules reproduce the current paper evaluator. For a new shared-detector
benchmark, dropping missed detections can bias the result; define and disclose
a common missing-detection policy for every method.

## Aggregation caveat

The original implementation filters invalid frames before splitting the valid
sequence into metric chunks. Therefore gaps between valid frames are compressed
for scoring. The reference scorer preserves this behavior for compatibility.
