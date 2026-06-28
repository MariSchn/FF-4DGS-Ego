# World-space hand placement eval — design (workstream C1)

**Goal.** Produce our rows for Table 5: world-space W-MPJPE / WA-MPJPE on the
HaWoR / Hand3R protocol, so we can sit next to HaWoR (HOI4D 51.8 / 22.5–27.4;
HOT3D 33.2 / 11.3) and Hand3R (HOI4D 42.6 / 38.0–56.7). This is the comparison
Cyrus explicitly asked for. It is a build, not a run (see ETA in the roadmap spec).

## What we have (reuse, do not rebuild)

- Model predicts, per 16-frame clip: `preds["hand_joints"]` ([1,S,64] MANO params),
  `preds["rendered_extrinsics"]` ([S,4,4] cam->world, clip-local), `preds["gs_depth"]`,
  `preds["rendered_intrinsics"]`.
- `compute_joints_from_batch(params, mano, dev)` -> `[B,S,2,16,3]` metric cam-frame joints.
- `solve_metric_scale(pred_joints, gs_depth, has_hand, cam_intr, clamp=(0.1,10))` -> scalar `s`.
- Cam->world lift (mirrors `reconstruct_4dgs.py` / `export_h2o_figure.py`):
  `j_world = (c2w[:3,:3] @ j_cam_metric.T).T + c2w[:3,3]*s`  (metric world frame).
- `hand_metrics.compute_similarity_transform(S1, S2)` -> Procrustes (sR+t) of S1 onto S2.
- GT world hands: `<seq>/hand_data/mano_hand_pose_trajectory.jsonl` (world frame), with
  `cam_extrinsics_cache.pt` and `resolve_extrinsics_convention()` for the w2c/c2w check.

## The one new piece: clip chaining

World-space methods (HaWoR) get a single global camera trajectory from SLAM. We get a
*clip-local* predicted trajectory per 16-frame clip (`rendered_extrinsics` is expressed in
that clip's own frame). To form one 128-frame world trajectory we **chain** consecutive
clips:

1. Run clips with an **overlap** of `O` frames (e.g. 16-frame clips, stride 8 -> O=8).
2. Lift each clip's hands to its clip-local world frame.
3. For consecutive clips A, B sharing O overlap frames, estimate the rigid+scale transform
   `T` that maps B's overlap joints onto A's (via `compute_similarity_transform` on the
   overlap region), and apply `T` to all of B. Compose along the segment so every clip
   lands in clip-0's frame = the global frame.
4. Result: `[128, 2, 16, 3]` predicted world joints for the segment.

This is the principled analogue of SLAM trajectory stitching using the model's own
per-clip geometry — exactly the "chain per-clip hands through the predicted camera
trajectory" step noted in the comparison table.

## Metrics (HaWoR/Hand3R convention)

- **W-MPJPE**: mean per-joint L2 between predicted and GT *world* joints over the whole
  128-frame segment, **no alignment** (absolute world placement). Averaged over segments.
- **WA-MPJPE**: per-segment **similarity-aligned** MPJPE — align the predicted world
  trajectory onto GT once per window with `compute_similarity_transform`, then MPJPE.
  HaWoR reports two windows; we mirror as:
  - **short**: align per non-overlapping sub-window of `w_short` frames (e.g. 16),
  - **long**: align once over the full 128-frame segment.
  (Validate `w_short`/`w_long` against the HaWoR paper before quoting; the harness takes
  them as flags.)

## On-cluster validation checklist (must do before quoting numbers)

1. **GT convention**: confirm `mano_hand_pose_trajectory.jsonl` joints are world-frame and
   match `cam_extrinsics_cache.pt` via `resolve_extrinsics_convention` (err < few mm).
2. **Scale**: per-clip `solve_metric_scale` vs one scale per segment — pick the stabler;
   log the spread.
3. **Chaining drift**: sanity-check by chaining GT clip-local frames and verifying recovery
   of the GT global trajectory (should be ~0 error); measure predicted-chain drift over 128.
4. **HaWoR window defs**: confirm `w_short`/`w_long` against their eval code.
5. Run on the same HOI4D / HOT3D sequences HaWoR/Hand3R report, not a different split.

## Files

- `scripts/world_space_metrics.py` — pure, testable: W-MPJPE, WA-MPJPE, overlap chaining.
- `scripts/eval_world_space.py` — harness: load ckpt, run clips over 128-frame segments,
  load GT world joints, call the metrics, write a JSON report.

## Output

A JSON + a printed table row per dataset:
`Ours  W-MPJPE=__  WA-MPJPE(short/long)=__/__  (HOI4D | HOT3D)` -> fills Table 5.
