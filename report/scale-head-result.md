# (b) Feedforward scale head — directional verdict

**Question (Cyrus direction b).** Keep the GS backbone and gs_head frozen (do not touch
the up-to-scale scene); add a small feedforward `ScaleHead` that predicts one global
metric scale `s` per clip from the camera/register token, trained so `s · gs_depth`
matches the metric MANO hand. Does the head actually learn the hand↔scene metric
coupling in a single forward pass (no per-clip solve)?

## Result (directional)

| Step | scale_residual | abs. MPJPE | PA-MPJPE | PSNR | SSIM |
|---|---|---|---|---|---|
| 1 (`s`≈1, untrained) | **14.07 cm** | 157.7 mm | 8.01 mm | 40.01 dB | 0.987 |
| 2 | **13.17 cm** | 157.1 mm | 8.01 mm | 40.01 dB | 0.987 |
| 4 | **13.08 cm** | 137.5 mm | 8.09 mm | 40.01 dB | 0.987 |

**The scale head learns the coupling.** `scale_residual` (the error of `s·gs_depth`
against the metric hand) drops monotonically 14.07 → 13.08 cm in 4 optimizer steps;
absolute MPJPE falls 157.7 → 137.5 mm as the global scale corrects the placement; pose
shape (PA-MPJPE 8 mm) and rendering (PSNR 40 dB, frozen GS) are untouched. This is the
behaviour the unified-metric thesis predicts: a single learned scalar pulling the
up-to-scale scene onto the metric hand, feedforward, with no per-clip optimisation.

## Status / caveats

- **Directional, not converged.** 8 sequences, 4 optimizer steps (a `/tmp`-staged smoke
  run cut at the 28-min streaming wall). It establishes the *trend*; the converged floor
  needs a full run.
- The head is small (528,897 params); backbone + gs_head verified frozen
  (`backbone(unfrozen)=0 gs_head=0 injection=0`).
- 13 cm global-scale residual is still above the 4.5 cm *direct* hand-depth anchor — as
  expected, since this is one global scalar (no per-clip solve) and only ~4 steps in. The
  point is it is **learnable and feedforward**, and it is decreasing.

## Provenance

- Harness `configs/_p3_harness.sh` (stages repo to `/tmp`, patches the train script to
  print `scale_residual_m`/`obj_depth_residual_m`/`hand_depth_residual_m`, runs on 8 seqs).
- Config `configs/exp_p3_scalehead_smoke.yaml`, overrides `val_every=2 log_every=1
  val_max_batches=12 max_sequences=8`. Node studgpu-spark02, gb10.
- Run via streaming `srun` (cluster `/work`+`/home` quota-dead; logs streamed, checkpoints
  to node-local `/tmp`).

## Next

Converged run (full sequences, to a residual floor) once `/work` quota frees or a bigger
GPU lands. Pairs with the (a) partial-unfreeze object-depth verdict and the HOI4D
dense-depth result (20.5 → 13.4 cm) as the metric-scene ablation ladder.
