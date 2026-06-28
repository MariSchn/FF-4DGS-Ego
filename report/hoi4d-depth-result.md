# HOI4D dense-depth result (finalized)

**Question.** Does the predicted Gaussian scene depth become metric/accurate when we
supervise it on *dense* GT depth — as opposed to the HOT3D object-render path, which
only has sparse, masked object depth? HOI4D has a real depth sensor, so this isolates
the claim **"dense depth supervision makes the feedforward scene metric"** with the
hand head *disabled* (`enable_hand: false`), no masking.

## Result

| Setting | Scene depth residual (val) | Δ vs baseline |
|---|---|---|
| Frozen NeoVerse backbone (no depth supervision) | **20.51 cm** | — |
| + dense GT-depth supervision, partial unfreeze (last 4 blocks) | **13.40 cm** | **−35%** |

- The residual is the mean depth error over **all** valid pixels (no hand/object
  masking): `n_valid ≈ 1.5–1.6 M` pixels per validation batch.
- Best checkpoint: `/work/scratch/dmonopoli/checkpoints/hoi4d_depth/best_depth.pt`
  (4.6 GB), saved at the step-60 new-best (13.40 cm).

## Provenance

- Job **99535** (`/work/scratch/dmonopoli/joblogs/99535_hoi4ddepth.out`), studgpu-spark01.
- Config `configs/exp_hoi4d_depth.yaml`: frozen NeoVerse + `unfreeze_last_n_blocks: 4`,
  dense depth loss (smooth-L1, `margin 0.05 m`, depth range 0.05–10 m), batch 2 ×
  grad-accum 4, lr 1e-5, AMP. Data `/work/scratch/dmonopoli/hoi4d` (14 G).
- Validation trace (baseline 20.51 cm throughout):
  - step 20 → **17.09 cm**
  - step 40 → **15.28 cm**
  - step 60 → **13.40 cm**  ← best, checkpoint saved
- The job was then **cancelled by SIGTERM at step 80** (preemption / wall-time on the
  spark node), so 13.40 cm is the *best-so-far*, **not** a converged number — training
  residuals were still oscillating 3–12 cm, i.e. there is clear headroom for a longer,
  uninterrupted run to push below ~10 cm.

## Why this matters

This is the **existence proof** for the metric-scene claim: on a frozen backbone the
scene depth does **not** become metric (HOT3D objects 62 → 135 cm, *worse*, when only
the hand anchor is available). But when a *real metric depth target* exists and a few
blocks are unfrozen, the residual drops a clean 35% with **no masking**. It is the
bridge between B2's frozen-backbone falsification and the unified-metric goal, and the
direct setup for the HOI4D ↔ Hand3R world-space comparison.

## Open / next

1. **Longer run to convergence** — the SIGTERM cut it at step 80; an uninterrupted run
   should tighten 13.40 cm further. Queue after the P3 (a)/(b) jobs clear (QOS = 1
   job/user, so it serializes).
2. **Hand3R comparison** — needs the manual HOI4D Hand-Pose download (user task) to put
   our C-MPJPE next to Hand3R's 42.6 on the same sequences.
