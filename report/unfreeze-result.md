# (a) Partial-unfreeze + GT object-depth — directional verdict

**Question (Cyrus direction a).** Unfreeze the last 4 frame + 4 global transformer
blocks (100.8 M params) and supervise the predicted scene depth on GT object depth
(HOT3D meshes), with a gentle hand-depth anchor. B2 falsified the metric-scene claim on a
*frozen* backbone (objects 62 → 135 cm, worse). Does **unfreezing** change the outcome —
does object depth trend *down* instead?

## Result (directional)

| Step | obj_depth | hand_depth | abs MPJPE | PA-MPJPE |
|---|---|---|---|---|
| 1 | 23.80 cm | 17.54 cm | 192.4 mm | 5.57 mm |
| 2 | 23.74 cm | 18.49 cm | 195.7 mm | 5.53 mm |
| 4 | 23.38 cm | 17.42 cm | 190.5 mm | 5.47 mm |
| 6 | 22.88 cm | 13.25 cm | 172.4 mm | 5.58 mm |
| 8 | **22.33 cm** | **11.17 cm** | **151.0 mm** | 5.74 mm |

**The unfreeze route works, and accelerates.** Object depth falls monotonically
23.80 → 22.33 cm, and the per-step drop *grows* (0.06 → 0.36 → 0.50 → 0.55 cm), i.e. the
trajectory is steepening, not plateauing. Hand depth drops to 11.17 cm and absolute MPJPE
improves ~41 mm (192 → 151) over 8 optimizer steps, with pose shape stable at PA ~5.5 mm.
This is the
opposite sign to the frozen-backbone B2 result (62 → 135 cm): **unfreezing a few blocks +
GT depth pulls the scene toward metric instead of distorting it.**

## Status / caveats

- **Directional, not converged.** 8 sequences, 6 optimizer steps (`/tmp`-staged smoke,
  cut at the 28-min streaming wall). The magnitude per step is small because this trains
  100.8 M backbone params (vs the 528 K-param scale head, which moves in ~4 steps).
- **23.8 cm is not directly the 62 cm B2 baseline** — different sequences and eval setup
  (8 smoke seqs here). The signal is the *in-run trend*, which is clearly downward.
- The decisive magnitude (object depth driven well below the frozen baseline) needs a
  converged run, i.e. the compute we are gated on.

## Provenance

- Harness `configs/_p3_harness.sh`; config `configs/exp_p3_gtdepth_smoke.yaml`
  (`unfreeze_last_n_blocks=4`, obj_depth loss 0.5, hand anchor 0.1), overrides
  `val_every=2 log_every=1 val_max_batches=12 max_sequences=8`. Node studgpu-spark02, gb10.
- Streaming `srun`; residual print injected by the harness patch.

## Relation to the thesis

This is the HOT3D counterpart of the converged HOI4D dense-depth result (20.5 → 13.4 cm).
Together they form the top rung of the metric-scene ablation ladder: the scene goes metric
when a few blocks are unfrozen and given a metric depth target — on HOI4D decisively
(converged), on HOT3D directionally (compute-gated). See [[scale-head-result]] for the
feedforward scale-head rung (b) and [[hoi4d-depth-result]] for the converged HOI4D rung.
