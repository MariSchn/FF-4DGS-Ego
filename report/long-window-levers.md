# The long-window number (W-MPJPE): what we know, what we ranked wrong, what to do

Consolidated 2026-08-07. This exists because the same ground was covered twice: the scale bias and
its mechanism were diagnosed in **June**, and on 2026-08-06 they were re-derived from scratch and
mis-ranked as the top lever. Read this before proposing any W-MPJPE work.

## The number

| set | W-MPJPE | oracle (GT trajectory) |
|---|---|---|
| 128f, 314 segments (`lwfinal_ours_online`) | **200.9** | 61.5 |
| lw60, 176 segments (`ours_decomp_both_lw60`) | 128.1 | 35.1 |

So roughly **70% of W is camera trajectory**, not hand.

## The lever ranking, from our own verified decomposition

`w-mpjpe-levers-hot3d` (adversarially verified 2026-06-25), in `world = R·(s·pj_cam) + s·t`:

| # | lever | size | status |
|---|---|---|---|
| 1 | **cross-clip chaining drift** | **~85 mm** | greedy Sim(3) seams accumulate. A global/BA Sim(3) swap is **already null** (global == greedy); a single Sim(3) cannot fix biased per-clip poses. Needs render-align, higher-DOF, or **removing chaining entirely** via a single long pass. |
| 2 | **hand root absolute depth** | ~70 mm | *"PROVEN lever: `kp3d_abs` retrain moved W 308→250, C_abs 115→53. The ONLY lever below the oracle floor. PRIMARY."* |
| 3 | scale bias | ~45 mm | **null on HOI4D.** Moves W on HOT3D (per-clip 219 vs pooled 252) but not here. |
| 4 | intra-clip per-frame pose | ~1 mm | negligible |

And the rotation/translation split (lw60, baseline 128.1):

| substitute GT | W | gain |
|---|---|---|
| rotation | 109.3 | −18.8 |
| translation | 80.3 | −47.8 |
| both | 35.1 | −93.0 |

Translation dominates rotation 2.5:1. **What is still unmeasured** is how much of that translation
term is scale *magnitude* versus *direction* - the GT-scale oracle (`W_MPJPE_gtscale`, commit
b7ca1a1) now emits it on every eval pass.

## Dead ends - do not rebuild

- **Within-clip camera-pose refinement** (`scripts/pose_refine.py`, `--refine_pose`). Render
  plumbing works (PSNR 25.3 dB) but as a W lever it is **harmful**: pooled W 251.7 → 292.3 (+40 mm),
  WA-long 95.4 → 131.3, even with photometric `improved=8/8`. The clip's Gaussians are pose-derived,
  so the photometric optimum is not the metric-truth pose.
- **Global/BA Sim(3) chaining swap** - already null, global == greedy.
- **Test-time temporal smoothing** - dead, 4%.
- **Scale variants as a W lever on HOI4D** - W is *flat* across per-clip / per-seq-median /
  per-seq-pooled on every sequence, despite a per-clip scale std of 0.227 on a mean of ~0.6.

## The scale bias: diagnosed in June, re-derived in August

**`scale-source-ablation-result`, 2026-06-27**, scoring `|s·gs_depth − GT dense sensor depth|` over
**non-hand** pixels:

| scale source | scene err | s |
|---|---|---|
| none (s=1) | 14.3 cm | 1.000 |
| hand per-clip | 45.8 cm | 0.742 |
| hand robust (per-seq, MAD) | 30.2 cm | 0.728 |
| oracle | 4.1 cm | 1.022 |

> **Mechanism (verbatim):** "the frozen feedforward backbone does NOT reconstruct the thin/foreground
> hand, so `gs_depth` AT the projected hand pixels reads the BACKGROUND behind the hand
> (~1/0.728 = 1.37x too far)... Note z_hand here is **GT**, so the bias is NOT a hand-pose-prediction
> error - it is the frozen scene depth being wrong at the hand."

**Verdict then: hand-as-scene-scale REFUTED.** Hand-robust (30.2 cm) is 2x *worse* than not scaling
at all (14.3 cm).

The 2026-08-06 measurement (`s_hand` 0.6208 vs `s_gt` 1.0230, ratio 0.578) is **the same finding**,
and the "one-sided background contamination" hypothesis proposed that day is **the same mechanism**,
re-derived without checking the record. Using GT hand depth, June had already excluded the hand as
the cause; August re-established it indirectly from C-abs.

### What of the August work survives

- The **low-order-statistic estimator** (`sample_depth_at_joints(window=3, reduce="min")`) is a
  genuine response the June work never tried: it refuted hand-as-scale but did not attempt a robust
  near-surface read. It may fix the scale *itself*, which matters for metric-honesty claims.
- The **18.4% clamp-floor rate** is a real defect regardless of whether the scale moves W.
- The **GT-scale oracle** re-tests the June "scale is null on HOI4D" verdict on current data.
- **Expected W payoff: near zero on HOI4D**, per the June refutation. The `zguard` A/B's primary
  win condition (W goes down) will most likely fail, and that is predictable in advance.

## What actually attacks the long number

1. **Remove chaining** - the #1 term at ~85 mm. This is what variable-length training plus a single
   long inference pass does (#48). **Gated on the memory probe**: if 100 frames do not fit, the
   single-pass form is dead and the fallback is fewer, longer chunks.
2. **Hand root absolute depth** - #2 at ~70 mm, and the only lever below the oracle-camera floor.
   `kp3d_abs` already moved W 308→250 once.
3. **Camera head** (#39) - never trained. `cam_token` sits frozen in the backbone, one token per
   frame, and GT extrinsics are now available on all 157 sequences (commit e55c3b4).
4. **Scale** (#63) - null on HOI4D by the June result. Worth finishing for correctness, not for W.

## Timing to a dynamic-frames number

Measured from `varlen_9732155_0.out`: 10 epochs, 6559 steps/epoch, and 6 h 54 m of wall clock
reached epoch 2 at 58% (~2.58 epochs), i.e. **~2.7 h/epoch → ~27 h per arm** for the full 10 epochs.
Both prior arms were killed by a time limit before finishing epoch 3, which is why there is no
result.

| stage | duration |
|---|---|
| memory probe (the gate) | ~40 min, after geo59 + regdump |
| varlen retrain, per arm | **~27 h** |
| eval | ~1 h |

Two arms in parallel ≈ 27 h of training; sequential ≈ 54 h. **The gate must pass first**, and the
matched-control arm (`fixed32`) is not optional: the window sweep found longer *fixed* windows hurt
articulation (16f C_rr 16.6 → 64f 28.7), so the random-length arm has to be compared against the
lock on identical data.

## The process lesson

The August scale work cost most of a day and reproduced a June result. The memory index already
carried `scale-source-ablation-result` and `hoi4d-world-space-results`, both of which name the
mechanism and the null W verdict. **Read the existing decomposition before proposing a W lever.**
