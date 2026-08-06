# Review scoreboard

Longitudinal record of every simulated review pass, so we can see whether the paper is actually
improving rather than just changing. One row per pass. Scores are the reviewers' ratings on the ICLR
ladder (2 reject / 4 marginally below / 6 marginally above / 8 accept).

**Read this with the anchors.** Measured real outcomes, from `openreview-evidence-log.md`:

| Real paper | Scores | Outcome |
|---|---|---|
| Human3R | 8/8/4 | Accept |
| JOSH | 6/6/4/6 | Accept |
| Fin3R | 4/5/5/4 (NeurIPS scale), all raised | Accept |
| Fuse-and-Refine | 2x borderline-reject, 2x borderline-accept | Accept |
| **G-CUT3R** | **4/4/4** | **Reject** |
| PhysHandi | 4/6/4/2 | Reject |
| SIGHT | 2/4/4/2 | Reject |
| 3D Affordance | 3/3/3/3 | Withdrawn |

A unanimous 4/4/4 is a reject. Accepts routinely contain a single 4. The target is a majority at 6.

---

## Pass 1 - 2026-08-06 - first calibrated panel

**Scores: 4 / 4 / 4 / 4. Decision: Reject.** Sits exactly on the G-CUT3R line.

| Reviewer | Lens | Rating | Conf | Sound | Present | Contrib |
|---|---|---|---|---|---|---|
| R1 | efficiency and cost | 4 | 4 | 2 | 3 | 2 |
| R2 | protocol and provenance | 4 | 4 | 2 | 2 | 2 |
| R3 | contribution and framing | 4 | 4 | 2 | 3 | 2 |
| R4 | domain expert | 4 | **5** | 2 | 3 | **3** |

Mean 4.0. Soundness unanimously 2. R4 at confidence 5 gave contribution 3, which no rejected paper in
the anchor set managed, so the ceiling is higher than the score suggests.

### What the panel found that was NOT already known

Two confirmed defects in the numbers themselves, both verified independently rather than taken on the
agents' word:

1. **Depth sampling normalised by a hardcoded 1408-px Aria frame** while HOI4D/H2O stores are at 224.
   A hand at image centre sampled at (0.9197, 0.0795), i.e. the frame corner. Affects every world
   number, the root-depth anchor, the training depth-anchor loss, and the scale-source ablation.
   Task #59.
2. **The Hand3R W-MPJPE cell (86.9) is unsourced**; our own tracker calls that number Hand3R's WA and
   puts its W at 125.8, which we would lose to. Task #58.

Plus, from R1's direct parameter count: the 1178.7 M total is backbone 916.23 + **camera head** 216.17
+ hand head 46.26, with no Gaussian branch. The 262.4 M trainable figure is the camera head, not the
Gaussian head as our own internal note claimed. And at `clip_len 16 stride 8` every frame is encoded
twice, so the deployed rate is 3.30 FPS, not 6.59; against HaWoR's 2.47 the honest ratio is 1.34x,
not 2.67x.

### Consensus weaknesses, all four reviewers

- Contribution 2 (hand-to-Gaussian injection) has never been trained or evaluated; `enable_gs: false`
  in every config, and `freeze_gs_head` freezes the injection too, so the described configuration is
  not expressible.
- Stated protocol contradicts the numbers: Sec 4.1 holds HOI4D out, Sec 4.6 reports its in-domain cost.
- Tables are not box-consistent, not coverage-matched, and built by two scorers with different segment
  enumeration and aggregation.
- No FLOPs, no peak memory, no per-component cost. Appendix C is a placeholder.
- No supplementary video, no failure case.

### Changes made in response (fill in as they land)

| # | Change | Status | Which reviewer it targets |
|---|---|---|---|
| 59 | Derive normalisation width from intrinsics (`2*cx`) instead of hardcoding 1408 | **CODE FIXED + 4 regression tests, 2026-08-06.** Evals must re-run | R2 W1 |
| 58 | Hand3R row | **DONE.** Row removed from `tab:world`, moved to prose with the reason | R4 W2 |
| 62 | kp2d loss Aria-hardcoded AND active at 0.05 in every run | found 2026-08-06, not yet fixed | new, defect hunt |
| - | GT-box comparison mixed two checkpoints (23.6 no-jitter vs 35.6 jitter) | **DONE.** Now 27.3 -> 35.6, matching the stated +8.3 | defect hunt |
| - | "within 13% (200.9-227.6)" excluded the three rows that widen it | **DONE.** Now 38% (200.9-276.6) | defect hunt |
| - | Re:InterHand listed as a training store after being measured non-video and dropped | **DONE.** Pool is now four stores throughout | defect hunt |
| - | DexYCB depth 0.857 (biased fast path) | **DONE.** Now 0.780 canonical; range is 0.339-0.780 | defect hunt |
| 49 | Cut contribution 2 or run it; retitle | | R1 W1, R3 W1, R4 W7 |
| 60 | Rebuild both tables: one scorer, box-consistent, full coverage | blocked by #59 re-run | R2 W3/W5, R4 W10 |
| 61 | Reconcile the four C-abs values; state joint-averaging convention | | R2 W3, R4 W4/W11 |
| 50/57 | Fill Appendix C with FLOPs, memory, per-component cost | | R1 W5 (named "the single change that would move my score") |
| 56 | Frozen-vs-unfrozen ablation | partial data exists (arms 0 and 1, cancelled at step 3000) | R1 W5, and the accept-pattern requirement |
| 53 | Appendix, supplementary video, failure figure | scaffold + 6 figure slots written | R4 W8/W9 |

### The fifth pass: cvpr-reviewer as defect hunter

Run alongside the panel with its verdict suppressed. 27 findings, of which the ones above were new.
It independently confirmed #59 by reading the code. Its most valuable unique find is #62: the same
1408 constant in the **training** kp2d loss, which the paper claims was zeroed and which every config
in the repo sets to 0.05. Keeping it as a defect finder with no score was the right call.

---

### Run log

**2026-08-06, first submission (9803703 / 9803742 / 9803744).** Two of three failed, and both
failures were informative rather than wasted.

- **dxcache32 9803742: SUCCESS.** DexYCB feature cache built, 5938 clips, 186 GB, `CACHE_VALIDATE
  max|diff|=0.000000`, rc=0. The earlier OOM was purely the 24 GB card; a 40 GB card fixed it with no
  other change. All 10 subjects present. **This unblocks the depth-bracketing claim**, which both R3
  and R4 named as the strongest result in the paper.
- **gsinj 9803703: refused to train, correctly.** The loss-recipe guard caught `kp3d_abs: 0.0`
  inherited from the 2025 GS-ablation template I built the config from. Fixed to the proven recipe
  (`kp3d_abs 1.0`, `transl 1.0`). The guard did exactly its job; see [[loss-recipe-kp3d-abs]].
- **unfrz 9803744: all four arms failed**, and chasing why exposed something worse. See below.

**The unfreeze experiment had never actually run.** Cached feature tokens are the *frozen* backbone's
output, so with `feature_cache_dir` set the backbone forward never executes and any parameter marked
trainable by `unfreeze_last_n_blocks` receives no gradient. Job 9674186's own logs show arm 1 with
`backbone(unfrozen)=25,197,056` alongside `[feature-cache] 6480/6480 clips have cached tokens`. Arms 0
and 1 were the same model, about nine GPU-hours each, and the 102.04 vs 101.00 difference reported
from them is noise. **Retracted.** A guard now refuses this combination outright, and the resubmitted
sweep runs uncached at 3 epochs matched across arms.

**2026-08-06, resubmitted:** gsinj **9834712** (arms on/off), unfrz **9834714** (n = 0,1,2,4).

---

## Pass 2 - (pending, after the free items land)

Target: move R2 and R4 off 4. Both said their blocking items are fixable without new science.
The bar to clear is a majority at 6, not the absence of a hostile reviewer.
