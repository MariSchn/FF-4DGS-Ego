# Open Lines Tracker

## TARGET: ICLR 2027. Abstract Sep 18 2026, paper Sep 25 2026. Set 2026-08-09.

Six and a half weeks, not the three months the CVPR-November assumption implied. Read this before
picking up anything below, because it changes what is worth doing rather than only how fast.

**Critical path is #39, the camera-head fine-tune.** It is the only item left that needs a full
training run; everything else is writing or evaluation and compresses. Its result also decides which
paper we write, so the introduction cannot be drafted before it lands. Recovering a meaningful share
of the 200.9 -> 61.5 oracle gap keeps the thesis at "method with analysis"; a null moves it to
"analysis", which then has to be written as such.

**The thesis moves.** Every strong architectural claim has been refuted by our own controls: DINOv2
matches the reconstruction backbone, scene-recon is not a lever, the scale is not a lever, box
geometry is null, and HaMeR_ft beats us 23.4 to 35.8 on identical boxes. What the measurements do
support is transfer: depth coverage of the training mixture cuts H2O zero-shot 184.8 -> 66.2, a 64%
reduction, for 6.9 mm in domain. That belongs on page one and is currently buried.

**Order.** Weeks 1-2 experimental: #39 on both GPUs, box-consistent tables, HaMeR_ft 23.4 moved INTO
the table beside a parameter column, and #49 settled - prefer RETITLING over evaluating the scene,
because evaluating opens a new attack surface at six weeks out while retitling closes one for free.
Weeks 3-4 rewrite. Week 5 figures A1-A6 and the supplementary video, whose absence is named
explicitly in our own reject samples, then ONE OpenReview panel on a finished draft. Week 6 fixes
and buffer.

**Cut:** TACO (a fifth dataset does not move a contribution score; the thesis does), EgoAllo,
**EgoForce**, and the variable-length run if its step-1600 read does not pay.

EgoForce was reinstated and then cut again the same day, and the second decision is the right one
under the new thesis: it is camera-space, and once we stop claiming to beat everyone on accuracy we
do not need an absolute SLAM-free comparator in a table. It remains CITED in related work with its
published figures and an explicit not-input-matched note, so the question "why is the published
SLAM-free absolute SOTA missing" has an answer rather than a silence. To reinstate: code, weights
and ARCTIC/H2O/HO3D/HOT3D loaders at github.com/dfki-av/EgoForce, ordinary Python requirements with
no SLAM to compile, and native HOT3D_PINHOLE support matching our own conversion. Independently of
the table, its Crop Intrinsics Token is the published mechanism our null box-geometry ablation
points at, and that deserves a sentence in the paper.

## 2026-08-08 - the backbone we call frozen is not the released one (#71, RESOLVED)

Started as a label check on the architecture figure ("0 of 24 blocks fine-tuned") and ended as a
provenance finding. Student jobs 104471-104477, scripts `~/verify_frozen{,2,3}.py`,
`~/verify_optim.py`, `~/find_ancestor.py`.

**Two things were wrong, for different reasons.**

*The denominator.* `visual_transformer.py` builds TWO ModuleLists of `depth` blocks, `frame_blocks`
(line 219) and `global_blocks` (line 234), and the forward runs frame[i] then global[i]. `depth=24`
means **48 encoder blocks**, not 24. Every "24 blocks" statement in the paper was wrong, and
`unfreeze_last_n_blocks: N` unfreezes the last N of EACH list, i.e. 2N blocks.

*The claim.* Both headline checkpoints differ from `models/NeoVerse/reconstructor.ckpt` in exactly
`frame_blocks[20:24]` + `global_blocks[20:24]`: 8 blocks, **100,788,224 parameters**, identical in
winner10ep (step 9900) and jitterrob10ep (step 11700). The other 40 blocks are BIT-IDENTICAL,
max|delta| exactly 0.0, which is what rules out a save-precision artefact: a dtype effect would
perturb all 48.

**But the hand run did not train them, and that matters.** Its optimizer state owns 134 tensors
totalling **46,256,160** parameters = 3.92% of the 1,178,658,867-parameter model, and no backbone
tensor was ever stepped. The paper's "46.3M trainable, 3.9%" is therefore CORRECT, and is now
sourced from Adam's per-parameter moments rather than from a config. The 8 blocks arrive at
INITIALISATION: `exp_p4_jitterrob.yaml:67` warm-starts from `checkpoints/hoi4d_depth/best_depth.pt`
(4.9 GB, a FULL-model dict), and `exp_hoi4d_depth.yaml:48` sets `unfreeze_last_n_blocks: 4`. The
ancestor scan confirms `best_depth.pt` carries exactly those 8 blocks and nothing else.

| checkpoint | blocks changed vs stock | params |
|---|---|---|
| `hoi4d_depth/best_depth.pt` (the warm start) | 8: frame/global[20:24] | 100,788,224 |
| `winner10ep_best.pt`, `jitterrob10ep_best.pt` | 8, same indices | 100,788,224 |
| `h2o_hand/best_cmpjpe.pt`, `h2o_hand_hires_full/`, `old_hires/` | **16**: frame/global[16:24] | 201,576,448 |

**Consequences, now written into the paper** (`4exp.tex`, new paragraph "The backbone we freeze is
not the released one"; `3method.tex:36` and the figure caption):
1. Reproduction from stock NeoVerse will NOT match our numbers. The depth stage is part of the recipe.
2. The warm-start contamination already disclosed in `sec:exp:data` is larger in SCOPE than written:
   it is 100.8M backbone parameters trained on HOI4D, not merely a head initialisation. The measured
   effect is unchanged (0.2 mm), because clean-152 already excludes the affected sequences.
3. The backbone-substitution ablation gave its reconstruction arm one extra stage of in-domain
   training that DINOv2 and random-init never got, and it STILL lost to DINOv2. The bias runs
   against that arm, so the null conclusion strengthens.

**OPEN, carried forward.** The H2O rows sit on a 16-block / 201.6M fine-tune. Any H2O statement
that says or implies "frozen backbone" has to be re-checked against that. Separately, #56's
frozen-vs-unfrozen ablation compares against a "frozen" arm that is itself depth-tuned, so its
framing needs a sentence.

**THE RULE.** A claim about a checkpoint is verified against that checkpoint's *weights*; a claim
about what was *trained* is verified against its *optimizer state*. Configs are not evidence:
`exp_p4_jitterrob.yaml` had been edited since the run it supposedly documented (its training block
now reads `epochs: 1`, `output_dir: rt_c1`) and the run's stdout was gone.

### Two false failures the same evening, both mine, both probes rather than the thing probed

- **gsplat "BUILD FAILED" in 21 s having never invoked nvcc.** gsplat 1.5.3 keeps `_C` in
  `gsplat.cuda._backend` and imports it lazily inside each wrapper call, so
  `getattr(_wrapper, '_C', None)` is None whether or not the extension compiled. Testing
  `from gsplat.cuda._backend import _C` triggers the real build: 3742 s of nvcc, then
  `GSPLAT_BUILD_OK`. #49 unblocked.
- **`lualatex` exiting 1 on `! Dimension too large`.** The `tab:world` caption had grown to ~450
  words; `\@makecaption` typesets the whole caption into one unbroken hbox to test single-line fit,
  and past TeX's maximum dimension (16383.99 pt) that measurement fails. Moved into a `minipage` of
  table notes under the tabular, no content lost. Paper now builds rc=0, 0 overfull, no undefined
  references.

### Baseline set, revised

EgoForce is **dropped** (task deleted). Verified from its abstract: it "recovers robust, absolute 3D
hand pose and its position from the user's (camera-space) viewpoint" and reports camera-space MPJPE.
No trajectory, no world evaluation. It cannot fill a world-table row, only a C-abs one.

**Replacement search, conducted 2026-08-09 with CODE AVAILABILITY as the gate.** The 2026 world-space
hand literature is almost entirely code-less right now:

| method | world-space? | code | usable as a baseline |
|---|---|---|---|
| StableHand (2605.18553) | yes, W/WA-MPJPE + Accel on HOT3D+ARCTIC | **none** | no |
| Hand3R (2602.03200) | yes, our closest competitor | **none found** | no |
| WHOLE (2602.22209) | yes, but needs metric-SLAMed Aria input | **none found** | no |
| EgoGrasp (2601.01050) | yes, H2O + HOI4D | **none found** | no |
| UniHand (2602.21631) | camera + world on HOT3D | **none found** | no |
| HaWoR (2501.02973) | yes | `ThunderVVV/HaWoR` | ALREADY a baseline (#30) |
| Dyn-HaMR | yes | released | IN FLIGHT (#31) |
| EgoAllo | world, but body SMPL-H + Aria MPS | released | out of regime, #44 |

Also searched and rejected, so nobody repeats the search: **HandFlow** (2607.11221, leads every
world/trajectory metric, WA 16.17 vs Dyn-HaMR 31.01) has no code. **EgoHandICL** (2601.19850,
ICLR 2026) HAS code but is not world-space: its metrics are P-MPJPE, P-MPVPE, F@k and MRRPE, all
root-relative or Procrustes-aligned camera-space. **HMP** (WACV 2024) optimises a global hand
trajectory but never estimates camera motion, so it is not world-space under a moving egocentric
camera. **HandDGP** (`nianticlabs/HandDGP`, code released) is camera-space by title and would only
be a fourth instance of the SLAM composition we already have three of.

**THE REFRAME THAT MATTERS.** Hand3R, our closest competitor, builds its own world-space comparison
from exactly **HaMeR-SLAM, WiLoR-SLAM and HaWoR**. That is the published field standard for this
metric, and our table already contains all three plus **Dyn-HaMR** and **HaPTIC+SLAM**. We are not
missing a world-space baseline; we have a SUPERSET of what the closest competitor uses. The real
gap is not coverage but input-matching (#30 HaWoR on detbox v3, #31 Dyn-HaMR currently scoring zero
segments). Fixing those two is worth more than any additional row.

Everything code-less above should be cited as concurrent work, numbers quoted and explicitly marked
non-comparable, and the authors emailed for code.

## 2026-08-07 (result) - #63 CLOSED AS NULL, #70 real but not an accuracy lever

Jobs 104381 + 104382, 30 matched HOI4D sequences, 360 segments, jitterrob, detbox v3. The win
condition was written into the job BEFORE it ran, with W primary, precisely so that improved
diagnostics could not be mistaken for success.

| arm | W | dW | s_med | ratio/GT | floor% | C_abs | C_rr |
|---|---|---|---|---|---|---|---|
| base | 32.23 | | 0.603 | 0.551 | 19.2% | 32.16 | 21.02 |
| z (#63 H1 behind-camera) | 32.33 | +0.10 | 0.633 | 0.588 | 16.9% | 32.16 | 21.02 |
| win (#63 H2 win3-min) | 32.26 | +0.03 | 0.609 | 0.556 | 19.2% | 32.16 | 21.02 |
| **hv (#70 hand gate)** | 32.44 | +0.21 | **0.772** | **0.668** | **0.0%** | 32.16 | 21.02 |
| hv + win | 32.48 | +0.24 | 0.778 | 0.672 | 0.0% | 32.16 | 21.02 |

C_abs and C_rr are bit-identical in all five. They are camera-frame and cannot depend on the scene
scale, so any movement would have voided the arm. None moved.

### The defect is real; the lever is not

The hand gate moves `s_med` 0.603 -> 0.772, narrowing the gap to the true 1.023 from 41% to 25%,
and removes **every** clamp-floor failure (19.2% -> 0.0%). A clamp-floor segment is a solve that
failed and returned the bound as though it were an estimate, so this is a genuine robustness gain.

And **W does not move**: every arm is between +0.03 and +0.24 mm, inside the +/-0.8 mm seed band.

### Two things this settles about #63

**H1 was a partial proxy for #70, not a mechanism of its own.** `z` reaches `s_med` 0.633 against
`hv`'s 0.772, about 17% of the effect. The arithmetic explains it: 94% of behind-camera samples are
the phantom hand, but only HALF the phantom hand's joints are behind the camera. `z` removes that
half; `hv` removes all of it.

**H2 is null even in combination.** On top of `hv` it contributes +0.006 to `s_med` and nothing
else. Its premise, one-sided background contamination making a low-order statistic the right
estimator, does not survive the actual depth map: the prediction is smooth enough that a 3x3
minimum is nearly the bilinear read. Visible in `report/fig_registration_steps.png`, where the
depth panel is blobby rather than sharp. A larger window or a lower quantile remains testable, but
the mechanism is not a W lever either way.

### The conclusion, reached three independent ways

The scale can now be substantially CORRECTED and W is indifferent. That agrees with
`hoi4d-world-space-results` (scale and pose both refuted as W levers on HOI4D) and with
`camera-head-is-the-lever` (hand within 2 mm of oracle while W sits at 200). Three routes, one
answer: **W is not scale-limited, and the long-window work belongs on the camera trajectory.**

### Decision

Adopt `--gate_scale_on_hand_valid` on CORRECTNESS and ROBUSTNESS grounds, and state in the paper
that it leaves W unchanged. Feeding joints from a hand the detector never found into a metric solve
is indefensible once a reviewer sees it, and removing 19.2% of failed solves is real. Presenting it
as an accuracy improvement would be false.

Job 104415 produces the paper-grade full-157 gated cells at seg30 and seg100. The 360-segment
numbers above are a matched contrast and must not enter the paper.

## 2026-08-07 (later) - a job reported COMPLETED while all four arms produced nothing

### The third instance of the same shape

zguard job 104380 ran the whole #63 2x2. Every arm died on
`FileNotFoundError: /home/dmonopoli/ckpt_backup/jitterrob10ep_best.pt`, because the local copy was
deleted once the HuggingFace upload verified its hashes. `sacct` reported **COMPLETED**: the arm
wrapper swallowed each non-zero exit, the readout script ran afterwards, printed `MISSING` on all
four rows, and exited 0. So H2 (the window-min estimator, the mechanism that actually predicts the
SIZE of the scale bias) has never had a reading at all.

This is the same failure as the `enable_gs` eval trap (exit 0, silently non-metric W/WA) and the
C-abs-725 untrained checkpoint. The pattern: **a run that reports success while producing
nothing**. It is expensive precisely because nothing draws attention to it.

### Fixed in three places

1. `build_model` now calls `_require_checkpoint_present`, which exits non-zero **before** a GPU
   hour is spent and names the restore route. The bare `FileNotFoundError` was actively
   misleading: it said the file was absent but not that it is one command away in the private HF
   repo, so the obvious response was to retrain and quietly change the headline numbers. The
   message says `Do NOT retrain` for exactly that reason.
2. `handgate.sbatch` records per-arm failures and **exits non-zero** if any arm failed or wrote an
   empty JSON, so the readout can no longer be the last word.
3. `tests/test_missing_checkpoint_is_loud.py` pins both properties, with a negative control that a
   present checkpoint is a no-op.

### The checkpoint is back

Restored from `mondraaa/worldhand4dgs-checkpoints` and **independently hash-verified** on the
cluster: `5f2f12ddc52fddf201bc9d04abfbece2fde4aae1ce2a09f15c74cd9e786651bc`, matching the recorded
value. It now lives in `/home/dmonopoli/ckpt_backup/` (5.09 GB) rather than scratch, because
`/work/scratch/dmonopoli` is **at quota** - `mkdir` there fails outright. Home went 8.6 G -> 14 G.

Note for next time: the login node has no `pip` and no `huggingface_hub`, and `venv_gb10` is
aarch64 so it cannot run there. The restore therefore uses plain `curl` with the bearer token fed
through `curl -K -` (stdin), which keeps it out of `ps` and off disk. Script: `~/hf_restore.sh`.

### Paper consistency swept at the same time

The world table row was updated to the post-#59 numbers (W 36.812 / WA 32.934) in f917e4c, but
**four prose sites still quoted the pre-fix 37.3 / 33.2**: the W-vs-WA-at-30-frames illustration,
the `\todo` naming the bolded cell, the segment-length sensitivity sentence, and the HaPTIC
shared-box comparison. All four corrected; `main.tex` recompiles clean at 28 pages. Same
checkpoint and same segments throughout - only `s` moved, and `s` multiplies camera translation
only, so W and WA move while C_abs does not.

## 2026-08-07 - the scale solve was eating a hand the detector never found (#70)

### Found by fixing a figure, not by reading code

Cyrus asked for "visual intermediate results for the registration steps". Building that panel
meant putting the projected joints on the RGB frame AND on the predicted scene depth. `gs_depth`
is stored 90 degrees rotated, so the first version drew the same joints at different pixel
positions in the two panels. Un-rotating the depth for display (`rot90(k=1)`, verified lossless:
it reproduces the eval's own store-frame sample with max abs difference **0.0**) put both hands in
one coordinate frame, and a second joint cluster appeared sitting on a door, a floor and a table,
but on no hand.

### MEASURED, 5 dumps, scene ZY20210800001_H1_C12

| | n per 16f clip | median ratio | frac z<=0 |
|---|---|---|---|
| slot 0 (never detected) | exactly 96 (6/frame) | **-0.019** | 50% |
| slot 1 (the visible hand) | 213-246 | 0.607-0.894 | 0-4.2% |

Slot 0 is **240 of 255 = 94.1%** of every behind-camera correspondence.

`hoi4d_detboxes_v3/ZY20210800001_H1_C12_N28_S200_s01_T2.pt` has `valid[:,0].sum() == 0` over all
**300** frames. The detector never found a left hand. So this is not a badly predicted real hand:
the store says the hand is absent, the model emits a default MANO into the empty slot, those
joints project to plausible in-frame pixels, and the purely geometric `ratio_validity_mask`
accepts them. `eval_sequence` already reads `hand_valid` and passes it to `build_views` for
conditioning; `predict_clip` was simply never given it.

### This reframes #63

The behind-camera population is mostly this phantom hand. `--require_positive_z` masks a symptom
whose cause is the absent slot, so the two must be measured on separate arms or they conflate.
Dropping slot 0 moves `s` per clip 0.6053->0.6233, 0.7395->0.8943, 0.7199->0.7636, 0.5901->0.6072,
0.6745->0.6968: toward `s_gt` 1.023, **not closing** it. `s` multiplies camera translation only,
so W/WA move and C_abs does not.

### Status: instrumented, OFF by default

`--gate_scale_on_hand_valid` ships OFF (commit 0221ba4), same discipline as the z-guard. The
absent-hand rate is recorded under both settings so the arms stay comparable. **Validate
criterion:** adopt only if the full 157-seq A/B moves `s_med` toward 1.0 AND does not worsen W/WA.
If the rate turns out to be ~0% on other stores, record that too rather than leaving the flag
dangling. Caveat: 5 clips of ONE HOI4D scene; check H2O and HOT3D for the same asymmetry.

### Also closed on the way

- **`downsample_probe.py:99` unpacked 4 values from the 5-tuple.** The same latent
  `ValueError: too many values to unpack` that killed every sequence of geo59's seg100 stage. The
  arity test written after that incident only scans `eval_world_space.py`, so it never saw this
  one. Fixed with starred unpacking.
- **The figure's labels were wrong.** `R1/R1/R2/R3` read as a typo. Fig. 1's caption names
  **three** steps and the first does two things, so the panels are now `R1a/R1b/R2/R3` and use the
  caption's own vocabulary rather than a second one.

## 2026-08-06 (late) - task #68 resolved: "detbox" means a different detector per method

### The answer was already in this file

D2-10, recorded 2026-07-19, states it outright:

> "Ours det-box rows (49.8 / 35.6) use detbox v3 ... HaMeR fu/fj/mt det-box rows (44.0 / 41.6 /
> 68.7) use **WiLoR-detector boxes**"

and the fix D2-10 built was the `--box file` driver that produced the **v3box** variant:

> "fj2 on the exact v3 boxes = **C_abs 23.4** / wrist 22.3 / C_rr 13.2"

with sanity anchors "fj2 GT 19.0 vs orig fj 20.8; own-det 38.6 vs 41.6".

| directory | boxes | C-abs |
|---|---|---|
| `hamer_fj2_gtbox_preds` | GT | 19.01 |
| `hamer_fj2_v3box_preds` | **our detbox v3** | **23.41** |
| `hamer_fj2_detbox_preds` | **WiLoR's detector** | 38.55 |

So "detbox" means our v3 for OUR rows and WiLoR's detector for HaMeR's rows. Same word, different
detector, no marker anywhere except a per-method convention nobody wrote down.

### Consequence 1: the abstract is right

"23.4 mm on identical boxes against our 35.8" traces to `hamer_fj2_v3box_eval` (23.41), which is
the locked detbox v3. Nothing to change.

### Consequence 2: two world-table rows are not input-matched, and I certified them this morning

`fj2_slam_detbox_preds` and `mt_slam_detbox_preds` are composed from the WiLoR-detector caches, so
HaMeR$_{ft}$+SLAM (37.4 / 36.8 / 53.6) and HaMeR$_{mt}$+SLAM (67.3 / 41.8 / 66.8) sit on a third
detector that is neither ours nor their own. **I marked both as SHARED earlier today by trusting
the `detbox` substring** - the exact failure I had flagged an hour before, in the commit that added
provenance stamping specifically because directory names are not provenance.

**Direction: it understates HaMeR_ft.** Input-matched it is 23.4 camera-frame against the 37.4
printed. The comparison that already goes against us goes against us harder.

### What was done, and what is deliberately not

Both rows now carry a footnote saying plainly that they are not on our boxes, naming WiLoR's
detector, giving the input-matched 23.4, and explaining why the world cells cannot be corrected the
same way. **Marked, not moved**: they belong in neither the shared nor the native column.

An input-matched HaMeR_ft+SLAM **world** row needs `hamer_fj2_v3box_preds` composed with the shared
trajectory. That directory was purged from Euler scratch (1 of 157 files) **and** the fine-tuned
checkpoint `ckpts/fj2_tuned.pt` went with it, so completing the row means **repeating the
fine-tune**, not running an eval. That is a decision, not an errand: repeat the fine-tune, or keep
the footnote and let the abstract carry the input-matched comparison. The footnote is honest either
way.

### Still unverified, and three inspection attempts FAILED (2026-08-07)

`wilor_slam_detbox_preds` (the world table's WiLoR+SLAM 43.7) is composed from
`wilor_detbox_truefocal_preds`, which comes from a **tarball with no producer script on either
cluster**. Its sibling `wilor_detbox_hd_preds` WAS built with `--box_dir hoi4d_detboxes_v3`, but
that directory is empty and its result json never existed, so it is not the one in the table.

Three attempts to settle it by inspection, all inconclusive, recorded so nobody repeats them:

1. **Validity-mask agreement with the v3 store.** VOID: scored the known NEGATIVE
   (`hamer_fj2_detbox_preds`) at 100% and the known POSITIVE (`haptic_detbox_preds`) at 79.5% -
   backwards. It measured "is this dir fully populated", not box source.
2. **2D IoU, first pass.** VOID, all zeros: the store holds `bboxes [N,2,4]` **normalised** per
   hand, and they were being compared against pixel coordinates.
3. **2D IoU with the correct layout**, calibrated on a matched known pair:

   | dir | median IoU vs v3 box |
   |---|---|
   | `hamer_fj2_v3box_preds` (known POSITIVE) | 0.2632 |
   | `hamer_fj2_detbox_preds` (known NEGATIVE) | 0.1817 |
   | `wilor_detbox_truefocal_preds` | 0.0345 |
   | `wilor_native_truefocal_preds` | 0.1752 |

   Separates in the right direction but by only 0.08, and the positive's own IoU is low. The
   WiLoR outlier at 0.0345 is very likely a **confound**: the "truefocal" rescaling changes
   predicted depth and hence projected extent, so IoU is not comparable across truefocal and
   non-truefocal dirs.

**Not determinable by inspection.** Resolve by regenerating the row from a known box source
(`build_native_baseline_preds --box_dir hoi4d_detboxes_v3` -> compose -> score 30/30), which now
stamps provenance automatically. Until then the 43.7 cell is box-source **unconfirmed**.

**The lesson that generalises:** every one of these tests was only trustworthy because it was run
on a known positive AND a known negative first. Two of the three were silently backwards and would
have produced a confident wrong answer without controls.

### Original note, superseded by the above


`wilor_slam_detbox_preds` comes from `wilor_detbox_truefocal_preds`. WiLoR has BOTH a `native` and
a `detbox` variant, which suggests `detbox` = ours there - but that is an inference from naming,
and naming is precisely what failed here. Confirm before the table is final (folded into #69).


## 2026-08-06 (evening) - the scale is 42% too small, and the cause is one-sided

### The measurement that reframes task #63

`ours_fix59_seg30.json`, 1884 segments, against the GT camera scale the eval already computes:

| quantity | value |
|---|---|
| `s_gt` median (true camera scale) | **1.0230** |
| `s_hand` median (what we solve) | **0.6208** |
| ratio hand/GT | **0.578** |
| on the 0.1 clamp floor | 296/1884 = 15.7% |

**#63 was filed as "16% of segments clamp to 0.1". That is the symptom.** The whole distribution
is biased low: on geo59 the population `s_med` median is 0.644 and the **maximum over 152 segments
is 0.982**, so we never reach the truth even from above. (Caveat: only 12 of 1884 segments carry
GT extrinsics, so the ratio is directionally solid, not tight.)

This is first-order, not a tail issue. `s` multiplies the camera **translation** only, and the W
decomposition names translation as the dominant lever (lw60: GT translation −47.8 mm vs GT
rotation −18.8 mm, jointly −93.0 mm). Our seg128 W is 200.9 against a GT-trajectory oracle of
61.5.

### Where the bias comes from, and why it is one-sided

Not the hand: C-MPJPE absolute is ~36 mm at ~0.7 m, about 5%, nowhere near 42%. So `d_scene` at
the sampled pixels reads roughly **1.7x too far**.

At a hand joint's pixel the nearest visible surface **is** the hand. So any misregistration,
sub-pixel error, or silhouette blur in the predicted depth blends in **background**, which is
strictly **farther**. `d_scene` can be pushed up and essentially never down, so `s = z/d` is
biased **down**. A hand at 0.5 m against background at 0.85 m gives a ratio of 1.7 - the size we
measure.

**Under one-sided contamination the correct estimator is a low-order statistic, not the mean-like
bilinear blend.** That is a statement about the noise model, not a tuning knob.

### Implemented, both OFF by default

- **H1** `--require_positive_z` - behind-camera joints (the projection clamps depth to
  `Z_MIN`=5 cm but returns the RAW depth, so such a joint lands at a plausible in-frame pixel and
  contributes a negative ratio the median eats).
- **H2** `--scene_depth_window 3 --scene_depth_reduce min` - one-sided contamination.

`sample_depth_at_joints` gained `window=`/`reduce=`, sampling a KxK neighbourhood in normalised
units so it stays resolution-independent. `"mean"` is deliberately **not** offered: averaging
re-introduces exactly the blend this exists to remove.

12 new tests pin both mechanisms on synthetic maps with exact truth, including a **no-op test on
flat depth**, so a "fix" that merely always reports something nearer would fail rather than pass.

### The 2x2 (job 104353, 30 matched sequences, seg30)

Only H2 predicts the **size** of the bias; H1 predicts the clamp tail. They are not exclusive,
hence a 2x2 rather than a shootout.

**Adopt only if** W-MPJPE goes down (primary, all segments) **and** `scale_ratio_med` moves toward
1.0 (corroboration) **and** `C_abs`/`C_rr` do **not** move - those are camera-frame and cannot
depend on the scene scale, so if they shift the arm is void and the readout prints VOID
automatically. A lower clamp rate alone is not success: excluding joints shrinks the population.
**If nothing moves W**, the scale is not the binding term and the camera head (#38/#39) is next.

### The GT-scale check was starved by an unrelated cache (fixed, commit e55c3b4)

I reported the 0.578 ratio with a caveat: only 12 of 1884 segments carried GT extrinsics, so it
was directional rather than tight. That caveat was wrong about the cause, and the cause was
fixable.

    157/157 sequences have cam_extrinsics_cache.pt
      1/157 sequences have gt_joints_2d_cache.pt

Two couplings turned that into 1/157 coverage. The cache block loads from disk only when ALL THREE
of {gt_joints_2d, cam_extrinsics, cam_intrinsics} exist, so a missing 2D cache sends the sequence
to a recompute path that returns `None` for a store without calibration - discarding extrinsics
that are sitting on disk. And both `clip["cam_extrinsics"]` and `out["cam_extrinsics"]` were
assigned inside the `gt_joints_2d` branches.

No consumer of extrinsics needs 2D joints. `cam_intrinsics` had already been decoupled for exactly
this reason, with a comment saying so, which is what makes the extrinsics coupling an oversight
rather than a design choice.

The fix is additive and **no metric can move**: `cam_extrinsics` is never fed to the model, and in
`eval_world_space` it is read only by the `s_gt` pair, the `diag_cam` print and `eval_oracle_cam`.
The GT-scale measurement now spans all 157 sequences instead of one, which turns 0.578 from
directional into quotable and makes the `scale_ratio_med` column of the 2x2 meaningful.

Also confirmed while here, from the registration dumps: the predicted camera centres are **not**
degenerate (per-clip spread 0.003-0.007 scene units over 16 frames, comfortably past the 1e-4
gate). So the low scale is not a collapsed trajectory.

### Settled on the side: the Fast3R index pool is NOT needed

`visual_transformer.py:505-524` - `cam_token`/`reg_token` have shape `(1, 2, X, C)`: position 0
for frame 0, position 1 shared by **every** remaining frame. There is no per-frame positional
encoding along the frame axis at all, so the model is permutation-equivariant across frames 2..S.
Fast3R needs the index pool because it *has* index embeddings trained only over 0..19; we have
nothing that can go out of range. Task #48 drops to the random frame count alone, which is already
implemented. No architecture work owed.


## 2026-08-06 (later) - the #59 geometry fix did NOT close #63; a behind-camera hole in the scale mask is the new suspect

### MEASURED, not inferred

On the in-flight `geo59` run (job 104348, post-#59 depth-sampling fix, HOI4D detbox v3, seg30),
the first **152 scored segments** give:

| quantity | value |
|---|---|
| segments solving exactly onto the `0.1` clamp floor | **28 / 152 = 18.4 %** |
| `s_med` median | 0.644 |
| `s_med` p25 / p75 | 0.571 / 0.705 |
| `s_med` max | 0.982 |

Two readings, both load-bearing:

1. **#63 is NOT a side effect of #59.** The clamp-failure rate was 15.7 % before the geometry fix
   and is 18.4 % after it, on the same store and protocol. Fixing where the depth is sampled did
   not fix how often the solve fails.
2. **The surviving scales are biased LOW.** The true camera-centre scale on this data is ~1.0
   (see `world-lift-uses-a-hand-derived-scale`), and the *maximum* observed `s_med` across 152
   segments is 0.982. A median that never reaches the truth from above is a median being pulled
   down, which is the same direction a negative contaminant would pull it.

### The candidate cause, with the mechanism pinned

`predict_clip`'s validity mask was `in_frame & sampled>0.01 & isfinite(z) & isfinite(sampled)`.
It never required `z > 0`. That matters because of a clamp that looks harmless:

```python
z = pred_joints[..., 2].clamp_min(Z_MIN)      # Z_MIN = 0.05 m, projection only
col = f * x / z + cx ;  row = f * y / z + cy
return grid_xy, pred_joints[..., 2]           # <- RAW, UNCLAMPED z
```

A joint 30 cm **behind** the camera is projected *as if it sat 5 cm in front of it*, so for a
joint near the optical axis it lands at a perfectly plausible in-frame pixel, while the depth
handed back for the ratio is still `-0.3`. The ratio `z/d_scene` is negative, `in_frame` is True,
and the median eats it. **The clamp that keeps the projection finite is exactly what hides the
sign error.** Proven without a GPU in `tests/test_negative_z_scale_solve.py` (5 tests):
`(0.02, 0.02, -0.3)` with a centred 224-px pinhole projects to grid `[0.141, 0.859]`, in frame.

### Status: instrumented, NOT yet fixed - and deliberately so

`--require_positive_z` exists and is **OFF by default** (commit bf3cd25). The behind-camera rate
is now recorded under both settings, so the A/B arms are comparable. This ordering is on purpose:
the same plausible-story-then-fix pattern produced the retracted #59 "material improvement" claim
earlier the same day. **Validate criterion:** run both arms on a fixed sequence subset; the guard
is adopted only if it lowers the clamp-failure rate AND moves `s_med` toward 1.0 AND does not
worsen W/WA. If the behind-camera rate turns out to be ~0 %, this hypothesis is dead and the
18.4 % has another cause - record that outcome too rather than leaving the flag dangling.

### Also closed on the way

- **The registration-panel dumper was reimplementing the eval.** `dump_registration_steps.py`
  built its own model-load + forward + projection, and silently diverged: no bfloat16 autocast,
  no `cond_flags`. It emitted a constant hand depth of `-0.0205 m` on all three sequences (every
  hand behind the camera), which would have gone in front of the supervisor as Fig. 1's
  registration panel. It now owns **no forward pass at all**: `predict_clip` gained a `steps_out`
  hook and emits the intermediates from inside the eval's own solve. A figure meant to show what
  the eval does is now fed by the eval.
- **`object_depth_loss` was fine; the TEST was wrong.** Its prover reported 53 mm after the fix.
  `gs_depth` is stored 90 degrees rotated, so feeding the same array as prediction and GT is not
  an identity test. Measured all four rotations: only `rot90(k=-1)` round-trips to 0.0000 mm.
  Test corrected, plus a companion asserting the other three do NOT round-trip.
- **A latent `NameError` in `--dense_link`.** `_dense_scene_points` referenced
  `frame_width_from_intr` while the only import sat inside `predict_clip`. That path would have
  crashed the first time it ran. Found by pyflakes, now a module-level import.
- **Test-suite hygiene.** All 5 stale bug-prover failures resolved: 2 converted to
  `xfail(strict)` tracking the still-open #64 (so they flip to FAILURE the day it is fixed rather
  than rotting green) plus a new regression test for the actual production fix
  (`set_default_frame_width`); 1 tolerance corrected from 1e-6 to 1e-4 because it was testing
  float32 eps (7.6e-6 at cx~114) rather than the resize convention it names. **121 passed,
  2 xfailed.**


## 2026-08-06 - an adversarial bug hunt found 11 defects; 9 fixed; TWO producers now disagree with data on disk

Two agents were run against the tree with one rule: a bug counts only with a test that FAILS on
current code. They returned 11 findings and INDEPENDENTLY converged on the joint permutation.
Commits: 2eed3b8, 8fa7d44. Proof tests live in `tests/test_hunt_a.py` and `tests/test_hunt_b.py`.

### 🔴 WHAT IS OWED, and it is a DATA problem not a code problem (task #65)

Two fixes changed what PRODUCERS emit without regenerating what already exists:

  (A) INTRINSICS RESIZE. `h2o_to_currentproto.py:178` and `preprocess_hoi4d.py:97` rescaled the
      principal point as `c*s`; correct is `(c+0.5)*s - 0.5`. `dexycb_to_ours.py:409` was ALREADY
      correct and documented why. So a mixed-store run right now spans TWO conventions ~0.34 px
      apart. Stores on disk still carry the old form. DECIDE: rebuild H2O+HOI4D, or pin the old
      convention. Do not silently mix.
  (B) NATIVE BASELINE PREDS. Every `<seq>.pt` from `build_native_baseline_preds.py` written before
      2026-08-06 has PERMUTED FINGER BLOCKS. Those dirs feed build_slam_baseline ->
      eval_worldspace_baseline, i.e. the WiLoR+SLAM / HaMeR+SLAM rows.
      LIKELY NOT IN THE PAPER: a perfect prediction scores ~53mm root-relative through the bad
      map, and every reported baseline root-relative number is BELOW that floor (WiLoR 27.2,
      HaMeR 30.3, WiLoR+SLAM 33.4, HaMeR+SLAM 34.7). CONFIRM per pred-dir by rescoring one
      sequence both ways. Do not close this on the inference alone.

### ✅ The 9 fixed, most severe first

  1. joint permutation, `build_native_baseline_preds.py:50` - source order, not smplx-16. Same
     index SET so no shape/range check saw it. Now canonical + an import-time tripwire against
     the REAL constant (a local copy cannot detect drift in the original).
  2. `frame_width_from_intr` = 2*cx - MY OWN 38a5803 FIX, half wrong. HOI4D's principal point is
     NOT centred: 2*cx=228.56, 2*cy=217.05, true frame 224. 2.0% error. Now documented as a last
     resort and warns; callers should pass `image_width`. NOT fully closed - see task #64.
  3. half-pixel offset in `project_joints_to_norm_pixels`. grid_sample(align_corners=False) reads
     pixel `x*W - 0.5`, so pixel centre k needs `(k+0.5)/W`. `eval_world_space` already INVERTED
     with the +0.5 and claimed to invert "the EXACT projection", so the round trip was provably
     inconsistent. Verified from first principles after the fix: a joint on pixel-centre u samples
     that pixel exactly (31.0000 / 53.0000 / 58.0000 / 63.0000).
  4. `object_depth_loss.py:69` built (col,row) while every other gs_depth consumer uses
     [(W-1)-row, col]: same tensor read 189mm apart. Now identical; verified independently with a
     constant map and a ramp, residual 0.000e+00 over 147 points.
  5. `contact_mask.py:35` sampled GT depth (plain image layout) with the ROTATED grid: 438mm of
     error against a 50mm threshold, i.e. the gate was noise. `rotated=False` flag added.
  6. `hand_to_gs_injection.py:105` clamped the destination but not the source crop, so a
     half-off-frame box injected byte-identical content to a box half its width. Partially
     out-of-frame hands are the COMMON case in egocentric video.
  7. the loss-effect guard (built after C-abs-725) silently skipped any weighted-but-untracked
     loss and printed PASSED. `root_anchor` was never accumulated and 5 shipped configs set it to
     1.0. Untracked + weighted is now a FAILURE.
  8. `dexycb_to_ours.py` drift tripwire compared a local literal to an identical local literal, so
     it could never fire, while its docstring claimed it pinned against H2O's constant.
  9. `extract_h2o_clips_mp4.py:22` applied the H2O remap TWICE, putting fingertips in kinematic
     base slots. Still a permutation of 0..20, so the paired PA check could not see it.

### 🟢 IMPACT ON THE PAPER: the headline numbers are UNTOUCHED

`exp_p4_jitterrob.yaml` - the config behind every reported number - has `obj_depth` absent,
`root_anchor: 0.0`, `hand_depth_anchor: 0.0`, `use_contact_gate: false`, `contact_cache_dir: null`.
Those paths never ran. C-MPJPE absolute (35.8) and root-relative (26.6) do not route through depth
sampling or the scale solve, so they cannot move - the same reason the 1408 fix left them
bit-identical. W-MPJPE and WA-MPJPE carry the 2% width error plus the half pixel and MUST be
re-run (task #65).

### 🔵 THE METHODOLOGICAL LESSON, which is the transferable part

Four tests I added in 38a5803 were WRONG in two ways: they inverted the projection without the
pixel-centre offset, and used `cx=cy=W/2` fixtures that satisfy the very assumption under test.
Both are the same error - **a test written to match current behaviour instead of derived from the
contract**. That is exactly why bug 2 survived the fix for bug 1. Rule: when replacing a hardcoded
constant with a derived one, test the derivation against the REAL stores, never against a fixture
built from the derivation.

Also: the two hunters encoded CONTRADICTORY assumptions about axis order. Agent findings are
leads, not verdicts - verify independently before acting.


## 2026-08-05 - haptic_detbox_preds is NOT a box variant, do not use it

Attempting to raise the long-window table from 3-of-6 to 4-of-6 box-consistent rows, I reached for
`haptic_detbox_preds` (157 seqs, has `world_joints`) in place of `haptic_slam_preds`. Its seg100
numbers looked like a large win: **W 81.5 / WA30 21.6 / WA100 31.4** against the own-detector
dir's **W 143.4 / WA30 36.3**. C_abs barely moved (158.6 vs 157.7), which is the wrong signature
for a box change and is what triggered the check.

DIRECT COMPARISON of the two dirs on the same sequence:

  cam_joints    meanAbsDiff = 0.0000 m   **BIT-IDENTICAL**
  world_joints  meanAbsDiff = 0.1797 m   different

**The boxes are the same.** Identical camera-frame joints prove it. The two dirs differ only in the
camera trajectory used for the world lift, so the 62 mm W "improvement" is a trajectory swap, not a
box fix. `ours_gttraj` (GT trajectory) sits at W 41.3, and 81.5 lies between that and the SLAM
number, so the detbox variant plausibly carries an oracle or otherwise privileged track.

**VERDICT: do not put `haptic_detbox_preds` in any table.** The long-window table stays at 3-of-6
box-consistent. HaWoR and Dyn-HaMR genuinely need detbox v3 runs (tasks #30, #31), and HaPTIC now
does too - the dir whose name implied it already had one does not.

This is the second time a `+SLAM`-style pred dir has silently encoded a different trajectory. Any
row swap must compare `cam_joints` first: if they are identical, the dirs differ in the LIFT, not
the input, and the world metrics are not comparable.

### seg100 rebuild, all rows on matched flags (--drop_partial_tail ON, --hands both, wa_short 30)

  row              W      WA30   WA100   C_rr    C_abs    nseg   box
  ours_gttraj     41.3    18.0    22.9   24.8     33.3    180*   detbox v3, ONLY 60 of 157 seqs
  haptic_slam     81.5**  21.6    31.4   29.7    158.6    471    see above, NOT a box variant
  hawor          133.6    40.3    61.0   32.6     87.7    468    OWN detector
  hamer_fj2      150.1    36.8    57.8   14.5     37.4    468    detbox v3
  hamer_mt       166.4    41.3    63.2   39.7     67.3    468    detbox v3
  dynhamr        195.4    49.0    69.0   59.9   1336.7    468    OWN detector
  wilor          292.4    46.4    64.8   55.8   1074.6    468    detbox v3

  * ours is NOT comparable to the rest at 180 vs 468 segments; it needs a full 157-seq re-run.
  ** this row was scored from haptic_detbox_preds and is therefore trajectory-confounded; rescore
     from haptic_slam_preds before use.

Note `--drop_partial_tail` was OFF in the earlier tables_regen build even though the paper claims
it is used "throughout". It is ON here, so these numbers are NOT comparable to that build.


## 2026-08-04 EVENING - dataset pool relocked, and the 725 causal claim cut down to 4mm

### 🔵 abs2x2: absolute supervision is worth ~4mm, NOT ~700mm

Matched 2x2, HOI4D-only so it isolates the loss and not the data, kp2d zeroed in every arm,
n=314 segments each. Two of four cells in:

  arm                                       C_abs    C_rr
  kp3d_abs ON,  transl ON  (full recipe)     22.8    17.7
  kp3d_abs OFF, transl ON                    27.1    15.3

Dropping kp3d_abs while keeping the direct translation L1 costs **4.3mm** of absolute placement
and **improves** articulation by 2.4mm. The archived control that read **725 is dead** - that was
an untrained checkpoint, and an abs-unsupervised model and an untrained one both emit a constant.
Any sentence of the form "without absolute supervision the model collapses" must be rewritten as
"absolute joint supervision is worth about 4mm". Recorded into `3method.tex` at the CRITICAL todo.
Arms `absON_trOFF` (running) and `absOFF_trOFF` (queued) anchor the floor; hold final wording.
NOTE the arms run `--keep_gs_off`, so their W/WA are non-metric and must never be quoted.

### ✅ FINAL POOL LOCKED (Dario): 5 train, 2 eval, both eval sets fully held out

TRAIN: HOT3D 0.339 | OakInk2 0.386 | ARCTIC 0.474 | DexYCB (new) | Re:InterHand ego (new)
EVAL:  H2O 0.503 | HOI4D 0.677 - **neither is ever trained on**, so every number is zero-shot.

Ego-Exo4D REMOVED and deleted from disk: it ships only triangulated joints, never MANO (verified
against all 351 annotation files and the official schema), and was the shallowest store at 0.313.

**CASCADE: every mixture recipe built so far is void.** mix3/mix4/mix5/shallowmix each contain
HOI4D or Ego-Exo4D. So does the unfreeze sweep now running, which trains on mix3 - it is a
DIAGNOSTIC of layer depth only and none of its absolute numbers may be reported as the recipe.
Every headline we own (23.6, 35.6, mix3's 66.2, both world tables) was trained on HOI4D and is
now provisional pending a retrain on the locked pool.

### ✅ DexYCB depth gate PASSED before committing 119GB

Downloaded `calibration.tar.gz` (14KB) first and measured the rig from the extrinsics: 9 cameras
per session across 10 sessions, camera centre to camera-ring centroid **median 0.877m**, 10/90
percentile 0.597/1.276. That is DEEPER than HOI4D's 0.677, which is exactly the coverage the
mixture lost. Caveat: this is a rig-geometry proxy, not a hand-depth measurement; confirm against
the MANO annotations once a subject tarball lands.

### ⚠️ DexYCB download is Google-Drive QUOTA blocked, not credential blocked

All 10 subject tarballs return "Too many users have viewed or downloaded this file recently",
documented to clear in up to 24h. The 14KB calibration file downloads fine, so it is purely a
large-file quota. A detached retry loop (`dl_retry.sh`, one pass per 45 min, skips what is on
disk) is running on the login node. Re:InterHand is unaffected - it serves from a public S3
bucket over plain wget at ~90MB/s.

### Disk: 1.85 TB -> 1.43 TB

Freed ~420GB: Ego-Exo4D raw+converted+featcache (180GB) and the `hoi4d_32f`/`hoi4d_64f` feature
caches (196GB) from the CLOSED window-length sweep, which are derived and regenerable. Headroom
to the 2.5TB soft quota is now 1.07TB against ~353GB of incoming data.


Every parallel line of investigation, its status, and **what "validated / closed" means** — so no line is
abandoned without an explicit verdict. Update the Status + Verdict columns as results land.

Legend — Status: 🟢 running · 🟡 queued · 🔵 results-in-needs-verdict · ⚪ open/parked · ✅ closed(validated) · ❌ closed(refuted)

_Last updated: 2026-08-03. Sections A-E below are a Jul-5 snapshot with dated updates prepended;
read the 2026-08-03 hygiene pass FIRST - it retracts two numbers that later sections still cite._

## 2026-08-04 RESULT - depth-diverse mixing confirmed, and the 725 line finally closed

### ✅ The depth-prior prediction HELD, quantitatively

  run                                  C_abs    C_rr   n_seg
  HOI4D-only -> H2O zero-shot          184.8    59.7     757
  **mix3 -> H2O zero-shot**            **66.2** 38.2     757
  HOI4D-only in-domain (HOI4D-157)      23.9    18.0     314
  mix3 in-domain (HOI4D-157)            30.8    16.6     314

mix3 = HOI4D + OakInk2 + ARCTIC, H2O fully held out. Both H2O rows are full-coverage on identical
757-segment splits, matched protocol 16/8, so they are directly comparable.

The transfer law fit on the single-dataset case was `C_abs ~= 60 + 0.50 * |depth shift|`. HOI4D
alone sits 174 mm deeper than H2O, so the law charges ~87 mm over the intercept and it scored
184.8. mix3's depths (HOI4D 0.677, ARCTIC 0.474, OakInk2 0.386) BRACKET H2O's 0.503, the shift term
collapses, and the law predicts ~60. **Observed 66.2.** The prediction was written into the job
script before the run.

**The trade, which is the paper-ready framing:** mixing costs **6.9 mm in-domain** and buys
**118.6 mm out-of-domain**, roughly 17:1. Report both columns; the in-domain cost is what makes the
out-of-domain gain credible. **Control still missing:** a same-size mixture that does NOT bracket
H2O, to separate depth coverage from simply more data. Do not claim the mechanism in print without
it.

### ✅ The 725 line is closed

The HOI4D-only control from its TRAINED checkpoint reads **C_abs 23.9 / C_rr 18.0**, consistent
with the 23.6 headline. 725 was the untrained checkpoint and nothing else. The §2 caveat below
stands as history; the datapoint itself is now superseded by a valid number.

### Window length, matched arms (all retrained at their own length, 314 segs)

  16/8  C_abs 30.8  C_rr 16.6 | 32/16  C_abs 28.4  C_rr 21.2 | 64/32  C_abs 37.2  C_rr 28.7

32 frames buys 2.4 mm of absolute placement and costs 4.6 mm of articulation; 64 is worse on
everything, so C_abs has an interior optimum near 32. W/WA barely move, so window length is not a
trajectory lever. Keep 16/8 as the LOCK.

### Infrastructure fixed (three silent-failure classes)

- `solve_similarity` dropped a whole run on ONE non-finite joint (SVD raise -> all 157 sequences
  skipped, 1h37m of GPU wasted). Now drops non-finite rows; verified a NaN row still recovers the
  exact scale (258fa01).
- Eval configs inherited `enable_gs: false` from training, silently making every W/WA non-metric
  while exiting 0. `_make_eval_cfg.py` now forces it on, with `--keep_gs_off` for hand-only models
  that never had the branch (8df4d17).
- `build_feature_cache` ignored `min_labelled_frames`. On Ego-Exo4D that was the difference between
  **6,124 clips / 100.5 GB and 88,807 clips / 1457 GB**, against 1.15 TB of scratch headroom
  (0c356e5). Cache built and verified at 96 GB / 6,124 files.

## 2026-08-03 HYGIENE PASS - retractions, and one confounded causal control

### 1. ❌ RETRACTED: the `ctxgate` clip-length sweep

The sweep quoted all through late July as evidence that longer clips degrade C-abs
(29.7 / 30.2 / 30.7 / 32.2 for cl 16/32/48/64) scored **40 segments, not 157**. A 0.5 mm spread
over 40 segments is noise. The mismatch penalty for clip_len != train num_frames is **UNMEASURED**.
Retracted in code, configs and memory (`e5d7f1a`); it never reached `report/` or the paper.
`scripts/run_chunk_sweep.py` now exists specifically to keep matched (paper-reportable) cells
visually separate from mismatched (diagnostic-only) ones, so this cannot recur silently.

### 2. ⚠️ CONFOUNDED, not refuted: the `C_abs 725` kp3d_abs causal control

D2-2 and D2-6 both cite **absloss0 (job 102437) C_abs 725 / rr 131** as proof that absolute 3D
supervision is causally necessary. Two problems found on 2026-08-03:

- Memory (`cabs-725-untrained-checkpoint`) records **725 as untrained-model noise** in a *different*
  run (`ctrleval`), where `best_val_loss.pt` sat at `global_step=1` because `val_every 3000`
  exceeded the run's 2222 steps. absloss0 reads **724.95** - the same value to 3 s.f.
- absloss0's per-sequence spread is only **662-782 mm**, ±8% around the median over 60 sequences.
  That flatness is what a model emitting a near-constant depth looks like, independent of input.
  (Contrast a genuinely trained-but-degraded model: the H2O zero-shot run spreads 112-280.)

Crucially this does **not** show kp3d_abs is unnecessary. An abs-*unsupervised* model and an
*untrained* model both collapse to a constant, so 725 cannot distinguish them - the experiment
does not separate its own hypothesis from the null. `results/absloss0_eval.json` records no
`global_step` and the 102437 log is gone, so it is **unverifiable as archived**.

**Action:** re-run the control. The guards that would have caught this now exist on both ends
(trainer `val_every` clamp; eval `_assert_checkpoint_is_trained`, `MIN_TRAINED_STEP = 10`, added
`1a5e7d9`). Until then, do not cite 725. kp3d_abs necessity is independently supported by
`loss-recipe-kp3d-abs` (omitting the key silently trains with no absolute supervision - cost two
runs) and by the 23.6-vs-24.9 weight comparison, so the *claim* is not in doubt - only this datapoint.

### 3. Partial-eval audit (all 76 Euler result files with segment counts)

Published tables are **unaffected** - every table-bearing file is full-coverage (1425-1506 segments).
Partial and therefore diagnostic-only: `ctxgate_cl{16,32,48,64}` (40), `ours_denselink_128` (16),
`h2o_zeroshot_jitterrob` (17), `win32_eval` (8), `diagcam` (3). Root cause was
`eval_world_space --max_seqs` defaulting to **4**, which twice produced fake-complete numbers;
the default is now **0 = ALL**, coverage is printed, and the run shouts `PARTIAL EVAL` below half.

### 4. ✅ H2O zero-shot 162 mm INVESTIGATED - not a bug, mechanism measured

Suspected bug; it is a dataset depth-prior carryover. Four causes eliminated (GT joint scramble via
anatomical gate; the `s=1.000` scale-degeneracy, which invalidates **W only** and never touches
C-abs; focal, H2O 198.07 vs HOI4D 219.92 at 224 px = 11%, far too small; box convention, gated by a
hard precondition marker). The actual mechanism: HOI4D holds hands **184 mm** further from the
camera than H2O does (median RIGHT wrist depth 0.687 vs 0.503 m). Over 148 sequences,
`corr(|z_prior - z_GT|, C_abs) = +0.759`, slope +0.50, intercept 59.9 mm:

  **C_abs_zeroshot ≈ 60 mm + 0.50 × |z_prior - z_GT|**

Report the zero-shot number WITH this decomposition - it is a quantified limitation that directly
motivates depth-diverse multi-dataset training, not a defect. Detail in memory
(`h2o-zeroshot-depth-prior`).

### 5. ❌ W/WA from job 104218 (`h2oeval2`) are INVALID - two independent flags

The log carries both `NO rendered_extrinsics IN preds - falling back to IDENTITY camera poses`
(the `identity-camera-pose-bug`, so zero camera motion) and `SCALE DEGENERATE (s=1.0)`. Only
**C_abs and C_rr** from that job are usable. Note this is the same `enable_gs=False` trap recorded
as methodological trap 2 in the 2026-07-27 section - it has now bitten twice, so any H2O world
number needs `model.enable_gs=True` with `gs_anchor_only` doing the render skip.

### 6. Infra note: feature cache must filter like the trainer

`build_feature_cache.py` did not honour `min_labelled_frames` while the trainer did (`f1f81a0`),
so on a sparse store it would write a full-size cache for clips carrying no supervision at all.
Ego-Exo4D labels ~2.3% of frames at 16.8 MB/clip. Fixed in `0c356e5`: defaults to the config value
so cache and training cannot disagree, and hard-exits on an empty post-filter dataset.


## 2026-07-28 CORRECTION — identity-camera-pose bug invalidates the "ours" world rows below

`predict_clip` reads camera poses from `preds["rendered_extrinsics"]`, which is published by the
rasterizer. The `gs_anchor_only` fast path (added to dodge the gsplat hang on gb10) returns
*before* the rasterizer, so the key was missing and `predict_clip` silently fell back to identity
`c2w` — **zero camera translation and zero camera rotation**. Fixed in `9dd474d` by republishing
the camera head's `camera_poses` in the fast path, plus a loud warning on the fallback.

Caught by: all three scene-scale variants bit-identical on 314/314 segments (scale multiplies only
the camera translation) → `--diag_cam` showed `pred_cam_excursion=0.000000 m` vs 19–46 mm of GT
camera motion per clip → `s_gt_med` NaN everywhere, since that stat needs centre motion > 1e-4.

**Contamination window, dated from git**: the offending line entered `eval_world_space` in
`d8faf8d` (2026-07-27). Only runs from that commit onward are affected.
**Invalidated** (do not quote): ours online self-chained, the rot/trans decomposition of our own
trajectory, the **G1 dense-chain** verdict, and the "scene scale is neutral" verdict (definitively
wrong — variants were bit-identical on 314/314 pre-fix, 0/3 post-fix).
**Valid, predate the flag**: chunk-link (205.2/202.1), the gravity and full-rotation oracles, the
oracle-depth gate, re16/velGT, and the short-window world headline 36.2/33.2/37.3.
**Unaffected**: all baseline rows (HaMeR/WiLoR+SLAM, HaWoR, Dyn-HaMR — they never call
`predict_clip`) and every camera-frame C-MPJPE/C-abs number (they ignore `c2w`).

**Unlocked by the fix**: the hand-derived scene scale is now measurable against the true camera
scale — `s_hand_med` 0.858 vs `s_gt_med` 1.111, ratio **0.574**. We under-scale the camera
trajectory by ~43%, and since W/WA align rigidly without re-solving scale, that converts straight
into trajectory shape error. This is the new leading candidate lever.

## 2026-07-27 UPDATE — offline-SLAM lever closed, trajectory error decomposed

Two more long-window levers tested. Everything below is 128-frame segments, `wa_short` 30, one
scorer (`eval_worldspace_baseline`), matched 60-sequence subset, matched segment counts.

- ❌ **G1 dense-chain / MonST3R-style linking** (job 103689, 8-seq gate): W_dchain 108.6 vs
  W_global 111.2; the robust variant is worse (184.7). Per-clip dense scene geometry is itself
  drift-inconsistent, so solving seams from it cannot fix the global track. Do **not** build the
  windowed-graph optimizer.
- ❌ **Offline SLAM / BA** (Euler job 8794646). Composing our metric camera-frame hands with the
  DROID/HaWoR track gives **ours+SLAM W 128.1** (WA_s 27.4, WA_l 45.0, C_abs 32.6) against
  WiLoR+SLAM 143.2, HaWoR 147.2, HaMeR+SLAM 150.6. We are the best offline row, but SLAM does not
  rescue the long window — it lands near our own chained trajectory.
- ❌ **Trajectory-scale sub-lever** (job 8795576). Oracle-forcing each sequence's track scale to GT
  made W slightly *worse* (ours 128.1 → 137.1; HaMeR 150.6 → 159.7; WiLoR 143.2 → 151.8). The
  per-sequence scale is ill-conditioned (median 1.58, range 0.11–3.85) because egocentric HOI4D
  clips barely translate.
- ✅ **Oracle ceiling** (job 8796151). Our hands + the GT camera track: **W 40.8** (WA_s 17.7,
  WA_l 21.7). That is what a perfect trajectory buys, and it bounds every trajectory lever.
- ✅ **Rotation-vs-translation decomposition** (job 8798019). Same hands, swapping one half of the
  SE(3) for GT: SLAM R + SLAM t **128.1** → GT R + SLAM t **109.3** → SLAM R + GT t **80.3** →
  GT R + GT t **35.1**. Track error vs GT is 4.79° rotation / 102.5 mm centre. Translation is the
  larger single term, but neither half alone recovers even half the gap — the two are strongly
  **coupled** (same effect the gravity gate saw: predicted rotation and translation co-drift and
  partially cancel in hand placement). ⇒ a trajectory head must regress **full 6-DoF jointly**;
  rotation-only (gravity/IMU-style) or translation-only correction is provably capped at ~80–110.

Two methodological traps worth remembering, both of which produced wrong numbers before being
caught:
1. A "+SLAM" prediction dir can silently encode the **GT** camera track. The student-side
   `hamer_slam_preds` does (`audit_slam_trajectory.py`: 0.0 mm residual, Sim(3) scale 1.0000), which
   briefly made ours+SLAM look like W 40.8 = a 3× win. Always audit a pred dir's implied trajectory
   before quoting a world number from it.
2. `model.enable_gs=False` in a world eval silently zeroes the scene scale (s→1.0, "SCALE
   DEGENERATE" in the log) and inflates every W/WA metric. Keep it **True**; the gsplat render is
   already skipped via `model.gs_anchor_only`.

## 2026-07-26 UPDATE — world-space levers closed + Dyn-HaMR baseline

Long-window (128f) W-MPJPE lever hunt is COMPLETE — three candidate levers tested, all ❌ neutral,
so the bottleneck is now diagnosed and defensible:
- ❌ **Chunk-and-link chaining** (linker, incl. rigid variant): W_global 205 → W_link 202 (~1.5%).
  Neutral. `ours_chunklink_128_v4.json`.
- ❌ **Scene scale** (per-seq pooled/median): W_spool 202.7 / W_smed 201.6. Neutral.
- ❌ **Hand absolute depth** (oracle: perfect GT wrist depth, `--oracle_depth`): W 182.8 → 180.5
  (−1.3%) on a 50-seq subset. Neutral → **DA3 dense-depth world anchor CLOSED** (not the lever).
- ✅ **Diagnosis**: re16 (rigid re-anchor to GT every 16f) = 26 vs actual 182 ⇒ the long-window W
  bottleneck is **camera-trajectory rigid drift** (rotation+translation of per-clip predicted
  cameras), which no post-hoc op on hand predictions can touch. Only a better global camera
  trajectory (SLAM / scene BA ⇒ offline; scene-recon already refuted) would move it. For the paper:
  30f-window W is the headline (we win 37.3); the 128f long-window is a characterized limitation.

🔵→✅ **Dyn-HaMR input-matched baseline**: clean 30/30-seq re-run (batch-artifact freeze fixed) →
WA-MPJPE **48.1** (293 windows); absolute C/W omitted (input-matched setup ⇒ depth_constraint
inactive ⇒ degenerate world_scale, inherent not a bug). Table + PDF/PNG updated. Full-157 extension
chained on Euler (jobs 8612865→8628169→8628170, resumable, ~24h).

## HEADLINE RESULT (2026-07-05)

**Absolute camera-frame MPJPE (Hand3R's "C-MPJPE") = 23.6 mm on 157 held-out HOI4D sequences**
(winner10ep: 10-epoch kp3d_abs=1.0 cached retrain; root-relative 17.3; best val 20.5). Supersedes the
3-epoch 24.9 (kp3d_abs=0.5). Ckpt: `/home/dmonopoli/ckpt_backup/winner10ep_best.pt`, HF upload
`hoi4d_full/winner10ep_best_23p6mm.pt` in flight. Clean-152 / scene-disjoint-132 re-slices still to be
computed from `winner10ep_eval_test.json` per-seq entries (no GPU needed).
Reference: Hand3R 42.6 (online SOTA), HaWoR 51.8 (offline), Cyrus target 40.
Prior headline (3-epoch base, still the HF model-card number until updated): 24.9 / rr 18.6.
Robust under leakage discounts: clean-152 (warm-start seqs excluded) = 25.1, scene-disjoint-132 = 24.8.
Checkpoint + model card: HF `mondraaa/ff4dgs-ego-ckpts` (`hoi4d_full/base_best_mpjpe_3ep.pt`, val 18.7).
Recipe: frozen backbone, HaMeR-style head, warm start best_depth.pt, kp3d_abs 0.5, 3 epochs (7938 steps,
cosine annealed), best-of-val selection, 367-seq train / 157-seq test (stratified 7:3, seed 4023).

**The ~100 mm era is explained**: (1) undertraining — the 1-epoch runs did ~60–380 optimizer steps; a
3-epoch control with nothing else changed dropped 114→52 mm; (2) data — 11 train seqs → 367 took 52→24.9
(with kp3d_abs upweighted). The method was never the limit.

## A. Closed lines (verdicts on the old plan)

| # | Line | Status | Verdict |
|---|------|--------|---------|
| A1 | kp3d_abs sweep on the OLD 8-seq train | ✅ | Superseded: "tapped out ~99mm" was a data-scale artifact, not a lever limit. At 367 seqs the same lever reaches 24.9. |
| A2 | DA3 hand-depth probe | ✅ | DA3 at-wrist ~30–40mm after scale fix — but the head now beats it (~25mm), so DA3-as-reference is no longer needed for C-abs. Keep for W-space if useful. |
| A3 | Unified-model (unfreeze) | ❌ | CLOSED: partial-unfreeze added nothing over frozen at matched steps (52.3 vs 53.0 on 11 seqs). Frozen + head is the recipe. |
| A4 | Contact/DA3 anchor (RootDepthRefine) | ❌ | CLOSED: neutral-to-harmful in all fair tests (3-arm: 114 ctrl vs 122/125 anchored). Do not rebuild. Coupling story returns via B3 (global-context ablation) instead. |
| A5 | Scene-metric claim ("hand rescales scene") | ❌ | REFUTED (B2 eval 62→135cm). Scene recon = frozen 3rd-party, not a paper contribution (audit 2026-07-02). |
| A6 | Test-time smoothing / within-clip pose refine | ❌ | Dead (4% / harmful). |
| A7 | HOI4D data expansion (11 → 525 usable seqs) | ✅ | Done: yinloonga HOI4D_release.zip mirror + full handpose zip; preprocess bit-identical-validated; split 367/157 (seed 4023, `/home/dmonopoli/hoi4d_split.json`). |
| A8 | Feature cache (frozen-backbone tokens) | ✅ | Built: ~10x faster head training (scripts/build_feature_cache.py + data.feature_cache_dir). Corrupt-clip tolerant. Node /tmp only (wiped between jobs — rebuild ~4.5h, resumable). |
| A9 | kp3d_abs sweep at scale (0.25/0.5/1.0/2.0, 2-epoch cached arms, 60-seq eval) | ✅ | **kp3d_abs=1.0 wins** (32.9 vs 38.5/46.1/36.4). Single-seed; the 0.5-arm reading is noise-suspect. Arms are ranking-only (undertrained vs live 3-epoch). |
| A10 | Leakage audit of the 367/157 split | ✅ | Train∩test=0. 5 test seqs seen by warm-start: effect +0.2mm (negligible; report clean-152). 25 sibling-take seqs: scene-disjoint-132 = 24.8 (no change). |
| A11 | Fig. 5 qualitative + 3D error figure | ✅ | REDONE 2026-07-04 with the real Figure-5 pipeline (`blender_figure5.py --matte --film_white --orbit --hand_clear`, assets `fig_hoi4d/fig5_hoi4d/`): `report/figures/hoi4d_fig5_gt_vs_pred.png` + `_alt.png` (wavy splat scene + matte pink GT / blue pred MANO meshes, frame 207, winner2ep ckpt — caption must not claim the 24.9 ckpt) and `hoi4d_test_error3d_side.png` (skeleton overlay, 25.0mm frame). |

## B. Running / queued (cluster job chain)

| # | Line | Job | Status | Validate / close criterion |
|---|------|-----|--------|----------------------------|
| B1 | **Winner promote**: winner@2ep full-157 eval, then **10-epoch kp3d_abs=1.0** cached train + full-157 eval | 102007 | ✅ | **10ep = 23.6 abs / 17.3 rr full-157 — new headline** (val best 20.5). winner@2ep full-157 = 34.2 (confirms sweep arms were ranking-only/undertrained). More epochs kept helping through 10; curve not obviously saturated → a 20ep probe is a cheap C-line if we want the last mm. |
| B2 | **Bbox robustness eval** (jitter:0.2 + fixed:0.30, winner10ep ckpt, TEST60 stride 16) | 102058 | 🔵 results-in | jitter:0.2 → **31.8 abs / 20.2 rr** (clean ≈ 22–23 on this subset): degrades ~+9mm but still well under Hand3R 42.6 → external claims survive detector-like noise. fixed:0.30 → **43.5 / 35.4**: box geometry is a real cue for BOTH depth and articulation (crop mis-sizing hurts everything, not selectively depth). C3 jitter-robust retrain queued (102084). |
| B3 | **Global-context ablation** (crop-only head, cached 2ep, kp3d_abs=1.0) | 102058 | 🔵 results-in | croponly 2ep = **36.7 abs / 26.8 rr** vs matched-budget cpush 2ep 32.9 / 23.9 → context helps ~+12% but UNIFORMLY (abs and rr equally), NOT selectively absolute. Strong "metric-from-scene-context" claim NOT supported as-is. Caveat: ROI-aligned crop tokens come from the globally-attended backbone feature map, so "crop-only" still carries implicit scene context — ablation is weak by construction. 10ep matched rerun in 102084; framing must soften to "scene-image features suffice; global cross-attention adds a uniform ~12%". |
| B4 | **Data-scale curve** (11/50/150 train seqs @ ~2700-step budget) | 102058 | ✅ | **11→55.0, 50→43.2, 150→34.3, 367→32.9 (matched budget) / 23.6 (full 10ep)** on TEST60/full. Clean monotone curve. Key split: root-relative saturates by ~50 seqs (25.1→25.0→23.9) while ABSOLUTE keeps improving with data → metric placement is the data-hungry part. Not saturated at 367 → more data (multi-rig, other categories) is a live lever. |
| B5 | **Multi-rig zero-shot** (60 seqs each from rigs 002/003/004, eval-only, rig-001-trained ckpt) | 102059 | ❌ blocked | 102059 FAILED: rig-002 seqs all die in preprocess (`recover_K: no usable kps2D frames` — handpose annotation layout differs from rig 001 under `ZY.../H2/...`). Needs a layout adaptation pass; parked behind higher-value lines. |
| B6 | **World-space W/WA-MPJPE** with winner10ep | 102118 | ✅ | **W 195.6 (was 353), WA-long 54.7, WA-short 22.4, re-anchor@16 24.1** (40 seqs × 2×128-frame segs, robust_scale). vs Hand3R 86.9 WA / 125.8 W: we now WIN aligned-world (54.7 < 86.9) and lose raw W (195.6 > 125.8) — chaining drift dominates (velGT oracle = 55.0 confirms per-frame poses are fine, trajectory accumulation is the residual). C-abs on this subset 23.2 ✓ consistent with headline. Next W lever if wanted: re-anchoring / drift correction, NOT better hand pose. |
| B7 | **C3 jitter-robust retrain** (train WITH jitter:0.2, 10ep winner recipe) + **detector-box E2E eval** (real WiLoR boxes) | 102132→102386 | 🔵 results-in | Train succeeded Jul 6 (102132) but /home quota zeroed all eval JSONs; eval-only recovery 102386 ✅ 2026-07-09. **jitterrob (TEST60): clean 26.9 / jitter:0.2 27.3 / fixed:0.30 31.2 C-abs** — jitter training buys near-total box robustness (+0.4mm under jitter vs +9mm for winner10ep; fixed-offset penalty 43.5→31.2) at ~3mm clean cost vs winner10ep (~23–24 on this subset). **detbox_eval (winner10ep + real WiLoR det boxes): C-abs 140.6 / rr 40.7** — degradation is UNIFORM (median 143.5, 50/60 seqs >100mm, 0 >300mm; clean-good seqs 20-30mm jump to 170-275) → NOT a few catastrophic seqs. DIAGNOSED (IoU study, 18k frames): det-vs-GT meanIoU 0.383, 18.8% frames IoU==0, det boxes 1.53x too wide. Three input flaws, all in the det-cache builder: detection ran on 224px frames (hand 30-50px → 80.5% recall + sloppy boxes), expand_box never squares (GT protocol = tight×1.5→square), carry-forward fallback decays to IoU 0 on long miss runs (S179 seqs: 51-72% frames IoU<0.3 → 233-275mm). So 140.6 measures a broken input regime, not the model. Fix = **detbox v2**: detect on the HD 1080² frames (Euler HD store being built for HaPTIC), scale boxes into the 224 store coords, apply the exact training protocol, carry-forward on miss; re-eval winner10ep + jitterrob. jitterrob×flawed-detbox cell (102395) still lands as the training-side-mitigation datapoint. Crop-only 10ep ckpt was never banked (nobank) → lost; retrain only if the ablation row is needed. |

**2026-07-05 ~12:08 incident**: /home hit the 20GB quota during 102059's multi-rig preprocess (two 4.7GB
ckpts + preprocess output). All /home writes failed silently → 102059's world result lost (0-byte json),
102093 baselines died at startup (empty log), 102094 c3 config writes failed. Recovered: winner2ep.pt
deleted (~5GB free). Lesson: bank at most ONE 4.7GB ckpt in /home at a time; HF is the durable store
(winner10ep upload verified OK).

**2026-07-05 ~18:15 second incident (self-inflicted)**: quota cleanup deleted `hoi4d_pp_full` (636MB),
which was the LIVE preprocessed store — `hoi4d_train/`+`hoi4d_test/` are symlink farms into it. No /home
snapshots. All inputs survive (split json, handpose zip on scratch, mirror mp4s, deterministic validated
preprocess) → regeneration running (102116, ~4h), banked numbers unaffected. Also fixed en route: WiLoR
baseline install (wilor-mini is git-only, not PyPI) + HaMeR model_config.yaml fetch (team25 has bare ckpt).
Lesson: before rm of any data dir, check inbound symlinks from consumers (`ls -l hoi4d_test/ | head`).
Outcome: regen 102116 completed in 42 min, 524/524 bit-identical store restored + extrinsics 157/157.
World eval 102118 ✅ (see B6). Baselines attempt 2 (102117) failed on chumpy build isolation (aborts the
whole wilor-mini git install) + missing pyrender for HaMeR → fixed (numpy<2 + setuptools<70 first, chumpy
--no-build-isolation, opencv 4.x pin, +pyrender/scikit-image/trimesh), attempt 3 = 102124, then c3 = 102125.

## C. Open / planned (next 1–2 weeks)

| # | Line | Depends on | Criterion |
|---|------|------------|-----------|
| C1 | Baselines (HaMeR, WiLoR) rerun on OUR split | port from H2O baseline work | ✅ **DONE 2026-07-06 (job 102128, all 157 test seqs, same-input 224px protocol, true-focal conversion)**: **WiLoR C-abs 218.2 / rr 67.9 / wrist 208** (det recall 80.5%, fallback 19.5%); **HaMeR (same WiLoR det boxes) C-abs 187.9 / rr 72.6 / wrist 188**. Ours: **23.6 / 17.3** → 8–9x absolute gap ON THE SAME SPLIT; the cross-split asterisk vs Hand3R 42.6 is now bracketed by two same-split crop baselines. Protocol caveats to report: 224px source frames penalize crop models on articulation (their rr ~70 vs typical ~25–40 on HD); absolute failure is structural (weak-persp), not resolution. HaMeR ckpt loaded strict=False with HF-space config on team25 bare ckpt — sanity-check key-match count before final table. **Hand3R has NO public code/ckpt/split-list** (arXiv 2602.03200) → quoted-with-footnote. HaWoR (public, sm_75-buildable) = stretch row. **UPDATE 2026-07-09 (job 102397, rr joint-order fix + per-SEQUENCE agg — these supersede the numbers above): WiLoR C-abs 206.3 / rr 32.2 / wrist 208.2 / PA 7.1; HaMeR C-abs 176.8 / rr 39.9 / wrist 188.0 / PA 8.1.** The old rr ~70 values were the joint-order bug, not a resolution penalty — with the fix their rr is a normal 32–40, so drop the "224px penalizes articulation" caveat and note instead that ours (23.6 abs) beats their root-relative. |
| C2 | 21-joint (fingertips) metric variant | none (tips via MANO vertices, hand_vis_utils) | Match Hand3R's joint count if theirs is 21; expect a slightly higher reading. |
| C3 | Bbox-robust retrain (train WITH jitter) | B2 outcome | → PROMOTED to B7 (job 102084, queued 2026-07-05): B2 deflated under jitter (+9mm). |
| C4 | H2O cross-dataset check | existing H2O pipeline | Second-dataset row. SCOPED 2026-07-09: the old H2O pipeline (pack_h2o npz + train_h2o_hand + eval_cmpjpe) is protocol-incompatible (per-clip agg, own loss, 21-joint remap); the ONE blocking build = npz→current-protocol store adapter (per-seq dirs w/ video_main_rgb.mp4 + hand_data caches), then winner recipe runs verbatim. Split = train subj1-3/test subj4; the banked 58.0 C-MPJPE is MIXED-SET (leakage caveat in run_b3_h2o_subj4.sbatch) — do not quote. h2o_packed store on scratch may need re-pack (dl_h2o.sh). **STATUS 2026-07-10: adapter BUILT + gate-validated (synthetic, 14/14 checks + negative control) and deployed to cluster (md5-verified); h2o_packed store GONE and scratch write-locked → re-pack plan = stream subject tars to /home. BLOCKED: h2odataset.ethz.ch returns HTTP 401 with the stored creds (worked for subject4 earlier) → credentials expired/rotated. NEEDS DARIO: re-register / refresh H2O dataset access (https://h2odataset.ethz.ch), then place creds on the cluster; everything downstream is staged and auto-runs (pack subj1 → convert+validate → gate report → review → train).** **UPDATE 2026-07-19 — VALID FRESH-TRAIN RESULT LANDED:** first result (job 102879: 64.4 C_abs with C_rr>C_abs) was a GT joint-remap SCRAMBLE bug (21→16 selector composed for the wrong layout; thumb slots held fingertips); fixed (H2O16_IDX=H2O_TO_MANO[:16] in eval_cmpjpe.py + h2o_to_currentproto.py, new anatomical bone-length gate 4), store fully reconverted (102962, 611 gate-4 PASS / 0 FAIL), retrained (102963, kp3d cured 0.08→0.006). **Corrected: train subj1-3 → test subj4 (45 seqs, right hand, same protocol): C_abs 48.2 / C_rr 25.6 / wrist_abs 44.2 mm** (results/h2o_fresh_eval.json; scrambled artifacts parked as *_SCRAMBLEDGT_102879). Establishes the recipe transfers to a 2nd dataset; NOT yet calibrated (no H2O baseline) — reviewer pass 6 M3: minimum = metric-tuned HaMeR on H2O; best = HOI4D→H2O zero-shot for both our head and fu-HaMeR (unrun; needs box-convention bridge + fu ckpt banking + ls_polle/Cyrus sign-off). ALL pre-fix H2O numbers computed through eval_cmpjpe's h2o remap (incl. the banked 58.0) are void. **AUDIT 2026-07-19 (job 102988):** 48.2 has a per-SCENE placement bias (group medians k2 32.1 / k1 33.8 / h2 43.3 / o1 50.1 / h1 63.2 / o2 64.8; C_rr uniform) with intrinsics (f=198.1 uniform), depth range (med_z 0.39-0.42 all groups, corr -0.37), box size, and train/test depth coverage ALL ruled out. Prime suspect = the kp2d loss (train_hand_head.py hardcodes Aria 1408px + 90° rotation; active at weight 0.05 in every recipe; raw 13.7 on H2O ≈ the entire train loss vs 2-6 on HOI4D). Control IN FLIGHT: job 102989 = same recipe + loss_weights.kp2d=0.0 → results/h2o_kp2d0_eval.json. If h1/o2 recover, the honest H2O number improves and an HOI4D kp2d=0 control becomes worth running (the 23.6 headline also trained with the term active). **kp2d0 RESULT 2026-07-19 15:03 (job 102989): artifact CONFIRMED but PARTIAL. Per-seq C_abs 48.2 → 44.6 (−3.6mm, −7.5%); train_loss collapsed 0.695→0.0077 (kp2d WAS ~99% of it). Per-scene medians kp2d0 vs baseline: k2 31.1/32.1, k1 32.6/33.8, h2 41.2/43.3, o1 50.0/50.1, h1 54.0/63.2 (−9.2), o2 58.8/64.8 (−6.0) — gain concentrates in exactly the two worst scenes; good scenes flat. So the mis-specified kp2d loss does bias absolute placement in hard scenes, but a large scene-dependent residual survives (h1/o2/o1 50-59mm vs k 31-33mm). CAUTION: the streaming "hand_metrics(all) MPJPE=24.9" is frame-pooled, NOT the comparable per-seq C_abs (44.6). ADOPT kp2d=0 as the cleaner H2O recipe (44.6). Residual candidates: scene-dependent H2O GT quality or genuine appearance difficulty. NEXT (needs Dario sign-off, cheap): HOI4D kp2d=0 control — 23.6 headline trained with same wrong term (HOI4D raw kp2d 2-6 → expect smaller shift, but headline is load-bearing → defensibility). Head banked ckpt_backup/h2o_kp2d0_head.pt.** **HOI4D CONTROL LAUNCHED 2026-07-19: job 103044 (auto/hoi4d_kp2d0.sbatch), two MATCHED arms at headline seed 42 sharing one recon cache — off (kp2d=0.0) then on (kp2d=0.05 shipped default), both full-157 GT-box via eval_hand_cam_anchor. Results results/hoi4d_kp2d_{off,on}_eval.json, marker HOI4D_KP2D0_ALL_DONE. Compares against banked 23.6±0.8; the matched on-arm removes node/cache confound so the delta reads above ~0.8mm seed variance.** |
| C5 | Fine kp3d_abs grid (0.7/1.4) + multi-seed | B1 | Only if 10ep suggests weight sensitivity matters at full training. |
| C6 | Multi-rig TRAINING | B5 outcome | Only if zero-shot degrades badly. |

## D. Paper framing (draft thesis, for Cyrus)

**"Feedforward scene reconstructors already carry the information for metric-absolute hand placement —
we identify it, extract it with a lightweight head, and beat dedicated online systems 2x."**

In hand: 24.9 vs 42.6 headline; error anatomy (articulation solved: rr 18.6, PA ~5–8; absolute placement
was the residual); supervision lever measured (A9); explanation of the ~100mm era (undertraining + data).
Pending: WHERE the signal lives (B3 scene-context; B2 box-size cue), W-space transfer (B6), cross-rig (B5),
data curve (B4). Negative results to include: hand-as-global-scale refuted, anchors neutral, smoothing
dead, unfreeze unnecessary — they sharpen the analysis framing.

Integrity notes that ship with any claim: GT-derived boxes until B2/C3 close; single-rig until B5;
16-joint metric until C2; our split ≠ Hand3R's split (protocol matched, lists differ, theirs unpublished).

### D2. CVPR-reviewer verdict (2026-07-09, two independent passes: publication-potential + protocol audit)

**Verdict: Reject as constituted; items below get to Borderline+/main-track-viable.** Full text in
session memory (`cvpr-review-verdict-2026-07-09`). THE attack: oracle GT boxes (box size = depth cue;
fixed:0.30 → 43.5 > Hand3R 42.6, so the "win" disappears without GT localization) + zero fair
absolute-placement baseline actually run (WiLoR/HaMeR are weak-persp straw men; their abs numbers are
harness artifacts — WiLoR H2O PA 47.6 but abs 614.9). New actionable protocol bugs from the audit:

| # | Finding | Fix | Status |
|---|---|---|---|
| D2-1 | Aggregation mismatch: ours = per-seq mean over seqs; baselines = per-128-frame segment mean (eval_hand_cam_anchor.py:113 vs eval_worldspace_baseline.py:104) | one scorer, one aggregation, regenerate table | OPEN |
| D2-2 | Baselines never given GT boxes (box-source asymmetry both directions) | rerun WiLoR/HaMeR WITH our GT boxes + ours with detbox v2 = parity both ways | **HALF-DONE 2026-07-10 (job 102436, TEST60, per-seq):** WiLoR GT-box C_abs 243.9/rr 25.8 (det-box same split: 205.8/27.8); HaMeR GT-box 164.7/33.1 (det-box: 166.7/36.6). GT boxes fix misses + help rr but abs does NOT close (HaMeR flat, WiLoR 38mm WORSE → weak-persp depth is calibrated to its own detector's box stats = box-convention artifact). Baseline absolute failure is structural. Remaining half = ours on detbox v2 (Euler chain). **Reviewer pass 2 (2026-07-10) adds:** (a) the GT-box-worse result opens an OOD-box counter-attack ("you fed WiLoR a foreign box convention then called it structural") — fix = run WiLoR/HaMeR in their FULLY NATIVE regime (native detector+padding+crop res+focal, full-res HOI4D) as the primary baseline row, demote det/GT-box rows to a box-convention ablation, state the tz=2f/(box*s) mechanism explicitly; (b) conclusion-bearing parity table must be full-157 (or pre-declared identical subset for ours+baselines), TEST60 is diagnostic-only; (c) reviewer's detbox v2 calibration for reference: E2E <=~50mm keeps the claim alive, ~80mm borderline, ~140 fatal. **Dario 2026-07-10: (a)+(b) APPROVED and dispatched to the baseline agent (Euler es_tang, native detector + native padding + native crop res + scaled_focal_length translation as primary, true-focal logged as secondary, full-157, per-seq agg); (c) pre-registered headline policy NOT adopted — headline framing decided after seeing detbox v2.** **Progress 2026-07-10 eve:** first Euler jobs (6576437/6576424) died at evaluator selftest = missing `dill` in venv recipe (env gap); fixed + resubmitted as 6605600 (native full-res E2E) + 6605597 (WiLoR GT-box full-157). GT-box parity COMPLETE on full-157: WiLoR GT-box 240.0 abs / 30.5 rr, HaMeR GT-box 168.3 / 36.0 — full split confirms TEST60 (GT boxes don't rescue abs; WiLoR ~34mm worse than own det boxes). **NATIVE E2E DONE (6605600, 156 seqs): WiLoR true-focal 83.4 abs / 27.2 rr; HaMeR true-focal 88.0 / 30.3; published-conversion rows ~24.5m (non-metric, document as such). Native det recall 91-100% vs 80.5% on 224px crops. Crop-regime 170-240 numbers were input artifacts — primary comparison is now ours-GT-box 23.6 vs native-E2E 83-88; detbox v2 (resubmitted 7152642 after dill fix, prior 6737582 was a FALSE GREEN) supplies our honest E2E side. **CLOSED 2026-07-12 (baseline agent final report):** W-lifting question ANSWERED from job code — all baseline W rows (WiLoR 62.5 / HaMeR 84.0, native AND GT-box) use GT extrinsics (cam_extrinsics_cache.pt, X_w=R^T(X_c−t), EXTR_SANITY 0.000mm) = oracle-camera tier, NEVER protocol-comparable to our own-trajectory W 195.6 (footnote as oracle-lifted or drop). Dropped seq = ZY20210800001_H1_C14_N34_S207_s03_T2 (transient mirror pull miss, recorded in seq_misses, recoverable). Deviation flags for the paper: HaMeR own-detector infeasible (ViTDet/mmcv-1.x vs torch-2.3 harness) → WiLoR detector + HaMeR-native rescale 2.0 substituted, footnote required; HaMeR GT-box rows = tight GT box × native 2.0 rescale. absloss0 control (102437): C_abs 725 / rr 131 (TEST60) — kp3d_abs causally necessary. **⚠️ 2026-08-03: this datapoint is CONFOUNDED, do not cite — see the 2026-08-03 hygiene pass §2.** JSONs: Euler $SCRATCH/results/{wilor,hamer}_native[_truefocal]_eval.json + native_fullres_inline.json + *_gtbox157_*; student /home/dmonopoli/results/. Remaining in D2-2: detbox v2 only (7152642). HaPTIC HD: C_rr 25.7 valid; C_abs 6360 still published-units, needs true-focal pass.** |
| D2-3 | Hard-coded "(Hand3R 42.6)" in eval printouts (eval_world_space.py:616, eval_hand_cam_anchor.py:208) invites invalid direct comparison | remove; footnote "quoted, not comparable" | OPEN |
| D2-4 | root_depth_anchor ref_scale 0.892 fit on test GT — every anchor result is test-set-tuned (headline unaffected) | refit on disjoint calibration split before citing any anchor number | **TOOLING DONE 2026-07-10, run pending.** Contamination CONFIRMED: probe (da3_hand_probe v2, job 101840) fit on 8 of the old 11 seqs; 5 of those 11 are in the current 157 TEST split, and the anchor A/B evals of that era ran on the same 11 (fit-on-eval). Note 0.892 = measured DA3/GT ratio; the applied constant is 1.121 = 1/0.892. Built: `scripts/fit_ref_scale.py` (train-only refit, hard-fails on any test/unknown seq, seq-level bootstrap CI) + `ANCHOR_REF_SCALE` env override in root_depth_anchor.py (non-breaking). Blocker: DA3 wrist caches exist only for the old 11 seqs (5-6 train) — for a tight CI, build train-split DA3 caches first (gb10, venv_gb10 + /tmp/da3libs recipe). Then rerun anchor evals with ANCHOR_REF_SCALE=<refit>. Headline C-abs unaffected (anchor off). **REFIT DONE 2026-07-10 (job 102463):** 40 train seqs / 966 frames -> median ratio 0.9571, **ref_scale = 1.0449 (95% CI 1.0075-1.0678)**. The old test-fit 1.121 is OUTSIDE the CI = contaminated AND over-correcting ~7%. Any cited anchor number must be rerun with ANCHOR_REF_SCALE=1.0449 (env override, no code edit). JSON: /home/dmonopoli/ref_scale_refit.json. |
| D2-5 | hoi4d_to_haptic.py:77 sx=W/(2cx) conflates principal point with focal scale — near-exact for our center crops (error ~0.1%), fragile in general | scale f,cx,cy by true resolution ratio, keep off-center pp | polish |
| D2-6 | Missing ablations: backbone-swap (frozen recon vs frozen DINO vs random-init) = THE novelty defense; kp3d_abs zeroed-loss causal control | run both pre-submission | **kp3d_abs control DONE 2026-07-10 (job 102437 absloss0, TEST60):** zeroing ONLY kp3d_abs (kp3d-rel 0.05, kp2d 0.05, transl 1.0, priors all unchanged vs winner recipe) collapses C_abs to 725mm (winner ~23.6) and C_rr to 131 — absolute 3D supervision is causally necessary; features + relative losses alone do not yield absolute placement. JSON: /home/dmonopoli/results/absloss0_eval.json. **⚠️ 2026-08-03: CONFOUNDED — 724.95 matches retracted untrained-model noise to 3 s.f., per-seq spread is a flat 662-782, and the JSON records no global_step while the log is gone. Unverifiable as archived; needs a re-run under the new trained-checkpoint guards. See the 2026-08-03 hygiene pass §2 (the kp3d_abs claim itself stands on other evidence).** **Backbone-swap DONE 2026-07-11 (jobs 102484/102485), null hypothesis CONFIRMED:** full-157 GT-box per-seq C_abs/C_rr — recon 23.6/17.3, frozen DINOv2-L **21.9/14.8 (BEATS recon)**, random-init frozen 27.7/18.3. "Reconstruction features encode metric depth" is dead as a novelty claim; performance comes from the recipe (kp3d_abs + GT-box geometry + crop tokens + 367 seqs). Deviations to state: new arms trained head from scratch (favors recon row — DINO won anyway); DINO C=1024 vs 2048. JSONs results/bb_{dino,random}_eval_test.json. Contribution narrative must reposition to recipe/protocol/analysis; detbox v2 becomes THE load-bearing number (random-arm strength corroborates GT-box-as-depth-cue). **2026-07-15: ordering now needs 3-seed error bars to be citable (see D2-7). detbox v2 boxes BUILT (job 7199201, 157/157, mean det recall 0.829) but eval 102815 KILLED 2026-07-16 at n=94 (partial C_abs 160.1, INVALID): surgical audit found build_detboxes_v2.py used the HOT3D square-to-max-side+clamp convention while the HOI4D store (preprocess_hoi4d.py:444-455) is rectangular x1.5 per-dim UNCLAMPED, and training/hamer_head consume the stored box verbatim ([cx,cy,w,h] geometry injection = the depth cue) — convention mismatch, not detector quality (meanIoU 0.245 = the tight-rect-vs-1.5-square ceiling; C_rr only 41.5 vs 17.3). Also full-frame [0,0,1,1] boxes before first detection. detbox v3 BUILT 2026-07-16 (Euler 7303584; 7297824 requeued at 8G past a ~32G/user QOS mem cap): n=157, mean_recall=0.829, meanIoU=0.346 (vs v2 0.245 — convention fix confirmed), BUT med_w_ratio=1.703 / med_h_ratio=1.677 → detector boxes ~1.7x GT size per dim (silhouette+margin vs joint hull), predicts elevated raw C_abs via the [w,h] depth cue; jitterrob (2nd ckpt in same eval) is the trained mitigation. Shrink factor would need TRAIN-side HD frames (test157_hd only exists) — small train-HD pull via the mirror if needed; fitting on test is forbidden. Boxes md5-verified to student (05a86069), EVAL RUNNING job 102825 (farm 157/157, winner10ep then jitterrob, results detboxv3_{winner,jr}_eval.json); seeds preempted w/ user approval + resubmitted (102826 w10, 102827 dino). Reviewer pass 3 tightened the accept bar: detbox v2 ≤~50 accept-tier, ~50-80 borderline, >~90 fatal (must beat FAIR native baselines 83-88, not the old 188-218). Paper draft repaired: 218/188 removed from headline → native 83.4/88.0; DINO row added; Hand3R marked non-comparable; hard-coded Hand3R printout removed from eval_world_space.py. **detbox v3 WINNER RESULT 2026-07-16 15:45 (job 102825, full n=157, per-seq agg): C_abs = 49.8 / C_rr = 41.0 — UNDER the ≤50 accept-tier line, beats metric-tuned HaMeR det-box 68.7 by 27%, beats native 83.4/88.0. The predicted 1.7x-box-bias depth collapse did NOT happen (~2/157 seqs >90). E2E claim HOLDS in both regimes (ours 23.6 GT / 49.8 det vs mt-HaMeR 39.8 GT / 68.7 det). jitterrob det-box pass running next in same job (detboxv3_jr_eval.json). **jitterrob RESULT (ALL_DONE 17:41): C_abs = 35.6 / C_rr = 26.5 full-157 — new headline E2E row. Beats mt-HaMeR det-box 68.7 by 48% and even mt-HaMeR GT-box 39.8; C_rr 26.5 beats native full-res WiLoR 27.2, killing the pass-4 "worse articulator E2E" MAJOR without a retrain. Box-scale-aug retrain demoted to optional polish. Open cell: jitterrob full-157 GT-box clean number (TEST60 26.9). detbox line CLOSED.** |
| D2-7 | Single-seed everywhere; jitter sensitivity shows variance matters | 3-seed final recipe, mean±std | **IN PROGRESS 2026-07-15 (task #15, agent a00c4b1):** reviewer pass 3 upgraded this to MANDATORY — the backbone-swap ordering (DINO 21.9 / recon 23.6 / random 27.7) is plausibly within single-seed noise, so the D2-6 analysis conclusion is UNLICENSED without error bars. Staging 2 extra winner10ep-recipe seeds (existing 23.6 = seed 1) + 2 DINO-arm seeds if GPU budget allows (259/800 team-h). Queue discipline: submits ONLY after H2O training (h2o_train10ep) is in the student queue (H2O priority). Deliverable = mean±std per arm + whether DINO-beats-recon survives. w10_seeds.sbatch staged. |
| D2-8 | "It's just a metric-tuned HaMeR" — the natural next reviewer attack after the backbone null | fine-tune ONLY WiLoR/HaMeR camera/transl head with kp3d_abs on our 367 train seqs, eval full-157 | **IN PROGRESS 2026-07-15 (task #16, agent afdaaacd, reviewer pass 3 recommendation):** freeze all but the weak-persp cam/transl head, kp3d_abs L1 on abs cam-frame 3D kpts, same 367-seq train data + true-focal lift; eval own-detector boxes (primary, native regime) + GT boxes (secondary). Runs on Euler es_tang (idle, independent of student QOS), ~6h head-only cap. Either outcome publishable: gap closes → confirms recipe-not-backbone cleanly; gap holds → our arch genuinely helps. Pre-empts the strongest post-null attack. **SUBMITTED 2026-07-15 job 7284898 (Euler es_tang). Two documented deviations: (1) target HaMeR not WiLoR — WiLoR-mini's pred_cam is fused inside the ViT backbone (no isolable weak-persp head, no train entrypoint); HaMeR's MANOTransformerDecoderHead has a clean `deccam=nn.Linear(dim,3)`. Unfreeze ONLY deccam (weak-persp [s,tx,ty]); backbone/decoder/decpose/decshape/MANO frozen → C_rr invariant by construction, only absolute translation adapts. (2) Run the 224-store GT-box regime (= our winner's exact 23.6mm headline regime) instead of native HD — streaming 524 HD mp4s in 6h is infeasible w/ mirror IncompleteRead flakiness, and 224-GT-box is the MOST DIRECT rebuttal (metric-tuned HaMeR vs our headline, same regime); native HD already covered by the 83.4/88.0 rows. So GT-box=primary, det-box=secondary + a no-tune GT-box control measuring the C_rr-invariance. Caveat for the paper: unfreezing only a 3-output Linear is the cleanest isolation but a reviewer could call it too narrow — if the result is ambiguous, consider also unfreezing part of the decoder. Row lands as "HaMeR deccam kp3d_abs fine-tune".** **DONE 2026-07-16 (job 7284898, all 3 eval passes, per-seq agg): tuned GT-box C_abs=39.8 / C_rr=36.0 / wrist=48.6 (n=156); tuned det-box (WiLoR detector, recall 0.921) C_abs=68.7 / C_rr=39.9; no-tune GT-box control 168.3/36.0 = bit-identical to the Jul-10 hamer_gtbox157 run (harness validated; C_rr untouched by tuning as designed — pure metric calibration). Train L1 28.4->22.2->20.7mm over 3 epochs. READ-OUT: metric tuning closes 168->39.8 but ours 23.6 still leads the fair matched-regime bar by 16mm (40%); the E2E fair bar for detbox v3 is now 68.7 (tighter than native 83.4/88.0). OPEN before table ships: why n=156 on the mt_gtbox pass (1 seq dropped); JSONs Euler results/hamer_mt_{gtbox,detbox}_*.json + hamer_notune_gtbox_*.json. Watcher agent died at session limit AFTER pass 3 started — numbers pulled directly by main agent.** |
| D2-9 | PROVENANCE for the full-fine-tune control rows (table rows "HaMeR full fine-tune 632M" and "+ jitter aug") — reviewer pass 6 CRITICAL C1: these load-bearing rows were absent from this tracker | log job IDs + JSONs | **CLOSED 2026-07-19 (doc-only; experiments were done 07-16/17).** (1) **hunfz, Euler job 7402476** (DONE 2026-07-17 00:31, full-157, per-seq): same kp3d_abs L1 harness as D2-8 but joints NOT detached; two capacity arms — hd (MANO decoder unfrozen, lr 3e-5, 4ep): GT-box 26.5/17.0/wrist 26.0, det-box 54.1/21.2/50.2; **fu (FULL 632M network unfrozen, lr 1e-5, 4ep, train L1 15.1→9.3): GT-box 21.4 / C_rr 11.6 / wrist 24.1; det-box 44.0 / 16.1 / 43.7.** JSONs Euler results/hamer_{hd,fu}_{gtbox,detbox}_eval.json; sbatch auto/euler_hunfz.sbatch (md5 866c00b5). (2) **hujit, Euler job 7481943** (DONE 2026-07-17 14:13, full-157, per-seq, 2-pass JSON-verified): **fj = fu + matched box-jitter aug** (amplitude 0.2, 4-draw, per-sample deterministic, on tight box pre-×2-expand): GT-box 20.83/11.61/wrist 21.7; det-box 41.64/14.87/40.32; **fus1 = fu seed 1**: GT 21.39/11.7/24.39, det 45.07/15.85/45.15 (fu abs seed-stable). JSONs results/hamer_{fj,fus1}_{gtbox,detbox}_eval.json. These are the rows behind table footnote d (gate-1 robustness control: fj degrades +20.8 GT→det vs ours+jitter +8.3). |
| D2-10 | **Box-source parity for the headline E2E crossover — reviewer pass 6 CRITICAL C2 (open).** Ours det-box rows (49.8 / 35.6) use detbox v3 (our YOLO cache, mean_recall 0.829, meanIoU 0.346, med box ratio ~1.7x GT); the HaMeR fu/fj/mt det-box rows (44.0 / 41.6 / 68.7) use WiLoR-detector boxes (recall 0.921). Different detectors ⇒ the crossover "ours 35.6 < fj 41.6" is not yet input-matched, and box scale is the dominant depth cue. | one eval pass, no retrain: score fj (and fu) on the detbox v3 boxes via a convention bridge (v3 store [cx,cy,w,h] → HaMeR tight-box + rescale 2.0), OR ours on the WiLoR det boxes; report detector, recall, and med box-ratio in footnote d | **LAUNCHED 2026-07-19: Euler job 7775057 (euler_d210.sbatch).** fj ckpts were never banked (NODE_SCRATCH, wiped) and its pred dumps carry no boxes, so the run RETRAINS fj (tag fj2, identical harness/seed/jitter; ckpt banked to $S/ckpts/fj2_tuned.pt — also a prerequisite for the H2O zero-shot arm), re-runs GT-box + det-box as retrain-equivalence sanity anchors (expect ~20.8 / ~41.6), then evals on the EXACT detbox v3 boxes via a new `--box file` driver mode (patch_d210.py, clamp-to-frame like the gt/det paths). Decision: hamer_fj2_v3box_eval.json vs ours 35.6 on identical boxes. **RESULT (D210_ALL_DONE 2026-07-19 13:12, full-157, per-seq): CROSSOVER REVERSED.** fj2 on the exact v3 boxes = **C_abs 23.4 / wrist 22.3 / C_rr 13.2** vs ours(jitterrob) 35.6/34.8/26.5 on identical boxes — the full fine-tune wins E2E by 12.2mm (34%) once inputs are matched, and its GT→det degradation is SMALLER (+4.4 vs our +8.3). Sanity anchors reproduced (fj2 GT 19.0 vs orig fj 20.8; own-det 38.6 vs 41.6; retrain ~2mm stronger = run-to-run variance on the FT, worth a table note). READ-OUT: the pass-5/6 "detector-robustness of the frozen head" claim is REFUTED — the old +20.8-vs-+8.3 slope compared different box sources (HaMeR's own live detector produces harder boxes than v3). fu/fj own-detector rows stay in the table but relabeled; footnote d rewritten; ours+jitter det-box bolding removed (fj2-v3 row now best). Remaining affirmative story = parameter-efficiency (within ~3mm GT at ~1/100 trainable params) + analysis; gate 1 of the pass-5 survival path is DEAD. fj2 ckpt banked ($S/ckpts/fj2_tuned.pt) — reusable for the H2O zero-shot arm. |

Confirmed SOUND by the audit (stop re-litigating): metric definitions + self-tests, units, smplx-16
joint order incl. OP2SMPLX16, frame/clip alignment (offset=start, stride match), valid-mask asymmetry
actually favors baselines. Rebuttal rule: no new experiments in rebuttal — everything above must be
in the submission.

## E. Known infra facts (operational)

- SLURM: `-w` node pins NOT honored; node /tmp wiped between jobs; sbatch spools scripts at submit
  (patching a queued job's file does nothing — resubmit); compute nodes have internet; QOS = 1 running /
  3 submitted per user.
- Quotas: scratch ~65G+inodes (FULL — never touch venv_gb10), /home 20G. Checkpoints → node /tmp during
  runs; bank via `srun --jobid=<id> --overlap` watchers; durable storage = HF (`~/.hf_token`, repo in
  `~/.hf_repo`).
- Feature cache: stride-16 keyed `<seq>_<offset>.pt`; cached training must set data.clip_stride=16.
- Eval cost ~2.3 min/seq (stride 8), ~1.2 (stride 16); full-157 ≈ 3h at stride 16.
- Reliable big-file transfer Mac↔cluster: SSH ControlMaster + `scp -o ControlPath=/tmp/clctl`
  (pty base64 corrupts MB-scale payloads).
