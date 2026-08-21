# Confirmed defects

Defects that have been **measured**, not suspected. Read this before debugging anything that looks
like a slow job, a bad metric, or an implausible number: it has probably happened before.

## The bar for an entry

An entry is admissible only when all four are present. Anything short of that belongs in the issue
tracker or a scratch note, not here.

1. **Evidence.** A `file:line`, a measured number, or a job id. Not "I recall" and not "probably".
2. **Reproduction.** The exact command, config, or condition that triggers it.
3. **Signature.** What the failure looks like from outside, especially when it does not raise. This
   is the part that saves the next person, because a defect you can recognise costs minutes and one
   you cannot costs a day.
4. **Status.** Fixed with the commit or file:line, or open with what is still needed.

Delete an entry when the defect becomes unreachable, for example when the code path is removed. Do
not delete one because it was fixed: the signature stays useful.

---

## 1. `while d.dim() > 3: d = d.squeeze(1)` hangs forever on channel-last depth

**Evidence.** `preds["gs_depth"]` is `[B, S, Hd, Wd, 1]`, measured `(1, 8, 224, 224, 1)`.
`gs_depth[0]` is then 4-D with axis 1 of size 224, and `squeeze(1)` is a no-op on a non-unit axis,
so `dim()` never decreases.

**Reproduction.** Any `eval_world_space` run with `--dump_traj`, which is the only caller that sets
`depth_out` (`scripts/eval_world_space.py:544`, `:1172`).

**Signature.** No exception, no log line, one core at 100%. SLURM kills the job at the walltime and
the log ends with `CANCELLED DUE TO TIME LIMIT`. It reads as "the eval is slow", so the natural
reaction is to ask for more walltime, which never helps. It cost two one-hour jobs on 2026-08-19.

**Status.** Fixed at `scripts/eval_world_space.py:261-272`, handling both layouts and raising on
anything else. The same pattern was already documented at `:421` as "the clip-1 hang" but had only
been fixed at the second site. Verified against `[S,1,H,W]`, `[S,H,W,1]` and `[S,H,W]`.

---

## 2. Rewriting `preds["gs_depth"]` after the forward does not change the render

**Evidence.** `scripts/gs_metrics.py:render_views_from_predictions` reads `preds["splats"]`, which
the forward already built from the depth (`models/models/rasterization.py:721-738`). A probe that
scaled `preds["gs_depth"]` by 1e-8 through 1e4 got `L1=0.02426` and `render_mean=0.5840` at every
factor, identical to five digits across twenty decades.

**Reproduction.** Mutate `preds["gs_depth"]` between the forward and the rasterizer call.

**Signature.** A result that is *too* clean. A quantity that does not move at all across many
decades is almost never a real invariance, it is an input that never arrived. Treat a perfectly
flat sweep as a broken instrument until proven otherwise.

**Status.** Fixed in `scripts/probe_depth_vs_photometric.py` by scaling `splat.means` instead, plus
a guard that aborts when `render_mean` is identical at every factor rather than printing a verdict.

---

## 3. The Gaussian depth saturates at `exp(20)` and cannot recover

**Evidence.** `diffsynth/.../heads/dense_head.py:18` sets `_EXP_ACT_MAX = 20.0` and `:354` applies
`torch.exp(x.clamp(max=_EXP_ACT_MAX))`. Measured on `checkpoints/gsinj2_on`: every pixel of every
frame equals `4.85165e8`, and the fraction at the clamp is `1.0000`. `clamp` has zero gradient
above its maximum, so the unit is dead and no downstream loss can move it.

**Reproduction.** Train with `enable_gs: true`, `gs_l1`/`gs_lpips` above zero and no loss on the
depth, e.g. `configs/exp_gsinj2_on.yaml`. Saturation is already complete in the earliest surviving
checkpoint, roughly nine minutes into the run.

**Signature.** RGB renders look normal and PSNR *improves* while SSIM drops and LPIPS worsens
(measured 24.60 to 29.35 dB with SSIM 0.869 to 0.815). The Gaussian depth map renders as one flat
colour, since a percentile colormap over a constant has nothing to spread. Nothing in the training
log reports depth magnitude, so the run looks healthy throughout.

**The degeneracy is double, and the cause is that the objective has no parallax.** Measured splat
scales are `1.01e-08` on the trained head against `0.0016` on both healthy ones. Footprint is
scale/depth, so its Gaussians are fourteen orders of magnitude smaller and survive only through the
rasterizer's minimum screen-space dilation, which makes every one of them exactly one pixel: the
render is a point-sampled copy of the input, not a reconstruction. Scaling positions and extents
together does not recover it, so this is not a wrong gauge, both quantities ran away in opposite
directions.

Why it drifts: `scripts/train_hand_head.py:1262-1263` hardcodes identity `camera_poses` and
`camera_intrs`, and the loss re-renders the **input** views from Gaussians unprojected from those
same views (`scripts/gs_metrics.py:64`). With no viewpoint change there is no parallax, depth is
unobservable, and scaling the whole cloud is a flat direction with the clamp as its absorbing end.
The loss itself is not blind: on a healthy model the same loss has a sharp optimum at the true depth
(35.47 dB, twenty down one decade either way, `scripts/probe_depth_vs_photometric.py`).

| model | splat scales | depth | best PSNR |
|---|---|---|---|
| pretrained NeoVerse | 0.00161 | 0.926 m | 35.47 |
| v1, frozen head + anchor 1.0 | 0.00158 | 0.902 m | 33.46 |
| v2, trained head, anchor 0.0 | 1.01e-08 | 4.85e8 m | 29.59 |

Three settings changed together between v1 and v2, all pushing the same way: `hand_depth_anchor`
1.0 to **0.0**, which was the only term referencing a real length and therefore the only thing
pinning the gauge, `gs_l1`/`gs_lpips` 0.1 to **1.0**, and `freeze_gs_head` true to **false**.

**Status.** Open. `scripts/train_hand_head.py` now logs `gs_depth median` and `at_clamp` every
`log_every`, so a collapse is visible while it happens. The clamp still turns a divergence into a
silent dead unit; it was added to stop `ExpBackward0` NaNs (`dense_head.py:11-17`), so removing it
needs a replacement bound.

---

## 4. `hand_scene_registration_loss` sends no gradient to the depth by default

**Evidence.** The default direction is `bidirectional`
(`scripts/train_hand_head.py:2243`), and in that branch
`scene = s * sampled` where `sampled = sampled_g.detach()`
(`scripts/hand_scene_registration_loss.py:104-108`). Only `scene_follows_hand` uses the
gradient-carrying `sampled_g`.

**Reproduction.** Set `loss_weights.hand_scene_registration` above zero without setting
`hand_scene_registration.direction`.

**Signature.** The loss term appears in the log and decreases, while the Gaussian depth is
untouched. It looks like the coupling is being trained when only the scale and the hand are moving.

**Status.** Open, and correct by design for the scale-head route it was written for. Any run whose
purpose is to constrain the depth must set `direction: scene_follows_hand` explicitly.

---

## 5. The registration loss is computed with different settings in training and validation

**Evidence.** The training call passes `margin`, `depth_min`, `conf_thresh` and `direction`
(`scripts/train_hand_head.py:2959-2965`). The validation call passes none of them
(`:1581-1583`), so validation always uses the function defaults.

**Signature.** The reported validation term does not track the objective being optimised, and the
gap grows with any non-default setting.

**Status.** Open. Harmless while the weight is zero.

---

## 6. Training builds the Gaussian views with identity intrinsics

**Evidence.** `scripts/train_hand_head.py:1262-1263` hardcodes `camera_poses = eye(4)` and
`camera_intrs = eye(3)`, while the real `cam_intrinsics` are already in the batch and are used by
the 2D and anchor losses.

**Signature.** Training-time splat geometry is in no metric frame, and under identity intrinsics
unprojecting and reprojecting through the same camera returns the original pixel for any depth.

**Status.** Open, and a candidate root cause for defect 3.

---

## 7. `checkpoints/mix5` contains no Gaussian-head weights at all

**Evidence.** Zero `gs_head`, `hand_to_gs` and `gs_renderer` keys in the state dict, because
`configs/train_mix5_all.yaml:68` sets `enable_gs: false` and the module is never built. Evals load
the base with `strict=False`, so the Gaussian branch stays pretrained NeoVerse.

**Signature.** An A/B that reads as "frozen head against trained head" is really "pretrained head
against trained head". Anything attributed to freezing is attributed wrongly.

**Status.** Open as a documentation issue: the comparison is valid, its label is not.

---

## 8. 4DGT renders are black, and a shell bug hid it

**Evidence.** `gs_4dgt_native1.sbatch` had a whitespace-only line between a `\` continuation and
its `+render_out=` argument, so the shell ended the command early and tried to execute
`+render_out=...` as a program: `line 37: +render_out=...: No such file or directory`, and all ten
sequences reported `FOURDGTFAIL`. Removing that blank line makes the job report
`GS4DGTIV_DONE ok=10 fail=0 seqs=13 frames=640`. The 640 frames are 504x504 PNGs of about 1.6 kB
each, uniformly black, and the shared scorer gives whole-frame **PSNR 4.78, SSIM 0.0010,
LPIPS 1.0144** against ground truth.

**Reproduction.** `sbatch gs_4dgt_native1.sbatch`, then look at any file under
`gs_out/fourdgt_native1/<seq>/renders/`.

**Signature.** Two independent failures stacked, and the outer one hid the inner one. Fixing the
shell quoting turned `ok=0 fail=10` into `ok=10 fail=0` while changing nothing about whether the
output is usable. A 1.6 kB PNG at 504x504 is a constant image; file size alone is enough to tell.
Note also the resolution mismatch: the ground-truth export is 224x224.

**Status.** Open. The invocation is fixed, the renderer is not. The earlier `fourdgt_parity` attempt
produced the same black output, so this is not caused by the view count.

---

## 10. Cached intrinsics are at source resolution, so every non-HOI4D store projects out of frame

**Evidence.** The preprocessors write `hand_data/cam_intrinsics.pt` at the resolution of the source
video, while every frame reaching the model has been scaled to cover and centre-cropped to 224
(`diffsynth/utils/auxiliary.py:126-142`, reached because `load_video` defaults to
`resize_mode="center_crop"` and the dataset does not override it). Measured, one sequence per store:

| store | source video | `2*cx` x `2*cy` | joint on the optical axis lands at |
|---|---|---|---|
| arctic | 2800x2000 | 2657 x 1966 | 3.0 |
| taco | 1920x1080 | 1935 x 1054 | 2.2 |
| oakink2 | 848x480 | 864 x 504 | 1.9 |
| hot3d | 1408x1408 | 1407 x 1407 | 3.1 |
| dexycb | 480x480 | 467 x 477 | 1.0 |
| **hoi4d** | **224x224** | 229 x 217 | **0.51** |

Normalised coordinates outside `[0, 1]` are discarded as out of frame, so on five of the six stores
`in_frame` is uniformly false. HOI4D escapes only because its video is already 224x224.

**Reproduction.** Any scene-scale solve outside HOI4D. `sgt_hot3d` job 11212664 reported
`12/12 clips FAILED the solve (100.0%)` with `EMPTY population (no correspondence passed the
geometric mask; ungated=0)` on every clip. `ungated=0` is the diagnostic that separates this from
the hand-validity gate: it means the population was empty **before** any gate ran.

**Signature.** Three ways it shows up, none of which raise:

- every clip's solved scale sits exactly on a clamp bound, with zero variance across a segment;
- `hand_depth_anchor` evaluates to `0.0` while its configured weight is non-zero, so a run reports a
  loss recipe it is not applying;
- world-space metrics are computed from `s = 1.0` and are therefore non-metric, while camera-frame
  metrics are untouched and look healthy.

**Status.** Fixed 2026-08-20. `HOT3DHandDataset` now carries the cached intrinsics across the same
cover-and-centre-crop once per sequence (`_intr_to_render_frame`, `scripts/train_hand_head.py`), and
`eval_world_space.predict_clip` applies the same mapping to the copy it loads straight off disk.
Two compensations that then corrected a second time were removed: the rescale inside
`project_joints_to_norm_pixels` and the one in `metric_views.intr_3x3`, which forced the principal
point to the frame centre and moved arctic's focal from 270 to 291.

The mapping is the identity on HOI4D, so every published number on that store is unchanged. It also
makes the `IMAGE_WIDTH = 2.0 * cx` heuristic used elsewhere in the trainer true for the first time
on the other five stores.

After the fix, `sgt_hot3d` job 11228854 solves 37 of 56 clips with zero failures and yields a HOT3D
scene unit of 1.140 median over 75 segments.

**Still owed.** The three converters that write non-square sources, arctic, oakink2 and taco, have
never had a scale or anchor number produced on them, so there is nothing to re-check but also
nothing verified end to end. `b2_render_object_depth.py` reads the raw file and infers
`W = H = 2*cx`, which is self-consistent because it renders in the native frame, and correct for
HOT3D because that store is square.

---

## 11. A held-out view renders black, because a Gaussian is drawn only into its own frame

**Evidence.** `GaussianSplatRenderer.forward` gates every Gaussian per view
(`rasterization.py:321`): it draws only when `splats.timestamp == -1`, when the timestamp equals
the view's, or when the view falls inside a forward/backward motion window. Those windows need
`forward_timestamp` / `backward_timestamp`, which the motion module supplies; `enable_motion` is
false in every config and every call site passes `use_motion=False`, so only exact match can fire.
Splats carry their own frame's timestamp, so a frame is rendered from the Gaussians unprojected
from that same frame.

Measured on the pretrained checkpoint, 12 HOT3D clips, 2 of 16 frames held out:

| | per-frame | fused static |
|---|---|---|
| held-out PSNR | 9.65 | 22.06 |
| held-out SSIM | 0.019 | 0.679 |
| target dark-pixel fraction | **1.0000** | 0.0029 |
| context PSNR | 36.18 | 23.79 |

**Reproduction.** Any forward with `build_views_metric` and `n_targets > 0` before 2026-08-20, or
any render of a frame the clip did not unproject.

**Signature.** The held-out PSNR is not merely low, it is the PSNR of a black image, and the
context number is *high* at the same time. A single averaged score hides both: with 14 context
frames against 2 targets the mean still reads well. The tell is the gap, and the coverage check
that separates "wrong" from "nothing drawn" is the mean and dark-pixel fraction of the render, not
its PSNR.

**Status.** Fixed 2026-08-20. `build_views_metric` sets `is_static=True`, which routes every
Gaussian through `_create_constant_gaussians` at `timestamp=-1` (`rasterization.py:833`) so the
clip fuses into one cloud visible from every view. Two consequences had to be handled with it:

- that path calls `prune_gs`, which needs `torch_scatter`; the package is absent from `train_env`,
  so `scatter_sum` was `None` and the static path raised `TypeError: 'NoneType' object is not
  callable`. It is now implemented in plain torch via `index_add_`, verified against the reference
  semantics on 1-D and 2-D inputs. Skipping the merge instead is not viable: an unpruned clip left
  training at iteration 0 after 13 minutes.
- rendering one fused cloud into every view is **16x the work** of drawing each frame's own set
  into one view, measured as 24.7 s/it against 3.1. That is the price of scoring a reconstruction
  rather than a copy, and the arms were resized to an effective batch of 8 to pay it.

**What it means for numbers already reported.** Every PSNR this project has published sits in the
per-frame regime: `tab:gs` rows at 29.47 / 27.66, the frozen row at 37.93, the 36.96 -> 37.92
quoted in the same section, and the 37.79 / 35.27 measured on 2026-08-19. None of them is a
reconstruction score. `4exp.tex:105` also calls the table "novel-view synthesis", which it is not:
the rows score the input views. The comparison against AnySplat is the visible symptom, a frozen
model appearing to beat a dedicated splatting method by 15 dB, where the honest fused measurement
puts the same backbone at 23.79 against its 22.43.

---

## 12. The bbox cache's real MANO GT loads with `has_mano=False`, masking every param loss

**Evidence.** `HOT3DHandDataset` takes the no-jsonl branch's placeholder flag (`seq_has_mano =
False`, `train_hand_head.py` cache path), then overwrites `gt_per_frame` with the cache's real
MANO (`gt_per_frame[:] = list(cached["gt"])`) without restoring the flag. Measured on
`hot3d_pinhole_f609/P0001_10a27bf7`: `has_mano=False` while `gt.abs().sum()=796.5`.

**Reproduction.** Any cached-crop run on a store whose sequences resolve GT through the bbox
cache. The metric arm run 11274159 aborted at step 50 with `transl / global_orient / hand_pose /
betas contributed EXACTLY 0.0`.

**Signature.** `kp3d` and `kp3d_abs` fire normally, the four MANO param terms read exactly 0.0,
and nothing raises unless the loss-effect guard is reached. In a MIXED pool the guard never fires,
because the other stores keep the running averages nonzero, which is how mix5 and both fusion arms
trained with HOT3D contributing no MANO parameter supervision at all, silently.

**Status.** Fixed 2026-08-20, gated. `data.cache_gt_is_mano: true` restores the flag when the
cached GT is non-trivially nonzero; the DEFAULT is false, deliberately, because flipping it changes
what trains: `abl_fusion_on` is already trained under the masked behaviour, so its OFF arm must
run masked too or the pair measures two things. The metric injection arms enable it symmetrically.
Re-audit whether HOT3D's missing param supervision matters for mix5 before the final training.

---

## 13. `gs_depth_logit` held only the LAST DPT chunk: the anti-collapse teacher saw 6 of 14 frames

**Evidence.** `DPTHead.forward` defaults `frames_chunk_size=8` and `worldmirror.py:479` never
overrides it, so 14 context frames run as chunks `[0:8]` and `[8:14]`. `activate_head` stashed
`self._last_raw_attr = attr` per chunk (`dense_head.py:355`), each call overwriting the last, and
in pre-reshape layout `[B*S_chunk, H, W, C]`. `preds["gs_depth_logit"]` therefore carried frames
8-13 only, shaped `[B*6, H, W]`. The teacher head stashes the same way, so the shapes matched and
`gs_depth_logit_loss` ran without any error.

**Reproduction.** CPU: a toy `DPTHead(is_gsdpt=True)` forward with `frames_chunk_size=2` on 5
frames left `_last_raw_attr` at `[B*2(last chunk), H, W, C]` before the fix.

**Signature.** Nothing raises. The logit teacher term and its `frac_at_clamp` / `logit_median`
diagnostics cover only the trailing chunk; the `gs_depth median` console line reads the correctly
concatenated activated depth, so the two diagnostics can disagree. The completed metric ON run
11295502 trained under this: its teacher never touched context frames 0-7.

**Same root cause, worse symptom.** `gs_depth_target: "depth_head"` sliced the target with
`[:, :gs_depth_logit.shape[1]]` (a no-op on the real layout) and then the alignment loop in
`gs_depth_logit_loss` spun forever on the 4-dim-vs-3-dim pair: the branch could never have worked,
it would have hung the job at its first step, silently, until the walltime.

**Status.** Fixed 2026-08-20: `DPTHead.forward` reassembles the stash across chunks to
`[B, S, H, W, C]` (verified chunked == unchunked on CPU), and the loss's alignment loop raises on
an unalignable pair instead of looping. The OFF control and the depth-head arm pick the fix up at
start; ON is re-queued as `mgson2` (11349071) so the A/B pair trains on identical code. The
completed ON (18.37 dB, +0.35 over init) remains valid as a result but its teacher coverage was
partial.

---

## 14. "Training the Gaussian head" never optimizes the attribute decoder

**Evidence.** The optimizer's param groups collect `model.gs_head.parameters()` only
(`train_hand_head.py:2078-2080`, consumed at `:2148-2151` and `:2579`). The convs that turn
`gs_feat` into quats/scales/opacities/SH/weights live in `model.gs_renderer.gs_head` and
`gs_renderer.gs_head_dynamic` (`rasterization.py:487-491`) and appear in no group, under either
value of `freeze_gs_head`. Their `requires_grad` stays true, so grads accumulate and are never
stepped or zeroed.

**Signature.** Every "trained Gaussian head" claim in this repo is about the DPT depth/feature
head with a FIXED attribute decoder. Any statement about "the scales the head learned" (e.g. the
1e-8 splat scales of defect 3) describes a fixed decoder driven by drifted features.

**Status.** Open, deliberate for now: adding the decoder to the optimizer changes the recipe, so
it must not happen mid-A/B. Decide after the ON2/OFF pair lands.

---

## 15. `is_static` selects WHICH attribute decoder runs: identity-era training used the dynamic one, fused evals use the static one

**Evidence.** `enable_dynamic_gs_attr` defaults true (`worldmirror.py:39`) and per-frame
`torch.where(is_static, gs_params_static, gs_params_dynamic)` picks the conv
(`rasterization.py:538-548`). `build_views` sets `is_static` zeros (`train_hand_head.py:1301`);
`build_views_metric` sets ones (`metric_views.py:99`), as do `eval_heldout_gs` and
`run_ours_gs --static`. Both decoders exist pretrained in the NeoVerse ckpt (3 keys each,
verified 2026-08-20).

**Signature.** A gsinj/gsinj2-era checkpoint scored in the fused-static regime decodes its
trained `gs_feat` through a conv that training never exercised. The metric pipeline is internally
consistent (trains and evals static). Cross-era comparisons, including the frozen-vs-trained
figure if rendered fused, straddle the mismatch.

**Status.** Open. Affects interpretation of old runs, not the current arms.

---

## 16. `run_validation` renders through the eval branch while training renders through GT cameras

**Evidence.** `run_validation` calls `model.eval()` (`train_hand_head.py:1422`) and the renderer
branches on `self.training` (`rasterization.py:551-583`): eval re-predicts cameras, scale-aligns
their translations, and unprojects with `gsdepth+predcamera`; training uses GT cameras. So the
validation `gs_l1`/`gs_lpips` inside `val_loss`, and the best-checkpoint criterion, score a
different render than the training objective. `eval_heldout_gs.py:129-133` sets `model.train()`
for exactly this reason. Side effect: the eval branch writes scale-aligned translations in place
into a view of `predictions["camera_poses"]` (`rasterization.py:571` via `prepare_cameras`), so
poses read after a non-training render are mutated.

**Status.** Open. The metric arms ship `hand_head_final.pt`, not the val-best checkpoint, so the
completed results stand; fix before any run whose selection depends on val_loss.

---

## 17. The Gaussian ATTRIBUTE grid is rotated 90° cw against the image; the feature maps and `gs_depth` are not

**Evidence.** Two instruments, one clip set, 2026-08-20. (a) The ownership probe's layout sweep
renders the suppressed subset alone under five candidate mask layouts and scores overlap with the
GT hand boxes: `rot90cw` 1.000, `transpose` 0.524, identity 0.085, `rot90ccw` 0.000
(`$S/results/ownership_ab_mix5hand_v2.json`, `grid_layout_sweep`). (b) The impulse test adds a
constant to the top-left quadrant of every per-layer projected feature, the tensors the
hand-to-GS injection writes into, and the render diff peaks in image TL (0.445), with the
`gs_depth` positive control also TL (0.666) (`$S/results/grid_impulse_mix5hand.json`).

**Reading.** The DPT-internal feature maps and `gs_depth` are image-aligned, so the injection's
box-guided writes, `hand_depth_anchor`'s pixel sampling and the depth-logit masks address the
right pixels. The rotation sits only between an image-space [H, W] mask and the per-Gaussian
attribute layout that `scene_opacity_mask` multiplies. Any consumer that indexes Gaussian
attributes with image coordinates must map through `rot90cw` first.

**Consequence already measured.** The first ownership A/B (revealed MAE 0.0943 vs 0.0795, read as
decontamination evidence) suppressed a 90°-wrong subset. Under the correct mask the verdict
reverses: suppression is worse in revealed regions (paired B-A +0.058, CI [0.046, 0.079], 3/12
clip wins) and indistinguishable on co-covered pixels (-0.002, ns), with the revealed gap
explained by coverage loss (alpha 0.999 -> 0.785; white-background MAE 0.100 -> 0.209). The same
fix un-broke arm C: inserted hand Gaussians now render (sentinel 22.9k target px changed) and
halve the hand-region MAE, 0.066 -> 0.034.

**Status.** Instrument fixed in `scripts/probe_ownership_ab.py` (sweep + `to_grid`). Root cause
in the attribute reshape not yet localized.
