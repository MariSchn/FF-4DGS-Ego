# Submission plan, ICLR 2027

Written 2026-08-05 after three review passes: an adversarial CVPR reviewer, a supervisor-style prose
reviewer, and a structural gap analysis against HaWoR / Hand3R / EgoForce / WildHands. Both reviewers
returned **Reject** independently, for different and compounding reasons.

Decisions already taken and NOT open for re-litigation:

- **Sec. 3.2 (hand to Gaussian injection) is NOT demoted.** It stays a numbered contribution. That
  means option (b) below is not available: the run has to happen.
- **Depth-diverse mixing is promoted** from an Experiments observation to a stated contribution.

---

## P0 - blocks submission, and no rebuttal can repair it

### P0.1 Decompose the world-space oracle (the scale confound)

`eval_world_space.py:105-116` multiplies the camera translation by a **hand-derived** scene scale
`s = median(z_hand / gs_depth)`, clamped to `[0.1, 10]`. The headline run records `s_med 0.707`
against a true scale near `1.0`, and the code's own comment at `:500-505` calls the pooled variant
"biased (+14-27% vs the true cam-center scale) AND heavy-tailed".

Consequences: (i) this is the hand-supplies-scene-scale mechanism our own ablation refuted (0.728 vs
oracle 1.022); (ii) the GT-camera oracle that moves W from `200.9` to `61.5` removes rotation,
translation **and** this ~29% scale error at once, so "long-window error is camera-dominated" is not
earned.

- [ ] Run the three-row decomposition: **GT scale only / GT rotation only / GT full**. One eval sweep.
- [ ] State which scale variant produced every tabulated W. `eval_world_space.py:493-495` offers
      per-clip, per-sequence-median and per-sequence-pooled, and calls the pooled one "the principled
      sequence-level solve". **If the tables used the pooled variant, the "online" claim fails
      outright**, because it is a whole-sequence statistic.
- [x] Method text corrected: it no longer claims "no post-hoc rescaling stage". `Sections/3method.tex`.

### P0.2 Two baseline cells are artifacts of our own harness

Our internal `report/table_world_comparison.tex` already marks these `\na`; the paper prints them.

- [ ] HaPTIC C-abs `153.6` to `\na`. Root cause is our adapter: weak-perspective lift
      `tz ~ 2*focal/(256*s)` with `hoi4d_to_haptic.py` rescaling the 224-store focal, so `tz` tracks
      the injected focal. Internal note: "C_abs BROKEN both - never report as metric."
- [ ] Dyn-HaMR C-abs `1327.4` / W `276.6` to `\na`. Input matching bypasses its metric-depth stage;
      only its scale-invariant WA (`49.3`) is meaningful.

### P0.3 Frozen-vs-unfrozen ablation (task #56)

Added 2026-08-05 from the OpenReview corpus. **Every frozen-backbone paper in our family gets asked
"why frozen?", and the answer format decided the outcome in all three cases we have full threads for.**

| Paper | Answered with | Outcome |
|---|---|---|
| Human3R (ICLR 2026) | a table: WA-MPJPE `267.9` frozen vs `1359.3` with the decoder unfrozen | **Accept** |
| Fin3R (NeurIPS 2025) | a table: full fine-tune `26.35` vs frozen baseline `28.40` AUC@5 | **Accept** |
| G-CUT3R (ICLR 2026) | prose only: *"we did not see any improvement from training the encoder"* | **Reject** |

Human3R's AC named it as one of three things that resolved the concerns: *"the ablation study on the
joint-train decoder justified their design choice, empirically proving that full fine-tuning causes
catastrophic forgetting of scene priors."* Fin3R's authors had run it, cut it "in the interest of
conciseness", and had it demanded back by two reviewers independently.

We do not have this ablation. `backbone-swap` varies *which frozen backbone* (recon `23.6` / DINOv2
`21.9` / random `27.7`), which answers a different question. Without a frozen-vs-unfrozen arm,
"frozen" reads as an unjustified inheritance, and that feeds straight into the efficiency attack in
P1 as well.

- [ ] Arm 1: frozen backbone + trained hand branch (our recipe)
- [ ] Arm 2: + unfrozen last N encoder blocks (`unfreeze_last_n_blocks` already exists)
- [ ] Arm 3: + fully unfrozen backbone
- [ ] Report C-abs, C-rr and one world metric per arm, matched data and schedule
- Scaffolding exists: `configs/_p3a_unfreeze.sbatch`. GPU required.

**Other OpenReview-derived items now tracked as tasks:** #49 title-claim audit (P0), #50 FLOPs and
inherited-vs-added cost (P0), #53 appendix + supplementary video (P0), #51 input provenance table,
#52 degradation curve, #54 lock the metric set, #57 accepted efficiency-table format. Full evidence
with verbatim reviewer quotes: `report/openreview-evidence-log.md`.

### P0.3 Tab. 1 and Tab. 2 contradict each other

Tab. 1 gives WiLoR+SLAM and HaPTIC+SLAM `\na` under "no metric depth"; Tab. 2, same 157 sequences,
prints WiLoR+SLAM at C-abs `84.0` / W `219.0`. Also, `84.0` appears to be WiLoR's *native full-res*
camera-frame number fused with a SLAM-composed world number under one row label.

- [ ] One policy across both tables; per-row statement of which pipeline produced which column.

### P0.4 Baseline regime asymmetry

`report/table_world_comparison.tex` footnote *b*: HaWoR scores **49.7 / 34.7 / 42.3** on its own
detector and **87.7 / 40.1 / 53.4** on ours. The paper reports only the second. Hand3R's published
HOI4D table independently puts HaWoR at `51.77 / 41.28`, consistent with its native numbers.

Worse: our headline model is trained **with box-jitter augmentation**, i.e. train-time adaptation to
the exact box distribution every baseline is force-fed without adaptation (`ours 49.8` without it,
`35.6` with). Neither the augmentation nor the `49.8` appears anywhere in the paper.

- [ ] Add a native-regime column for every baseline alongside the shared-box column.
- [ ] Disclose box-jitter augmentation in implementation details; report `49.8`.
- [ ] Either give baselines the same jitter-augmented adaptation, or state plainly that the
      shared-box column advantages us by train-time adaptation.

---

## P1 - the five items, in the requested order

### 1. "Trained on target dataset?" column in every HOI4D table

Self-inflicted: `eq:transfer` is **our own** claim that train-test depth distance governs absolute
error, so a table comparing an in-domain-trained model against cross-dataset baselines without that
column undermines itself. Yes in our row, No in every other.

- [ ] Add the column. Baseline training sets are already known: HaWoR on HOT3D/ARCTIC/DexYCB/HO3D,
      WiLoR on HaMeR's ten plus BEDLAM/ARCTIC/Re:InterHand/HOT3D, HaMeR on its 2.72M pool, HaPTIC on
      ARCTIC/InterHand/DexYCB/H2O, Dyn-HaMR trains nothing. Note HaPTIC trains on H2O, so H2O is not
      clean zero-shot for the whole table either.

**Doable now, no GPU.**

### 2. Cross-attention ablation arm

The only untested mechanism left. Sec. 3.1 asserts "the attention is what carries scene evidence into
the hand decoder" and nothing measures it. The switch already exists:
`diffsynth/auxiliary_models/worldmirror/models/heads/hamer_head.py:124-126`, `use_global_context`,
with a comment that reads like our own hypothesis. It is in no config and no run.

- [ ] One training run with `use_global_context=False`, everything else matched.
- [ ] Companion floor: **constant-depth predictor** - place the wrist at the training-set median
      depth along the ray through the box centre, score C-abs. Every diagnostic we have says absolute
      placement is prior-driven (no-abs-supervision gives 682.8mm, "the signature of a network
      predicting a near-constant depth"); this measures the floor the method must clear. Cheap, no
      training.

### 3. GS injection: run it under a corrected `freeze_gs_head`

Sec. 3.2 stays, so it must be trained and evaluated.

- [ ] **Code fix first.** `train_hand_head.py` currently freezes the injection convolutions along
      with the head:
      ```python
      if freeze_gs_head:
          for p in gs_head_params + injection_params:   # <- injection must NOT be here
              p.requires_grad = False
      ```
      We need head frozen, injection trainable. No config combination gives that today.
- [ ] One run with `enable_gs: true` under the patched flag.
- [ ] Report **hand-region-masked** PSNR/LPIPS as the primary metric, not full-frame. The masked
      metric already exists in the trainer.
- [ ] Until it lands, the snowflake on the Gaussian head in Fig. 1 is an intention, not a fact.

### 4. Three figures

- [ ] **Page-1 teaser.** Hand3R Fig. 1 is the model: a three-panel taxonomy (root-relative isolated /
      global with disjoint pipelines / ours single-pass). Does in one image what our Related Work
      does in four paragraphs. Blocked on the Intro existing.
- [ ] **Metric-depth side-by-side vs HaWoR and WiLoR, same scene.** The comparison our central claim
      invites and that no figure makes. Needs baseline meshes dumped into a shared scene.
- [ ] **Depth-vs-time trajectory plot.** Cheapest of the three; predictions already dumped. EgoForce
      Fig. 16 is the reference. Plots exactly what C-abs measures and exactly where WiLoR/HaMeR fail
      (84.0/89.0 absolute vs 27.2/30.3 root-relative).

Current qualitative figure exists but its panels come from June checkpoints and three unrelated
sequences; regenerate from the final checkpoint, one sequence shown three ways.

### 5. Limitations section + appendix

We have **no appendix at all** (`\appendix` commented out, `main.tex:85`). HaWoR ships 4 supplementary
sections, EgoForce 5. The 25 open `\todo` blocks are exactly the class of material that belongs there.

- [ ] Uncomment `\appendix`; add: per-store box convention table, HOI4D split-leakage audit,
      undistortion derivation, Re:InterHand distortion model, formal metric definitions.
- [ ] Write Limitations. Owns: camera head untrained and dominating long-window W; Sec. 3.2 status;
      HaMeR_ft beating us on matched boxes; the scale solve.

---

## P2 - found by the gap analysis, cheap, high value

- [ ] **Re-run `eq:geom` in angular units.** `g_h` is in PIXELS. WildHands (ECCV'24) argues the
      transferable quantity is angular position `atan((x-px)/fx)`; EgoForce's Crop Intrinsics Token
      does exactly this and measures CS-MJE `123.4 -> 76.6`. We train across five stores and three
      camera models, which is precisely where a pixel encoding cannot work - and our box-geometry
      ablation duly returns null (0.7mm). **The null is likely a diagnosis, not a dead end.**
- [ ] **State hand set and joint set everywhere.** `eval_world_space.py:652` scores C-MPJPE on the
      RIGHT HAND ONLY over **16** joints (no fingertips); W/WA use both hands over 32. HaMeR / WiLoR /
      Hand3R report **21** joints including fingertips, the highest-error ones. Until this is stated,
      tabulating our 36.2 against Hand3R's 42.6 is not valid, nor is the claim that EgoForce's CS-MJE
      "is the same quantity as our absolute C-MPJPE".
- [ ] **Missed-detection policy.** The eval validity mask *is* the detector's success mask, so missed
      hands contribute no error: the "honest end-to-end" 35.6 is scored on **82.9%** of hands, the
      GT-box 23.6 on 100%. Report a recall-penalised metric or an intersection protocol.
- [ ] **FPS honesty.** `fps_probe.py` runs with **GS off** at the superseded T=16, so 6.59 FPS times a
      configuration that produces no scene. Report hands-only and hands+Gaussians at the final T.
- [ ] **One configuration table.** Eight different "our HOI4D number" circulate (22.8 / 23.6 / 23.9 /
      30.8 / 35.2 / 35.6 / 35.8 / 36.2) with varying sequence and segment counts. One table mapping
      checkpoint, training set, T, box source, jitter on/off, seq count, GS on/off to every number.
- [ ] **Temporal metric (Accel).** HaWoR and Dyn-HaMR both report it; we are the online row in a table
      of offline methods, so jitter is the standard objection and we answer it nowhere. Computable
      from existing dumps.
- [ ] **Camera trajectory on its own (ATE / ATE-S).** Turns "characterized limitation" into a
      measurement. GT extrinsics already in the loader.
- [ ] **Fix the Related Work differentiator.** `2rw.tex` says "no information flows from scene
      geometry back into the hand estimate" across a citation group that includes Hand3R and Human3R,
      which predict humans and pointmaps from a *shared* backbone. That sentence carves out our
      novelty and is not true of the closest prior work.

---

## Resolved today, no action needed

- **W / WA naming: we are correct.** HaWoR supp. 8.3: "after aligning the first frames and aligning
  the entire trajectories, respectively", and its Tab. 3 has W > WA throughout (33.20 vs 11.27).
  Ours matches (200.9 vs 35.0). EgoGrasp's *prose* also matches; it is EgoGrasp's *Table 2* that
  inverts (H2O W 5.75 vs WA 46.37). **Do not tabulate against EgoGrasp's numbers until explained.**
  Two real differences remain to state per table: HaWoR's WA aligns the entire trajectory, ours per
  window; our W is rigid with no rescaling, EgoGrasp's is Procrustes and absorbs scale.
- **MANO dimensions corrected.** `HAND_PARAM_DIM = 32 # pos(3) + rot(4) + pose(15) + betas(10)`.
  Quaternion rotation, 15-dim pose. An earlier draft copied EgoGrasp's axis-angle shape and summed
  to 61.
- Four prose contradictions fixed: HOI4D protocol, zero-shot in Tab. 1 caption, the forbidden scale
  claim, the `q` notation collision.

---

## Known gap with no owner yet

**We have no H2O world-space numbers at all.** `report/` holds only HOI4D world evals
(`hoi4d_world_eval_*.json`). We have H2O camera-frame results, but nothing that could occupy a row in
EgoGrasp's H2O W/WA/PA table. If we want to appear in that comparison, it is a new eval run - and
per the note above, it should not be tabulated against their numbers until their inversion is
explained.

---

## E: FIRST RESULT (pilot), 2026-08-05

Computed locally from `report/figures/hoi4d_test_error3d_joints.npz` (253 valid frames, 16 joints,
camera-frame). **Pilot only** - one diagnostic dump, not the 157-seq test set.

Construction: keep articulation fixed, slide the wrist along its own ray to a constant depth. The
camera sits at the origin in camera coordinates, so scaling the wrist vector by `z_c/z` is exactly
"along the ray through it". The constant used is the **median of the test sample itself**, i.e. an
ORACLE constant, which makes the floor artificially strong; a real predictor using the training
median would do worse.

| | C-abs | note |
|---|---|---|
| **Model** | **25.1 mm** | C-rr 19.2 |
| Floor, GT articulation + oracle constant depth | 38.7 mm | |
| Floor, our articulation + oracle constant depth | 45.3 mm | the like-for-like comparison |

Wrist depth alone: **model 17.7 mm vs constant 37.8 mm**.

**Read-out: the model beats the constant-depth floor by 20.2 mm, 45%, against an oracle constant.**
The floor is NOT close to the reported number, which is what a skeptical reviewer assumes by
default. This strengthens rather than weakens the learned-depth-prior framing: the prior does real
per-frame work instead of collapsing to one number. Owed: rerun on the full 157-seq set using the
TRAIN median rather than the test median.

## D: BLOCKED, needs a dump run

The local `report/hoi4d_world_eval_*.json` files carry only scalar per-segment errors
(`W_MPJPE`, `WA_MPJPE_short/long`), not per-frame joints, so the segment-length x alignment-window
sweep cannot be computed locally. `eval_world_space.py` does support `--dump_cam_dir` and
`dump_list`, so this needs one evaluator pass with dumps enabled; after that the sweep itself is
free and repeatable.

## Cluster state after the 2026-08-05 triage

- varlen (9732155) KILLED at ~6h50. Yielded ONE validation per arm, both at `global_step 1`
  (untrained, ~500 mm). No usable science. It did expose the `val_every` x `grad_accum` bug and the
  real epoch cost (~4.1 h, so 10 epochs could never fit in 24 h).
- `dxcache32` RESUBMITTED as **9795073**. Cancelling it earlier was a mistake: DexYCB (0.780 m) is
  the ONLY store deep enough to bracket H2O (0.503 m), so the bracketing arm of experiment A cannot
  be built without it. Surviving T=32 caches are arctic (0.474), hot3d (0.339), oakink2 (0.386) -
  all shallower than the target.
- `featcache/` (T=16) is EMPTY, so the T=16 mix3 result (66.2) cannot be reproduced or matched.
  **Decision: run both arms of A fresh at T=32** and restate the mixing headline at the locked
  protocol.
- `train_shallowmix_control.yaml` exists but its run (9594595, Aug 4) died `rc=1` after the step-1
  validation, and it used Ego-Exo4D, which was later removed for shipping no MANO. Rebuild it with
  hot3d in place of egoexo4d.

---

# SIX-WEEK SCHEDULE (2026-08-05 to ~2026-09-16)

One GPU, 24 h wall limit, jobs serialize. **The organising fact: most of the highest-value work
costs zero GPU.** The cluster is the bottleneck for three results; everything else is writing and
rescoring, and that can run in parallel from day one.

## Week 1 (Aug 5-11) - reframe + free fixes + A starts

GPU (dependency chain, must be in this order):
- [ ] `dxcache32` (9795073) finishes -> DexYCB T=32 cache exists
- [ ] **A, arm 1: bracketing** (DexYCB 0.780 + 2 shallow), T=32, volume-matched
- [ ] **A, arm 2: non-bracketing** (3 shallow), T=32, identical clip count

Zero GPU, do all of it this week:
- [ ] **Rewrite around the new claim.** Title, abstract, intro, and a Method cut to what survives.
      This is the single biggest task in the six weeks and it does not need a single result.
- [ ] E into `tab:boxsweep` as the constant-depth floor row (already computed)
- [ ] Two artifact baseline cells -> `\na` (HaPTIC C-abs 153.6, Dyn-HaMR 1327.4/276.6)
- [ ] Resolve the Tab. 1 / Tab. 2 contradiction; one policy, per-row pipeline provenance
- [ ] Disclose box-jitter augmentation + report the non-augmented 49.8
- [ ] State hand set and joint set everywhere (right hand only, 16 joints, no fingertips)
- [ ] Native-regime column for every baseline (HaWoR 49.7 alongside 87.7)

## Week 2 (Aug 12-18) - A lands, B runs, draft v1

GPU:
- [ ] **B: multi-store transfer sweep**, zero training. One existing checkpoint evaluated zero-shot
      on HOT3D 0.339 / OakInk2 0.386 / ARCTIC 0.474 / H2O 0.503 / DexYCB 0.780. Five genuinely
      held-out between-store points, plus leave-one-out prediction error. Control the box convention
      across stores FIRST or the sweep measures conventions.
- [ ] **C: world re-run with the scale decomposition** (+GT scale only / +GT rot only / +GT full),
      full 157 seqs rather than 60, with `--dump_cam_dir` on so **D unblocks for free**

Zero GPU:
- [ ] **D: protocol sweep** on the dumps from C. Segment {30,50,100,128} x window {2,10,30}.
      Becomes a standalone figure with a call to action. Check whether method RANKING flips anywhere
      in the grid; if it does, that alone earns the section.
- [ ] Limitations section + `\appendix` (uncomment `main.tex:85`); start migrating the 25 todos

## Week 3 (Aug 19-25) - the contribution-2 decision point

GPU:
- [ ] **Patch `freeze_gs_head`** so it freezes the head but NOT the injection convs (one line), then
      one run with `enable_gs: true`. Sec. 3.2 stays by your decision, so this is mandatory.
- [ ] Hand-region-masked PSNR/LPIPS eval for it
- [ ] If budget: `use_global_context=False` at 10 epochs, matched

Zero GPU:
- [ ] Rerun E on the full 157-seq set with the TRAIN median, not the test median
- [ ] Figures: depth-vs-time trajectory plot (cheapest, predictions already dumped)

## Week 4 (Aug 26-Sep 1) - figures + last experiments

- [ ] Page-1 teaser (three-panel taxonomy, Hand3R Fig. 1 shape). Needs the intro to exist first.
- [ ] Metric-depth side-by-side vs HaWoR/WiLoR, same scene, true depth. Needs baseline meshes dumped.
- [ ] Regenerate the qualitative figure from the final checkpoint, one sequence three ways
- [ ] Optional, cheap, possible real finding: **`eq:geom` in angular units** (`atan((x-px)/fx)`)
      per WildHands/EgoForce. The pixel encoding cannot transfer across 3 camera models, so the
      0.7 mm null is plausibly a diagnosis rather than a dead end.

## Week 5 (Sep 2-8) - integrate

- [ ] Every number traced to one configuration table (checkpoint, train set, T, box source, jitter
      on/off, seq count, GS on/off). Eight different "our HOI4D number" currently circulate.
- [ ] Full internal read-through; rerun the adversarial + style reviewers on the finished draft
- [ ] Fix whatever they find

## Week 6 (Sep 9-16) - polish and buffer

- [ ] Supplementary assembled, licences, code/checkpoint release statement
- [ ] Deliberately empty for slippage. Something will slip.

---

## Kill criteria, decide by end of Week 2

**Stop and target a later venue if:**
- **A comes back near 66 mm for the non-bracketing arm.** Depth coverage is then not the mechanism,
  it is just more data, and the headline contribution ceases to exist. There is no fallback thesis.
- **B comes back non-monotone, or r below ~0.5 across stores.** The relation is then a HOI4D<->H2O
  coincidence, and the same verdict applies.

Everything else that could fail removes a supporting claim, not the thesis:
- scale decomposition unfavourable -> contribution 2 dies, contribution 1 survives
- fusion ablation null -> supports the thesis under the new framing
- constant-depth floor close to the model -> already refuted, E shows a 45% margin

## What NOT to do, decided

- No backbone unfreezing. The GT->detector asymmetry (+12.0 ours vs +4.4 HaMeR_ft) points at crop
  RESOLUTION, not capacity; unfreezing voids the ~1 TB cache and makes runs ~10x slower; and it
  destroys the DINOv2 control, which is one of the few genuinely strong results.
- No TACO acquisition, no EgoForce/EgoAllo/Human3R baselines, no rebuilding the baseline tables at
  T=32. Each is a schedule killer for a marginal gain.
