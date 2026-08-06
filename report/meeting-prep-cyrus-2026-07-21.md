# Study guide for the Cyrus meeting (2026-07-21)

Written to be read by someone who knows nothing about the project. It explains every
concept from zero, then summarizes what changed in the two months since the
"Feedforward 4D Gaussians for Egocentric Vision" report, then gives the current numbers
and the likely questions.

---

## 0. The one-paragraph summary

The old report was a **scene-reconstruction** paper: put a hand prior into a frozen 4D
Gaussian world model (NeoVerse/VGGT) and show it improves scene rendering (PSNR 36.96 →
37.92 dB on HOT3D). Over the last two months the project **pivoted to a hand-pose paper**:
the real, defensible contribution is a lightweight head on a frozen backbone that predicts
the hand in **absolute metric camera coordinates** (not just relative shape), evaluated
against strong hand baselines on **HOI4D** (and H2O). The current headline is that in the
**global / world-space** setting, short-window, with our own detector boxes, we beat the
published SOTA (HaWoR, Hand3R) on absolute camera-frame joint error and on short-window
world trajectory error, at a tiny fraction of their trainable parameters.

---

## 1. The architecture (recap, unchanged since the report)

```
monocular egocentric video (T frames)
        │
        ▼
  FROZEN VGGT / NeoVerse backbone   ──►  per-frame: camera pose, DEPTH map, scene Gaussians
        │ (multi-level dense feature tokens)
        ▼
  HaMeR-style HAND HEAD (transformer decoder)   ──►  MANO hand params (both hands)
        ▲                                              per hand: translation t (3),
   hand bounding box (feature-space crop)              global rotation, pose (15 joints),
        │                                              shape β
   local crop tokens  ⊕  global scene tokens (cross-attention = "local-global fusion")
```

Key terms:

- **VGGT / NeoVerse backbone**: a pretrained "visual geometry" transformer that, from raw
  video, predicts camera poses, a dense depth map, and 3D Gaussians for the scene. We keep
  it **frozen** (never train it) to preserve its geometry knowledge and keep training cheap.
  All the trainable capacity lives in the hand head and the small injection layers.
- **MANO**: the standard parametric hand model. A hand is described by ~48 numbers
  (a 3D translation, a global rotation, 15 joint rotations, 10 shape parameters). Feeding
  those numbers through MANO gives you a 3D hand mesh and 21 3D joint positions. So "predict
  the hand" = "predict these MANO numbers".
- **HaMeR head**: the transformer that regresses MANO params. HaMeR is a well-known hand
  reconstruction network; we reuse its head design but run it on our frozen backbone's
  features instead of its own ResNet/ViT.
- **Feature-space cropping**: instead of cropping the image around each hand and running a
  whole network per hand (expensive), we crop the already-computed feature map around the
  hand box. One backbone pass serves the whole frame regardless of how many hands.
- **Local-global fusion**: the hand crop tokens cross-attend to the full-frame tokens, so
  the hand prediction "sees" the surrounding scene (needed to place the hand correctly in
  the world, which a local crop alone cannot determine).

---

## 2. Coordinate frames and "root-relative" (essential for every metric)

A predicted hand can be judged in three different frames. This distinction is the single
most important thing to understand for the metrics.

- **Camera frame (absolute):** the hand's 3D joints expressed in the camera's own
  coordinate system, in real metres, including how far the hand is from the camera. Getting
  this right requires predicting the **absolute depth** of the hand ("the hand is 42 cm from
  the camera"). This is HARD from a single image.
- **Root-relative:** take the hand's joints and subtract the wrist (the "root") position, so
  the wrist sits at the origin. This throws away *where* the hand is and keeps only its
  *shape / articulation* (finger bending, pose). This is comparatively EASY: you don't need
  absolute depth, just the local geometry.
- **World frame (global):** every frame's camera-frame hand is placed into one shared world
  using the camera trajectory (extrinsics), so you can see the hand move through the room
  over the whole clip.

**Why it matters:** almost every method is good at root-relative shape and bad at absolute
placement. Our contribution is specifically about **absolute** placement, so we must always
be clear which frame a number is in.

---

## 3. The metrics — what each one measures

All are joint/vertex errors in millimetres, lower is better. "MPJPE" = Mean Per-Joint
Position Error (average distance between predicted and ground-truth joints). "MPVPE" is the
same on mesh vertices.

| Metric | Frame | What it measures | Notes |
|---|---|---|---|
| **C-MPJPE (a.k.a. C-abs)** | camera, absolute | Full 3D joint error including absolute depth. **This is our headline / the hard one.** | Hand3R calls this "C-MPJPE"; it is our "C_abs". |
| **C-rr (root-relative MPJPE / "local MPJPE")** | camera, wrist-centered | Hand *shape/articulation* only, wrist error removed | Easy; most methods are competitive here |
| **wrist_abs** | camera, absolute | Error of just the wrist (root) position | Isolates "how well do we place the hand" from "how well do we bend the fingers"; when placement dominates, wrist_abs ≈ C_abs |
| **W-MPJPE (global trajectory, "W")** | world | Take each frame's camera-frame prediction, lift it into one world frame via the camera extrinsics, **rigidly align the first window**, then measure joint error against world-frame GT. Keeps global drift. | Sensitive to trajectory drift + scale. Long windows explode; short windows are fair. |
| **WA-MPJPE (window-aligned)** | world | Same as W but align the **whole trajectory** rigidly before measuring, so global drift/scale is absorbed. Measures shape-in-world after best rigid fit. | "WA" = window-aligned. Easier than W. |
| **PA-MPJPE / PA-MPVPE** | Procrustes-aligned | Align prediction to GT with rotation+translation+**scale** per frame, then measure. Removes placement AND scale entirely. | This is what the OLD report used. It hides exactly the thing we now care about (absolute placement). |
| **AUC-V / AUC-J** | — | Area under the "percent of correct keypoints vs threshold" curve for vertices/joints | Summary of the PA curves; higher is better |

**Short vs long window (for W/WA):** a "window" is a run of consecutive frames aligned
together. Short = 30 continuous frames (the Hand3R protocol), long = 120. We follow Cyrus:
**short window only**, because feedforward/online methods aren't built to hold a
long-duration global trajectory (drift accumulates), and it isn't a fair or flattering
comparison. Our own measurements confirmed W grows with window length (≈33 mm at 30 frames,
≈196 mm over a full sequence) — the error is low-frequency drift, not per-frame jitter.

**The W-MPJPE mechanics in one line** (your own phrasing, which is correct): W-MPJPE is
joint error in the **world** frame, not the camera frame — you place each frame's
camera-frame prediction into one shared world frame using the camera trajectory
(extrinsics), rigidly align the first window, then measure against world-frame GT.

---

## 4. The losses (training objective)

We train the heads jointly (backbone stays frozen). The total loss is a weighted sum:

- **Photometric losses (scene):** L1 + **LPIPS** between the rendered Gaussian frame and the
  real frame. LPIPS = a "perceptual" distance using deep features (looks-similar rather than
  pixel-identical).
- **MANO parameter loss:** L2 on the predicted MANO numbers (translation, rotation, pose,
  shape) vs GT.
- **kp3d (3D joint loss):** L2 between predicted and GT 3D joints. This comes in two flavours,
  and the distinction is central to the whole project:
  - root-relative kp3d — supervises shape only.
  - **kp3d_abs** — supervises the **absolute** 3D joints (with depth). See §6.
- **kp2d (2D reprojection loss):** project the predicted 3D joints into the image and compare
  to GT 2D keypoints. (Caveat we found: the kp2d term was hardcoded for the Aria/HOT3D image
  convention — 1408 px width + a 90° rotation — which is wrong for plain-pinhole 224 px HOI4D/
  H2O data. It was mildly *harmful* on both datasets; turning it off (weight 0) slightly
  improves the headline. See §11.)

---

## 5. The datasets and the baselines (the cast)

**Datasets**
- **HOT3D**: egocentric hand-object dataset captured with Project Aria glasses. The old
  report trained/evaluated only here. High-quality MANO + camera tracks.
- **HOI4D**: large egocentric hand-object dataset with **dense ground-truth depth**. This is
  now our primary benchmark, because it lets us measure absolute placement honestly and
  matches the protocol Hand3R reports on.
- **H2O**: a second hand-object dataset, used as the cross-dataset check ("does the recipe
  generalize beyond HOI4D?").

**Baselines (other methods we compare against)**
- **HaMeR**: strong single-image hand reconstruction transformer. Off-the-shelf it is not
  trained on egocentric viewpoints, so we also train "metric-tuned" and "fully fine-tuned"
  versions for fair comparison.
- **WiLoR**: another strong hand detector+reconstructor; we use its detector for boxes and
  compare its reconstruction.
- **HaWoR**: hand-in-world method = a per-frame hand estimator + **DROID-SLAM** for the camera
  trajectory + Metric3D for scale. This is the main *world-space* baseline. (This is the one
  we are currently building on Euler to run ourselves rather than only cite.)
- **Hand3R**: the closest concurrent method — online 4D hand-scene reconstruction. It reports
  HOI4D global metrics (C-MPJPE, W, WA), which is exactly our comparison table.

- **DROID-SLAM**: a learned SLAM system (Simultaneous Localization And Mapping). Given a video
  it recovers the **camera trajectory** (where the camera was each frame) and a sparse map. In
  HaWoR it is what ties the per-frame hand estimates into a single world frame. It needs custom
  CUDA extensions and is the painful part of the HaWoR build. Important nuance we learned:
  HaWoR estimates the hand in a *world* frame, so even the *camera-frame* joints are obtained
  by transforming world→camera using SLAM — i.e. you cannot skip SLAM even if you only want the
  camera-frame number.

---

## 6. kp3d_abs — the main lever (know this cold)

**Problem:** predicting the *shape* of a hand is easy; predicting *how far it is from the
camera* (absolute depth) is hard, and that is what C-MPJPE / W depend on.

**kp3d_abs** is simply the 3D-joint loss applied to **absolute** joint positions (with their
real depth), given a nonzero weight in training. Turning this weight up (0 → 0.3 → 0.5 → 1.0)
forces the head to actually learn absolute depth instead of only relative shape.

**Evidence it is the lever:**
- On HOT3D it moved absolute camera-frame error from ~115 mm down to ~53 mm and world W from
  308 → 250 while relative shape stayed as good as Hand3R.
- Controls confirmed it causally: the *transl_z_weight* knob alone (just weighting the
  translation-depth term) did **not** help; kp3d_abs did. This is why the final recipe uses
  kp3d_abs = 1.0.
- The final HOI4D headline (23.6 mm C_abs) is trained with kp3d_abs; a zeroed-loss control
  confirmed the number depends on it.

One subtlety worth stating: absolute depth and world scale are **coupled** — the hand's
metric scale is `z_hand / scene_depth`, so improving absolute depth also changes the scale
used to lift into the world. That coupling is why fixing C_abs did not automatically fix W.

---

## 7. Unfreezing — how many layers, and where

There are two different "unfreeze" stories; keep them separate.

1. **Our final model:** backbone **fully frozen**; only the **hand head + injection layers**
   are trainable. That is ~**46 million** trainable parameters. (For comparison, fully
   fine-tuning HaMeR is ~632 M — we train roughly 1/14 as many parameters.) This
   parameter-efficiency is a real selling point.
2. **An exploration (HOT3D era):** we tried unfreezing the **last 4 frame blocks + last 4
   global blocks** of the backbone (via `unfreeze_last_n_blocks`) plus kp3d_abs. On the small
   11-sequence set it cut C_abs 114 → 53 mm, which showed absolute depth can be squeezed out
   architecturally. But with more data (367 sequences) + kp3d_abs, the **frozen** head alone
   reaches 23.6 mm, so the shipped recipe keeps the backbone frozen.
3. **Baselines we unfroze for fairness:** to compare against a "properly tuned" HaMeR we ran
   `fu` = full unfreeze (fine-tune all of HaMeR) and `fj` = full unfreeze + box jitter. These
   are the strong baselines that made us tighten our claims (see §9, §10).

---

## 8. jitterbox / "jitterrob" (box jitter augmentation)

The hand head crops around a **bounding box**. At training time we knew the box exactly (GT
box); at test time a real detector gives noisier boxes. "**Box jitter**" augmentation randomly
perturbs the training box (shifts/scales it) so the head learns to tolerate imperfect boxes.
`jitterrob` / `jitter` is our box-jitter-trained model; `fj` is the box-jitter HaMeR baseline.
This matters because our robustness-to-detector-boxes claim rests on it: with our own detector
boxes (the realistic "end-to-end" setting) the jitter-trained model degrades much less.

- End-to-end (our detector boxes), C_abs: jitter-trained **35.6** vs no-aug 49.8.
- Root-relative C_rr with jitter: 26.5; wrist_abs 34.8.

---

## 9. GT-box confound and "box parity" (why we keep re-checking)

A subtle trap: if *we* use ground-truth boxes but the baselines use *detector* boxes, our
advantage might just be "we had easier inputs". So we spent effort on **box parity** —
scoring everyone on the *same* boxes:

- With ground-truth boxes, HaMeR fine-tuned is actually a bit better than us (GT-box: fj 20.8
  vs ours 23.6) — we do **not** claim to be more accurate given identical clean boxes.
- The honest, defensible framing is a **crossover**: we start slightly worse with clean
  GT boxes but end up **more robust** with realistic detector boxes, at ~14× fewer trainable
  parameters. (A recent check even showed the end-to-end crossover shrinks when boxes are
  perfectly matched, so we lead with parameter-efficiency + the world-space win, not "we beat
  HaMeR end-to-end".)

This is exactly the kind of self-auditing Cyrus and reviewers reward; be ready to explain it
plainly.

---

## 10. Backbone-swap ablation (a "null result" that reshaped the story)

Question: is our accuracy coming from the *reconstruction features* of the special backbone,
or just from the recipe (kp3d_abs + good boxes + data)?

We swapped the backbone: reconstruction backbone vs a plain **frozen DINOv2** vs a
**random-initialized** backbone (3 seeds each):
- recon: 23.6 ± 0.8
- DINO: 22.2 ± 0.7 (as good or better)
- random-init: 27.7 (clearly worse)

**Conclusion:** any frozen ViT + our recipe reaches ~22–24 mm. So "reconstruction features
encode hand depth" is **not** the novelty. The driver is the **recipe** (kp3d_abs + geometry
injection + data). This is why the paper's story is now "recipe + analysis +
parameter-efficient robustness", not "our backbone is special". Honest, and it survives
scrutiny.

---

## 11. Depth: DA3 vs DINO vs "which depth do we actually use?"

Common confusion — three different things:
- **The depth we actually use:** the frozen **VGGT/NeoVerse backbone's own predicted depth
  map**, plus the head's absolute-depth prediction trained by kp3d_abs. On HOI4D that
  predicted depth is genuinely good when supervised (AbsRel ~0.08).
- **DINO:** only appeared in the **backbone-swap ablation** as an alternative *feature*
  backbone (not a depth source). It showed the backbone is interchangeable (§10).
- **DA3 (Depth Anything 3):** an external metric-depth model we **probed** as a possible
  reference for a wrist "anchor". At the ground-truth wrist pixel DA3 was ~40 mm (great), but
  at the *predicted* wrist pixel it was no better than our head, so the anchor was a **wash**
  and DA3 is **not** in the final model. Useful to know it was tried and honestly reported as
  neutral.

So: "we use VGGT's depth + kp3d_abs-trained head depth; DINO and DA3 were ablation/probe
experiments, not part of the shipped model."

---

## 12. Contact anchor (tried, honestly reported as neutral)

Idea: when the hand touches an object, its depth equals the object/scene depth there, so you
could "anchor" the hand's absolute depth to the scene at contact frames.

- Implemented with an explicit GT contact mask (|wrist_z − scene_depth_at_wrist| < 5 cm).
- Result on HOI4D: **neutral** — it removes the harm of a naive wide "contact band" but does
  not *add* accuracy, because the head's own placement is already good and the frozen scene
  depth at the thin hand is unreliable (it reads the background behind the hand).
- On HOT3D a periodic *oracle* re-anchor to GT collapses W (324 → 65), which proved the
  headroom is in absolute placement, but no *feedforward* reference we have (contact or DA3)
  is clean enough to realize it.

Takeaway line: "the post-hoc anchor is dead; absolute placement has to come from the head
itself (kp3d_abs), not from a scene/contact reference."

---

## 13. Temporal head / test-time filtering (tried, dead — good to know)

Cyrus asked whether a temporal component could help the world trajectory. We checked before
building it:
- **Test-time smoothing** (low-pass the trajectory): buys only ~4% and saturates → the W
  error is **low-frequency drift**, not high-frequency jitter, so smoothing cannot fix it.
- **Velocity / motion head:** even feeding *perfect* GT motion did **not** lower W. Fixing
  motion is the wrong target.
- What *would* fix W is periodic **absolute** re-placement (the anchor oracle), not temporal
  filtering.

So there is **no temporal head** in the model, and test-time filtering is intentionally off.
This matches Cyrus's own instinct that filtering was inelegant.

---

## 14. What was actually done in the last two months (report → now)

1. **Reframed the contribution** from "hand prior helps scene rendering" (PSNR) to
   "feedforward **absolute** hand pose", because reviewers value the absolute-placement result
   and the scene-metric claims did not hold up (frozen backbone doesn't model the thin hand).
2. **Moved the benchmark to HOI4D** (dense GT depth, matches Hand3R's protocol) and added
   **H2O** as a second dataset.
3. **Found and validated kp3d_abs** as the absolute-placement lever (§6); locked the recipe.
4. **Ran the honest controls** reviewers demand: backbone-swap null (§10), GT-box confound /
   box parity (§9), kp3d_abs causal control, 3-seed error bars, data-leakage audit,
   world-space quarantine.
5. **Built strong baselines** ourselves: metric-tuned HaMeR, fully-fine-tuned HaMeR, box-jitter
   HaMeR (fu/fj), native WiLoR/HaMeR — so the comparison is fair, not cherry-picked.
6. **Killed dead ends cleanly** (with evidence): contact/DA3 anchor (wash), temporal/velocity
   head + test-time smoothing (dead), within-clip pose refinement (harmful), hand-as-scene-scale
   (refuted).
7. **Built the global / world-space comparison** (Cyrus's current top priority): short-window
   C-MPJPE / W / WA against HaWoR, Hand3R, HaMeR+SLAM, WiLoR+SLAM.
8. **Parameter-efficiency sweep** (Cyrus's "minimize params" ask): how small can the head get
   before accuracy breaks (§16).
9. **In progress:** running HaWoR ourselves on Euler (a real self-run row instead of only
   citing its paper numbers), and a native-HD HaPTIC baseline.

---

## 15. Current verified numbers (all pulled from result JSONs)

**HOI4D absolute camera-frame (C_abs), full 157 sequences:**
- Ours, GT boxes: **23.6 mm** (3-seed 23.57 ± 0.84); with the kp2d bug fixed, 22.9.
- Ours, our own detector boxes, box-jitter (end-to-end): **35.6 mm** C_abs / 26.5 C_rr /
  34.8 wrist.
- Baselines on HOI4D (absolute camera C-MPJPE, from the Hand3R paper): HaMeR+SLAM 248.2,
  WiLoR+SLAM 252.2, HaWoR 51.8, Hand3R 42.6.
- Metric-tuned HaMeR (our fair baseline): 39.8 GT-box / 68.7 detector-box. Native off-the-shelf
  WiLoR 83.4 / HaMeR 88.0.

**Global / world-space, short 30-frame window, our detector boxes (the current headline table):**

| Method | C-MPJPE ↓ | WA-MPJPE ↓ | W-MPJPE ↓ |
|---|---|---|---|
| HaMeR + SLAM | 248.2 | 52.7 | 140.8 |
| WiLoR + SLAM | 252.2 | 52.9 | 146.9 |
| HaWoR | 51.8 | **22.5** | 41.3 |
| Hand3R | 42.6 | 38.0 | 86.9 |
| **Ours (feedforward, no SLAM)** | **33.8** | 28.0 | **33.2** |

We win C-MPJPE and short-window W; second on WA (HaWoR wins that). All "ours" cells come from
one end-to-end run (jitter model, detector boxes, predicted extrinsics).

**Backbone swap (3-seed, GT-box C_abs):** recon 23.57 ± 0.84 / DINO 22.24 ± 0.68 /
random 27.7.

---

## 16. Parameter-efficiency sweep (Cyrus's "minimize params" question)

How small can the head be and still work (from-scratch, full-157 GT-box C_abs):

| Head size | Trainable params | C_abs |
|---|---|---|
| full (dim 1024, depth 6) | 46.3 M | 23.1 |
| half (dim 512) | 19.5 M | 24.98 |
| quarter (dim 256) | 8.8 M | 27.27 |
| tiny (dim 256, depth 3) | 5.27 M | 26.63 |

(See `report/figures/head_param_sweep.png`.)

Story: the head shrinks ~9× (46 M → 5 M) for only ~3.5 mm of degradation and still stays far
ahead of the SLAM baselines — i.e. the method is cheap, which reinforces the parameter-efficiency
angle (vs 632 M to fully fine-tune HaMeR).

**Important honesty note (be ready for this):** these sweep points are **single-seed**, and our
3-seed std is ±0.8 mm. So the tiny (26.63) vs quarter (27.27) "inversion" is **within noise** —
do NOT claim tiny beats quarter. The real, monotonic lever is **width**: dim 256 (tiny/quarter)
≈ 25–27, dim 512 (half) 25.0, dim 1024 (full) 23.1.

**Cyrus's steer (2026-07-21) — the sweep axis should change:** *do not reduce `dim`* (keep the
head full width); instead sweep **how many layers are trained** — "only head" (frozen backbone +
head, our current recipe) vs "**last X backbone layers unfrozen + head**". That is the meaningful
"how much do we need to tune" question, and it connects to our earlier result that unfreezing the
last 4+4 backbone blocks cut HOT3D C_abs 114→53. Planned experiment: at full head width, sweep
n_unfreeze ∈ {0 (head-only), last-1, last-2, last-4} backbone blocks on HOI4D (367 train / 157
test), reporting trainable params and C_abs. Trade-off to state: unfreezing backbone blocks
requires backprop through the backbone (no frozen-feature cache → slower training, more params
per block). The dim-sweep above is now a *secondary* "the head itself is small" point, not the
headline params result.

---

## 17. Where we stand + what is running right now

- **Story / venue:** honest current position is a **3DV/WACV**-strength paper as constituted,
  with **CVPR (Nov 2026)** the target if the global-comparison win holds and the last
  reviewer must-haves close. Lead with: (1) global/world-space win vs HaWoR/Hand3R short
  window, (2) parameter efficiency, (3) rigorous honest controls. Do **not** over-claim "we
  beat HaMeR end-to-end" — own the GT-box/articulation trade openly.
- **Running now:** HaWoR self-run build on Euler (so we report a self-run row, not just cite);
  the parameter sweep finishes its smallest head tonight.
- **Still queued:** DINO 3-seed + a trainable-params column in the tables; native-HD HaPTIC
  baseline; H2O cross-dataset calibration.

---

## 18. Likely Cyrus questions and crisp answers

- *"Is the global comparison fair and full?"* — Short window only (feedforward's fair
  regime), our own detector boxes (not GT), predicted extrinsics (not oracle), same metric
  definitions as Hand3R. We win C-MPJPE and W-short; second on WA. We are also building HaWoR
  ourselves so at least one baseline row is self-run, not cited.
- *"How many parameters do you tune?"* — ~46 M (head + injection), ≈1/14 of full HaMeR
  fine-tuning; and it degrades gracefully down to ~5 M (see §16).
- *"Why is W bad on long windows?"* — W is low-frequency trajectory drift that accumulates
  with duration; short-window is the honest feedforward regime, which is also what Hand3R
  reports. We showed smoothing/velocity heads can't fix it; only absolute re-placement can.
- *"Isn't your win just from ground-truth boxes?"* — No: box-parity checks show we don't claim
  accuracy superiority given identical clean boxes; the claim is robustness to realistic
  detector boxes + parameter efficiency + the world-space win.
- *"What makes the backbone special?"* — Nothing (honestly): the backbone-swap null shows the
  recipe (kp3d_abs + geometry + data), not the features, is the driver.
- *"Did the anchor / temporal ideas work?"* — Reported honestly as neutral/dead with evidence;
  absolute placement must come from the head (kp3d_abs), not post-hoc anchoring or filtering.

---

*Numbers here are pulled from result JSONs and the memory ledger; the world-space table and
HOI4D headline are the load-bearing results — double-check the exact cell before quoting a
decimal in the meeting.*
