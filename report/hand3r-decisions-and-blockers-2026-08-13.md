# Hand3R comparison: the table, the blocker, and what Cyrus has to decide

## 1. The table, with Hand3R's published numbers next to ours

Two protocols, side by side, NOT a comparison. Every difference is listed under it.

| Method | Type | Pipeline | C-MPJPE | Short WA | Short W | Long WA | Long W |
|---|---|---|---|---|---|---|---|
| **Hand3R's Table II** (their split, GT boxes, 21 joints) ||||||||
| HaMeR-SLAM | offline | multi-stage | 248.23 | 52.69 | 140.75 | 85.46 | 218.05 |
| WiLoR-SLAM | offline | multi-stage | 252.24 | 52.91 | 146.91 | 87.51 | 223.00 |
| HaWoR | offline | multi-stage | 51.77 | 22.54 | 41.28 | 27.40 | 58.62 |
| Hand3R | online | one-stage | 42.6 | 38.04 | 86.87 | 56.71 | 125.81 |
| **Ours** (our 157-seq split, detbox v3, 16 joints, their W gauge) ||||||||
| HaMeR + SLAM | offline | multi-stage | 87.9 | 34.7 | 103.3 | 57.5 | 219.9 |
| WiLoR + SLAM | offline | multi-stage | 83.4 | 33.4 | 96.8 | 56.3 | 209.9 |
| HaWoR | offline | multi-stage | 87.7 | 40.1 | 118.4 | 61.0 | 229.4 |
| HaPTIC + SLAM | offline | multi-stage | (157.1) | 36.7 | (117.0) | 57.1 | (263.1) |
| Dyn-HaMR | offline | optimisation | (1336.7) | 49.3 | (164.8) | 69.0 | (328.2) |
| **WorldHand4DGS** | online | one-stage | **35.4** | **28.1** | **73.0** | **43.3** | **121.2** |

Differences between the two blocks: split (67.5% of our test set is their training data),
boxes (theirs GT-derived on every row, ours a shared detector everywhere), joints (21 with
fingertips vs 16), hand set (theirs right-preferred with left fallback, ours right only),
training data (Hand3R = DexYCB stage 1 + HOI4D stage 2; ours = five datasets, HOI4D held out
in the final protocol).

### What survives the caveats

Two observations do not depend on the split, because they are internal to each block.

**Hand3R loses to HaWoR on every world metric in its own table, and says why.** Sec. IV-C:
"we observe a performance gap in global metrics... Offline approaches inherently benefit from
global bundle adjustment, which utilizes future frames to refine camera trajectories. In contrast,
online methods are restricted to causal estimation, where accumulated drift is inevitable."
They frame the online-vs-offline gap as structural and unbridgeable.

**In our block, the online method beats every offline one, including HaWoR.** 28.1 against 40.1
short WA, 73.0 against 118.4 short W. If that holds under the matched protocol, it contradicts the
structural claim, and that is a stronger result than any margin in the table.

**The SLAM rows agree across blocks and the HaWoR rows do not.** Long W: HaMeR 219.9 vs 218.05,
WiLoR 209.9 vs 223.00, both within a few percent despite everything differing. HaWoR is 229.4 vs
58.62. The +SLAM baselines are reproducible across protocols and HaWoR is not, which points at the
box source: their HaWoR runs GT-derived boxes in its native regime, ours runs detbox v3.

## 2. THE BLOCKER: we have no hand-pose GT for the 222 clips

Verified today, three ways:

- `HOI4D_release.zip` on the yinloonga mirror: 15,700 entries, **all `align_rgb` video**. No poses.
- `HOI4D_annotations.zip`: 1,820,009 entries, all objpose / 2dseg / 3dseg / action. **Zero handpose.**
- The mirror repo holds only those two plus `camera_params.zip`.

Our own coverage: `hoi4d_pp_full` has processed GT for **524** sequences, which is exactly our
367 train + 157 test. Of the 222 mutually-held-out clips it covers **0**.

So the videos for the 222 are scriptable and the ground truth is not. HOI4D hand pose is
**OneDrive-only** and has to be fetched by hand ([[hoi4d-pose-acquisition]]). Until someone does
that, the matched-protocol evaluation cannot run, for us or for Hand3R.

**This is the single item on the critical path, and it needs a human with a browser, not a job.**

## 3. What we can do without asking Hand3R anything

Everything except the Hand3R row. Their code and checkpoints are private, so only they can produce
it. Ours, HaMeR+SLAM, WiLoR+SLAM, HaWoR, HaPTIC+SLAM and Dyn-HaMR are all ours to run.

The right order is therefore: get the GT, build the 222-clip store, run our six rows, and only then
send them a package that already contains everything except their own row. That is also a better
thing to send, because it shows the protocol working rather than proposing it.

## 4. The missing-detection policy, and why this one

Their README: "dropping missed detections can bias the result; define and disclose a common
missing-detection policy for every method."

**Proposal: score the intersection, identically for all methods, and publish the coverage.**

1. A frame is scorable when the shared detector fired AND the GT hand is annotated.
2. That frame set is computed once and applied to every method, ours included. Since all methods
   receive the same boxes, the set is identical by construction, so no method is advantaged.
3. A method that internally refuses a frame it was given is recorded separately as a per-method
   coverage number and printed in the table.
4. Report the frame count and the coverage percentage per row, not just the error.

Why not the alternatives: a fixed penalty per miss is arbitrary and unfalsifiable, and it lets the
penalty constant decide the ranking. Interpolating a missing box invents an input and hides the
detector's failure inside the pose error. Letting each method fall back to its own detector is
exactly the box-source confound we spent this project removing.

The one honest cost of the intersection rule is that it measures pose quality on frames where
detection succeeded, not end-to-end system quality including detection. We state that, and we
report coverage so a reader can see how much was dropped.

## 5. What to ask Cyrus

Only decisions that are his, with a recommendation on each.

1. **Fetch the HOI4D hand-pose GT from OneDrive?** Blocks everything else. Nobody can script it.
   *Recommend: yes, today, it is the critical path.*
2. **Accept the offer to have the Hand3R authors run their model for us?** This means sending an
   external group our detector boxes, prediction format, scorer and eventually results before
   submission. *Recommend: yes. It is the only route to a real Hand3R row, they asked to see
   unfavourable results too, and they offered first.*
3. **Does the HOI4D evaluation set move from our 157 to the 222?** Changes every HOI4D number in
   the paper. *Recommend: report both. The 157 keeps continuity with every ablation; the 222 is the
   only set on which Hand3R can be compared.*
4. **Adopt Hand3R's W gauge as the primary one?** It doubles every global number we report.
   *Recommend: yes, report both columns. It is the better measurement, ours degenerates at 30
   frames, and theirs costs us less than any baseline so our lead widens.*
5. **The missing-detection policy above.** *Recommend: approve as written.*
6. **TACO's feature cache may not fit.** See below. *Recommend: he should know before we spend the
   quota.*

## 6. Is the five-dataset protocol ready? No.

Stores converted: ARCTIC 267, OakInk2 109, HOT3D 198, DexYCB 2969, TACO 2311, HOI4D 367 train /
157 test. Ego-Exo4D is empty and dropped (no MANO).

32-frame feature caches exist for **four** of the five: arctic, dexycb, hot3d, oakink2, totalling
1.2 TB over ~37.6k clips. **TACO has no cache.**

Two problems with building it:

- **Space.** Scratch is at 2.19 TB of a 2.70 TB hard quota, so about 500 GB is free. At the
  measured ~32 MB per clip, TACO's 2,311 sequences plausibly land between 0.8 and 1.1 TB. It does
  not fit. Options: a coarser stride for TACO, subsampling the sequences, or deleting a cache we
  are done with.
- **The config is stale.** `train_mix5_all.yaml` still lists hoi4d + oakink2 + arctic + hot3d +
  egoexo4d, which includes a dataset that is empty and HOI4D, which the final protocol holds out.

So: five stores exist, four are trainable, and the fifth needs a space decision before it can be.
