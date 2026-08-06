# Scaling up training to a mixed dataset set — dataset plan

> **SUPERVISOR CONSTRAINTS (Cyrus, 2026-07-21) — these override the survey below:**
> 1. **PINHOLE cameras only.** Cannot mix-train across camera types. Fish-eye datasets
>    (Aria: HOT3D, Ego-Exo4D) must be **undistorted to pinhole** first
>    (repo: `scripts/preprocessing/preprocess_undistort.py`, `pinhole_undistort_ops.py`).
>    HOI4D + H2O are already pinhole.
> 2. **ARCTIC — accurate, include in training** (Cyrus has used it; verify ego view is pinhole).
> 3. **AssemblyHands — caution:** some images are grayscale → may not suit; verify or drop.
> 4. **Ego-Exo4D → TRAINING, not held-out test.** Massive + complex daily activities, harder than
>    lab-setting datasets; common practice trains on it. (Fish-eye → undistort.) A *different*
>    held-out "new" test dataset is therefore needed.
> 5. **HELD-OUT TEST = OakInk2 (CONFIRMED).** Pinhole verified from the toolkit (cam_intr = 3x3 K,
>    no fisheye coeffs); egocentric view + MANO GT; scriptable from HuggingFace
>    kelvin34501/OakInk-v2. Never trained on -> the generalization test.
>
> **Revised train-mix leaning:** HOI4D + H2O (have, pinhole) + ARCTIC + HOT3D (undistort) +
> Ego-Exo4D (undistort, big). **Step 0 (HOI4D+H2O) is unaffected — both pinhole.**


Compiled 2026-07-18. Purpose: move from single-dataset training (HOI4D) + two-dataset eval
(HOI4D, H2O) to a **mixed 3-5 dataset** training regime (the WiLoR / HaWoR recipe), with **one
"extremely new" held-out dataset** reserved purely for testing (true generalization). Target
protocol is unchanged: **absolute camera-frame 3D joint error (C-MPJPE / C_abs) in metric mm**,
egocentric.

All dataset facts below were web-verified by three research passes (URLs cited per row). Numbers
flagged *unconfirmed* were not on an authoritative page and must be double-checked before they
enter a paper.

---

## 0. What we already have (repo / cluster state)

The training pipeline consumes one cache format only: `HOT3DHandDataset` auto-discovers
sequences and loads a per-sequence cache set (`cam_intrinsics.pt`, `cam_extrinsics_cache.pt`,
`gt_joints_cache_world.pt`, `gt_joints_cache_cam_v2.pt`, `gt_joints_2d_cache.pt`,
`hand_bboxes_*.pt`). **Any new dataset must ship a preprocessor that emits this exact cache
set** — this is the real integration cost, not the download.

| Dataset | State in repo | Preprocessor | Notes |
|---|---|---|---|
| **HOI4D** | preprocessed (11 dense-depth seqs staged; expansion pipeline to ~525 rig-001 seqs exists) | `scripts/preprocessing/preprocess_hoi4d.py` | current train set; annotation root staged at `hoi4d_gt/` |
| **H2O** | preprocessed / eval-wired | `scripts/preprocessing/h2o_to_currentproto.py` | current 2nd eval set; store is SQUARE+CLAMP box convention (see memory: h2o-box-convention-audit) |
| **HOT3D** | **raw only** (team25/data), no cache built | none yet (needs one emitting the cache set) | egocentric, metric MANO; world-space eval already runs on it |

So three datasets are effectively in hand. Only HOT3D still needs a preprocessor. Every other
candidate below needs both a download and a new preprocessor.

---

## 1. Candidate datasets (metric 3D hand pose)

Legend for the key columns: **Metric?** = true metric 3D GT + calibrated camera (so absolute
C_abs is meaningful) vs weak-perspective/relative. **View** = Ego / Exo / Both. **MANO?** = ships
MANO params (vs 3D joints only). **DL** = download friction.

| # | Dataset | Metric 3D GT + intr/extr | View | Size (approx) | MANO? | HOI vs hands-only | License / download | DL friction |
|---|---|---|---|---|---|---|---|---|
| 1 | **HOI4D** | Yes — dense Kinect depth, per-frame SLAM extrinsics, 21-jt MANO (right hand) | Ego | ~4000 seqs full release; we use rig-001 subset | Yes (48+10+trans) | HOI (rigid objects) | non-commercial; RGB via HF `Livioni/hoi4d`, intrinsics/extrinsics via HF `yinloonga/HOI4D`, **handpose GT OneDrive/Baidu only** | have it |
| 2 | **H2O** | Yes — Azure Kinect RGB-D, intr+extr+GT cam poses, MANO both hands | **Ego + Exo** (5 cams: 4 static + 1 head) | 571,645 frames, 4 subj, 8 objects | Yes | HOI (bimanual) | register + EULA at h2odataset.ethz.ch, then `download_script.py --username --password` | have it |
| 3 | **HOT3D** | Yes — marker mocap GT, Aria/Quest3 calibration, MANO + UmeTrack | **Ego** (multi-view head) | 19 subj, 33 objects, ~1.5M multi-view frames; HOT3D-Clips = 3,832×150f | Yes | HOI (rigid) | HOT3D License (gated) on HF `projectaria/hot3d`, then scriptable via projectaria_tools | have raw |
| 4 | **AssemblyHands** | Yes — triangulated 3D joints (mm, ~4.2mm err), intr+extr; **no MANO (joints only)** | **Ego + Exo** | ~3.0M imgs (~490K ego); challenge split 384K/32K/62K | No | HOI (assembly, bimanual) | CC-BY-NC 4.0; **Google Drive (manual)** + toolkit | medium-manual |
| 5 | **ARCTIC** | Yes — 54-cam Vicon, MANO(2)+SMPL-X+articulated obj, full intr+extr | 8 Exo + **1 Ego** | ~2.1M frames, 10 subj, 11 articulated objects | Yes | HOI (articulated, bimanual) | MPI non-commercial; **register/EULA then scriptable** download scripts | medium |
| 6 | **DexYCB** | Yes — 8× Azure Kinect RGB-D, MANO + 3D joints, calibrated rig | **Exo only** | 582K frames, 1000 seqs, 10 subj, 20 YCB obj, ~119 GB | Yes | HOI (grasp, single-hand) | CC-BY-NC 4.0; **direct Google Drive** (~119 GB) | medium (big but scriptable) |
| 7 | **FreiHAND** | Yes — MANO + 3D joints in meters, per-sample K; **no extrinsics** (single-view) | Exo | 32,560 unique (130K aug) + 3,960 eval | Yes | Hands-only (green-screen) | research-only; **direct zips, scriptable, no gate** | low |
| 8 | **HanCo** | Yes — 8-cam intr+dist+extr, MANO fits + 21 kpts | Exo (8-cam multi-view) | 107,538 timesteps × 8 = 860K imgs, ~75 GB | Yes | Mostly hands-only | research-only; **direct links, scriptable** | low |
| 9 | **InterHand2.6M** | Yes — 3D joints world mm, MANO, intr+extr | Exo (studio) | ~2.6M @5fps (~80 GB) / ~12.5M @30fps | Yes | **Hands-only** (two-hand, no objects) | CC-BY-NC 4.0; packaged releases, scriptable-ish | low-medium |
| 10 | **ContactPose** | Yes — multi-Kinect intr+extr, MANO + 3D joints + contact | Exo | 2,306 grasps, 50 subj, 25 obj, >2.9M RGB-D | Yes (fit code) | HOI (functional grasp) | **MIT**; python download utils, scriptable | low |
| 11 | **OakInk** | Yes — 4-view intr+extr, MANO + 21×3 cam-space | Exo | ~230K frames, 792 clips, 12 subj | Yes | HOI (grasp/handover) | code MIT; **data Google-Form gated** | medium-manual |
| 12 | **OakInk2** | Yes — intr+extr, MANO + SMPL-X, obj 6DoF | **1 Ego + 3 Exo** | 627 seqs, ~4.01M frames, 75 obj (CVPR'24) | Yes | HOI (bimanual, long-horizon) | HF `kelvin34501/OakInk-v2` + `script/download.py`, **scriptable** | low-medium |
| 13 | **EgoDexter** | Partial — **3D fingertips only**, no full skeleton, no MANO; RGB-D intr+w2c | **Ego** | 4 seqs, 4 actors, 3,190 frames, ~2 GB | No | HOI (clutter) | non-commercial; **direct wget** | low (but tiny + fingertips-only) |
| 14 | **FPHA** | mm 3D joints from magnetic mocap (no MANO); RGB-D intr+extr | **Ego** | >100K frames, 45 actions, 6 subj | No | HOI + hands-only | academic; **registration form (manual)**; visible sensors on hand (appearance bias) | medium-manual |
| 15 | **Ego-Exo4D (hand pose)** | Metric 3D joints (mm), Aria SLAM + calib; **task withholds cam pose**, in-the-wild (auto-GT weak) | **Ego + Exo** | 740 participants, 1,286 h; hand GT: ~340K manual 3D + ~21M auto 3D | No (joints) | activity/HOI (in-the-wild) | Ego4D DLA; **access request + CLI**, then scriptable (`--benchmarks handpose`) | medium |
| 16 | **Ego4D** | **No metric 3D hand GT** (2D FHO boxes + contact only) | Ego | ~3,670 h | No | 2D only | Ego4D DLA + CLI | n/a — unusable for metric 3D |
| 17 | **GRAB** | Metric SMPL-X + MANO, **no real cameras** (mocap render only) | none (render your own) | 10 subj, 51 obj | Yes | Whole-body HOI grasp | MPI EULA (manual); SMPL-X/MANO separate | n/a for real-RGB metric eval |

Sources: HOT3D facebookresearch.github.io/hot3d + HF projectaria/hot3d + arXiv 2411.19167 ·
AssemblyHands assemblyhands.github.io + arXiv 2304.12301 · ARCTIC arctic.is.tue.mpg.de + arXiv
2204.13662 · DexYCB dex-ycb.github.io · Ego-Exo4D docs.ego-exo4d-data.org + arXiv 2311.18259 ·
FreiHAND lmb.informatik.uni-freiburg.de/projects/freihand + arXiv 1909.04349 · HanCo
lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html · InterHand2.6M
mks0601.github.io/InterHand2.6M · EgoDexter handtracker.mpi-inf.mpg.de · ContactPose
contactpose.cc.gatech.edu · FPHA guiggh.github.io/publications/first-person-hands · OakInk(2)
oakink.net + oakink.net/v2 + HF kelvin34501/OakInk-v2 · H2O taeinkwon.com/projects/h2o +
h2odataset.ethz.ch + arXiv 2104.11181 · GRAB grab.is.tue.mpg.de.

---

## 2. What WiLoR / HaWoR / HaMeR actually train on

This anchors "mixed 4-5 datasets" to what SOTA does. **HaMeR's 10-dataset mix is the base recipe;
WiLoR extends it; HaWoR reuses WiLoR's frozen backbone and trains temporal modules on a small
egocentric/HOI set.**

**HaMeR** (arXiv:2312.05251) — 10 datasets, ~2.7M examples:
FreiHAND, HO3D, MTC, RHD, InterHand2.6M, H2O3D, DexYCB, COCO-WholeBody, Halpe, MPII-NZSL.
(~5% in-the-wild.) Introduced **HInt** (Hands-in-the-Wild 2D-keypoint + occlusion annotations on
Hands23, EPIC-VISOR, Ego4D — 40.4K hands, mainly an eval benchmark).
URLs: arxiv.org/abs/2312.05251, github.com/geopavlakos/hamer.

**WiLoR** (arXiv:2409.12259) — HaMeR's 10 **plus** BEDLAM, ARCTIC, Re:InterHand, **HOT3D**, and its
own **WHIM** (~2M in-the-wild YouTube hand images, auto-annotated, released; models CC-BY-NC-ND).
Paper says "thirteen sources" — the enumerated names + WHIM exceed 13, so treat the exact count as
uncertain. URLs: arxiv.org/abs/2409.12259, github.com/rolpotamias/WiLoR.

**HaWoR** (arXiv:2501.02973) — freezes WiLoR's ViT backbone, trains temporal/motion modules on a
**small egocentric + HOI set**: **HOT3D (~573K), ARCTIC (~165K), DexYCB (~169K), HO3D (~66K)** ≈ 1M
frames. Eval on HOT3D (held-out), DexYCB, EPIC-KITCHENS. URLs: arxiv.org/abs/2501.02973,
github.com/ThunderVVV/HaWoR.

**Takeaway for us.** The intersection of "used by SOTA" ∩ "metric + egocentric-relevant + we can
plausibly get" is exactly **HOT3D, ARCTIC, DexYCB** (all in WiLoR *and* HaWoR), plus our own HOI4D
and H2O. That is the natural mixed set, and it makes our training regime directly comparable to
HaWoR's.

---

## 3. Compatibility with our absolute-metric (C_abs) protocol

Our loss is `kp3d_abs` (absolute 3D-joint MPJPE in the camera frame). Requirements: **metric 3D
joint GT + camera intrinsics** (extrinsics only needed for world-space, not for C_abs). MANO is
convenient but not mandatory — joint-only GT (AssemblyHands, FPHA, Ego-Exo4D) can still supervise
`kp3d_abs`; it just can't supervise MANO pose/shape.

**TRUE metric, absolute-C_abs-valid (use freely):**
HOI4D, H2O, HOT3D, DexYCB, ARCTIC, AssemblyHands, HanCo, InterHand2.6M, ContactPose, OakInk,
OakInk2, FreiHAND (single-view, K present), EgoDexter (fingertips only), FPHA (magnetic-mocap
joints), Ego-Exo4D (manual-GT portion).

**Needs a known focal / intrinsics fed in (all of the above ship K):** every metric row provides
intrinsics; none are weak-perspective. The weak-perspective failure mode is on the *baseline* side
(WiLoR/HaMeR dummy focal=5000), not the training data.

**Not usable for C_abs:** Ego4D (no 3D hand GT), GRAB (no real cameras — mocap render only).

**Caveats that bite our protocol:**
- AssemblyHands / FPHA / Ego-Exo4D / EgoDexter give **joints, not MANO** → supervise `kp3d_abs`
  only; the MANO-param branch is unsupervised on those samples (mask it).
- FPHA has **magnetic sensors + tape physically visible** on the hand in RGB → appearance bias.
- Ego-Exo4D's 21M *automatic* 3D annotations are model-generated (weak GT); only the ~340K manual
  ones are lab-grade — reserve it for **testing**, not training-on-auto-GT.
- H2O / HOI4D box conventions differ (square+clamp vs rectangular) — already logged (memory:
  h2o-box-convention-audit); harmonize per-dataset in the preprocessor.

**Egocentric (our target regime):** HOI4D, H2O (ego cam4), HOT3D, AssemblyHands (ego rig), ARCTIC
(1 ego view), OakInk2 (1 ego view), EgoDexter, FPHA, Ego-Exo4D. Third-person only: DexYCB,
FreiHAND, HanCo, InterHand2.6M, ContactPose, OakInk.

---

## 4. Recommendation

### 4a. Training mix (~5 datasets) — favor egocentric + metric + gettable

1. **HOI4D** — have it; egocentric, dense depth. Keep as anchor.
2. **H2O** — have it (preprocessed); egocentric + exo, metric MANO both hands.
3. **HOT3D** — have raw; egocentric mocap-metric MANO; **trained on by both WiLoR and HaWoR** →
   direct comparability. Only needs a preprocessor.
4. **AssemblyHands** — egocentric (+exo) triangulated metric 3D, big + bimanual assembly diversity.
   Joint-only (mask MANO branch). Adds real egocentric domain shift.
5. **ARCTIC** — 1 ego view + bimanual articulated-object HOI, full MANO; **trained on by both WiLoR
   and HaWoR**. Register-then-scriptable.

Swap option: replace ARCTIC or AssemblyHands with **DexYCB** if download friction on the gated sets
is too high — DexYCB is trivially scriptable (direct Google Drive, ~119 GB), metric MANO, and used
by all three SOTA methods; the cost is it is **third-person only** (weakens the "egocentric" story).

Rationale: 4 of the 5 are egocentric or have an ego view; all 5 are true-metric; three of them
(HOI4D, H2O, HOT3D) need zero new downloads; and HOT3D+ARCTIC put us on the same training data as
HaWoR/WiLoR, so "we mix like SOTA" is defensible.

### 4b. Held-out test dataset (NOT in the mix, recent, challenging)

**Primary: Ego-Exo4D hand-pose benchmark (2024).** Best "extremely new / true generalization"
story: in-the-wild egocentric Aria, 123 scene contexts, 740 participants, metric 3D GT, and a
standardized benchmark task that *withholds camera pose* — the hardest realistic generalization
test, and none of the training mix overlaps it. Access = Ego4D DLA + CLI (`--benchmarks handpose`),
then scriptable. Use the **manual-GT** split for reporting (treat auto-GT as weak).

**Easier-access alternative: OakInk2 (CVPR 2024).** Recent, has a real egocentric view, ships
MANO+SMPL-X, lab-grade GT, and is **fully scriptable from HuggingFace** — lower friction than
Ego-Exo4D if the DLA turnaround is slow. Downside: less "in-the-wild" than Ego-Exo4D, so a weaker
generalization claim.

Do **not** hold out a dataset that any training member overlaps (e.g. holding out DexYCB while
training on HO3D/ARCTIC shares objects/subjects) — Ego-Exo4D and OakInk2 are cleanly disjoint from
the proposed mix.

### 4c. Small scaling-up first step (prove the mixing pipeline before the full 5)

**Step 0 (zero download): HOI4D + H2O + HOT3D.** All three are already on the cluster (HOI4D & H2O
preprocessed; HOT3D raw). The only new code is a HOT3D preprocessor emitting the `HOT3DHandDataset`
cache set (the eval path already reads HOT3D, so the loader contract is known). This proves the
multi-dataset **sampler/mixing** machinery (per-dataset box conventions, hand-validity masks,
right-hand-only vs both-hands, intrinsics plumbing) on data we already trust, with no download or
licensing wait.

**Step 1 (one new download): add AssemblyHands** (egocentric, joint-only → exercises the
MANO-branch-masking path) *or* **DexYCB** (trivial scriptable download → exercises a third-person
domain). Pick DexYCB first if you want the mixing proven fastest; AssemblyHands if you want to stay
egocentric.

**Step 2 (full mix): add ARCTIC** (register/EULA) to reach the 5-set regime and match HaWoR/WiLoR
training data.

---

## 5. Download / access notes for the top picks

Ranked by download+licensing friction (lowest first). "Scriptable" = runnable on a cluster scratch
dir without a browser.

| Rank | Dataset | Registration / EULA | Approx size | Scriptable? | How |
|---|---|---|---|---|---|
| — | HOI4D | none for RGB/calib; handpose GT is OneDrive/Baidu (manual) | ~14 GB subset staged | partial | HF `Livioni/hoi4d` (RGB), HF `yinloonga/HOI4D` (`camera_params.zip`, annotations via `remotezip` range-extract); GT already local |
| — | H2O | **register + EULA** (h2odataset.ethz.ch) | full set (GB unconfirmed) | yes after login | `download_script.py --username --password --mode {all,ego,pose}` |
| 1 | **HOT3D** | **accept HOT3D License** (gated HF) | large (GB unconfirmed); we have raw | yes | HF `projectaria/hot3d` + `projectaria_tools`; raw already on cluster → just build cache |
| 2 | DexYCB | none reported (CC-BY-NC) | ~119 GB (or 13 sub-archives) | yes | direct Google Drive links from dex-ycb.github.io (use `gdown`) |
| 3 | FreiHAND | none | few GB | yes | direct zips from lmb.informatik.uni-freiburg.de/projects/freihand (`wget`) |
| 3 | HanCo | none | ~75 GB | yes | direct links from the HanCo page (`wget`) |
| 3 | ContactPose | none (MIT) | GB unconfirmed | yes | python download utils, github.com/facebookresearch/ContactPose |
| 4 | OakInk2 | EULA not surfaced (preview) | 627 seqs / ~4M frames | yes | HF `kelvin34501/OakInk-v2` + `script/download.py` (needs MANO v1.2 + SMPL-X v1.1) |
| 5 | ARCTIC | **MPI account + EULA** | hundreds of GB (cropped variant offered) | yes after registration | account at arctic.is.tue.mpg.de → provided download scripts |
| 6 | Ego-Exo4D | **Ego4D DLA + access request** | hand subset (multi-GB) | yes after approval | official `egoexo` CLI, `--benchmarks handpose` |
| 7 | AssemblyHands | CC-BY-NC (no form) but **Google Drive manual** | ~490K ego imgs | manual | Google Drive links from assemblyhands.github.io + toolkit |
| 8 | OakInk (v1) | **Google Form gated** | ~230K frames | no | annotations zip via Google Form |
| 8 | FPHA | **registration form** | >100K frames | no | form at guiggh.github.io |
| 8 | GRAB | **MPI EULA** | — | no (and no real RGB) | grab.is.tue.mpg.de |

Cluster caveat (from project memory): `/work` scratch quota has been hit repeatedly and login-node
reapers kill long downloads — stage large pulls (DexYCB ~119 GB, HanCo ~75 GB, ARCTIC) via SLURM
jobs to node-local `/tmp`, or stream-and-delete, rather than downloading on the login node.

---

## Bottom line

- **Train (5):** HOI4D + H2O + HOT3D (all already on the cluster) + AssemblyHands + ARCTIC.
  Swap DexYCB in for AssemblyHands/ARCTIC if gated-download turnaround is a blocker.
- **Held out (1):** Ego-Exo4D hand-pose benchmark (best true-generalization story); OakInk2 as the
  lower-friction fallback.
- **First step:** prove the mixing sampler on HOI4D+H2O+HOT3D (zero new download; only a HOT3D
  preprocessor), then add one easy dataset (DexYCB fastest, or AssemblyHands to stay egocentric),
  then ARCTIC for the full 5.
- **Integration cost is the preprocessor, not the download:** every new dataset must emit the
  `HOT3DHandDataset` cache set; joint-only datasets (AssemblyHands, Ego-Exo4D) need the MANO
  branch masked and `kp3d_abs`-only supervision.
