# FF-4DGS-Ego — Research Synthesis & Weekend Plan (2026-06-19)

> **STALE THESIS WARNING (added 2026-07).** This is a dated session snapshot, kept as a
> historical record. Its framing ("hand as metric anchor for a feedforward Gaussian scene",
> "first feedforward full-scene 3DGS made metric by the hand") was experimentally falsified
> in 2026-07 (scale-source ablation: hand-as-global-scene-scale 0.728 vs oracle 1.022; the
> 4DGS backbone is frozen third-party, Gaussian rendering is off). The current thesis is
> absolute camera-frame egocentric hand pose from a frozen feedforward recon backbone; see
> `report/related-work-positioning.md` and `report/publication-plan.md`. The literature
> pointers below are still useful; do not reuse the positioning or the plan.

**Status:** session knowledge dump (session hit limit; compaction failed). This file is the
single source of truth for resuming. Everything below was produced 2026-06-19 from a stride-4
experiment + 3 web-research agents + a full codebase-feasibility dive + a HOI4D-eval feasibility
investigation.

---

## TL;DR — what to do next (ranked)

1. **Hand3R HOI4D world-space head-to-head** (the paper's money table). Pipeline already mostly
   exists. Reproduce Hand3R's HOI4D protocol; show our hand-anchor scale reduces long-window
   drift below their **W-MPJPE 125.8 mm (long)**. *Tier A, no training, needs cluster GPU.*
2. **Render-and-PnP / photometric per-clip pose refinement** in `eval_world_space.py` — attacks the
   proven bottleneck (per-clip camera pose; oracle GT-cameras cut W-MPJPE 188→115 mm). We HAVE a
   differentiable gsplat renderer (RGB+ED). *Tier A, no training, but needs Gaussian-exposure work.*
3. **Re-run the 3 background research agents** (CVPR/ICCV/3DV/ECCV/NeurIPS 2025-26, additive) — they
   died when the process exited. Prompts preserved at the bottom of this doc.
4. **Tier B (needs `/work` freed):** unfreeze camera head + hi-res hand crops → world-space eval.

---

## Part 1 — Experimental state

### Stride-4 result (task beh4xr0gh, completed)
Larger overlap (stride 8→4) gives at most a marginal change; **global ≡ greedy confirmed a 3rd time.**
```
[P0001_23fa0ee8 seg0] greedy W=130.5 WA(s/l)=43.0/93.9 | global W=130.7 WA(s/l)=43.1/94.0 (128f)
[P0001_4bf4e21a seg0] greedy W=nan  WA(s/l)=63.0/140.8 | global W=nan  WA(s/l)=63.4/141.3 (128f)
[P0001_550ea2ac seg0] greedy W=67.7 WA(s/l)=33.5/56.1  | global W=67.7 WA(s/l)=33.6/56.2  (128f)
OURS greedy W-MPJPE=99.1 WA(s/l)=38.2/75.0 ; global 99.2/38.3/75.1 (n=2 valid)
```
(n=2 not comparable to the n=9 headline — one seq went W=nan from a degenerate first window. The
*per-seq global=greedy equality* is the only thing this run needed to show, and it did.)

### Drift levers — ALL exhausted, conclusively null/negative
- Global (BA-style) chaining ≡ greedy (160.9=160.9 across 6 seqs; 130.5≈130.7 stride-4).
- Robust/Huber Umeyama: no gain.
- Intrinsics conditioning (`cond_flags=[0,0,1]`): WORSE (160.9→172.9, OOD).
- Stride 8→4 (larger overlap): marginal.
- **TALO (CVPR'26, arXiv:2512.02341) formally proves a single Sim(3) can't repair bad per-clip poses
  → independently confirms our null.** The fix is per-clip pose, or higher-DOF global deformation.

### Oracle-camera diagnostic = the whole story
Replacing predicted cameras with GT extrinsics cuts **W-MPJPE 188 → 115 mm**. So the bottleneck is the
**frozen backbone's per-clip CAMERA POSE** (not chaining, not the hand head). Our oracle ceiling
(11.5 cm) ≈ HaWoR's 11.3 cm → **if we fix per-clip pose we reach SLAM-parity placement, SLAM-free.**

### Table 5 (report/scale-table.tex, line 79) — complete, compiles to 1 page
```
Ours (FF-4DGS-Ego, SLAM-free)  &  ---  &  41.6 / 94.0  &  HOT3D     (W-MPJPE=188mm in note, n=9)
```
**New 2026 competitors to ADD (found this session):**

| Method | HOT3D W-MPJPE | SLAM-free? | note |
|---|---|---|---|
| WHOLE (2602.22209) | **10.4 cm** (WA 0.58 cm, first-frame aligned) | ✗ needs Aria metric SLAM | new SOTA / our SLAM upper-bound |
| HaWoR (already in table) | 11.3 cm | ✗ | |
| **EgoForce (2605.12498)** | 43.9 mm *(camera-space MJE)* | **✓** | the SLAM-free bar to cite |
| Ours | 18.8 cm | ✓ | |
| (oracle cameras) | **11.5 cm** | ✓ | our ceiling = HaWoR-parity |

Honest framing: ~1.7× worse than SLAM on absolute placement, **but SLAM-free, and our ceiling is
HaWoR-parity.** The ablation findings (global≡greedy, intrinsics-worse, stride-4-marginal) are
text/appendix, not headline.

---

## Part 2 — Strategic research synthesis (3 web agents)

### Finding 1 — We are the *contrarian* on metric scale (defensible, must be proven)
The whole 2025-26 field (HaWoR, Dyn-HaMR, EgoGrasp, WHOLE) gets scale from a **metric-depth network
with the hand region MASKED OUT**, because near-field hand depth is where depth nets fail. We do the
opposite — the hand *is* the anchor. The same near-field unreliability is our **argument** (known-size
hand sidesteps monocular scale ambiguity) AND a risk (noisy anchor). **EgoForce (2026, arXiv:2605.12498)
independently validates "known hand/forearm geometry as metric anchor", SLAM-free** — strong support.
→ The HOI4D "mask the hand" ablation is now a **core thesis test**, not reviewer-defense:
hand-anchor vs masked-scene-depth vs fused. If masking the hand *degrades* metric recovery → direct
evidence the hand drives the coupling.

### Finding 2 — Closest competitor IS our template, and it admits our opening
**Hand3R (arXiv:2602.03200, Feb 2026)** — first online, feedforward, SLAM-free joint 4D hand+scene.
Frozen HaMeR + frozen CUT3R scene model, fused via "scene-aware visual prompting." **Metric scale
from the scene model (CUT3R), NOT the hand.** Reports HOI4D (we have it):
C-MPJPE 42.6, **W-MPJPE 86.9 short / 125.8 long**, WA-MPJPE 38.0/56.7 — and explicitly concedes
long-sequence drift ("accumulated drift is inevitable"). Local DexYCB PA-MPJPE 4.83 mm.
→ **THE experiment: reproduce their HOI4D protocol, beat their 125.8 long-drift with the hand anchor.**

### Finding 3 — Novelty verdict: novel *combination*, not novel *primitive* → MOVE FAST
"Human/body as metric ruler" is **crowded on the body side** (2024-26, trending up):
- **UniSH** (CVPR'26, 2601.01222) — closest architecture: one feedforward pass → metric scene
  pointmaps + camera + metric SMPL via an AlignNet scale predictor. Differs: SMPL body (not MANO
  hand), pointmaps (not 3DGS), third-person (not ego). **Must cite as direct body analog.**
- **HAC "Humans as Checkerboards"** (ICCV'25, 2407.00574) — metric HMR body + contact-joint depths
  calibrate monocular SLAM scale. Closest *concept*, motion-only not GS.
- **MfH "Metric from Human"** (NeurIPS'24, OpenReview GA8TVtxudf) — HMR body dims as scene-independent
  metric scale prior for zero-shot metric *depth*. Strongest published *principle* precedent.
- SynCHMR (CVPR'24, 2405.14855), MetricHMSR (CVPR'26, 2506.09919).

The exact intersection — **feedforward + full-scene 3DGS + MANO hand + egocentric + no-IMU** — is
**currently UNCLAIMED**, but UniSH already landed the *body* version (Jan'26), so the hand-GS-ego
variant is the obvious, likely-contested next step. **Defensible claim:** *first feedforward,
marker-free, IMU-free, full-scene 3DGS made metric by anchoring to the in-scene MANO hand in
egocentric HOI video.* Frame around the intersection, not "human as ruler." Move fast.

### Backbone confirmed
**WorldMirror** (Tencent Hunyuan, arXiv:2510.10726, ICML'26) — feedforward, emits Gaussians, accepts
depth/intrinsics/pose as **optional priors**, **NOT natively metric**. The intended hook to make it
metric is feeding a metric prior → **our in-scene MANO hand is exactly that prior.** VGGT lineage is
normalized/not-metric. Natively-metric peers (CUT3R, MoGe-2, MapAnything, AMB3R) get scale from a
learned monocular prior — the signal that's unreliable in ego hand-object scenes.

---

## Part 3 — The bottleneck and the levers (ranked by impact × single-GPU feasibility)

1. **GS-CPR-style render-and-PnP per-clip pose refinement (arXiv:2408.11085) — DO FIRST.**
   We already produce Gaussians+depth+RGB. Per frame: render at predicted pose → match real↔render →
   lift via rendered depth → PnP+RANSAC, warm-started from feedforward pose. Gradient-free, no backprop
   through frozen backbone, ~<180 ms/frame. **Caveat (our setting):** Gaussians are predicted jointly
   with poses per clip → per-clip self-consistency means the cleanest leverage is *cross-clip*
   (refine clip i+1 vs clip i's map = drift reduction) or sharpening the feedforward pose toward its
   own photometric optimum. Add AnyCam-style (2503.23282) confidence-weighting to downweight moving hands.
2. **TALO drop-in global-consistency layer (arXiv:2512.02341)** — training-free, replaces Sim(3)
   Umeyama with TPS higher-DOF + globally-propagated control points. Beats VGGT-Long/VGGT-SLAM on ATE.
   Our global=greedy null says we need *higher-DOF*, not just joint Sim(3). Weekend-portable.
3. **LoRA partial-tune toward window-relative pose (VGGT-HPE recipe 2604.10106 + Anchor3R framing
   2606.05035)** — LoRA r=8/alpha=16 on qkv+MLP of last ~6-8 blocks + trainable pose/scale head,
   single GPU ~1 day, EMA(0.999) + WiSE-FT(α≈0.5) to protect priors. *Needs cluster training.*

**Free A/B worth trying:** swap frozen VGGT backbone for **π³/Pi3 (arXiv:2507.13347)** — best published
per-clip pose (halves VGGT's Sintel ATE), permutation-equivariant (kills 16-frame ordering bias), no
retraining for an initial test.

**Test-time refinement options:** GS-CPR (2408.11085), iComMa (2312.09031), AnyCam (2503.23282),
PROFusion (2509.24236, blur-robust gradient-free), Diff3R (2604.01030), JOGS (2510.26117).
Always warm-start from feedforward pose + add a matching/PnP term (3R-GS caveat 2504.04294).

**NOT feasible for us (flagged):** MASt3R-SfM global BA (~27 min/200 img), AMB3R backend (~80 H100-hr),
VGGT-SLAM (under-constrained SL(4), diverged >60%), full backbone fine-tune (regresses metric accuracy).

---

## Part 4 — Codebase feasibility map (what we can actually change)

### Camera pose
`diffsynth/auxiliary_models/worldmirror/models/heads/camera_head.py` — **9-vec**: t[0:3], quat_wxyz[3:7],
FOV[7:9] (relu). Iterative refine `steps=4`. In `worldmirror.py`: `cam_seq=self.cam_head(token_list)`
(L322) → `cam_params=cam_seq[-1]` → `transform_camera_vector` → `preds["camera_poses"]` (c2w, L325-326).
**CameraHead is separately addressable** (register token only) → can unfreeze ONLY it.

### Gaussian renderer with depth — render-and-PnP IS implementable
`diffsynth/auxiliary_models/worldmirror/models/models/rasterization.py`:
`rasterize_splats(splats, viewmats, Ks, width, height)` → gsplat `rasterization(..., render_mode="RGB+ED")`
returns `(colors[H,W,3], depth[H,W,1], alphas)`. **Differentiable wrt viewmats (pose).** `render()`
returns `preds["rendered_extrinsics"]`, `preds["rendered_depths"]`, `preds["gs_depth"]`.
⚠️ viewmats are detached (L295). For render-and-PnP, need the raw Gaussian params (means/quats/scales/
opacities/colors) exposed in inference preds — **not currently exposed; gs_head produces gs_feat+gs_depth,
splats built inside render(). This is the one piece of plumbing to add.**

### Clip chaining (`scripts/world_space_metrics.py`)
- `chain_trajectories_by_overlap(clip_worlds, overlap)` — greedy per-seam Sim(3) (7-DOF), seam-avg.
- `chain_trajectories_global(clip_worlds, overlap, iters=8, robust=True)` — BA-style alternating
  consensus, clip0 pinned (gauge). Swap is plug-and-play. Robust IRLS Huber available.
- `solve_similarity` (weighted Umeyama), `solve_similarity_robust`, `w_mpjpe`, `wa_mpjpe`.

### Heads (wired, OFF by default)
- `scale_head.py` — `ScaleHead(dim_in=2*dim)`: register-token MLP → `exp(log_s).clamp(0.1,10)`. Enable
  `enable_scale_head:true`. Supervised by `scale_head_loss.py` (smooth-L1 `s*sampled_depth - metric_hand_depth`).
  **COMPLETE — just flip the flag.**
- `hires_hand_encoder.py` — `HiResHandEncoder` (ResNet18/50 trunk, 256px crop → tokens). Wired in
  `hamer_head.py` behind `model.hires_hand:true`; 224px path byte-for-byte unchanged when off.

### Unfreeze (`scripts/train_hand_head.py` ~L1280-1320)
`unfreeze_last_n_blocks: N` → unfreezes last N `frame_blocks` + last N `global_blocks`; sets
`_backbone_trainable=True`. `freeze_backbone:false` trains everything. **No LoRA/adapter** — discrete
block on/off only. Losses: kp3d / kp3d_abs (cam/world joint MPJPE), hand_depth_anchor, obj_depth,
scale_head, gs_l1/gs_lpips.

### Hand head → world joints
`hamer_head.py`: per-hand [pos3, quat4, pose15(PCA), betas10] = 32; [N,2,32]→[N,64].
`compute_joints_from_batch(params[B,S,64], mano, device)` → [B,S,2,16,3] camera-frame metres.
World lift in `eval_world_space.py:lift_clip_to_world` (L51-72): `world[k]=R@(s*pj[k])+(s*t)` with
`s=solve_metric_scale(...)`. Hand crop fed to head = 8×8 patches of the 224 feature map (the #26 hi-res concern).

### Feasibility verdict
| lever | difficulty | status |
|---|---|---|
| render-and-PnP pose refinement | MEDIUM | implementable; need to expose Gaussians in preds |
| higher-DOF global chaining | EASY | plug-and-play swap |
| learned scale head supervised by hand | EASY | complete, gated by flag |
| LoRA/partial-unfreeze of camera head | MEDIUM | block on/off yes; no LoRA |

---

## Part 5 — HOI4D world-space eval (Hand3R reproduction) — THE money experiment

### ⛔ DATA BLOCKER found 2026-06-20 (probed cluster)
The downloaded HOI4D is a **minimal subset**: flat `<seq>/images/*.png` (300/seq) + `raw_depth/*.png`,
NOT the canonical release tree `preprocess_hoi4d.py` expects (`align_rgb/image.mp4`, `3Dseg/output.log`).
- **18 image seqs downloaded; 11 have matching handpose GT** at
  `hoi4d_handpose_gt/Hand_pose/handpose_right_hand/<ZY.../H1/C*/N*/S*/s*/T*>/<frame>.pickle`
  (234–300 pickles/seq; keys `poseCoeff[48]=3 global aa+45 pose, beta[10], trans[3] cam-frame m, kps2D`).
  Flat dir `ZY..._H1_C11_N07_S185_s02_T2` ↔ handpose `ZY.../H1/C11/N07/S185/s02/T2` (`tr '_' '/'`).
- Intrinsics present: `camera_params/camera_params/<ZY...>/intrin.npy` (double-nested; 4 cameras).
- **NO camera extrinsics anywhere** (no `3Dseg/output.log`, no pose files). HOI4D `trans` is camera-frame.

**Consequence — splits the HOI4D experiments:**
- ✅ **Camera-frame C-MPJPE** (Hand3R 42.6 / HaWoR 51.77 head-to-head): VIABLE NOW (handpose + intrinsics,
  no extrinsics). This is task #18's goal. Adapt `eval_cmpjpe.py` (H2O-only) for HOI4D pickles.
- ✅ **Scale-source ablation** (hand-anchor vs masked dense `raw_depth` vs fused, camera-frame metric depth
  at the hand): VIABLE NOW (`hoi4d_depth_dataset.py` loads raw_depth; no extrinsics).
- ⛔ **World-space W/WA-MPJPE** ("beat 125.8 long-drift"): BLOCKED — needs per-frame camera poses to lift
  GT cam-frame hands to world. Options: (a) download HOI4D camera poses / 3Dseg (needs space; `/work` full),
  (b) keep the world-space story on HOT3D (already works, GT extrinsics present), (c) get poses for a few
  seqs to node-local /tmp inside one job + preprocess + eval in-job. `preprocess_hoi4d.py` also needs
  adapting: images/ (not image.mp4), handpose path differs, camera_params double-nested.

**Decision pending from user:** invest in HOI4D pose download to unblock world-space, vs do the unblocked
C-MPJPE + scale-ablation now and keep world-space on HOT3D.

### Original plan (still valid once poses exist)
**Good news: the pipeline mostly exists.** `scripts/preprocessing/preprocess_hoi4d.py` emits the EXACT
cache set `HOT3DHandDataset` loads on the cache-HIT path (so it never hits HOT3D-specific code), and
`HOT3DHandDataset.discover_sequences` auto-discovers HOI4D seqs (`video_main_rgb.mp4` +
`hand_data/mano_hand_pose_trajectory.jsonl`).

### What preprocess_hoi4d.py produces (per seq, under `<out>/<seq>/hand_data/`)
`cam_intrinsics.pt [3]`, `cam_extrinsics_cache.pt [N,4,4]` T_camera_world, `gt_joints_cache_world.pt
[N,2,16,3]`, `gt_joints_cache_cam_v2.pt [N,2,16,3]`, `gt_joints_2d_cache.pt`, `hand_bboxes_v2_rf1.5_resHxW.pt`
({bboxes, valid, gt64}). **Right hand only → index RH=1; left hand stays invalid (valid mask handles it).**
HOI4D GT = `handpose/refinehandpose_right/<seq>/<frame>.pickle` {poseCoeff[48]=3 global aa + 45 full pose,
beta[10], trans(camera frame, m)}; intrinsics `camera_params/<cam>/intrin.npy`; extrinsics
`3Dseg/output.log` (Open3D per-frame 4×4); RGB `align_rgb/image.mp4` (1920×1080@15fps), depth `raw_depth/*.png`
(16-bit mm).

### ⚠️ THE ONE REAL RISK: extrinsics direction (output.log)
`load_extrinsics` ASSUMES the parsed matrix is `T_camera_world` (world→cam). Open3D/Redwood `.log`
convention is ambiguous (loaders disagree). **De-risk before scaling:** pick 1 seq with depth+hand,
back-project a depth pixel → 3D cam point → transform by extrinsic (as-is, then inverted) → compare to
handpose GT `trans`; whichever makes the hand visible / lands in the scene cloud is correct. Set the
`invert` flag accordingly. If wrong, ALL world-space GT caches are flipped. Also VERIFY (2) the 45-full→
15-PCA pose conversion (flat_hand_mean) and (3) cam→world lift, by overlaying `gt_joints_2d` on the RGB.

### HOI4D data on cluster
`/work/scratch/dmonopoli/hoi4d` (14G): ~dozens of `ZY20210800001_H1_C*_N*_S*_s*_T*` seqs, each with
`images/` + `raw_depth/`. `camera_params.zip`, `seq1.tar.gz` (1.1G, re-downloadable).
`/work/scratch/dmonopoli/hoi4d_handpose_gt/Hand_pose/handpose_right_hand/ZY.../...` = GT.
**NO `gt_joints_cache_world.pt` built yet** — preprocessing has not been run on these seqs.

### Build plan (minimal)
1. Run `preprocess_hoi4d.py` on 1 seq (`--limit 50`) → verify caches + 2D overlay + extrinsics direction.
2. Adapt `eval_world_space.py` for HOI4D (mostly: point `--data_root` at preprocessed HOI4D; right-hand-only
   means half the 32 joints are always invalid — confirm `gt_valid`/metrics handle it; HOI4D `cam_intrinsics`
   is `[focal,cx,cy]` single-focal square — matches `_intr_3x3`).
3. Batch-preprocess available HOI4D seqs (to node-local `/tmp` inside a streaming srun; `/work` is full).
4. Run W/WA-MPJPE + C-MPJPE; compare to Hand3R (C-MPJPE 42.6, W 86.9/125.8, WA 38.0/56.7).
   Also build the **scale-source ablation**: hand-anchor vs masked-scene-depth (UniDepth-V2/Metric3D-v2)
   vs fused. HOI4D has dense `raw_depth` → `scripts/hoi4d_depth_dataset.py:HOI4DDepthDataset` already loads it.

### Reusable pieces
`preprocess_hoi4d.py` (full I/O), `HOT3DHandDataset.discover_sequences`, `hand_metrics.py`
(`joints_and_vertices_from_params`), `world_space_metrics.py` (all metrics), `eval_world_space.py`
(HOT3D driver to adapt), `hoi4d_depth_dataset.py` (dense depth). C-MPJPE driver: adapt `eval_cmpjpe.py`
(currently H2O-only) for HOI4D's `gt_joints_cache_cam_v2.pt`.

---

## Part 6 — Cluster constraints & infra (critical, preserve)

- **gb10 Grace-Blackwell aarch64**, 114 GB GPU. Login `student-cluster1 ≡ studgpu-spark01` (shares /tmp).
  `spark02` separate box (own /tmp). **QOS = 1 job/user serial.** Partitions: `jobs` (7-day), `interactive`
  (2h). Streaming `srun` capped ~30 min (expect timeout 1800s). gsplat JIT ~70s first run, cached `/tmp/xc`.
- **`/work` per-user quota EXHAUSTED** (hard cap). Filesystem 13T/16T (77%) but USER quota hit — can't even
  write a KB SLURM log (`joblogs: Disk quota exceeded`). **→ `sbatch` BLOCKED (no log path); no persisted
  checkpoints.** Cache cleanup (pip_cache/xdg_cache/wandb ~150-250 MB) did NOT relieve the quota.
  Reclaimable big items: `hoi4d/seq1.tar.gz` (1.1G, re-downloadable). **Classifier blocks me deleting
  experiment checkpoints + mass `scancel -u` → give the user `!`-prefixed commands.**
- **`/home` block+inode exhausted** (can't create/edit files there). Only node-local **`/tmp` (860G)** writable.
- **Active venv** `/work/scratch/dmonopoli/venv_gb10` (aarch64 — NEVER import/pip torch on x86 login node;
  use `py_compile` or a gb10 srun).
- **Existing checkpoints:** `/work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt` (4.7G, NEEDED),
  `.../hoi4d_depth/best_depth.pt` (4.9G), `models/NeoVerse/reconstructor.ckpt` (6G, just re-downloaded).
  No hires/unfreeze checkpoint exists.
- **SSH/secrets:** cluster password stored in `/tmp/.ethpw` (chmod 600) — value REDACTED (was committed in 2595eaf; rotate); helpers
  `expect /tmp/clssh.exp '<cmd>'`, push `expect /tmp/clscp.exp <local> <remote>`, pull
  `expect /tmp/clpull.exp <remote> <local>`. Restore pw:
  `printf '%s' '<CLUSTER_PASSWORD>' > /tmp/.ethpw && chmod 600 /tmp/.ethpw`.
  **Plain `ssh eth-cluster` from the Mac shell does NOT auth (no key/askpass) — use the expect helpers.**
  H2O creds streamed from env (`$H2O_USER:$H2O_PASS`), REDACTED, don't persist. H100/A100 DENIED by Cyrus.
- **Self-staging launcher pattern (works for >stage files into node-local /tmp):** base64 a self-contained
  bash script that `cp`s `$SRC/scripts` into `/tmp/run`, symlinks diffsynth/models/configs, runs the eval,
  prints results to stdout (which streams back over ssh). Used for the stride-4 run. The base64 blob bloats
  the task output — filter with `grep -avE '[A-Za-z0-9+/=]{120,}'`.

---

## Part 7 — Weekend plan (today = Fri 2026-06-19; Cyrus reviews ~Mon)

**Tier A — no training, streaming srun (30 min), attacks the proven bottleneck:**
- A1. **Hand3R HOI4D world-space head-to-head** + scale-source ablation (Part 5). Highest value, mostly
  built. Beat their 125.8 long-drift. Watch the extrinsics-direction risk.
- A2. **Render-and-PnP per-clip pose refinement** in `eval_world_space.py` → re-run HOT3D world-space.
  Target W-MPJPE 18.8 → toward the 11.5 cm oracle ceiling. Needs Gaussian-exposure plumbing first.

**Tier B — needs `/work` freed + sbatch:**
- B1. Unfreeze camera head (`unfreeze_last_n_blocks`; camera head separately unfreezable) → world eval.
- B2. Hi-res hand crops (`hires_hand:true`, 256px, wired) → world eval.

To unlock Tier B: free ~a few GB (re-downloadable HOI4D tarballs). Inspect first:
`! ssh eth-cluster 'du -sh /work/scratch/dmonopoli/hoi4d/*; ls -la /work/scratch/dmonopoli/hoi4d/*.tar.gz'`
(then delete extracted-and-redundant tarballs). NOTE: this requires the expect helper, not plain ssh.

**Recommendation:** start A1 (Hand3R HOI4D) — it's the paper-defining result and mostly built. A2 second.

---

## Part 8 — Key papers (round 1, ~45) + pending round 2

### Round-1 paper set (cite/skip-in-future-searches)
VGGT, Pi3/π³ (2507.13347), CUT3R (2501.12387), MV-DUSt3R+ (2412.06974), Spann3R (2408.16061),
MASt3R-SfM (2409.19152), Fast3R (2501.13928), Light3R (2501.14914), StreamVGGT (2507.11539),
TALO (2512.02341), TTT3R (2509.26645), VGGT-Long (2507.16443), Anchor3R (2606.05035), LoGeR (2603.03269),
GS-CPR (2408.11085), iComMa (2312.09031), AnyCam (2503.23282), PROFusion (2509.24236), Diff3R (2604.01030),
JOGS (2510.26117), VGGT-HPE (2604.10106), 3R-GS (2504.04294), Surgical-FT (2210.11466), EndoDAC (2405.08672),
WiSE-FT (2109.01903), HaWoR (2501.02973), Dyn-HaMR (2412.12861), Hand3R (2602.03200), WiLoR (2409.12259),
HaMeR (2312.05251), EgoAllo/HMP (2410.03665), WHOLE (2602.22209), EgoGrasp (2601.01050),
EgoForce (2605.12498), HORT (2503.21313), HOLD (2311.18448), PAD-Hand (2603.26068), intuitive-physics-hand
(2508.01835), UniDepth/V2 (2403.18913 / 2502.20110), Metric3D/v2 (2307.10984 / 2404.15506), Depth-Pro
(2410.02073), DepthAnythingV2 (2406.09414), Marigold (2312.02145), MoGe-2 (2507.02546), UniK3D (2503.16591),
DepthAnything3 (2511.10647), HAC (2407.00574), SynCHMR (2405.14855), MetricHMSR (2506.09919),
UniSH (2601.01222), MfH (NeurIPS'24 GA8TVtxudf), WorldMirror (2510.10726), MapAnything (2509.13414),
AMB3R (2511.20343), NoPoSplat (2410.24207), Splatt3R (2408.13912), GS-LRM (2404.19702), PM-Loss (2506.05327).
(Some well-known IDs cited from memory — 30-sec confirm before they enter the paper.)

### Round-2 research agents (RE-RUN 2026-06-20 — DONE; full results in Part 10 below). Additive: skip the list above.
- **Agent R1** (CVPR'26 + 3DV'26 + ICLR'26, also WACV'26/SIGGRAPH'25): feedforward 3D/4D GS, feedforward
  pose/geometry backbones resisting drift WITHOUT SLAM, making up-to-scale metric, test-time refinement,
  LoRA-adapting frozen 3D foundation models. Return: NEW papers only, top-5 leads by impact×feasibility,
  flag collisions with hand-anchor-metric idea.
- **Agent R2** (ICCV'25 + ECCV'24/'26 + NeurIPS'25): egocentric 3D hand / hand-object / world-grounded 4D
  hand, hand-scene joint recon, hand/human-as-metric-anchor, new ego hand-object datasets/benchmarks/
  leaderboards 2025-26. Return: NEW only, top-5, flag collisions.
- **Agent R3** (collision + protocol intel): newest/closest papers to "hand/body/known-object as metric
  anchor for feedforward 3DGS/NeRF in ego video" ranked by threat; EXACT eval protocols for HOT3D
  (HaWoR 33.2/11.3, WHOLE 10.41/0.58) and HOI4D (Hand3R 42.6, W 86.9/125.8, WA 38.0/56.7) — alignment
  (first-frame vs per-window Procrustes), window size, segment length, 21 vs 16 joints, right-hand-only,
  metric-scale handling; reusable eval toolkits (HOT3D challenge, Ego-Exo4D); 2026 challenges to target.

(Full prompts were in the launch calls; reconstruct from the bullets above.)

---

## Part 9 — Open decisions / notes
- Table 5: add WHOLE + EgoForce rows; reframe our row as "SLAM-free, ceiling = HaWoR-parity."
- Masking ablation reframed from reviewer-defense → **core thesis test** (hand-anchor vs masked-scene-depth).
- The "weekend training runs you asked to line up" (hires + unfreeze) = Tier B, gated on `/work`.
- Tasks: #18 (HOI4D track) and #27 (world-space eval) still in_progress; #26 (hi-res crops) in_progress.
  Consider adding: render-and-PnP, Hand3R-HOI4D-world-space, round-2-research.

---

## Part 10 — Round-2 venue deep-dive (2026-06-20)
Re-ran the three failed venue agents in foreground (3 parallel general-purpose). Verified arXiv IDs where
possible; numbers marked *provisional* were not fetched from the PDF table directly — confirm before citing.

### 10.1 — The confirmed head-to-head bar (Hand3R Table II, HOI4D, VERIFIED)
Hand3R = `2602.03200`, **arXiv-only (submitted 3 Feb 2026), NOT peer-reviewed, no code/split released.**

| Method | Type | C-MPJPE↓ | 30f WA↓ | 30f W↓ | 100f WA↓ | 100f W↓ |
|---|---|---|---|---|---|---|
| HaMeR-SLAM | Offline | 248.23 | 52.69 | 140.75 | 85.46 | 218.05 |
| WiLoR-SLAM | Offline | 252.24 | 52.91 | 146.91 | 87.51 | 223.00 |
| HaWoR | Offline | 51.77 | 22.54 | 41.28 | 27.40 | 58.62 |
| **Hand3R** | Online | **42.6** | 38.04 | 86.87 | 56.71 | **125.81** |

- **Both target numbers CONFIRMED in one table: C-MPJPE 42.6, 100f-W-MPJPE 125.81 (HOI4D).** This is the
  "beat 125.8 long-drift" goal — now exact.
- Protocol (verified prose): C-MPJPE = absolute cam-frame, no align. W = align **first frame(s)** (rigid,
  no scale). WA = **full-trajectory Procrustes** (Sim(3), with scale). Windows 30 / 100 frames. Datasets =
  **DexYCB + HOI4D only (NO HOT3D, NO H2O).**
- **Narrative gift:** Hand3R *beats* HaWoR on camera-frame C-MPJPE (42.6<51.77) but *loses badly* in world
  space (100f-W 125.81 vs 58.62) — online single-pass trades world stability for cam accuracy. Our
  hand-anchor pitch targets exactly that world-space gap.
- **Contestability flags (reproducibility risk):** (1) alignment Procrustes-vs-Sim(3) and scale-free-vs-
  scale is UNDERSPECIFIED → report both variants or our number is contestable in their favor; (2) no split
  released → must reconstruct (likely standard 7:3) and state sequences explicitly; (3) C-MPJPE root-vs-
  centroid alignment is paper-specific. **HOT3D head-to-head is more reproducible (HaWoR ships code+split).**
- **Do NOT merge the two HOI4D protocols.** EgoGrasp reports single-digit HOI4D W-MPJPE (HaWoR 9.04 there);
  Hand3R reports tens-to-hundreds. Different conventions → keep as separate comparison blocks.

### 10.2 — Novelty threats (closest first)
- **🔴 Hand3R (2602.03200)** — frozen hand expert (HaMeR-ish) + frozen 4D-scene FM (CUT3R) via scene-aware
  visual prompting → joint online metric hand+scene. **The most direct architectural twin.** Differentiate
  on: explicit **3DGS Gaussians** vs CUT3R recurrent state; scale anchored to **metric MANO hand** vs generic
  scene-memory. arXiv-only + concurrent → benchmark head-on, not just cite.
- **🔴 EgoGrasp (2601.01050)** — aligns DepthAnything3 depth to metric via **mean ratio of MANO-model depth
  to estimated depth in hand regions** = literally our hand-as-metric-anchor, on H2O + HOI4D, vs HaWoR/
  Dyn-HaMR. Provisional H2O W 6.84 / WA 40.93; HOI4D 8.61 / 192.06 (labels look swapped — verify). Our
  deltas: frozen feedforward **3DGS** (vs their DA3+WiLoR+SAM3D pipeline), full 4DGS, HOT3D/Aria coverage.
- **🔴 UniSH (2601.01222)** — feedforward joint scene+human metric recon using SMPL height prior as scene's
  metric reference. Same recipe, but full-body **SMPL / multi-view, not ego-monocular hand.** Architectural twin.
- **🟡 Hand-4DGS — `arXiv:2606.19156` (VERIFIED 2026-06-20, submitted 17 Jun 2026, arXiv-only, NOT peer-
  reviewed)** — authors Jeongmin Bae, Seoha Kim, **Marc Pollefeys**, **Mahdi Rad**, Youngjung Uh, **Taein
  Kwon** (Yonsei + ETH/Microsoft Spatial AI Lab + Oxford VGG). First feedforward 4D-hand-from-ego-video,
  mesh-guided (MANO) Gaussians ~60 FPS, 2D-supervised, refines HaMeR init. Project page
  jeongminb.github.io/hand-4dgs, code "Coming Soon." **HAND-ONLY: no scene, NO metric-scale claim, NO
  world-space/W-MPJPE metric, eval on H2O + ARCTIC only (no HOT3D/HOI4D).** → Nearest-NAME collision but
  occupies NONE of our contribution (joint scene+hand, metric-via-hand-anchor). **Reinforces our novelty.**
  Cite + differentiate on (1) joint scene 3DGS + hand, (2) metric scale by anchoring hand into metric scene.
  Genuinely concurrent + high-credibility (Pollefeys/Rad) → reviewers WILL know it; must cite explicitly.
- **Human-as-anchor for SMPL bodies is essentially SOLVED:** MetricHMSR (2506.09919, "recovered metric human
  as a geometric anchor"), SynCHMR (2405.14855, "Human-aware Metric SLAM"), JOSH (2501.02158, contact-coupled
  joint scene+human), SHARE (2510.15342, scene→human grounding — inverse of our P1 scene_follows_hand),
  PhySIC (2510.11649), Human3R (2510.06219, ICLR'26, body analog of our exact thesis). → Scope our novelty
  TIGHTLY to **egocentric-monocular + MANO-hand anchor + frozen up-to-scale 3DGS** (the one combo none occupy).
- **Premise threat — feedforward backbones now give metric scale WITHOUT a hand:** AMB3R (2511.20343, CVPR'26),
  MapAnything (2509.13414, 3DV'26, explicit factored metric-scale token), MoGe-2 (2507.02546), Any4D
  (2512.10935, 3DV'26), PLANA3R. → Defend by framing hand-anchor as **drift-robust in the near-field** where
  global scene-metric is weakest (corroborated by EgoForce + our own "scene-depth-at-hand unreliable" finding).

### 10.3 — Implementable levers (new)
- **⭐ "Learning 3D Reconstruction with Priors in Test Time" (2604.03878)** — test-time constraint
  optimization of a **frozen DUSt3R-family backbone** (our exact setup); our metric hand IS such a prior.
  Generalizes our single global Sim(3) Procrustes → higher-DOF per-prediction opt, >50% pointmap-error cut.
  **Most directly adoptable.**
- **VGGT-SLAM (2505.12549, NeurIPS'25)** — SL(4) 15-DOF submap factor graph; the canonical "Sim(3) is
  inadequate for uncalibrated cams" cite + adoptable alignment. Pair with **probabilistic Procrustes soft-
  dustbin (2507.18541)** as robust upgrade to our hard Sim(3).
- **Render-and-PnP family** (beyond GS-CPR): "Rethinking Pose Refinement in 3DGS under uncertainty"
  (2603.16538, CVPR'26 — uncertainty weighting addresses our "unreliable scene-depth-at-hand"); epipolar
  3DGS pose refine (2508.17876); iGaussian fast feedforward inversion (2511.14149).
- **DyTact (2506.03103, 3DV'26 Oral)** — MANO-bound 2D Gaussian surfels + differentiable contact-map render:
  cleanest recipe for **binding our scene Gaussians to the MANO hand head.**
- **Contact/collision priors (cheap differentiator, tightens scale via penetration gradient):**
  (1) **PROX (1908.06963)** scene-SDF penetration — cheapest, battle-tested, SMPL-X→MANO direct, code out.
  (2) **CHOIS (2312.03913)** point-cloud-SDF contact+penetration — exact precedent for "Gaussians-as-points".
  (3) **Interaction-Aware 4D-GS (2511.14540)** native hand-Gaussian↔object-Gaussian penetration — our rep
  exactly; no code → reimplement + must-cite for novelty positioning.
- **ForeHOI (2602.06226)** — feedforward HOI object branch (~100x faster than opt, evals on HOT3D): drop-in
  to extend hand-only → full HOI while staying feedforward.

### 10.4 — Protocol/dataset GT facts (verified)
- W-MPJPE / WA-MPJPE / PA-MPJPE lineage = **SLAHMR (2302.12827)** defines them; WHAM (2312.07531) fixed
  100-frame segments + first-2-frame W + RTE%; reused by TRAM/GVHMR/HaWoR. C-MPJPE = non-standard umbrella
  (root vs centroid align is paper-specific — verify per-paper).
- **HOT3D:** mm-accurate gravity-aligned metric extrinsics (Aria MPS), MANO+UmeTrack, but **NO dense GT depth**
  (sparse SLAM points only). HaWoR's native benchmark; held-out participants P0004/05/06/08/16/20.
- **HOI4D:** per-frame SLAM extrinsics (`3Dseg/output.log`), **real dense GT depth** (Kinect v2 / D455),
  MANO 21-jt. Standard 7:3 random sequence split. ← our dense-depth pivot + Hand3R head-to-head live here.
- **H2O:** per-frame 4×4 extrinsics, **dense GT depth** (Azure Kinect), both hands 21-jt, leave-subject4-out
  split. ← EgoGrasp's H2O numbers live here.

### 10.5 — What changed the plan
1. **Beat-125.8 is now a precise, sourced target** with the full HOI4D table → wire our world-space eval to
   reproduce Hand3R's exact protocol (both alignment variants) and the 7:3 split.
2. **Two direct twins (Hand3R, EgoGrasp) + two architectural twins (UniSH, Hand-4DGS)** → novelty must be
   scoped to ego-monocular + MANO + frozen-3DGS, and these become **head-on baselines, not citations.**
3. **Verify Hand-4DGS venue/ID urgently** (related-work freeze risk; no clean ID found).
4. **New cheap differentiators to consider:** PROX/CHOIS/Interaction-4DGS penetration loss; DyTact-style MANO-
   bound surfels; test-time prior-optimization (2604.03878) as the render-and-PnP-class drift lever.

---

## Part 11 — HOI4D world-space data acquisition (2026-06-20): UNBLOCKED for 11 seqs
The world-space HOI4D head-to-head (vs Hand3R 100f-W 125.81) was blocked on missing camera poses.
RESOLVED — all 4 modalities now in hand for 11 sequences (the dense-depth set: C1/C2/C3/C5/C7 +
C12×2/C13×2/C14×2, all `ZY20210800001_H1_*`, ~300 frames each, 2792 handpose frames total).

**Source map (the Livioni HF mirror is a STRIPPED repackage — RGB+depth+frame-list JSON only, NO poses):**
- RGB ✓ — `huggingface.co/datasets/Livioni/hoi4d/<seq>.tar.gz` → `images/*.jpg` (on cluster /work/.../hoi4d).
- Intrinsics ✓ — `huggingface.co/datasets/yinloonga/HOI4D/resolve/main/camera_params.zip` (2.1 KB,
  ungated) → `camera_params/ZY20210800001/intrin.npy` (fx≈1060, cx≈971, cy≈523 @1920×1080).
- Extrinsics ✓ — `yinloonga/HOI4D/HOI4D_annotations.zip` (22 GB, range-extractable via `remotezip`;
  central dir = 1.82M entries, ~23 s to index) → `3Dseg/output.log` (Open3D 5-line blocks, frame0
  identity = poses relative to frame 0; ~58 KB/seq). Extracted all 11 in 34 s, no full download.
- Handpose GT ✓ — **OneDrive/Baidu only, NO scriptable mirror anywhere** (exhaustively confirmed). Already
  had it locally for exactly these 11 seqs at `hoi4d_gt/Hand_pose/handpose_right_hand/` (manual download).
  Pickle keys verified: poseCoeff[48], beta[10], trans[3] (cam-frame m), kps2D[21,2]. Right hand only.

**Staged + validated:** assembled a preprocess-ready annotation root at `hoi4d_gt/` (camera_params/,
`<seq>/3Dseg/output.log`, `handpose/refinehandpose_right` → symlink to Hand_pose/handpose_right_hand).
All 3 loaders (`load_intrinsics`/`load_extrinsics`/`load_mano_frame` in preprocess_hoi4d.py) resolve
against it for a test seq (300/300 frames). `preprocess_hoi4d.py` patched to read the Livioni `images/`
frame-dir (underscored seq) as RGB fallback + new `--rgb_root` arg (py_compile clean).

**Remaining (cluster-only, next session — needs gb10 + venv_gb10 + SSH expect helpers):**
1. Push `hoi4d_gt/` annotation root to cluster (tiny, ~tar+scp via expect helper).
2. Run `preprocess_hoi4d.py` per seq: `--hoi4d_root <pushed hoi4d_gt> --rgb_root /work/scratch/dmonopoli/hoi4d
   --seq <slashed> --out <preprocessed>`. Writes the HOT3DHandDataset cache set (incl. gt_joints_cache_world.pt).
   ⚠️ STILL VERIFY the 3 flagged unknowns on seq 1 before scaling: (1) output.log direction (world→cam vs
   cam→world — back-project depth / check gt_joints_2d overlays the hand in RGB), (2) 45→15 PCA pose, (3)
   cam→world lift. Output is /work-bound (quota!) — write to node-local /tmp on the gb10 job if /work is full.
3. Run `eval_world_space.py` on the 11 seqs → our W/WA-MPJPE (short 30f / long 100f) + C-MPJPE.
4. Reproduce Hand3R protocol EXACTLY: 7:3 split (state seqs), first-frame W vs full-traj-Procrustes WA,
   report BOTH alignment variants (their Procrustes-vs-Sim(3) is underspecified). Target: 100f-W < 125.81.
