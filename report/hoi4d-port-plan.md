# HOI4D port plan + competitive landscape (CVPR dense-depth pivot)

Research date: 2026-06-17. Target: CVPR ~Nov 2026. Goal: port HOI4D (dense RGB-D,
Hand3R's benchmark) into our feedforward 4DGS+MANO pipeline for dense metric-depth
supervision + a direct SOTA comparison. See [[cvpr-target-dense-depth-pivot]].

## 1. Competitive landscape — READ THIS FIRST (it shapes the pitch)

**The "egocentric hand + metric 4D scene" space got hot in 2025-2026.** Our nearest
neighbours all appeared in the last ~12 months:

| Method | Venue | Scene repr | Metric scale from | HOI4D? | Code |
|---|---|---|---|---|---|
| **Hand3R** | arXiv 2602.03200 (Feb-2026 **preprint**, no venue) | CUT3R point cloud | CUT3R native metric + L_abs/L_trans on HOI4D | **yes** (C/W/WA-MPJPE) | **none** |
| HaWoR | CVPR 2025 | — (SLAM) | Metric3D + scale-opt | only as re-run baseline | yes (ThunderVVV/HaWoR) |
| EgoGrasp | arXiv 2601.01050 | — | — | **yes** (+H2O) | — |
| WHOLE | arXiv 2602.22209 | — | — | no (HOT3D+H2O) | — |

**Hand3R is our near-twin:** HaMeR hand expert + CUT3R 4D scene foundation model,
coupled, egocentric, metric, on HOI4D. That is almost our pitch. **So our novelty
cannot be "metric hand + scene egocentric" — that's taken.** Our defensible edge:
- We output **renderable Gaussians** (novel-view synthesis), not just a point cloud.
- We recover metric from an **up-to-scale** GS backbone **via the hand coupling +
  dense depth**, rather than leaning on a natively-metric foundation model (CUT3R).
- **Unified single feedforward pass**: Gaussians + depth + camera + hand together.

Good news: Hand3R has **no code, no published split, no venue** -> it's concurrent
work, not an established baseline we're "behind." We cite it as concurrent and we
must re-implement its protocol ourselves.

## 2. The number to beat
**C-MPJPE** (camera-space absolute joint error, **no alignment**, mm) — the metric-
scale headline. Hand3R = **42.6 mm**; HaWoR = 51.77 mm. Our new abs-MPJPE + WRIST_mm
metrics map directly onto C-MPJPE (this is why Cyrus's "absolute MPJPE + wrist" ask
matters). Also report W-MPJPE (align first frames) + WA-MPJPE (full Procrustes).
No standard split exists across papers (Hand3R 30/100-frame clips; EgoGrasp 128-frame
Procrustes). **We define a clean protocol (EgoGrasp 128-frame is the cleanest doc'd)
and re-run HaWoR (code available) as the in-house baseline.**

## 3. HOI4D facts (for the preprocessor)
- Access: hoi4d.github.io, CC BY-NC 4.0, OneDrive + Baidu mirror, modality-split
  downloads (RGB / Depth / Camera Params / Hand Pose / Annotations / CAD). Depth
  alone = 127 GB; full ~hundreds GB. Only `release.txt` seqs are public.
- Seq path: `ZY.../H*/C*/N*/S*/s*/T*/`. RGB `align_rgb/image.mp4` (1920x1080, 15fps,
  ~300 frames). Depth `align_depth/depth_video.avi` -> **must** decode via official
  `utils/decode.py` -> 16-bit PNG `%05d.png`, **millimeters (/1000 -> m)**, aligned
  to RGB, range ~0.01-10 m (RealSense D455 + Kinect v2).
- Intrinsics: `camera_params/{cam_id}/intrin.npy` 3x3 (fx,fy,cx,cy), one per camera ID.
- Extrinsics: `3Dseg/output.log` Open3D/Redwood per-frame 4x4. **Verify world->cam vs
  cam->world empirically** (loaders disagree) by back-projecting depth into `raw_pc.pcd`.
- Hands: separate tree `handpose/refinehandpose_right/.../{frame}.pickle` = MANO
  `poseCoeff` 48 (3 global axis-angle + **45 full pose, no PCA**), `beta` 10,
  `trans` (camera frame, m), `kps2D`. **Right hand**; left-hand release unconfirmed.
  `ManoLayer(use_pca=False, ncomps=45, flat_hand_mean=True, side='right')`,
  joints = `manolayer(theta,beta)/1000 + trans`.
- Objects: `objpose/{frame}.json` center (m, cam frame) + Euler XYZ rot + dims; CAD
  meshes per instance; category-level.
- Reusable loaders: **egomono4d/dataset_hoi4d.py** (closest 4D pipeline: RGB+depth
  /1000 + intrin + output.log poses), **zerchen/hort_train/cocoify_hoi4d.py** (MANO
  +intrinsics), ThermoHands/read_hoi4d.py, Vp-SoLo/HOI4D-prepare-4D-data (bug-fixed).

## 4. Our input contract (what the preprocessor must emit per sequence)
`preprocessed/{seq}/`:
- `video_main_rgb.mp4`
- `hand_data/cam_intrinsics.pt` `[3]=[focal,cx,cy]` at the video frame size
- `hand_data/cam_extrinsics_cache.pt` `[N,4,4]` T_camera_world
- `hand_data/mano_hand_pose_trajectory.jsonl` (per frame: hand_poses{id:{wrist_xform
  {t_xyz,q_wxyz}, pose[15], betas[10]}})
- `hand_data/gt_joints_cache_{world,cam_v2}.pt` `[N,2,16,3]`
- `hand_data/gt_joints_2d_cache.pt` `[N,2,16,3]` (u,v,conf)
- `hand_data/hand_bboxes_v2_*.pt` (bboxes [N,2,4], valid [N,2], gt [N,64])
- (new) dense depth cache for supervision — see §6.

Our MANO pack = 32/hand: `[0:3]` transl, `[3:7]` quat wxyz, `[7:22]` **15 PCA pose**,
`[22:32]` 10 betas. (preprocess_undistort.py; dataset HOT3DHandDataset in train_hand_head.py.)

## 5. The 3 conversion gotchas
1. **MANO pose basis mismatch:** HOI4D = 45 full pose; ours = 15 PCA. Don't convert
   lossily — run HOI4D's ManoLayer to get 3D **joints/verts** and supervise on those
   (kp3d, kp3d_abs, kp2d, transl, global_orient, betas); drop/relax the 15-PCA pose
   param-loss for HOI4D (or fit PCA coeffs offline). Joints are what the metric needs.
2. **Camera vs world frame:** HOI4D hand `trans`/joints are camera-frame; we need real
   per-frame extrinsics (output.log) for the GS reconstruction (parallax). Lift
   camera-frame hand -> world via inverse extrinsics to fill gt_joints_cache_world,
   let the pipeline reproduce camera-frame as usual. Verify output.log direction first.
3. **Depth-RGB alignment + resolution:** decode.py output is RGB-aligned; confirm
   decoded depth dims; mask zeros as invalid; clamp [0.01,10] m.

## 6. Dense-depth supervision hook (the whole point)
We already have `object_depth_loss(gs_depth, gt_obj_depth[B,S,R,R], gt_obj_mask[B,S,R,R])`
sampling gs_depth on a normalized grid (resolution-independent). For HOI4D, replace
the per-frame mesh render in `HOT3DHandDataset._render_clip_obj_depth` with **native
dense depth**: load decoded 16-bit PNG -> /1000 -> resize to R -> `gt_obj_depth`,
`gt_obj_mask = depth>0`. **No masking to objects** -> dense supervision -> directly
answers Cyrus's degradation worry. The loss + training wiring are unchanged.

## 7. Recommended sequencing
1. Get one short HOI4D sequence (a few hundred MB) + decode RGB+depth.
2. **Backbone-compat smoke** (highest risk): forward our frozen backbone on HOI4D RGB,
   eyeball Gaussians/depth sanity. If it breaks here, rethink before porting.
3. Write `scripts/preprocessing/preprocess_hoi4d.py` (reuse egomono4d / hort_train I/O):
   emit the §4 contract for a handful of sequences.
4. Dense-depth path in the dataset (§6) + a HOI4D config.
5. Define the eval protocol (§2), re-run HaWoR baseline, report C/W/WA-MPJPE.
6. Train: frozen-backbone + dense-depth + hand, then partial-unfreeze variant.

H2O is the secondary dataset (egocentric, both hands) — same machinery, later.
