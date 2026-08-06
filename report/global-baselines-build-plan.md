# Global-baseline build plan (1b HaPTIC-HD, HaWoR camera-frame)

Status 2026-07-19. Scoped and CLIs confirmed; Euler-side execution (repo/weights) pending.
Both run on the Euler es_tang share (2080ti, sm_75, venv_haptic torch 2.1.1), NOT the
Blackwell/ARM student nodes. Scored with `scripts/eval_worldspace_baseline.py`
(`--segment_len 30 --wa_short 30` for the short window Cyrus wants).

## Metric availability (the honest constraint)
- Cited rows (Hand3R, HaWoR-paper, HaMeR+SLAM, WiLoR+SLAM): only C-MPJPE + WA + W (their
  Table II). No abs-wrist, no local-MPJPE. Cannot self-run Hand3R (no code).
- Self-run rows (Ours, HaPTIC, HaWoR-camera): full set EXCEPT WA/W need a trajectory.
  Ours has its own estimated extrinsics -> all metrics. HaPTIC/HaWoR-camera give the
  camera-frame trio (abs-MPJPE, abs-wrist, local); their WA/W stay paper-cited.

## 1b - HaPTIC native-HD rerun
Pipeline scripts already exist. Prior 224px run: C_rr 28.7 / WA_long 35.3 VALID, C_abs
broken (2.73x, weak-persp miscalib at tiny hand). Prior HD attempt (job 6612959): C_rr
25.7 (better) but C_abs 6360 = still published-conversion units -> the true-focal metric
conversion is the ONE unresolved step.
Steps:
1. Re-fetch 157 HOI4D test videos at native HD (yinloonga HOI4D_release mirror; see
   hoi4d-data-expansion-pipeline). ~tens of GB, stream+delete per seq.
2. `scripts/hoi4d_to_haptic.py --data_root <HD test> --out_root <haptic_in>` - writes HD
   frames + intrinsics rescaled to HD res (script already handles "HD re-extract";
   focal.txt from cam_intrinsics rescaled).
3. Run HaPTIC on haptic_in (venv_haptic) -> per-frame pkls (cJoints/wJoints).
4. `scripts/haptic_to_worldeval.py` -> pred_dir of <seq>.pt (cam_joints/world_joints,
   metres, smplx-16).
5. **UNRESOLVED: true-focal metric conversion.** HaPTIC's absolute depth is in its own
   weak-persp/published units at our scale; WiLoR/HaMeR baseline scripts force an explicit
   true-focal crop-to-full conversion that HaPTIC's internal pipeline skips. Must inject the
   same conversion (using true HD focal) into haptic_to_worldeval.py before scoring abs.
   Do NOT apply an empirical depth rescale (reviewer-indefensible).
6. Score `eval_worldspace_baseline.py --segment_len 30 --wa_short 30`.

## HaWoR - camera-frame only (skip DROID-SLAM)
Repo github.com/ThunderVVV/HaWoR. Weights: hawor.ckpt (+model_config.yaml), detector.pt
(WiLoR), metric_depth_vit_large (Metric3D), MANO. NOT droid.pth (SLAM-only).
Model output (from eval_hawor_hot3d.py): MANO params + joints; `pred_rotmat` root orient,
`pred_trans` translation, joints (T,21,3). `demo.py` has `--vis_mode cam` = camera-frame.
Steps:
1. Clone repo on Euler; fetch the 4 weight sets (HF + GDrive + MANO); build/verify env
   (repo says torch>=1.13 ok; try venv_haptic torch 2.1.1 first, else fresh 1.13/cu117 -
   but Metric3D + hand net are plain torch, no custom CUDA).
2. **GO/NO-GO RESOLVED 2026-07-19 = GO** (verified from demo.py source, no clone needed).
   Camera-frame hand output is UPSTREAM of DROID-SLAM: flow is detect_track_video ->
   hawor_motion_estimation (hand model fwd, cam frame) -> hawor_infiller (per-frame cam
   MANO: pred_trans/pred_rot/pred_hand_pose/pred_betas) -> hawor_slam (world lift, runs
   AFTER). run_mano()/run_mano_left() give cam-frame joints/verts BEFORE the R_x world
   transform. pred_trans is metric from the hand model (it precedes Metric3D, which is at
   the SLAM step -> scales the camera trajectory, not the hand). So build the adapter from
   motion_estimation+infiller+run_mano and STOP before hawor_slam. Only W-MPJPE needs SLAM.
   Adapter's first data check: sanity pred_trans wrist depth vs GT (~0.68m) to confirm
   metric; if off, add Metric3D (plain torch, still no DROID-SLAM).
3. Adapter: feed our 157 HOI4D test frames (224 store or HD) + our detector boxes
   (v3, reuse) into HaWoR hand est -> per-frame cam joints (21 -> smplx-16 remap) -> metres,
   true-focal convention -> pred_dir <seq>.pt (cam_joints; world_joints = NaN, we don't run
   SLAM).
4. Score `eval_worldspace_baseline.py` for the camera-frame trio; leave WA/W as paper 22.5/41.3.

### HaWoR concrete execution recipe (gathered 2026-07-20 from repo source)
CLONE: `git clone --recursive https://github.com/ThunderVVV/HaWoR.git` (submodules, but we
skip the DROID-SLAM build entirely).
ENV: repo wants python3.10 + torch1.13.0+cu117 + requirements.txt + pytorch-lightning==2.2.4
(--no-deps) + torchmetrics==1.4.0. SKIP `thirdparty/DROID-SLAM setup.py install` (SLAM only).
Try Euler es_tang venv_haptic (torch2.1.1) first; if hawor.ckpt/pl load fails, build a fresh
py3.10 venv on Euler scratch.
WEIGHTS (all public HF, no token needed; wget):
  hawor.ckpt  -> huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt  -> weights/hawor/checkpoints/
  infiller.pt -> .../hawor/checkpoints/infiller.pt -> weights/hawor/checkpoints/
  model_config.yaml -> .../hawor/model_config.yaml -> weights/hawor/
  detector.pt -> huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -> weights/external/
  SKIP droid.pth (SLAM). Metric3D metric_depth_vit_large_800k.pth (GDrive 1eT2gG-...) — try WITHOUT
  first (go/no-go: pred_trans is metric from the hand model, precedes Metric3D); add only if import/run needs it.
MANO (the blocker — DO NOT try the manual license download): SEARCH the cluster for existing
MANO_RIGHT.pkl / MANO_LEFT.pkl (HaMeR at team25/models/hamer, our smplx-16 pipeline, projectaria)
and SYMLINK into _DATA/data/mano/MANO_RIGHT.pkl + _DATA/data_left/mano_left/MANO_LEFT.pkl.
ADAPTER call chain (camera-frame, NO slam) per demo.py:
  from scripts.scripts_test_video.detect_track_video import detect_track_video
  from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
  from hawor.utils.process import run_mano, run_mano_left
  start,end,seq_folder,imgfiles = detect_track_video(args)           # or inject our detbox v3
  frame_chunks_all, img_focal = hawor_motion_estimation(args,start,end,seq_folder)
  pred_trans,pred_rot,pred_hand_pose,pred_betas,pred_valid = hawor_infiller(args,start,end,frame_chunks_all)
  out = run_mano(pred_trans[hi:hi+1,a:b], pred_rot[...], pred_hand_pose[...], betas=pred_betas[...])
  # out["joints"] = (B,T,21,3) CAMERA frame, transl applied = ABSOLUTE metric cam joints. STOP (no hawor_slam).
  # remap 21->smplx-16, metres, true-focal -> pred_dir <seq>.pt {cam_joints:[N,2,16,3]}; world_joints=NaN.
OPEN (finalize from cloned source): exact args fields for detect_track_video/motion_estimation
(image dir vs video), the 21-joint order for the ->16 remap, whether pred_trans wrist depth is
metric (sanity vs GT ~0.68m). Euler double-hop via scratchpad/run_euler.exp (untested euler pw).

## Effort / risk
- 1b: ~1 day; single unresolved step (true-focal conversion). Medium confidence.
- HaWoR-camera: ~1-2 days; gated on step-2 verification (cam-frame extractability). If that
  fails, no self-run row -> cite-only. Confirm before committing the full build.
- Both are Euler-side + weight downloads -> execute when supervisable (login node reboots
  Sun midnight). Neither is "single-submit ready" yet; this plan makes each deterministic.
