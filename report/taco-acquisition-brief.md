# TACO: how to download it, and what will bite you

Approved by Cyrus on 2026-08-10 as the fifth training store. This is the working brief: what to
fetch, what to skip, what to check before trusting a single frame of it.

TACO is *Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding*, CVPR 2024, Liu et
al. Repo `leolyliu/TACO-Instructions`, project page `taco2024.github.io`.

## Why it qualifies, and how we know

Every claim below is read off their loader code, `dataset_utils/project_pose_to_egocentric_view.py`,
not off the abstract. That distinction cost us a week on Re:InterHand, whose abstract said
"egocentric views" and whose data turned out to be independently sampled viewpoints.

- **The camera is worn.** Their capture setup: "a **helmet** equipped with a **Realsense L515**
  camera is **worn by the actor**", 1920x1080 at 30 Hz. So the trajectory is real head motion.
- **The pose ground truth is mocap, not SLAM.** Six infrared NOKOV Mars4H cameras. Better than
  anything we could reconstruct.
- **Per-frame extrinsics exist and their direction is stated in the code**, line 113:
  `egocentric_extrinsics = np.load(...)  # world_to_camera, shape = (N_frame, 4, 4)`. That is
  already our `cam_extrinsics_cache` convention, so no direction guessing.
- **Intrinsics are constant per sequence**, one 3x3 in a text file. This is what Re:InterHand could
  not offer and what makes a one-focal-per-sequence store possible with no warping.
- **Full MANO**, `hand_pose` (48,) axis-angle plus `hand_trans` (3,) and `hand_shape` (10,), under
  `ManoLayer(use_pca=False, ncomps=45, center_idx=0)`. Exactly what `split_pose48` already parses.
- **No distortion coefficients anywhere.** Plain pinhole, so no undistort pass, unlike HOT3D.

## Which version to take

| | sequences | source | verdict |
|---|---|---|---|
| pre-release | 244 | OneDrive | **start here** |
| version 1 | 2317 | Dropbox | store is affordable, full feature cache is not |

The store is cheap and the feature cache is not. Our converted stores are mp4 plus annotations:
ARCTIC is 22 GB for 267 sequences. Its **feature cache is 354 GB** for the same 267, because the
cache is one 33.6 MB file per 32-frame clip. So cache size tracks total frames, not sequence count.

Euler scratch is at 2.12 of 2.50 TB soft, 2.70 hard. That leaves roughly 380 GB before the soft
limit, which is about 11000 cached clips. 244 sequences is the same order as ARCTIC (267) and HOT3D
(198) and will fit. 2317 will not, and a half-cached store is worse than a smaller one because the
sampler will stall on cache misses.

**Decision rule:** download the full version 1 if bandwidth is free, convert all of it, then build
the feature cache for a subset sized so that `n_clips * 33.6 MB` stays under the remaining quota.
Measure `n_clips` after conversion, do not guess it.

## Downloading

Both hosts are shared-folder links, and neither has a working public listing API. I tried the
OneDrive `shares/u!<base64>` endpoint on `api.onedrive.com` and `graph.microsoft.com`; both return
401. So the folder has to be opened in a browser once. After that it is scriptable.

**Take exactly these three subtrees.** The bulk of TACO is 12-view allocentric video we will never
look at, and the depth videos are only worth having if the dense-depth line is revived.

    Egocentric_RGB_Videos/<triplet>/<sequence>/color.mp4
    Egocentric_Camera_Parameters/<triplet>/<sequence>/egocentric_intrinsic.txt
    Egocentric_Camera_Parameters/<triplet>/<sequence>/egocentric_frame_extrinsic.npy
    Hand_Poses/<triplet>/<sequence>/{left,right}_hand.pkl
    Hand_Poses/<triplet>/<sequence>/{left,right}_hand_shape.pkl

    SKIP: Allocentric_RGB_Videos, Marker_Removed_Allocentric_RGB_Videos,
          Allocentric_Camera_Parameters, Object_Poses, Object_Models, Egocentric_Depth_Videos

**Note the README's directory tree does not list `Egocentric_Camera_Parameters`.** Their own code
does, at line 75. If the folder is missing from the share, that is the first thing to raise with
the authors, because without it the dataset is useless to us.

The practical route onto the cluster, since neither cluster has a browser:

1. Open the Dropbox (or OneDrive) folder in a browser, navigate into one of the three subtrees,
   and start the download so the host generates a zip.
2. Cancel the local download and copy the generated direct URL out of the browser's download list
   (`chrome://downloads`, right click, copy link address).
3. `wget` that URL on the cluster. The URL is time-limited, so do this promptly and re-copy it if
   it expires.

Do this per subtree rather than for the whole share, or you will pull the allocentric video too.
`Egocentric_Camera_Parameters` and `Hand_Poses` are small and can go in one shot;
`Egocentric_RGB_Videos` is the large one and is worth taking per triplet.

Run the transfer under `sbatch`, not on the login node. The login-node reaper has killed long
downloads before.

The BaiduNetDisk mirror splits large files. If you end up there:

    cat Egocentric_Depth_Videos_split.* > Egocentric_Depth_Videos.zip

## Four gates before trusting any of it

Each of these caught a real defect on some other store. Run them in this order and stop at the
first failure.

1. **Temporal coherence.** Consecutive-frame `|d extrinsic translation|` and `|d focal|`. This is
   the exact test Re:InterHand failed, at 755 mm and 166 px per frame. TACO should be smooth and
   small. If it is not, it is not a video dataset and the whole plan is off.
2. **Frame-count alignment.** `len(mp42imgs(color.mp4))` against `N_frame` from the extrinsics.
   Their own script prints `[error] losing frames in the egocentric video, skip!`, so the mismatch
   is a known upstream condition. Our store contract is that video frame `t` is cache row `t`; a
   half-frame offset here is what inflated a HaWoR baseline by 2.5x once.
3. **Depth gate.** Median wrist depth, against the pool and the held-out sets:

       HOT3D 0.339   OakInk2 0.386   ARCTIC 0.474   DexYCB 0.780
       held out:  H2O 0.503   HOI4D 0.677

   We do not need TACO to be deep, DexYCB already covers that end. Just record where it lands, so
   the depth-coverage argument stays honest.
4. **Anatomical and joint-order gate.** Bone lengths after the 21 to 16 remap. A scrambled remap is
   what corrupted every H2O number we produced until bone lengths caught it, and it is silent
   otherwise.

## Then

Write `scripts/preprocessing/taco_to_ours.py` against the store contract that
`scripts/preprocessing/dexycb_to_ours.py` documents in its header, which is derived from what the
training loader actually reads rather than from anything we invented. Reuse `split_pose48`. Fill
the absent-hand slot with **zeros and valid=False, never NaN**: the keypoint loss multiplies by a
per-joint confidence and `NaN * 0` is `NaN`, which poisons every gradient in the batch.
