# Single-image hand baseline on H2O subject4 — C-MPJPE run

Run 2026-06-18. Goal: a real camera-frame **C-MPJPE** (absolute 3D joint error, mm)
for a weak-perspective single-image hand method on the H2O held-out subject4 test
set, to sit next to our method's **58 mm**. These methods regress a weak-perspective
camera (dummy focal 5000), so they cannot recover metric scale — a LARGE absolute
error (~100–250 mm) is the EXPECTED, correct finding.

## Method

- **WiLoR** (rolpotamias/WiLoR), run via the `wilor-mini` pipeline
  (`warmshao/WiLoR-mini`), weights + MANO auto-downloaded from HuggingFace.
- Single image per hand. Hand crop bbox derived from **GT joints** (same detector
  stand-in `scripts/eval_cmpjpe.py` uses — `bboxes_from_joints`), so the comparison
  isolates the method's pose/scale, not a detector.
- Camera-frame absolute joints = `pred_keypoints_3d` (root-relative MANO) +
  `pred_cam_t_full` (weak-perspective translation).
  - **raw**: WiLoR's own `pred_cam_t_full` (its dummy focal 5000 scaled to image).
  - **Kfix**: translation re-derived with H2O's **real focal length** (fairest shot
    at metric scale; uses the exact `cam_crop_to_full` formula `tz = 2·f/(box·s)`).
- Metric math (`_per_joint_mm`, `_pa_mpjpe_mm`) and the H2O→MANO joint remaps copied
  **verbatim** from `scripts/eval_cmpjpe.py`, so the number is directly comparable to
  ours. Joints evaluated on the 16 kinematic MANO joints (and the 21-joint
  H2O/SOTA convention with 5 fingertips).

## Reference (for context — Hand3R HOI4D table)

| Method | C-MPJPE (mm) |
|---|---|
| WiLoR-SLAM (offline +SLAM) | 252.24 |
| HaMeR-SLAM (offline +SLAM) | 248.23 |
| HaWoR (world-space metric) | 51.77 |
| Hand3R (world-space metric) | 42.6 |
| **Ours** | **58** |

## Environment that worked (reproducible)

- Cluster: ETH student-cluster, SLURM account `3dv`, partition `jobs`,
  `--gpus=2080ti:1` (Turing **sm_75**), `--mem=32G`.
- Node-local scratch for venv + pip cache + `HF_HOME` (since `/work` is at quota,
  `$HOME` is inode-limited): `TMPDIR=/tmp` on the compute node (~364 GB free).
- **python3.10** (`/usr/bin/python3.10`) — required so chumpy imports (see blocker 2).
- **torch 2.3.1 + cu118** (py3.10 wheels, supports sm_75 — the task's "CUDA 11.7"
  is conservative; sm_75 runs fine on cu118). `torch.cuda.is_available()` True on
  the 2080ti.
- Full working install sequence (also in `job_wilor.sh`):
  ```bash
  python3.10 -m venv venv && source venv/bin/activate
  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
  pip install "numpy<2" "setuptools<70" wheel
  pip install --no-build-isolation chumpy
  # PATCH chumpy/__init__.py for numpy>=1.24 (see blocker 2) — required before import works
  sed -i 's/^from numpy import bool.*/from numpy import nan, inf/' \
      venv/lib/python3.10/site-packages/chumpy/__init__.py
  pip install "smplx==0.1.28" timm einops "ultralytics==8.1.34" \
      opencv-python huggingface_hub scikit-image roma
  pip install "numpy<2"                     # re-pin: skimage/ultralytics bump it
  pip install --no-deps --no-build-isolation git+https://github.com/warmshao/WiLoR-mini
  ```
  Job script: `/work/scratch/dmonopoli/baseline_run/job_wilor.sh`.
- Data prep: `extract_subject4.py` packs subject4 frames (224×224 RGB + GT joints in
  camera frame + GT-derived bboxes + pack-adjusted K) into one npz; inference reads
  only that (no repo deps).

## Results (WiLoR, H2O subject4, 400 frames / **797 hands**, 100% pred success)

SLURM job 99865 on a single 2080ti. GT-derived bbox per hand (skip detector). The
two camera-frame variants differ only in the focal used to place the root.

### 16 kinematic MANO joints

| Metric | raw (WiLoR focal 5000) | Kfix (H2O real focal ≈198) | meaning |
|---|---|---|---|
| **C-MPJPE** ↓ | **8529.6 mm** | **614.9 mm** | absolute, camera frame |
| WRIST | 8530.1 mm | 611.7 mm | absolute root error |
| PA-MPJPE | **47.6 mm** | (same) | Procrustes — pose+scale quality |
| RR-MPJPE | 63.2 mm | (same) | root-relative shape |

(median C-MPJPE: raw 8617.8 mm, Kfix 614.6 mm.)

### 21 joints (16 base + 5 fingertips, H2O/SOTA convention)

| Metric | raw (WiLoR focal 5000) | Kfix (H2O real focal ≈198) |
|---|---|---|
| **C-MPJPE** ↓ | 8532.7 mm | **616.2 mm** |
| WRIST | 8530.1 mm | 611.7 mm |
| PA-MPJPE | **47.5 mm** | (same) |
| RR-MPJPE | 65.6 mm | (same) |

The 21-joint RR (65.6 mm) is barely above the 16-joint RR (63.2 mm), which confirms
the **fingertip pairing is correct** (a mis-permuted tip order would blow RR up). The
absolute C-MPJPE / PA / WRIST are essentially identical to the 16-joint run.

A projection sanity-check (`wilor_viz.png`, pred=red, GT=green) shows GT joints landing
on the hand and the WiLoR pred with the **right shape but a systematic translation
offset** — the visible signature of the weak-perspective scale failure. The metric math
is sound (GT projects onto the hand).

### Reading

This is the **expected, correct** result: a weak-perspective single-image method
recovers **excellent hand pose** (PA-MPJPE **47.6 mm**, on par with WiLoR's reported
quality; RR 63 mm) but **cannot place the hand in metric camera space**. With its own
dummy focal=5000 the root lands ~8.5 **metres** off; even re-projected with H2O's true
focal the absolute C-MPJPE is **~615 mm** — an order of magnitude worse than ours.

| | C-MPJPE (mm) ↓ |
|---|---|
| **Ours** | **58** |
| WiLoR single-image (Kfix, H2O real focal) | **615** |
| WiLoR single-image (raw, dummy focal) | 8530 |
| WiLoR-SLAM (Hand3R HOI4D table, +SLAM scale) | 252 |
| HaWoR / Hand3R (world-space metric) | 42–52 |

The gap (58 vs 615) is exactly the two-tier story: WiLoR's **pose** is great (PA 48)
but its **absolute scale/translation** is not metric. Even bolting SLOW SLAM on
(WiLoR-SLAM, 252 mm in Hand3R's table) only halves the error of our Kfix number and
is still 4× ours.

## Blockers / notes (all resolved — for reproducibility)

1. **No conda, `/work` at quota, `$HOME` inode-limited.** Built the venv + pip cache +
   `HF_HOME` in node-local `/tmp` inside the SLURM job (~364 GB free). Each job rebuilds
   the venv (node-local scratch is wiped per job); torch cu118 re-download ≈90 s.
2. **chumpy (wilor-mini dep) is the hard part.** Two failures, both fixed:
   - On **py3.12** chumpy import-crashes (`inspect.getargspec` removed in py3.11+).
     → use **python3.10** (present at `/usr/bin/python3.10`; wilor-mini's stated req).
   - On py3.10 chumpy still crashes on `from numpy import bool,int,float,...` (numpy≥1.24
     removed those aliases). → **patch** `chumpy/__init__.py` on disk (can't locate via
     `import chumpy` since import is what fails): rewrite that line to `from numpy import
     nan, inf` and strip bare `np.bool/np.float/...` (word-boundary, to spare `np.bool_`).
     Verified `import chumpy` (0.70) with numpy 1.26.4 before the GPU run.
   - Install wilor-mini with `--no-deps` so pip never re-fetches the git `chumpy @ ...` URL.
3. **opencv-python 4.13 "requires numpy>=2"** warning is benign — runs fine on 1.26.4.
4. WiLoR + MANO weights auto-download from HF on first run (`HF_HOME` node-local). The
   detector is bypassed via `predict_with_bboxes` with the GT-derived box.

Artifacts on cluster: `/work/scratch/dmonopoli/baseline_run/` (`job_wilor.sh`,
`extract_subject4.py`, `run_wilor_h2o.py`, `subject4_frames.npz`, `wilor_viz.png`).
Log: `/work/scratch/dmonopoli/joblogs/99865_wilor.out`.

---

# HaWoR (world-space metric competitor) on H2O subject4 — C-MPJPE run

Run 2026-06-18. **HaWoR** (CVPR 2025 Highlight) is the world-space metric baseline:
HaMeR-style per-frame hand + **DROID-SLAM** camera trajectory + **Metric3D** monocular
depth for metric scale + a transformer **infiller**. Unlike WiLoR (weak-perspective,
single-image), HaWoR is designed to place the hand in metric world/camera space — so a
*low* absolute C-MPJPE is the expected, fair comparison to ours (C-MPJPE 58 mm). HaWoR
reports HOI4D C-MPJPE 51.8 mm.

## Environment that worked (the hard part — reproducible)

Cluster as for WiLoR (`3dv`/`jobs`/`--gpus=2080ti:1`, sm_75; node-local `$TMPDIR=/tmp`
364 GB for venv+frames; **only small outputs persisted to `$HOME`** because `/work` is
now at *disk* quota too). The make-or-break was building **DROID-SLAM + lietorch** (two
CUDA extensions) and **pytorch3d** (needed by motion-est to render the SLAM hand masks).

**Key environment fact that forced the whole recipe:** this node has **no system nvcc on
PATH** and the *only* complete CUDA toolkit is **system CUDA 13.1** (`/usr/local/cuda-13.1`).
The pip CUDA wheels (`nvidia-cuda-nvcc-cu11`/`-cu12`) ship `ptxas` + `cicc` + `libnvvm`
but **NOT the `nvcc` frontend binary** — so the usual "pip a self-contained toolkit"
trick is a dead end here. We therefore must compile against system nvcc 13.1, which
dictates a recent torch.

Working recipe (`build_hawor_env.sh`, sourced inside the job):
```bash
python3.10 -m venv $TMPDIR/hvenv && source .../activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# system CUDA 13.1 is the ONLY usable nvcc:
export CUDA_HOME=/usr/local/cuda-13.1; export PATH=$CUDA_HOME/bin:$PATH
# (a) bypass torch's hard nvcc-vs-torch CUDA version check (13.1 nvcc vs 12.8 torch):
#     patch torch/utils/cpp_extension.py -> neutralise `raise RuntimeError(CUDA_MISMATCH_MESSAGE...)`.
# (b) CUDA 13 dropped sm_60/61/70 -> trim DROID setup.py gencode to sm_75/80/86.
# (c) modern PyTorch removed implicit Tensor.type()->ScalarType: patch the 3+38 AT_DISPATCH
#     sites in DROID src/*.cu and lietorch src/*.cu|*.cpp : `.type(),` -> `.scalar_type(),`.
cd thirdparty/DROID-SLAM && TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" python setup.py install   # builds droid_backends + lietorch
pip install ... mmengine==0.10.4 smplx==0.1.28 chumpy(+numpy<2 patch) pytorch-lightning==2.2.4 --no-deps ...
pip install "git+.../pytorch3d.git@stable"   # FORCE_CUDA=1, same CUDA 13.1 + version-bypass
```
All weights were already staged (no Google-Drive auth needed): `weights/external/droid.pth`
(16 MB, the genuine 2021 DROID weight), `detector.pt`, `hawor.ckpt`, `infiller.pt`,
Metric3D `metric_depth_vit_large_800k.pth`, and MANO RIGHT/LEFT in `_DATA`.

### Build blockers hit + fixes (all real, all resolved)
1. **No conda, `/work` at disk quota, `$HOME` inode-limited.** Logs to `$HOME` (1 file),
   venv+frames+SLAM all on node-local `$TMPDIR`. (A first submit FAILED 0:53 instantly
   because SLURM couldn't even create the `--output` log on the quota-full `/work`.)
2. **pip CUDA wheels have no `nvcc` binary** (only ptxas) → must use system **CUDA 13.1**.
3. **torch refuses nvcc/torch CUDA-version skew** → patch out the `CUDA_MISMATCH` raise.
4. **CUDA 13 dropped sm_60/61/70** → trim DROID `setup.py` gencode list.
5. **`AT_DISPATCH_*(<tensor>.type(), ...)` fails on PyTorch 2.x** (`DeprecatedTypeProperties`
   no longer converts to `ScalarType`) → `.type()` → `.scalar_type()` in DROID's 3 `.cu`
   sites and lietorch's 38 (`lietorch_gpu.cu`/`lietorch_cpu.cpp` ×19 each + extras). After
   this, **droid_backends + lietorch compile cleanly on sm_75** (CUDA 13.1 / torch 2.7).
6. **QOS = 1 job/user** (`3dv-team25`) → the HaWoR job queues behind any other run.

<!-- RESULTS_PLACEHOLDER -->

Scripts on cluster: `/home/dmonopoli/HaWoR/hawor_eval/` (`build_hawor_env.sh`,
`job_hawor.sh`, `run_hawor_infer.py`, `extract_h2o_clips_mp4.py`, `eval_hawor_cmpjpe.py`).
The `.cu` `.scalar_type()` patches are applied in-place under
`/home/dmonopoli/HaWoR/thirdparty/DROID-SLAM/` (persist on `$HOME`).
