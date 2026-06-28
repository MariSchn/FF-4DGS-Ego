# B1 result — does a metric-depth FM recover depth AT THE HAND as well as our anchor?

**Question.** Can an off-the-shelf metric-depth foundation model (UniDepth-V2,
ViT-L/14) recover *metric* depth **at the hand** as well as our in-scene hand
anchor (residual 4.5 cm, HOT3D)? If yes, the contribution is weakened; if no, the
anchor is the better metric source and the contribution holds.

## Headline number

| Depth source @ hand | mean | median | p90 | N joints |
|---|---|---|---|---|
| **UniDepth-V2 (ViT-L/14, metric)** | **15.67 cm** | **8.14 cm** | **37.74 cm** | 6311 |
| **Our hand anchor (ours)** | **4.52 cm** | **4.16 cm** | **8.92 cm** | 6311 |
| **ratio (FM / ours)** | **3.47×** | — | — | — |

Same 6311 GT hand-joint locations for both columns — directly comparable ("how far
off is this depth source AT the hand?"). Evaluated over 60 clips × 16 frames from
`b1_bundle.pt` (resolution 224×224, checkpoint `p2_warmstart/best_mpjpe.pt`).

## Verdict: HOLDS

Decision rule (stated up front):
- UniDepth hand-depth error **>~10 cm → anchor wins, contribution holds.**
- <~5 cm → FM matches, contribution weakened.
- in between → report exact numbers.

UniDepth-V2's **mean hand-depth error is 15.7 cm (3.5× our 4.5 cm)** — comfortably
above the 10 cm threshold. Even on the median (8.1 cm) UniDepth is ~2× worse than
our anchor's median (4.2 cm), and its tail is far heavier (p90 37.7 cm vs 8.9 cm).
A generic metric-depth foundation model **cannot replace the in-scene hand anchor**:
it is metric-plausible globally but unreliable precisely at the hand, where we need
it. The contribution holds.

Nuance worth a sentence in the paper: the gap is largest in the tail (p90 4.2×),
i.e. UniDepth is occasionally roughly right at the hand but frequently very wrong;
the anchor is consistently tight. Mean (3.5×) and p90 (4.2×) tell the cleanest story;
median (2.0×) is the conservative lower bound.

## How it was run (env recipe)

- **Job:** SLURM `99969` `ff4dgs-b1unidepth`, partition `jobs`, account `3dv`,
  `--gpus=2080ti:1` (sm_75), 32 G, node `studgpu-node13`. **COMPLETED in 2:36.**
  (Prior attempt `99963` was cancelled by a QOS shuffle; `99967` then FAILED at the
  import — missing `matplotlib`; `99969` is the fixed re-run.)
- **Script:** `scripts/b1_unidepth_2080ti.sh` → builds a **node-local** (`/tmp`)
  cu118 venv and runs `scripts/eval_unidepth_b1.py`. Everything node-local because
  `$HOME` is inode-limited and `/work` is at disk quota; logs + JSON go to
  `/home/dmonopoli/b1_logs/` (2 tiny files).
- **Env:** `python3.10` venv, `torch==2.3.1+cu118` / `torchvision==0.18.1` (clean
  pip, no custom CUDA build — UniDepthV2 has no custom CUDA op, so it works on
  sm_75; torch ≤ 2.6 required, it broke on Blackwell torch ≥ 2.7).
- **UniDepth deps:** `numpy<2 einops timm huggingface-hub pillow>=10.2.0 scipy
  opencv-python-headless tqdm matplotlib wandb h5py tabulate termcolor trimesh
  imageio`, then `git clone lpiccinelli-eth/UniDepth` + `pip install --no-deps -e`.
- **Weights:** `lpiccinelli/unidepth-v2-vitl14` (HF, downloaded at runtime).
- **Eval:** `UniDepthV2.infer(img, K)` per frame → metric depth map → bilinearly
  sampled at GT hand pixels → abs error vs trusted metric `hand_z`; `gs_at_hand`
  gives our anchor's error at the same pixels.
- **Result JSON:** `/home/dmonopoli/b1_logs/b1_unidepth.json` (on cluster).

### The one blocker, fixed
`99967` FAILED at MILESTONE-1:
```
File ".../unidepth/utils/visualization.py", line 8, in <module>
    import matplotlib.pyplot as plt
ModuleNotFoundError: No module named 'matplotlib'
```
`unidepth.utils.__init__` unconditionally imports `visualization.py`, which imports
`matplotlib` and `wandb` at module load — pulled in by `from unidepth.models import
UniDepthV2` even though we never train/log. Fix: add `matplotlib wandb` (+ other
pure-python utils deps) to the pip line. After the fix, MILESTONE-1 passed
(`depth map: shape=(224,224) min=0.417 max=2.379 median=1.366 m`) and the full eval
completed cleanly. Non-fatal warnings only: xFormers / NystromAttention /
EdgeGuidedLocalSSI not compiled → slower non-CUDA-optimized path (still produced the
60-clip eval in ~30 s).

## Metric3D
Not run. UniDepth-V2 alone is decisive (3.5× → HOLDS); a second FM is not needed to
clear the decision rule. Can be added later as a robustness check if a reviewer asks.
