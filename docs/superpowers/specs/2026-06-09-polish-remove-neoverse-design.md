# Design: Strip NeoVerse video-diffusion, keep the 4DGS + hand solution

**Date:** 2026-06-09
**Branch:** `polish` (worktree at `.claude/worktrees/polish`, based off `origin/main` @ `a8a1821`)
**Status:** Approved (design), pending spec review

## Problem

This repo (`FF-4DGS-Ego`) is a fork of **NeoVerse** — a video-diffusion 4D world
model built on DiffSynth-Studio. The team's actual contribution is a **4D
Gaussian-Splatting + egocentric hand-reconstruction** solution living in
`scripts/`, which reuses only a narrow slice of the vendored `diffsynth/`
library (the WorldMirror reconstructor + a few utils).

The bundled NeoVerse video-diffusion machinery (generation pipelines, the full
diffusion model zoo, prompters, schedulers, the Gradio demo) is dead weight: it
is never imported by the team's solution and clutters the repo. Goal: delete it,
keep only the final solution, and tidy the kept code — without disturbing a
second agent's in-flight work on `feat/hand-scene-metric-coupling`.

## Isolation strategy

The primary working tree (`feat/hand-scene-metric-coupling`) holds another
agent's **uncommitted** work (modified `scripts/*`, untracked `report/`,
`poster_assets/`, `configs/exp_p1*`, `tests/test_p1_losses.py`). All work for
this task happens in an **isolated git worktree** on the new `polish` branch,
based off `origin/main`. The other working tree is never touched. The
`models/hamer` git submodule is never modified.

## Keep / Delete boundary

Grounded in a static import-closure trace (pure-AST, no execution) seeded from
every `scripts/*.py` entry point. Baseline: 281 `diffsynth/**/*.py` files.

### DELETE — NeoVerse video-diffusion

**Repo root**
- `app.py` — Gradio NeoVerse demo
- `inference.py` — novel-trajectory video generation entry point

**`diffsynth/` subtrees (entirely)**
- `pipelines/` — all ~20 image/video generation pipelines (`wan_video*`, `sd*`,
  `sdxl*`, `flux*`, `hunyuan*`, `cog_video`, `svd_video`, `step_video`,
  `dancer`, `pipeline_runner`, `base`, …)
- `prompters/`, `schedulers/`, `tokenizer_configs/`, `controlnets/`,
  `processors/`, `vram_management/`, `distributed/`, `trainers/`, `lora/`
- `extensions/` — FastBlend, ImageQualityMetric, ESRGAN, RIFE (diffusion
  pre/post-processing + quality metrics)
- `auxiliary_models/depth_anything_3/` — the **alternative** DA3 reconstructor.
  The team's solution uses **WorldMirror** (confirmed: every script imports
  `worldmirror`, none import `depth_anything_3`). **Confirmed delete.**
- `utils/app.py`, `utils/visualization_tools.py` — demo / Gradio helpers

**`diffsynth/models/` — the diffusion model zoo (~60 files)**
- All `*_dit.py`, `*_vae*.py`, `*_unet.py`, `*_text_encoder.py`,
  `*_controlnet.py`, `*_ipadapter.py`, `*_motion.py`, `svd_*`, `wan_video_*`
  (incl. `wan_video_neoverse_controller.py`), `omnigen*`, `nexus_gen*`,
  `step1x_connector`, `cog_*`, `hunyuan_*`, `flux_*`, `sd*`, `sdxl*`, `tiler`,
  `lora`, `downloader` (if unused after slimming)

**Other**
- `README.md` → **rewritten** for FF-4DGS-Ego
- `examples/trajectories/*` — NeoVerse trajectory inputs (`*.json`, `*.npz`).
  Keep `examples/videos/*` as reconstruction test inputs. **Confirmed.**
- `requirements.txt` — prune diffusion-only deps (gradio, deepspeed, peft, …)
- NeoVerse references in `batch.sh`, `train_gb10.sh`, `StudentCluster.md`,
  `docs/trajectory_format.md` — clean or remove

### KEEP — the final solution

- **All `scripts/*.py`** — `train_hand_head`, `reconstruct_4dgs`, `eval_*`,
  `gs_metrics`, `hamer_losses`, `hand_depth_anchor_loss`, `hand_metrics`,
  `hand_vis_utils`, `view_4dgs`, `visualize_hand_bboxes`,
  `diagnose_crop_vs_fullframe`, `test_hand_to_gs_injection_placement`
- `diffsynth/auxiliary_models/worldmirror/` — backbone + heads + the utils
  actually reached by the closure
- `diffsynth/utils/auxiliary.py` + `utils/__init__.py` (no diffusion deps)
- `diffsynth/data/video.py` + `data/__init__.py`
- `diffsynth/models/utils.py` — `hash_state_dict_keys` (needed by worldmirror)
- `diffsynth/configs/model_config.py` — **slimmed** (see below)
- `diffsynth/models/model_manager.py`, `models/__init__.py` — **slimmed**
- `configs/` (hand-training / ablation), `models/` (MANO, hamer submodule,
  checkpoint placeholder), `docs/` (coordinate_system, etc.)

## The `ModelManager` chokepoint (the one delicate change)

`scripts/reconstruct_4dgs.py` is the **only** kept file that touches
`ModelManager`, and only to load the reconstructor:

```python
model_manager = ModelManager()
model_manager.load_model(reconstructor_path, device=..., torch_dtype=torch.bfloat16)
reconstructor = model_manager.fetch_model("reconstructor")
```

`ModelManager` is generic, data-driven machinery: `model_config.py` holds a
`hash → model_class` loader table (the reconstructor entry maps to
`WorldMirror`), and `model_manager.py` top-imports the entire zoo only for
re-export convenience. So instead of rewriting the loader in
`reconstruct_4dgs.py` (risky — must replicate checkpoint detection), we
**slim the hub** and keep `reconstruct_4dgs.py` byte-for-byte:

1. `diffsynth/configs/model_config.py` → keep only the `WorldMirror`
   reconstructor loader entry; drop the DA3 entry and all diffusion-model
   imports.
2. `diffsynth/models/model_manager.py` → strip the top-level zoo imports
   (lines importing `SDUNet`, `SDXLUNet`, `*DiT`, `*VAE*`, etc.); keep the
   generic `ModelManager` machinery + whatever minimal imports remain
   referenced (`downloader`, `utils`).
3. `diffsynth/models/__init__.py` (`from .model_manager import *`) and
   `diffsynth/auxiliary_models/__init__.py` (drop `DepthAnything3Reconstructor`)
   → trim to the reconstructor.

Net effect: `reconstruct_4dgs.py` stays behavior-identical, and the ~60-file
diffusion zoo moves from "kept-by-closure" to deleted.

## Tidy pass (kept code)

After deletions:
1. Fix any now-dead imports/references to deleted modules across the kept tree.
2. Prune `requirements.txt` to what the kept solution actually needs.
3. Clean NeoVerse references in `batch.sh` / `train_gb10.sh` / `StudentCluster.md`.
4. Run `/simplify` over `scripts/*` (reuse, dedupe, clarity) — **no behavior change**.

## Verification

Constraint: this dev host (macOS) has `torch`/`numpy`/`pytest`/`yaml` but **no
GPU, no `decord`/`gsplat`/CUDA, and no model checkpoints**. There is also **no
committed test suite on `main`** (the feat-branch `tests/` is untracked).

Therefore verification is **static**:
- Re-run the import-closure tracer → assert **zero** kept-file imports resolve
  into any deleted module (no dangling references).
- `python3 -m py_compile` across the entire kept tree → syntactic integrity.
- Targeted `python -c "import …"` for kept modules whose deps are installable
  here (e.g. `diffsynth.utils.auxiliary`, `diffsynth.data`,
  `diffsynth.models` after slimming) to prove the slimmed `ModelManager`
  imports cleanly.

**Cannot verify here (flag for the user to smoke-test in the GPU env before merge):**
- `python -m scripts.reconstruct_4dgs --input_path examples/videos/robot.mp4`
- `python -m scripts.train_hand_head --config configs/train_hand_head.yaml`

## Scope / non-goals

- No refactor of kept logic beyond `/simplify` (no file-splitting of
  `train_hand_head.py` — explicitly out of scope per chosen "tidy" level).
- No changes to the `models/hamer` submodule.
- No changes to the other agent's working tree or branch.
- Does not download checkpoints or run training/reconstruction.

## Risks

- **Slimming `model_manager.py`/`model_config.py` too aggressively** could drop
  an import the reconstructor path still needs. Mitigation: import-check the
  slimmed `diffsynth.models` + `fetch_model` code path statically; keep the
  diff minimal and reviewable.
- **Hidden dynamic imports** (string-based model loading) not caught by AST.
  Mitigation: grep for `importlib`, `__import__`, and string module paths in
  the kept tree before finalizing.
