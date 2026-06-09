# Polish: Strip NeoVerse, Keep 4DGS+Hand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the bundled NeoVerse video-diffusion machinery from the `FF-4DGS-Ego` repo, keeping only the team's 4D-Gaussian-Splatting + egocentric hand-reconstruction solution, and tidy the kept code — all on an isolated `polish` branch off `main`.

**Architecture:** The repo vendors DiffSynth-Studio (`diffsynth/`). The kept solution (`scripts/*`) imports only a narrow slice: `auxiliary_models.worldmirror`, `utils.auxiliary`, `data.video`, `models.utils`, and `models.ModelManager`. We delete the rest of `diffsynth/` plus root demo scripts, and **slim the `ModelManager` import hub** (4 files) so `reconstruct_4dgs.py` stays behavior-identical while the ~60-file diffusion model zoo is removed. Verification is static (py_compile + AST dangling-import check) because this host has no GPU/checkpoints.

**Tech Stack:** Python 3.10, PyTorch, git worktrees. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-09-polish-remove-neoverse-design.md`

---

## File Structure (decisions locked here)

**Kept `diffsynth/` after the cut:**
- `diffsynth/__init__.py` — trimmed to `from .data import *` + `from .models import *`
- `diffsynth/auxiliary_models/__init__.py` — drop DA3 import
- `diffsynth/auxiliary_models/worldmirror/**` — backbone + heads + utils (unchanged)
- `diffsynth/configs/model_config.py` — **rewritten** minimal (WorldMirror entry + stubs)
- `diffsynth/configs/__init__.py` — unchanged (empty)
- `diffsynth/data/{__init__,video}.py` — unchanged
- `diffsynth/utils/{__init__,auxiliary}.py` — unchanged
- `diffsynth/models/{__init__,model_manager,downloader,utils}.py` — `model_manager.py` slimmed, others unchanged

**Deleted:** root `app.py`, `inference.py`; `diffsynth/{pipelines,prompters,schedulers,tokenizer_configs,controlnets,processors,vram_management,distributed,trainers,lora,extensions}/`; `diffsynth/utils/{app,visualization_tools}.py`; all `diffsynth/models/*.py` except the 4 kept; `diffsynth/auxiliary_models/depth_anything_3/`; `examples/trajectories/`; rewrite `README.md`.

---

## Task 0: Pre-flight — create the static verifier

**Files:**
- Create: `/tmp/verify_polish.py` (dev tool, not committed)

- [ ] **Step 1: Write the verifier**

This is the gate used after every deletion task. It (a) `py_compile`s every kept `.py`
file and (b) AST-parses every `diffsynth.*` import in the kept tree and flags any
that resolve to a missing file (a dangling reference into a deleted module).

```python
# /tmp/verify_polish.py
import ast, os, sys, py_compile

ROOT = "/Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish"
PKG = "diffsynth"
PKG_DIR = os.path.join(ROOT, PKG)

def kept_py_files():
    out = []
    for base in (PKG_DIR, os.path.join(ROOT, "scripts")):
        for dp, _d, fs in os.walk(base):
            if "__pycache__" in dp:
                continue
            for f in fs:
                if f.endswith(".py"):
                    out.append(os.path.join(dp, f))
    return out

def resolve(mod):
    rel = mod.replace(".", os.sep)
    for p in (os.path.join(ROOT, rel + ".py"), os.path.join(ROOT, rel, "__init__.py")):
        if os.path.isfile(p):
            return True
    # maybe module.symbol -> resolve parent module
    parent = ".".join(mod.split(".")[:-1])
    if parent:
        rel = parent.replace(".", os.sep)
        for p in (os.path.join(ROOT, rel + ".py"), os.path.join(ROOT, rel, "__init__.py")):
            if os.path.isfile(p):
                return True
    return False

def diffsynth_imports(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    cur = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
    parent = cur if path.endswith("__init__.py") else ".".join(cur.split(".")[:-1])
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] == PKG:
                    out.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                bp = parent.split(".")
                if n.level > 1:
                    bp = bp[: -(n.level - 1)]
                base = ".".join(bp)
                mod = f"{base}.{n.module}" if n.module else base
            else:
                mod = n.module or ""
            if mod.split(".")[0] == PKG:
                out.add(mod)
    return out

def main():
    files = kept_py_files()
    compile_errs, dangling = [], []
    for f in files:
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            compile_errs.append(f"{f}: {e}")
        try:
            for mod in diffsynth_imports(f):
                if not resolve(mod):
                    dangling.append(f"{os.path.relpath(f, ROOT)} -> {mod}")
        except SyntaxError as e:
            compile_errs.append(f"{f}: {e}")
    print(f"Checked {len(files)} kept .py files")
    if compile_errs:
        print(f"\nCOMPILE ERRORS ({len(compile_errs)}):")
        for e in compile_errs: print("  ", e)
    if dangling:
        print(f"\nDANGLING diffsynth IMPORTS ({len(dangling)}):")
        for d in dangling: print("  ", d)
    if compile_errs or dangling:
        sys.exit(1)
    print("\nVERIFY OK: no compile errors, no dangling diffsynth imports.")

main()
```

- [ ] **Step 2: Confirm no dynamic/string-based model loading in the kept tree**

Run:
```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
grep -rnE "importlib|__import__" scripts/ diffsynth/auxiliary_models/worldmirror/ diffsynth/models/{model_manager,downloader,utils}.py diffsynth/utils/ diffsynth/data/ 2>/dev/null
```
Expected: only benign matches (none that build a deleted `diffsynth.pipelines/models.*` path from a string). The `huggingface_model_loader_configs` string-redirects live in `model_config.py`, which we rewrite in Task 3. If a kept file dynamically imports a to-be-deleted module, STOP and reassess.

- [ ] **Step 3: Baseline the verifier on the un-cut tree**

Run: `python3 /tmp/verify_polish.py`
Expected: `VERIFY OK` (the current tree compiles and has no dangling imports). If it already fails, fix the verifier before proceeding — do not start deleting against a red baseline.

---

## Task 1: Delete root NeoVerse entry points

**Files:**
- Delete: `app.py`, `inference.py`

- [ ] **Step 1: Remove the files**

```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
git rm app.py inference.py
```

- [ ] **Step 2: Confirm nothing kept references them**

Run:
```bash
grep -rnE "^\s*(import|from)\s+(app|inference)\b" scripts/ diffsynth/ 2>/dev/null
```
Expected: no output.

- [ ] **Step 3: Verify**

Run: `python3 /tmp/verify_polish.py`
Expected: `VERIFY OK`.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove NeoVerse demo entry points (app.py, inference.py)"
```

---

## Task 2: Trim `diffsynth/__init__.py` and delete pure-diffusion subtrees

**Files:**
- Modify: `diffsynth/__init__.py`
- Delete: `diffsynth/{pipelines,prompters,schedulers,tokenizer_configs,controlnets,processors,vram_management,distributed,trainers,lora,extensions}/`, `diffsynth/utils/{app,visualization_tools}.py`

- [ ] **Step 1: Trim the package init FIRST (so `import diffsynth` never references deleted subpackages)**

Replace the entire contents of `diffsynth/__init__.py` with:

```python
from .data import *
from .models import *
```

- [ ] **Step 2: Delete the diffusion subtrees**

```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
git rm -r diffsynth/pipelines diffsynth/prompters diffsynth/schedulers \
          diffsynth/tokenizer_configs diffsynth/controlnets diffsynth/processors \
          diffsynth/vram_management diffsynth/distributed diffsynth/trainers \
          diffsynth/lora diffsynth/extensions
git rm diffsynth/utils/app.py diffsynth/utils/visualization_tools.py
```

- [ ] **Step 3: Verify**

Run: `python3 /tmp/verify_polish.py`
Expected: `VERIFY OK`. (If dangling imports appear pointing into `diffsynth/models/*` zoo files, that is expected and resolved in Task 3 — but at THIS point the only remaining references to the zoo are inside `diffsynth/models/` and `diffsynth/configs/model_config.py`, which are still present, so the verifier should still pass. If it flags `diffsynth.lora` / `diffsynth.pipelines` from a kept file, STOP — a kept file unexpectedly depends on a deleted subtree.)

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove NeoVerse video-diffusion subpackages (pipelines, prompters, schedulers, etc.)"
```

---

## Task 3: Slim the `ModelManager` hub and delete the model zoo + DA3

**Files:**
- Rewrite: `diffsynth/configs/model_config.py`
- Modify: `diffsynth/models/model_manager.py`, `diffsynth/auxiliary_models/__init__.py`
- Delete: all `diffsynth/models/*.py` except `__init__.py`, `model_manager.py`, `downloader.py`, `utils.py`; `diffsynth/auxiliary_models/depth_anything_3/`

- [ ] **Step 1: Rewrite `diffsynth/configs/model_config.py` to the minimal surface**

`model_manager.py` imports exactly `model_loader_configs, huggingface_model_loader_configs, patch_model_loader_configs`; `downloader.py` imports `preset_models_on_huggingface, preset_models_on_modelscope, Preset_model_id`. The preset dicts are pure-string data and are only consulted when loading by preset id (we load by explicit path), so they can be emptied. Replace the **entire** file with:

```python
from typing_extensions import Literal, TypeAlias

from ..auxiliary_models import WorldMirror

# Single-file loader detection table.
# Format: (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
# Only the WorldMirror reconstructor remains after the NeoVerse diffusion zoo was removed.
model_loader_configs = [
    (None, "1a1d001a35f78f3a7796a1e719ead340", ["reconstructor"], [WorldMirror], "civitai"),
]

# HuggingFace-format and patch loaders are unused by the reconstructor path.
huggingface_model_loader_configs = []
patch_model_loader_configs = []

# Preset download tables (consumed by downloader.py). Emptied: checkpoints are
# loaded by explicit --reconstructor_path, not by preset id.
preset_models_on_huggingface = {}
preset_models_on_modelscope = {}

Preset_model_id: TypeAlias = Literal["reconstructor"]
```

- [ ] **Step 2: Slim `diffsynth/models/model_manager.py`**

Remove the zoo convenience-imports (every `from .<zoo_module> import ...` line in the
header block — i.e. all of them EXCEPT `from .downloader import ...`, `from .utils import ...`,
and `from ..configs.model_config import ...`) and the `from .lora import get_lora_loaders`
line. Then remove the now-orphaned `load_lora` method.

Concretely:
1. In the import header (roughly lines 6–50), delete every `from .X import …` line where `X` is a deleted zoo module (`sd_*`, `sdxl_*`, `sd3_*`, `svd_*`, `hunyuan_*`, `flux_*`, `cog_*`, `omnigen`, `nexus_gen*`, `step*`, `wan_video_*`, `*_controlnet`, `*_ipadapter`, `*_motion`) **and** the line `from .lora import get_lora_loaders`. Keep:
   - `from .downloader import download_models, download_customized_models, Preset_model_id, Preset_model_website`
   - `from ..configs.model_config import model_loader_configs, huggingface_model_loader_configs, patch_model_loader_configs`
   - `from .utils import load_state_dict, init_weights_on_device, hash_state_dict_keys, split_state_dict_with_prefix`
   - any non-diffsynth imports (torch, os, etc.)
2. Delete the `load_lora` method (the block beginning `def load_lora(self, file_path="", state_dict={}, lora_alpha=1.0):` and its body, ending just before the next `def`). It is the only consumer of `get_lora_loaders` and is unused by the reconstructor path.

- [ ] **Step 3: Confirm the slimmed `model_manager.py` references no deleted class**

Run:
```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
grep -nwE "SDUNet|SDXLUNet|SD3DiT|SVDUNet|HunyuanDiT|FluxDiT|CogDiT|WanModel|OmniGenTransformer|NeoVerseControlBranch|get_lora_loaders|load_lora" diffsynth/models/model_manager.py
```
Expected: no output. If `load_lora`/`get_lora_loaders` still appears, the method/import was not fully removed.

- [ ] **Step 4: Drop the DA3 import from `diffsynth/auxiliary_models/__init__.py`**

Replace its contents with:

```python
from .worldmirror.models.models.worldmirror import WorldMirror
```

- [ ] **Step 5: Delete the model zoo and DA3**

```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
# delete every models/*.py except the 4 kept
find diffsynth/models -maxdepth 1 -name '*.py' \
  ! -name '__init__.py' ! -name 'model_manager.py' \
  ! -name 'downloader.py' ! -name 'utils.py' -print -exec git rm {} +
git rm -r diffsynth/auxiliary_models/depth_anything_3
```

- [ ] **Step 6: Verify (static gate)**

Run: `python3 /tmp/verify_polish.py`
Expected: `VERIFY OK`, with zero dangling imports — this proves no kept file references the deleted zoo / DA3 / lora.

- [ ] **Step 7: Best-effort runtime import check (may be skipped if deps missing)**

Run:
```bash
python3 -c "from diffsynth.models import ModelManager; print('ModelManager import OK')" 2>&1 | tail -5
```
Expected: `ModelManager import OK`. If it fails with `ModuleNotFoundError: modelscope` (or another *external* dep absent on this host), that is an environment limitation — record it and rely on the static gate. If it fails with an error referencing a **deleted diffsynth module**, STOP and fix.

- [ ] **Step 8: Commit**

```bash
git commit -m "refactor: slim ModelManager to the reconstructor; delete diffusion model zoo + DA3"
```

---

## Task 4: Clean examples, requirements, and run-scripts

**Files:**
- Delete: `examples/trajectories/`
- Modify: `requirements.txt`, `batch.sh`, `train_gb10.sh`, `StudentCluster.md`, `docs/trajectory_format.md`

- [ ] **Step 1: Remove NeoVerse trajectory inputs (keep example videos)**

```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
git rm -r examples/trajectories
```

- [ ] **Step 2: Evidence-based requirements pruning**

For each diffusion-only candidate, confirm it is unused by the kept tree, then remove it:
```bash
for dep in gradio deepspeed peft accelerate ftfy; do
  echo "== $dep =="
  grep -rniE "\b(import|from)\s+$dep\b" scripts/ diffsynth/ 2>/dev/null | head
done
```
Remove from `requirements.txt` only the candidates with **no** import hits (expected: all five — `gradio` was app.py, `deepspeed`/`peft`/`accelerate` were `diffsynth/trainers`, `ftfy` was prompt cleaning). Delete the corresponding lines and the now-empty `# Demo` section header if `trimesh`/`viser` are its only survivors. Leave everything else untouched (conservative — a slightly fat requirements file is safer than a broken env).

- [ ] **Step 3: Scrub NeoVerse references in run-scripts and docs**

```bash
grep -rniE "neoverse|inference\.py|app\.py|wan|diffusion|trajector" \
  batch.sh train_gb10.sh StudentCluster.md docs/trajectory_format.md docs/coordinate_system.md 2>/dev/null
```
For each hit: if the line is purely about NeoVerse video generation (e.g. a `python inference.py …` invocation, a `models/NeoVerse` diffusion-weights download, a trajectory-format section), remove or rewrite it to reference the kept workflow (`scripts/reconstruct_4dgs.py`, `scripts/train_hand_head.py`). If `docs/trajectory_format.md` documents *only* the NeoVerse novel-trajectory format, `git rm` it. Keep `venv`/cluster/SLURM boilerplate that is environment setup, not NeoVerse.

- [ ] **Step 4: Verify**

Run: `python3 /tmp/verify_polish.py`
Expected: `VERIFY OK` (no code touched, but confirms tree still clean).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: drop NeoVerse trajectory examples, prune diffusion deps, scrub run-script refs"
```

---

## Task 5: Rewrite README for FF-4DGS-Ego

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Replace `README.md`** with a project-accurate document. Draft:

```markdown
# FF-4DGS-Ego

Feed-forward **4D Gaussian-Splatting reconstruction with egocentric hand
recovery**. A WorldMirror reconstructor backbone predicts per-frame 4D Gaussians
and camera parameters from monocular video; a HaMeR/MANO hand head, coupled to
the reconstructed scene, recovers metric-scale 3D hands.

> This repository began as a fork of DiffSynth-Studio / NeoVerse. The video
> generation (diffusion) stack has been removed; only the reconstruction +
> hand-recovery solution remains.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git submodule update --init --recursive   # HaMeR
```

Place model weights under `models/` (reconstructor checkpoint, MANO, HaMeR) —
see `models/Put checkpoints here.txt`.

## Reconstruct 4D Gaussians from a video

```bash
python -m scripts.reconstruct_4dgs --input_path examples/videos/robot.mp4 --render_video
```

Outputs (to `outputs/reconstruction/`): `gaussians.pt`, `camera_params.json`,
`gaussians_frame0000.ply`, and an optional `render.mp4`.

## Train the hand head

```bash
python -m scripts.train_hand_head --config configs/train_hand_head.yaml
```

Configs for ablations live in `configs/`. SLURM/cluster launch examples:
`batch.sh`, `train_gb10.sh`.

## Evaluation

```bash
python -m scripts.eval_hand_head   --config configs/train_hand_head.yaml
python -m scripts.eval_metric_scale --help
python -m scripts.eval_reconstruction --help
```

## Repository layout

| Path | Purpose |
|------|---------|
| `scripts/` | Reconstruction, hand-head training, evaluation, visualization |
| `diffsynth/auxiliary_models/worldmirror/` | Reconstructor backbone + heads (camera, dense, HaMeR, GS injection) |
| `diffsynth/models/` | `ModelManager` checkpoint loader (reconstructor only) |
| `diffsynth/{utils,data}/` | Video I/O and geometry helpers |
| `configs/` | Training / ablation configs |
| `models/` | Checkpoints (reconstructor, MANO, HaMeR submodule) |

## License

See `LICENSE.txt`.
```

Adjust any command flag that does not match the actual `argparse` in the
referenced script (cross-check `--help` for each before finalizing).

- [ ] **Step 2: Commit**

```bash
git add README.md docs/
git commit -m "docs: rewrite README for FF-4DGS-Ego (reconstruction + hand recovery)"
```

---

## Task 6: Tidy the kept scripts (`/simplify`)

**Files:**
- Modify: `scripts/*.py` (reuse/dedupe/clarity only — **no behavior change**)

- [ ] **Step 1: Fix any now-dead imports across the kept tree**

```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
grep -rnE "neoverse|wan_video|pipelines|prompters|schedulers" scripts/ 2>/dev/null
```
Remove/repair any remaining import or reference to deleted modules. (Expected: none, since the verifier already passed — this is a human-readable double check.)

- [ ] **Step 2: Run the `/simplify` skill over `scripts/`**

Invoke the `simplify` skill scoped to the kept scripts. It targets reuse,
deduplication, efficiency, and clarity — **quality only, no bug-hunting, no
behavior change**. Review its diff against the spec's "no behavior change"
constraint before applying.

- [ ] **Step 3: Verify**

Run: `python3 /tmp/verify_polish.py`
Expected: `VERIFY OK`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: tidy kept scripts (reuse/dedupe/clarity, no behavior change)"
```

---

## Task 7: Final verification and hand-off

- [ ] **Step 1: Full static gate**

Run: `python3 /tmp/verify_polish.py`
Expected: `VERIFY OK`.

- [ ] **Step 2: Confirm no NeoVerse/diffusion references survive anywhere kept**

```bash
cd /Users/mondra/git/3dv/FF-4DGS-Ego/.claude/worktrees/polish
grep -rniE "neoverse" --include='*.py' --include='*.md' --include='*.sh' --include='*.yaml' . | grep -v docs/superpowers/
```
Expected: empty (or only intentional historical mentions in the rewritten README's fork note).

- [ ] **Step 3: Summarize the diff**

```bash
git log --oneline main..HEAD
git diff --stat main..HEAD | tail -5
```

- [ ] **Step 4: Produce the GPU smoke-test checklist for the user**

These CANNOT run on this host (no GPU/checkpoints). The user must run them in the
training env before merging `polish`:
1. `python -m scripts.reconstruct_4dgs --input_path examples/videos/robot.mp4`
2. `python -m scripts.train_hand_head --config configs/train_hand_head.yaml` (a few steps)
3. `python -m scripts.eval_hand_head --config configs/train_hand_head.yaml`

---

## Self-Review

**Spec coverage:**
- Delete root entry points → Task 1 ✓
- Delete diffusion subtrees → Task 2 ✓
- Slim ModelManager + delete zoo + DA3 → Task 3 ✓
- examples/trajectories, requirements, run-script refs → Task 4 ✓
- README rewrite → Task 5 ✓
- Tidy (`/simplify`) → Task 6 ✓
- Static verification + GPU smoke-test hand-off → Task 0/7 ✓
- "Do not touch other agent's tree / hamer submodule" → enforced by worktree isolation (no task modifies them) ✓

**Placeholder scan:** No TBD/TODO; every code/edit step shows concrete content or an exact command + expected output. The two judgement steps (4.3 run-script scrub, 6.2 `/simplify`) specify the decision rule rather than a fixed diff because their output depends on file contents — acceptable and bounded.

**Type/name consistency:** `model_config.py` exports exactly the six names its two importers (`model_manager.py`, `downloader.py`) consume: `model_loader_configs`, `huggingface_model_loader_configs`, `patch_model_loader_configs`, `preset_models_on_huggingface`, `preset_models_on_modelscope`, `Preset_model_id`. `Preset_model_website` stays defined in `downloader.py` (unchanged). Kept `diffsynth/models/` set (`__init__`, `model_manager`, `downloader`, `utils`) is consistent across Tasks 2–3.
