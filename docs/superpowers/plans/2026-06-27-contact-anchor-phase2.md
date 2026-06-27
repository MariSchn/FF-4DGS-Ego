# Contact Anchor Phase 2 (contact-gated root depth) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the contact anchor's wide `|d_scene - wrist_z| < band_m` proxy gate with an explicit contact signal, so the anchor pulls the predicted hand root toward the scene depth ONLY where the scene depth is reliable (within 5 cm of true contact), and prove it lowers camera-frame C-abs on HOI4D.

**Architecture:** A standalone offline pass caches a per-frame boolean contact signal (`|wrist_z_GT - GT_dense_depth_at_wrist| < 5 cm`) next to the existing HOI4D hand caches. `RootDepthRefine.forward` and `apply_root_anchor` gain an optional external `contact` gate that, when supplied, replaces the `band_m` proxy. Training and the camera-frame eval thread the cached contact mask through to both the gate and the anchor loss. An isolation sbatch (control / proxy-band / contact-gated, × kp3d_abs on/off) at the stable 120-step operating point measures C-abs.

**Tech Stack:** PyTorch, the existing `diffsynth...hand_depth_sampling` (`project_joints_to_norm_pixels`, `sample_depth_at_joints`), pytest (pure-logic tests run on the x86 login/dev machine — never import `venv_gb10` there), SLURM/gb10 for training.

**Evidence motivating this plan (do not re-derive):** the scale-source contact-stratified test (commit 291931d, job 101529) found `s_contact=1.005` (n=2614) vs `s_noncontact=0.804` (n=13086) vs `oracle=1.022` — the hand recovers the scene's metric scale almost perfectly AT contact, biased low in free space. The current gate `band_m=0.5` (50 cm, `configs/exp_p4_contact_hoi4d.yaml`) fires across both, so it pulls toward biased free-space depth. The 800-step p4big arms diverged (C-abs 92→366); the 120-step / grad_accum=1 smoke is the stable operating point.

**Non-circular note:** contact is defined by GT only (GT wrist depth vs GT dense sensor depth). The anchor still corrects the PREDICTED hand. The eval's contact-gated arm is an honest **oracle-gate upper bound** on the mechanism; the proxy-band arm is the deployable comparison. State this framing in the paper.

---

## File Structure

- Create `scripts/contact_mask.py` — pure contact predicate + a sampling wrapper. One responsibility: decide per-hand contact from GT wrist depth + GT dense depth.
- Create `scripts/build_contact_cache.py` — offline pass: per seq, load `gt_joints_cache_cam_v2.pt` + `cam_intrinsics.pt` + `raw_depth/`, write `hand_data/contact_cache.pt` (`[N,2]` bool). Reuses `scripts/contact_mask.py`.
- Modify `diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py` — `forward` accepts optional `contact`.
- Modify `scripts/root_depth_anchor.py` — `apply_root_anchor` accepts optional `contact_mask`.
- Modify `scripts/train_hand_head.py` — load `contact_cache`, pass to `apply_root_anchor`, gate `root_anchor_loss` by contact.
- Modify `scripts/eval_hand_cam_anchor.py` (+ its `predict_clip`) — load `contact_cache`, `--contact_gate {proxy,oracle,off}`.
- Create `run_hoi4d_contact_phase2_gb10.sbatch` — the isolation experiment.
- Create `tests/test_contact_gate.py` — pure-logic unit tests (Tasks 1, 3, 4).

---

## Task 1: Contact predicate + sampling wrapper

**Files:**
- Create: `scripts/contact_mask.py`
- Test: `tests/test_contact_gate.py`

- [ ] **Step 1: Write the failing test for the pure predicate**

```python
# tests/test_contact_gate.py
import torch
from scripts.contact_mask import is_contact

def test_is_contact_true_when_within_threshold():
    wrist_z   = torch.tensor([[[0.40, 0.40]]])   # [B=1,S=1,2]
    dense_at  = torch.tensor([[[0.42, 0.99]]])   # hand0 2cm off surface, hand1 59cm off
    in_frame  = torch.tensor([[[True, True]]])
    m = is_contact(wrist_z, dense_at, in_frame, thresh_m=0.05)
    assert m.tolist() == [[[True, False]]]

def test_is_contact_false_when_out_of_frame_or_no_depth():
    wrist_z  = torch.tensor([[[0.40, 0.40]]])
    dense_at = torch.tensor([[[0.40, 0.00]]])    # hand1 has no valid depth (0)
    in_frame = torch.tensor([[[False, True]]])   # hand0 out of frame
    m = is_contact(wrist_z, dense_at, in_frame, thresh_m=0.05)
    assert m.tolist() == [[[False, False]]]
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/test_contact_gate.py -k is_contact -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.contact_mask'`

- [ ] **Step 3: Implement the pure predicate + the sampling wrapper**

```python
# scripts/contact_mask.py
"""Per-hand contact signal: the GT wrist sits ON the visible GT surface.
Used to gate the contact anchor where the scene depth is reliable (the
scale-source test showed s_contact=1.005 within 5cm of contact, biased
0.804 elsewhere). GT-only -> non-circular w.r.t. the predicted hand."""
import torch
from scripts.root_depth_anchor import (  # reuse the same import-with-fallback
    project_joints_to_norm_pixels, sample_depth_at_joints,
)

WRIST_J = 0
DEPTH_MIN = 0.01

def is_contact(wrist_z, dense_at_wrist, in_frame, thresh_m: float = 0.05):
    """All [B,S,2] (per hand). Contact iff in-frame, valid surface depth, and
    |wrist_z - dense_at_wrist| < thresh_m. Returns bool [B,S,2]."""
    valid = in_frame & (dense_at_wrist > DEPTH_MIN) & torch.isfinite(dense_at_wrist) & torch.isfinite(wrist_z)
    return valid & ((wrist_z - dense_at_wrist).abs() < thresh_m)

def wrist_contact_mask(wrist_cam, dense_depth, cam_intr, thresh_m: float = 0.05):
    """wrist_cam [B,S,2,3] cam-frame GT wrist (m); dense_depth [B,S,1,H,W] GT metric
    depth; cam_intr [B,3]. Returns contact bool [B,S,2]."""
    grid_xy, z = project_joints_to_norm_pixels(wrist_cam.unsqueeze(3), cam_intr)  # [B,S,2,1,2],[B,S,2,1]
    d, in_frame = sample_depth_at_joints(dense_depth, grid_xy)                    # [B,S,2,1]
    return is_contact(z[..., 0], d[..., 0], in_frame[..., 0], thresh_m)
```

Note: `scripts/root_depth_anchor.py` already imports `project_joints_to_norm_pixels` / `sample_depth_at_joints` with a dev-machine fallback; re-export them from there so this module needs no new import handling.

- [ ] **Step 4: Run the test, verify it passes**

Run: `python -m pytest tests/test_contact_gate.py -k is_contact -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/contact_mask.py tests/test_contact_gate.py
git commit -m "feat(contact): per-hand GT contact predicate + wrist sampling wrapper"
```

---

## Task 2: Offline contact-cache builder

**Files:**
- Create: `scripts/build_contact_cache.py`

- [ ] **Step 1: Implement the builder (mirrors eval_scale_source's loaders)**

```python
# scripts/build_contact_cache.py
"""Write hand_data/contact_cache.pt ([N,2] bool) per HOI4D seq: GT wrist on the
visible GT surface (|z_wrist_GT - GT_dense_depth_at_wrist| < thresh). Reuses the
preprocessed cam caches + raw_depth; same center-crop/resize as eval_scale_source.
Run (gb10 venv): python -m scripts.build_contact_cache --pp /tmp/hoi4d_pp \
    --raw /work/scratch/dmonopoli/hoi4d --res 224 --thresh_m 0.05"""
import argparse, glob, os
import cv2, numpy as np, torch
from scripts.contact_mask import wrist_contact_mask

RH = 1

def _center_square(a):
    h, w = a.shape[:2]; s = min(h, w)
    return a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]

def _load_depth(path, res):
    d = _center_square(cv2.imread(path, cv2.IMREAD_ANYDEPTH)).astype(np.float32) / 1000.0
    d = cv2.resize(d, (res, res), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--res", type=int, default=224); ap.add_argument("--thresh_m", type=float, default=0.05)
    args = ap.parse_args()
    seqs = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(args.pp, "*")))
            if os.path.exists(os.path.join(d, "hand_data", "gt_joints_cache_cam_v2.pt"))]
    for sq in seqs:
        hd = os.path.join(args.pp, sq, "hand_data")
        gtj = torch.load(os.path.join(hd, "gt_joints_cache_cam_v2.pt"), map_location="cpu").float()  # [N,2,16,3]
        ci = torch.load(os.path.join(hd, "cam_intrinsics.pt"), map_location="cpu").float().view(1, 3)
        deps = sorted(glob.glob(os.path.join(args.raw, sq, "raw_depth", "*.png")))
        n = min(len(deps), gtj.shape[0])
        out = torch.zeros(gtj.shape[0], 2, dtype=torch.bool)
        for t in range(n):
            wrist = gtj[t:t + 1, :, 0, :].unsqueeze(0)           # [1,1,2,3]
            d = _load_depth(deps[t], args.res).reshape(1, 1, 1, args.res, args.res)
            out[t] = wrist_contact_mask(wrist, d, ci, args.thresh_m)[0, 0]
        torch.save(out, os.path.join(hd, "contact_cache.pt"))
        print(f"[{sq}] contact frames RH: {int(out[:, RH].sum())}/{n}", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check + commit (no GPU needed to author; it runs in the sbatch)**

```bash
python -c "import ast; ast.parse(open('scripts/build_contact_cache.py').read()); print('AST OK')"
git add scripts/build_contact_cache.py
git commit -m "feat(contact): offline contact_cache builder (GT wrist on GT surface)"
```

---

## Task 3: Optional external contact gate in RootDepthRefine

**Files:**
- Modify: `diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py:29-35`
- Test: `tests/test_contact_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_contact_gate.py
from diffsynth.auxiliary_models.worldmirror.models.heads.root_depth_refine import RootDepthRefine

def test_external_contact_gate_overrides_band():
    m = RootDepthRefine(hidden=4, conf_thresh=0.1, band_m=0.05)
    with torch.no_grad():                       # force a non-zero delta so gating is observable
        m.net[-1].weight.fill_(1.0); m.net[-1].bias.fill_(0.5)
    wrist_z = torch.tensor([[[0.40, 0.40]]]); d_scene = torch.tensor([[[0.90, 0.90]]])  # disagree 0.50 >> band
    conf = torch.ones(1, 1, 2); in_frame = torch.ones(1, 1, 2, dtype=torch.bool)
    contact = torch.tensor([[[True, False]]])
    delta, gate = m(wrist_z, d_scene, conf, in_frame, contact=contact)
    assert gate.tolist() == [[[True, False]]]   # contact fires hand0 despite 50cm disagreement
    assert delta[0, 0, 1].item() == 0.0          # hand1 gated off

def test_falls_back_to_band_when_no_contact():
    m = RootDepthRefine(hidden=4, conf_thresh=0.1, band_m=0.05)
    wrist_z = torch.tensor([[[0.40, 0.40]]]); d_scene = torch.tensor([[[0.42, 0.90]]])  # 2cm, 50cm
    conf = torch.ones(1, 1, 2); in_frame = torch.ones(1, 1, 2, dtype=torch.bool)
    _, gate = m(wrist_z, d_scene, conf, in_frame)   # no contact arg -> band proxy
    assert gate.tolist() == [[[True, False]]]
```

- [ ] **Step 2: Run, verify it fails**

Run: `python -m pytest tests/test_contact_gate.py -k contact_gate -v`
Expected: FAIL — `forward() got an unexpected keyword argument 'contact'`

- [ ] **Step 3: Implement (minimal edit to forward)**

```python
    def forward(self, wrist_z, d_scene, conf, in_frame, contact=None):
        """All inputs [B, S, 2] (per hand). Returns (delta_z [B,S,2], gate bool [B,S,2]).
        When ``contact`` (bool [B,S,2]) is given it REPLACES the |disagree|<band_m proxy:
        fire where the hand actually touches the surface (scene depth reliable there)."""
        disagree = d_scene - wrist_z
        feats = torch.stack([wrist_z, d_scene, disagree, conf], dim=-1)  # [B,S,2,4]
        delta = self.net(feats).squeeze(-1)  # [B,S,2]
        reliable = contact.bool() if contact is not None else (disagree.abs() < self.band_m)
        gate = in_frame & (conf > self.conf_thresh) & reliable
        return delta * gate.float(), gate
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_contact_gate.py -k contact_gate -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add diffsynth/auxiliary_models/worldmirror/models/heads/root_depth_refine.py tests/test_contact_gate.py
git commit -m "feat(contact): RootDepthRefine optional explicit contact gate (replaces band proxy)"
```

---

## Task 4: Thread contact_mask through apply_root_anchor

**Files:**
- Modify: `scripts/root_depth_anchor.py:30-50`
- Test: `tests/test_contact_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_contact_gate.py
from scripts.root_depth_anchor import apply_root_anchor

def test_apply_root_anchor_passes_contact_to_gate():
    m = RootDepthRefine(hidden=4, conf_thresh=0.0, band_m=0.001)  # band so tight only contact can fire
    with torch.no_grad():
        m.net[-1].weight.fill_(1.0); m.net[-1].bias.fill_(0.3)
    B, S = 1, 1
    pred = torch.zeros(B, S, 2, 1, 3); pred[..., 2] = 0.5         # wrist depth 0.5m, single joint
    gs = torch.full((B, S, 1, 8, 8), 0.9)                         # scene 0.9m -> 40cm disagree
    cam_intr = torch.tensor([[1.0, 0.5, 0.5]])
    contact = torch.tensor([[[True, False]]])
    _, delta, info = apply_root_anchor(m, pred, gs, None, cam_intr, contact_mask=contact)
    assert bool(info["gate"][0, 0, 0]) is True and bool(info["gate"][0, 0, 1]) is False
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest tests/test_contact_gate.py -k apply_root_anchor -v`
Expected: FAIL — `apply_root_anchor() got an unexpected keyword argument 'contact_mask'`

- [ ] **Step 3: Implement (edit signature + the module call)**

In `scripts/root_depth_anchor.py`, change the signature and the `module(...)` call:

```python
def apply_root_anchor(module, pred_joints, gs_depth, gs_depth_conf, cam_intr, contact_mask=None):
```
and
```python
    delta_z, gate = module(wrist_z, d_scene, conf, in_frame, contact=contact_mask)  # [B,S,2]
```
Leave the rest of the function unchanged.

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_contact_gate.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/root_depth_anchor.py tests/test_contact_gate.py
git commit -m "feat(contact): apply_root_anchor forwards optional contact_mask to the gate"
```

---

## Task 5: Training integration (load cache, gate loss, pass mask)

**Files:**
- Modify: `scripts/train_hand_head.py` (the hand dataset `__getitem__`/collate that builds `clip`/`batch`; the anchor block ~`:1771-1833`)

- [ ] **Step 1: Load contact_cache in the dataset, attach to the batch**

Find where the dataset loads `gt_joints_cache_cam_v2.pt` per sequence (grep `gt_joints_cache_cam_v2`). Immediately after, load the sibling contact cache (decoupled from any 2D/extrinsics gate, same lesson as the cam_intrinsics fix in 06099df):

```python
        _contact_p = os.path.join(hand_dir, "contact_cache.pt")
        seq_contact = torch.load(_contact_p, map_location="cpu").bool() if os.path.exists(_contact_p) else None
```
Slice it with the same frame indices used for `seq_gt_joints` and attach to the per-clip dict:
```python
        clip["contact"] = seq_contact[frame_idx] if seq_contact is not None else None  # [S,2] bool or None
```
and stack into the batch the same way other per-clip tensors are stacked (`None` -> omit key).

- [ ] **Step 2: Pass the mask into apply_root_anchor (train_hand_head ~:1786)**

```python
                        pred_joints, _ra_delta, _ra_info = apply_root_anchor(
                            model.root_depth_refine, pred_joints,
                            preds["gs_depth"], preds.get("gs_depth_conf"),
                            batch["cam_intrinsics"],
                            contact_mask=batch.get("contact"),   # None -> band proxy (back-compat)
                        )
```
(Match the existing positional args at that call site; only ADD `contact_mask=`.)

- [ ] **Step 3: Gate the anchor loss by contact when present (~:1831)**

`root_anchor_loss` already gates by `gate & has_hand`. Because Task 3 folds contact into `gate`, no signature change is needed — the loss now only supervises contact frames automatically. Confirm `_ra_gate` passed to `root_anchor_loss` is the gate returned in `_ra_info["gate"]` (it is). No edit unless that wiring differs.

- [ ] **Step 4: Smoke-compile on the login node (x86, py_compile only — never import venv_gb10 here)**

Run: `python -m py_compile scripts/train_hand_head.py && echo OK`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_hand_head.py
git commit -m "feat(contact): train_hand_head loads contact_cache, gates anchor by true contact"
```

---

## Task 6: Eval integration (contact-gate switch)

**Files:**
- Modify: `scripts/eval_hand_cam_anchor.py` (+ its `predict_clip` anchor call)

- [ ] **Step 1: Add `--contact_gate {proxy,oracle,off}` (default proxy = current behaviour)**

```python
    ap.add_argument("--contact_gate", choices=["proxy", "oracle", "off"], default="proxy",
                    help="oracle = gate by cached GT contact (mechanism ceiling); "
                         "proxy = the deployable |disagree|<band_m gate; off = no anchor.")
```

- [ ] **Step 2: Load contact_cache per seq and pass to predict_clip**

Where `predict_clip` calls `apply_root_anchor`, pass `contact_mask=clip_contact if args.contact_gate == "oracle" else None`, and when `--contact_gate off` skip the anchor entirely (set `model.enable_root_anchor=False` for the run). Load `contact_cache.pt` beside the seq's joint cache; slice with the clip frame indices.

- [ ] **Step 3: Syntax check + commit**

```bash
python -m py_compile scripts/eval_hand_cam_anchor.py && echo OK
git add scripts/eval_hand_cam_anchor.py
git commit -m "feat(contact): eval_hand_cam_anchor --contact_gate proxy|oracle|off"
```

---

## Task 7: Isolation experiment sbatch

**Files:**
- Create: `run_hoi4d_contact_phase2_gb10.sbatch`

- [ ] **Step 1: Author the sbatch (preprocess -> build contact cache -> 6 arms -> eval)**

Model on `run_hoi4d_p4_big.sbatch` (same 11 seqs, KVALS, preprocess loop, `run_arm`). Differences:
- After preprocess, build the contact cache: `python -u -m scripts.build_contact_cache --pp "$PP" --raw "$HOI4D" --res 224 --thresh_m 0.05`.
- Stable operating point for EVERY arm: `training.max_steps=120 training.grad_accum_steps=1 training.val_every=60 training.log_every=20`.
- Arms (each = train then eval C-abs):
  1. `control`     `model.enable_root_anchor=false loss_weights.kp3d_abs=0.0`            → eval `--contact_gate off`
  2. `proxy_only`  `model.enable_root_anchor=true  loss_weights.kp3d_abs=0.0 model.root_anchor_kwargs.band_m=0.5`  → eval `--contact_gate proxy`
  3. `contact_only` `model.enable_root_anchor=true loss_weights.kp3d_abs=0.0`            → eval `--contact_gate oracle`
  4. `contact_kp`  `model.enable_root_anchor=true  loss_weights.kp3d_abs=0.3`            → eval `--contact_gate oracle`
- Decision: arm 3 `C-abs` < arm 1 (control) ⇒ the contact-gated anchor improves metric placement ON ITS OWN; arm 3 < arm 2 ⇒ contact gating beats the wide-band proxy.
- Checkpoints to node-local `/tmp`; results to `/work/scratch/dmonopoli/contact_p2_results` and `$HOME`; SLURM logs to `/work/scratch/dmonopoli/joblogs`. `#!/bin/bash`, `--time=03:00:00`.

- [ ] **Step 2: Syntax check, commit, push**

```bash
bash -n run_hoi4d_contact_phase2_gb10.sbatch && echo OK
git add run_hoi4d_contact_phase2_gb10.sbatch
git commit -m "feat(contact): Phase-2 isolation sbatch (control/proxy/contact x kp3d_abs)"
git push origin feat/hand-scene-metric-coupling
```

- [ ] **Step 3: Launch (from login node) and read C-abs per arm**

```bash
cd ~/FF-4DGS-Ego && git fetch origin && git reset --hard origin/feat/hand-scene-metric-coupling
sbatch run_hoi4d_contact_phase2_gb10.sbatch && squeue --me
```

---

## Self-Review

**Spec coverage:** gate redesign (Task 3) ✓; explicit contact signal (Tasks 1–2) ✓; train integration (Task 5) ✓; eval oracle/proxy switch (Task 6) ✓; isolation + stable operating point + kp3d_abs isolation (Task 7) ✓; non-circular oracle framing (header + Task 6) ✓.

**Placeholder scan:** Tasks 1–4 contain complete code + tests. Tasks 5–6 reference exact files and call sites with the surgical edit shown; they are integration edits into a 1900-line file, so they cite grep anchors (`gt_joints_cache_cam_v2`, the `apply_root_anchor` call) rather than fragile line numbers — verify the anchor call site matches before editing.

**Type consistency:** `contact` is bool `[B,S,2]` everywhere — `is_contact`/`wrist_contact_mask` return it, `RootDepthRefine.forward(contact=)` consumes it, `apply_root_anchor(contact_mask=)` forwards it, `build_contact_cache` saves `[N,2]` bool sliced to `[S,2]` per clip. `contact_cache.pt` is the agreed filename in Tasks 2/5/6.

**Risk flagged for execution:** Task 5's batch-stacking of a possibly-`None` `contact` must match how the dataset already handles optional per-clip tensors; if any seq lacks `contact_cache.pt` the run must fall back to the band proxy, not crash. The oracle eval gate uses GT — report arm 3/4 as an oracle-gate ceiling, with the proxy (arm 2) as the deployable number.
