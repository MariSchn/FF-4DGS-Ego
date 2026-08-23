# Reciprocal Hand-Scene Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce target-blind, unseen-dataset, causally controlled evidence for both hand-to-scene and scene-to-hand improvement, lock the world-space protocol, and rewrite the paper around only the effects that survive.

**Architecture:** Context RGB is passed through WorldMirror exactly once and target RGB never reaches the model. Target cameras are attached only to an immutable render request. Predicted MANO is interpolated from context frames and composited as a separate Gaussian set. Scene-to-hand causality is tested with a frozen baseline plus a parameter-matched residual conditioner whose only treatment-specific inputs are stop-gradient scene geometry and camera motion.

**Tech Stack:** Python, PyTorch, pytest, WorldMirror/NeoVerse Gaussian rasterizer, AnySplat, SLURM, JSON provenance, LaTeX.

**Spec:** `docs/superpowers/specs/2026-08-23-reciprocal-hand-scene-evidence-design.md`

## Global Constraints

- Never pass target RGB, target MANO, target masks, or target depth through a model forward in a proposed-method arm.
- Bootstrap paired deltas by sequence, never by frame.
- HOI4D and H2O are absent from every training pool used for the headline zero-shot tests.
- Preserve the user's existing dirty worktree and commit only files owned by the current task.
- Use ASCII punctuation and never add agent co-authors to commits.
- A GPU result is unreadable until provenance, hand placement, camera convention, and target-blindness gates pass.
- Only one GPU job may be runnable at a time.

---

### Task 1: Context-only render protocol

**Files:**
- Create: `scripts/insertion_protocol.py`
- Create: `tests/test_insertion_protocol.py`
- Modify: `scripts/metric_views.py`

**Interfaces:**
- Produces: `FramePartition.build(num_frames, target_indices) -> FramePartition`
- Produces: `slice_frame_mapping(mapping, indices, frame_count) -> dict`
- Produces: `target_render_predictions(preds, c2w, intrinsics, timestamps) -> dict`
- Produces: `build_context_views_metric(...) -> tuple[dict, FramePartition]`
- Consumes: existing `build_views_metric`, WorldMirror prediction dictionaries, and rasterizer camera keys.

- [ ] **Step 1: Write failing partition tests**

```python
def test_partition_preserves_temporal_neighbors_without_reordering_targets():
    p = FramePartition.build(16, [6, 10])
    assert p.context == (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15)
    assert p.targets == (6, 10)
    assert p.neighbors == {6: (5, 7), 10: (9, 11)}

def test_partition_rejects_adjacent_or_boundary_targets():
    with pytest.raises(ValueError):
        FramePartition.build(16, [0])
    with pytest.raises(ValueError):
        FramePartition.build(16, [6, 7])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_insertion_protocol.py`

Expected: collection fails because `scripts.insertion_protocol` does not exist.

- [ ] **Step 3: Implement the immutable partition and frame slicer**

`FramePartition` stores original frame indices and a lookup from original index to context position. `slice_frame_mapping` slices only tensors whose second dimension equals the declared frame count and refuses an unregistered ambiguous frame tensor. It never appends target placeholders.

- [ ] **Step 4: Add failing render-camera tests**

```python
def test_target_render_predictions_replaces_only_render_camera_fields():
    out = target_render_predictions(preds, c2w, K, ts)
    assert out is not preds
    assert out["splats"] is preds["splats"]
    assert torch.equal(out["rendered_extrinsics"], c2w)
    assert torch.equal(out["rendered_intrinsics"], K)
    assert torch.equal(out["rendered_timestamps"], ts)

def test_target_render_predictions_does_not_accept_target_rgb():
    assert "img" not in inspect.signature(target_render_predictions).parameters
```

- [ ] **Step 5: Implement target-only camera attachment**

Return a shallow copy of `preds`, preserve the splat objects, validate `[B,T,4,4]`, `[B,T,3,3]`, and `[B,T]`, and replace only `rendered_extrinsics`, `rendered_intrinsics`, and `rendered_timestamps`.

- [ ] **Step 6: Add `build_context_views_metric`**

Build full metric camera tensors outside the model, slice the model view to context frames, set `is_target` false for every model frame, and set `is_static` true. Return the target camera tensors separately through `FramePartition`; never store target images in the model view.

- [ ] **Step 7: Run focused and existing metric-view tests**

Run: `pytest -q tests/test_insertion_protocol.py tests/test_random_frames.py tests/test_extrinsics_decoupled.py`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/insertion_protocol.py scripts/metric_views.py tests/test_insertion_protocol.py
git commit -m "fix: make held-out insertion structurally target blind"
```

---

### Task 2: End-to-end target-leak regression

**Files:**
- Create: `scripts/probe_target_blindness.py`
- Create: `tests/test_target_blindness_contract.py`
- Create: `configs/cluster/target_blindness_smoke.sbatch`

**Interfaces:**
- Consumes: `build_context_views_metric`, `target_render_predictions`, `build_model`, and a real packed clip.
- Produces: a JSON artifact with per-tensor maximum absolute differences across original, zero, and random target payloads.

- [ ] **Step 1: Write a static contract test that fails on the old probe**

```python
def test_insertion_probe_calls_model_with_context_view_only():
    source = Path("scripts/probe_insertion_48.py").read_text()
    assert "build_context_views_metric" in source
    assert "model(context_views" in source
```

Run: `pytest -q tests/test_target_blindness_contract.py`

Expected: FAIL because the current probe calls `model(views)` over context plus target.

- [ ] **Step 2: Implement the real-forward probe**

Load one real clip and checkpoint. Hold context tensors fixed. Create original, zero, and seeded-random target RGB payloads only in the scorer-side batch. Run the production context-only path for each payload and compare context MANO, interpolated MANO, splat attributes, inserted hand attributes, and target renders. Exit nonzero on any difference above declared tolerances.

- [ ] **Step 3: Add the legacy-leak diagnostic mode**

`--legacy-all-frame` deliberately reproduces the old all-frame call and must report a nonzero hand-parameter difference. This is evidence that the test can detect the defect, not merely that the new path passes.

- [ ] **Step 4: Run a local import/CLI test**

Run: `python scripts/probe_target_blindness.py --help` and `pytest -q tests/test_target_blindness_contract.py`

Expected: both pass after Task 3 wires the production probe.

- [ ] **Step 5: Submit the cluster E2E smoke**

Run legacy mode first and require `TARGET_LEAK_REPRODUCED`. Run corrected mode second and require `TARGET_BLIND_OK`. Store both JSON artifacts and SLURM logs.

- [ ] **Step 6: Commit**

```bash
git add scripts/probe_target_blindness.py tests/test_target_blindness_contract.py configs/cluster/target_blindness_smoke.sbatch
git commit -m "test: gate insertion on real-forward target invariance"
```

---

### Task 3: Dataset-agnostic insertion evaluator

**Files:**
- Modify: `scripts/probe_insertion_48.py`
- Create: `scripts/insertion_dataset.py`
- Create: `tests/test_insertion_dataset.py`
- Create: `report/protocols/insertion_hoi4d_sequences.txt`
- Create: `report/protocols/insertion_h2o_sequences.txt`

**Interfaces:**
- Consumes: packed-store `HOT3DHandDataset`, which is generic despite its name, plus `FramePartition`.
- Produces: one result JSON per dataset with A/P/G/C/N1/N3/Z10 arms and a shared provenance schema.

- [ ] **Step 1: Write failing manifest and unseen-store tests**

```python
def test_manifest_is_exact_not_last_n_sequences(tmp_path):
    manifest = tmp_path / "split.txt"
    manifest.write_text("seq_b\nseq_a\n")
    assert resolve_manifest(root, manifest) == [root / "seq_b", root / "seq_a"]

def test_training_roots_cannot_overlap_evaluation_manifest():
    with pytest.raises(RuntimeError, match="overlap"):
        assert_unseen(eval_sequences, training_roots)
```

- [ ] **Step 2: Implement manifest loading and overlap audit**

Resolve every sequence path, require camera calibration and hand supervision, hash the ordered manifest, and compare real paths against all roots embedded in the checkpoint/config provenance. Missing provenance is a hard failure for a headline run.

- [ ] **Step 3: Refactor the probe to context-only data flow**

Call `model(context_views, is_inference=False, use_motion=False)`. Interpolate MANO using original temporal neighbors through the partition lookup. Attach only target cameras to each render prediction. Score target RGB from a separate tensor never passed to the model.

- [ ] **Step 4: Add sensor-depth and boundary metrics**

Read packed sensor depth when present. Score valid target silhouette depth MAE and an occlusion-boundary band. Record explicit `unavailable` rather than silently omitting depth on a store without it.

- [ ] **Step 5: Preserve causal controls and correct the interpretation**

Keep N1 as a tolerance arm, require N3/Z10 to lose P's improvement, and compare G minus P rather than claiming they match. Keep C Gaussian count equal to P.

- [ ] **Step 6: Generate frozen manifests**

Choose sequences from HOI4D and H2O that are absent from the mix5 roots, span participants, and carry all required camera/hand/depth fields. Commit exact manifests before the full run.

- [ ] **Step 7: Run tests and smoke one clip per dataset**

Run: `pytest -q tests/test_insertion_protocol.py tests/test_insertion_dataset.py tests/test_target_blindness_contract.py`

Cluster smoke: one sequence, two clips, all arms. Abort unless median predicted hand depth is 0.4 to 0.9 m, centroid-to-box-center is below the box diagonal, every proposed arm inserts visible alpha, and target blindness passes.

- [ ] **Step 8: Run full Gate 1 sequentially**

Run HOI4D then H2O with paired sequence bootstrap. The analysis script prints PASS or FAIL from the frozen Gate 1 definition and never promotes a secondary metric when the primary metric fails.

- [ ] **Step 9: Commit**

```bash
git add scripts/probe_insertion_48.py scripts/insertion_dataset.py tests/test_insertion_dataset.py report/protocols/insertion_*_sequences.txt
git commit -m "feat: evaluate target-blind hand insertion on unseen stores"
```

---

### Task 4: AnySplat insertion parity

**Files:**
- Modify: `scripts/run_anysplat_heldout.py`
- Create: `scripts/anysplat_hand_adapter.py`
- Create: `tests/test_anysplat_hand_adapter.py`
- Create: `configs/cluster/anysplat_insertion.sbatch`

**Interfaces:**
- Consumes: the Gate 1 manifests and context-only MANO insertion sets.
- Produces: AnySplat A/P/G/C/Z10 renders through the same scorer and target-pose protocol.

- [ ] **Step 1: Inspect AnySplat's live Gaussian dataclass and decoder contract**

Record means, covariance/scale representation, SH/color convention, opacity activation, batch dimensions, and target camera convention from the checked-out cluster repository. Do not infer the schema from tensor shapes alone.

- [ ] **Step 2: Write a failing adapter parity test**

Construct one synthetic opaque Gaussian in front of an identity camera. Render it through the native AnySplat path and through the adapter-composed set. Require the same pixel center, color, alpha ordering, and depth ordering.

- [ ] **Step 3: Implement `append_hand_gaussians`**

Convert context-derived MANO Gaussian means, colors/SH, opacity, rotations, and scales into AnySplat's native container. Preserve scene tensors and concatenate only the Gaussian axis.

- [ ] **Step 4: Remove target-image pose leakage**

Keep `target_pose=interp` as the proposed protocol. The AnySplat camera head must not receive target images. Abort if a configuration selects `camera_head` for a headline insertion run.

- [ ] **Step 5: Run adapter tests and one-sequence smoke**

Require visible hand-only alpha and verify Z10 moves the footprint/depth in the expected direction.

- [ ] **Step 6: Run Gate 2 only if Gate 1 passed**

Evaluate HOI4D then H2O using A/P/G/C/Z10 and the identical sequence bootstrap. Record both absolute AnySplat metrics and paired insertion deltas.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_anysplat_heldout.py scripts/anysplat_hand_adapter.py tests/test_anysplat_hand_adapter.py configs/cluster/anysplat_insertion.sbatch
git commit -m "feat: compose predicted hands with AnySplat"
```

---

### Task 5: Causal scene-to-hand conditioner

**Files:**
- Create: `diffsynth/auxiliary_models/worldmirror/models/heads/scene_geometry_conditioner.py`
- Create: `scripts/scene_conditioning.py`
- Modify: `diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py`
- Modify: `scripts/train_hand_head.py`
- Modify: `scripts/eval_hand_head.py`
- Create: `tests/test_scene_geometry_conditioner.py`
- Create: `tests/test_scene_conditioning_controls.py`
- Create: `configs/exp_scenecond_{cap,scene,depth,camera,shuffle,const}.yaml`

**Interfaces:**
- Produces: `SceneGeometryConditioner.forward(hand_state, scene_points, scene_conf, camera_motion, valid) -> delta_translation`
- Produces: `build_scene_descriptor(gs_depth, gs_conf, intrinsics, camera_poses, hand_boxes, hand_valid, mode, permutation) -> SceneDescriptor`
- Consumes: frozen baseline MANO outputs and stop-gradient scene/camera outputs.

- [ ] **Step 1: Write failing zero-init and shape tests**

```python
def test_zero_init_preserves_warm_started_hand():
    delta = module(hand, points, conf, cameras, valid)
    assert torch.equal(delta, torch.zeros_like(delta))

def test_only_root_translation_changes():
    refined = apply_scene_conditioning(params, delta)
    assert torch.equal(refined[..., 3:], params[..., 3:])
```

- [ ] **Step 2: Write failing corruption-sensitivity tests**

Train a tiny deterministic synthetic example where root depth is a function of a scene plane. Require the scene arm to fit, require a shuffled descriptor to fail, and require equal trainable parameter counts for cap/scene/depth/camera/shuffle/const.

- [ ] **Step 3: Implement annulus sampling in image coordinates**

Use image-aligned `gs_depth`, not the rotated Gaussian attribute grid. Sample a fixed number of valid points outside the hand box but inside an expanded annulus. Backproject with real intrinsics and transform with camera-to-world poses. Carry confidence and a validity mask. Detach the descriptor before the conditioner.

- [ ] **Step 4: Implement the set encoder and residual head**

Encode each point as relative xyz, normal/confidence, and frame-relative camera motion. Masked pool the set, concatenate baseline root/confidence/box geometry, and predict bounded metric `delta_xyz`. Zero-initialize the final layer.

- [ ] **Step 5: Implement closed control modes**

Use an enum, not free text. `cap` zeros all scene channels, `scene` uses all channels, `depth` zeros motion, `camera` zeros point features, `shuffle` applies a deterministic cross-sequence permutation, and `const` supplies median depth plus identity motion. Every mode executes the same module.

- [ ] **Step 6: Freeze causal inputs and enforce warm-start recipe**

Freeze backbone, Gaussian/depth/camera heads, and baseline hand head. Train only the conditioner with absolute metric 3D loss. Abort if HOI4D or H2O appears in a training root, if a warm-start checkpoint is absent, or if parameter counts differ across arms.

- [ ] **Step 7: Run unit tests and a 50-step loss-effect smoke**

Run: `pytest -q tests/test_scene_geometry_conditioner.py tests/test_scene_conditioning_controls.py tests/test_hand_depth_sampling.py tests/test_scene_depth_window.py`

The cluster smoke requires nonzero conditioner gradients for `scene`, finite losses, a hand-placement sanity check, and an inference corruption response above a fixed numerical tolerance.

- [ ] **Step 8: Run matched arms sequentially**

Run Hcap, Hscene, Hdepth, Hcamera, Hshuffle, and Hconst with the same seed and schedule. Evaluate each zero-shot on the frozen HOI4D and H2O manifests. Run Horacle as evaluation-only headroom if packed sensor depth and GT cameras are available.

- [ ] **Step 9: Run a second seed only when the first seed satisfies the mechanism controls**

Do not spend a second seed if Hscene equals Hshuffle/Hconst or if the conditioner ignores corruption.

- [ ] **Step 10: Commit**

```bash
git add diffsynth/auxiliary_models/worldmirror/models/heads/scene_geometry_conditioner.py scripts/scene_conditioning.py diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py scripts/train_hand_head.py scripts/eval_hand_head.py tests/test_scene_geometry_conditioner.py tests/test_scene_conditioning_controls.py configs/exp_scenecond_*.yaml
git commit -m "feat: add causal scene-conditioned hand placement"
```

---

### Task 6: Locked world-space scorer

**Files:**
- Create: `report/protocols/world_space_v1.json`
- Create: `scripts/world_protocol.py`
- Modify: `scripts/eval_world_space.py`
- Modify: `scripts/eval_worldspace_baseline.py`
- Modify: `scripts/make_world_table.py`
- Create: `tests/test_world_protocol.py`
- Extend: `tests/test_w_gauge_degeneracy.py`
- Extend: `tests/test_hand3r_protocol_parity.py`

**Interfaces:**
- Produces: `WorldProtocol.load(path) -> WorldProtocol`
- Produces: `validate_prediction_provenance(record, protocol) -> None`
- Produces result metadata `matches_locked_protocol: true` only after exact validation.

- [ ] **Step 1: Write failing closed-schema tests**

Require dataset manifest hash, exact frames, box source, hand set, joint set, clip length 16, stride 8, segment length 128, camera source, W gauge, WA gauge, unit, and validity rule. Missing or mismatched fields raise before scoring.

- [ ] **Step 2: Encode the canonical protocol**

W uses one first-window rigid alignment with scale fixed and reuses it through the 128-frame segment. WA uses the declared per-window similarity. Store all conventions as closed enums.

- [ ] **Step 3: Add synthetic algebra tests**

Verify invariance to one constant world rigid gauge, sensitivity to multiplicative scale drift, sensitivity to accumulated rotation drift, deterministic missing-hand handling, and correct tail/segment boundaries.

- [ ] **Step 4: Enforce protocol in both ours and baseline scorers**

No scorer may emit `matches_locked_protocol: true` based on a CLI promise. It is computed from resolved inputs and hashes. Non-comparable results are still saved but carry a structured mismatch list.

- [ ] **Step 5: Regenerate every table candidate**

Run ours and baselines over identical manifests. Put methods lacking raw predictions or matching joints/gauge in a separate non-comparable block. Audit sequence and segment counts before metric values.

- [ ] **Step 6: Commit**

```bash
git add report/protocols/world_space_v1.json scripts/world_protocol.py scripts/eval_world_space.py scripts/eval_worldspace_baseline.py scripts/make_world_table.py tests/test_world_protocol.py tests/test_w_gauge_degeneracy.py tests/test_hand3r_protocol_parity.py
git commit -m "fix: lock world-space evaluation provenance and gauge"
```

---

### Task 7: Reciprocal 2-by-2 evaluation

**Files:**
- Create: `scripts/eval_reciprocal_coupling.py`
- Create: `tests/test_reciprocal_table.py`
- Create: `report/drafts/tab_reciprocal_ready.tex`

**Interfaces:**
- Consumes: Gate 1 insertion artifacts, Gate 3 conditioner checkpoints, and Gate 6 protocol-valid world results.
- Produces: Independent, composition-only, geometry-conditioned-hand, and reciprocal rows with paired deltas.

- [ ] **Step 1: Write result-contract tests**

Refuse input artifacts with different manifest hashes, camera protocols, hand checkpoints, or missing target-blindness attestations.

- [ ] **Step 2: Evaluate the four configurations**

Keep dataset, checkpoint family, boxes, cameras, and scorer identical. Apply scene conditioning before MANO interpolation and hand insertion in the reciprocal arm.

- [ ] **Step 3: Check both gains survive simultaneously**

The reciprocal row passes only when its scene delta retains Gate 1's sign/control behavior and its hand delta retains Gate 3's sign/control behavior.

- [ ] **Step 4: Generate the LaTeX table from JSON**

Do not hand-copy metric values. Include confidence intervals, sequence counts, and explicit target-blind/unseen labels.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_reciprocal_coupling.py tests/test_reciprocal_table.py report/drafts/tab_reciprocal_ready.tex
git commit -m "eval: measure reciprocal hand-scene gains"
```

---

### Task 8: Evidence-driven paper rewrite

**Files:**
- Modify: the Overleaf-synced paper sources after verifying their current local state
- Modify or replace: `report/drafts/abstract-and-claims-patch-2026-08-22.md`
- Create: `report/evidence-ledger-2026-08-23.json`
- Create: final generated figures and tables under the paper asset directory

**Interfaces:**
- Consumes: only PASS/FAIL gate artifacts and protocol-valid tables.
- Produces: title, abstract, introduction, method, figures, and contributions consistent with the surviving evidence.

- [ ] **Step 1: Build the evidence ledger**

For every sentence-level claim, record supporting artifact paths, dataset manifests, control outcomes, and whether it passed its predefined gate.

- [ ] **Step 2: Select the thesis from the gate outcomes**

Use reciprocal wording only if Gates 1, 2, and 3 pass. Use compositional wording if only Gates 1 and 2 pass. Use scene-conditioned hand wording if only Gate 3 passes. Remove mutual improvement if neither passes.

- [ ] **Step 3: Rewrite title, abstract, introduction, and contributions**

Remove unsupported feature injection, hand-derived scale calibration, camera-frame SOTA, and 4D claims. State AnySplat's absolute advantage even if insertion is complementary.

- [ ] **Step 4: Replace method and result tables**

Add the context-only data-flow diagram, the causal-control table, the reciprocal 2-by-2 table, and the locked world-space table. Clearly separate oracle and non-comparable rows.

- [ ] **Step 5: Add qualitative mechanism figures**

Show A/P/G/C/Z10 target renders and depth/alpha boundaries. For scene-to-hand, show world placement against reconstructed surfaces plus shuffled/constant controls. Never choose only winning clips; use a predeclared selection rule and include failures.

- [ ] **Step 6: Compile and audit**

Compile the PDF, inspect every page, verify references/table labels, and search for contradicted phrases including `mutual`, `feature injection`, `scale calibration`, `state-of-the-art`, and `4D`.

- [ ] **Step 7: Commit paper-source changes without editing generated changelogs**

Use a human-authored commit message with no agent co-author.

---

## Completion Criteria

The program is complete when every gate has a machine-readable outcome, all published numbers pass the locked provenance contract, the reciprocal claim is either supported by both directional interventions or removed, and the compiled paper contains no statement contradicted by the evidence ledger.
