# Open Lines Tracker

Every parallel line of investigation, its status, and **what "validated / closed" means** — so no line is
abandoned without an explicit verdict. Update the Status + Verdict columns as results land.

Legend — Status: 🟢 running · 🟡 queued · 🔵 results-in-needs-verdict · ⚪ open/parked · ✅ closed(validated) · ❌ closed(refuted)

_Last updated: 2026-07-05 (morning). See memory for the live state; the table below is a Jul-5 snapshot._

## 2026-07-28 CORRECTION — identity-camera-pose bug invalidates the "ours" world rows below

`predict_clip` reads camera poses from `preds["rendered_extrinsics"]`, which is published by the
rasterizer. The `gs_anchor_only` fast path (added to dodge the gsplat hang on gb10) returns
*before* the rasterizer, so the key was missing and `predict_clip` silently fell back to identity
`c2w` — **zero camera translation and zero camera rotation**. Fixed in `9dd474d` by republishing
the camera head's `camera_poses` in the fast path, plus a loud warning on the fallback.

Caught by: all three scene-scale variants bit-identical on 314/314 segments (scale multiplies only
the camera translation) → `--diag_cam` showed `pred_cam_excursion=0.000000 m` vs 19–46 mm of GT
camera motion per clip → `s_gt_med` NaN everywhere, since that stat needs centre motion > 1e-4.

**Invalidated** (do not quote): every "ours" world/W number below, the rot/trans decomposition of
our own trajectory, and the "scene scale is neutral" verdict. The chunk-link and dense-chain
"neutral" verdicts must be **re-verified** — they ran through the same path.
**Unaffected**: all baseline rows (HaMeR/WiLoR+SLAM, HaWoR, Dyn-HaMR — they never call
`predict_clip`) and every camera-frame C-MPJPE/C-abs number (they ignore `c2w`).

**Unlocked by the fix**: the hand-derived scene scale is now measurable against the true camera
scale — `s_hand_med` 0.858 vs `s_gt_med` 1.111, ratio **0.574**. We under-scale the camera
trajectory by ~43%, and since W/WA align rigidly without re-solving scale, that converts straight
into trajectory shape error. This is the new leading candidate lever.

## 2026-07-27 UPDATE — offline-SLAM lever closed, trajectory error decomposed

Two more long-window levers tested. Everything below is 128-frame segments, `wa_short` 30, one
scorer (`eval_worldspace_baseline`), matched 60-sequence subset, matched segment counts.

- ❌ **G1 dense-chain / MonST3R-style linking** (job 103689, 8-seq gate): W_dchain 108.6 vs
  W_global 111.2; the robust variant is worse (184.7). Per-clip dense scene geometry is itself
  drift-inconsistent, so solving seams from it cannot fix the global track. Do **not** build the
  windowed-graph optimizer.
- ❌ **Offline SLAM / BA** (Euler job 8794646). Composing our metric camera-frame hands with the
  DROID/HaWoR track gives **ours+SLAM W 128.1** (WA_s 27.4, WA_l 45.0, C_abs 32.6) against
  WiLoR+SLAM 143.2, HaWoR 147.2, HaMeR+SLAM 150.6. We are the best offline row, but SLAM does not
  rescue the long window — it lands near our own chained trajectory.
- ❌ **Trajectory-scale sub-lever** (job 8795576). Oracle-forcing each sequence's track scale to GT
  made W slightly *worse* (ours 128.1 → 137.1; HaMeR 150.6 → 159.7; WiLoR 143.2 → 151.8). The
  per-sequence scale is ill-conditioned (median 1.58, range 0.11–3.85) because egocentric HOI4D
  clips barely translate.
- ✅ **Oracle ceiling** (job 8796151). Our hands + the GT camera track: **W 40.8** (WA_s 17.7,
  WA_l 21.7). That is what a perfect trajectory buys, and it bounds every trajectory lever.
- ✅ **Rotation-vs-translation decomposition** (job 8798019). Same hands, swapping one half of the
  SE(3) for GT: SLAM R + SLAM t **128.1** → GT R + SLAM t **109.3** → SLAM R + GT t **80.3** →
  GT R + GT t **35.1**. Track error vs GT is 4.79° rotation / 102.5 mm centre. Translation is the
  larger single term, but neither half alone recovers even half the gap — the two are strongly
  **coupled** (same effect the gravity gate saw: predicted rotation and translation co-drift and
  partially cancel in hand placement). ⇒ a trajectory head must regress **full 6-DoF jointly**;
  rotation-only (gravity/IMU-style) or translation-only correction is provably capped at ~80–110.

Two methodological traps worth remembering, both of which produced wrong numbers before being
caught:
1. A "+SLAM" prediction dir can silently encode the **GT** camera track. The student-side
   `hamer_slam_preds` does (`audit_slam_trajectory.py`: 0.0 mm residual, Sim(3) scale 1.0000), which
   briefly made ours+SLAM look like W 40.8 = a 3× win. Always audit a pred dir's implied trajectory
   before quoting a world number from it.
2. `model.enable_gs=False` in a world eval silently zeroes the scene scale (s→1.0, "SCALE
   DEGENERATE" in the log) and inflates every W/WA metric. Keep it **True**; the gsplat render is
   already skipped via `model.gs_anchor_only`.

## 2026-07-26 UPDATE — world-space levers closed + Dyn-HaMR baseline

Long-window (128f) W-MPJPE lever hunt is COMPLETE — three candidate levers tested, all ❌ neutral,
so the bottleneck is now diagnosed and defensible:
- ❌ **Chunk-and-link chaining** (linker, incl. rigid variant): W_global 205 → W_link 202 (~1.5%).
  Neutral. `ours_chunklink_128_v4.json`.
- ❌ **Scene scale** (per-seq pooled/median): W_spool 202.7 / W_smed 201.6. Neutral.
- ❌ **Hand absolute depth** (oracle: perfect GT wrist depth, `--oracle_depth`): W 182.8 → 180.5
  (−1.3%) on a 50-seq subset. Neutral → **DA3 dense-depth world anchor CLOSED** (not the lever).
- ✅ **Diagnosis**: re16 (rigid re-anchor to GT every 16f) = 26 vs actual 182 ⇒ the long-window W
  bottleneck is **camera-trajectory rigid drift** (rotation+translation of per-clip predicted
  cameras), which no post-hoc op on hand predictions can touch. Only a better global camera
  trajectory (SLAM / scene BA ⇒ offline; scene-recon already refuted) would move it. For the paper:
  30f-window W is the headline (we win 37.3); the 128f long-window is a characterized limitation.

🔵→✅ **Dyn-HaMR input-matched baseline**: clean 30/30-seq re-run (batch-artifact freeze fixed) →
WA-MPJPE **48.1** (293 windows); absolute C/W omitted (input-matched setup ⇒ depth_constraint
inactive ⇒ degenerate world_scale, inherent not a bug). Table + PDF/PNG updated. Full-157 extension
chained on Euler (jobs 8612865→8628169→8628170, resumable, ~24h).

## HEADLINE RESULT (2026-07-05)

**Absolute camera-frame MPJPE (Hand3R's "C-MPJPE") = 23.6 mm on 157 held-out HOI4D sequences**
(winner10ep: 10-epoch kp3d_abs=1.0 cached retrain; root-relative 17.3; best val 20.5). Supersedes the
3-epoch 24.9 (kp3d_abs=0.5). Ckpt: `/home/dmonopoli/ckpt_backup/winner10ep_best.pt`, HF upload
`hoi4d_full/winner10ep_best_23p6mm.pt` in flight. Clean-152 / scene-disjoint-132 re-slices still to be
computed from `winner10ep_eval_test.json` per-seq entries (no GPU needed).
Reference: Hand3R 42.6 (online SOTA), HaWoR 51.8 (offline), Cyrus target 40.
Prior headline (3-epoch base, still the HF model-card number until updated): 24.9 / rr 18.6.
Robust under leakage discounts: clean-152 (warm-start seqs excluded) = 25.1, scene-disjoint-132 = 24.8.
Checkpoint + model card: HF `mondraaa/ff4dgs-ego-ckpts` (`hoi4d_full/base_best_mpjpe_3ep.pt`, val 18.7).
Recipe: frozen backbone, HaMeR-style head, warm start best_depth.pt, kp3d_abs 0.5, 3 epochs (7938 steps,
cosine annealed), best-of-val selection, 367-seq train / 157-seq test (stratified 7:3, seed 4023).

**The ~100 mm era is explained**: (1) undertraining — the 1-epoch runs did ~60–380 optimizer steps; a
3-epoch control with nothing else changed dropped 114→52 mm; (2) data — 11 train seqs → 367 took 52→24.9
(with kp3d_abs upweighted). The method was never the limit.

## A. Closed lines (verdicts on the old plan)

| # | Line | Status | Verdict |
|---|------|--------|---------|
| A1 | kp3d_abs sweep on the OLD 8-seq train | ✅ | Superseded: "tapped out ~99mm" was a data-scale artifact, not a lever limit. At 367 seqs the same lever reaches 24.9. |
| A2 | DA3 hand-depth probe | ✅ | DA3 at-wrist ~30–40mm after scale fix — but the head now beats it (~25mm), so DA3-as-reference is no longer needed for C-abs. Keep for W-space if useful. |
| A3 | Unified-model (unfreeze) | ❌ | CLOSED: partial-unfreeze added nothing over frozen at matched steps (52.3 vs 53.0 on 11 seqs). Frozen + head is the recipe. |
| A4 | Contact/DA3 anchor (RootDepthRefine) | ❌ | CLOSED: neutral-to-harmful in all fair tests (3-arm: 114 ctrl vs 122/125 anchored). Do not rebuild. Coupling story returns via B3 (global-context ablation) instead. |
| A5 | Scene-metric claim ("hand rescales scene") | ❌ | REFUTED (B2 eval 62→135cm). Scene recon = frozen 3rd-party, not a paper contribution (audit 2026-07-02). |
| A6 | Test-time smoothing / within-clip pose refine | ❌ | Dead (4% / harmful). |
| A7 | HOI4D data expansion (11 → 525 usable seqs) | ✅ | Done: yinloonga HOI4D_release.zip mirror + full handpose zip; preprocess bit-identical-validated; split 367/157 (seed 4023, `/home/dmonopoli/hoi4d_split.json`). |
| A8 | Feature cache (frozen-backbone tokens) | ✅ | Built: ~10x faster head training (scripts/build_feature_cache.py + data.feature_cache_dir). Corrupt-clip tolerant. Node /tmp only (wiped between jobs — rebuild ~4.5h, resumable). |
| A9 | kp3d_abs sweep at scale (0.25/0.5/1.0/2.0, 2-epoch cached arms, 60-seq eval) | ✅ | **kp3d_abs=1.0 wins** (32.9 vs 38.5/46.1/36.4). Single-seed; the 0.5-arm reading is noise-suspect. Arms are ranking-only (undertrained vs live 3-epoch). |
| A10 | Leakage audit of the 367/157 split | ✅ | Train∩test=0. 5 test seqs seen by warm-start: effect +0.2mm (negligible; report clean-152). 25 sibling-take seqs: scene-disjoint-132 = 24.8 (no change). |
| A11 | Fig. 5 qualitative + 3D error figure | ✅ | REDONE 2026-07-04 with the real Figure-5 pipeline (`blender_figure5.py --matte --film_white --orbit --hand_clear`, assets `fig_hoi4d/fig5_hoi4d/`): `report/figures/hoi4d_fig5_gt_vs_pred.png` + `_alt.png` (wavy splat scene + matte pink GT / blue pred MANO meshes, frame 207, winner2ep ckpt — caption must not claim the 24.9 ckpt) and `hoi4d_test_error3d_side.png` (skeleton overlay, 25.0mm frame). |

## B. Running / queued (cluster job chain)

| # | Line | Job | Status | Validate / close criterion |
|---|------|-----|--------|----------------------------|
| B1 | **Winner promote**: winner@2ep full-157 eval, then **10-epoch kp3d_abs=1.0** cached train + full-157 eval | 102007 | ✅ | **10ep = 23.6 abs / 17.3 rr full-157 — new headline** (val best 20.5). winner@2ep full-157 = 34.2 (confirms sweep arms were ranking-only/undertrained). More epochs kept helping through 10; curve not obviously saturated → a 20ep probe is a cheap C-line if we want the last mm. |
| B2 | **Bbox robustness eval** (jitter:0.2 + fixed:0.30, winner10ep ckpt, TEST60 stride 16) | 102058 | 🔵 results-in | jitter:0.2 → **31.8 abs / 20.2 rr** (clean ≈ 22–23 on this subset): degrades ~+9mm but still well under Hand3R 42.6 → external claims survive detector-like noise. fixed:0.30 → **43.5 / 35.4**: box geometry is a real cue for BOTH depth and articulation (crop mis-sizing hurts everything, not selectively depth). C3 jitter-robust retrain queued (102084). |
| B3 | **Global-context ablation** (crop-only head, cached 2ep, kp3d_abs=1.0) | 102058 | 🔵 results-in | croponly 2ep = **36.7 abs / 26.8 rr** vs matched-budget cpush 2ep 32.9 / 23.9 → context helps ~+12% but UNIFORMLY (abs and rr equally), NOT selectively absolute. Strong "metric-from-scene-context" claim NOT supported as-is. Caveat: ROI-aligned crop tokens come from the globally-attended backbone feature map, so "crop-only" still carries implicit scene context — ablation is weak by construction. 10ep matched rerun in 102084; framing must soften to "scene-image features suffice; global cross-attention adds a uniform ~12%". |
| B4 | **Data-scale curve** (11/50/150 train seqs @ ~2700-step budget) | 102058 | ✅ | **11→55.0, 50→43.2, 150→34.3, 367→32.9 (matched budget) / 23.6 (full 10ep)** on TEST60/full. Clean monotone curve. Key split: root-relative saturates by ~50 seqs (25.1→25.0→23.9) while ABSOLUTE keeps improving with data → metric placement is the data-hungry part. Not saturated at 367 → more data (multi-rig, other categories) is a live lever. |
| B5 | **Multi-rig zero-shot** (60 seqs each from rigs 002/003/004, eval-only, rig-001-trained ckpt) | 102059 | ❌ blocked | 102059 FAILED: rig-002 seqs all die in preprocess (`recover_K: no usable kps2D frames` — handpose annotation layout differs from rig 001 under `ZY.../H2/...`). Needs a layout adaptation pass; parked behind higher-value lines. |
| B6 | **World-space W/WA-MPJPE** with winner10ep | 102118 | ✅ | **W 195.6 (was 353), WA-long 54.7, WA-short 22.4, re-anchor@16 24.1** (40 seqs × 2×128-frame segs, robust_scale). vs Hand3R 86.9 WA / 125.8 W: we now WIN aligned-world (54.7 < 86.9) and lose raw W (195.6 > 125.8) — chaining drift dominates (velGT oracle = 55.0 confirms per-frame poses are fine, trajectory accumulation is the residual). C-abs on this subset 23.2 ✓ consistent with headline. Next W lever if wanted: re-anchoring / drift correction, NOT better hand pose. |
| B7 | **C3 jitter-robust retrain** (train WITH jitter:0.2, 10ep winner recipe) + **detector-box E2E eval** (real WiLoR boxes) | 102132→102386 | 🔵 results-in | Train succeeded Jul 6 (102132) but /home quota zeroed all eval JSONs; eval-only recovery 102386 ✅ 2026-07-09. **jitterrob (TEST60): clean 26.9 / jitter:0.2 27.3 / fixed:0.30 31.2 C-abs** — jitter training buys near-total box robustness (+0.4mm under jitter vs +9mm for winner10ep; fixed-offset penalty 43.5→31.2) at ~3mm clean cost vs winner10ep (~23–24 on this subset). **detbox_eval (winner10ep + real WiLoR det boxes): C-abs 140.6 / rr 40.7** — degradation is UNIFORM (median 143.5, 50/60 seqs >100mm, 0 >300mm; clean-good seqs 20-30mm jump to 170-275) → NOT a few catastrophic seqs. DIAGNOSED (IoU study, 18k frames): det-vs-GT meanIoU 0.383, 18.8% frames IoU==0, det boxes 1.53x too wide. Three input flaws, all in the det-cache builder: detection ran on 224px frames (hand 30-50px → 80.5% recall + sloppy boxes), expand_box never squares (GT protocol = tight×1.5→square), carry-forward fallback decays to IoU 0 on long miss runs (S179 seqs: 51-72% frames IoU<0.3 → 233-275mm). So 140.6 measures a broken input regime, not the model. Fix = **detbox v2**: detect on the HD 1080² frames (Euler HD store being built for HaPTIC), scale boxes into the 224 store coords, apply the exact training protocol, carry-forward on miss; re-eval winner10ep + jitterrob. jitterrob×flawed-detbox cell (102395) still lands as the training-side-mitigation datapoint. Crop-only 10ep ckpt was never banked (nobank) → lost; retrain only if the ablation row is needed. |

**2026-07-05 ~12:08 incident**: /home hit the 20GB quota during 102059's multi-rig preprocess (two 4.7GB
ckpts + preprocess output). All /home writes failed silently → 102059's world result lost (0-byte json),
102093 baselines died at startup (empty log), 102094 c3 config writes failed. Recovered: winner2ep.pt
deleted (~5GB free). Lesson: bank at most ONE 4.7GB ckpt in /home at a time; HF is the durable store
(winner10ep upload verified OK).

**2026-07-05 ~18:15 second incident (self-inflicted)**: quota cleanup deleted `hoi4d_pp_full` (636MB),
which was the LIVE preprocessed store — `hoi4d_train/`+`hoi4d_test/` are symlink farms into it. No /home
snapshots. All inputs survive (split json, handpose zip on scratch, mirror mp4s, deterministic validated
preprocess) → regeneration running (102116, ~4h), banked numbers unaffected. Also fixed en route: WiLoR
baseline install (wilor-mini is git-only, not PyPI) + HaMeR model_config.yaml fetch (team25 has bare ckpt).
Lesson: before rm of any data dir, check inbound symlinks from consumers (`ls -l hoi4d_test/ | head`).
Outcome: regen 102116 completed in 42 min, 524/524 bit-identical store restored + extrinsics 157/157.
World eval 102118 ✅ (see B6). Baselines attempt 2 (102117) failed on chumpy build isolation (aborts the
whole wilor-mini git install) + missing pyrender for HaMeR → fixed (numpy<2 + setuptools<70 first, chumpy
--no-build-isolation, opencv 4.x pin, +pyrender/scikit-image/trimesh), attempt 3 = 102124, then c3 = 102125.

## C. Open / planned (next 1–2 weeks)

| # | Line | Depends on | Criterion |
|---|------|------------|-----------|
| C1 | Baselines (HaMeR, WiLoR) rerun on OUR split | port from H2O baseline work | ✅ **DONE 2026-07-06 (job 102128, all 157 test seqs, same-input 224px protocol, true-focal conversion)**: **WiLoR C-abs 218.2 / rr 67.9 / wrist 208** (det recall 80.5%, fallback 19.5%); **HaMeR (same WiLoR det boxes) C-abs 187.9 / rr 72.6 / wrist 188**. Ours: **23.6 / 17.3** → 8–9x absolute gap ON THE SAME SPLIT; the cross-split asterisk vs Hand3R 42.6 is now bracketed by two same-split crop baselines. Protocol caveats to report: 224px source frames penalize crop models on articulation (their rr ~70 vs typical ~25–40 on HD); absolute failure is structural (weak-persp), not resolution. HaMeR ckpt loaded strict=False with HF-space config on team25 bare ckpt — sanity-check key-match count before final table. **Hand3R has NO public code/ckpt/split-list** (arXiv 2602.03200) → quoted-with-footnote. HaWoR (public, sm_75-buildable) = stretch row. **UPDATE 2026-07-09 (job 102397, rr joint-order fix + per-SEQUENCE agg — these supersede the numbers above): WiLoR C-abs 206.3 / rr 32.2 / wrist 208.2 / PA 7.1; HaMeR C-abs 176.8 / rr 39.9 / wrist 188.0 / PA 8.1.** The old rr ~70 values were the joint-order bug, not a resolution penalty — with the fix their rr is a normal 32–40, so drop the "224px penalizes articulation" caveat and note instead that ours (23.6 abs) beats their root-relative. |
| C2 | 21-joint (fingertips) metric variant | none (tips via MANO vertices, hand_vis_utils) | Match Hand3R's joint count if theirs is 21; expect a slightly higher reading. |
| C3 | Bbox-robust retrain (train WITH jitter) | B2 outcome | → PROMOTED to B7 (job 102084, queued 2026-07-05): B2 deflated under jitter (+9mm). |
| C4 | H2O cross-dataset check | existing H2O pipeline | Second-dataset row. SCOPED 2026-07-09: the old H2O pipeline (pack_h2o npz + train_h2o_hand + eval_cmpjpe) is protocol-incompatible (per-clip agg, own loss, 21-joint remap); the ONE blocking build = npz→current-protocol store adapter (per-seq dirs w/ video_main_rgb.mp4 + hand_data caches), then winner recipe runs verbatim. Split = train subj1-3/test subj4; the banked 58.0 C-MPJPE is MIXED-SET (leakage caveat in run_b3_h2o_subj4.sbatch) — do not quote. h2o_packed store on scratch may need re-pack (dl_h2o.sh). **STATUS 2026-07-10: adapter BUILT + gate-validated (synthetic, 14/14 checks + negative control) and deployed to cluster (md5-verified); h2o_packed store GONE and scratch write-locked → re-pack plan = stream subject tars to /home. BLOCKED: h2odataset.ethz.ch returns HTTP 401 with the stored creds (worked for subject4 earlier) → credentials expired/rotated. NEEDS DARIO: re-register / refresh H2O dataset access (https://h2odataset.ethz.ch), then place creds on the cluster; everything downstream is staged and auto-runs (pack subj1 → convert+validate → gate report → review → train).** **UPDATE 2026-07-19 — VALID FRESH-TRAIN RESULT LANDED:** first result (job 102879: 64.4 C_abs with C_rr>C_abs) was a GT joint-remap SCRAMBLE bug (21→16 selector composed for the wrong layout; thumb slots held fingertips); fixed (H2O16_IDX=H2O_TO_MANO[:16] in eval_cmpjpe.py + h2o_to_currentproto.py, new anatomical bone-length gate 4), store fully reconverted (102962, 611 gate-4 PASS / 0 FAIL), retrained (102963, kp3d cured 0.08→0.006). **Corrected: train subj1-3 → test subj4 (45 seqs, right hand, same protocol): C_abs 48.2 / C_rr 25.6 / wrist_abs 44.2 mm** (results/h2o_fresh_eval.json; scrambled artifacts parked as *_SCRAMBLEDGT_102879). Establishes the recipe transfers to a 2nd dataset; NOT yet calibrated (no H2O baseline) — reviewer pass 6 M3: minimum = metric-tuned HaMeR on H2O; best = HOI4D→H2O zero-shot for both our head and fu-HaMeR (unrun; needs box-convention bridge + fu ckpt banking + ls_polle/Cyrus sign-off). ALL pre-fix H2O numbers computed through eval_cmpjpe's h2o remap (incl. the banked 58.0) are void. **AUDIT 2026-07-19 (job 102988):** 48.2 has a per-SCENE placement bias (group medians k2 32.1 / k1 33.8 / h2 43.3 / o1 50.1 / h1 63.2 / o2 64.8; C_rr uniform) with intrinsics (f=198.1 uniform), depth range (med_z 0.39-0.42 all groups, corr -0.37), box size, and train/test depth coverage ALL ruled out. Prime suspect = the kp2d loss (train_hand_head.py hardcodes Aria 1408px + 90° rotation; active at weight 0.05 in every recipe; raw 13.7 on H2O ≈ the entire train loss vs 2-6 on HOI4D). Control IN FLIGHT: job 102989 = same recipe + loss_weights.kp2d=0.0 → results/h2o_kp2d0_eval.json. If h1/o2 recover, the honest H2O number improves and an HOI4D kp2d=0 control becomes worth running (the 23.6 headline also trained with the term active). **kp2d0 RESULT 2026-07-19 15:03 (job 102989): artifact CONFIRMED but PARTIAL. Per-seq C_abs 48.2 → 44.6 (−3.6mm, −7.5%); train_loss collapsed 0.695→0.0077 (kp2d WAS ~99% of it). Per-scene medians kp2d0 vs baseline: k2 31.1/32.1, k1 32.6/33.8, h2 41.2/43.3, o1 50.0/50.1, h1 54.0/63.2 (−9.2), o2 58.8/64.8 (−6.0) — gain concentrates in exactly the two worst scenes; good scenes flat. So the mis-specified kp2d loss does bias absolute placement in hard scenes, but a large scene-dependent residual survives (h1/o2/o1 50-59mm vs k 31-33mm). CAUTION: the streaming "hand_metrics(all) MPJPE=24.9" is frame-pooled, NOT the comparable per-seq C_abs (44.6). ADOPT kp2d=0 as the cleaner H2O recipe (44.6). Residual candidates: scene-dependent H2O GT quality or genuine appearance difficulty. NEXT (needs Dario sign-off, cheap): HOI4D kp2d=0 control — 23.6 headline trained with same wrong term (HOI4D raw kp2d 2-6 → expect smaller shift, but headline is load-bearing → defensibility). Head banked ckpt_backup/h2o_kp2d0_head.pt.** **HOI4D CONTROL LAUNCHED 2026-07-19: job 103044 (auto/hoi4d_kp2d0.sbatch), two MATCHED arms at headline seed 42 sharing one recon cache — off (kp2d=0.0) then on (kp2d=0.05 shipped default), both full-157 GT-box via eval_hand_cam_anchor. Results results/hoi4d_kp2d_{off,on}_eval.json, marker HOI4D_KP2D0_ALL_DONE. Compares against banked 23.6±0.8; the matched on-arm removes node/cache confound so the delta reads above ~0.8mm seed variance.** |
| C5 | Fine kp3d_abs grid (0.7/1.4) + multi-seed | B1 | Only if 10ep suggests weight sensitivity matters at full training. |
| C6 | Multi-rig TRAINING | B5 outcome | Only if zero-shot degrades badly. |

## D. Paper framing (draft thesis, for Cyrus)

**"Feedforward scene reconstructors already carry the information for metric-absolute hand placement —
we identify it, extract it with a lightweight head, and beat dedicated online systems 2x."**

In hand: 24.9 vs 42.6 headline; error anatomy (articulation solved: rr 18.6, PA ~5–8; absolute placement
was the residual); supervision lever measured (A9); explanation of the ~100mm era (undertraining + data).
Pending: WHERE the signal lives (B3 scene-context; B2 box-size cue), W-space transfer (B6), cross-rig (B5),
data curve (B4). Negative results to include: hand-as-global-scale refuted, anchors neutral, smoothing
dead, unfreeze unnecessary — they sharpen the analysis framing.

Integrity notes that ship with any claim: GT-derived boxes until B2/C3 close; single-rig until B5;
16-joint metric until C2; our split ≠ Hand3R's split (protocol matched, lists differ, theirs unpublished).

### D2. CVPR-reviewer verdict (2026-07-09, two independent passes: publication-potential + protocol audit)

**Verdict: Reject as constituted; items below get to Borderline+/main-track-viable.** Full text in
session memory (`cvpr-review-verdict-2026-07-09`). THE attack: oracle GT boxes (box size = depth cue;
fixed:0.30 → 43.5 > Hand3R 42.6, so the "win" disappears without GT localization) + zero fair
absolute-placement baseline actually run (WiLoR/HaMeR are weak-persp straw men; their abs numbers are
harness artifacts — WiLoR H2O PA 47.6 but abs 614.9). New actionable protocol bugs from the audit:

| # | Finding | Fix | Status |
|---|---|---|---|
| D2-1 | Aggregation mismatch: ours = per-seq mean over seqs; baselines = per-128-frame segment mean (eval_hand_cam_anchor.py:113 vs eval_worldspace_baseline.py:104) | one scorer, one aggregation, regenerate table | OPEN |
| D2-2 | Baselines never given GT boxes (box-source asymmetry both directions) | rerun WiLoR/HaMeR WITH our GT boxes + ours with detbox v2 = parity both ways | **HALF-DONE 2026-07-10 (job 102436, TEST60, per-seq):** WiLoR GT-box C_abs 243.9/rr 25.8 (det-box same split: 205.8/27.8); HaMeR GT-box 164.7/33.1 (det-box: 166.7/36.6). GT boxes fix misses + help rr but abs does NOT close (HaMeR flat, WiLoR 38mm WORSE → weak-persp depth is calibrated to its own detector's box stats = box-convention artifact). Baseline absolute failure is structural. Remaining half = ours on detbox v2 (Euler chain). **Reviewer pass 2 (2026-07-10) adds:** (a) the GT-box-worse result opens an OOD-box counter-attack ("you fed WiLoR a foreign box convention then called it structural") — fix = run WiLoR/HaMeR in their FULLY NATIVE regime (native detector+padding+crop res+focal, full-res HOI4D) as the primary baseline row, demote det/GT-box rows to a box-convention ablation, state the tz=2f/(box*s) mechanism explicitly; (b) conclusion-bearing parity table must be full-157 (or pre-declared identical subset for ours+baselines), TEST60 is diagnostic-only; (c) reviewer's detbox v2 calibration for reference: E2E <=~50mm keeps the claim alive, ~80mm borderline, ~140 fatal. **Dario 2026-07-10: (a)+(b) APPROVED and dispatched to the baseline agent (Euler es_tang, native detector + native padding + native crop res + scaled_focal_length translation as primary, true-focal logged as secondary, full-157, per-seq agg); (c) pre-registered headline policy NOT adopted — headline framing decided after seeing detbox v2.** **Progress 2026-07-10 eve:** first Euler jobs (6576437/6576424) died at evaluator selftest = missing `dill` in venv recipe (env gap); fixed + resubmitted as 6605600 (native full-res E2E) + 6605597 (WiLoR GT-box full-157). GT-box parity COMPLETE on full-157: WiLoR GT-box 240.0 abs / 30.5 rr, HaMeR GT-box 168.3 / 36.0 — full split confirms TEST60 (GT boxes don't rescue abs; WiLoR ~34mm worse than own det boxes). **NATIVE E2E DONE (6605600, 156 seqs): WiLoR true-focal 83.4 abs / 27.2 rr; HaMeR true-focal 88.0 / 30.3; published-conversion rows ~24.5m (non-metric, document as such). Native det recall 91-100% vs 80.5% on 224px crops. Crop-regime 170-240 numbers were input artifacts — primary comparison is now ours-GT-box 23.6 vs native-E2E 83-88; detbox v2 (resubmitted 7152642 after dill fix, prior 6737582 was a FALSE GREEN) supplies our honest E2E side. **CLOSED 2026-07-12 (baseline agent final report):** W-lifting question ANSWERED from job code — all baseline W rows (WiLoR 62.5 / HaMeR 84.0, native AND GT-box) use GT extrinsics (cam_extrinsics_cache.pt, X_w=R^T(X_c−t), EXTR_SANITY 0.000mm) = oracle-camera tier, NEVER protocol-comparable to our own-trajectory W 195.6 (footnote as oracle-lifted or drop). Dropped seq = ZY20210800001_H1_C14_N34_S207_s03_T2 (transient mirror pull miss, recorded in seq_misses, recoverable). Deviation flags for the paper: HaMeR own-detector infeasible (ViTDet/mmcv-1.x vs torch-2.3 harness) → WiLoR detector + HaMeR-native rescale 2.0 substituted, footnote required; HaMeR GT-box rows = tight GT box × native 2.0 rescale. absloss0 control (102437): C_abs 725 / rr 131 (TEST60) — kp3d_abs causally necessary. JSONs: Euler $SCRATCH/results/{wilor,hamer}_native[_truefocal]_eval.json + native_fullres_inline.json + *_gtbox157_*; student /home/dmonopoli/results/. Remaining in D2-2: detbox v2 only (7152642). HaPTIC HD: C_rr 25.7 valid; C_abs 6360 still published-units, needs true-focal pass.** |
| D2-3 | Hard-coded "(Hand3R 42.6)" in eval printouts (eval_world_space.py:616, eval_hand_cam_anchor.py:208) invites invalid direct comparison | remove; footnote "quoted, not comparable" | OPEN |
| D2-4 | root_depth_anchor ref_scale 0.892 fit on test GT — every anchor result is test-set-tuned (headline unaffected) | refit on disjoint calibration split before citing any anchor number | **TOOLING DONE 2026-07-10, run pending.** Contamination CONFIRMED: probe (da3_hand_probe v2, job 101840) fit on 8 of the old 11 seqs; 5 of those 11 are in the current 157 TEST split, and the anchor A/B evals of that era ran on the same 11 (fit-on-eval). Note 0.892 = measured DA3/GT ratio; the applied constant is 1.121 = 1/0.892. Built: `scripts/fit_ref_scale.py` (train-only refit, hard-fails on any test/unknown seq, seq-level bootstrap CI) + `ANCHOR_REF_SCALE` env override in root_depth_anchor.py (non-breaking). Blocker: DA3 wrist caches exist only for the old 11 seqs (5-6 train) — for a tight CI, build train-split DA3 caches first (gb10, venv_gb10 + /tmp/da3libs recipe). Then rerun anchor evals with ANCHOR_REF_SCALE=<refit>. Headline C-abs unaffected (anchor off). **REFIT DONE 2026-07-10 (job 102463):** 40 train seqs / 966 frames -> median ratio 0.9571, **ref_scale = 1.0449 (95% CI 1.0075-1.0678)**. The old test-fit 1.121 is OUTSIDE the CI = contaminated AND over-correcting ~7%. Any cited anchor number must be rerun with ANCHOR_REF_SCALE=1.0449 (env override, no code edit). JSON: /home/dmonopoli/ref_scale_refit.json. |
| D2-5 | hoi4d_to_haptic.py:77 sx=W/(2cx) conflates principal point with focal scale — near-exact for our center crops (error ~0.1%), fragile in general | scale f,cx,cy by true resolution ratio, keep off-center pp | polish |
| D2-6 | Missing ablations: backbone-swap (frozen recon vs frozen DINO vs random-init) = THE novelty defense; kp3d_abs zeroed-loss causal control | run both pre-submission | **kp3d_abs control DONE 2026-07-10 (job 102437 absloss0, TEST60):** zeroing ONLY kp3d_abs (kp3d-rel 0.05, kp2d 0.05, transl 1.0, priors all unchanged vs winner recipe) collapses C_abs to 725mm (winner ~23.6) and C_rr to 131 — absolute 3D supervision is causally necessary; features + relative losses alone do not yield absolute placement. JSON: /home/dmonopoli/results/absloss0_eval.json. **Backbone-swap DONE 2026-07-11 (jobs 102484/102485), null hypothesis CONFIRMED:** full-157 GT-box per-seq C_abs/C_rr — recon 23.6/17.3, frozen DINOv2-L **21.9/14.8 (BEATS recon)**, random-init frozen 27.7/18.3. "Reconstruction features encode metric depth" is dead as a novelty claim; performance comes from the recipe (kp3d_abs + GT-box geometry + crop tokens + 367 seqs). Deviations to state: new arms trained head from scratch (favors recon row — DINO won anyway); DINO C=1024 vs 2048. JSONs results/bb_{dino,random}_eval_test.json. Contribution narrative must reposition to recipe/protocol/analysis; detbox v2 becomes THE load-bearing number (random-arm strength corroborates GT-box-as-depth-cue). **2026-07-15: ordering now needs 3-seed error bars to be citable (see D2-7). detbox v2 boxes BUILT (job 7199201, 157/157, mean det recall 0.829) but eval 102815 KILLED 2026-07-16 at n=94 (partial C_abs 160.1, INVALID): surgical audit found build_detboxes_v2.py used the HOT3D square-to-max-side+clamp convention while the HOI4D store (preprocess_hoi4d.py:444-455) is rectangular x1.5 per-dim UNCLAMPED, and training/hamer_head consume the stored box verbatim ([cx,cy,w,h] geometry injection = the depth cue) — convention mismatch, not detector quality (meanIoU 0.245 = the tight-rect-vs-1.5-square ceiling; C_rr only 41.5 vs 17.3). Also full-frame [0,0,1,1] boxes before first detection. detbox v3 BUILT 2026-07-16 (Euler 7303584; 7297824 requeued at 8G past a ~32G/user QOS mem cap): n=157, mean_recall=0.829, meanIoU=0.346 (vs v2 0.245 — convention fix confirmed), BUT med_w_ratio=1.703 / med_h_ratio=1.677 → detector boxes ~1.7x GT size per dim (silhouette+margin vs joint hull), predicts elevated raw C_abs via the [w,h] depth cue; jitterrob (2nd ckpt in same eval) is the trained mitigation. Shrink factor would need TRAIN-side HD frames (test157_hd only exists) — small train-HD pull via the mirror if needed; fitting on test is forbidden. Boxes md5-verified to student (05a86069), EVAL RUNNING job 102825 (farm 157/157, winner10ep then jitterrob, results detboxv3_{winner,jr}_eval.json); seeds preempted w/ user approval + resubmitted (102826 w10, 102827 dino). Reviewer pass 3 tightened the accept bar: detbox v2 ≤~50 accept-tier, ~50-80 borderline, >~90 fatal (must beat FAIR native baselines 83-88, not the old 188-218). Paper draft repaired: 218/188 removed from headline → native 83.4/88.0; DINO row added; Hand3R marked non-comparable; hard-coded Hand3R printout removed from eval_world_space.py. **detbox v3 WINNER RESULT 2026-07-16 15:45 (job 102825, full n=157, per-seq agg): C_abs = 49.8 / C_rr = 41.0 — UNDER the ≤50 accept-tier line, beats metric-tuned HaMeR det-box 68.7 by 27%, beats native 83.4/88.0. The predicted 1.7x-box-bias depth collapse did NOT happen (~2/157 seqs >90). E2E claim HOLDS in both regimes (ours 23.6 GT / 49.8 det vs mt-HaMeR 39.8 GT / 68.7 det). jitterrob det-box pass running next in same job (detboxv3_jr_eval.json). **jitterrob RESULT (ALL_DONE 17:41): C_abs = 35.6 / C_rr = 26.5 full-157 — new headline E2E row. Beats mt-HaMeR det-box 68.7 by 48% and even mt-HaMeR GT-box 39.8; C_rr 26.5 beats native full-res WiLoR 27.2, killing the pass-4 "worse articulator E2E" MAJOR without a retrain. Box-scale-aug retrain demoted to optional polish. Open cell: jitterrob full-157 GT-box clean number (TEST60 26.9). detbox line CLOSED.** |
| D2-7 | Single-seed everywhere; jitter sensitivity shows variance matters | 3-seed final recipe, mean±std | **IN PROGRESS 2026-07-15 (task #15, agent a00c4b1):** reviewer pass 3 upgraded this to MANDATORY — the backbone-swap ordering (DINO 21.9 / recon 23.6 / random 27.7) is plausibly within single-seed noise, so the D2-6 analysis conclusion is UNLICENSED without error bars. Staging 2 extra winner10ep-recipe seeds (existing 23.6 = seed 1) + 2 DINO-arm seeds if GPU budget allows (259/800 team-h). Queue discipline: submits ONLY after H2O training (h2o_train10ep) is in the student queue (H2O priority). Deliverable = mean±std per arm + whether DINO-beats-recon survives. w10_seeds.sbatch staged. |
| D2-8 | "It's just a metric-tuned HaMeR" — the natural next reviewer attack after the backbone null | fine-tune ONLY WiLoR/HaMeR camera/transl head with kp3d_abs on our 367 train seqs, eval full-157 | **IN PROGRESS 2026-07-15 (task #16, agent afdaaacd, reviewer pass 3 recommendation):** freeze all but the weak-persp cam/transl head, kp3d_abs L1 on abs cam-frame 3D kpts, same 367-seq train data + true-focal lift; eval own-detector boxes (primary, native regime) + GT boxes (secondary). Runs on Euler es_tang (idle, independent of student QOS), ~6h head-only cap. Either outcome publishable: gap closes → confirms recipe-not-backbone cleanly; gap holds → our arch genuinely helps. Pre-empts the strongest post-null attack. **SUBMITTED 2026-07-15 job 7284898 (Euler es_tang). Two documented deviations: (1) target HaMeR not WiLoR — WiLoR-mini's pred_cam is fused inside the ViT backbone (no isolable weak-persp head, no train entrypoint); HaMeR's MANOTransformerDecoderHead has a clean `deccam=nn.Linear(dim,3)`. Unfreeze ONLY deccam (weak-persp [s,tx,ty]); backbone/decoder/decpose/decshape/MANO frozen → C_rr invariant by construction, only absolute translation adapts. (2) Run the 224-store GT-box regime (= our winner's exact 23.6mm headline regime) instead of native HD — streaming 524 HD mp4s in 6h is infeasible w/ mirror IncompleteRead flakiness, and 224-GT-box is the MOST DIRECT rebuttal (metric-tuned HaMeR vs our headline, same regime); native HD already covered by the 83.4/88.0 rows. So GT-box=primary, det-box=secondary + a no-tune GT-box control measuring the C_rr-invariance. Caveat for the paper: unfreezing only a 3-output Linear is the cleanest isolation but a reviewer could call it too narrow — if the result is ambiguous, consider also unfreezing part of the decoder. Row lands as "HaMeR deccam kp3d_abs fine-tune".** **DONE 2026-07-16 (job 7284898, all 3 eval passes, per-seq agg): tuned GT-box C_abs=39.8 / C_rr=36.0 / wrist=48.6 (n=156); tuned det-box (WiLoR detector, recall 0.921) C_abs=68.7 / C_rr=39.9; no-tune GT-box control 168.3/36.0 = bit-identical to the Jul-10 hamer_gtbox157 run (harness validated; C_rr untouched by tuning as designed — pure metric calibration). Train L1 28.4->22.2->20.7mm over 3 epochs. READ-OUT: metric tuning closes 168->39.8 but ours 23.6 still leads the fair matched-regime bar by 16mm (40%); the E2E fair bar for detbox v3 is now 68.7 (tighter than native 83.4/88.0). OPEN before table ships: why n=156 on the mt_gtbox pass (1 seq dropped); JSONs Euler results/hamer_mt_{gtbox,detbox}_*.json + hamer_notune_gtbox_*.json. Watcher agent died at session limit AFTER pass 3 started — numbers pulled directly by main agent.** |
| D2-9 | PROVENANCE for the full-fine-tune control rows (table rows "HaMeR full fine-tune 632M" and "+ jitter aug") — reviewer pass 6 CRITICAL C1: these load-bearing rows were absent from this tracker | log job IDs + JSONs | **CLOSED 2026-07-19 (doc-only; experiments were done 07-16/17).** (1) **hunfz, Euler job 7402476** (DONE 2026-07-17 00:31, full-157, per-seq): same kp3d_abs L1 harness as D2-8 but joints NOT detached; two capacity arms — hd (MANO decoder unfrozen, lr 3e-5, 4ep): GT-box 26.5/17.0/wrist 26.0, det-box 54.1/21.2/50.2; **fu (FULL 632M network unfrozen, lr 1e-5, 4ep, train L1 15.1→9.3): GT-box 21.4 / C_rr 11.6 / wrist 24.1; det-box 44.0 / 16.1 / 43.7.** JSONs Euler results/hamer_{hd,fu}_{gtbox,detbox}_eval.json; sbatch auto/euler_hunfz.sbatch (md5 866c00b5). (2) **hujit, Euler job 7481943** (DONE 2026-07-17 14:13, full-157, per-seq, 2-pass JSON-verified): **fj = fu + matched box-jitter aug** (amplitude 0.2, 4-draw, per-sample deterministic, on tight box pre-×2-expand): GT-box 20.83/11.61/wrist 21.7; det-box 41.64/14.87/40.32; **fus1 = fu seed 1**: GT 21.39/11.7/24.39, det 45.07/15.85/45.15 (fu abs seed-stable). JSONs results/hamer_{fj,fus1}_{gtbox,detbox}_eval.json. These are the rows behind table footnote d (gate-1 robustness control: fj degrades +20.8 GT→det vs ours+jitter +8.3). |
| D2-10 | **Box-source parity for the headline E2E crossover — reviewer pass 6 CRITICAL C2 (open).** Ours det-box rows (49.8 / 35.6) use detbox v3 (our YOLO cache, mean_recall 0.829, meanIoU 0.346, med box ratio ~1.7x GT); the HaMeR fu/fj/mt det-box rows (44.0 / 41.6 / 68.7) use WiLoR-detector boxes (recall 0.921). Different detectors ⇒ the crossover "ours 35.6 < fj 41.6" is not yet input-matched, and box scale is the dominant depth cue. | one eval pass, no retrain: score fj (and fu) on the detbox v3 boxes via a convention bridge (v3 store [cx,cy,w,h] → HaMeR tight-box + rescale 2.0), OR ours on the WiLoR det boxes; report detector, recall, and med box-ratio in footnote d | **LAUNCHED 2026-07-19: Euler job 7775057 (euler_d210.sbatch).** fj ckpts were never banked (NODE_SCRATCH, wiped) and its pred dumps carry no boxes, so the run RETRAINS fj (tag fj2, identical harness/seed/jitter; ckpt banked to $S/ckpts/fj2_tuned.pt — also a prerequisite for the H2O zero-shot arm), re-runs GT-box + det-box as retrain-equivalence sanity anchors (expect ~20.8 / ~41.6), then evals on the EXACT detbox v3 boxes via a new `--box file` driver mode (patch_d210.py, clamp-to-frame like the gt/det paths). Decision: hamer_fj2_v3box_eval.json vs ours 35.6 on identical boxes. **RESULT (D210_ALL_DONE 2026-07-19 13:12, full-157, per-seq): CROSSOVER REVERSED.** fj2 on the exact v3 boxes = **C_abs 23.4 / wrist 22.3 / C_rr 13.2** vs ours(jitterrob) 35.6/34.8/26.5 on identical boxes — the full fine-tune wins E2E by 12.2mm (34%) once inputs are matched, and its GT→det degradation is SMALLER (+4.4 vs our +8.3). Sanity anchors reproduced (fj2 GT 19.0 vs orig fj 20.8; own-det 38.6 vs 41.6; retrain ~2mm stronger = run-to-run variance on the FT, worth a table note). READ-OUT: the pass-5/6 "detector-robustness of the frozen head" claim is REFUTED — the old +20.8-vs-+8.3 slope compared different box sources (HaMeR's own live detector produces harder boxes than v3). fu/fj own-detector rows stay in the table but relabeled; footnote d rewritten; ours+jitter det-box bolding removed (fj2-v3 row now best). Remaining affirmative story = parameter-efficiency (within ~3mm GT at ~1/100 trainable params) + analysis; gate 1 of the pass-5 survival path is DEAD. fj2 ckpt banked ($S/ckpts/fj2_tuned.pt) — reusable for the H2O zero-shot arm. |

Confirmed SOUND by the audit (stop re-litigating): metric definitions + self-tests, units, smplx-16
joint order incl. OP2SMPLX16, frame/clip alignment (offset=start, stride match), valid-mask asymmetry
actually favors baselines. Rebuttal rule: no new experiments in rebuttal — everything above must be
in the submission.

## E. Known infra facts (operational)

- SLURM: `-w` node pins NOT honored; node /tmp wiped between jobs; sbatch spools scripts at submit
  (patching a queued job's file does nothing — resubmit); compute nodes have internet; QOS = 1 running /
  3 submitted per user.
- Quotas: scratch ~65G+inodes (FULL — never touch venv_gb10), /home 20G. Checkpoints → node /tmp during
  runs; bank via `srun --jobid=<id> --overlap` watchers; durable storage = HF (`~/.hf_token`, repo in
  `~/.hf_repo`).
- Feature cache: stride-16 keyed `<seq>_<offset>.pt`; cached training must set data.clip_stride=16.
- Eval cost ~2.3 min/seq (stride 8), ~1.2 (stride 16); full-157 ≈ 3h at stride 16.
- Reliable big-file transfer Mac↔cluster: SSH ControlMaster + `scp -o ControlPath=/tmp/clctl`
  (pty base64 corrupts MB-scale payloads).
