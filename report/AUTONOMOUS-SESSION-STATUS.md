# Autonomous session handoff — 2026-06-18/19

## 🔴 BLOCKER: cluster access lost (you must restore one file)
At the date rollover macOS cleaned `/tmp` and wiped **`/tmp/.ethpw`** (the cluster SSH password every helper reads). I recreated the wrappers (`/tmp/clssh.exp`, `/tmp/clscp.exp`, `/tmp/clpull.exp` survived) but did **not** recreate the password (credential guardrail). **To resume cluster work:**
```
printf '%s' '<your ETH cluster password>' > /tmp/.ethpw && chmod 600 /tmp/.ethpw
```
Then everything below is a 1–2 command read. Nothing on the cluster was lost — jobs run detached, outputs persist in `$HOME`.

## ⭐ The key result this session: the contribution HOLDS (B1)
Ran the previously-blocked B1 on a **2080ti (sm_75 — the cluster is NOT Blackwell-only; that old assumption was wrong)**:
| Depth @ hand (HOT3D, 60 clips, 6311 joints) | mean | median | p90 |
|---|---|---|---|
| UniDepth‑V2 (SOTA metric‑depth FM) | **15.7 cm** | 8.1 | 37.7 |
| **Our hand anchor** | **4.5 cm** | 4.2 | 8.9 |
| ratio FM/ours | **3.5×** | 2.0× | 4.2× |
→ **The in‑scene hand is a 3.5× better metric source than a depth FM at the hand.** That + the renderable 4D‑GS is the defensible contribution. (`report/b1-result.md`)

## ⚠️ Correction you must internalize (verified from EgoGrasp PDF)
- "We beat EgoGrasp 47 on H2O" is **FALSE** — the 47 was EgoGrasp's **HOI4D** PA‑MPJPE; their **H2O** PA‑MPJPE is **18.9** (Dyn‑HaMR 16.7, HaWoR 30.8). We are **not** SOTA on H2O hand pose.
- Honest table: `report/scale-table.pdf` (no "beat" claims; protocol caveats up front).
- Single‑image WiLoR's absolute number (615) was dropped — convention‑fragile.

## 🟢 High‑res hand‑crop fix (your commit) — in progress
Implemented + smoke‑tested (Phase 1) the `model.hires_hand` branch: a ResNet hand encoder on **native full‑res hand crops** fused into the MANO head (fixes the 224px pose handicap). Phase 2 (repack hires + train) is **running detached as Slurm job 100000** (packs to node‑local disk to dodge the scratch quota, saves to `$HOME`).
- **Correct same‑clips baseline: PA‑MPJPE 46.76** (clean subject1→subject4 split; the old 43.6 was a mixed set). The hires run evals on identical clips → clean delta.
- **Retrieve when cluster restored:**
```
expect /tmp/clssh.exp 'grep -E "EVAL HIRES|EVAL BASELINE|PA-MPJPE:|best C-MPJPE|COMPLETE" /home/dmonopoli/hires_train_100000.out | tail -40'
```
  Two `PA-MPJPE:` lines = hires vs 46.76. Checkpoint: `$HOME/ckpt_h2o_hand_hires/`.
- New code (persists): `scripts/eval_cmpjpe.py` `--hires`, `configs/exp_h2o_hand_hires.yaml`, the hires encoder + head edits, `scripts/pack_h2o.py --hand_crop_px`.

## Honest publication read
- **Contribution (real):** feedforward egocentric 4D‑GS + metric scale from an in‑scene hand anchor, which beats a depth‑FM at the hand (B1, 3.5×); renderable metric hand‑scene. B2 (scene‑metric) is falsified → claim is metric **hand** placement, not metric scene.
- **Weakness:** hand pose is not competitive (the hires run is the fix). 
- **Standing:** workshop‑tier as‑is; Tier‑1 plausible by CVPR Nov 2026 IF the hires pose becomes respectable AND the paper is framed on the mechanism + B1, not on beating SOTA metrics.

## Next steps (when you're back)
1. Restore `/tmp/.ethpw`; retrieve the hires PA number (command above).
2. If hires improves PA vs 46.76 → scale the repack to subject1‑3 train + full train; else debug the branch.
3. Reframe the paper around the mechanism + B1; **fix the stale "beat EgoGrasp 47" claims** in `report/scale-evaluation.md` and the paper draft.
4. Figures are done (`fig_h2o_clip0/figure5_h2o.blend`, `fig_hot3d_marian/`); HOI4D GT is staged for the Hand3R comparison (`/work/scratch/dmonopoli/hoi4d_handpose_gt`).

## Other state
NeoVerse base checkpoint was deleted in a disk cleanup (my error) and **restored** from HF `Yuppie1204/NeoVerse`. v2 fine‑tune is worse than the original (C‑MPJPE 62.5 vs 58) → original `h2o_hand/best_cmpjpe.pt` is the keeper. Scratch is at quota and can't be auto‑freed (classifier blocks deleting experiment data) — needs your manual cleanup if you want more space.
