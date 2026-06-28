# Resume runbook — 2026-06-27 (B3 + convergence, launch-ready)

Everything below runs on the **cluster login node**. I can't SSH, so these are the exact
copy-paste commands. Cluster is at commit `d93dfce` from the contact run; if unsure, sync first:

```bash
cd /home/dmonopoli/FF-4DGS-Ego
git fetch origin && git reset --hard origin/feat/hand-scene-metric-coupling
```

(No new commits since the contact run, so this is a no-op safety check.)

---

## De-risked already (so you don't waste a job)

- **B3 packer URL + creds: VERIFIED LIVE.** `subject4_ego_v1_1.tar.gz` → HTTP 200, **43 GB**.
  Creds authenticate (subject1 also 200). The `_v1` fallback correctly 404s, so the packer's
  primary `_v1_1` URL is the right one — no fallback needed. The 43 GB **streams** through
  `pack_h2o.py` and never lands on disk (no quota hit), fits the 4 h CPU window.

## ⚠ One hygiene flag (not blocking)

These 5 scripts/configs the launchers call are **untracked in git** (present on the cluster
from earlier manual staging, and `git reset --hard` preserves untracked files, so the launchers
still run): `scripts/pack_h2o.py`, `scripts/eval_cmpjpe.py`, `scripts/train_hoi4d_depth.py`,
`configs/exp_h2o_hand.yaml`, `configs/exp_hoi4d_depth.yaml`. They are NOT on `origin`, so a
`git clean -fdx` on the cluster would break the launchers. **Commit them when you're back** (I
held off mutating the deploy while you were away and couldn't verify a remote checkout).

---

## Launch sequence

Set the H2O creds first (temporary academic creds — from our chat; expire ~6 days; do NOT commit):

```bash
export H2O_USER=<from chat>   # the 5-char user
export H2O_PASS=<from chat>   # the hex pass
```

### DEFAULT (set-and-forget — both results waiting when you next check)

```bash
cd /home/dmonopoli/FF-4DGS-Ego
sbatch pack_h2o_subject4.sh            # CPU, concurrent, ~20-40min, does NOT touch the QOS=1 GPU slot
sbatch run_hoi4d_converge_gb10.sbatch  # GPU (QOS=1), ~3-4h — runs productively while the pack streams
# once BOTH of the above clear `squeue -u dmonopoli`:
sbatch run_b3_h2o_subj4.sbatch         # GPU, ~1h  (needs the subject4 pack to have finished)
```

### If you want B3 FIRST (it's your priority #1 and cheap)

```bash
sbatch pack_h2o_subject4.sh            # CPU
# wait for it to finish (watch `squeue -u dmonopoli`; ~20-40min), THEN:
sbatch run_b3_h2o_subj4.sbatch         # GPU, ~1h — the held-out H2O Table-2 numbers
sbatch run_hoi4d_converge_gb10.sbatch  # GPU, ~3-4h — start after B3 clears the slot
```

QOS=1 = one GPU job at a time. The packer is CPU so it overlaps either GPU job freely.

### Optional 3rd GPU job — Cyrus #31 (root-depth supervision, lowest priority)

```bash
sbatch run_hoi4d_rootdepth_gb10.sbatch   # GPU, ~1-1.5h (3 short arms at the 120-step point)
```

Tests Cyrus's "supervise root translation with larger weights" config-only: the knob
`loss_weights.transl_z_weight` (already wired, defaulted to 1.0) up-weights the camera-depth
axis of the root-translation loss. Arms: control(1.0) vs rootz3(3.0) vs rootz3_absbump(3.0,
kp3d_abs 0.5), anchor off, C-abs isolated. **Decision:** rootz3 C-abs < control(~105.8) =>
up-weighting helps; a dedicated root-depth *head* is the follow-up build only if the knob
plateaus. Already committed + pushed (commit 73abda3); fully simulated locally (parses, the
override creates transl_z_weight=3.0 -> ParameterLoss axis-2 weight). Run it after B3 +
convergence since it's lowest priority.

---

## Monitoring

```bash
squeue -u dmonopoli                                            # what's queued/running
ls -t /work/scratch/dmonopoli/joblogs/ | head                 # newest logs
tail -f /work/scratch/dmonopoli/joblogs/<jobid>_h2opack4.out  # packer progress
tail -f /work/scratch/dmonopoli/joblogs/b3_<jobid>.out        # B3 result
tail -f /work/scratch/dmonopoli/joblogs/hoi4dconv_<jobid>.out # convergence (look for 'best depth residual ... cm')
```

## What each result means

- **B3** — read C-MPJPE / WRIST / PA-MPJPE for the **fine-tuned** head on subject-4. These are
  the true held-out Table-2 numbers (replace the mixed-set 58.0 / 36.6). The default-head row
  (never saw H2O) is always held-out. CAVEAT: if fine-tuned C-MPJPE comes back ≈58, train/test
  overlapped → needs a subject1-only retrain for a clean number.
- **Convergence** — the final `best depth residual ... cm` line is the converged Table 4/6
  number, replacing the SIGTERM'd 13.4 best-so-far. Target: sub-10 cm. The TERM trap copies
  `best_depth.pt` to `/work/.../hoi4d_depth_conv/` and `~/hoi4dconv_best_depth.pt` even on a
  wall-clock kill.
