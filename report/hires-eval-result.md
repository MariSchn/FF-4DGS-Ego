# Hires hand-crop: decisive eval (Task a) + full-data train (Task b)

**Question (a).** On the *identical* subject4 held-out clips, does the high-res
hand-crop model (`hires_hand: true`, native 192 px crops) beat the 224 px baseline's
clean PA-MPJPE of **46.76 mm**?

**Status.** _Filled in when job 100028 completes — see jobs below._

---

## Task (a) — hires vs baseline, same subject4 clips (21-joint, Procrustes)

Standalone gb10 job **100028** (`taska_eval.sbatch`): packs subject4 at
`--hand_crop_px 192` to node-local `$TMPDIR`, then evaluates both checkpoints on the
SAME clips:

- **hires:** `eval_cmpjpe --hires --ckpt $HOME/ckpt_h2o_hand_hires/best_cmpjpe.pt --joints21` (subset hires ckpt from job 100000)
- **baseline 224 px:** `eval_cmpjpe --ckpt /work/scratch/dmonopoli/checkpoints/h2o_hand/best_cmpjpe.pt --joints21` (NO `--hires`)

| Model | PA-MPJPE | C-MPJPE | RR-MPJPE | WRIST | clips (N) |
|---|---|---|---|---|---|
| 224 px baseline (trained) | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| hires (192 px crops) | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

Reference: clean baseline PA on subject4 was **46.76 mm** (prior session, 1953 clips).

**Verdict:** _pending job 100028 — does hires PA beat 46.76?_

---

## Task (b) — full hires train (subject1+2+3 → subject4)

Self-contained gb10 job **100029** (`taskb_train.sbatch`, dependency
`afterany:100028`): packs subject1+2+3 (train) + subject4 (eval) at
`--hand_crop_px 192` → node-local, trains `train_h2o_hand` with `hires_hand: true`
to the 8 h cap, saves best ckpt to `$HOME/ckpt_h2o_hand_hires_full/`, then evals
PA-MPJPE on subject4 (hires + baseline, same clips).

- **Job id:** 100029
- **Status:** PENDING (`QOSMaxJobsPerUserLimit`, dependency `afterany:100028` unfulfilled) — runs after Task (a). 8 h run; not waited on.

This is the fair full-data hires number (the subset run / job 100000 trained on
subject1 only).

---

## Jobs

| Job | What | State |
|---|---|---|
| 100000 | subset hires train (subject1 only) → produces the (a) hires ckpt | RUNNING (near 8 h cap) |
| 100028 | Task (a): hires-vs-baseline eval on subject4 | PENDING → runs when 100000 frees the QOS slot |
| 100029 | Task (b): full hires train subject1-3, eval subject4 | PENDING (dep afterany:100028) |
