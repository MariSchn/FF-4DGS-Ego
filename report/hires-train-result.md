# Phase 2: high-res hand-crop training — PA-MPJPE result

**Question.** Does training the hand head WITH the high-res hand-crop branch
(`model.hires_hand: true`, native 192 px crops fused into the HaMeR head) improve
PA-MPJPE over the 224 px baseline?

**Status: hires run LAUNCHED & EXECUTING on the cluster (job 100000); baseline
confirmed (46.76 mm). The hires number could not be read back this session because
cluster access was lost mid-run** — the SSH password file `/tmp/.ethpw` that every
`expect` helper reads was wiped when `/tmp` was cleaned at the date rollover, and the
cluster accepts only publickey (no `~/.ssh/id_rsa` present) or password. The Slurm job
is detached and keeps running regardless; its log and checkpoint land in `$HOME`. See
**"How to retrieve the hires result"** below — it is a 2-command read once `/tmp/.ethpw`
is restored.

---

## Headline (apples-to-apples, same clips)

| Model | PA-MPJPE (21-jt, Procrustes) | split / clips |
|---|---|---|
| 224 px warm-start **baseline** | **46.76 mm** | subject4 held-out, 1953 clips |
| **hires (192 px crops)** | _pending (job 100000)_ | subject4 held-out, ~1953 clips |

**Note on the 43.6 reference.** The 43.6 mm in the brief was measured on the older
mixed `h2o_packed` eval set. On the clean cross-subject split used here
(**train subject1 → test subject4**), the *same* warm-start checkpoint scores
**PA-MPJPE 46.76 mm** (21-joint). That 46.76 is the correct apples-to-apples
baseline for this run — both numbers are produced on the identical subject4 clips
with identical eval code, so the hires-vs-baseline delta is clean regardless of the
absolute level. (For context, EgoGrasp on H2O is ~47 mm, so 46.76 is already
SOTA-grade articulation.)

C-MPJPE / WRIST for the warm-start baseline on this split are high (233 / 291 mm)
because that checkpoint was never trained on H2O absolute scale; PA-MPJPE removes
the global similarity transform, so it isolates the articulation/shape quality the
hires branch is meant to improve.

---

## Setup (the run)

- **Cross-subject split (H2O standard, smallest viable):** train **subject1** (47 seqs,
  3414 clips), test **subject4** (46 seqs, 3881 train-loader clips / 1953 eval clips).
- **Hires data:** re-packed in-job at native **192 px** hand crops
  (`scripts.pack_h2o --hand_crop_px 192`), GT-joint-projected full-res bboxes. 93 npz
  total, all carrying `hand_crop_L/R`.
- **Model:** WorldMirror hand head (HaMeR) + ResNet18 hires encoder (58.0 M hand params),
  warm-started from `hand_head_final.pt` (104 missing keys = the fresh `hires_encoder.*`
  weights, by design), last 8 backbone blocks unfrozen (201.6 M).
- **Train:** lr 1e-5 cosine→1e-6, bs 2 × grad-accum 4, bf16 AMP, abs+RR L2 loss,
  grad-clip 5.0, **max_steps 1500** (bounds the 8 h cap).
- **Eval:** `scripts.eval_cmpjpe --hires --joints21 --subject subject4` on the trained
  best checkpoint, plus the 224 px baseline on the same clips. PA-MPJPE = Umeyama
  Procrustes (CPU).

## Disk constraint — how it was handled

- **Pack → node-local `$TMPDIR`** on the compute node (studgpu-spark02, **813 GB free**),
  never to scratch. The 2×37 GB ego tars are streamed (`curl | pack_h2o`), never stored.
- **First attempt (job 99978) hit the exact quota wall:** `os.makedirs` of the scratch
  `output_dir` failed with `OSError: [Errno 122] Disk quota exceeded`
  (`/work/scratch/dmonopoli/checkpoints/h2o_hand_hires`) — scratch is inode/quota-exhausted
  and cannot be freed, as warned. The pack + model build + warm-start all succeeded; only
  the checkpoint dir write failed, so training aborted before step 1.
- **Fix (job 100000):** checkpoint `output_dir` moved to **`$HOME`**
  (`/home/dmonopoli/ckpt_h2o_hand_hires`, 11 TB free, ~unlimited inodes). One self-contained
  job: pack → train → eval → save ckpt to $HOME. Slurm logs to `$HOME`.

## Code touched for the eval

- `scripts/eval_cmpjpe.py`: added `--hires` (feeds packed `hand_crops` into the hires-hand
  branch at eval; reads `crop_px` from the config) — without it the hires encoder would get
  no input and the comparison would be invalid. Pushed to the cluster.
- `configs/exp_h2o_hand_hires.yaml`: Phase-2 config (`hires_hand: true`, `crop_px: 192`,
  `max_steps: 1500`, `output_dir` → `$HOME`).
- `configs/_hires_train.sbatch` → `/home/dmonopoli/hires_train.sbatch`: the packing +
  training + dual-eval job.

## Jobs

- **99978** — first attempt, FAILED at train start (scratch quota). Gave the baseline:
  **PA-MPJPE 46.76 mm** (224 px, subject4, 1953 clips).
- **100000** — the real run (output → $HOME). _In progress._

## How to retrieve the hires result (after access is restored)

The blocker is purely read-access: `/tmp/.ethpw` (the cluster SSH password the `expect`
helpers read) was deleted when macOS cleaned `/tmp` at the day boundary. Restore that
0600 file with the cluster password (same as session start), then:

```bash
# 1. The dual-eval block at the END of the job log has both PA-MPJPE numbers:
expect /tmp/clssh.exp 'grep -E "EVAL HIRES|EVAL BASELINE|C-MPJPE:|PA-MPJPE:|RR-MPJPE:|WRIST:|best C-MPJPE|COMPLETE|Disk quota|Traceback" /home/dmonopoli/hires_train_100000.out | tail -40'

# 2. The trained checkpoint (saved to $HOME, NOT scratch):
expect /tmp/clssh.exp 'ls -la /home/dmonopoli/ckpt_h2o_hand_hires/'
```

The job log prints, in order: the training VAL C-MPJPE curve, then
`=== EVAL HIRES ckpt (PA-MPJPE, 21-joint, subject4) ===` with the **hires PA-MPJPE**,
then `=== EVAL BASELINE ... ===` which should reproduce **46.76 mm**. Compare the two
PA-MPJPE lines — that single delta is the answer to "did hires help?".

If the job log shows it hit the 8 h wall mid-training, the **best checkpoint is still
saved** (the trainer checkpoints on every val improvement to `best_cmpjpe.pt`); re-run
just the hires eval against it:

```bash
expect /tmp/clssh.exp 'cd /home/dmonopoli/FF-4DGS-Ego && source venv_gb10/bin/activate && \
  python -m scripts.eval_cmpjpe --config configs/exp_h2o_hand_hires.yaml \
    --ckpt /home/dmonopoli/ckpt_h2o_hand_hires/best_cmpjpe.pt \
    --h2o <node-local pack is gone; re-pack subject4 or point at a persisted pack> \
    --subject subject4 --joints21 --hires --limit 400'
```
(Re-eval needs the subject4 hires npz; the node-local pack is wiped at job end. If a
re-eval is needed, re-pack only subject4 to `$HOME` once and keep it.)

## Next step if hires helps

Scale train to **subject1-3** (the full cross-subject train set) at 192 px and re-eval on
subject4; the pipeline (pack flag, dataset, head, eval `--hires`) is already in place.
Persist the pack to `$HOME` (not node-local) once, so eval re-runs don't re-stream 37 GB
tars — node-local `$TMPDIR` is wiped when the job ends.
