# What this paper claims

Set by the group lead, 2026-08-19, and it decides what work counts:

**The contribution is unifying world-space scene reconstruction with hand tracking, and showing the
two improve each other.** Hand pose does not need to be state of the art in the camera frame. A
result that improves camera-frame accuracy without demonstrating the coupling does not serve the
claim.

"Improve each other" is two directions and both need evidence:

- **scene to hand**, which we have. Global scene features condition the hand head, and the depth
  coverage of the training mixture governs transfer.
- **hand to scene**, which we do not have yet. The injection measurement is confounded: it was made
  on a checkpoint whose Gaussian head had collapsed, where both arms degrade over training and the
  best number in the run is the no-injection arm at epoch 1. See docs/confirmed-defects.md entry 3.

So the load-bearing open item is a hand-to-scene improvement measured on a model whose Gaussian
branch is not degenerate.

# Working in this repo

**Read [docs/confirmed-defects.md](docs/confirmed-defects.md) start to finish before touching a
job, a metric, or a number.** Not only when something looks wrong: several entries describe defects
that produced *plausible* numbers for weeks. It holds measured defects with their evidence, their
reproduction and, most usefully, what each one looks like from outside when it does not raise.
Add to it only what you have confirmed: the bar for an entry is stated at the top of that file.

## Mistakes already made here, which keep coming back

These are not hypotheticals. Each one has cost a day or more, and most have recurred after being
fixed once, which is why they are listed separately from the defect file.

**Changing a resource request to get out of the queue.** `--gres=gpumem` selects the GPU model, not
just its memory: 40g picks an A100 at 2.11 s/it and 24g a TITAN at 7.52. Lowering it once bought an
afternoon of chasing a phantom slowdown, and a second time landed a job on a compute capability the
`TORCH_CUDA_ARCH_LIST` did not cover, so gsplat had no kernel and all ten sequences failed. Cutting
`--cpus-per-task` has the same character: it changes which nodes are reachable. Treat every SBATCH
resource line as part of the experiment.

**Believing an exit code.** SLURM reported `COMPLETED` for jobs that had failed with `rc=1`, because
the sbatch ended with `ts "... rc=$?"` and exited with the status of `ts`. Every dependency in that
chain then fired on a failure. An sbatch must end with `exit $rc`. Separately, a zero exit says
nothing about whether an artifact exists: check the artifact.

**Reading a number without checking which variable it came from.** `grep -oE 'anchor=[0-9.]+'`
matched `root_anchor=0` in a parameter count and was read as the anchor loss being dead. A healthy
run was killed on that reading and the restart cost nine hours of queue. Print the whole line, not
the match.

**Trusting an averaged metric over a mixed population.** A clip of 14 good frames and 2 black ones
still averages well. When a number covers subsets that should differ, report the subsets.

**Assuming a fix generalises across stores.** HOI4D is the only store whose video is already at the
render resolution, so it is the only one where several conventions happen to hold. A change verified
on HOI4D alone has been wrong on the other five more than once.

Three habits that file exists to enforce.

**A zero exit code is not a result.** Check the artifact. A 4DGT run reported `ok=10 fail=0 rc=0`
while producing renders no scorer could read, and two eval jobs died on the walltime with no error
because a loop could not terminate.

**Verify the input arrived before trusting the output.** A measurement that does not move across
twenty decades is an instrument that was never connected, not an invariance.

**Instrument before you wait.** A job with no per-stage progress line cannot be diagnosed while it
runs, only re-run.

# Config flags that must be right

Every entry below has cost us a wrong number that a green exit hid. Check the flag against the
config before trusting any metric, and cite the file:line when you report that you did.

## World-space evaluation

Never hand-write an eval config. `scripts/_make_eval_cfg.py` exists to force these three, and its
module docstring records the three runs each trap already spoiled.

| Flag | Value | What breaks without it |
|---|---|---|
| `model.enable_gs` | `true` | No `gs_depth`, so the scene-scale solve silently falls back to `s = 1.0` and every W/WA is non-metric. Ruined h2oeval2, mix3fix, ctrleval2. Camera-frame C-MPJPE is unaffected, which is why it goes unnoticed (`scripts/_make_eval_cfg.py:7-12`) |
| `model.gs_anchor_only` | `true` | Paired with the above. Keeps the depth correspondences and skips gsplat rasterization, so correctness is nearly free (`scripts/_make_eval_cfg.py:43`) |
| `model.enable_cam` | `true` | `camera_poses` is published only under this flag. Without it the world lift uses identity poses, `s_gt_med` comes out NaN, and every scene-scale variant returns a bit-identical number (`scripts/_make_eval_cfg.py:48-51`). `eval_world_space.py` raises rather than continuing |

`--keep_gs_off` opts out, and is correct only for a camera-frame eval where scale is irrelevant.

## Training

| Flag | Value | What breaks without it |
|---|---|---|
| `loss_weights.kp3d_abs` | `1.0` | No absolute supervision at all. C_abs degrades several-fold while C_rr looks healthy, so the failure reads as a data problem. The measured control is C_abs 725 against C_rr 131. Guarded at `scripts/train_hand_head.py:1925-1941`, which aborts unless `--allow_recipe_drift` |
| `loss_weights` (rest) | `transl 1.0, kp3d 0.05, kp2d 0.0, betas 0.01, global_orient 0.01, hand_pose 0.01` | `PROVEN_LOSS_RECIPE`, `scripts/train_hand_head.py:1762`. Drift from it is reported as a note, not an abort |
| `loss_weights.kp2d` | `0.0` | Deliberately off. Its implementation hardcodes the Aria 1408 px frame and a 90-degree rotation, while HOI4D and H2O cache GT 2D unrotated in res-pixel units, so the term compares two frames at two scales |
| `metric_scale.enable` | `true`, clamp `[0.1, 10.0]` | `scripts/train_hand_head.py:1945` |
| `data.resolution` | present, e.g. `[224, 224]` | `set_default_frame_width` is called from it (`scripts/train_hand_head.py:2174`, `scripts/eval_world_space.py:1187`). Without it the width falls back to `2*cx`, which on HOI4D is 228.55 against a true 224 because the principal point is off-centre, and every depth sample lands off-pixel |
| `visualization.mano_model_folder` | set | `scripts/train_hand_head.py:2003` raises |
| `training.output_dir` | unique per arm | The trainer resumes whenever its output directory holds a checkpoint. One gsinj arm resumed at epoch 2 with a decayed LR while the other started clean, both reported rc=0, and the 6.33 dB gap between them was an artifact |

## Training the Gaussian head

`enable_gs: true` alone does not train it. All three are needed together, as in
`configs/exp_gsinj2_on.yaml`:

- `model.enable_gs: true`
- `loss_weights.gs_l1` and `gs_lpips` above zero
- `model.freeze_gs_head: false`

A config with `enable_gs: false` and `gs_l1: 0.0` leaves the head as frozen NeoVerse. An eval then
forces `enable_gs` on and produces a depth map that exists without ever having been trained, and
nothing in the output distinguishes the two cases.

## Reading a result

A zero exit code says nothing about whether the output can enter a table. 4DGT reported
`ok=10 fail=0 rc=0` while producing renders that no scorer could use. Check the artifact, not the
return code.
