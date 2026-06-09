# FF-4DGS-Ego

Feed-forward **4D Gaussian-Splatting reconstruction with egocentric hand recovery**.
A WorldMirror reconstructor backbone predicts per-frame 4D Gaussians and camera
parameters from monocular video; a HaMeR/MANO hand head, coupled to the
reconstructed scene, recovers metric-scale 3D hands.

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
python scripts/reconstruct_4dgs.py --input_path examples/videos/robot.mp4
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--render_video` | off | Re-render input views and save `render.mp4` |
| `--static_scene` | off | Treat scene as static (no temporal Gaussians) |
| `--output_dir` | `outputs/reconstruction` | Where to write results |
| `--num_frames` | 120 | Number of frames to process |
| `--sampling` | `uniform` | Frame-sampling strategy (`uniform` or `first`) |
| `--frame_offset` | 0 | Skip this many frames at the start |
| `--height` / `--width` | 336 | Input resolution |
| `--resize_mode` | `center_crop` | `center_crop` or `resize` |
| `--save_all_frames_ply` | off | Save a `.ply` for every frame (large) |
| `--fps` | 16 | FPS for the output render video |
| `--reconstructor_path` | `models/NeoVerse/reconstructor.ckpt` | Path to reconstructor checkpoint |

Outputs (written to `--output_dir`): `gaussians.pt`, `camera_params.json`,
`gaussians_frame0000.ply`, and an optional `render.mp4`.

## Train the hand head

```bash
python -m scripts.train_hand_head --config configs/train_hand_head.yaml
```

Hydra-style key-value overrides are accepted after `--config`:

```bash
python -m scripts.train_hand_head --config configs/train_hand_head.yaml batch_size=8
```

Ablation configs live in `configs/`. SLURM launch example: `batch.sh`.

## Evaluation

**Hand head** (keypoint and MANO metrics):

```bash
python -m scripts.eval_hand_head \
    --config configs/train_hand_head.yaml \
    --ckpt checkpoints/default/best_val_loss.pt
```

Key flags: `--sweep` (evaluate all checkpoints in `--ckpt-dir`), `--out <path.json>`,
`--batch-size`, `--num-workers`, `--device`, `--sanity`, `--limit-clips`.

**GS head** (Gaussian-splatting rendering metrics):

```bash
python -m scripts.eval_gs_head \
    --config configs/train_hand_head.yaml \
    --ckpt checkpoints/default/best_val_loss.pt
```

Accepts the same flags as `eval_hand_head` plus `--lpips-net {alex,vgg}`.

**Reconstruction quality** (PSNR / SSIM / LPIPS vs. original video):

```bash
python scripts/eval_reconstruction.py \
    --original examples/videos/robot.mp4 \
    --reconstruction outputs/reconstruction \
    --height 336 --width 336
```

Key flags: `--num_frames`, `--output_csv <path>`, `--lpips_net {alex,vgg}`, `--no_gpu`.

## Repository layout

| Path | Purpose |
|------|---------|
| `scripts/` | Reconstruction, hand-head training, evaluation, visualization |
| `diffsynth/auxiliary_models/worldmirror/` | Reconstructor backbone + heads (camera, dense, HaMeR, GS injection) |
| `diffsynth/models/` | `ModelManager` checkpoint loader (reconstructor only) |
| `diffsynth/{utils,data}/` | Video I/O and geometry helpers |
| `configs/` | Training / ablation configs |
| `models/` | Checkpoints (reconstructor, MANO, HaMeR submodule) |
| `examples/videos/` | Sample input clips |

## License

See `LICENSE.txt`.
