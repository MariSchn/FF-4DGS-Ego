# FF-4DGS-Ego

Feed-forward **4D Gaussian-Splatting reconstruction with egocentric hand recovery**.
A WorldMirror reconstructor backbone predicts per-frame 4D Gaussians and camera
parameters from monocular video; a HaMeR/MANO hand head, coupled to the
reconstructed scene, recovers metric-scale 3D hands.

> This repository began as a fork of DiffSynth-Studio / NeoVerse. The video
> generation (diffusion) stack has been removed; only the reconstruction +
> hand-recovery solution remains.

## Setup

Tested on CUDA 12.1 / PyTorch 2.3.1 and CUDA 12.8 / PyTorch 2.7.1. Install
PyTorch first (it is intentionally not pinned in `requirements.txt` so you can
match your CUDA version), then the rest:

```bash
conda create -n ff4dgs python=3.10 -y && conda activate ff4dgs

# PyTorch — pick the wheel matching your CUDA (example: CUDA 12.1)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Gaussian-splatting deps (match the torch/CUDA above)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git

git submodule update --init --recursive   # HaMeR
```

Place model weights under `models/` — the WorldMirror reconstructor checkpoint
(`models/NeoVerse/reconstructor.ckpt`, shipped in the
[NeoVerse release](https://huggingface.co/Yuppie1204/NeoVerse)), plus MANO and
the HaMeR submodule. See `models/Put checkpoints here.txt`.

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
| `examples/hot3d_sample/` | HOT3D egocentric sample (the data the hand head uses) |
| `examples/videos/` | Demo input clips for the reconstruction script |

## External libraries

All third-party libraries used, with the versions this project was developed and
tested against (Python 3.10, CUDA 12.8 / PyTorch 2.7.1). PyTorch and the
Gaussian-splatting CUDA extensions are installed separately (see [Setup](#setup))
so they match your CUDA toolkit; everything else is pinned in
[`requirements.txt`](requirements.txt).

**Installed separately (CUDA-matched):**

| Library | Version |
|---------|---------|
| [torch](https://pytorch.org) | 2.7.1 (cu128); also tested 2.3.1 (cu121) |
| [torchvision](https://github.com/pytorch/vision) | 0.22.1; also tested 0.18.1 |
| [torch-scatter](https://github.com/rusty1s/pytorch_scatter) | 2.1.2 |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | 1.5.3 |

**Pinned in `requirements.txt`:**

| Library | Version |
|---------|---------|
| [transformers](https://github.com/huggingface/transformers) | 4.57.6 |
| [safetensors](https://github.com/huggingface/safetensors) | 0.7.0 |
| [einops](https://github.com/arogozhnikov/einops) | 0.8.2 |
| [numpy](https://numpy.org) | 2.4.6 |
| [Pillow](https://python-pillow.org) | 12.2.0 |
| [tqdm](https://github.com/tqdm/tqdm) | 4.68.0 |
| [sentencepiece](https://github.com/google/sentencepiece) | 0.2.1 |
| [imageio](https://github.com/imageio/imageio) (+ [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)) | 2.37.3 (ffmpeg 0.6.0) |
| [opencv-python](https://github.com/opencv/opencv-python) | 4.13.0.92 |
| [decord](https://github.com/dmlc/decord) | 0.6.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | 0.36.2 |
| [modelscope](https://github.com/modelscope/modelscope) | 1.37.1 |
| [lpips](https://github.com/richzhang/PerceptualSimilarity) | 0.1.4 |
| [scikit-image](https://scikit-image.org) | 0.26.0 |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | 1.9.0 |
| [plyfile](https://github.com/dranjan/python-plyfile) | 1.1.4 |
| [jaxtyping](https://github.com/patrick-kidger/jaxtyping) | 0.3.10 |
| [scipy](https://scipy.org) | 1.17.1 |
| [matplotlib](https://matplotlib.org) | 3.10.9 |
| [evo](https://github.com/MichaelGrupp/evo) | 1.36.5 |
| [e3nn](https://github.com/e3nn/e3nn) | 0.6.0 |
| [addict](https://github.com/mewwts/addict) | 2.4.0 |
| [moviepy](https://github.com/Zulko/moviepy) | 1.0.3 |
| [trimesh](https://github.com/mikedh/trimesh) | 4.12.2 |
| [viser](https://github.com/nerfstudio-project/viser) | 1.0.30 |
| [omegaconf](https://github.com/omry/omegaconf) | 2.3.0 |
| [tensorboard](https://github.com/tensorflow/tensorboard) | 2.20.0 |
| [wandb](https://github.com/wandb/wandb) | 0.27.1 |
| [pandas](https://pandas.pydata.org) | 3.0.3 |
| [smplx](https://github.com/vchoutas/smplx) | 0.1.28 |
| [projectaria-tools](https://github.com/facebookresearch/projectaria_tools) | 2.1.1 |

**Model assets** (downloaded separately, not pip packages): the
[MANO](https://mano.is.tue.mpg.de) hand model and the
[HaMeR](https://github.com/geopavlakos/hamer) submodule. See [Setup](#setup).

> On aarch64 (e.g. GH200) the training environment substitutes the equivalent
> `opencv-python-headless` and `decord2` builds, since the stock `opencv-python`
> / `decord` wheels are x86-only.

## Notes

- **No Android app / APK**: this project is Python/PyTorch only, so no `.apk` is included.
- **Sample data**: a sample of the egocentric data this project actually uses is in
  [`examples/hot3d_sample/`](examples/hot3d_sample/): a HOT3D frame with our MANO
  hand-mesh overlay, plus a provenance README and a download link. The
  reconstruction demo also ships generic monocular input clips under
  `examples/videos/` (for example `examples/videos/robot.mp4`).

## License

See `LICENSE.txt`.
