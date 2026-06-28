# High-resolution hand-crop encoder branch (Phase 1: implement + smoke test)

## Why

Our PA-MPJPE on H2O is 43.6, largely because the whole pipeline runs at 224 px.
WiLoR scores 47.6 on our 224 px protocol vs ~6-9 at full res, so the resolution is
the handicap, not the head. This adds a **high-resolution hand-crop encoder branch**:
a small ResNet consumes a native-res (256 px) crop tightly around each hand and its
features are fused into the HaMeR head's cropped-feature context right before MANO
regression. Behind a config flag (`model.hires_hand`), default **false** — the 224 px
path is byte-for-byte unchanged.

## What changed (files, one line each)

- `diffsynth/auxiliary_models/worldmirror/models/heads/hires_hand_encoder.py` — **new**: `HiResHandEncoder`, a torchvision ResNet (18/34/50) trunk → spatial feature map → `[M, T, dim]` tokens (ImageNet-normalised input, pretrained if available).
- `diffsynth/auxiliary_models/worldmirror/models/heads/hamer_head.py` — `HamerManoHead` gains `hires_hand` / `hires_hand_kwargs` ctor args and a `hand_crops` forward arg; encodes the per-hand native crop and **concatenates** its tokens onto each hand's crop-context before the regression cross-attention (variable `n_ctx`); `enhanced_crop_tokens` (the GS-injection view) is left at its original shape.
- `diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py` — reads `hires_hand` / `hires_hand_kwargs`, passes them to `HamerManoHead`, forwards `views["hand_crops"]` to the head when `hires_hand` is on, and records `hires_hand` in the stored config.
- `scripts/pack_h2o.py` — `--hand_crop_px N` (default 0 = off): keeps the original full-res RGB per frame and saves native per-hand crops `hand_crop_L`/`hand_crop_R` `[N,P,P,3] uint8` derived from GT-joint-projected bboxes (`_bbox_from_joints_fullres`, full-frame pixels — no center-square-crop adjustment since we crop the original image).
- `scripts/h2o_dataset.py` — `H2ODepthDataset(hires_hand=, hand_crop_px=)` returns `hand_crops` `[S,2,3,P,P]` in [0,1]; uses the packed `hand_crop_L/R` when present, else **upscales the 224 frame as a shape-only placeholder** (smoke tests before the real repack).
- `scripts/train_hand_head.py` — `build_views(..., hand_crops=None)` now threads `hand_crops` into `views`.
- `scripts/train_h2o_hand.py` — reads `model.hires_hand` / `hires_hand_kwargs.crop_px`, passes them to the dataset, and feeds `batch["hand_crops"]` into `build_views` in both the train and eval loops.
- `configs/exp_h2o_hand.yaml` — added `model.hires_hand: false` + `hires_hand_kwargs {arch: resnet18, pretrained: true, crop_px: 256}`.

## Fusion point (the key design choice)

The HaMeR head ROI-aligns backbone features in the hand bbox → `crop_to_global`
cross-attn → per-hand context tokens `[N*2, crop^2, dim]`. The hires encoder produces
`[N*2, T, dim]` tokens from the native crop; these are **concatenated along the token
axis** so the MANO regression transformer cross-attends over BOTH the 224 px feature
crop and the high-frequency native crop. Dims stay at `dim` throughout; the regression
attention is over an arbitrary token count, so no other shapes change. Invalid/absent
hands have their hires tokens zeroed (same as the existing crop tokens).

## Verification done

- `py_compile` on all 8 edited files — OK.
- Isolated CPU forward+backward of `HamerManoHead` with `hires_hand=True`: params `[1,2,64]`, conf `[1,2,1]`, `enhanced_crop_tokens` unchanged at `[4,64,256]`; gradients finite into **both** the hires encoder (`proj.weight.grad`) and the backbone tokens. hires-OFF path identical in shape.
- `pack_h2o` crop helpers: valid hand → non-zero 256×256 crop, absent hand → zeros, bbox projects to sane pixels.
- Dataset: placeholder, real (`hand_crop_L/R`), and OFF paths all return correct shapes; OFF emits no `hand_crops` key.
- **End-to-end gb10 smoke test (job 99971, venv_gb10 aarch64, real `train_h2o_hand.py`, `hires_hand: true`, `max_steps=3`)**: **COMPLETED, exit 0**. Built the model (hand head 58.0M params incl. the ResNet18 hires encoder), indexed 184 H2O seqs, baseline eval, then 3 forward+backward steps. Loss decreased `0.674 → 0.597 → 0.506` (abs `558 → 468 → 401 mm`), gradients finite (`320 → 98 → 69`). This used the **placeholder** crop path (224→256 upscale) with a fresh head over 3 steps, so the absolute mm are meaningless — it only proves the branch trains end to end without shape/autograd errors. Real accuracy needs the `--hand_crop_px 256` repack (below).

## How to re-pack the high-res data (for real training)

Re-run the H2O pack with `--hand_crop_px 256` (streams the ego tar; needs only PIL+numpy,
runs on `interactive-cpu`). This re-creates the npz with the extra `hand_crop_L/R` arrays
**in addition to** the existing 224 px arrays:

```bash
# per subject (subject1..subject4); H2O credentials in $U/$P
curl -sL -u "$U:$P" https://h2odataset.ethz.ch/data/dataset/subject1_ego_v1_1.tar.gz \
  | python3 -m scripts.pack_h2o --out /work/scratch/dmonopoli/h2o_packed_hires --res 224 --hand_crop_px 256
```

Note: the hires npz is larger (extra 2×256×256×3 uint8 per frame). Point training's
`data.data_root` at the hires dir. Storage: `/work` is at quota — clear space or pack to
a scratch path with room before repacking all four subjects.

## How to train (Phase 2 — the human kicks this off)

1. Re-pack with `--hand_crop_px 256` (above) → `h2o_packed_hires`.
2. In `configs/exp_h2o_hand.yaml`: set `model.hires_hand: true` and `data.data_root: /work/scratch/dmonopoli/h2o_packed_hires`.
3. Submit on a gb10 node with venv_gb10 (template: `/home/dmonopoli/smoke_hires.sbatch`, account `3dv`, `--gpus=gb10:1`, logs to `$HOME`):

```bash
python -m scripts.train_h2o_hand --config configs/exp_h2o_hand.yaml
```

The hires encoder is part of `model.hand_head`, so it is already included in the
optimizer's `hand_params` — no trainer change needed.

## Open issues / notes

- **Warm-start mismatch**: the existing `hand_head_final.pt` warm-start predates the hires
  branch; loading is `strict=False`, so the new `hires_encoder.*` params init fresh (fine).
  The smoke config set `warm_start_hand_head: null` to keep logs clean; for real training
  keep the warm-start — only the new encoder weights are uninitialised.
- **Placeholder vs real crops**: until the repack lands, the dataset upscales the 224 frame
  (shape-correct, no real high-frequency detail). Real gains require the `--hand_crop_px 256`
  repack. The smoke test exercises the placeholder path on purpose.
- **Bbox source**: hires crops are derived from GT joints (a detector stand-in, same protocol
  as `bboxes_from_joints`/`eval_cmpjpe`). At true inference time a real hand detector would
  supply the bbox; for the H2O protocol GT-derived bboxes are standard.
- **Encoder choice**: started with ResNet18 (pretrained) for speed; `arch` is configurable
  (`resnet34`/`resnet50`). A ViT could be swapped in later behind the same `hires_hand_kwargs`.

## Exact next step to start real training

Repack subject1-4 with `--hand_crop_px 256` to a scratch dir with free space, point
`data.data_root` there, set `model.hires_hand: true`, and submit `train_h2o_hand` on gb10.
