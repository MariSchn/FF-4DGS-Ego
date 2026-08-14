"""Write an eval copy of a training config that points at a trained checkpoint.

eval_world_space has no --ckpt flag: it loads the head via model.warm_start_hand_head. Building
this as a file rather than a nested heredoc avoids the shell expanding the Python quoting, which
silently corrupted an earlier job script.

WHY THIS FORCES enable_gs. Training configs set ``model.enable_gs: false`` because the Gaussian
branch is not supervised in a hand-only run. Copying that verbatim into a WORLD-SPACE eval is a
silent trap: without the Gaussian branch there are no ``gs_depth`` correspondences, the scene
scale estimator falls back to ``s = 1.0``, and every W/WA number comes out non-metric while the
run still exits 0. That trap has produced bad world numbers three separate times (h2oeval2,
mix3fix, ctrleval2). Camera-frame C-MPJPE is unaffected, which is exactly why it goes unnoticed.

We therefore force ``enable_gs: true`` and pair it with ``gs_anchor_only: true``, which keeps the
depth correspondences the scale solve needs while skipping the expensive gsplat rasterization, so
correctness costs almost nothing. ``--keep_gs_off`` opts out for a camera-frame-only eval where
the scale is genuinely irrelevant.

WHY THIS ALSO FORCES enable_cam. ``camera_poses`` is published only under ``enable_cam``
(worldmirror ``_gen_all_preds``), and it is what the ``gs_anchor_only`` fast path republishes as
``rendered_extrinsics``. With the camera head off, the world lift falls back to identity poses:
zero camera translation, so every scene-scale variant returns a bit-identical number and
``s_gt_med`` comes out NaN. A world eval without a camera trajectory measures hand-joint chaining,
not world placement.
"""
import sys

import yaml

args = [a for a in sys.argv[1:] if a != "--keep_gs_off"]
keep_off = "--keep_gs_off" in sys.argv[1:]
src, ckpt, out = args[0], args[1], args[2]

cfg = yaml.safe_load(open(src))
cfg["model"]["warm_start_hand_head"] = ckpt

if keep_off:
    print("WARNING: --keep_gs_off set, leaving enable_gs as-is. Any W/WA metric from this eval "
          "will be NON-METRIC (scale degenerates to 1.0). Camera-frame C-MPJPE stays valid.")
else:
    was = cfg["model"].get("enable_gs")
    cfg["model"]["enable_gs"] = True
    cfg["model"]["gs_anchor_only"] = True
    if was is not True:
        print(f"forced model.enable_gs {was} -> True (+ gs_anchor_only) so the world-space scale "
              f"solve has gs_depth correspondences; pass --keep_gs_off to override")

    was_cam = cfg["model"].get("enable_cam")
    cfg["model"]["enable_cam"] = True
    if was_cam is not True:
        print(f"forced model.enable_cam {was_cam} -> True so the model publishes camera_poses; "
              f"without it the world lift silently uses identity camera poses")

yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print("eval config -> {}  (ckpt={}, num_frames={}, enable_gs={}, enable_cam={})".format(
    out, ckpt, cfg["data"]["num_frames"],
    cfg["model"].get("enable_gs"), cfg["model"].get("enable_cam")))
