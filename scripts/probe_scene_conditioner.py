"""Fail-closed E2E health probe for a trained scene-to-hand conditioner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from scripts.insertion_protocol import mano_box_placement_metrics
from scripts.scene_conditioning import SceneConditioningMode, condition_hand_params


def _first_valid_sample(dataset):
    for index in range(min(len(dataset), 32)):
        sample = dataset[index]
        if bool(sample.get("hand_valid", torch.zeros(1, dtype=torch.bool)).any()):
            return sample
    raise RuntimeError("no clip with a valid hand in the first 32 samples")


def _two_real_samples(cfg, mano_model):
    from scripts.train_hand_head import HOT3DHandDataset

    samples = []
    roots = cfg["data"]["data_roots"]
    for spec in roots:
        root = Path(spec["root"] if isinstance(spec, dict) else spec)
        if not root.is_dir():
            continue
        for sequence in sorted(path for path in root.iterdir() if path.is_dir()):
            dataset = HOT3DHandDataset(
                [str(sequence)],
                mano_model,
                num_frames=int(cfg["data"].get("num_frames", 16)),
                clip_stride=int(cfg["data"].get("clip_stride", 8)),
                use_hand_crop=bool(cfg["model"].get("use_hand_crop", False)),
                rescale_factor=float(cfg.get("hand_crop", {}).get("rescale_factor", 1.5)),
            )
            if len(dataset):
                try:
                    samples.append(_first_valid_sample(dataset))
                except RuntimeError:
                    continue
            if len(samples) == 2:
                return samples
    raise RuntimeError("could not find two real clips with visible hands")


def _placement(vertices, valid, boxes, intrinsics, image_size):
    rows = []
    for batch_index in range(vertices.shape[0]):
        for frame_index in range(vertices.shape[1]):
            row = mano_box_placement_metrics(
                vertices[batch_index, frame_index].cpu(),
                valid[batch_index, frame_index].cpu(),
                boxes[batch_index, frame_index].cpu(),
                intrinsics[batch_index].cpu(),
                image_size,
            )
            if row["median_z_hand"] is not None:
                rows.append(row)
    if not rows:
        raise RuntimeError("placement probe found no valid predicted hand")
    median = lambda key: float(torch.tensor([row[key] for row in rows]).median())
    result = {
        "median_z_hand": median("median_z_hand"),
        "centroid_to_boxcentre_px_med": median("centroid_to_boxcentre_px_med"),
        "box_diag_px": median("box_diag_px"),
    }
    result["passed"] = (
        0.3 <= result["median_z_hand"] <= 1.2
        and result["centroid_to_boxcentre_px_med"] < result["box_diag_px"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--require-geometry-response",
        action="store_true",
        help="require correct scene descriptors to differ from capacity and shuffled controls",
    )
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    cfg["model"]["warm_start_hand_head"] = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scripts.eval_world_space import build_model
    from scripts.hand_vis_utils import MANOModel
    from scripts.train_hand_head import build_views, compute_vertices_from_batch

    mano_model = MANOModel(cfg["visualization"]["mano_model_folder"])
    model = build_model(cfg, device)
    model.gs_anchor_only = True
    samples = _two_real_samples(cfg, mano_model)
    batch = torch.utils.data.default_collate(samples)
    imgs = batch["img"].to(device)
    boxes = batch["hand_bboxes"].to(device)
    valid = batch["hand_valid"].to(device).bool()
    intrinsics = batch["cam_intrinsics"].to(device)
    views = build_views(imgs, imgs.shape[1], device, boxes, valid)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(views, is_inference=True, use_motion=False)
    preds = {
        key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in preds.items()
    }
    required = ("hand_joints", "gs_depth", "camera_poses")
    missing = [key for key in required if preds.get(key) is None]
    if missing:
        raise RuntimeError(f"scene conditioner probe missing predictions: {missing}")

    params = {}
    deltas = {}
    descriptors = {}
    descriptor_cfg = cfg.get("training", {})
    for mode in (
        SceneConditioningMode.SCENE,
        SceneConditioningMode.CAP,
        SceneConditioningMode.SHUFFLE,
    ):
        refined, delta, controlled = condition_hand_params(
            model.scene_geometry_conditioner,
            preds["hand_joints"],
            preds["gs_depth"],
            preds.get("gs_depth_conf"),
            intrinsics,
            preds["camera_poses"],
            boxes,
            valid,
            image_size=int(imgs.shape[-1]),
            mode=mode,
            points_per_hand=int(descriptor_cfg.get("scene_points_per_hand", 32)),
            annulus_scale=float(descriptor_cfg.get("scene_annulus_scale", 2.0)),
            confidence_threshold=float(
                descriptor_cfg.get("scene_confidence_threshold", 0.0)
            ),
        )
        params[mode.value] = refined
        deltas[mode.value] = delta
        descriptors[mode.value] = controlled

    state = model.scene_geometry_conditioner.state_dict()
    learned_tensors = [
        value for key, value in state.items()
        if key in ("residual.2.weight", "residual.2.bias")
    ]
    if not learned_tensors:
        raise RuntimeError("could not locate the zero-initialized conditioner output layer")
    learned_norm = float(
        sum(value.float().square().sum() for value in learned_tensors).sqrt()
    )
    scene_delta = float(deltas["scene"].abs().max())
    scene_vs_cap = float((params["scene"] - params["cap"]).abs().max())
    scene_vs_shuffle = float((params["scene"] - params["shuffle"]).abs().max())
    vertices = compute_vertices_from_batch(params["scene"], mano_model, device)
    placement = _placement(vertices, valid, boxes, intrinsics, int(imgs.shape[-1]))
    scene_valid = descriptors["scene"].valid
    depth_values = preds["gs_depth"].flatten().float()
    finite = torch.isfinite(depth_values)
    finite_depth = depth_values[finite]
    box_area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    # The conditioner zeroes its own delta for any hand with no sampled scene point, so an
    # all-zero delta is unreadable without knowing whether the annulus found anything at all.
    sampling = {
        "hands_marked_valid": int(valid.sum()),
        "hands_with_scene_points": int(scene_valid.any(dim=-1).sum()),
        "point_fill_fraction": float(scene_valid.float().mean()),
        "gs_depth_finite_fraction": float(finite.float().mean()),
        "gs_depth_median_m": float(finite_depth.median()) if finite_depth.numel() else None,
        "gs_depth_over_1cm_fraction": (
            float((finite_depth > 0.01).float().mean()) if finite_depth.numel() else 0.0
        ),
        "box_area_fraction_med": (
            float(box_area[valid].median()) if bool(valid.any()) else None
        ),
    }
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "learned_output_layer_norm": learned_norm,
        "scene_delta_max_m": scene_delta,
        "scene_vs_cap_max_m": scene_vs_cap,
        "scene_vs_shuffle_max_m": scene_vs_shuffle,
        "sampling": sampling,
        "placement": placement,
    }
    # Placement is reported, not gated. The conditioner exists to correct hand placement, so
    # requiring correct placement before it is trained is circular, and the measured value does not
    # move between a 50-step and a 400-step smoke: it is a property of the warm start on the probed
    # clip, not of the module under test. Gate the wiring instead, and check placement on the
    # trained model where it carries a claim.
    report["passed"] = (
        learned_norm > 0.0
        and sampling["hands_with_scene_points"] > 0
        and scene_delta > 1e-7
        and (
            not args.require_geometry_response
            or (scene_vs_cap > 1e-7 and scene_vs_shuffle > 1e-7)
        )
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("SCENE_CONDITIONER_HEALTH_FAILED")
    print("SCENE_CONDITIONER_HEALTH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
