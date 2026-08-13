#!/usr/bin/env python3
"""Standalone NumPy reference scorer for the Hand3R HOI4D protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


METRIC_NAMES = (
    "C-MPJPE",
    "PA-MPJPE",
    "MPJPE",
    "WA-MPJPE",
    "W-MPJPE",
    "MRE",
    "RTE",
)


def _as_joints(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (21, 3):
        raise ValueError(f"{name} must have shape [T, 21, 3], got {value.shape}")
    return value


def _align(
    target: np.ndarray, source: np.ndarray, fixed_scale: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment matching the Table-II PyTorch evaluator."""
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    if target.shape != source.shape or target.shape[0] == 0:
        raise ValueError("Alignment inputs must be non-empty and shape-matched")

    mu_target = target.mean(axis=0)
    mu_source = source.mean(axis=0)
    target_centered = target - mu_target
    source_centered = source - mu_source
    covariance = target_centered.T @ source_centered / target.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ correction @ vt

    if fixed_scale:
        scale = 1.0
    else:
        source_variance = np.square(source_centered).sum() / target.shape[0]
        if source_variance <= np.finfo(np.float64).eps:
            raise ValueError("Cannot estimate scale from a degenerate source")
        scale = float((singular_values * np.diag(correction)).sum() / source_variance)

    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation, translation


def _apply(points: np.ndarray, transform: Tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    return scale * (np.asarray(points) @ rotation.T) + translation


def _mpjpe(gt: np.ndarray, pred: np.ndarray, to_mm: float) -> float:
    return float(np.linalg.norm(gt - pred, axis=-1).mean() * to_mm)


def _per_frame_aligned(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    aligned = []
    for gt_frame, pred_frame in zip(gt, pred):
        aligned.append(_apply(pred_frame, _align(gt_frame, pred_frame)))
    return np.stack(aligned)


def _rte(gt_root: np.ndarray, pred_root: np.ndarray) -> float:
    pred_aligned = _apply(pred_root, _align(gt_root, pred_root, fixed_scale=True))
    total_displacement = np.linalg.norm(np.diff(gt_root, axis=0), axis=-1).sum()
    total_error = np.linalg.norm(gt_root - pred_aligned, axis=-1).sum()
    return float(total_error / (total_displacement + 1e-8) * 100.0)


def camera_metrics(gt_cam: np.ndarray, pred_cam: np.ndarray, to_mm: float) -> Dict[str, float]:
    return {
        "C-MPJPE": _mpjpe(gt_cam, pred_cam, to_mm),
        "PA-MPJPE": _mpjpe(gt_cam, _per_frame_aligned(gt_cam, pred_cam), to_mm),
    }


def world_metrics(
    gt_world: np.ndarray,
    pred_world: np.ndarray,
    chunk_length: int,
    to_mm: float,
) -> Dict[str, float]:
    values = {name: [] for name in ("MPJPE", "WA-MPJPE", "W-MPJPE", "MRE", "RTE")}
    for start in range(0, len(gt_world), chunk_length):
        end = min(len(gt_world), start + chunk_length)
        if end - start < 10:
            continue
        gt = gt_world[start:end]
        pred = pred_world[start:end]
        values["MPJPE"].append(_mpjpe(gt, pred, to_mm))
        pred_wa = _apply(pred, _align(gt, pred))
        values["WA-MPJPE"].append(_mpjpe(gt, pred_wa, to_mm))
        pred_w = _apply(pred, _align(gt[:2], pred[:2], fixed_scale=True))
        values["W-MPJPE"].append(_mpjpe(gt, pred_w, to_mm))
        gt_root, pred_root = gt[:, 0], pred[:, 0]
        values["MRE"].append(float(np.linalg.norm(gt_root - pred_root, axis=-1).mean() * to_mm))
        values["RTE"].append(_rte(gt_root, pred_root))
    return {
        name: float(np.mean(items)) if items else float("nan")
        for name, items in values.items()
    }


def score_clip(
    gt_cam: np.ndarray,
    pred_cam: np.ndarray,
    gt_world: Optional[np.ndarray] = None,
    pred_world: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    pred_valid_mask: Optional[np.ndarray] = None,
    chunk_length: int = 100,
    unit: str = "m",
) -> Optional[Dict[str, float]]:
    """Score one clip; return None when paper-protocol validity checks reject it."""
    gt_cam = _as_joints(gt_cam, "gt_cam")
    pred_cam = _as_joints(pred_cam, "pred_cam")
    if gt_cam.shape != pred_cam.shape:
        raise ValueError("Camera-frame GT and prediction shapes differ")
    frame_count = len(gt_cam)
    mask = np.ones(frame_count, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != (frame_count,):
        raise ValueError(f"valid_mask must have shape [{frame_count}]")
    if pred_valid_mask is not None:
        pred_mask = np.asarray(pred_valid_mask, dtype=bool)
        if pred_mask.shape != (frame_count,):
            raise ValueError(f"prediction valid_mask must have shape [{frame_count}]")
        mask &= pred_mask

    mask &= np.isfinite(gt_cam).all(axis=(1, 2))
    if mask.sum() < 10:
        return None
    finite_prediction = np.isfinite(pred_cam).all(axis=(1, 2))
    nonfinite_count = int((mask & ~finite_prediction).sum())
    if nonfinite_count / int(mask.sum()) > 0.5:
        return None
    mask &= finite_prediction
    if mask.sum() < 10:
        return None

    # Normalize internally to meters so the RTE epsilon exactly matches the
    # original evaluator even when the submitted arrays use millimeters.
    spatial_scale = 1.0 if unit == "m" else 0.001
    gt_cam = gt_cam * spatial_scale
    pred_cam = pred_cam * spatial_scale
    to_mm = 1000.0
    result = camera_metrics(gt_cam[mask], pred_cam[mask], to_mm)

    if (gt_world is None) != (pred_world is None):
        raise ValueError("gt_world and pred_world must be provided together")
    if gt_world is not None:
        gt_world = _as_joints(gt_world, "gt_world") * spatial_scale
        pred_world = _as_joints(pred_world, "pred_world") * spatial_scale
        if gt_world.shape != gt_cam.shape or pred_world.shape != gt_cam.shape:
            raise ValueError("World and camera arrays must have identical shapes")
        world_mask = mask & np.isfinite(gt_world).all(axis=(1, 2)) & np.isfinite(pred_world).all(axis=(1, 2))
        if world_mask.sum() >= 10:
            result.update(world_metrics(gt_world[world_mask], pred_world[world_mask], chunk_length, to_mm))
    return result


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _nan_to_none(value):
    if isinstance(value, dict):
        return {key: _nan_to_none(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate_manifest(manifest: Path, chunk_length: int, unit: str) -> Dict[str, object]:
    per_clip: Dict[str, Dict[str, float]] = {}
    skipped = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip_id = row["clip_id"]
            gt_path = Path(row["gt_path"])
            pred_path = Path(row["pred_path"])
            if not gt_path.is_absolute():
                gt_path = manifest.parent / gt_path
            if not pred_path.is_absolute():
                pred_path = manifest.parent / pred_path
            gt = _load_npz(gt_path)
            pred = _load_npz(pred_path)
            metrics = score_clip(
                gt_cam=gt["joints_cam"],
                pred_cam=pred["joints_cam"],
                gt_world=gt.get("joints_world"),
                pred_world=pred.get("joints_world"),
                valid_mask=gt.get("valid_mask"),
                pred_valid_mask=pred.get("valid_mask"),
                chunk_length=chunk_length,
                unit=unit,
            )
            if metrics is None:
                skipped.append(clip_id)
            else:
                per_clip[clip_id] = metrics

    summary = {}
    for metric in METRIC_NAMES:
        values = [item[metric] for item in per_clip.values() if metric in item and np.isfinite(item[metric])]
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "count": len(values),
            }
        else:
            summary[metric] = {"mean": None, "std": None, "median": None, "count": 0}
    return _nan_to_none({"summary": summary, "per_clip": per_clip, "skipped_clips": skipped})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--chunk-length", type=int, choices=(30, 100), default=100)
    parser.add_argument("--unit", choices=("m", "mm"), default="m")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    results = evaluate_manifest(args.manifest, args.chunk_length, args.unit)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
