"""Target-blind frame and render-camera plumbing for hand insertion experiments.

The model is allowed to see context frames only. Target cameras are attached to an already-built
prediction dictionary after the forward pass, immediately before rasterization. Keeping those two
operations separate makes target blindness a data-flow invariant rather than an attention mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from scripts.metric_views import build_views_metric


_FRAME_KEYS = frozenset({
    "img",
    "gt",
    "gt_joints",
    "gt_joints_cam",
    "gt_joints_2d",
    "hand_bboxes",
    "hand_valid",
    "hand_crops",
    "cam_extrinsics",
    "camera",
    "camera_poses",
    "camera_intrs",
    "depthmap",
    "depth",
    "sensor_depth",
    "valid_mask",
    "is_target",
    "is_static",
    "timestamp",
    "frame_index",
    "has_mano",
    "contact",
})


@dataclass(frozen=True)
class FramePartition:
    """Immutable original-index partition for an interior held-out protocol."""

    num_frames: int
    context: tuple[int, ...]
    targets: tuple[int, ...]
    _neighbors: tuple[tuple[int, int, int], ...]
    _context_positions: Mapping[int, int]

    @classmethod
    def build(cls, num_frames: int, target_indices: Sequence[int]) -> "FramePartition":
        num_frames = int(num_frames)
        raw_targets = tuple(int(i) for i in target_indices)
        if num_frames < 3:
            raise ValueError(f"num_frames must be at least 3, got {num_frames}")
        if not raw_targets:
            raise ValueError("at least one held-out target is required")
        if len(set(raw_targets)) != len(raw_targets):
            raise ValueError(f"target indices must be unique, got {raw_targets}")

        targets = tuple(sorted(raw_targets))
        for target in targets:
            if target <= 0 or target >= num_frames - 1:
                raise ValueError(
                    f"target {target} must be interior to [0, {num_frames - 1}]"
                )
            if target - 1 in targets or target + 1 in targets:
                raise ValueError(
                    f"target {target} has a held-out temporal neighbor; interpolation would leak"
                )

        target_set = set(targets)
        context = tuple(i for i in range(num_frames) if i not in target_set)
        positions = MappingProxyType({frame: pos for pos, frame in enumerate(context)})
        neighbors = tuple((target, target - 1, target + 1) for target in targets)
        return cls(
            num_frames=num_frames,
            context=context,
            targets=targets,
            _neighbors=neighbors,
            _context_positions=positions,
        )

    @property
    def neighbors(self) -> dict[int, tuple[int, int]]:
        return {target: (lower, upper) for target, lower, upper in self._neighbors}

    def context_position(self, original_frame: int) -> int:
        return self._context_positions[int(original_frame)]


@dataclass(frozen=True)
class TargetRenderCameras:
    """Target-only rasterizer inputs. Deliberately has no appearance field."""

    c2w: torch.Tensor
    intrinsics: torch.Tensor
    timestamps: torch.Tensor


def slice_frame_mapping(
    mapping: Mapping[str, Any],
    indices: Sequence[int],
    frame_count: int,
    *,
    frame_keys: frozenset[str] = _FRAME_KEYS,
) -> dict[str, Any]:
    """Return a shallow copy with registered `[B,S,...]` tensors sliced on S.

    Unknown tensors whose second dimension happens to equal ``frame_count`` are rejected. That
    guard forces new per-frame batch fields to be registered instead of silently carrying targets
    into a context-only forward.
    """

    frame_count = int(frame_count)
    idx = tuple(int(i) for i in indices)
    if not idx:
        raise ValueError("indices must not be empty")
    if min(idx) < 0 or max(idx) >= frame_count:
        raise IndexError(f"indices {idx} are outside [0, {frame_count - 1}]")

    out: dict[str, Any] = {}
    for key, value in mapping.items():
        is_frame_tensor = (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and int(value.shape[1]) == frame_count
        )
        if key in frame_keys:
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                if int(value.shape[1]) != frame_count:
                    raise ValueError(
                        f"registered frame tensor {key!r} has S={value.shape[1]}, "
                        f"expected {frame_count}"
                    )
                out[key] = value[:, idx]
            else:
                out[key] = value
        elif is_frame_tensor:
            raise KeyError(
                f"tensor {key!r} looks per-frame with shape {tuple(value.shape)} but is not "
                "registered in insertion_protocol._FRAME_KEYS"
            )
        else:
            out[key] = value
    return out


def target_render_predictions(
    preds: Mapping[str, Any],
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    timestamps: torch.Tensor,
) -> dict[str, Any]:
    """Attach target cameras to context-only splats without accepting target appearance."""

    if "splats" not in preds:
        raise KeyError("preds must contain context-only 'splats'")
    if c2w.ndim != 4 or tuple(c2w.shape[-2:]) != (4, 4):
        raise ValueError(f"c2w must be [B,T,4,4], got {tuple(c2w.shape)}")
    if intrinsics.ndim != 4 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(
            f"intrinsics must be [B,T,3,3], got {tuple(intrinsics.shape)}"
        )
    if timestamps.ndim != 2:
        raise ValueError(f"timestamps must be [B,T], got {tuple(timestamps.shape)}")

    batch_targets = tuple(c2w.shape[:2])
    if tuple(intrinsics.shape[:2]) != batch_targets or tuple(timestamps.shape) != batch_targets:
        raise ValueError(
            "target camera fields disagree: "
            f"c2w={tuple(c2w.shape)}, intrinsics={tuple(intrinsics.shape)}, "
            f"timestamps={tuple(timestamps.shape)}"
        )
    if len(preds["splats"]) != batch_targets[0]:
        raise ValueError(
            f"splats batch {len(preds['splats'])} != target camera batch {batch_targets[0]}"
        )

    out = dict(preds)
    out["rendered_extrinsics"] = c2w
    out["rendered_intrinsics"] = intrinsics
    out["rendered_timestamps"] = timestamps
    return out


def build_context_views_metric(
    imgs: torch.Tensor,
    *,
    device: torch.device | str,
    cam_extrinsics: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    res: int,
    target_indices: Sequence[int],
    hand_bboxes: torch.Tensor | None = None,
    hand_valid: torch.Tensor | None = None,
    crop_local_output: bool = False,
    hand_crops: torch.Tensor | None = None,
    frame_index: torch.Tensor | None = None,
) -> tuple[dict[str, Any], TargetRenderCameras, FramePartition]:
    """Build a model view containing only context RGB and separate target cameras.

    The full mapping exists only long enough to apply one camera convention consistently. The
    returned model view has no target slot, while the target object has no RGB field by type.
    """

    if imgs.ndim != 5:
        raise ValueError(f"imgs must be [B,S,3,H,W], got {tuple(imgs.shape)}")
    num_frames = int(imgs.shape[1])
    partition = FramePartition.build(num_frames, target_indices)

    full_views = build_views_metric(
        imgs,
        num_frames,
        device,
        cam_extrinsics,
        cam_intrinsics,
        res,
        hand_bboxes=hand_bboxes,
        hand_valid=hand_valid,
        n_targets=0,
        crop_local_output=crop_local_output,
        hand_crops=hand_crops,
        frame_index=frame_index,
        static=True,
    )
    context_views = slice_frame_mapping(
        full_views,
        partition.context,
        frame_count=num_frames,
    )
    context_views["is_target"] = torch.zeros_like(context_views["is_target"])
    context_views["is_static"] = torch.ones_like(context_views["is_static"])

    target_index = list(partition.targets)
    target_cameras = TargetRenderCameras(
        c2w=full_views["camera_poses"][:, target_index],
        intrinsics=full_views["camera_intrs"][:, target_index],
        timestamps=full_views["timestamp"][:, target_index],
    )
    return context_views, target_cameras, partition
