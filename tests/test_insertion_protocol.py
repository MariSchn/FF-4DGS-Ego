from __future__ import annotations

import inspect

import pytest
import torch

from scripts.insertion_protocol import (
    FramePartition,
    build_context_views_metric,
    slice_frame_mapping,
    target_render_predictions,
)


def test_partition_preserves_temporal_neighbors_without_reordering_targets():
    partition = FramePartition.build(16, [6, 10])

    assert partition.context == (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15)
    assert partition.targets == (6, 10)
    assert partition.neighbors == {6: (5, 7), 10: (9, 11)}
    assert partition.context_position(5) == 5
    assert partition.context_position(11) == 9


@pytest.mark.parametrize("targets", ([0], [15], [6, 7], [6, 6], []))
def test_partition_rejects_invalid_targets(targets):
    with pytest.raises(ValueError):
        FramePartition.build(16, targets)


def test_partition_rejects_out_of_range_context_lookup():
    partition = FramePartition.build(16, [6, 10])

    with pytest.raises(KeyError):
        partition.context_position(6)


def test_slice_frame_mapping_slices_only_registered_frame_tensors():
    mapping = {
        "img": torch.arange(2 * 5 * 3).reshape(2, 5, 3),
        "camera": torch.arange(2 * 5 * 4 * 4).reshape(2, 5, 4, 4),
        "intrinsics": torch.ones(2, 3),
        "label": "clip",
    }

    out = slice_frame_mapping(mapping, [0, 2, 4], frame_count=5)

    assert torch.equal(out["img"], mapping["img"][:, [0, 2, 4]])
    assert torch.equal(out["camera"], mapping["camera"][:, [0, 2, 4]])
    assert out["intrinsics"] is mapping["intrinsics"]
    assert out["label"] == "clip"
    assert torch.equal(mapping["img"], torch.arange(2 * 5 * 3).reshape(2, 5, 3))


def test_slice_frame_mapping_rejects_bad_indices():
    mapping = {"img": torch.zeros(1, 5, 3)}

    with pytest.raises(IndexError):
        slice_frame_mapping(mapping, [0, 5], frame_count=5)


def test_target_render_predictions_replaces_only_render_camera_fields():
    splats = [object()]
    preds = {
        "splats": splats,
        "hand_joints": torch.ones(1, 3, 64),
        "rendered_extrinsics": torch.eye(4).view(1, 1, 4, 4),
        "rendered_intrinsics": torch.eye(3).view(1, 1, 3, 3),
        "rendered_timestamps": torch.zeros(1, 1, dtype=torch.long),
    }
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    timestamps = torch.tensor([[6, 10]])

    out = target_render_predictions(preds, c2w, intrinsics, timestamps)

    assert out is not preds
    assert out["splats"] is splats
    assert out["hand_joints"] is preds["hand_joints"]
    assert torch.equal(out["rendered_extrinsics"], c2w)
    assert torch.equal(out["rendered_intrinsics"], intrinsics)
    assert torch.equal(out["rendered_timestamps"], timestamps)
    assert preds["rendered_extrinsics"].shape[1] == 1


def test_target_render_predictions_does_not_accept_target_rgb():
    assert "img" not in inspect.signature(target_render_predictions).parameters


def test_build_context_views_metric_never_puts_target_rgb_in_model_view():
    images = torch.arange(1 * 6 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 6, 3, 4, 4)
    w2c = torch.eye(4).view(1, 1, 4, 4).repeat(1, 6, 1, 1)
    w2c[:, :, 0, 3] = -torch.arange(6, dtype=torch.float32) * 0.01
    intr = torch.tensor([[4.0, 2.0, 2.0]])
    boxes = torch.zeros(1, 6, 2, 4)
    valid = torch.ones(1, 6, 2, dtype=torch.bool)
    frame_index = torch.arange(6).view(1, 6)

    context_views, target_cameras, partition = build_context_views_metric(
        images,
        device="cpu",
        cam_extrinsics=w2c,
        cam_intrinsics=intr,
        res=4,
        target_indices=[2, 4],
        hand_bboxes=boxes,
        hand_valid=valid,
        frame_index=frame_index,
    )

    assert partition.context == (0, 1, 3, 5)
    assert torch.equal(context_views["img"], images[:, [0, 1, 3, 5]])
    assert context_views["img"].shape[1] == 4
    assert not context_views["is_target"].any()
    assert context_views["is_static"].all()
    assert torch.equal(context_views["timestamp"], torch.tensor([[0, 1, 3, 5]]))
    assert torch.equal(target_cameras.timestamps, torch.tensor([[2, 4]]))
    assert target_cameras.c2w.shape == (1, 2, 4, 4)
    assert target_cameras.intrinsics.shape == (1, 2, 3, 3)
    assert not hasattr(target_cameras, "images")


@pytest.mark.parametrize(
    ("c2w_shape", "intr_shape", "ts_shape"),
    [
        ((1, 2, 3, 4), (1, 2, 3, 3), (1, 2)),
        ((1, 2, 4, 4), (1, 1, 3, 3), (1, 2)),
        ((1, 2, 4, 4), (1, 2, 3, 3), (1, 1)),
    ],
)
def test_target_render_predictions_rejects_mismatched_camera_shapes(
    c2w_shape, intr_shape, ts_shape
):
    preds = {"splats": [object()]}

    with pytest.raises(ValueError):
        target_render_predictions(
            preds,
            torch.zeros(c2w_shape),
            torch.zeros(intr_shape),
            torch.zeros(ts_shape, dtype=torch.long),
        )
