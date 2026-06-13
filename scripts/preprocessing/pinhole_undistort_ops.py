"""Lens-undistort helpers (fisheye -> pinhole), torch-free.

Extracted verbatim from scripts/hand_alignment/align_hand_to_neoverse.py so the
preprocessing pipeline can import them WITHOUT pulling in that module's heavy
WorldMirror/torch-CUDA dependencies. This matters because preprocessing runs on
CPU-only x86 nodes (projectaria_tools has no aarch64 binding, so it cannot run
on the gb10/ARM training nodes, and the WorldMirror import drags in CUDA
extensions that fail to import on a CPU node).

Only projectaria_tools + numpy are required here.
"""
from __future__ import annotations

import numpy as np
from projectaria_tools.core import calibration as aria_calib

ARIA_FRAME_SIZE = 1408  # native Aria fisheye resolution


def build_pinhole_target(fisheye_calib, focal):
    """Linear (pinhole) camera matching the source image size and physical
    position (T_Device_Camera), in SENSOR orientation, at the given focal."""
    return aria_calib.get_linear_camera_calibration(
        ARIA_FRAME_SIZE, ARIA_FRAME_SIZE, float(focal), "pinhole_target",
        fisheye_calib.get_transform_device_camera(),
    )


def undistort_mp4_frame(frame_mp4_rgb, fisheye_calib, pinhole_calib):
    """Warp an MP4 (display-orientation) fisheye frame to MP4-orientation pinhole.

    MP4 is rotated 90 deg CW from the sensor orientation the calibration
    describes, so rotate CCW to align with the calibration, undistort, then
    rotate CW back to upright/MP4 orientation.
    """
    frame_sensor = np.rot90(frame_mp4_rgb, k=1).copy()
    und_sensor = aria_calib.distort_by_calibration(
        frame_sensor, pinhole_calib, fisheye_calib,
    )
    return np.rot90(und_sensor, k=-1).copy()
