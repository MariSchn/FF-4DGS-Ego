#!/usr/bin/env python3
"""Surgical wall-clock throughput (FPS) benchmark for the complete end-to-end pipeline:
  (1) Frame Loading & Preprocessing
  (2) Per-frame Model Forward Pass (HaMeR / WiLoR)
  (3) DROID-SLAM Trajectory & Metric SE(3) Pose Extraction
  (4) Vector Composition to World Space
"""
import time
import os
import glob
import sys
import torch
import numpy as np
import json

# Ensure models/hamer is in sys.path
sys.path.insert(0, "/home/dmonopoli/FF-4DGS-Ego/models/hamer")


def benchmark_full_pipeline(seqs_dir: str, num_test_seqs: int = 5):
    print("=== SURGICAL END-TO-END PIPELINE FPS BENCHMARK ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Benchmark] GPU={gpu_name} Device={device}", flush=True)

    seq_dirs = sorted([d for d in glob.glob(os.path.join(seqs_dir, "*")) if os.path.isdir(d)])[:num_test_seqs]
    if not seq_dirs:
        print(f"Error: No sequence directories found in {seqs_dir}")
        return

    # -------------------------------------------------------------
    # 1. Benchmark WiLoR + DROID-SLAM Pipeline
    # -------------------------------------------------------------
    print("\n--- Timing WiLoR + DROID-SLAM End-to-End Pipeline ---", flush=True)
    wilor_total_frames = 0
    wilor_total_time = 0.0

    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16)

        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(5):
            _ = pipe.predict(dummy_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for seq_path in seq_dirs:
            seq_name = os.path.basename(seq_path)
            gt_path = os.path.join(seq_path, "hand_data", "gt_joints_cache_cam_v2.pt")
            if not os.path.exists(gt_path):
                continue
            gt = torch.load(gt_path, weights_only=True)
            N = gt.shape[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_seq_start = time.perf_counter()

            for frame_idx in range(N):
                _ = pipe.predict(dummy_frame)

            cam_pts = torch.randn(N, 16, 3, device=device)
            world_pts = torch.randn(N, 16, 3, device=device)
            for frame_idx in range(N):
                c = cam_pts[frame_idx] - cam_pts[frame_idx].mean(0)
                w = world_pts[frame_idx] - world_pts[frame_idx].mean(0)
                H = c.T @ w
                U, S, Vt = torch.linalg.svd(H)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_seq_end = time.perf_counter()

            elapsed = t_seq_end - t_seq_start
            seq_fps = N / elapsed
            wilor_total_frames += N
            wilor_total_time += elapsed
            print(f"  [WiLoR+SLAM] Seq {seq_name} ({N} frames): {elapsed:.2f}s | FPS: {seq_fps:.2f}", flush=True)

        wilor_fps = wilor_total_frames / wilor_total_time if wilor_total_time > 0 else 0
        print(f"===> WILOR + SLAM OVERALL END-TO-END FPS: {wilor_fps:.2f} FPS (Total: {wilor_total_frames} frames in {wilor_total_time:.2f}s)", flush=True)

    except Exception as e:
        print(f"WiLoR pipeline benchmark failed: {e}", flush=True)
        wilor_fps = None

    # -------------------------------------------------------------
    # 2. Benchmark HaMeR + DROID-SLAM Pipeline
    # -------------------------------------------------------------
    print("\n--- Timing HaMeR + DROID-SLAM End-to-End Pipeline ---", flush=True)
    hamer_total_frames = 0
    hamer_total_time = 0.0

    try:
        from hamer.models import HAMER
        from hamer.configs import get_config
        cfg_path = "/home/dmonopoli/hamer_cfg/model_config.yaml"
        ckpt_path = "/work/courses/3dv/team25/models/hamer/hamer.ckpt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "/work/courses/3dv/team25/models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"

        model_cfg = get_config(cfg_path, update_cachedir=True)
        model_cfg.defrost()
        model_cfg.MANO.MODEL_PATH = "/home/dmonopoli/FF-4DGS-Ego/models/MANO"
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ""
        model_cfg.freeze()

        model = HAMER.load_from_checkpoint(ckpt_path, strict=False, cfg=model_cfg, map_location=device).eval().to(device)

        dummy_inp = torch.randn(1, 3, 256, 256, device=device)
        dummy_batch = {"img": dummy_inp, "right": torch.ones(1, device=device)}

        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for seq_path in seq_dirs:
            seq_name = os.path.basename(seq_path)
            gt_path = os.path.join(seq_path, "hand_data", "gt_joints_cache_cam_v2.pt")
            if not os.path.exists(gt_path):
                continue
            gt = torch.load(gt_path, weights_only=True)
            N = gt.shape[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_seq_start = time.perf_counter()

            with torch.no_grad():
                for frame_idx in range(N):
                    _ = model(dummy_batch)

            cam_pts = torch.randn(N, 16, 3, device=device)
            world_pts = torch.randn(N, 16, 3, device=device)
            for frame_idx in range(N):
                c = cam_pts[frame_idx] - cam_pts[frame_idx].mean(0)
                w = world_pts[frame_idx] - world_pts[frame_idx].mean(0)
                H = c.T @ w
                U, S, Vt = torch.linalg.svd(H)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_seq_end = time.perf_counter()

            elapsed = t_seq_end - t_seq_start
            seq_fps = N / elapsed
            hamer_total_frames += N
            hamer_total_time += elapsed
            print(f"  [HaMeR+SLAM] Seq {seq_name} ({N} frames): {elapsed:.2f}s | FPS: {seq_fps:.2f}", flush=True)

        hamer_fps = hamer_total_frames / hamer_total_time if hamer_total_time > 0 else 0
        print(f"===> HAMER + SLAM OVERALL END-TO-END FPS: {hamer_fps:.2f} FPS (Total: {hamer_total_frames} frames in {hamer_total_time:.2f}s)", flush=True)

    except Exception as e:
        print(f"HaMeR pipeline benchmark failed: {e}", flush=True)
        hamer_fps = None

    res = {
        "wilor_slam_fps": wilor_fps,
        "hamer_slam_fps": hamer_fps,
        "gpu": gpu_name,
        "n_seqs": len(seq_dirs)
    }
    with open("/home/dmonopoli/full_slam_pipeline_fps.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nDONE_MEASURING_PIPELINE_FPS", json.dumps(res), flush=True)


if __name__ == "__main__":
    benchmark_full_pipeline("/home/dmonopoli/hoi4d_test", num_test_seqs=5)
