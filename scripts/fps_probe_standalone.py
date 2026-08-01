#!/usr/bin/env python3
"""Standalone neural network inference throughput (FPS) probe for baseline hand models (WiLoR, HaMeR).
Times the pure neural network forward pass + MANO joint prediction latency on GPU (excluding DROID-SLAM).
"""
import time
import torch
import numpy as np


def benchmark_wilor():
    print("=== Benchmarking Standalone WiLoR Forward Pass ===", flush=True)
    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16)

        # Create a dummy image (224x224x3 RGB)
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            _ = pipe.predict(dummy_frame)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed loop over 100 frames
        latencies = []
        n_iters = 100
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = pipe.predict(dummy_frame)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        mean_lat = float(np.mean(latencies))
        fps = 1.0 / mean_lat
        print(f"[WiLoR Standalone] Mean Latency: {mean_lat * 1000:.2f} ms | Throughput: {fps:.2f} FPS", flush=True)
        return fps, mean_lat
    except Exception as e:
        print(f"[WiLoR Standalone] Benchmark Failed: {e}", flush=True)
        return None, None


def benchmark_hamer():
    print("=== Benchmarking Standalone HaMeR Forward Pass ===", flush=True)
    try:
        from hamer.models import load_hamer
        from hamer.utils import recursive_to
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_hamer()
        model = model.to(device).eval()

        dummy_img = torch.randn(1, 3, 256, 256, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_img)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed loop over 100 frames
        latencies = []
        n_iters = 100
        with torch.no_grad():
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _ = model(dummy_img)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append(t1 - t0)

        mean_lat = float(np.mean(latencies))
        fps = 1.0 / mean_lat
        print(f"[HaMeR Standalone] Mean Latency: {mean_lat * 1000:.2f} ms | Throughput: {fps:.2f} FPS", flush=True)
        return fps, mean_lat
    except Exception as e:
        print(f"[HaMeR Standalone] Benchmark Failed: {e}", flush=True)
        return None, None


if __name__ == "__main__":
    w_fps, w_lat = benchmark_wilor()
    h_fps, h_lat = benchmark_hamer()
    print(f"DONE_STANDALONE_FPS wilor={w_fps} hamer={h_fps}", flush=True)
