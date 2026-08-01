#!/usr/bin/env python3
"""Standalone neural network inference throughput (FPS) probe for baseline hand models (WiLoR, HaMeR).
Excludes offline DROID-SLAM to measure pure neural network forward pass throughput on GPU.
"""
import time
import torch
import numpy as np


def benchmark_wilor():
    print("=== Benchmarking WiLoR Neural Network Forward Pass ===", flush=True)
    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16)

        # Create a dummy image matching 224x224 input
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            _ = pipe.predict(dummy_frame)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed loop
        latencies = []
        n_iters = 100
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = pipe.predict(dummy_frame)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        mean_lat = np.mean(latencies)
        fps = 1.0 / mean_lat
        print(f"WiLoR Standalone NN Latency: {mean_lat * 1000:.2f} ms | FPS: {fps:.2f}", flush=True)
        return fps
    except Exception as e:
        print(f"WiLoR benchmark failed: {e}", flush=True)
        return None


def benchmark_hamer():
    print("=== Benchmarking HaMeR Neural Network Forward Pass ===", flush=True)
    try:
        from hamer.models import load_hamer
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

        # Timed loop
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

        mean_lat = np.mean(latencies)
        fps = 1.0 / mean_lat
        print(f"HaMeR Standalone NN Latency: {mean_lat * 1000:.2f} ms | FPS: {fps:.2f}", flush=True)
        return fps
    except Exception as e:
        print(f"HaMeR benchmark failed: {e}", flush=True)
        return None


if __name__ == "__main__":
    w_fps = benchmark_wilor()
    h_fps = benchmark_hamer()
    print("DONE_FPS_BASELINES", flush=True)
