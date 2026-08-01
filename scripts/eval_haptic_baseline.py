#!/usr/bin/env python3
"""Run HaPTIC baseline evaluation using our predictive detector boxes and standard global metrics.

Pipeline:
  1. Prepare HOI4D sequence data using scripts/hoi4d_to_haptic.py (injecting our predictive detector boxes).
  2. Run HaPTIC model inference across all HOI4D test sequences.
  3. Convert HaPTIC pickle outputs to eval_worldspace_baseline format using scripts/haptic_to_worldeval.py.
  4. Run eval_worldspace_baseline scoring script and save final JSON results.
"""
import os
import sys
import argparse
import subprocess
import json


def run_cmd(cmd: str):
    print(f"[Run] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[Error] Command failed with exit code {res.returncode}: {cmd}")
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", default="/home/dmonopoli/hoi4d_test")
    ap.add_argument("--haptic_dir", default="/home/dmonopoli/haptic")
    ap.add_argument("--work_dir", default="/work/scratch/dmonopoli/haptic_run")
    ap.add_argument("--out_json", default="/home/dmonopoli/results/haptic_baseline_eval.json")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # 1. Clone HaPTIC repo if missing
    if not os.path.exists(args.haptic_dir):
        print(f"Cloning HaPTIC from GitHub to {args.haptic_dir}...", flush=True)
        run_cmd(f"git clone https://github.com/JudyYe/haptic.git {args.haptic_dir}")

    # 2. Format HOI4D dataset into HaPTIC input format using OUR predictive detector boxes
    haptic_in = os.path.join(args.work_dir, "haptic_input")
    os.makedirs(haptic_in, exist_ok=True)
    print("=== Step 1: Preprocessing HOI4D sequences with OUR predictive detector boxes ===", flush=True)
    run_cmd(f"python -m scripts.hoi4d_to_haptic --data_root {args.test_root} --out_root {haptic_in}")

    # 3. Run HaPTIC model inference
    haptic_out = os.path.join(args.work_dir, "haptic_output")
    os.makedirs(haptic_out, exist_ok=True)
    print("=== Step 2: Running HaPTIC Model Inference ===", flush=True)
    run_cmd(f"python {args.haptic_dir}/demo.py --data_dir {haptic_in} --out_dir {haptic_out}")

    # 4. Convert HaPTIC outputs to eval_worldspace_baseline contract
    pred_dir = os.path.join(args.work_dir, "eval_preds")
    os.makedirs(pred_dir, exist_ok=True)
    print("=== Step 3: Converting HaPTIC outputs to eval_worldspace_baseline contract ===", flush=True)
    run_cmd(f"python -m scripts.haptic_to_worldeval --haptic_out {haptic_out} --data_root {args.test_root} --pred_dir {pred_dir}")

    # 5. Run standard global evaluation scorer
    print("=== Step 4: Scoring HaPTIC under standard global metrics ===", flush=True)
    run_cmd(f"python -m scripts.eval_worldspace_baseline --data_root {args.test_root} --pred_dir {pred_dir} --out {args.out_json}")

    print(f"=== HaPTIC Evaluation Complete -> {args.out_json} ===", flush=True)
    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            data = json.load(f)
        print("AGGREGATE RESULTS:", json.dumps(data.get("aggregate", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
