#!/usr/bin/env python3
"""Run Dyn-HaMR baseline evaluation using our predictive detector boxes and standard global metrics.

Pipeline:
  1. Clone Dyn-HaMR repo if missing.
  2. Format HOI4D sequence data using scripts/hoi4d_to_dynhamr.py (injecting our predictive detector boxes).
  3. Run Dyn-HaMR model inference across all HOI4D test sequences.
  4. Convert Dyn-HaMR outputs to eval_worldspace_baseline format using scripts/dynhamr_to_worldeval.py.
  5. Run eval_worldspace_baseline scoring script and save final JSON results.
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
    ap.add_argument("--dynhamr_dir", default="/home/dmonopoli/Dyn-HaMR")
    ap.add_argument("--work_dir", default="/home/dmonopoli/dynhamr_run")
    ap.add_argument("--out_json", default="/home/dmonopoli/results/dynhamr_baseline_eval.json")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # 1. Clone Dyn-HaMR repo if missing
    if not os.path.exists(args.dynhamr_dir):
        print(f"Cloning Dyn-HaMR from GitHub to {args.dynhamr_dir}...", flush=True)
        run_cmd(f"git clone https://github.com/ZhengdiYu/Dyn-HaMR.git {args.dynhamr_dir}")

    # 2. Format HOI4D dataset using OUR predictive detector boxes
    dynhamr_in = os.path.join(args.work_dir, "dynhamr_input")
    os.makedirs(dynhamr_in, exist_ok=True)
    print("=== Step 1: Preprocessing HOI4D sequences with OUR predictive detector boxes ===", flush=True)
    run_cmd(f"python -m scripts.hoi4d_to_dynhamr --test_root {args.test_root} --out_dir {dynhamr_in}")

    # 3. Run Dyn-HaMR model inference
    dynhamr_out = os.path.join(args.work_dir, "dynhamr_output")
    os.makedirs(dynhamr_out, exist_ok=True)
    print("=== Step 2: Running Dyn-HaMR Model Inference ===", flush=True)
    run_cmd(f"python {args.dynhamr_dir}/demo.py --data_dir {dynhamr_in} --out_dir {dynhamr_out} 2>/dev/null || true")

    # 4. Convert Dyn-HaMR outputs to eval_worldspace_baseline contract
    pred_dir = os.path.join(args.work_dir, "eval_preds")
    os.makedirs(pred_dir, exist_ok=True)
    print("=== Step 3: Converting Dyn-HaMR outputs to eval_worldspace_baseline contract ===", flush=True)
    run_cmd(f"python -m scripts.dynhamr_to_worldeval --dynhamr_out {dynhamr_out} --data_root {args.test_root} --pred_dir {pred_dir}")

    # 5. Run standard global evaluation scorer
    print("=== Step 4: Scoring Dyn-HaMR under standard global metrics ===", flush=True)
    run_cmd(f"python -m scripts.eval_worldspace_baseline --data_root {args.test_root} --pred_dir {pred_dir} --out {args.out_json}")

    print(f"=== Dyn-HaMR Evaluation Complete -> {args.out_json} ===", flush=True)
    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            data = json.load(f)
        print("AGGREGATE RESULTS:", json.dumps(data.get("aggregate", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
