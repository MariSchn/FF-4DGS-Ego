#!/usr/bin/env python3
"""Download OakInk-v2 dataset directly from Hugging Face (kelvin34501/OakInk-v2) into /home/dmonopoli/oakink2.
"""
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download


def main():
    repo_id = "kelvin34501/OakInk-v2"
    out_dir = "/home/dmonopoli/oakink2"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Downloading OakInk-v2 from Hugging Face ({repo_id}) ===", flush=True)
    print(f"Target Directory: {out_dir}", flush=True)

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=out_dir,
        resume_download=True,
        max_workers=8
    )
    print(f"=== OakInk-v2 Download Complete -> {local_path} ===", flush=True)


if __name__ == "__main__":
    main()
