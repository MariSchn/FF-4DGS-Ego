#!/bin/bash

#SBATCH --gpus=gb10:1
#SBATCH --time=24:00:00
#SBATCH --account=3dv
#SBATCH --job-name=ff4dgs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

python3 -m venv venv_gb10
source venv_gb10/bin/activate

# Redirect pip cache to /work to avoid home directory quota issues
export PIP_CACHE_DIR=/scratch/miliev/pip_cache

# PyPI provides CUDA-enabled aarch64 wheels for PyTorch directly
pip install torch==2.7.1 torchvision==0.22.1
pip install -r requirements.txt
# Build torch-scatter from source; --no-build-isolation lets setup.py see the installed torch
pip install torch-scatter --no-build-isolation
pip install gsplat
pip install chumpy --no-build-isolation
pip install pyyaml

# Build decord from local source (no aarch64 PyPI wheel available)
cd decord/build && make -j$(nproc) && cd ../..
pip install ./decord/python
