#!/bin/bash

python3 -m venv venv_gb10
source venv_gb10/bin/activate

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install gsplat
pip install chumpy --no-build-isolation
pip install pyyaml
