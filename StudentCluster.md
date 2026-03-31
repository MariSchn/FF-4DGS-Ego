
# Quick Start for Student Cluster


### Config git repo
- git init
- git remote add origin https://github.com/MariSchn/FF-4DGS-Ego/tree/main
- git pull origin main

### Create Venv
- python3 -m venv venv
- source venv/bin/activate

### Install Packages (CUDA 12.8) (Recommended)
- pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
- pip install -r requirements.txt
- pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
- pip install gsplat
- pip install chumpy --no-build-isolation
	- use the modified version of chumpy, from the repo

### Install Packages (CUDA 12.1) (Not Revommended)
- pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
- pip install -r requirements.txt
- pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
...

### Download Model weights
- wget https://huggingface.co/Yuppie1204/NeoVerse/resolve/main/reconstructor.ckpt
- mkdir models/NeoVerse
- mv reconstructor.ckpt models/NeoVerse

### Config Weights & Biases
- wandb login

### Relevant Paths
- /work/courses/3dv/team25/data
- /home/{user}/

- tag = 3dv

### Running

Simple commands:
- srun -A 3dv -t 00:10 -o nvidia-smi.out nvidia-smi

Interactive jobs, for 60 minutes
- srun --pty -A 3dv -t 60 bash --login

Sbatch

batch.sh:
```bash
#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account={tag}
#SBATCH --output=nvidia-smi.out

# run stuff, ex.:
ls
nvidia-smi

```

type into cmd:
```
sbatch batch.sh
```