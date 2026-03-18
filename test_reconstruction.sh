#!/bin/bash
#SBATCH -A 3dv
#SBATCH -p jobs
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /work/scratch/dmonopoli/recon_test_%j.out

# 1. Entra nella cartella del progetto
cd ~/FF-4DGS-Ego

# 2. Attiva l'ambiente Conda
source ~/miniconda3/bin/activate neoverse

# 3. Ottimizzazione Memoria CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Configura i percorsi
CLIP_NAME="P0001_10a27bf7"
INPUT_PATH="/work/courses/3dv/team25/data/hot3d_aria/sequences/$CLIP_NAME/video_main_rgb.mp4"
OUTPUT_DIR="/work/scratch/dmonopoli/output_test_$CLIP_NAME"

mkdir -p $OUTPUT_DIR

echo "Avvio test con 16 frame a 256x256 su 2080ti..."

# 5. Esegui la ricostruzione
python scripts/reconstruct_4dgs.py \
    --input_path "$INPUT_PATH" \
    --reconstructor_path "models/reconstructor.ckpt" \
    --output_dir "$OUTPUT_DIR" \
    --num_frames 16 \
    --height 224 \
    --width 224 \
    --sampling "first" \
    --render_video