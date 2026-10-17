#!/bin/bash
# Steps 3-5 in ONE GPU job: tile (on the 16 cores that come with the GPU), then embed images and
# reports for each model. Useful when the CPU partitions are full but A100s are idle (as on 10/17).
# Every step skips work that is already done, so resubmitting resumes.
#   sbatch slurm/pipeline_gpu.sbatch                             # pathgen quilt plip (+conch if set up)
#   sbatch --export=ALL,MODELS="conch pathgen" slurm/pipeline_gpu.sbatch
# MUSK runs in its own env: sbatch --array=1 slurm/embed.sbatch
#SBATCH --job-name=tf-pipe
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
source activate.sh main
MODELS=${MODELS:-"pathgen quilt plip"}

python 02_tile.py --workers 16 || exit 1
for m in $MODELS; do
    echo "===== $m ====="
    python 03_embed_images.py --model "$m" --workers 8 && \
        python 04_embed_reports.py --model "$m" || echo "!!! $m failed, continuing"
done
