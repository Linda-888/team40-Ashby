#!/bin/bash
# Step 1: load each model on 5 tiles. Submit from the repo root: sbatch slurm/check_models.sbatch
#SBATCH --job-name=tf-check
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
source activate.sh main
python 00_check_models.py --models conch pathgen quilt plip
source activate.sh musk
python 00_check_models.py --models musk
