#!/bin/bash
# Limitation fixes (15_robustness.py): stronger statistics, a trained attacker, tile pooling.
# Runs on cached embeddings only (CPU work; the interactive GPU lane just starts fastest).
#   cd ~/team40-Ashby && sbatch robust.sbatch
#SBATCH --job-name=tf-robust
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=robust-%j.out

cd "$SLURM_SUBMIT_DIR"
export TF_ROOT=/work/nvme/bfqi/lhuang17/tissue-fingerprints
export TF_WORK=/work/nvme/bfqi/lhuang17/tissue-fingerprints
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
~/demo-env/bin/python 15_robustness.py --out ~/robustness
echo ROBUST_DONE
