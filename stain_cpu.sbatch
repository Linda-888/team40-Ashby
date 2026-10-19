#!/bin/bash
# Feature 1, Part B: re-embed tiles with PLIP as-is and after Macenko stain normalization, then
# re-run Test 1. CPU only (team CPU allocation, no GPU hours). Submit from ~/team40-Ashby:
#   sbatch stain_cpu.sbatch
#SBATCH --job-name=tf-stain
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=stain-%j.out

cd "$SLURM_SUBMIT_DIR"
export TF_ROOT=/work/nvme/bfqi/lhuang17/tissue-fingerprints
export TF_WORK=/work/nvme/bfqi/lhuang17/tissue-fingerprints
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
~/demo-env/bin/python 14_fingerprint_sees.py stain --threads 32 --tiles-per-slide 50 --out ~/fingerprint_sees
echo STAIN_DONE
