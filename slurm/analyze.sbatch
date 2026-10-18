#!/bin/bash
# Steps 05-11 for the given models in a batch job instead of the shared login node. Large embeddings
# (MUSK is 1024-dim) make the scrambling and linkage analyses heavy multi-core work.
#   sbatch --export=ALL,MODELS=musk slurm/analyze.sbatch
# Afterwards, on the login node: bash sync_results.sh
#SBATCH --job-name=tf-analyze
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
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
MODELS=${MODELS:-"pathgen quilt plip musk"}

for m in $MODELS; do
    echo "===== $m ====="
    python 05_tests.py --model "$m" && \
    python 06_scramble.py --model "$m" && \
    python 07_remove.py --model "$m" && \
    python 09_heatmaps.py --model "$m" && \
    python 10_linkage.py --model "$m" && \
    python 10_linkage.py --model "$m" --split site || echo "!!! $m failed"
done
python 08_scorecard.py
python 11_figures.py
echo ANALYZE_DONE
