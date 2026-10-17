#!/bin/bash
# Steps 4-5: one array task per model, images then reports.
# 0=conch 1=musk 2=pathgen (the plan). Ungated backups: sbatch --array=3-4 slurm/embed.sbatch
# (3=quilt 4=plip). One model only: sbatch --array=2 slurm/embed.sbatch
# Re-running skips slides that are already embedded. The GPU budget is small (10 GPU-hours per
# person); 60 patients take minutes per model, so 2 hours is plenty.
#SBATCH --job-name=tf-embed
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
MODELS=(conch musk pathgen quilt plip)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
if [ "$MODEL" = musk ]; then source activate.sh musk; else source activate.sh main; fi

python 03_embed_images.py --model "$MODEL"
if [ -f "$TF_REPORTS_CSV" ]; then
    python 04_embed_reports.py --model "$MODEL"
else
    echo "No reports CSV at $TF_REPORTS_CSV, skipping report embeddings"
fi
