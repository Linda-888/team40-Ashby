#!/bin/bash
# Step 3: tile every slide in slides.csv, split across 4 array tasks. Pure CPU work, so it runs
# on the team's CPU allocation (bfqi-delta-cpu) and costs no GPU hours.
#SBATCH --job-name=tf-tile
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
source activate.sh main
python 02_tile.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4 --workers 32
