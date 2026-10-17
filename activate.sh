# Usage: source activate.sh main    (CONCH, PathGen-CLIP, tiling, analysis)
#        source activate.sh musk    (MUSK only)
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
source "$TF_ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$TF_ENVS/tf-${1:-main}"
