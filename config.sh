# Shared settings for every script and sbatch job. Source this: `source config.sh`
# Edit the three paths below once you know where things live on Delta.

export TF_ACCOUNT="bfqi-delta-gpu"

# Everything (code, envs, slides, tiles, embeddings, results) lives on the team's NVMe work space:
# 4.9 TB, versus 500 GB of /projects/bfqi shared by ~200 hackathon users. Not backed up.
export TF_ROOT="/work/nvme/bfqi/${USER}/tissue-fingerprints"
export TF_WORK="${TF_ROOT}"

# The hackathon's shared data has no TCGA whole slides, so tcga_download.py fetches them from the GDC here.
export TF_SLIDES_DIR="${TF_ROOT}/slides"

# TCGA-Reports text corpus (patient_filename, text), also fetched by tcga_download.py.
export TF_REPORTS_CSV="${TF_ROOT}/data/TCGA_Reports.csv"

export TF_ENVS="${TF_ROOT}/envs"
export HF_HOME="${TF_ROOT}/hf_cache"          # model weights downloaded once, shared by the team
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"   # your token stays private
export TF_PATHGEN_CKPT="${TF_ROOT}/weights/pathgenclip.pt"

# If you find pre-loaded CONCH / MUSK weights on Delta, point these at them (a file or its folder)
# and no Hugging Face access is needed. (None were in /projects/bfqi/shared as of 10/17.) CONCH wants pytorch_model.bin, MUSK wants model.safetensors.
# Leave empty to download from Hugging Face instead (needs approved access).
export TF_CONCH_CKPT=""
export TF_MUSK_CKPT=""
