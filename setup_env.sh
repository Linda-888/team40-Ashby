#!/bin/bash
# One-time setup on a Delta LOGIN node (it needs internet). Takes ~20 min.
#   bash setup_env.sh            # everything
#   bash setup_env.sh weights    # only (re)download model weights
#
# Two envs on purpose: MUSK pins torch==2.0.1 / timm==0.9.8, which would fight CONCH and
# PathGen-CLIP. Every script imports models lazily, so the same code runs in either env.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

mkdir -p "$TF_ROOT"/{data,emb,results,weights,src} "$TF_WORK/tiles" "$TF_ENVS" "$HF_HOME"

install_conda() {
    # Our own Miniforge in project space: no guessing Delta module names, shared by the team.
    if [ ! -x "$TF_ROOT/miniforge3/bin/conda" ]; then
        curl -fsSL -o "$TF_WORK/miniforge.sh" \
            https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash "$TF_WORK/miniforge.sh" -b -p "$TF_ROOT/miniforge3"
        rm "$TF_WORK/miniforge.sh"
    fi
    source "$TF_ROOT/miniforge3/etc/profile.d/conda.sh"
}

install_main() {
    # CONCH + PathGen-CLIP + OpenSlide + analysis.
    [ -d "$TF_ENVS/tf-main" ] || conda create -y -p "$TF_ENVS/tf-main" python=3.10
    conda activate "$TF_ENVS/tf-main"
    # torch >= 2.6: transformers refuses to load .bin checkpoints (e.g. PLIP) on older torch (CVE-2025-32434)
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    pip install "timm>=0.9.8,<1.0" "transformers<5" tokenizers open_clip_torch \
        openslide-python openslide-bin scikit-learn scipy pandas matplotlib pillow tqdm \
        huggingface_hub ftfy regex h5py ipykernel
    [ -d "$TF_ROOT/src/CONCH" ] || git clone https://github.com/mahmoodlab/CONCH.git "$TF_ROOT/src/CONCH"
    pip install -e "$TF_ROOT/src/CONCH"
    python -m ipykernel install --user --name tf-main --display-name "tf-main (CONCH, PathGen)"
    conda deactivate
}

install_musk() {
    [ -d "$TF_ENVS/tf-musk" ] || conda create -y -p "$TF_ENVS/tf-musk" python=3.10
    conda activate "$TF_ENVS/tf-musk"
    [ -d "$TF_ROOT/src/MUSK" ] || git clone https://github.com/lilab-stanford/MUSK.git "$TF_ROOT/src/MUSK"
    pip install -r "$TF_ROOT/src/MUSK/requirements.txt"
    pip install -e "$TF_ROOT/src/MUSK"
    pip install pandas tqdm scikit-learn scipy matplotlib huggingface_hub ipykernel
    python -m ipykernel install --user --name tf-musk --display-name "tf-musk (MUSK)"
    conda deactivate
}

download_weights() {
    conda activate "$TF_ENVS/tf-main"
    # CONCH and MUSK are gated. Skipped when config.sh points at local copies (none were pre-loaded on 10/17);
    # otherwise request access on Hugging Face first. Your token stays in your home dir (HF_TOKEN_PATH).
    python - <<'EOF'
import os
from huggingface_hub import get_token, snapshot_download, hf_hub_download
# Ungated models first, so they are ready even if gated access is still pending.
snapshot_download("wisdomik/QuiltNet-B-32")
snapshot_download("vinid/plip")
print("QuiltNet and PLIP downloaded")
need_conch, need_musk = not os.environ.get("TF_CONCH_CKPT"), not os.environ.get("TF_MUSK_CKPT")
if need_conch or need_musk:
    print("TF_CONCH_CKPT / TF_MUSK_CKPT not set, downloading gated weights from Hugging Face")
    if not get_token():
        print("No Hugging Face token yet: skipping CONCH/MUSK. Log in, then run: bash setup_env.sh weights")
        raise SystemExit(0)
    if need_conch:
        snapshot_download("MahmoodLab/conch")
    if need_musk:
        hf_hub_download("xiangjx/musk", filename="model.safetensors",
                        local_dir=os.path.join(os.environ["HF_HOME"], "musk"))
    print("gated weights downloaded")
EOF
    if [ ! -f "$TF_PATHGEN_CKPT" ]; then
        curl -fL -o "$TF_PATHGEN_CKPT" https://pub-7a38cc906afa44a4a01533c288d0b1af.r2.dev/pathgenclip.pt
    fi
    conda deactivate
}

install_conda
if [ "${1:-all}" = "weights" ]; then
    download_weights
else
    install_main
    install_musk
    download_weights
fi
echo "Done. Next: sbatch slurm/check_models.sbatch  (or: source activate.sh main; python 00_check_models.py)"
