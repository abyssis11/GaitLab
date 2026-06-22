#!/usr/bin/env bash
set -euo pipefail

# Reference setup for the experimental monocap_v2 SAM3D Body backend.
# This script is intentionally not called by the pipeline. Run commands manually
# so Hugging Face access, CUDA wheels, and external repo state stay explicit.

ENV_NAME="${ENV_NAME:-monocap-sam3d-body}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SAM3D_REPO="${SAM3D_REPO:-${REPO_ROOT}/external/sam-3d-body}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/models/sam3d_body/sam-3d-body-dinov3}"

echo "Create conda env:"
echo "  conda create -n ${ENV_NAME} python=3.11 -y"
echo "  conda activate ${ENV_NAME}"
echo
echo "Install PyTorch CUDA stack, then SAM3D Body requirements:"
echo "  pip install --upgrade pip"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
echo "  git clone https://github.com/facebookresearch/sam-3d-body.git ${SAM3D_REPO}"
echo "  pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub"
echo "  pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps"
echo "  pip install git+https://github.com/microsoft/MoGe.git"
echo
echo "Download Hugging Face assets after access is granted:"
echo "  mkdir -p ${CHECKPOINT_DIR}/assets"
echo "  # Place model.ckpt at: ${CHECKPOINT_DIR}/model.ckpt"
echo "  # Place mhr_model.pt at: ${CHECKPOINT_DIR}/assets/mhr_model.pt"
echo "  # ViT-H fallback may also be placed at models/sam3d_body/sam-3d-body-vith/."
echo
echo "Verify from repo root:"
echo "  /home/denik/miniconda3/envs/${ENV_NAME}/bin/python monocap_v2/scripts/check_sam3d_prereqs.py --strict"
