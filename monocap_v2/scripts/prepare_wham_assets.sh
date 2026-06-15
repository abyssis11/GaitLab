#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-monocap-wham}"
CONDA_BIN="${CONDA_BIN:-/home/denik/miniconda3/bin/conda}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHAM_REPO="${WHAM_REPO:-${REPO_ROOT}/external/WHAM}"
SMPL_DIR="${SMPL_DIR:-${REPO_ROOT}/models/smpl}"
CHECKPOINT_DIR="${WHAM_REPO}/checkpoints"
BODY_MODEL_DIR="${WHAM_REPO}/dataset/body_models"

mkdir -p "${CHECKPOINT_DIR}" "${BODY_MODEL_DIR}/smpl"

link_smpl() {
  local source_name="$1"
  local target_name="$2"
  local source_path="${SMPL_DIR}/${source_name}"
  local target_path="${BODY_MODEL_DIR}/smpl/${target_name}"

  if [[ ! -e "${source_path}" ]]; then
    echo "Missing SMPL source file: ${source_path}" >&2
    return 1
  fi
  if [[ ! -e "${target_path}" ]]; then
    ln -s "${source_path}" "${target_path}"
  fi
}

download_if_missing() {
  local file_name="$1"
  local gdrive_id="$2"
  local output_path="${CHECKPOINT_DIR}/${file_name}"

  if [[ -s "${output_path}" ]]; then
    echo "Found ${output_path}"
    return
  fi
  "${CONDA_BIN}" run -n "${ENV_NAME}" gdown "https://drive.google.com/uc?id=${gdrive_id}&export=download&confirm=t" -O "${output_path}"
}

link_smpl "basicmodel_m_lbs_10_207_0_v1.1.0.pkl" "SMPL_MALE.pkl"
link_smpl "basicmodel_f_lbs_10_207_0_v1.1.0.pkl" "SMPL_FEMALE.pkl"
link_smpl "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" "SMPL_NEUTRAL.pkl"

BODY_ARCHIVE="${WHAM_REPO}/dataset/body_models.tar.gz"
if [[ ! -s "${BODY_MODEL_DIR}/smpl_mean_params.npz" ]]; then
  "${CONDA_BIN}" run -n "${ENV_NAME}" gdown "https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t" -O "${BODY_ARCHIVE}"
  tar -xzf "${BODY_ARCHIVE}" -C "${WHAM_REPO}/dataset"
fi

download_if_missing "wham_vit_w_3dpw.pth.tar" "1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB"
download_if_missing "wham_vit_bedlam_w_3dpw.pth.tar" "19qkI-a6xuwob9_RFNSPWf1yWErwVVlks"
download_if_missing "hmr2a.ckpt" "1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y"
download_if_missing "dpvo.pth" "1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT"
download_if_missing "yolov8x.pt" "1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo"
download_if_missing "vitpose-h-multi-coco.pth" "1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso"

echo "WHAM assets are prepared under ${WHAM_REPO}."
echo "Run: ${CONDA_BIN} run -n ${ENV_NAME} python ${REPO_ROOT}/monocap_v2/scripts/check_wham_prereqs.py"
