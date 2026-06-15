#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-monocap-wham}"
CONDA_BIN="${CONDA_BIN:-/home/denik/miniconda3/bin/conda}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHAM_REPO="${WHAM_REPO:-${REPO_ROOT}/external/WHAM}"
TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.0.2}"
PYTORCH_CUDA="${PYTORCH_CUDA:-11.8}"

"${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.9
"${CONDA_BIN}" install -y -n "${ENV_NAME}" \
  "pytorch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "pytorch-cuda=${PYTORCH_CUDA}" \
  -c pytorch -c nvidia
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

if [[ ! -d "${WHAM_REPO}/.git" ]]; then
  git clone --recursive https://github.com/yohanshin/WHAM.git "${WHAM_REPO}"
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install setuptools==59.5.0
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --no-build-isolation -r "${WHAM_REPO}/requirements.txt"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -v -e "${WHAM_REPO}/third-party/ViTPose"
"${CONDA_BIN}" install -y -n "${ENV_NAME}" gxx=9.5 -c conda-forge

DPVO_DIR="${WHAM_REPO}/third-party/DPVO"
mkdir -p "${DPVO_DIR}/thirdparty"
if [[ ! -d "${DPVO_DIR}/thirdparty/eigen-3.4.0" ]]; then
  wget -O "${DPVO_DIR}/eigen-3.4.0.zip" https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m zipfile -e "${DPVO_DIR}/eigen-3.4.0.zip" "${DPVO_DIR}/thirdparty"
fi

echo "Base WHAM env setup is complete. Defaults use PyTorch ${TORCH_VERSION} with CUDA ${PYTORCH_CUDA} for RTX 40-series WSL compatibility."
echo "DPVO native build is intentionally separate because WHAM's CUDA 11.3-era DPVO dependencies conflict with this stack."
echo "Next: ${REPO_ROOT}/monocap_v2/scripts/prepare_wham_assets.sh"
echo "Then: ${CONDA_BIN} run -n ${ENV_NAME} python ${REPO_ROOT}/monocap_v2/scripts/check_wham_prereqs.py"
