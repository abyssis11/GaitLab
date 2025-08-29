set -euo pipefail
GPU=${1:-}
sudo apt update -y
sudo apt install -y --no-install-recommends python3-venv python3-pip git ffmpeg build-essential pkg-config libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
if [[ "$GPU" == "--gpu" ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
echo "Done. Activate with: source .venv/bin/activate"
