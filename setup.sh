#!/usr/bin/env bash
set -e

echo "=== SDKN GPU Setup gestartet ==="

# --- Base directory ---
export BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)"
cd "${BASEDIR}"

# --- Conda environment ---
ENV_NAME="sdkn-gpu-env"

# --- Sicherstellen, dass 'conda activate' funktioniert ---
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Channels konfigurieren ..."
conda config --remove-key channels || true
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels conda-forge
conda config --set channel_priority strict

# --- Existierendes Environment entfernen ---
if conda env list | grep -q "${ENV_NAME}"; then
    echo "Removing existing environment '${ENV_NAME}' ..."
    conda env remove -n "${ENV_NAME}" -y
fi

# --- Environment neu erstellen mit Python 3.10 ---
echo "Creating conda environment '${ENV_NAME}' with Python 3.10 ..."
conda create -y -n "${ENV_NAME}" python=3.10

# --- Aktivieren ---
conda activate "${ENV_NAME}"
echo "Environment aktiv: $CONDA_DEFAULT_ENV"

# --- Core Python packages ---
echo "Installing core packages..."
conda install -y \
    numpy==1.25.0 \
    pandas==2.0.3 \
    scipy==1.12.0 \
    scikit-learn==1.3.1 \
    matplotlib==3.8.0 \
    tqdm==4.66.1 \
    pyyaml==6.0 \
    fsspec==2023.1.0 \
    pyDeprecate==0.4.0 \
    setuptools==64.0.3

# --- GPU-fähiges PyTorch + Lightning installieren ---
echo "Installing CUDA-enabled PyTorch + Lightning ..."
conda install -y \
    pytorch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    pytorch-cuda=12.1 \
    pytorch-lightning==2.4.0 \
    torchmetrics==0.11.0 \
    tensorboard==2.14.0 \
    -c pytorch -c nvidia

# --- Additional packages ---
echo "Installing misc packages..."
conda install -y liac-arff==2.5.0

# --- GPU Test ---
echo "=== GPU Test wird ausgeführt ==="
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
else:
    print("Keine CUDA GPU gefunden.")
EOF

echo "=== Setup complete. Environment '${ENV_NAME}' fertig und GPU-enabled. ==="

read -p "Setup abgeschlossen. Enter drücken zum Beenden..."
