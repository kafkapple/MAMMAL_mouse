#!/bin/bash
# MAMMAL_mouse Environment Setup for gpu03 server
# Conda: ~/anaconda3, Env: mammal_stable

set -e
CONDA_PATH="$HOME/anaconda3"
ENV_NAME="mammal_stable"

echo "=== MAMMAL_mouse Setup for gpu03 server ==="
echo "Conda: $CONDA_PATH"
echo "Environment: $ENV_NAME"

# Initialize conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Check if environment exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists. Updating..."
    conda activate $ENV_NAME
else
    echo "Creating environment $ENV_NAME..."
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME
    
    # Install PyTorch via pip (more stable than conda)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install PyTorch3D (prebuilt wheel for cu121/py310/torch2.5.1)
    pip install fvcore iopath
    pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html
fi

# Install/update requirements
cd "$(dirname "$0")/../.."
pip install -r requirements.txt

echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
