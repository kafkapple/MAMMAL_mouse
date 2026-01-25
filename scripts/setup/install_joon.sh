#!/bin/bash
# MAMMAL_mouse Environment Setup for joon (gpu05/bori) server
# Conda: ~/miniconda3, Env: mammal_stable

set -e
CONDA_PATH="$HOME/miniconda3"
ENV_NAME="mammal_stable"

echo "=== MAMMAL_mouse Setup for joon server ==="
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
    
    # Install PyTorch with CUDA
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    # Install PyTorch3D
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"
fi

# Install/update requirements
cd "$(dirname "$0")/../.."
pip install -r requirements.txt

echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
