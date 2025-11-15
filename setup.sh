#!/bin/bash

# MAMMAL_mouse Environment Setup Script
# This script creates the mammal_stable environment and installs all dependencies
# Run this script only once during initial setup or when updating dependencies

set -e  # Exit immediately if a command exits with a non-zero status

echo "========================================="
echo "MAMMAL_mouse Environment Setup"
echo "========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA found: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo "⚠️  Warning: CUDA not found in PATH"
    echo "The installation will proceed, but GPU acceleration may not work"
fi

# --- Clean conda cache ---
echo ""
echo "--- Cleaning conda cache ---"
conda clean --all -y

# --- Remove existing environment if present ---
echo ""
echo "--- Checking for existing mammal_stable environment ---"
if conda env list | grep -q "^mammal_stable "; then
    echo "Found existing mammal_stable environment. Removing..."
    conda env remove -n mammal_stable -y
fi

# --- Create new environment ---
echo ""
echo "--- Creating mammal_stable environment with Python 3.10 ---"
conda create -n mammal_stable python=3.10 -y

echo ""
echo "--- Activating mammal_stable environment ---"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mammal_stable

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "mammal_stable" ]]; then
    echo "❌ Error: Failed to activate mammal_stable environment"
    exit 1
fi
echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# --- Install PyTorch with CUDA 11.8 ---
echo ""
echo "--- Installing PyTorch 2.0.0 with CUDA 11.8 ---"
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# --- Install NumPy and TensorBoard ---
echo ""
echo "--- Installing NumPy and TensorBoard ---"
pip install "numpy<2.0" tensorboard==2.13.0

# --- Install requirements.txt ---
echo ""
echo "--- Installing packages from requirements.txt ---"
pip install -r requirements.txt

# --- Install PyTorch3D dependencies ---
echo ""
echo "--- Installing PyTorch3D dependencies (fvcore, iopath) ---"
pip install fvcore iopath

# --- Install PyTorch3D ---
echo ""
echo "--- Installing PyTorch3D 0.7.5 for PyTorch 2.0.0 + CUDA 11.8 ---"
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html

# --- Verify installation ---
echo ""
echo "========================================="
echo "Installation Complete! Verifying..."
echo "========================================="

# Test PyTorch
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'✅ CUDA device: {torch.cuda.get_device_name(0)}')"
fi

# Test PyTorch3D
python -c "import pytorch3d; print(f'✅ PyTorch3D {pytorch3d.__version__} installed')"

# Test Hydra
python -c "import hydra; print('✅ Hydra installed')"

# Test OpenCV
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} installed')"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "    conda activate mammal_stable"
echo ""
echo "To run preprocessing on a video:"
echo "    bash run_preprocess.sh"
echo ""
echo "To run 3D fitting:"
echo "    bash run_fitting.sh"
echo ""
