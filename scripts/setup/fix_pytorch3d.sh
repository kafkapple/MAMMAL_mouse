#!/bin/bash
# Fix PyTorch3D compatibility issue
# This script reinstalls PyTorch3D compiled for your specific PyTorch version

echo "================================================"
echo "Fixing PyTorch3D Compatibility Issue"
echo "================================================"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mammal_stable

echo ""
echo "Current versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
pip show pytorch3d | grep Version

echo ""
echo "Uninstalling existing PyTorch3D..."
pip uninstall pytorch3d -y

echo ""
echo "Installing PyTorch3D 0.7.5 (compatible with PyTorch 2.0.0)..."
echo "This will compile from source and may take 5-10 minutes..."

# Install dependencies
pip install fvcore iopath

# Install PyTorch3D from source at specific version
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

echo ""
echo "================================================"
echo "Verifying installation..."
echo "================================================"

# Test import
python -c "
import torch
import pytorch3d
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ PyTorch3D: {pytorch3d.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')

# Test PyTorch3D import
from pytorch3d import _C
print('✓ PyTorch3D C++ extensions loaded successfully')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✓ PyTorch3D fixed successfully!"
    echo "================================================"
    echo ""
    echo "You can now run mesh fitting:"
    echo "  ./run_mesh_fitting_cropped.sh data/100-KO-male-56-20200615_cropped"
else
    echo ""
    echo "================================================"
    echo "✗ Installation failed. Please check errors above."
    echo "================================================"
    exit 1
fi
