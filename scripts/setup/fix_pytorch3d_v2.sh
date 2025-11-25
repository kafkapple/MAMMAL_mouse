#!/bin/bash
# Fix PyTorch3D compatibility issue - Version 2
# Downloads CUB and compiles PyTorch3D from source

echo "================================================"
echo "Fixing PyTorch3D Compatibility Issue (v2)"
echo "================================================"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mammal_stable

echo ""
echo "Current versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo ""
echo "Step 1: Downloading NVIDIA CUB..."
cd /tmp
rm -rf cub-1.17.2 cub-1.17.2.tar.gz
wget https://github.com/NVIDIA/cub/archive/1.17.2.tar.gz -O cub-1.17.2.tar.gz
tar -xzf cub-1.17.2.tar.gz
export CUB_HOME=/tmp/cub-1.17.2
echo "✓ CUB downloaded to: $CUB_HOME"

echo ""
echo "Step 2: Setting CUDA environment..."
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "✓ CUDA_HOME: $CUDA_HOME"

echo ""
echo "Step 3: Uninstalling existing PyTorch3D..."
pip uninstall pytorch3d -y

echo ""
echo "Step 4: Installing build dependencies..."
pip install fvcore iopath

echo ""
echo "Step 5: Installing PyTorch3D 0.7.5 from source..."
echo "This will compile C++/CUDA code and may take 10-15 minutes..."
echo ""

# Install with verbose output
pip install --verbose "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Step 6: Verifying installation..."
    echo "================================================"

    python -c "
import torch
import pytorch3d
from pytorch3d import _C
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ PyTorch3D: {pytorch3d.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print('✓ PyTorch3D C++ extensions loaded successfully')
"

    if [ $? -eq 0 ]; then
        echo ""
        echo "================================================"
        echo "✓✓✓ PyTorch3D FIXED SUCCESSFULLY! ✓✓✓"
        echo "================================================"
        echo ""
        echo "You can now run mesh fitting:"
        echo "  ./run_mesh_fitting_cropped.sh data/100-KO-male-56-20200615_cropped"
        echo ""
        exit 0
    else
        echo ""
        echo "✗ Verification failed. Check errors above."
        exit 1
    fi
else
    echo ""
    echo "================================================"
    echo "✗ Installation failed."
    echo "================================================"
    echo ""
    echo "Common solutions:"
    echo "1. Check CUDA toolkit is installed: nvcc --version"
    echo "2. Ensure you have GCC: gcc --version"
    echo "3. Try with fewer parallel jobs: MAX_JOBS=2 pip install ..."
    echo ""
    echo "For more help, see: docs/PYTORCH3D_FIX.md"
    exit 1
fi
