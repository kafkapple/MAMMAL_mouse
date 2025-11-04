#!/bin/bash

# MAMMAL_mouse Fitting Script
# This script runs the 3D mouse model fitting on preprocessed data

set -e  # Exit immediately if a command exits with a non-zero status

echo "========================================="
echo "MAMMAL_mouse 3D Fitting"
echo "========================================="
echo ""

# Check if environment exists
if ! conda env list | grep -q "^mammal_stable "; then
    echo "❌ Error: mammal_stable environment not found"
    echo "Please run setup.sh first to create the environment"
    exit 1
fi

# Activate environment
echo "--- Activating mammal_stable environment ---"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mammal_stable

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "mammal_stable" ]]; then
    echo "❌ Error: Failed to activate mammal_stable environment"
    exit 1
fi
echo "✅ Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Set EGL rendering (for headless servers)
export PYOPENGL_PLATFORM=egl

# Set CUDA paths (adjust if your CUDA is installed elsewhere)
if [ -d "/usr/local/cuda-11.8" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    echo "✅ CUDA 11.8 paths configured"
fi

# Display configuration
echo "--- Configuration ---"
python -c "
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load('conf/config.yaml')
print(f'Mode: {cfg.mode}')
print(f'Data directory: {cfg.data.data_dir}')
print(f'Views to use: {cfg.data.views_to_use}')
print(f'Frames: {cfg.fitter.start_frame} to {cfg.fitter.end_frame}')
print(f'With render: {cfg.fitter.with_render}')

# Check if data directory exists
if not os.path.exists(cfg.data.data_dir):
    print(f'⚠️  Warning: Data directory does not exist: {cfg.data.data_dir}')
"
echo ""

# Check if required data files exist
echo "--- Checking required data files ---"
python -c "
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load('conf/config.yaml')
data_dir = cfg.data.data_dir

required_files = [
    'new_cam.pkl',
    'keypoints2d_undist/result_view_0.pkl',
    'videos_undist/0.mp4',
    'simpleclick_undist/0.mp4'
]

all_exist = True
for file in required_files:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        print(f'✅ {file}')
    else:
        print(f'❌ {file} - NOT FOUND')
        all_exist = False

if not all_exist:
    print()
    print('⚠️  Some required files are missing.')
    print('Please run preprocessing first: bash run_preprocess.sh')
    exit(1)
" || exit 1

echo ""

# Run fitting
echo "--- Running fitter_articulation.py ---"
echo "This may take several minutes depending on the number of frames..."
echo ""

python fitter_articulation.py

echo ""
echo "========================================="
echo "Fitting Complete!"
echo "========================================="
echo ""
echo "Results saved to: mouse_fitting_result/results/"
echo ""
echo "Output files:"
echo "  - obj/                   3D mesh files (.obj)"
echo "  - params/                Fitting parameters (.pkl)"
echo "  - render/                Rendered overlays (.png)"
echo "  - fitting_keypoints_*.png  Keypoint visualizations"
echo ""
echo "To create a video from the output images, run:"
echo "  ffmpeg -framerate 10 -i mouse_fitting_result/results/render/fitting_%d.png \\"
echo "         -c:v libx264 -pix_fmt yuv420p -y mouse_fitting_result/results/output.mp4"
echo ""
