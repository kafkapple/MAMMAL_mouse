#!/bin/bash

# MAMMAL_mouse Preprocessing Script
# This script runs the preprocessing pipeline to generate masks and keypoints
# from a single-view video

set -e  # Exit immediately if a command exits with a non-zero status

echo "========================================="
echo "MAMMAL_mouse Preprocessing"
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
cfg = OmegaConf.load('conf/config.yaml')
print(f'Mode: {cfg.mode}')
if cfg.mode == 'single_view_preprocess':
    print(f'Input video: {cfg.preprocess.input_video_path}')
    print(f'Output directory: {cfg.preprocess.output_data_dir}')
else:
    print('⚠️  Warning: config.yaml mode is not set to single_view_preprocess')
    print('Please update conf/config.yaml to set mode: single_view_preprocess')
"
echo ""

# Run preprocessing
echo "--- Running preprocess.py ---"
python preprocess.py

echo ""
echo "========================================="
echo "Preprocessing Complete!"
echo "========================================="
echo ""
echo "Output files generated in the configured output_data_dir:"
echo "  - videos_undist/0.mp4          (original video)"
echo "  - simpleclick_undist/0.mp4     (mask video)"
echo "  - keypoints2d_undist/result_view_0.pkl  (2D keypoints)"
echo "  - new_cam.pkl                  (camera parameters)"
echo ""
echo "Next steps:"
echo "1. Update conf/config.yaml:"
echo "   - Set mode: multi_view"
echo "   - Set data.data_dir to your output_data_dir"
echo "   - Set data.views_to_use: [0]"
echo "2. Run: bash run_fitting.sh"
echo ""
