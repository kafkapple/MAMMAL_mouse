#!/bin/bash
# Run all 4 main experiments sequentially
#
# Usage:
#   ./run_all_experiments.sh           # Full run (100 frames)
#   ./run_all_experiments.sh --debug   # Debug mode (5 frames)
#   ./run_all_experiments.sh --frames 50  # Custom frame count
#
# GPU Selection:
#   GPU_ID=0 ./run_all_experiments.sh  # Use GPU 0
#   GPU_ID=1 ./run_all_experiments.sh  # Use GPU 1 (default)
#
# Results will be saved to results/fitting/ with timestamps

set -e  # Exit on error

# ===== GPU Configuration =====
# Force both CUDA compute and EGL rendering to use the same GPU
# Default: GPU 1 (can override with GPU_ID environment variable)
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EGL_DEVICE_ID="$GPU_ID"
export DISPLAY=""  # Disable X11 rendering (headless mode)
export PYOPENGL_PLATFORM=egl

# Parse arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS="$EXTRA_ARGS $1"
    shift
done

EXPERIMENTS=(
    "baseline_6view_keypoint"
    "monocular_keypoint"
    "sixview_no_keypoint"
    "sixview_sparse_keypoint"
)

echo "================================================"
echo "Running All 4 Experiments"
echo "GPU: $GPU_ID (CUDA=$CUDA_VISIBLE_DEVICES, EGL=$EGL_DEVICE_ID)"
echo "Arguments: $EXTRA_ARGS"
echo "================================================"
echo ""

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================"
    echo "[$CURRENT/$TOTAL] $exp"
    echo "========================================"

    ./run_experiment.sh "$exp" $EXTRA_ARGS

    echo ""
    echo "Completed: $exp"
    echo ""
done

echo "================================================"
echo "All experiments completed!"
echo "================================================"
echo ""
echo "Results saved in: results/fitting/"
echo ""
echo "To generate comparison report:"
echo "  python scripts/evaluate_experiment.py results/fitting/<baseline_folder> \\"
echo "      --compare results/fitting/<monocular_folder> \\"
echo "               results/fitting/<no_keypoint_folder> \\"
echo "               results/fitting/<sparse_folder>"
