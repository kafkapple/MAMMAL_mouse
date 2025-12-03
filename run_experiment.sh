#!/bin/bash
# Run MAMMAL mesh fitting experiments
#
# Usage:
#   ./run_experiment.sh <experiment_name> [--debug] [--frames N]
#
# Experiments:
#   baseline_6view_keypoint   - 6-view RGB with 22 keypoints (paper baseline)
#   monocular_keypoint        - Single-view RGB with 22 keypoints
#   sixview_no_keypoint       - 6-view RGB silhouette only (no keypoints)
#   sixview_sparse_keypoint   - 6-view RGB with 3 sparse keypoints
#
# Examples:
#   ./run_experiment.sh baseline_6view_keypoint --debug    # Quick test (5 frames)
#   ./run_experiment.sh baseline_6view_keypoint            # Full run (100 frames)
#   ./run_experiment.sh monocular_keypoint --frames 50     # Custom frame count

set -e

# ===== GPU Configuration =====
# Auto-detect server and set GPU (can override with GPU_ID env var)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

# Parse arguments
EXPERIMENT="$1"
DEBUG_MODE=false
CUSTOM_FRAMES=""

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            DEBUG_MODE=true
            shift
            ;;
        --frames|-f)
            CUSTOM_FRAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate experiment name
VALID_EXPERIMENTS=(
    "baseline_6view_keypoint"
    "monocular_keypoint"
    "sixview_no_keypoint"
    "sixview_sparse_keypoint"
    "quick_test"
)

if [[ -z "$EXPERIMENT" ]]; then
    echo "================================================"
    echo "MAMMAL Mesh Fitting Experiments"
    echo "================================================"
    echo ""
    echo "Available experiments:"
    for exp in "${VALID_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
    echo ""
    echo "Usage: ./run_experiment.sh <experiment_name> [--debug] [--frames N]"
    echo ""
    exit 0
fi

# Check if experiment exists
if [[ ! -f "conf/experiment/${EXPERIMENT}.yaml" ]]; then
    echo "Error: Experiment config not found: conf/experiment/${EXPERIMENT}.yaml"
    echo "Available experiments:"
    ls conf/experiment/*.yaml | xargs -n1 basename | sed 's/.yaml//'
    exit 1
fi

# Build command
CMD="python fitter_articulation.py experiment=$EXPERIMENT"

# Debug mode: override with minimal settings
if [[ "$DEBUG_MODE" == "true" ]]; then
    echo "================================================"
    echo "DEBUG MODE: Quick test with 5 frames"
    echo "================================================"
    CMD="$CMD fitter.end_frame=5"
    CMD="$CMD optim.solve_step0_iters=5"
    CMD="$CMD optim.solve_step1_iters=20"
    CMD="$CMD optim.solve_step2_iters=10"
fi

# Custom frame count
if [[ -n "$CUSTOM_FRAMES" ]]; then
    CMD="$CMD fitter.end_frame=$CUSTOM_FRAMES"
fi

echo ""
echo "================================================"
echo "Experiment: $EXPERIMENT"
echo "Debug mode: $DEBUG_MODE"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, EGL_DEVICE_ID=$EGL_DEVICE_ID)"
if [[ -n "$CUSTOM_FRAMES" ]]; then
    echo "Frames: $CUSTOM_FRAMES"
fi
echo "================================================"
echo ""
echo "Running: $CMD"
echo ""

# Run
eval $CMD

echo ""
echo "================================================"
echo "Experiment complete: $EXPERIMENT"
echo "================================================"
