#!/bin/bash
# Run MAMMAL mesh fitting experiments
#
# Usage:
#   ./run_experiment.sh <experiment_name> [--debug] [--frames N]
#
# ============ Experiment Groups ============
#
# Group 1: BASELINE (Paper reference)
#   baseline_6view_keypoint   - 6-view RGB + 22 keypoints (MAMMAL paper baseline)
#
# Group 2: Keypoint Ablation (6-view fixed)
#   sixview_sparse_keypoint   - 6-view + 3 sparse keypoints (nose, body, tail)
#   sixview_no_keypoint       - 6-view + silhouette only (no keypoints)
#
# Group 3: Viewpoint Ablation (sparse 3 keypoints fixed)
#   sparse_5view              - 5-view (0,1,2,3,4) + sparse 3
#   sparse_4view              - 4-view (0,1,2,3) + sparse 3
#   sparse_3view              - 3-view diagonal (0,2,4) + sparse 3
#   sparse_2view              - 2-view opposite (0,3) + sparse 3
#
# ============ Examples ============
#   ./run_experiment.sh baseline_6view_keypoint --debug    # Quick test (5 frames)
#   ./run_experiment.sh baseline_6view_keypoint            # Full run (100 frames)
#   ./run_experiment.sh sparse_3view --frames 50           # Custom frame count

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
    # Group 1: Baseline (Paper reference)
    "baseline_6view_keypoint"    # 6-view + 22 keypoints (MAMMAL paper baseline)
    # Group 2: Keypoint ablation (6-view fixed)
    "sixview_sparse_keypoint"    # 6-view + 3 sparse keypoints
    "sixview_no_keypoint"        # 6-view + silhouette only (no keypoints)
    # Group 3: Viewpoint ablation (sparse 3 keypoints fixed)
    "sparse_5view"               # 5-view (0,1,2,3,4) + sparse 3
    "sparse_4view"               # 4-view (0,1,2,3) + sparse 3
    "sparse_3view"               # 3-view diagonal (0,2,4) + sparse 3
    "sparse_2view"               # 2-view opposite (0,3) + sparse 3
    # Other
    "monocular_keypoint"
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
