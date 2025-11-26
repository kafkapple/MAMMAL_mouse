#!/bin/bash
# Silhouette-only fitting experiments
# Run sequentially with different configurations to compare results

set -e  # Exit on error

# Environment setup
export PYOPENGL_PLATFORM=egl
PYTHON="${HOME}/miniconda3/envs/mammal_stable/bin/python"

# Default parameters
INPUT_DIR="${1:-data/examples/markerless_mouse_1_nerf}"
START_FRAME="${2:-0}"
END_FRAME="${3:-2}"  # Debug mode: only 2 frames
LOG_DIR="results/silhouette_experiments"

echo "================================================"
echo "Silhouette-Only Fitting Experiments"
echo "================================================"
echo "Input: $INPUT_DIR"
echo "Frames: $START_FRAME - $END_FRAME"
echo "Log dir: $LOG_DIR"
echo "================================================"

mkdir -p "$LOG_DIR"

# Function to run experiment
run_experiment() {
    local exp_name="$1"
    local extra_args="$2"

    echo ""
    echo "=========================================="
    echo "Running: $exp_name"
    echo "=========================================="

    local log_file="$LOG_DIR/${exp_name}.log"

    $PYTHON fitter_articulation.py \
        dataset=default_markerless \
        fitter.start_frame=$START_FRAME \
        fitter.end_frame=$END_FRAME \
        fitter.use_keypoints=false \
        fitter.with_render=true \
        --input_dir "$INPUT_DIR" \
        $extra_args \
        2>&1 | tee "$log_file"

    echo "Log saved to: $log_file"
    echo ""
}

# Experiment 1: Baseline (default silhouette settings)
run_experiment "exp1_baseline" \
    "silhouette.iter_multiplier=2.0 silhouette.theta_weight=10.0 silhouette.use_pca_init=true"

# Experiment 2: More iterations
run_experiment "exp2_more_iters" \
    "silhouette.iter_multiplier=3.0 silhouette.theta_weight=10.0 silhouette.use_pca_init=true"

# Experiment 3: Higher regularization
run_experiment "exp3_high_reg" \
    "silhouette.iter_multiplier=2.0 silhouette.theta_weight=15.0 silhouette.bone_weight=3.0 silhouette.use_pca_init=true"

# Experiment 4: No PCA initialization (for comparison)
run_experiment "exp4_no_pca" \
    "silhouette.iter_multiplier=2.0 silhouette.theta_weight=10.0 silhouette.use_pca_init=false"

echo ""
echo "================================================"
echo "All experiments completed!"
echo "================================================"
echo ""
echo "Results saved in: results/fitting/"
echo "Logs saved in: $LOG_DIR/"
echo ""
echo "To compare results, view the render images:"
echo "  ls -lt results/fitting/*/render/fitting_0_sil.png"
