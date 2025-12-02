#!/bin/bash
# Run all 4 main experiments sequentially
#
# Usage:
#   ./run_all_experiments.sh           # Full run (100 frames)
#   ./run_all_experiments.sh --debug   # Debug mode (5 frames)
#   ./run_all_experiments.sh --frames 50  # Custom frame count
#
# Results will be saved to results/fitting/ with timestamps

set -e  # Exit on error

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
