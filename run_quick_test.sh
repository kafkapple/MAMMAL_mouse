#!/bin/bash
# Quick test run for mesh fitting (first 3 frames, no rendering)
# Usage: ./run_quick_test.sh [dataset_type]

DATASET_TYPE=${1:-"default_markerless"}

echo "================================================"
echo "Quick Test: Mesh Fitting"
echo "================================================"
echo "Dataset: $DATASET_TYPE"
echo "Frames: 0-2 (3 frames total)"
echo "Rendering: Disabled"
echo "================================================"

case $DATASET_TYPE in
    "default_markerless")
        python fitter_articulation.py \
          dataset=default_markerless \
          fitter.start_frame=0 \
          fitter.end_frame=2 \
          fitter.with_render=false
        ;;

    "cropped")
        python fit_cropped_frames.py \
          data/100-KO-male-56-20200615_cropped \
          --output-dir results/quick_test_cropped \
          --max-frames 3
        ;;

    *)
        echo "Unknown dataset type: $DATASET_TYPE"
        echo "Available options: default_markerless, cropped"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Quick test complete!"
echo "================================================"
