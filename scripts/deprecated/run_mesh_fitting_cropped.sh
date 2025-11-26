#!/bin/bash
# Run mesh fitting on cropped frames dataset
# Usage: ./run_mesh_fitting_cropped.sh [data_dir] [output_dir] [max_frames]

DATA_DIR=${1:-"data/100-KO-male-56-20200615_cropped"}
OUTPUT_DIR=${2:-"results/cropped_fitting"}
MAX_FRAMES=${3:-""}

# Enable headless rendering (required for servers without display)
export PYOPENGL_PLATFORM=egl

echo "================================================"
echo "Mesh Fitting: Cropped Frames (Silhouette-based)"
echo "================================================"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"

if [ -z "$MAX_FRAMES" ]; then
    echo "Max Frames: ALL"
    echo "================================================"

    python fit_cropped_frames.py \
      "$DATA_DIR" \
      --output-dir "$OUTPUT_DIR"
else
    echo "Max Frames: $MAX_FRAMES"
    echo "================================================"

    python fit_cropped_frames.py \
      "$DATA_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --max-frames "$MAX_FRAMES"
fi

echo ""
echo "================================================"
echo "Fitting complete! Results saved to: $OUTPUT_DIR"
echo "================================================"
