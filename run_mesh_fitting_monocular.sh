#!/bin/bash
# Run monocular mesh fitting on single images (RGB + mask)
# Usage: ./run_mesh_fitting_monocular.sh [input_dir] [output_dir] [max_images] [-- additional_args]
#
# Input directory should contain:
#   - *_rgb.png or *_cropped.png: RGB images
#   - *_mask.png: Corresponding binary masks
#
# Examples:
#   ./run_mesh_fitting_monocular.sh data/frames/ results/output
#   ./run_mesh_fitting_monocular.sh data/frames/ results/output 5           # first 5 images
#   ./run_mesh_fitting_monocular.sh data/frames/ results/output - -- --keypoints none
#   ./run_mesh_fitting_monocular.sh data/100-KO-male-56-20200615_4x/cropped/ results/shank3_4x/ 5 -- --keypoints none

# Parse arguments
INPUT_DIR=${1:-"data/test_images"}
OUTPUT_DIR=${2:-"results/monocular_fitting"}
MAX_IMAGES=""

shift 2 2>/dev/null

# Third arg: max_images (if numeric or "-" for none)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_IMAGES=$1
    shift
elif [[ "$1" == "-" ]]; then
    shift
fi

# Collect additional arguments after "--"
EXTRA_ARGS=""
if [[ "$1" == "--" ]]; then
    shift
    EXTRA_ARGS="$@"
fi

echo "================================================"
echo "Mesh Fitting: Monocular (Single Image)"
echo "================================================"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
if [ -n "$MAX_IMAGES" ]; then
    echo "Max Images: $MAX_IMAGES"
fi
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra Args: $EXTRA_ARGS"
fi
echo "================================================"

# Enable headless rendering (required for servers without display)
export PYOPENGL_PLATFORM=egl

# Build command
CMD="python fit_monocular.py"
CMD="$CMD --input_dir \"$INPUT_DIR\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""

if [ -n "$MAX_IMAGES" ]; then
    CMD="$CMD --max_images $MAX_IMAGES"
fi

if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "================================================"
echo "Fitting complete! Results saved to: $OUTPUT_DIR"
echo "  - *_mesh.obj: 3D mesh files (Blender compatible)"
echo "  - *_overlay.png: Visualization with mesh+mask+keypoints"
echo "  - *_params.pkl: MAMMAL parameters"
echo "================================================"
