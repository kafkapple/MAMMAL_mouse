#!/bin/bash
# Run monocular mesh fitting on single images (RGB + mask)
# Usage: ./run_mesh_fitting_monocular.sh [input_dir] [output_dir] [options]
#
# Input directory should contain:
#   - *_rgb.png: RGB images
#   - *_mask.png: Corresponding binary masks
#
# Examples:
#   ./run_mesh_fitting_monocular.sh /path/to/images results/output
#   ./run_mesh_fitting_monocular.sh /path/to/images results/output --detector superanimal
#   ./run_mesh_fitting_monocular.sh /path/to/images results/output --max-images 10

INPUT_DIR=${1:-"data/test_images"}
OUTPUT_DIR=${2:-"results/monocular_fitting"}
shift 2 2>/dev/null  # Remove first two positional args

# Default options
DETECTOR="geometric"
MAX_IMAGES=""
KEYPOINTS="all"  # all, head, spine, limbs, tail, or comma-separated indices

# Parse additional options
while [[ $# -gt 0 ]]; do
    case $1 in
        --detector)
            DETECTOR="$2"
            shift 2
            ;;
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --keypoints)
            KEYPOINTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "Mesh Fitting: Monocular (Single Image)"
echo "================================================"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Detector: $DETECTOR"
echo "Keypoints: $KEYPOINTS"
if [ -n "$MAX_IMAGES" ]; then
    echo "Max Images: $MAX_IMAGES"
fi
echo "================================================"

# Enable headless rendering (required for servers without display)
export PYOPENGL_PLATFORM=egl

# Build command
CMD="python fit_monocular.py --input_dir \"$INPUT_DIR\" --output_dir \"$OUTPUT_DIR\" --detector $DETECTOR"

if [ -n "$MAX_IMAGES" ]; then
    CMD="$CMD --max_images $MAX_IMAGES"
fi

if [ "$KEYPOINTS" != "all" ]; then
    CMD="$CMD --keypoints $KEYPOINTS"
fi

# Execute
eval $CMD

echo ""
echo "================================================"
echo "Fitting complete! Results saved to: $OUTPUT_DIR"
echo "  - *_mesh.obj: 3D mesh files (Blender compatible)"
echo "  - *_overlay.png: Visualization with mesh+mask+keypoints"
echo "  - *_params.pkl: MAMMAL parameters"
echo "================================================"
