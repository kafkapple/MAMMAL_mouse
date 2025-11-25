#!/bin/bash
# Launch Keypoint Annotation Tool

# Check conda environment
if [ "$CONDA_DEFAULT_ENV" != "mammal_stable" ]; then
    echo "Error: Please activate mammal_stable environment first:"
    echo "  conda activate mammal_stable"
    exit 1
fi

FRAMES_DIR="${1:-data/100-KO-male-56-20200615_cropped}"
OUTPUT_FILE="${2:-data/keypoints_manual.json}"
PORT="${3:-7861}"

echo "========================================="
echo "Keypoint Annotation Tool"
echo "========================================="
echo "Frames Dir: $FRAMES_DIR"
echo "Output:     $OUTPUT_FILE"
echo "Port:       $PORT"
echo ""
echo "Access at: http://localhost:$PORT"
echo "========================================="
echo ""

python keypoint_annotator_v2.py \
    "$FRAMES_DIR" \
    --output "$OUTPUT_FILE" \
    --port "$PORT"
