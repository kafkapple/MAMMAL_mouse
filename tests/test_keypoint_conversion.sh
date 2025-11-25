#!/bin/bash
# Test script for keypoint conversion pipeline
#
# This demonstrates the full workflow:
# 1. Manual annotation with keypoint_annotator_v2.py
# 2. Convert JSON to MAMMAL PKL format
# 3. Use with MAMMAL mesh fitter

set -e  # Exit on error

echo "===================================================="
echo "Keypoint Conversion Pipeline Test"
echo "===================================================="

# Configuration
CROPPED_DIR="data/100-KO-male-56-20200615_cropped"
ANNOTATIONS_JSON="data/annotations/keypoints.json"
OUTPUT_PKL="$CROPPED_DIR/keypoints2d_undist/result_view_0.pkl"
NUM_FRAMES=20

echo ""
echo "Step 1: Manual Annotation (Optional)"
echo "----------------------------------------------------"
echo "If you haven't annotated keypoints yet, run:"
echo ""
echo "  python keypoint_annotator_v2.py \\"
echo "    $CROPPED_DIR \\"
echo "    --output $ANNOTATIONS_JSON"
echo ""
echo "Or use SAM annotations (not recommended for mesh fitting):"
echo ""
echo "  python extract_sam_keypoints.py \\"
echo "    $CROPPED_DIR \\"
echo "    --output $ANNOTATIONS_JSON"
echo ""

# Check if annotations exist
if [ ! -f "$ANNOTATIONS_JSON" ]; then
    echo "❌ No annotations found at: $ANNOTATIONS_JSON"
    echo ""
    echo "Please annotate keypoints first using one of the methods above."
    exit 1
fi

echo "✅ Found annotations: $ANNOTATIONS_JSON"
echo ""

echo "Step 2: Convert JSON to MAMMAL PKL format"
echo "----------------------------------------------------"

python convert_keypoints_to_mammal.py \
    --input "$ANNOTATIONS_JSON" \
    --output "$OUTPUT_PKL" \
    --num-frames $NUM_FRAMES \
    --visualize 0 \
    --viz-output "data/annotations/conversion_check.png"

echo ""
echo "Step 3: Verify converted data"
echo "----------------------------------------------------"

python -c "
import pickle
import numpy as np

with open('$OUTPUT_PKL', 'rb') as f:
    data = pickle.load(f)

print(f'Shape: {data.shape}')
print(f'Dtype: {data.dtype}')
print(f'Frame 0 keypoints with confidence > 0:')
frame0 = data[0]
for i, (x, y, c) in enumerate(frame0):
    if c > 0:
        print(f'  Keypoint {i:2d}: ({x:6.1f}, {y:6.1f}) conf={c:.2f}')
"

echo ""
echo "===================================================="
echo "✅ Conversion pipeline test complete!"
echo "===================================================="
echo ""
echo "Next steps:"
echo "  1. Use the PKL file with MAMMAL mesh fitter"
echo "  2. Set up dataset config to point to this data"
echo "  3. Run mesh fitting with fitter_articulation.py"
echo ""
