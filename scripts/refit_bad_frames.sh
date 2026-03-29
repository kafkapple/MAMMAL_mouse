#!/bin/bash
# Re-fit 23 bad frames with accurate optim (6-view + 22 keypoints)
# Single invocation with frame_list to avoid redundant data loading
#
# Usage: CUDA_VISIBLE_DEVICES=4 bash scripts/refit_bad_frames.sh
# Source: FaceLift Neural Texture IoU < 0.7 (cam_003 projection)

set -e
cd "$(dirname "$0")/.."

source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate mammal_stable

OUTPUT_DIR="results/fitting/refit_accurate_23"

echo "=== MAMMAL Refit: 23 bad frames ==="
echo "Config: accurate_6views + with_render=false"
echo "Output: $OUTPUT_DIR"
echo ""

CUDA_VISIBLE_DEVICES=4 python fitter_articulation.py \
    experiment=accurate_6views \
    result_folder="$OUTPUT_DIR" \
    '+fitter.frame_list=[720,1320,1920,2040,2160,2760,3600,5160,5520,5880,6000,6120,6960,7200,8280,8400,9360,9480,9840,10080,10680,10800,11880]' \
    fitter.with_render=false \
    fitter.generate_visualizations=false

echo ""
echo "=== Done ==="
echo "OBJ files:"
ls "$OUTPUT_DIR/obj/" 2>/dev/null | wc -l
