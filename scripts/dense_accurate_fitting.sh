#!/bin/bash
# Dense accurate fitting for interpolation analysis
# Fits a continuous range of M5 frames with accurate config
#
# Usage:
#   bash scripts/dense_accurate_fitting.sh <GPU_ID> <START_M5> <END_M5>
#   bash scripts/dense_accurate_fitting.sh 6 0 100     # M5 frames 0-99 on GPU 6
#   bash scripts/dense_accurate_fitting.sh 7 100 200   # M5 frames 100-199 on GPU 7

set -e
cd "$(dirname "$0")/.."
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate mouse

GPU_ID=${1:?Usage: $0 <GPU_ID> <START_M5> <END_M5>}
START_M5=${2:?Provide START_M5 index}
END_M5=${3:?Provide END_M5 index}

# M5 to video frame conversion: video_frame = M5 * 5
START_VID=$((START_M5 * 5))
END_VID=$((END_M5 * 5))
N_FRAMES=$((END_M5 - START_M5))

OUTPUT_DIR="results/fitting/dense_accurate_${START_M5}_${END_M5}"

echo "=== Dense Accurate Fitting ==="
echo "GPU: $GPU_ID"
echo "M5 frames: $START_M5 - $END_M5 ($N_FRAMES frames)"
echo "Video frames: $START_VID - $END_VID (interval=5)"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID

python fitter_articulation.py \
    experiment=accurate_6views \
    result_folder="$OUTPUT_DIR" \
    fitter.start_frame=$START_VID \
    fitter.end_frame=$END_VID \
    fitter.interval=5 \
    fitter.with_render=false \
    fitter.generate_visualizations=false

echo ""
echo "=== Done: $(date) ==="
echo "OBJs: $(ls "$OUTPUT_DIR/obj/"*.obj 2>/dev/null | wc -l)/$N_FRAMES"
