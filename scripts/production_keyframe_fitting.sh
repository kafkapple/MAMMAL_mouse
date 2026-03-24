#!/bin/bash
# Production keyframe fitting: 900 keyframes (interval=4 M5 frames) with accurate config
#
# Usage:
#   bash scripts/production_keyframe_fitting.sh <GPU_ID> <PART>
#   PART: 1 (M5 0-899), 2 (M5 900-1799), 3 (M5 1800-2699), 4 (M5 2700-3599)
#
# Full run requires 4 GPUs × ~13h = ~52h total
# Each part: 225 keyframes × 14min/frame = ~52h/4 = ~13h

set -e
cd "$(dirname "$0")/.."
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate mouse

GPU_ID=${1:?Usage: $0 <GPU_ID> <PART 1-4>}
PART=${2:?Provide PART number 1-4}

# M5 frame ranges per part (interval=4, so every 4th M5 = every 20th video frame)
case $PART in
    1) START_VID=0;    END_VID=4500 ;;   # M5 0-899
    2) START_VID=4500;  END_VID=9000 ;;   # M5 900-1799
    3) START_VID=9000;  END_VID=13500 ;;  # M5 1800-2699
    4) START_VID=13500; END_VID=18000 ;;  # M5 2700-3599
    *) echo "Invalid PART: $PART (use 1-4)"; exit 1 ;;
esac

# interval=20 video frames = 4 M5 frames
OUTPUT_DIR="results/fitting/production_keyframes_part${PART}"

echo "=== Production Keyframe Fitting Part $PART ==="
echo "GPU: $GPU_ID"
echo "Video frames: $START_VID - $END_VID (interval=20)"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"

export CUDA_VISIBLE_DEVICES=$GPU_ID

python fitter_articulation.py \
    experiment=accurate_6views \
    result_folder="$OUTPUT_DIR" \
    fitter.start_frame=$START_VID \
    fitter.end_frame=$END_VID \
    fitter.interval=20 \
    fitter.with_render=false \
    fitter.generate_visualizations=false

echo ""
echo "=== Part $PART Done: $(date) ==="
echo "OBJs: $(ls "$OUTPUT_DIR/obj/"*.obj 2>/dev/null | wc -l)"
