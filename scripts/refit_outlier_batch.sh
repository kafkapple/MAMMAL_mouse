#!/bin/bash
# Batch accurate refit for 152 severe belly-dent outlier frames
# Expected runtime: ~16-18h overnight on single GPU
#
# Usage:
#   bash scripts/refit_outlier_batch.sh <frame_list_file> <output_base_dir>
#
# Example:
#   bash scripts/refit_outlier_batch.sh conf/frames/outlier_severe_152.txt \
#        results/fitting/refit_outliers_152/

set -e

FRAME_LIST="${1:-conf/frames/outlier_severe_152.txt}"
OUT_BASE="${2:-results/fitting/refit_outliers_152/}"

mkdir -p "$OUT_BASE/logs"
mkdir -p "$OUT_BASE/obj"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mammal_blackwell

N_TOTAL=$(wc -l < "$FRAME_LIST")
echo "Refitting $N_TOTAL outlier frames with accurate config (step1=200, step2=50)"
echo "Output: $OUT_BASE"
echo "Start: $(date)"

COUNT=0
while read fid; do
    COUNT=$((COUNT + 1))
    FID_PAD=$(printf "%06d" $fid)
    OUT_OBJ="$OUT_BASE/obj/step_2_frame_${FID_PAD}.obj"
    if [ -f "$OUT_OBJ" ]; then
        echo "[$COUNT/$N_TOTAL] frame $fid already done, skip"
        continue
    fi
    echo "[$COUNT/$N_TOTAL] refitting frame $fid ..."
    FRAME_END=$((fid + 5))
    LOG="$OUT_BASE/logs/frame_${FID_PAD}.log"
    ./run_experiment.sh baseline_6view_keypoint \
        fitter.start_frame=$fid \
        fitter.end_frame=$FRAME_END \
        fitter.interval=5 \
        optim=accurate \
        > "$LOG" 2>&1
    # Find most recent fitting dir (timestamped by Hydra) — use `ls -d` with full path
    LATEST_DIR=$(ls -td results/fitting/markerless_mouse_1_nerf_v012345_kp22_2026* 2>/dev/null | head -1)
    SRC_OBJ="$LATEST_DIR/obj/step_2_frame_${FID_PAD}.obj"
    if [ -n "$LATEST_DIR" ] && [ -f "$SRC_OBJ" ]; then
        cp "$SRC_OBJ" "$OUT_OBJ"
        rm -rf "$LATEST_DIR"  # cleanup to avoid disk filling up
    else
        echo "[WARN] frame $fid: $SRC_OBJ not found, skipping cleanup"
    fi
done < "$FRAME_LIST"

echo "Done: $(date)"
echo "Total refitted: $(ls $OUT_BASE/obj/ | wc -l) / $N_TOTAL"
