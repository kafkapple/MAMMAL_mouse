#!/bin/bash
# Run SAM Annotator for video frames
# Usage: bash run_sam_annotator.sh [port]

PORT=${1:-7860}
FRAMES_DIR="/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_frames"
ANNOTATIONS_DIR="${FRAMES_DIR}/annotations"

echo "================================================================================"
echo "Starting SAM Annotator for 100-KO-male-56-20200615"
echo "================================================================================"
echo "Frames directory: ${FRAMES_DIR}"
echo "Annotations output: ${ANNOTATIONS_DIR}"
echo "Server port: ${PORT}"
echo "================================================================================"
echo ""
echo "Access the web interface:"
echo "  Local: http://localhost:${PORT}"
echo "  SSH tunnel: ssh -L ${PORT}:localhost:${PORT} joon@server"
echo ""
echo "================================================================================"
echo ""

# Check SAM checkpoint
SAM_CHECKPOINT="${HOME}/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Warning: SAM checkpoint not found at ${SAM_CHECKPOINT}"
    echo "Checking for alternative checkpoints..."

    # Try other checkpoint names
    for ckpt in "${HOME}/dev/segment-anything-2/checkpoints/"*.pt; do
        if [ -f "$ckpt" ]; then
            echo "Found checkpoint: $ckpt"
            SAM_CHECKPOINT="$ckpt"
            break
        fi
    done
fi

echo "Using SAM checkpoint: ${SAM_CHECKPOINT}"
echo ""

# Run SAM annotator using simple CLI (avoids Hydra conflicts)
cd /home/joon/dev/mouse-super-resolution

# Check if we're in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: Not in conda environment. Trying to activate mammal_stable..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate mammal_stable
fi

# Use the simple CLI wrapper
conda run -n mammal_stable python sam_annotator/cli.py \
    --input "${FRAMES_DIR}" \
    --output "${ANNOTATIONS_DIR}" \
    --pattern "*.png" \
    --checkpoint "${SAM_CHECKPOINT}" \
    --device cuda \
    --host 0.0.0.0 \
    --port ${PORT}
