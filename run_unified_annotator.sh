#!/bin/bash
# Launch unified annotator with common configurations
#
# Usage:
#   ./run_unified_annotator.sh [input_dir] [output_dir] [mode]
#
# Examples:
#   ./run_unified_annotator.sh data/frames data/annotations both
#   ./run_unified_annotator.sh data/frames data/annotations keypoint

set -e

# Default arguments
INPUT_DIR=${1:-"data/100-KO-male-56-20200615_cropped"}
OUTPUT_DIR=${2:-"data/annotations"}
MODE=${3:-"both"}

# SAM checkpoint path
SAM_CHECKPOINT="$HOME/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt"

# Server port
PORT=7860

echo "===================================================="
echo "Unified Mouse Annotation Tool"
echo "===================================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Mode:   $MODE"
echo ""

# Check input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check SAM checkpoint if mask mode
if [ "$MODE" == "mask" ] || [ "$MODE" == "both" ]; then
    if [ ! -f "$SAM_CHECKPOINT" ]; then
        echo "⚠️  Warning: SAM checkpoint not found: $SAM_CHECKPOINT"
        echo "   Mask annotation will be disabled."
        echo ""
        echo "To install SAM2:"
        echo "  cd ~/dev"
        echo "  git clone https://github.com/facebookresearch/segment-anything-2.git"
        echo "  cd segment-anything-2"
        echo "  pip install -e ."
        echo "  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O checkpoints/sam2_hiera_large.pt"
        echo ""
        read -p "Continue without SAM? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        SAM_CHECKPOINT=""
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting annotator..."
echo ""
echo "📍 Access via: http://localhost:$PORT"
echo "📍 SSH tunnel: ssh -L $PORT:localhost:$PORT user@server"
echo ""

# Launch annotator
if [ -z "$SAM_CHECKPOINT" ]; then
    # Without SAM
    python unified_annotator.py \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --mode keypoint \
        --port $PORT
else
    # With SAM
    python unified_annotator.py \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --mode "$MODE" \
        --sam-checkpoint "$SAM_CHECKPOINT" \
        --port $PORT
fi
