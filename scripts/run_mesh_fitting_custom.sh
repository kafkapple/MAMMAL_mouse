#!/bin/bash
# Run mesh fitting on custom dataset with flexible configuration
# Usage: ./run_mesh_fitting_custom.sh <dataset_config> <data_dir> [start_frame] [end_frame]

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset_config> <data_dir> [start_frame] [end_frame] [with_render]"
    echo ""
    echo "Example:"
    echo "  $0 cropped /path/to/data 0 100 false"
    echo ""
    echo "Available dataset configs:"
    echo "  - default_markerless"
    echo "  - cropped"
    echo "  - upsampled"
    echo "  - shank3"
    echo "  - custom"
    exit 1
fi

DATASET_CONFIG=$1
DATA_DIR=$2
START_FRAME=${3:-0}
END_FRAME=${4:-10}
WITH_RENDER=${5:-false}

echo "================================================"
echo "Mesh Fitting: Custom Configuration"
echo "================================================"
echo "Dataset Config: $DATASET_CONFIG"
echo "Data Directory: $DATA_DIR"
echo "Start Frame: $START_FRAME"
echo "End Frame: $END_FRAME"
echo "With Render: $WITH_RENDER"
echo "================================================"

# Enable headless rendering (required for servers without display)
export PYOPENGL_PLATFORM=egl

python fitter_articulation.py \
  dataset=$DATASET_CONFIG \
  data.data_dir="$DATA_DIR" \
  fitter.start_frame=$START_FRAME \
  fitter.end_frame=$END_FRAME \
  fitter.with_render=$WITH_RENDER \
  result_folder=results/custom_fitting/

echo ""
echo "================================================"
echo "Fitting complete! Check outputs/ directory"
echo "================================================"
