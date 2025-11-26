#!/bin/bash
# Run mesh fitting on default markerless dataset
# Usage: ./run_mesh_fitting_default.sh [start_frame] [end_frame] [-- additional_args]
#
# Examples:
#   ./run_mesh_fitting_default.sh 0 10
#   ./run_mesh_fitting_default.sh 0 10 -- --keypoints none
#   ./run_mesh_fitting_default.sh 0 10 -- --input_dir /path/to/data --keypoints none

START_FRAME=${1:-0}
END_FRAME=${2:-50}

# Collect additional arguments after "--"
EXTRA_ARGS=""
shift 2 2>/dev/null
if [[ "$1" == "--" ]]; then
    shift
    EXTRA_ARGS="$@"
fi

echo "================================================"
echo "Mesh Fitting: Multi-View"
echo "================================================"
echo "Start Frame: $START_FRAME"
echo "End Frame: $END_FRAME"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra Args: $EXTRA_ARGS"
fi
echo "================================================"

# Enable headless rendering (required for servers without display)
export PYOPENGL_PLATFORM=egl

python fitter_articulation.py \
  dataset=default_markerless \
  fitter.start_frame=$START_FRAME \
  fitter.end_frame=$END_FRAME \
  fitter.with_render=true \
  $EXTRA_ARGS

echo ""
echo "================================================"
echo "Fitting complete!"
echo "================================================"
