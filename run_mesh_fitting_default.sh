#!/bin/bash
# Run mesh fitting on default markerless dataset
# Usage: ./run_mesh_fitting_default.sh [start_frame] [end_frame]

START_FRAME=${1:-0}
END_FRAME=${2:-50}
INTERVAL=${3:-1}
WITH_RENDER=${4:-true}

echo "================================================"
echo "Mesh Fitting: Default Markerless Dataset"
echo "================================================"
echo "Start Frame: $START_FRAME"
echo "End Frame: $END_FRAME"
echo "Interval: $INTERVAL"
echo "With Render: $WITH_RENDER"
echo "================================================"

# Enable headless rendering (required for servers without display)
export PYOPENGL_PLATFORM=egl

python fitter_articulation.py \
  dataset=default_markerless \
  fitter.start_frame=$START_FRAME \
  fitter.end_frame=$END_FRAME \
  fitter.interval=$INTERVAL \
  fitter.with_render=$WITH_RENDER \
  result_folder=results/markerless_fitting/

echo ""
echo "================================================"
echo "Fitting complete! Check outputs/ directory"
echo "================================================"
