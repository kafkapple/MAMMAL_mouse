#!/bin/bash
# Run mesh fitting on default markerless dataset
# Usage: ./run_mesh_fitting_default.sh [experiment] [start_frame] [end_frame] [-- additional_args]
#
# Examples:
#   ./run_mesh_fitting_default.sh                           # default (frame 0-50)
#   ./run_mesh_fitting_default.sh quick_test                # use experiment config
#   ./run_mesh_fitting_default.sh quick_test 0 5            # experiment + frame range
#   ./run_mesh_fitting_default.sh - 0 10                    # no experiment, frame 0-10
#   ./run_mesh_fitting_default.sh - 0 10 -- --keypoints none  # with extra args
#
# Available experiments (conf/experiment/):
#   quick_test  - 5 frames, minimal iterations (fast debugging)
#   (add more as needed)

# Parse arguments
EXPERIMENT=""
START_FRAME=""
END_FRAME=""

# First arg: experiment name or "-" for none
if [[ -n "$1" && "$1" != "--" ]]; then
    if [[ "$1" != "-" ]]; then
        EXPERIMENT="$1"
    fi
    shift
fi

# Second arg: start_frame (if numeric)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    START_FRAME=$1
    shift
fi

# Third arg: end_frame (if numeric)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    END_FRAME=$1
    shift
fi

# Collect additional arguments after "--"
EXTRA_ARGS=""
if [[ "$1" == "--" ]]; then
    shift
    EXTRA_ARGS="$@"
fi

echo "================================================"
echo "Mesh Fitting: Multi-View"
echo "================================================"
if [ -n "$EXPERIMENT" ]; then
    echo "Experiment: $EXPERIMENT"
fi
if [ -n "$START_FRAME" ]; then
    echo "Start Frame: $START_FRAME (CLI override)"
fi
if [ -n "$END_FRAME" ]; then
    echo "End Frame: $END_FRAME (CLI override)"
fi
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra Args: $EXTRA_ARGS"
fi
echo "================================================"

# Enable headless rendering (required for servers without display)
export PYOPENGL_PLATFORM=egl

# Build command
CMD="python fitter_articulation.py"

if [ -n "$EXPERIMENT" ]; then
    # Experiment config handles dataset selection
    CMD="$CMD experiment=$EXPERIMENT"
    # CLI frame range overrides experiment config (only if specified)
    if [ -n "$START_FRAME" ]; then
        CMD="$CMD fitter.start_frame=$START_FRAME"
    fi
    if [ -n "$END_FRAME" ]; then
        CMD="$CMD fitter.end_frame=$END_FRAME"
    fi
else
    # No experiment: use default_markerless with CLI args
    CMD="$CMD dataset=default_markerless"
    CMD="$CMD fitter.start_frame=${START_FRAME:-0}"
    CMD="$CMD fitter.end_frame=${END_FRAME:-50}"
    CMD="$CMD fitter.with_render=true"
fi

if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "================================================"
echo "Fitting complete!"
echo "================================================"
