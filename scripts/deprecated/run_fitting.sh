#!/bin/bash
# Run mesh fitting on cropped frames

# Check conda environment
if [ "$CONDA_DEFAULT_ENV" != "mammal_stable" ]; then
    echo "Error: Please activate mammal_stable environment first:"
    echo "  conda activate mammal_stable"
    exit 1
fi

# Set CUDA and PyTorch library paths
export LD_LIBRARY_PATH=/home/joon/.local/lib/python3.10/site-packages/torch/lib:/usr/local/cuda-12.4/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH updated"

cd /home/joon/dev/MAMMAL_mouse

# Run fitting
python fit_cropped_frames.py \
    data/100-KO-male-56-20200615_cropped \
    --output-dir results/cropped_fitting \
    --max-frames "${1:-2}"

echo ""
echo "Fitting complete! Check results in: results/cropped_fitting/"
