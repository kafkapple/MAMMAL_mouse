# PyTorch3D Installation Guide

## Problem

PyTorch3D compilation failing with symbol errors or CUDA issues.

## Quick Solution Options

### Option 1: Use Conda (Easiest, Recommended)

```bash
# Activate your environment
conda activate mammal_stable

# Uninstall existing pytorch3d
pip uninstall pytorch3d -y

# Install from conda-forge
conda install -c conda-forge pytorch3d -y
```

**Pros:**
- Pre-compiled binary
- Fast (no compilation)
- Usually works

**Cons:**
- May not have exact version match
- Might have minor compatibility issues

### Option 2: Manual Compilation with CUB

If conda doesn't work, compile from source:

```bash
# 1. Download NVIDIA CUB
cd /tmp
wget https://github.com/NVIDIA/cub/archive/1.17.2.tar.gz -O cub-1.17.2.tar.gz
tar -xzf cub-1.17.2.tar.gz

# 2. Set environment variables
export CUB_HOME=/tmp/cub-1.17.2
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. Activate environment
conda activate mammal_stable

# 4. Uninstall existing
pip uninstall pytorch3d -y

# 5. Install from source
MAX_JOBS=4 pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Option 3: Disable PyTorch3D Rendering (Workaround)

If installation fails completely, you can still run mesh fitting without PyTorch3D rendering:

**Modify `fit_cropped_frames.py`:**

Add this at the top:
```python
USE_PYTORCH3D = False  # Set to True when pytorch3d is installed
```

Then wrap PyTorch3D imports:
```python
if USE_PYTORCH3D:
    from pytorch3d.renderer import (
        OrthographicCameras,
        # ... other imports
    )
    from preprocessing_utils.silhouette_renderer import (
        SilhouetteRenderer, SilhouetteLoss, visualize_silhouette_comparison
    )
```

**Trade-off:** You'll lose silhouette rendering but core fitting will still work with keypoints.

---

## Verification

After installation, verify:

```bash
conda activate mammal_stable
python -c "
import torch
import pytorch3d
from pytorch3d import _C
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch3D: {pytorch3d.__version__}')
print('Success!')
"
```

---

## Run Mesh Fitting

After successful installation:

```bash
# Test with cropped frames
./run_mesh_fitting_cropped.sh \
  data/100-KO-male-56-20200615_cropped \
  results/test \
  3
```

---

## Still Having Issues?

### Check CUDA Installation

```bash
nvcc --version
# Should show CUDA 11.8
```

If not installed:
```bash
# Install CUDA 11.8
# Follow: https://developer.nvidia.com/cuda-11-8-0-download-archive
```

### Check GCC Version

```bash
gcc --version
# Should be GCC 7.x - 11.x
```

### System Requirements

- NVIDIA GPU with CUDA support
- CUDA 11.8 installed
- GCC 7.x - 11.x
- At least 8GB RAM
- ~5GB free disk space

---

## Alternative: Use Docker

If native installation fails, use Docker:

```bash
# Use official PyTorch container with PyTorch3D
docker run -it --gpus all pytorch/pytorch:2.0.0-cuda11.8-cudnn8-devel bash

# Inside container
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

---

**Last Updated:** 2025-11-17
