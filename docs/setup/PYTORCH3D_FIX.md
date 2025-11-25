# PyTorch3D Compatibility Fix

## Problem

When running mesh fitting scripts, you may encounter this error:

```
ImportError: /path/to/pytorch3d/_C.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK3c105Error4whatEv
```

## Root Cause

PyTorch3D binary was compiled with a different PyTorch version than currently installed. This creates ABI (Application Binary Interface) incompatibility.

**Your environment:**
- PyTorch: 2.0.0+cu118
- PyTorch3D: 0.7.8 (precompiled binary, incompatible)

**Solution:** Reinstall PyTorch3D 0.7.5 compiled from source for your specific PyTorch version.

---

## Quick Fix (Recommended)

Run the provided fix script:

```bash
cd /home/joon/dev/MAMMAL_mouse
./fix_pytorch3d.sh
```

This will:
1. Uninstall existing PyTorch3D
2. Install PyTorch3D 0.7.5 from source (5-10 minutes)
3. Verify installation

---

## Manual Fix (Alternative)

If the script doesn't work, follow these steps:

### Step 1: Activate Environment

```bash
conda activate mammal_stable
```

### Step 2: Uninstall Existing PyTorch3D

```bash
pip uninstall pytorch3d -y
```

### Step 3: Install Build Dependencies

```bash
# Install required packages
pip install fvcore iopath

# Ensure you have build tools
conda install gcc_linux-64 gxx_linux-64 -y
```

### Step 4: Install PyTorch3D from Source

```bash
# Install specific version compatible with PyTorch 2.0.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

**Note:** This will compile C++/CUDA extensions and takes 5-10 minutes.

### Step 5: Verify Installation

```bash
python -c "
import torch
import pytorch3d
from pytorch3d import _C
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch3D: {pytorch3d.__version__}')
print('PyTorch3D C++ extensions loaded successfully!')
"
```

Expected output:
```
PyTorch: 2.0.0+cu118
PyTorch3D: 0.7.5
PyTorch3D C++ extensions loaded successfully!
```

---

## Troubleshooting

### Issue 1: Compilation Fails with CUDA Errors

**Error:**
```
error: command '/usr/local/cuda/bin/nvcc' failed
```

**Solution:**
```bash
# Check CUDA version
nvcc --version

# Ensure CUDA 11.8 is available
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Retry installation
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Issue 2: Compilation Fails with GCC Errors

**Error:**
```
error: no matching function for call to 'at::Tensor::item()'
```

**Solution:**
```bash
# Use conda's GCC
conda install gcc_linux-64 gxx_linux-64 -y

# Retry installation
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Issue 3: Out of Memory During Compilation

**Error:**
```
c++: internal compiler error: Killed
```

**Solution:**
```bash
# Reduce parallel compilation jobs
export MAX_JOBS=2
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Issue 4: Still Get Symbol Errors After Reinstall

**Error:**
```
undefined symbol: _ZNK3c105Error4whatEv
```

**Solution:**
```bash
# Clear pip cache
pip cache purge

# Completely remove pytorch3d
pip uninstall pytorch3d -y
rm -rf ~/miniconda3/envs/mammal_stable/lib/python3.10/site-packages/pytorch3d*

# Fresh install
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

---

## Alternative: Use Pre-built Wheels (Faster)

If compilation fails, you can try pre-built wheels:

### For PyTorch 2.0.0 + CUDA 11.8

```bash
# Uninstall existing
pip uninstall pytorch3d -y

# Install from conda-forge (may not have exact version)
conda install -c conda-forge pytorch3d -y

# OR try fvcore wheels (if available)
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html
```

**Note:** Pre-built wheels may not be available for all PyTorch/CUDA combinations.

---

## Verification After Fix

Test that mesh fitting works:

```bash
# Quick test (3 frames)
./run_quick_test.sh cropped

# Full test
./run_mesh_fitting_cropped.sh \
  data/100-KO-male-56-20200615_cropped \
  results/test \
  5
```

Expected output:
```
Found 20 cropped frames
Loading mouse body model...
Fitting frames: 100%|████████████| 5/5
Processing complete!
```

---

## Why This Happens

PyTorch3D includes compiled C++/CUDA extensions that are tightly coupled to PyTorch's C++ ABI. When you install PyTorch3D from pip or conda, you get a pre-compiled binary that may not match your exact PyTorch version.

**Solutions:**
1. **Compile from source** (recommended) - Ensures perfect compatibility
2. **Use exact matching versions** - Install PyTorch3D wheel for your PyTorch version
3. **Downgrade PyTorch** - Match the PyTorch3D binary (not recommended)

---

## Prevention for Future

When setting up new environments:

```bash
# Install PyTorch first
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Then install PyTorch3D from source
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

# NOT from pip/conda (may be incompatible)
# pip install pytorch3d  # Don't do this
```

---

## Reference

- PyTorch3D GitHub: https://github.com/facebookresearch/pytorch3d
- Installation Guide: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- Known Issues: https://github.com/facebookresearch/pytorch3d/issues

---

**Last Updated:** 2025-11-17
