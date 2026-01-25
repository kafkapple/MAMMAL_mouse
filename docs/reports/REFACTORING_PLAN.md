# MAMMAL_mouse Refactoring Plan

**Date**: 2025-01-25
**Status**: Draft
**Risk Level**: Low-Medium

---

## 1. Goals

1. **원본 코드 최소 수정**: 원본 MAMMAL 코드의 핵심 로직은 유지
2. **모듈화**: 추가된 기능들을 별도 패키지(`mammal_ext/`)로 분리
3. **결과 폴더 단일화**: 분산된 로그/결과 폴더 통합
4. **코드 정리**: 중복/미사용 코드 제거

---

## 2. Current Issues Summary

### 2.1 Duplicate Files (3 files)
| File | Count | Action |
|------|-------|--------|
| `data_seaker_video_new.py` | 3 | Keep root only |
| `visualize_DANNCE.py` | 1 (moved) | Already in scripts/analysis/ |

### 2.2 Unused Imports (25+ imports)
| File | Unused Count | Critical |
|------|--------------|----------|
| `fitter_articulation.py` | 18+ | 1 duplicate |
| `articulation_th.py` | 5 | No |
| `fit_monocular.py` | 3 | No |
| `data_seaker_video_new.py` | 1 | No |

### 2.3 Deprecated Scripts (14 files)
Location: `scripts/deprecated/`
- All have modern replacements
- Can be safely deleted

### 2.4 Result Folders (Fragmented)
| Folder | Size | Purpose |
|--------|------|---------|
| `results/fitting/` | 21GB | Main results |
| `results/logs/` | 2MB | Hydra logs |
| `outputs/` | 18MB | Debug images |
| `logs/` (root) | 36KB | Runtime logs |
| `wandb/` | 4.4MB | WandB |
| `wandb_sweep_results/` | 32MB | Sweep exports |
| `results/fitting/_backup/` | 1.8GB | Old backups |

---

## 3. Phase 1: Cleanup (Risk: Very Low)

### 3.1 Delete Deprecated Files

```bash
# Step 1: Verify no active references
grep -r "scripts/deprecated" /home/joon/dev/MAMMAL_mouse/*.py
grep -r "scripts/deprecated" /home/joon/dev/MAMMAL_mouse/*.sh

# Step 2: Delete deprecated folder
rm -rf scripts/deprecated/

# Step 3: Delete duplicate data_seaker files
rm scripts/analysis/data_seaker_video_new.py
rm scripts/utils/data_seaker_video_new.py

# Step 4: Delete backup file
rm utils.py.bak
```

### 3.2 Archive Old Results

```bash
# Archive backup folder (1.8GB)
cd /home/joon/dev/MAMMAL_mouse
tar -czvf archive/results_backup_20250125.tar.gz results/fitting/_backup/
rm -rf results/fitting/_backup/

# Create archive directory
mkdir -p archive/
```

### 3.3 Unit Test

```bash
# Verify main functionality still works
./run_experiment.sh baseline_6view_keypoint --debug
python -c "from utils import pack_images, rodrigues_batch; print('OK')"
```

---

## 4. Phase 2: Result Consolidation (Risk: Low)

### 4.1 Target Structure

```
results/                      # Unified root
├── fitting/                  # Experiment results (existing)
│   └── <exp_name>_<date>/
├── logs/                     # All logs
│   ├── hydra/               # Hydra run logs (existing, from results/logs/)
│   └── runtime/             # Runtime logs (from root logs/)
├── sweep/                    # Sweep results (existing)
├── debug/                    # Debug outputs (from outputs/)
├── visualizations/           # Generated videos/images
└── wandb/                    # WandB (existing, from results/wandb/)
```

### 4.2 Migration Commands

```bash
# Step 1: Create new structure
mkdir -p results/logs/runtime
mkdir -p results/debug

# Step 2: Move root logs
mv logs/* results/logs/runtime/ 2>/dev/null || true
rmdir logs 2>/dev/null || true

# Step 3: Move outputs to debug
mv outputs/* results/debug/ 2>/dev/null || true
rmdir outputs 2>/dev/null || true

# Step 4: Move wandb_sweep_results
mv wandb_sweep_results/* results/sweep/ 2>/dev/null || true
rmdir wandb_sweep_results 2>/dev/null || true

# Step 5: Move root wandb if not using cloud
# (Skip if using WandB cloud)
# mv wandb/* results/wandb/
```

### 4.3 Update .gitignore

```gitignore
# Results (add if not present)
results/
archive/

# Remove old entries
# logs/       # Now consolidated
# outputs/    # Now consolidated
# wandb_sweep_results/  # Now consolidated
```

### 4.4 Update Config Paths

**File**: `conf/config.yaml`
```yaml
# Update result_folder if needed
result_folder: results/fitting/

hydra:
  run:
    dir: results/logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

---

## 5. Phase 3: Module Extraction (Risk: Medium)

### 5.1 New Package Structure

```
mammal_ext/                   # Extension package
├── __init__.py
├── config/                   # Configuration utilities
│   ├── __init__.py
│   ├── gpu.py               # GPU auto-detection
│   └── loss_weights.py      # Loss weight parsing
├── fitting/                  # Fitting extensions
│   ├── __init__.py
│   ├── sparse_keypoints.py  # Sparse keypoint support
│   └── debug_viz.py         # Debug visualization
├── visualization/            # Move from visualization/
│   ├── __init__.py
│   ├── mesh_visualizer.py
│   ├── video_generator.py
│   └── ...
└── preprocessing/            # Move from preprocessing_utils/
    ├── __init__.py
    └── ...
```

### 5.2 Step-by-Step Extraction

#### Step 3.1: Create Package

```bash
mkdir -p mammal_ext/config
mkdir -p mammal_ext/fitting

touch mammal_ext/__init__.py
touch mammal_ext/config/__init__.py
touch mammal_ext/fitting/__init__.py
```

#### Step 3.2: Extract GPU Config

**Create**: `mammal_ext/config/gpu.py`
```python
"""GPU configuration utilities."""

import os
import socket

# Server-specific GPU defaults
GPU_DEFAULTS = {
    'gpu05': '1',   # gpu05: use GPU 1
    'bori': '0',    # bori: use GPU 0 (only 1 GPU)
}

def get_default_gpu() -> str:
    """Get default GPU based on hostname."""
    hostname = socket.gethostname().split('.')[0]
    return GPU_DEFAULTS.get(hostname, '0')

def configure_gpu(gpu_id: str = None) -> str:
    """Configure GPU environment variables.

    Args:
        gpu_id: Explicit GPU ID. If None, uses GPU_ID env or hostname default.

    Returns:
        The configured GPU ID.
    """
    if gpu_id is None:
        gpu_id = os.environ.get('GPU_ID',
                                os.environ.get('CUDA_VISIBLE_DEVICES',
                                               get_default_gpu()))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['EGL_DEVICE_ID'] = gpu_id
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['DISPLAY'] = ''  # Disable X11 for headless rendering

    return gpu_id
```

**Update**: `fitter_articulation.py` (lines 1-17)
```python
# Replace hardcoded GPU config with:
from mammal_ext.config.gpu import configure_gpu
configure_gpu()

import numpy as np
# ... rest of imports
```

#### Step 3.3: Extract Loss Weights

**Create**: `mammal_ext/config/loss_weights.py`
```python
"""Loss weight configuration utilities."""

from typing import Dict, Any, Optional
from omegaconf import DictConfig

# Default loss weights (MAMMAL paper values)
DEFAULT_LOSS_WEIGHTS = {
    "theta": 3.0,
    "3d": 2.5,
    "2d": 0.2,
    "bone": 0.5,
    "scale": 0.5,
    "mask": 10.0,
    "chest_deformer": 0.1,
    "stretch": 1.0,
    "temp": 0.25,
    "temp_d": 0.2,
}

def get_loss_weights(cfg: Optional[DictConfig] = None) -> Dict[str, float]:
    """Get loss weights from config or defaults.

    Args:
        cfg: Hydra config with loss_weights section.

    Returns:
        Dictionary of loss weights.
    """
    weights = DEFAULT_LOSS_WEIGHTS.copy()

    if cfg is None:
        return weights

    lw_cfg = getattr(cfg, 'loss_weights', None)
    if lw_cfg is None:
        return weights

    for key in weights:
        if hasattr(lw_cfg, key):
            weights[key] = getattr(lw_cfg, key)

    return weights
```

#### Step 3.4: Unit Tests

**Create**: `tests/test_mammal_ext.py`
```python
"""Tests for mammal_ext package."""

import pytest
import os

def test_gpu_config():
    """Test GPU configuration."""
    from mammal_ext.config.gpu import configure_gpu, get_default_gpu

    # Test default detection
    default = get_default_gpu()
    assert isinstance(default, str)

    # Test configuration
    gpu_id = configure_gpu('0')
    assert gpu_id == '0'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'

def test_loss_weights():
    """Test loss weight configuration."""
    from mammal_ext.config.loss_weights import get_loss_weights, DEFAULT_LOSS_WEIGHTS

    # Test defaults
    weights = get_loss_weights()
    assert weights == DEFAULT_LOSS_WEIGHTS

    # Test with config
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'loss_weights': {
            'theta': 5.0,
            '2d': 0.5,
        }
    })
    weights = get_loss_weights(cfg)
    assert weights['theta'] == 5.0
    assert weights['2d'] == 0.5
    assert weights['bone'] == 0.5  # Default preserved
```

### 5.3 Move Existing Modules

```bash
# Step 1: Move visualization module
mv visualization mammal_ext/

# Step 2: Update imports in scripts
sed -i 's/from visualization/from mammal_ext.visualization/g' scripts/*.py

# Step 3: Move preprocessing module
mv preprocessing_utils mammal_ext/preprocessing

# Step 4: Update imports
sed -i 's/from preprocessing_utils/from mammal_ext.preprocessing/g' scripts/*.py
```

---

## 6. Phase 4: Import Cleanup (Risk: Low)

### 6.1 Fix Duplicate Import

**File**: `fitter_articulation.py`
```python
# Line 43: Keep this one
from omegaconf import DictConfig

# Line 68: REMOVE this duplicate
# from omegaconf import DictConfig  # DELETE
```

### 6.2 Remove Unused Imports

**File**: `fitter_articulation.py` - Remove:
```python
# Remove these unused imports:
# import torch.nn as nn
# import torch.functional as F
# from bodymodel_th import BodyModelTorch
# from scipy.spatial.transform import Rotation
# from torch.utils.tensorboard import SummaryWriter

# Remove unused PyTorch3D imports:
# OrthographicCameras, DirectionalLights, Materials,
# HardFlatShader, HardGouraudShader, HardPhongShader,
# TexturesUV, TexturesVertex, AmbientLights, PointLights
```

**File**: `articulation_th.py` - Remove:
```python
# from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
# from scipy.spatial.transform import Rotation
# from time import time
```

### 6.3 Replace Wildcard Imports

**Before**:
```python
from utils import *
```

**After**:
```python
from utils import (
    pack_images, rodrigues_batch, Rmat2axis,
    undist_points_cv2, draw_keypoints,
    colormap, bones, bone_color_index, g_colors, joint_color_index,
    DebugGridCollector, compress_existing_debug_folder
)
```

---

## 7. Implementation Checklist

### Phase 1: Cleanup ✅ (Completed 2025-01-25)
- [x] Delete `scripts/deprecated/` (14 files)
- [x] Delete duplicate `data_seaker_video_new.py` files (2 files)
- [x] Delete `utils.py.bak`
- [x] Update `.gitignore` with archive/
- Commit: `764f9f2`

### Phase 2: Result Consolidation ✅ (Completed 2025-01-25)
- [x] Create unified results structure
- [x] Migrate `logs/` → `results/logs/runtime/`
- [x] Migrate `outputs/` → `results/debug/`
- [x] Migrate `wandb_sweep_results/` → `results/sweep/`
- [x] Remove `uvmap_experiments/` (empty)
- [x] Update `conf/config.yaml` hydra paths
- Commit: `326777c`

### Phase 3: Module Extraction ✅ (Completed 2025-01-25)
- [x] Create `mammal_ext/` package
- [x] Extract GPU config → `mammal_ext/config/gpu.py`
- [x] Extract loss weight config → `mammal_ext/config/loss_weights.py`
- [x] Extract keypoint weight config → `mammal_ext/config/keypoint_weights.py`
- [x] Write unit tests → `tests/test_mammal_ext.py`
- [x] Update `fitter_articulation.py` imports
- Commit: `ed4af14`

### Phase 4: Import Cleanup ✅ (Completed 2025-01-25)
- [x] Remove duplicate DictConfig import
- [x] Remove 18+ unused imports (fitter_articulation.py)
- [x] Remove 4 unused imports (articulation_th.py)
- [x] Replace `from utils import *` with explicit imports
- Commit: `72fd3b7`

### Deferred (Optional Future Work)
- [ ] Move `visualization/` to `mammal_ext/visualization/`
- [ ] Move `preprocessing_utils/` to `mammal_ext/preprocessing/`
- [ ] Archive `results/fitting/_backup/` to external storage

---

## 8. Rollback Plan

각 Phase 완료 후 Git commit을 생성하여 롤백 가능하도록 함:

```bash
# Phase 1 완료 후
git add -A && git commit -m "chore: cleanup deprecated files and duplicates"

# Phase 2 완료 후
git add -A && git commit -m "refactor: consolidate result folders"

# Phase 3 완료 후
git add -A && git commit -m "refactor: extract mammal_ext module"

# Phase 4 완료 후
git add -A && git commit -m "refactor: cleanup imports"
```

문제 발생 시:
```bash
git revert HEAD  # 마지막 commit 롤백
```

---

## 9. Timeline Estimate

| Phase | Description | Complexity |
|-------|-------------|------------|
| Phase 1 | Cleanup | Simple (file deletion) |
| Phase 2 | Result consolidation | Simple (folder moves + config update) |
| Phase 3 | Module extraction | Medium (code refactoring + testing) |
| Phase 4 | Import cleanup | Simple (import edits) |

---

## Appendix: Quick Reference Commands

```bash
# Full cleanup (Phase 1)
rm -rf scripts/deprecated/
rm scripts/analysis/data_seaker_video_new.py scripts/utils/data_seaker_video_new.py
rm utils.py.bak

# Verify no breakage
./run_experiment.sh baseline_6view_keypoint --debug

# Consolidate results (Phase 2)
mkdir -p results/logs/runtime results/debug
mv logs/* results/logs/runtime/ 2>/dev/null; rmdir logs 2>/dev/null
mv outputs/* results/debug/ 2>/dev/null; rmdir outputs 2>/dev/null

# Run tests
python -m pytest tests/ -v
```
