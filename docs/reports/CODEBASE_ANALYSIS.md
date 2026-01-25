# MAMMAL_mouse Codebase Analysis Report

**Date**: 2025-01-25
**Author**: AI-assisted analysis
**Purpose**: Original vs Current code comparison, issue identification, and refactoring plan

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Original Repository Structure](#2-original-repository-structure)
3. [Current Codebase Structure](#3-current-codebase-structure)
4. [Changes by File/Folder](#4-changes-by-filefolder)
5. [Issues Identified](#5-issues-identified)
6. [Log/Result Folder Analysis](#6-logresult-folder-analysis)
7. [Refactoring Plan](#7-refactoring-plan)

---

## 1. Executive Summary

### Key Findings

| Category | Original | Current | Status |
|----------|----------|---------|--------|
| Total Python files | ~10 | ~85+ | **Large expansion** |
| Configuration | argparse (hardcoded) | Hydra (YAML) | **Major improvement** |
| Code lines (fitter) | ~530 | ~1712 | **3x increase** |
| New modules | - | 6 | Visualization, preprocessing, uvmap |
| Duplicate files | 0 | 3+ | **Needs cleanup** |
| Deprecated code | 0 | 14 files | **Needs review** |

### Overall Assessment
현재 코드베이스는 원본을 대폭 확장하여 다음을 추가:
- Hydra 기반 설정 시스템
- 다양한 실험 설정 (28개 experiment configs)
- 시각화 파이프라인
- 전처리 유틸리티
- UV 맵핑/텍스처 최적화 모듈

그러나 **코드 중복**, **사용되지 않는 파일**, **결과 폴더 분산** 문제가 있음.

---

## 2. Original Repository Structure

```
MAMMAL_mouse/ (Original - anl13/MAMMAL_mouse)
├── fitter_articulation.py     # Main fitting (~530 lines)
├── bodymodel_th.py            # PyTorch body model
├── bodymodel_np.py            # NumPy body model
├── articulation_th.py         # Joint articulation
├── evaluate.py                # Evaluation utils
├── data_seaker_video_new.py   # Data loader
├── visualize_DANNCE.py        # Visualization
├── mouse_22_defs.py           # Keypoint definitions
├── utils.py                   # Utilities
├── run.sh                     # Simple run script
├── requirements.txt
├── README.md
├── mouse_model/               # 3D model assets
│   ├── mouse.pkl
│   ├── keypoint22_mapper.json
│   └── mouse_txt/            # Mesh data (21 files)
├── colormaps/                 # Color maps (4 files)
├── figs/                      # Result images
└── mouse_fitting_result/      # Output directory
```

**특징**:
- argparse로 CLI 설정 (`--start`, `--end`, `--date`)
- 하드코딩된 파라미터 (keypoint_weight, term_weights)
- 단일 run.sh 스크립트

---

## 3. Current Codebase Structure

```
MAMMAL_mouse/ (Current)
├── [CORE - Modified from Original]
│   ├── fitter_articulation.py    # 1712 lines (+Hydra, +configs)
│   ├── bodymodel_th.py           # Unchanged
│   ├── bodymodel_np.py           # Unchanged
│   ├── articulation_th.py        # Modified (+Hydra paths)
│   ├── fit_monocular.py          # NEW: Monocular fitting
│   ├── data_seaker_video_new.py  # Modified
│   └── mouse_22_defs.py          # Unchanged
│
├── [NEW - Configuration System]
│   └── conf/
│       ├── config.yaml           # Main Hydra config
│       ├── experiment/           # 28 experiment configs
│       ├── optim/                # 4 optimization configs
│       ├── frames/               # Frame range configs
│       ├── dataset/              # Dataset configs
│       └── preprocess/           # Preprocessing configs
│
├── [NEW - Modules]
│   ├── utils/                    # Refactored utilities
│   │   ├── __init__.py          # Exports from original utils.py
│   │   └── debug_grid.py        # Debug visualization
│   ├── visualization/            # Visualization pipeline
│   │   ├── mesh_visualizer.py
│   │   ├── video_generator.py
│   │   ├── rerun_exporter.py
│   │   ├── textured_renderer.py
│   │   └── camera_paths.py
│   ├── preprocessing_utils/      # Preprocessing tools
│   │   ├── mask_processing.py
│   │   ├── keypoint_estimation.py
│   │   ├── silhouette_renderer.py
│   │   ├── sam_inference.py
│   │   └── yolo_keypoint_detector.py
│   └── uvmap/                    # UV/Texture optimization
│       ├── uv_pipeline.py
│       ├── uv_renderer.py
│       ├── texture_optimizer.py
│       └── wandb_sweep.py
│
├── [NEW - Scripts]
│   └── scripts/
│       ├── compare_experiments.py
│       ├── evaluate_experiment.py
│       ├── mesh_animation.py
│       ├── deprecated/          # 14 deprecated files
│       ├── tests/               # 15 test scripts
│       ├── utils/               # 11 utility scripts
│       ├── annotators/          # 2 annotation tools
│       └── debug/               # 3 debug scripts
│
├── [NEW - Run Scripts]
│   ├── run_experiment.sh         # Main experiment runner
│   ├── run_all_experiments.sh
│   ├── run_mesh_fitting_default.sh
│   └── run_silhouette_experiments.sh
│
├── [DATA & RESULTS]
│   ├── data/                     # Input data
│   ├── results/                  # Main results (22GB)
│   │   ├── fitting/             # Fitting results
│   │   ├── logs/                # Experiment logs
│   │   └── sweep/               # Sweep results
│   ├── outputs/                  # Debug outputs (18MB)
│   ├── logs/                     # Root logs (36KB)
│   ├── wandb/                    # WandB logs (4.4MB)
│   └── wandb_sweep_results/      # Sweep results (32MB)
│
├── [ASSETS]
│   ├── mouse_model/
│   ├── assets/
│   │   └── colormaps -> ../colormaps (symlink)
│   ├── models/                   # YOLO/SAM models
│   └── exports/                  # Export outputs
│
└── [DOCS]
    └── docs/
        ├── MOC.md               # Documentation entry point
        ├── practical/
        └── reference/
```

---

## 4. Changes by File/Folder

### 4.1 Core Files (Modified from Original)

#### `fitter_articulation.py` (530 → 1712 lines, **+223%**)

| Change | Description |
|--------|-------------|
| Hydra integration | `@hydra.main()` decorator, `DictConfig` config |
| GPU auto-detection | Socket-based hostname detection for GPU selection |
| Configurable weights | `loss_weights`, `keypoint_weights` from YAML |
| Step-specific mask weights | `mask_step0/1/2` for per-step silhouette loss |
| Sparse keypoint support | `sparse_keypoint_indices` for subset fitting |
| Debug grid collector | `DebugGridCollector` for iteration visualization |
| Path resolution | `hydra.utils.to_absolute_path()` for paths |

**Key additions**:
```python
# GPU auto-config (lines 1-17)
_gpu_defaults = {'gpu05': '1', 'bori': '0'}
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id

# Hydra config (lines 42-44)
import hydra
from omegaconf import DictConfig

# Configurable weights (lines 126-151)
self.term_weights = {
    "theta": getattr(lw_cfg, 'theta', 3.0) if lw_cfg else 3.0,
    ...
}
```

#### `articulation_th.py`
- Added `hydra.utils` for path resolution
- Same core functionality

#### `data_seaker_video_new.py`
- Modified for multi-dataset support
- Added Hydra path compatibility

### 4.2 New Modules

#### `conf/` - Configuration System
| Folder | Files | Purpose |
|--------|-------|---------|
| `experiment/` | 28 | Experiment presets (view/keypoint combinations) |
| `optim/` | 4 | Optimization settings (fast, accurate, paper) |
| `frames/` | 4+ | Frame range configs |
| `dataset/` | 3 | Dataset definitions |

#### `utils/` - Refactored Utilities
- **Before**: Single `utils.py` file
- **After**: Package with modular components
  - `__init__.py`: Exports original functions
  - `debug_grid.py`: New debug visualization tools

#### `visualization/` - New Visualization Pipeline
| File | Purpose |
|------|---------|
| `mesh_visualizer.py` | 3D mesh rendering |
| `video_generator.py` | Video creation |
| `rerun_exporter.py` | Rerun.io export |
| `textured_renderer.py` | Textured mesh rendering |
| `camera_paths.py` | Camera trajectory tools |

#### `preprocessing_utils/` - Preprocessing Tools
| File | Purpose |
|------|---------|
| `mask_processing.py` | Silhouette mask processing |
| `keypoint_estimation.py` | Keypoint detection |
| `sam_inference.py` | SAM segmentation |
| `yolo_keypoint_detector.py` | YOLO pose detection |
| `superanimal_detector.py` | SuperAnimal model |

#### `uvmap/` - UV/Texture Optimization
| File | Purpose |
|------|---------|
| `uv_pipeline.py` | UV mapping pipeline |
| `texture_optimizer.py` | Texture optimization |
| `wandb_sweep.py` | Hyperparameter sweeps |

### 4.3 New Scripts

| Category | Files | Purpose |
|----------|-------|---------|
| `scripts/` | 15+ | Main analysis scripts |
| `scripts/deprecated/` | 14 | Deprecated implementations |
| `scripts/tests/` | 15 | Test scripts |
| `scripts/utils/` | 11 | Utility scripts |

---

## 5. Issues Identified

### 5.1 Duplicate Files

| File | Locations | Status |
|------|-----------|--------|
| `data_seaker_video_new.py` | Root, `scripts/analysis/`, `scripts/utils/` | **Different versions** |
| `visualize_DANNCE.py` | (original location moved) | In `scripts/analysis/` |
| `evaluate.py` | `scripts/deprecated/` | Deprecated |

**Recommendation**:
- Keep only `data_seaker_video_new.py` in root
- Delete duplicates in `scripts/*/`

### 5.2 Unused/Deprecated Code

#### `scripts/deprecated/` (14 files, can be deleted)
```
deprecated/
├── evaluate.py              # Replaced by scripts/evaluate_experiment.py
├── fit_cropped_frames.py    # Obsolete workflow
├── fit_silhouette_prototype.py
├── interactive_mesh_viewer.py
├── keypoint_annotator.py    # Replaced by scripts/annotators/
├── run_fitting_legacy.sh
├── run_fitting.sh
├── run_keypoint_annotator.sh
├── run_mesh_fitting_cropped.sh
├── run_sam_annotator.sh
├── run.sh
├── visualize_gt_keypoints.py  # Replaced by visualize_gt_keypoints_hires.py
└── visualize_mesh_sequence.py
```

#### Backup Files
```
utils.py.bak                # 4.4KB - Can be deleted
results/fitting/_backup/    # 1.8GB - Review and archive
```

### 5.3 Import Conflicts

#### `from utils import *` Pattern
**Problem**: Ambiguous imports when both `utils.py` (deleted) and `utils/` package exist.

**Affected files**:
- `fitter_articulation.py:38` - `from utils import *`
- `data_seaker_video_new.py` - `from utils import *`

**Current workaround**: `utils/__init__.py` exports all functions.

**Recommendation**: Use explicit imports:
```python
# Instead of:
from utils import *

# Use:
from utils import pack_images, rodrigues_batch, draw_keypoints
```

### 5.4 Potential Issues

| Issue | Location | Severity |
|-------|----------|----------|
| Hardcoded paths | Various scripts | Medium |
| Unused imports | Some scripts | Low |
| Missing `__init__.py` | `scripts/` subfolders | Low (not packages) |
| WandB config in code | `uvmap/wandb_sweep.py` | Low |

---

## 6. Log/Result Folder Analysis

### Current State (Fragmented)

| Folder | Size | Purpose | Issues |
|--------|------|---------|--------|
| `results/fitting/` | 20GB+ | Main fitting results | **Primary location** |
| `results/logs/` | Varies | Hydra run logs | OK |
| `results/sweep/` | Large | Sweep results | OK |
| `outputs/` | 18MB | Debug outputs | **Should merge into results/** |
| `logs/` (root) | 36KB | Runtime logs | **Should merge into results/logs/** |
| `wandb/` | 4.4MB | WandB runs | OK (auto-managed) |
| `wandb_sweep_results/` | 32MB | Sweep exports | **Should merge into results/sweep/** |
| `results/fitting/_backup/` | 1.8GB | Old results | **Archive and remove** |

### Recommended Unified Structure

```
results/                      # Single results root
├── fitting/                  # Fitting experiment results
│   └── <experiment_name>_<date>/
│       ├── config.yaml       # Experiment config
│       ├── output_results.pkl
│       ├── videos/
│       ├── debug/
│       └── report.html
├── logs/                     # All logs consolidated
│   ├── hydra/               # Hydra run logs
│   └── runtime/             # Runtime logs
├── sweep/                    # Hyperparameter sweeps
├── wandb/                    # WandB runs (if not using cloud)
└── archive/                  # Old/backup results
```

### Migration Plan

1. Move `logs/` → `results/logs/runtime/`
2. Move `outputs/` → `results/debug/`
3. Move `wandb_sweep_results/` → `results/sweep/exports/`
4. Archive `results/fitting/_backup/` → external storage
5. Update all scripts to use unified paths

---

## 7. Refactoring Plan

### 7.1 Proposed Module Structure

```
MAMMAL_mouse/
├── [ORIGINAL - Minimal Changes]
│   ├── fitter_articulation.py   # Keep Hydra integration only
│   ├── bodymodel_th.py          # Unchanged
│   ├── bodymodel_np.py          # Unchanged
│   ├── articulation_th.py       # Minimal changes
│   └── data_seaker_video_new.py # Keep in root
│
├── [NEW MODULE - mammal_ext/]    # Extension module
│   ├── __init__.py
│   ├── config/                   # Config utilities
│   │   ├── __init__.py
│   │   ├── hydra_utils.py       # Hydra helpers
│   │   └── gpu_config.py        # GPU auto-detection
│   ├── fitting/                  # Fitting extensions
│   │   ├── __init__.py
│   │   ├── loss_weights.py      # Configurable loss weights
│   │   ├── sparse_keypoints.py  # Sparse keypoint support
│   │   └── step_configs.py      # Step-specific configs
│   ├── visualization/            # (move from visualization/)
│   ├── preprocessing/            # (move from preprocessing_utils/)
│   └── experiments/              # Experiment utilities
│       ├── __init__.py
│       ├── runner.py
│       └── evaluation.py
│
├── conf/                         # Keep Hydra configs
├── scripts/                      # Keep scripts (cleaned)
├── results/                      # Unified results
└── docs/                         # Documentation
```

### 7.2 Implementation Steps

#### Phase 1: Cleanup (Low Risk)
1. Delete `scripts/deprecated/` folder
2. Delete duplicate `data_seaker_video_new.py` files
3. Delete `utils.py.bak`
4. Archive `results/fitting/_backup/`

#### Phase 2: Result Consolidation (Low Risk)
1. Create unified `results/` structure
2. Migrate scattered log folders
3. Update `.gitignore`

#### Phase 3: Module Extraction (Medium Risk)
1. Create `mammal_ext/` package
2. Extract GPU config to `mammal_ext/config/gpu_config.py`
3. Extract loss weight handling to `mammal_ext/fitting/loss_weights.py`
4. Update imports in `fitter_articulation.py`
5. Unit test each extraction

#### Phase 4: Code Cleanup (Medium Risk)
1. Replace `from utils import *` with explicit imports
2. Add type hints to core functions
3. Document public APIs

### 7.3 Testing Strategy

#### Unit Tests Required
```
tests/
├── test_gpu_config.py        # GPU detection
├── test_loss_weights.py      # Loss weight parsing
├── test_sparse_keypoints.py  # Sparse keypoint indices
├── test_hydra_integration.py # Config loading
└── test_data_loader.py       # Data loading
```

#### Integration Tests
```
tests/integration/
├── test_fitting_pipeline.py  # End-to-end fitting
├── test_visualization.py     # Visualization outputs
└── test_experiment_runner.py # Experiment execution
```

### 7.4 Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| Phase 1 | Very Low | Git history preserved |
| Phase 2 | Low | Create symlinks first |
| Phase 3 | Medium | Unit test each step |
| Phase 4 | Medium | Keep original as fallback |

---

## Appendix A: File Comparison Summary

### Files Unchanged from Original
- `bodymodel_np.py`
- `bodymodel_th.py`
- `mouse_22_defs.py`
- `mouse_model/*` (assets)
- `colormaps/*` (assets)

### Files Significantly Modified
- `fitter_articulation.py` (+223%)
- `articulation_th.py` (Hydra paths)
- `data_seaker_video_new.py` (dataset support)

### Files Completely New
- All `conf/*.yaml` configs
- All `visualization/*.py` files
- All `preprocessing_utils/*.py` files
- All `uvmap/*.py` files
- All `scripts/*.py` files
- `utils/__init__.py`, `utils/debug_grid.py`
- Shell scripts (`run_experiment.sh`, etc.)

---

## Appendix B: Quick Commands

```bash
# Cleanup deprecated files
rm -rf scripts/deprecated/
rm scripts/analysis/data_seaker_video_new.py
rm scripts/utils/data_seaker_video_new.py
rm utils.py.bak

# Archive old results (run from project root)
tar -czvf results_backup_$(date +%Y%m%d).tar.gz results/fitting/_backup/
rm -rf results/fitting/_backup/

# Consolidate logs
mkdir -p results/logs/runtime
mv logs/* results/logs/runtime/
rmdir logs

# Run tests after changes
python -m pytest tests/ -v
```
