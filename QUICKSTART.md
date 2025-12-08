# Quick Start

## Prerequisites
```bash
conda activate mammal_stable
```

## Run Single Experiment
```bash
./run_experiment.sh <config_name> [--debug]
```

## All Experiment Commands

### Baseline
```bash
./run_experiment.sh baseline_6view_keypoint        # 6-view + 22 keypoints (paper reference)
```

### Keypoint Ablation (6-view fixed)
```bash
./run_experiment.sh baseline_6view_keypoint        # 22 keypoints (full)
./run_experiment.sh sparse_9kp_dlc                 # 9 keypoints (DeepLabCut style)
./run_experiment.sh sparse_7kp_mars                # 7 keypoints (MARS style)
./run_experiment.sh sparse_5kp_minimal             # 5 keypoints (nose, ears, neck, tail)
./run_experiment.sh sixview_sparse_keypoint        # 3 keypoints (nose, neck, tail)
./run_experiment.sh sixview_no_keypoint            # 0 keypoints (silhouette only)
```

### Viewpoint Ablation (sparse 3 keypoints fixed)
```bash
./run_experiment.sh sixview_sparse_keypoint        # 6 views
./run_experiment.sh sparse_5view                   # 5 views
./run_experiment.sh sparse_4view                   # 4 views
./run_experiment.sh sparse_3view                   # 3 views (diagonal: 0,2,4)
./run_experiment.sh sparse_2view                   # 2 views (opposite: 0,3)
```

### Viewpoint Ablation (22 keypoints fixed)
```bash
./run_experiment.sh views_6                        # 6 views
./run_experiment.sh views_5                        # 5 views
./run_experiment.sh views_4                        # 4 views
./run_experiment.sh views_3_diagonal               # 3 views (0,2,4)
./run_experiment.sh views_3_consecutive            # 3 views (0,1,2)
./run_experiment.sh views_2_opposite               # 2 views (0,3)
./run_experiment.sh views_1_single                 # 1 view (monocular)
```

### Silhouette Only (no keypoints)
```bash
./run_experiment.sh silhouette_only_6views
./run_experiment.sh silhouette_only_4views
./run_experiment.sh silhouette_only_3views
./run_experiment.sh silhouette_only_1view
```

### Utility
```bash
./run_experiment.sh quick_test                     # Fast debug (5 frames, minimal iters)
./run_experiment.sh accurate_6views                # High quality (more iterations)
```

## Batch Run All Ablations

### Keypoint Ablation (6-view, 22→9→7→5→3→0 keypoints)
```bash
cd /home/joon/dev/MAMMAL_mouse && \
./run_experiment.sh baseline_6view_keypoint && \
./run_experiment.sh sparse_9kp_dlc && \
./run_experiment.sh sparse_7kp_mars && \
./run_experiment.sh sparse_5kp_minimal && \
./run_experiment.sh sixview_sparse_keypoint && \
./run_experiment.sh sixview_no_keypoint
```

### Viewpoint Ablation (sparse 3 keypoints)
```bash
cd /home/joon/dev/MAMMAL_mouse && \
./run_experiment.sh sixview_sparse_keypoint && \
./run_experiment.sh sparse_5view && \
./run_experiment.sh sparse_4view && \
./run_experiment.sh sparse_3view && \
./run_experiment.sh sparse_2view
```

### Loop Style (alternative)
```bash
# Keypoint ablation
for exp in baseline_6view_keypoint sparse_9kp_dlc sparse_7kp_mars sparse_5kp_minimal sixview_sparse_keypoint sixview_no_keypoint; do
    ./run_experiment.sh $exp
done

# Viewpoint ablation
for exp in sixview_sparse_keypoint sparse_5view sparse_4view sparse_3view sparse_2view; do
    ./run_experiment.sh $exp
done
```

## Visualize Results
```bash
# Compare experiments
python scripts/compare_experiments.py "results/fitting/*sparse3*" -o comparison.html

# Mesh sequence video
python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh.mp4

# 360° rotation
python scripts/visualize_mesh_sequence.py results/fitting/xxx --rotating --output rotating.mp4
```

## Output Location
```
results/fitting/{dataset}_{views}_{keypoints}_{timestamp}/
├── render/step_2_frame_*.png    # Final mesh renders
├── obj/step_2_frame_*.obj       # 3D meshes (for downstream)
└── params/step_2_frame_*.pkl    # Model parameters
```

---

## UV Map Generation

### Prerequisites
```bash
pip install optuna  # For hyperparameter optimization
# Optional: pip install wandb (for dashboard visualization)
```

### Quick Run (using baseline fitting result)
```bash
# Run UV map pipeline
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --uv_size 512

# With photometric optimization
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --uv_size 512 \
    --optimize
```

### Hyperparameter Optimization

#### Optuna (recommended, no extra dependencies)
```bash
python -m uvmap.optuna_optimizer \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --n_trials 30 \
    --output_dir results/uvmap_optuna
```

#### WandB Sweep (dashboard visualization)
```bash
pip install wandb
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --count 30 \
    --project uvmap-optimization
```

### Ablation Study (grid search)
```bash
python -m uvmap.experiment_runner \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --output_dir results/uvmap_ablation \
    --quick  # Fast test with fewer combinations
```

### UV Map Output
```
results/uvmap/
├── texture_final.png      # Final UV texture map
├── confidence.png         # Per-pixel confidence
├── uv_mask.png           # Valid UV regions
└── texture.pt            # Tensor for downstream use
```

### Available Fitting Results (Baseline)
| Experiment | Path |
|------------|------|
| 6-view, 22kp | `markerless_mouse_1_nerf_v012345_kp22_20251206_165254` |
| 6-view, 9kp  | `markerless_mouse_1_nerf_v012345_sparse9_20251207_081918` |
| 6-view, 7kp  | `markerless_mouse_1_nerf_v012345_sparse7_20251207_172028` |
| 6-view, 5kp  | `markerless_mouse_1_nerf_v012345_sparse5_20251208_134918` |

---
**Details**: [docs/guides/experiments.md](docs/guides/experiments.md) | [docs/guides/output.md](docs/guides/output.md)
