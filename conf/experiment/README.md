# Experiment Configurations

## Quick Start: 4 Main Experiments

| Experiment | Views | Keypoints | Description |
|------------|-------|-----------|-------------|
| `baseline_6view_keypoint` | 6 | 22 (full) | **Paper Baseline**: Original MAMMAL setting |
| `monocular_keypoint` | 1 | 22 (full) | Single-view reconstruction |
| `sixview_no_keypoint` | 6 | 0 (none) | Silhouette-only mode |
| `sixview_sparse_keypoint` | 6 | 3 (sparse) | Minimal keypoints (nose, body, tail) |

### Run Commands

```bash
# ===== Debug Mode (5 frames, fast) =====
./run_experiment.sh baseline_6view_keypoint --debug
./run_experiment.sh monocular_keypoint --debug
./run_experiment.sh sixview_no_keypoint --debug
./run_experiment.sh sixview_sparse_keypoint --debug

# ===== Full Run (100 frames, paper settings) =====
./run_experiment.sh baseline_6view_keypoint
./run_experiment.sh monocular_keypoint
./run_experiment.sh sixview_no_keypoint
./run_experiment.sh sixview_sparse_keypoint

# ===== Custom Frame Range =====
./run_experiment.sh baseline_6view_keypoint --frames 50
```

### Direct Python Commands

```bash
# Debug mode
PYOPENGL_PLATFORM=egl python fitter_articulation.py \
    experiment=baseline_6view_keypoint \
    fitter.end_frame=5 \
    optim.solve_step0_iters=5 \
    optim.solve_step1_iters=20 \
    optim.solve_step2_iters=10

# Full run
PYOPENGL_PLATFORM=egl python fitter_articulation.py experiment=baseline_6view_keypoint
```

### Evaluation & Visualization

```bash
# Generate report
python scripts/evaluate_experiment.py results/fitting/<experiment_folder>

# Compare experiments
python scripts/evaluate_experiment.py results/fitting/baseline_xxx \
    --compare results/fitting/monocular_xxx results/fitting/silhouette_xxx

# Interactive mesh viewer
python scripts/interactive_mesh_viewer.py results/fitting/<experiment_folder>

# Export video
python scripts/interactive_mesh_viewer.py results/fitting/<experiment_folder> \
    --export-video output.mp4 --fps 15
```

---

## Dataset Info

| Item | Value |
|------|-------|
| **Path** | `data/examples/markerless_mouse_1_nerf/` |
| **Cameras** | 6 views (0~5) |
| **Total Frames** | 18,000 per video |
| **Default Sampling** | `end_frame=1000`, `interval=10` → 100 samples |

## Usage

```bash
# Run with experiment config
python fitter_articulation.py +experiment=<experiment_name>

# Override specific parameters
python fitter_articulation.py +experiment=views_6 fitter.end_frame=500

# Quick test (5 frames only)
python fitter_articulation.py +experiment=quick_test
```

## View Ablation Experiments (with Keypoints)

| Config | Views | Cameras | Description |
|--------|-------|---------|-------------|
| `views_6` | 6 | 0,1,2,3,4,5 | Full 6-view baseline |
| `views_5` | 5 | 0,1,2,3,4 | Drop camera 5 |
| `views_4` | 4 | 0,1,2,3 | 4 consecutive cameras |
| `views_3_diagonal` | 3 | 0,2,4 | 3 diagonal cameras (better coverage) |
| `views_3_consecutive` | 3 | 0,1,2 | 3 consecutive cameras |
| `views_2_opposite` | 2 | 0,3 | 2 opposite cameras (stereo-like) |
| `views_1_single` | 1 | 0 | Single view (monocular) |

## Silhouette-Only Experiments (No Keypoints)

| Config | Views | Regularization | Description |
|--------|-------|----------------|-------------|
| `silhouette_only_6views` | 6 | theta=10, bone=2 | Mask-only, 6 views |
| `silhouette_only_4views` | 4 | theta=10, bone=2 | Mask-only, 4 views |
| `silhouette_only_3views` | 3 | theta=15, bone=3 | Mask-only, 3 diagonal views |
| `silhouette_only_1view` | 1 | theta=20, bone=5 | Mask-only, single view (challenging) |

## Other Experiments

| Config | Description |
|--------|-------------|
| `quick_test` | Fast test (5 frames, minimal iterations) |
| `accurate_6views` | High-quality fitting (more iterations) |
| `ablation_mask_weight` | Keypoints + mask loss combined |

## Expected Output Folder Names

```
results/fitting/
├── markerless_mouse_1_nerf_v012345_20251202_143000/  # 6 views
│   ├── config.yaml          # Saved configuration
│   ├── params/              # Fitted parameters
│   ├── render/              # Visualizations
│   └── obj/                 # Exported meshes
├── markerless_mouse_1_nerf_v01234_20251202_143100/   # 5 views
├── markerless_mouse_1_nerf_v0123_20251202_143200/    # 4 views
├── markerless_mouse_1_nerf_v024_20251202_143300/     # 3 diagonal
└── markerless_mouse_1_nerf_v0_20251202_143400/       # 1 view
```

## Batch Experiments

### Run all view ablations:
```bash
for exp in views_6 views_5 views_4 views_3_diagonal views_2_opposite views_1_single; do
    echo "Running $exp..."
    python fitter_articulation.py +experiment=$exp
done
```

### Run all silhouette-only experiments:
```bash
for exp in silhouette_only_6views silhouette_only_4views silhouette_only_3views silhouette_only_1view; do
    echo "Running $exp..."
    python fitter_articulation.py +experiment=$exp
done
```

### Run with different frame ranges:
```bash
# Short test (10 frames)
python fitter_articulation.py +experiment=views_6 fitter.end_frame=100 fitter.interval=10

# Medium (100 frames)
python fitter_articulation.py +experiment=views_6 fitter.end_frame=1000 fitter.interval=10

# Full dataset
python fitter_articulation.py +experiment=views_6 fitter.end_frame=18000 fitter.interval=10
```

## Previous Experiments (from logs)

| Date | Experiment | Key Settings |
|------|------------|--------------|
| 2025-11-25 | 6-view fitting | `use_keypoints=true` (default) |
| 2025-11-26 | Silhouette-only | `use_keypoints=false`, 6 views |
