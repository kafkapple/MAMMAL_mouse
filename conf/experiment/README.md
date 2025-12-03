# Experiment Configurations

## Ablation Study Design

### Overview

| Group | Fixed Variable | Ablation Variable | Purpose |
|-------|---------------|-------------------|---------|
| **1. Baseline** | - | - | Paper reference (6-view + 22 keypoints) |
| **2. Keypoint Ablation** | 6-view | Keypoints (22→3→0) | Test keypoint dependency |
| **3. Viewpoint Ablation** | Sparse 3 keypoints | Views (6→5→4→3→2) | Test camera requirement |

---

## Group 1: Baseline (Paper Reference)

| Experiment | Views | Keypoints | Description |
|------------|-------|-----------|-------------|
| `baseline_6view_keypoint` | 6 (0-5) | 22 (full) | **MAMMAL Paper Baseline** |

### Commands
```bash
# Debug (2 frames, minimal iterations)
./run_experiment.sh baseline_6view_keypoint --debug

# Full run (5 frames default, use --frames N for more)
# Note: Frame count = temporal samples, NOT quality. Each frame fitted independently.
./run_experiment.sh baseline_6view_keypoint
```

---

## Group 2: Keypoint Ablation (6-view fixed)

| Experiment | Views | Keypoints | Description |
|------------|-------|-----------|-------------|
| `baseline_6view_keypoint` | 6 | 22 (full) | Full keypoints |
| `sixview_sparse_keypoint` | 6 | 3 (sparse) | Nose, body center, tail base only |
| `sixview_no_keypoint` | 6 | 0 (none) | Silhouette only (mask-based) |

### Sparse Keypoint Indices
- **idx_0**: Nose (weight: 5.0)
- **idx_18**: Tail base (weight: 3.0)
- **idx_21**: Body center (weight: 5.0)

### Commands
```bash
# Debug
./run_experiment.sh baseline_6view_keypoint --debug     # Full 22 keypoints
./run_experiment.sh sixview_sparse_keypoint --debug    # Sparse 3 keypoints
./run_experiment.sh sixview_no_keypoint --debug        # Silhouette only

# Full run
./run_experiment.sh baseline_6view_keypoint
./run_experiment.sh sixview_sparse_keypoint
./run_experiment.sh sixview_no_keypoint
```

---

## Group 3: Viewpoint Ablation (Sparse 3 keypoints fixed)

| Experiment | Views | Cameras | Keypoints | Description |
|------------|-------|---------|-----------|-------------|
| `sixview_sparse_keypoint` | 6 | 0,1,2,3,4,5 | sparse 3 | Reference |
| `sparse_5view` | 5 | 0,1,2,3,4 | sparse 3 | Drop camera 5 |
| `sparse_4view` | 4 | 0,1,2,3 | sparse 3 | 4 consecutive |
| `sparse_3view` | 3 | 0,2,4 | sparse 3 | Diagonal (better coverage) |
| `sparse_2view` | 2 | 0,3 | sparse 3 | Opposite (stereo-like) |

### Commands
```bash
# Debug
./run_experiment.sh sixview_sparse_keypoint --debug   # 6view + sparse3
./run_experiment.sh sparse_5view --debug              # 5view + sparse3
./run_experiment.sh sparse_4view --debug              # 4view + sparse3
./run_experiment.sh sparse_3view --debug              # 3view + sparse3 (diagonal)
./run_experiment.sh sparse_2view --debug              # 2view + sparse3 (opposite)

# Full run
./run_experiment.sh sixview_sparse_keypoint
./run_experiment.sh sparse_5view
./run_experiment.sh sparse_4view
./run_experiment.sh sparse_3view
./run_experiment.sh sparse_2view
```

---

## Config Parameter Reference

### Config Structure
```yaml
# Example: conf/experiment/sparse_3view.yaml
data:
  views_to_use: [0, 2, 4]        # Camera indices to use

fitter:
  render_cameras: [0, 2, 4]      # Should match views_to_use
  use_keypoints: true            # Enable keypoint loss
  keypoint_num: 22               # Total keypoint count
  sparse_keypoint_indices: [0, 18, 21]  # Active keypoint indices
  start_frame: 0
  end_frame: 100
  interval: 1
  with_render: true

optim:
  solve_step0_iters: 20          # Global positioning
  solve_step1_iters: 180         # Articulation fitting
  solve_step2_iters: 50          # Silhouette refinement

loss_weights:
  theta: 6.0                     # Pose regularization
  "2d": 0.6                      # 2D keypoint loss weight
  bone: 1.5                      # Bone length constraint
  scale: 1.5                     # Scale regularization
  mask_step0: 0.0                # Mask weight in step0
  mask_step1: 800.0              # Mask weight in step1
  mask_step2: 3000.0             # Mask weight in step2

keypoint_weights:
  default: 0.0                   # Default weight (0 for sparse mode)
  idx_0: 5.0                     # Nose weight
  idx_18: 3.0                    # Tail base weight
  idx_21: 5.0                    # Body center weight
  tail_step2: 10.0               # Tail weight in step2
```

### Code Application Locations

| Config Parameter | File | Line | Effect |
|-----------------|------|------|--------|
| `data.views_to_use` | `data_seaker_video_new.py:20` | Loads only specified cameras |
| `fitter.use_keypoints` | `fitter_articulation.py:153` | Disables 2D loss if false |
| `fitter.sparse_keypoint_indices` | `fitter_articulation.py:85` | Sets non-sparse weights to 0 |
| `keypoint_weights.default` | `fitter_articulation.py:88` | Base weight for all keypoints |
| `keypoint_weights.idx_*` | `fitter_articulation.py:98-103` | Per-keypoint weights |
| `loss_weights.*` | `fitter_articulation.py:125-141` | Term weights in optimization |
| `optim.solve_step*_iters` | `fitter_articulation.py:939-941` | Iteration counts |

---

## Expected Output Folder Names

```
results/fitting/
├── markerless_mouse_1_nerf_v012345_kp22_20251203_*/    # Baseline
├── markerless_mouse_1_nerf_v012345_sparse3_20251203_*/ # 6view sparse
├── markerless_mouse_1_nerf_v012345_noKP_20251203_*/    # 6view no keypoint
├── markerless_mouse_1_nerf_v01234_sparse3_20251203_*/  # 5view sparse
├── markerless_mouse_1_nerf_v0123_sparse3_20251203_*/   # 4view sparse
├── markerless_mouse_1_nerf_v024_sparse3_20251203_*/    # 3view sparse (diagonal)
└── markerless_mouse_1_nerf_v03_sparse3_20251203_*/     # 2view sparse (opposite)
```

---

## Customization Examples

### Change Sparse Keypoint Indices
```yaml
# In your config yaml
fitter:
  sparse_keypoint_indices: [0, 5, 10, 18, 21]  # Custom 5 keypoints

keypoint_weights:
  default: 0.0
  idx_0: 5.0
  idx_5: 3.0
  idx_10: 3.0
  idx_18: 3.0
  idx_21: 5.0
```

### Change Camera Configuration
```yaml
# Custom 3-view setup (consecutive cameras)
data:
  views_to_use: [0, 1, 2]

fitter:
  render_cameras: [0, 1, 2]
```

### CLI Override
```bash
# Override frame range
./run_experiment.sh sparse_3view --frames 200

# Override via direct python
python fitter_articulation.py experiment=sparse_3view \
    fitter.end_frame=200 \
    loss_weights.mask_step2=5000.0
```

---

## Batch Experiment Scripts

### Run all keypoint ablations
```bash
for exp in baseline_6view_keypoint sixview_sparse_keypoint sixview_no_keypoint; do
    echo "=== Running $exp ==="
    ./run_experiment.sh $exp
done
```

### Run all viewpoint ablations
```bash
for exp in sixview_sparse_keypoint sparse_5view sparse_4view sparse_3view sparse_2view; do
    echo "=== Running $exp ==="
    ./run_experiment.sh $exp
done
```

### Run complete ablation study
```bash
# All experiments
for exp in baseline_6view_keypoint sixview_sparse_keypoint sixview_no_keypoint \
           sparse_5view sparse_4view sparse_3view sparse_2view; do
    echo "=== Running $exp ==="
    ./run_experiment.sh $exp
done
```
