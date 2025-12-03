# Experiment Configurations

## 📊 Complete Experiment Matrix

### All Available Experiments (24 configs)

| # | Config Name | Views | Keypoints | Category |
|---|-------------|-------|-----------|----------|
| 1 | `baseline_6view_keypoint` | 6 | 22 (full) | **Baseline** |
| 2 | `sixview_sparse_keypoint` | 6 | 3 (sparse) | Keypoint Ablation |
| 3 | `sixview_no_keypoint` | 6 | 0 (none) | Keypoint Ablation |
| 4 | `sparse_5view` | 5 | 3 (sparse) | Viewpoint Ablation |
| 5 | `sparse_4view` | 4 | 3 (sparse) | Viewpoint Ablation |
| 6 | `sparse_3view` | 3 | 3 (sparse) | Viewpoint Ablation |
| 7 | `sparse_2view` | 2 | 3 (sparse) | Viewpoint Ablation |
| 8 | `monocular_keypoint` | 1 | 22 (full) | Monocular |
| 9 | `views_6` | 6 | 22 (full) | Viewpoint (Full KP) |
| 10 | `views_5` | 5 | 22 (full) | Viewpoint (Full KP) |
| 11 | `views_4` | 4 | 22 (full) | Viewpoint (Full KP) |
| 12 | `views_3_diagonal` | 3 | 22 (full) | Viewpoint (Full KP) |
| 13 | `views_3_consecutive` | 3 | 22 (full) | Viewpoint (Full KP) |
| 14 | `views_2_opposite` | 2 | 22 (full) | Viewpoint (Full KP) |
| 15 | `views_1_single` | 1 | 22 (full) | Viewpoint (Full KP) |
| 16 | `silhouette_only_6views` | 6 | 0 (none) | Silhouette Only |
| 17 | `silhouette_only_4views` | 4 | 0 (none) | Silhouette Only |
| 18 | `silhouette_only_3views` | 3 | 0 (none) | Silhouette Only |
| 19 | `silhouette_only_1view` | 1 | 0 (none) | Silhouette Only |
| 20 | `quick_test` | 6 | 22 (full) | Debug/Test |
| 21 | `accurate_6views` | 6 | 22 (full) | High Quality |
| 22 | `ablation_mask_weight` | 6 | 22 (full) | Loss Ablation |

### Experiment Matrix Visualization

```
                    Keypoints
                22 (full)    3 (sparse)    0 (none)
           ┌─────────────┬─────────────┬─────────────┐
Views   6  │ baseline_*  │ sixview_    │ sixview_no_ │
           │ views_6     │ sparse_kp   │ keypoint    │
           │ accurate_*  │             │ sil_only_6  │
           ├─────────────┼─────────────┼─────────────┤
        5  │ views_5     │ sparse_5v   │      -      │
           ├─────────────┼─────────────┼─────────────┤
        4  │ views_4     │ sparse_4v   │ sil_only_4  │
           ├─────────────┼─────────────┼─────────────┤
        3  │ views_3_*   │ sparse_3v   │ sil_only_3  │
           ├─────────────┼─────────────┼─────────────┤
        2  │ views_2_opp │ sparse_2v   │      -      │
           ├─────────────┼─────────────┼─────────────┤
        1  │ views_1_*   │      -      │ sil_only_1  │
           │ monocular_* │             │             │
           └─────────────┴─────────────┴─────────────┘
```

---

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
| `sixview_sparse_keypoint` | 6 | 3 (sparse) | Nose, tail base, neck only |
| `sixview_no_keypoint` | 6 | 0 (none) | Silhouette only (mask-based) |

### Sparse Keypoint Indices

**Note**: GT annotation (`mouse_22_defs.py`)과 Model definition (`keypoint22_mapper.json`)의 Head keypoints 순서가 다릅니다.
- GT: 0=L_ear, 1=R_ear, **2=nose**
- Model: 0=nose, 1=L_ear, 2=R_ear

실제 데이터는 GT 정의를 따르므로:
- **idx_2**: Nose (GT 기준) - weight: 5.0
- **idx_5**: Tail root - weight: 3.0
- **idx_3**: Neck - weight: 5.0

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
  # Sparse keypoint indices (GT annotation 기준 - mouse_22_defs.py):
  #   idx 2: Nose (GT에서 nose는 index 2)
  #   idx 5: Tail root
  #   idx 3: Neck
  sparse_keypoint_indices: [2, 5, 3]  # Active keypoint indices (GT 기준)
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
  idx_2: 5.0                     # Nose weight (GT index 2)
  idx_5: 3.0                     # Tail root weight
  idx_3: 5.0                     # Neck weight
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
# In your config yaml - Use GT indices from mouse_22_defs.py
# See "22 Keypoint Semantic Labels" section below for full mapping
fitter:
  sparse_keypoint_indices: [2, 3, 5, 6, 7]  # Custom 5 keypoints example

keypoint_weights:
  default: 0.0
  idx_2: 5.0    # Nose (GT index 2, NOT 0!)
  idx_3: 3.0    # Neck
  idx_5: 3.0    # Tail root
  idx_6: 3.0    # Tail mid
  idx_7: 3.0    # Tail end
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

---

## 22 Keypoint Semantic Labels

**Source**: `mouse_22_defs.py` (GT annotation 기준)

**중요**: GT annotation과 Model definition의 Head keypoint 순서가 다릅니다!

| Index | GT Label (mouse_22_defs.py) | Model Label (keypoint22_mapper) | Body Part |
|-------|-----------------------------|---------------------------------|-----------|
| 0 | **left_ear_tip** | nose | Head |
| 1 | **right_ear_tip** | left_ear | Head |
| 2 | **nose** | right_ear | Head |
| 3 | neck | neck | Body |
| 4 | body_middle | body | Body |
| 5 | tail_root | tail_base | Tail |
| 6 | tail_middle | tail_mid | Tail |
| 7 | tail_end | tail_tip | Tail |
| 8 | left_paw | L_forepaw_digit | L Front |
| 9 | left_paw_end | L_forepaw | L Front |
| 10 | left_elbow | L_ulna | L Front |
| 11 | left_shoulder | L_humerus | L Front |
| 12 | right_paw | R_forepaw_digit | R Front |
| 13 | right_paw_end | R_forepaw | R Front |
| 14 | right_elbow | R_ulna | R Front |
| 15 | right_shoulder | R_humerus | R Front |
| 16 | left_foot | L_hindpaw_digit | L Hind |
| 17 | left_knee | L_hindpaw | L Hind |
| 18 | left_hip | L_tibia | L Hind |
| 19 | right_foot | R_hindpaw_digit | R Hind |
| 20 | right_knee | R_hindpaw | R Hind |
| 21 | right_hip | R_tibia | R Hind |

### Recommended Sparse Keypoints (GT 기준)

For minimal annotation, use these 3 keypoints that span the body:
- **idx 2**: Nose (head) - GT에서 nose는 index 2!
- **idx 5**: Tail root (body junction)
- **idx 3**: Neck (body center)

```yaml
sparse_keypoint_indices: [2, 5, 3]  # GT 기준
```

---

## 📈 Results Comparison & Visualization

### Existing Experiment Results

| Timestamp | Views | Keypoints | Camera Config | Frames | Status |
|-----------|-------|-----------|---------------|--------|--------|
| `20251203_171841` | 6 | sparse 3 | 0,1,2,3,4,5 | 2 | ✅ Complete |
| `20251203_173315` | 5 | sparse 3 | 0,1,2,3,4 | 2 | ✅ Complete |
| `20251203_174322` | 4 | sparse 3 | 0,1,2,3 | 2 | ✅ Complete |
| `20251203_175157` | 3 | sparse 3 | 0,2,4 (diagonal) | 2 | ✅ Complete |
| `20251203_175826` | 2 | sparse 3 | 0,3 (opposite) | 2 | ✅ Complete |

### Result Folder Structure
```
results/fitting/{dataset}_{views}_{keypoints}_{timestamp}/
├── config.yaml                    # Reproducibility config
├── loss_history.json              # Training loss log
├── render/
│   ├── step_1_frame_000000.png    # Step1 result + keypoints
│   ├── step_2_frame_000000.png    # Step2 final result
│   ├── step_summary_frame_*.png   # 3-step comparison
│   ├── debug/                     # Iteration debug images
│   │   ├── step_0_frame_*_iter_*.png
│   │   └── step_1_frame_*_iter_*.png
│   └── keypoints/                 # GT vs Pred comparison
│       ├── step_1_frame_*_keypoints.png
│       └── step_1_frame_*_keypoints_compare.png
├── params/                        # Model parameters (PKL)
│   ├── step_1_frame_000000.pkl
│   └── step_2_frame_000000.pkl
└── obj/                           # 3D mesh files
    └── step_2_frame_000000.obj
```

### Compare Results Across Experiments

```bash
# Create comparison gallery from multiple experiments
python scripts/compare_experiments.py \
    results/fitting/markerless_mouse_1_nerf_v012345_sparse3_* \
    results/fitting/markerless_mouse_1_nerf_v01234_sparse3_* \
    results/fitting/markerless_mouse_1_nerf_v0123_sparse3_* \
    --output results/comparison_gallery.png

# Visualize single experiment as video
python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh.mp4

# Visualize with 360° rotation
python scripts/visualize_mesh_sequence.py results/fitting/xxx --rotating --output rotating.mp4
```

### Expected Quality vs Camera Count

Based on ablation experiments:

| Views | Expected Quality | Notes |
|-------|------------------|-------|
| 6 | ⭐⭐⭐⭐⭐ | Best - full coverage |
| 5 | ⭐⭐⭐⭐ | Slight degradation |
| 4 | ⭐⭐⭐⭐ | Good - still robust |
| 3 (diagonal) | ⭐⭐⭐ | Acceptable - coverage matters |
| 2 (opposite) | ⭐⭐ | Challenging - depth ambiguity |
| 1 | ⭐ | Very difficult - monocular |

### Expected Quality vs Keypoint Count

| Keypoints | Expected Quality | Notes |
|-----------|------------------|-------|
| 22 (full) | ⭐⭐⭐⭐⭐ | Best - strong constraints |
| 3 (sparse) | ⭐⭐⭐⭐ | Good - nose/neck/tail sufficient |
| 0 (none) | ⭐⭐ | Challenging - silhouette only |

---

## 🔬 Recommended Experiment Workflow

### Step 1: Debug Run (Quick Validation)
```bash
# 1. Test with debug mode first (2 frames, minimal iterations)
./run_experiment.sh sparse_3view --debug

# Check output exists
ls results/fitting/markerless_mouse_1_nerf_v024_sparse3_*/render/
```

### Step 2: Full Run (5+ frames)
```bash
# 2. Run full experiment
./run_experiment.sh sparse_3view

# Or with custom frame count
./run_experiment.sh sparse_3view --frames 100
```

### Step 3: Analyze Results
```bash
# 3. Visualize results
python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh.mp4

# Check keypoint accuracy
ls results/fitting/xxx/render/keypoints/
```

### Step 4: Compare with Baseline
```bash
# 4. Compare with baseline
./run_experiment.sh baseline_6view_keypoint

# Create comparison
python scripts/compare_experiments.py \
    results/fitting/xxx_baseline_* \
    results/fitting/xxx_sparse_* \
    --output comparison.png
```
