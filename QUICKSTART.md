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
./run_experiment.sh baseline_6view_keypoint        # 22 keypoints
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
```bash
# Keypoint ablation
for exp in baseline_6view_keypoint sixview_sparse_keypoint sixview_no_keypoint; do
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
**Details**: [docs/guides/experiments.md](docs/guides/experiments.md) | [docs/guides/output.md](docs/guides/output.md)
