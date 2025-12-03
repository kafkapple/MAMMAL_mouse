# MAMMAL_mouse

Markerless 3D mouse pose estimation and mesh reconstruction from multi-view video.

![mouse_model](assets/figs/mouse_1.png)

## Quick Links

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | All experiment commands |
| [Experiments Guide](conf/experiment/README.md) | Config details, ablation design |
| [Output Guide](docs/guides/output.md) | Result files, 3D prior usage |
| [Keypoint Reference](docs/KEYPOINT_REFERENCE.md) | 22 keypoints definition |
| [Full Usage Guide](docs/guides/full_usage_guide.md) | Detailed documentation |

## Installation

```bash
conda create -n mammal_stable python=3.10
conda activate mammal_stable
pip install -r requirements.txt
```

## Basic Usage

```bash
# Run experiment
./run_experiment.sh baseline_6view_keypoint

# Quick test (debug)
./run_experiment.sh quick_test --debug
```

→ See [QUICKSTART.md](QUICKSTART.md) for all commands

## Experiment Matrix

```
                Keypoints: 22(full)   3(sparse)   0(none)
              ┌───────────┬───────────┬───────────┐
Views:    6   │ baseline  │ sixview_  │ sixview_  │
              │           │ sparse    │ no_kp     │
          5   │ views_5   │ sparse_5v │     -     │
          4   │ views_4   │ sparse_4v │ sil_only  │
          3   │ views_3_* │ sparse_3v │ sil_only  │
          2   │ views_2   │ sparse_2v │     -     │
          1   │ views_1   │     -     │ sil_only  │
              └───────────┴───────────┴───────────┘
```

## Output Structure

```
results/fitting/{experiment}/
├── render/step_2_frame_*.png   # Mesh renders
├── obj/step_2_frame_*.obj      # 3D meshes (for NeRF/3DGS prior)
└── params/step_2_frame_*.pkl   # Model parameters
```

**Use as 3D prior:**
```python
import trimesh
mesh = trimesh.load("obj/step_2_frame_000000.obj")
vertices = mesh.vertices  # (N, 3)
```

## Data Structure

```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/0.mp4~5.mp4   # 6 camera views
├── new_cam.pkl                  # Camera parameters
└── keypoints2d_undist/          # 2D keypoint annotations
```

---
**License**: MIT | **Paper**: [MAMMAL](https://github.com/MAMMAL/paper)
