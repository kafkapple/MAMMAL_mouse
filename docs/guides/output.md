# Output Guide

## Result Folder Structure

```
results/fitting/{dataset}_{views}_{keypoints}_{timestamp}/
├── config.yaml                              # Experiment config (reproducibility)
├── loss_history.json                        # Training loss log
├── render/
│   ├── step_1_frame_000000.png              # Step1: Articulation fitting
│   ├── step_2_frame_000000.png              # Step2: Final result
│   ├── step_summary_frame_000000.png        # 3-step comparison (first frame)
│   ├── debug/
│   │   ├── step_0_frame_*_iter_*.png        # Step0 iterations
│   │   └── step_1_frame_*_iter_*.png        # Step1 iterations
│   └── keypoints/
│       ├── step_1_frame_*_keypoints.png     # Predicted only
│       └── step_1_frame_*_keypoints_compare.png  # GT vs Predicted
├── params/
│   ├── step_1_frame_000000.pkl              # Step1 parameters
│   └── step_2_frame_000000.pkl              # Step2 parameters (final)
└── obj/
    └── step_2_frame_000000.obj              # 3D mesh
```

## Fitting 3 Steps

| Step | Name | Optimized | Description |
|------|------|-----------|-------------|
| **0** | Global Positioning | `trans`, `rotation`, `scale` | Initial pose (joints frozen) |
| **1** | Articulation | `thetas`, `bone_lengths` | Joint angles & bones |
| **2** | Silhouette Refinement | All params | Mask loss enabled |

## Using as 3D Geometric Prior

### Option 1: OBJ Files (Recommended)

```python
import trimesh
import glob

# Single frame
mesh = trimesh.load("obj/step_2_frame_000000.obj")
vertices = mesh.vertices  # (N_verts, 3)
faces = mesh.faces        # (N_faces, 3)

# All frames
obj_files = sorted(glob.glob("obj/step_2_frame_*.obj"))
meshes = [trimesh.load(f) for f in obj_files]
```

### Option 2: PKL Parameters (Advanced)

```python
import pickle
from bodymodel_th import BodyModelTorch

with open("params/step_2_frame_000000.pkl", "rb") as f:
    params = pickle.load(f)

# params structure:
# {
#     "thetas": (1, 20, 3),       # Joint rotations (axis-angle)
#     "bone_lengths": (1, 20),    # Bone length offsets
#     "trans": (1, 3),            # 3D position (mm)
#     "rotation": (1, 3),         # Global rotation
#     "scale": (1, 1),            # Scale factor
#     "chest_deformer": (1, 1),   # Chest deformation
# }

# Reconstruct mesh
bodymodel = BodyModelTorch(device='cuda')
V, J = bodymodel.forward(
    params["thetas"], params["bone_lengths"],
    params["rotation"], params["trans"], params["scale"],
    params["chest_deformer"]
)
vertices = V[0].cpu().numpy()
```

## Multi-View Data Structure

```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/
│   ├── 0.mp4          # Camera 0
│   ├── 1.mp4          # Camera 1
│   └── ...            # Camera 2-5
├── new_cam.pkl        # Camera params (list of 6 dicts)
└── keypoints2d_undist/
    ├── result_view_0.pkl
    └── ...
```

**Camera params:**
```python
import pickle
with open("new_cam.pkl", "rb") as f:
    cams = pickle.load(f)  # List[Dict]

# cams[i] = {'K': (3,3), 'R': (3,3), 'T': (3,1), ...}
```

**Synchronization:** Same frame index = same timestamp across all views.

## Visualization Scripts

```bash
# Mesh sequence video
python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh.mp4

# 360° rotation
python scripts/visualize_mesh_sequence.py results/fitting/xxx --rotating -o rotating.mp4

# From OBJ only (no BodyModel)
python scripts/visualize_mesh_sequence.py results/fitting/xxx --use-obj -o mesh.mp4

# Compare experiments (HTML report)
python scripts/compare_experiments.py "results/fitting/*sparse*" -o comparison.html
```

---
**Related:** [QUICKSTART](../../QUICKSTART.md) | [Experiments](../../conf/experiment/README.md) | [Keypoints](../KEYPOINT_REFERENCE.md)
