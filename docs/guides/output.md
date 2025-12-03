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
│   │   ├── step0_frame_000000_grid.jpg      # Step0 all iterations (compressed grid)
│   │   └── step1_frame_000000_grid.jpg      # Step1 all iterations (compressed grid)
│   └── keypoints/
│       ├── step_1_frame_*_keypoints.png     # Predicted only
│       └── step_1_frame_*_keypoints_compare.png  # GT vs Predicted
├── params/
│   ├── step_1_frame_000000.pkl              # Step1 parameters
│   └── step_2_frame_000000.pkl              # Step2 parameters (final)
└── obj/
    └── step_2_frame_000000.obj              # 3D mesh
```

### Debug Grid Images

Iteration별 렌더링 결과는 개별 PNG 파일 대신 **압축된 grid JPEG**로 저장됩니다:

- **저장 용량**: ~95% 감소 (수백 MB → 수 MB)
- **형식**: 5열 grid, 320x240 썸네일, JPEG 85% 품질
- **한 눈에 확인**: 최적화 진행 과정을 단일 이미지로 확인 가능

```
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ iter 0  │ iter 1  │ iter 2  │ iter 3  │ iter 4  │
├─────────┼─────────┼─────────┼─────────┼─────────┤
│ iter 5  │ iter 6  │ iter 7  │ iter 8  │ iter 9  │
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

**기존 결과 압축 (post-processing):**
```python
from utils.debug_grid import compress_existing_debug_folder

# 기존 PNG 파일들을 grid로 압축
compress_existing_debug_folder("results/fitting/xxx/render/debug/", delete_originals=True)
```

## Fitting 3 Steps

| Step | Name | Optimized | Description |
|------|------|-----------|-------------|
| **0** | Global Positioning | `trans`, `rotation`, `scale` | Initial pose (joints frozen) |
| **1** | Articulation | `thetas`, `bone_lengths` | Joint angles & bones |
| **2** | Silhouette Refinement | All params | Mask loss enabled |

## Loss Values Explained

### 출력 값의 의미

**출력되는 `theta`, `2d`, `bone` 등은 weight 적용 전 개별 Loss 값 (raw)입니다.**

```
[Step1] iter 0: total=60.91 | theta:1.61 | 2d:129.63 | bone:1.61 | ...
                ↑ 가중합      ↑ raw loss (weight 적용 전)
```

### 계산 과정

```python
# 1. 개별 loss 계산 (출력되는 raw 값)
loss_theta = 1.61      # ← theta:1.61
loss_2d = 129.63       # ← 2d:129.63
loss_bone = 1.61       # ← bone:1.61

# 2. Weight 곱해서 total 계산
total = loss_theta * 3.0 +    # = 4.83
        loss_2d * 0.2 +       # = 25.93
        loss_bone * 0.5 + ... # = 0.81
      = 60.91                 # ← total=60.91
```

### Loss Terms 및 Default Weights

| Loss | Weight | Description |
|------|--------|-------------|
| **theta** | 3.0 | Joint angle regularization (초기값 0에 가깝게) |
| **2d** | 0.2 | 2D keypoint reprojection error (pixels²) |
| **bone** | 0.5 | Bone length constraint (기본값 유지) |
| **scale** | 0.5 | Scale regularization (목표: 115mm) |
| **mask** | Step별 | Silhouette IoU (Step0,1=0, Step2=3000) |
| **chest_d** | 0.1 | Chest deformation regularization |
| **stretch** | 1.0 | Bone stretch penalty (관절 연결 제약) |

### Typical Raw Values

| Loss | Step0 Range | Step1 Range | Step2 Range |
|------|-------------|-------------|-------------|
| theta | 0 (frozen) | 5→0.5 | 0.5→0.3 |
| 2d | 700→100 | 130→20 | 20→15 |
| bone | 0 (frozen) | 5→0.5 | 0.5→0.3 |
| mask | 0 (disabled) | 0 (disabled) | 0.15→0.05 |
| stretch | ~2 | 20→5 | 5→3 |

### Step별 Loss 변화 패턴

**Step 0 (Global Positioning)**
```
iter   0: total=284.14 | 2d:719.57 | scale:6.73 | stretch:2.25
iter 100: total= 52.30 | 2d:102.45 | scale:0.82 | stretch:1.85
```
- `2d` 급격히 감소 (전역 위치/스케일 맞춤)
- `theta`, `bone` = 0 (관절 고정)

**Step 1 (Articulation)**
```
iter   0: total=60.91 | theta:1.61 | 2d:129.63 | bone:1.61 | stretch:22.40
iter 200: total=12.35 | theta:0.42 | 2d: 23.18 | bone:0.35 | stretch: 4.82
```
- `theta`, `bone` 활성화 (관절 최적화)
- `stretch` 처음 높다가 안정화

**Step 2 (Silhouette)**
```
iter   0: total=15.28 | mask:0.12 | 2d:28.45
iter 100: total= 8.92 | mask:0.05 | 2d:16.32
```
- `mask` 활성화 (실루엣 정합)
- 미세 조정 단계

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
