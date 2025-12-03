# MAMMAL Mouse Mesh Fitting: 모델 및 메커니즘 분석

**Date**: 2025-11-25
**Purpose**: 연구 발표용 mesh fitting 시스템 체계적 설명

---

## 1. 문서 위치 안내

### 기본 사용법
| 문서 | 위치 | 내용 |
|-----|------|------|
| **Quick Start** | `README.md` | 설치, 데이터 준비, 실행 예시 |
| **Mesh Fitting Guide** | `docs/guides/fitting/MESH_FITTING_GUIDE.md` | 전체 가이드 |
| **Cheatsheet** | `docs/guides/fitting/MESH_FITTING_CHEATSHEET.md` | 빠른 참조 |
| **Monocular Fitting** | `docs/guides/fitting/MONOCULAR_FITTING_GUIDE.md` | 단일 카메라 피팅 |

### 데이터 준비/어노테이션
| 문서 | 위치 |
|-----|------|
| Annotation 가이드 | `docs/guides/annotation/` |
| Video 처리 | `docs/guides/preprocessing/` |

---

## 2. 특정 데이터로 Mesh Fitting 실행

### 2.1 데이터 준비 요구사항

```
your_data/
├── frame_000000_cropped.png      # 크롭된 이미지
├── frame_000000_mask.png         # Binary 마스크 (선택)
├── keypoints2d_undist/           # 2D 키포인트 (선택)
│   └── result_view_0.pkl
└── new_cam.pkl                   # 카메라 파라미터 (multi-view)
```

### 2.2 실행 명령어

**방법 1: Cropped Frames (마스크 기반)**
```bash
conda activate mammal_stable

# 기본 실행
python fit_cropped_frames.py \
    /path/to/your_data \
    --output-dir results/my_fitting \
    --max-frames 20

# Shell 스크립트 사용
./run_mesh_fitting_cropped.sh /path/to/your_data results/my_fitting 20
```

**방법 2: Multi-View (Hydra 설정)**
```bash
# Default dataset 사용
python fitter_articulation.py dataset=default_markerless

# Custom dataset 설정
python fitter_articulation.py \
    dataset=custom \
    data.data_dir="/path/to/your_data" \
    fitter.start_frame=0 \
    fitter.end_frame=100
```

**방법 3: Monocular (단일 카메라)**
```bash
python fit_monocular.py \
    --input_dir /path/to/frames \
    --output_dir results/monocular \
    --detector yolo \
    --max_images 50
```

---

## 3. Mesh Fitting 모델 구조

### 3.1 MAMMAL Mouse Model 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    MAMMAL Mouse Model                        │
├─────────────────────────────────────────────────────────────┤
│  Vertices: 14,522개 (3D mesh 정점)                          │
│  Joints:   140개 (Skeleton articulation)                    │
│  Keypoints: 22개 (Semantic landmarks)                       │
│  Faces:    ~7,200개 (Mesh triangles, reduced)               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 22 Semantic Keypoints

**Source**: `mouse_model/keypoint22_mapper.json`

| Index | Type | Joint/Vertex IDs | Semantic Meaning |
|-------|------|------------------|------------------|
| 0 | V | [12274, 12225] | **Nose** |
| 1 | V | [4966, 5011] | Left ear |
| 2 | V | [13492, ...] | Right ear |
| 3 | J | [64, 65] | **Neck** (neck_stretch) |
| 4 | V | [9043] | Body vertex |
| 5 | J | [48, 51] | **Tail base** (lumbar_vertebrae_0 + tail_0) |
| 6 | J | [54, 55] | Tail mid (tail_3 + tail_4) |
| 7 | J | [61] | **Tail tip** (tail_9_end) |
| 8 | J | [79] | Left forepaw digit |
| 9 | J | [74] | Left forepaw |
| 10 | J | [73] | Left ulna |
| 11 | J | [70] | Left humerus |
| 12 | J | [104] | Right forepaw digit |
| 13 | J | [99] | Right forepaw |
| 14 | J | [98] | Right ulna |
| 15 | J | [95] | Right humerus |
| 16 | J | [15] | Left hindpaw digit |
| 17 | J | [5] | Left hindpaw |
| 18 | J | [4] | Left tibia |
| 19 | J | [38] | Right hindpaw digit |
| 20 | J | [28] | Right hindpaw |
| 21 | J | [27] | Right tibia |

**Note**: `V` = Vertex average, `J` = Joint position

### 3.3 모델 파일 구조

```
mouse_model/
├── mouse.pkl                    # 전체 모델 파라미터
├── mouse_txt/
│   ├── vertices.txt             # T-pose 정점 좌표
│   ├── faces_vert.txt           # Face indices
│   ├── skinning_weights.txt     # LBS weights
│   ├── init_joint_trans.pkl     # 초기 joint 위치
│   ├── init_joint_rotvec.pkl    # 초기 joint 회전
│   ├── parents.pkl              # Skeleton hierarchy
│   └── bone_length_mapper.txt   # Bone length mapping
├── keypoint22_mapper.json       # Keypoint-to-mesh mapping
└── reg_weights.txt              # Regularization weights
```

---

## 4. Mesh Fitting 메커니즘

### 4.1 전체 파이프라인

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Input      │───▶│  Detection   │───▶│ Optimization │
│  (Image)     │    │  (Keypoints) │    │   (LBFGS)    │
└──────────────┘    └──────────────┘    └──────────────┘
                                              │
                                              ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Output     │◀───│   Skinning   │◀───│   Forward    │
│ (3D Mesh)    │    │    (LBS)     │    │  Kinematics  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 4.2 핵심 클래스 구조

```python
# fitter_articulation.py
class MouseFitter:
    def __init__(self):
        self.bodymodel = ArticulationTorch()  # 모델 정의
        self.renderer_mask = MeshRenderer()    # Differentiable renderer

    def fit(self, target):
        # 1. Parameter 초기화
        # 2. Loss 계산
        # 3. LBFGS 최적화
        # 4. 결과 저장

# articulation_th.py
class ArticulationTorch(Module):
    def forward(self, thetas, bone_lengths, R, T, s, chest_deformer):
        # 1. Bone length 적용
        # 2. Rotation matrix 계산 (Rodrigues)
        # 3. Forward Kinematics (G matrix)
        # 4. Linear Blend Skinning
        # 5. Global transformation (R, T, s)
        return V_final, J_final  # Vertices, Joints
```

### 4.3 최적화 파라미터

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `thetas` | [B, 140, 3] | Joint axis-angle rotations |
| `bone_lengths` | [B, 19] | Bone length scaling factors |
| `R` | [B, 3] | Global rotation (axis-angle) |
| `T` | [B, 3] | Global translation |
| `s` | [B, 1] | Global scale (~115) |
| `chest_deformer` | [B, 1] | Chest deformation |

### 4.4 Loss Functions

```python
# 전체 Loss = weighted sum of:
Loss = w_2d × L_2d           # 2D keypoint reprojection
     + w_θ × L_theta         # Joint angle regularization
     + w_bone × L_bone       # Bone length constraint
     + w_scale × L_scale     # Scale regularization
     + w_mask × L_mask       # Silhouette matching (optional)
     + w_stretch × L_stretch # Stretch-to constraints
     + w_temp × L_temporal   # Temporal smoothness
```

**Loss Weights (기본값)**:
```python
term_weights = {
    "theta": 3.0,      # Joint angle regularization
    "2d": 0.2,         # 2D reprojection
    "bone": 0.5,       # Bone length
    "scale": 0.5,      # Scale constraint
    "mask": 0,         # Silhouette (disabled by default)
    "stretch": 1.0,    # Stretch constraints
    "temp": 0.25,      # Temporal smoothness
}
```

---

## 5. Forward Kinematics 상세

### 5.1 Rodrigues Rotation

Axis-angle → Rotation matrix 변환:

```python
def rodrigues(r):
    """
    r: [batch, 1, 3] axis-angle
    return: [batch, 3, 3] rotation matrix
    """
    θ = ||r||  # rotation angle
    r̂ = r / θ  # unit axis

    R = cos(θ)I + (1-cos(θ))r̂r̂ᵀ + sin(θ)[r̂]×
```

### 5.2 Skeleton Hierarchy (G Matrix)

```python
def compute_G(S, chest_deformer):
    """
    S: [B, 140, 4, 4] - single affine transforms
    G: [B, 140, 4, 4] - global transforms

    G[0] = S[0]  # Root
    G[i] = G[parent[i]] @ S[i]  # Chain multiplication
    """
    for joint in range(140):
        parent = parents[joint]
        G[joint] = G[parent] @ S[joint]
    return G
```

### 5.3 Linear Blend Skinning (LBS)

```python
def skinning(G):
    """
    V_posed = Σ w_j × G_j × v_template

    w_j: skinning weight for joint j
    G_j: global transform of joint j
    v_template: T-pose vertex position
    """
    # 1. Remove joint translation from G
    G_normalized = G - joint_positions

    # 2. Weight blending
    G_blended = einsum("ijkm,nj->inkm", G_normalized, weights)

    # 3. Transform vertices
    V_posed = G_blended @ V_template
    return V_posed
```

---

## 6. Multi-Stage Optimization

### 6.1 Stage 0: Global Alignment

```python
def solve_step0(params, target, max_iters=20):
    """
    목적: 전체 위치/자세 정렬
    최적화: R, T, s만 (thetas, bone_lengths 고정)
    """
    params["thetas"].requires_grad_(False)
    params["bone_lengths"].requires_grad_(False)
    params["chest_deformer"].requires_grad_(False)

    optimizer = LBFGS(params.values())
    # ... optimization loop
```

### 6.2 Stage 1: Pose Refinement

```python
def solve_step1(params, target, max_iters=50):
    """
    목적: Joint angles 최적화
    최적화: R, T, s, thetas, bone_lengths
    """
    params["thetas"].requires_grad_(True)
    params["bone_lengths"].requires_grad_(True)

    optimizer = LBFGS(params.values())
    # ... optimization loop
```

### 6.3 Stage 2: Full Optimization

```python
def solve_step2(params, target, max_iters=100):
    """
    목적: 전체 파라미터 fine-tuning
    최적화: 모든 파라미터 + chest_deformer
    추가: Mask loss, Stretch constraints
    """
    params["chest_deformer"].requires_grad_(True)
    term_weights["mask"] = 0.5  # Enable mask loss
    term_weights["stretch"] = 1.0
```

---

## 7. 2D-3D Projection

### 7.1 Camera Model

```python
def calc_2d_keypoint_loss(keypoints_3d, target_2d):
    """
    keypoints_3d: [B, 22, 3] - 3D keypoints
    target_2d: [B, N_cams, 22, 3] - 2D keypoints + confidence
    """
    for cam_id in range(N_cameras):
        # 3D → Camera coordinates
        J3d_cam = R[cam_id] @ keypoints_3d + T[cam_id]

        # Camera → Image coordinates (pinhole projection)
        J2d = K[cam_id] @ J3d_cam
        J2d = J2d / J2d[..., 2]  # Normalize by Z

        # Loss with confidence weighting
        loss += ||J2d - target_2d|| × confidence × keypoint_weight
```

### 7.2 Differentiable Silhouette Rendering

```python
# PyTorch3D SoftSilhouetteShader
raster_settings = RasterizationSettings(
    image_size=(1024, 1152),
    blur_radius=log(1/1e-4 - 1) × 3e-5,
    faces_per_pixel=50
)

renderer_mask = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings),
    shader=SoftSilhouetteShader()
)

# Mask loss
rendered_mask = renderer_mask(mesh, cameras)
loss_mask = MSE(rendered_mask, target_mask)
```

---

## 8. Output Structure

```
results/fitting/
├── obj/
│   ├── frame_0000.obj    # 3D mesh (Blender/MeshLab 호환)
│   └── ...
├── params/
│   ├── frame_0000.pkl    # 최적화된 파라미터
│   └── ...
└── render/
    ├── frame_0000_view_0.png  # 시각화
    └── ...
```

### Parameter PKL 내용:
```python
{
    "thetas": [140, 3],         # Joint rotations
    "bone_lengths": [19],       # Bone scaling
    "rotation": [3],            # Global rotation
    "trans": [3],               # Global translation
    "scale": [1],               # Global scale
    "chest_deformer": [1]       # Chest deformation
}
```

---

## 9. Key References

1. **MAMMAL Framework**: An et al. (2023) - "Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL"
2. **Virtual Mouse Model**: Bolanos et al. (2021) - Nature Methods
3. **Linear Blend Skinning**: Lewis et al. (2000)
4. **PyTorch3D**: Meta AI Research

---

## 10. 핵심 코드 위치

| 기능 | 파일 | 함수/클래스 |
|-----|------|------------|
| Mesh Fitting | `fitter_articulation.py` | `MouseFitter` |
| Body Model | `articulation_th.py` | `ArticulationTorch` |
| Forward Kinematics | `articulation_th.py:333` | `forward()` |
| Skinning | `articulation_th.py:311` | `skinning()` |
| Loss Functions | `fitter_articulation.py:278` | `gen_closure()` |
| Optimization | `fitter_articulation.py:351` | `solve_step0/1/2()` |
