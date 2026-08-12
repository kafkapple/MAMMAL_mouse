# 251125 연구노트 — Auto Keypoint Pipeline 설계 및 Fitting 메커니즘 분석

## 목표
- Keypoint 없는 dataset 대상 자동 추정 파이프라인 아키텍처 설계
- MAMMAL mesh fitting의 FK, LBS, 3단계 최적화 메커니즘 상세 분석 (발표 준비용)

## 진행 내용

### 1. Keypoint Pipeline 아키텍처 설계

#### 기존 구현 현황

| 모듈 | 파일 | 상태 | Keypoints |
|------|------|------|-----------|
| Geometric (PCA) | `keypoint_estimation.py` | 완료 | 22 (mask 기반, no training) |
| SuperAnimal/DLC | `superanimal_detector.py` | 부분 | 27 (Nature Comm. 2024) |
| YOLO-Pose | `yolo_keypoint_detector.py` | 부분 | Custom |
| DANNCE→YOLO | `dannce_to_yolo.py` | 완료 | 데이터 변환 |

#### Pre-trained 모델 비교

| 모델 | Keypoints | 장점 | 단점 |
|------|-----------|------|------|
| **SuperAnimal-TopViewMouse** | 27 | Zero-shot 우수, fine-tuning 10-100x 효율 | DLC 환경 복잡, 27→22 매핑 필요 |
| **SuperAnimal-Quadruped** | 39 | 45+ 포유류 지원 | 범용적이라 마우스 정밀도 낮을 수 있음 |
| **YOLOv8-Pose** | Custom | 설치 간편, custom 학습 용이, 실시간 | 마우스 pretrained 없음 |
| **STPoseNet** | ~12 | YOLOv8 + Temporal + Kalman, 마우스 특화 | iScience 2024 |

#### 제안 아키텍처

```python
# 통합 인터페이스 (Factory pattern)
detector = KeypointDetector.create(
    backend="superanimal",  # "geometric", "yolo", "ensemble"
    config="configs/detectors/superanimal.yaml"
)
keypoints = detector.detect(image, mask=mask)  # Returns: (22, 3) [x, y, confidence]
```

모듈 구조: `mammal_keypoints/{detectors/, converters/, annotators/, training/}`

#### Keypoint Mapping: SuperAnimal (27) → MAMMAL (22)

- Head (0-5): nose, ears, eyes, head_center
- Spine (6-13): spine_1 ~ spine_8 (보간 가능)
- Limbs (14-17): 4 paws
- Tail (18-20): base, mid, tip
- Centroid (21): 전체 점 평균

#### Web Annotation 도구 비교

| 기능 | CVAT | Label Studio |
|------|------|-------------|
| Pose/Keypoint | 최적화 | 기본 지원 |
| Video 지원 | Interpolation 지원 | 기본 |
| ML 연동 | 제한적 | Active Learning 강력 |

**권장**: CVAT (Pose estimation 특화)

#### 구현 우선순위

1. **Phase 1** (1-2주): 통합 Detector interface + 기존 코드 리팩토링
2. **Phase 2** (1주): Gradio 기반 Web annotation (auto-detect + 수동 수정)
3. **Phase 3** (2주): YOLO/DLC fine-tuning pipeline
4. **Phase 4** (1주): pip installable package + Docker

### 2. MAMMAL Mesh Fitting 메커니즘 상세 분석

#### Mouse Model Spec

| 항목 | 값 |
|------|-----|
| Vertices | 14,522 |
| Joints | 140 |
| Keypoints | 22 (semantic landmarks) |
| Faces | ~7,200 (reduced) / 28,800 (full) |
| Bones | 19 (scaling factors) |

#### 22 Semantic Keypoints (mouse_model/keypoint22_mapper.json)

| Index | Type | 의미 |
|-------|------|------|
| 0 | V (vertex avg) | Nose |
| 1-2 | V | Left/Right ear |
| 3 | J (joint) | Neck |
| 5 | J | Tail base |
| 6-7 | J | Tail mid/tip |
| 8-15 | J | Forepaw/Ulna/Humerus (L/R) |
| 16-21 | J | Hindpaw/Tibia (L/R) |

#### Forward Kinematics Pipeline

```
Input Parameters → Rodrigues Rotation → Skeleton Hierarchy (G matrix) → LBS → Global Transform
```

1. **Rodrigues**: axis-angle [B, 140, 3] → rotation matrix [B, 140, 3, 3]
2. **G matrix**: `G[i] = G[parent[i]] @ S[i]` (chain multiplication)
3. **LBS**: `V_posed = sum(w_j * G_j * v_template)` (weighted blending)
4. **Global**: rotation R, translation T, scale s 적용

#### 최적화 파라미터

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `thetas` | [B, 140, 3] | Joint axis-angle rotations |
| `bone_lengths` | [B, 19] | Bone length scaling |
| `R` | [B, 3] | Global rotation (axis-angle) |
| `T` | [B, 3] | Global translation |
| `s` | [B, 1] | Global scale (~115) |
| `chest_deformer` | [B, 1] | Chest deformation |

#### Loss Functions

```python
Loss = w_2d * L_2d + w_theta * L_theta + w_bone * L_bone
     + w_scale * L_scale + w_mask * L_mask + w_stretch * L_stretch + w_temp * L_temporal
```

기본 weights: theta=3.0, 2d=0.2, bone=0.5, scale=0.5, mask=0 (default), stretch=1.0, temp=0.25

#### 3단계 최적화 상세

- **Step 0** (Global Alignment): R, T, s만, LBFGS, 20 iters
- **Step 1** (Pose Refinement): + thetas, bone_lengths, 50 iters
- **Step 2** (Full Optimization): + chest_deformer, mask loss (w=0.5), 100 iters

#### Differentiable Silhouette Rendering

```python
# PyTorch3D SoftSilhouetteShader
raster_settings = RasterizationSettings(
    image_size=(1024, 1152),
    blur_radius=log(1/1e-4 - 1) * 3e-5,
    faces_per_pixel=50
)
loss_mask = MSE(rendered_mask, target_mask)
```

## 핵심 발견
- MAMMAL 모델: 14,522 vertices, 140 joints, 22 keypoints — FK + LBS 기반
- Rodrigues → G matrix chain → LBS skinning 순서로 pose 적용
- 3단계 최적화: global alignment → pose → full (mask 포함)
- SuperAnimal-TopViewMouse가 마우스 keypoint auto-detection에 가장 적합한 후보
- 통합 인터페이스 설계로 detector backend 교체 용이하게 구현 가능

## 미해결 / 다음 단계
- Phase 1 구현: 통합 KeypointDetector 인터페이스
- SuperAnimal 27 → MAMMAL 22 매핑 테스트
- Web annotation server 구축 (Gradio + auto-detect)
- YOLO fine-tuning: MAMMAL 22 keypoints 직접 학습

---
*Sources: 251125_keypoint_pipeline_plan.md, 251125_mesh_fitting_mechanism.md*
