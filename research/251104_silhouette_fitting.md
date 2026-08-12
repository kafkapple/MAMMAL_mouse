# 251104 연구노트 — Silhouette 기반 Fitting

## 목표
- Keypoint 우회, SAM mask 기반 직접 fitting 구현
- PyTorch3D differentiable silhouette renderer 구축
- 기존 fitting 결과의 근본적 한계 분석 및 개선

## 진행 내용

### 1. Phase 1: Silhouette Renderer 구현

**파일**: `preprocessing_utils/silhouette_renderer.py`

**구현 컴포넌트**:
- `SilhouetteRenderer`: PyTorch3D MeshRenderer + SoftSilhouetteShader
- `SilhouetteLoss`: IoU, BCE, Dice loss (combined weighted)
- Helper 함수: mask 로딩, 시각화 (Green=target, Red=pred, Yellow=overlap)

**기술적 해결 사항**:

| 이슈 | 원인 | 해결 |
|------|------|------|
| Camera T shape mismatch | OpenCV T (3,1) vs PyTorch3D (N,3) | `.squeeze().unsqueeze(0)` |
| Bin size overflow warning | Rasterization 설정 | `max_faces_per_bin` 증가 고려 |
| BodyModel attribute 혼동 | `BodyModelTorch` vs `ArticulationTorch` | `ArticulationTorch` + `faces_vert_np` 사용 |

**Rasterization Settings**: image_size=(480, 640), blur_radius=log(1/1e-4 - 1)*1e-5, faces_per_pixel=50

### 2. 현재 Fitting 품질 검증 (Frame 0)

```
Predicted silhouette coverage: 1.15%
Target mask coverage: 17.78%
IoU: 0.0000 (완전 실패)
BCE Loss: 18.7151
```

**시각화 분석**: mesh가 중앙에 작고 수직 (생쥐와 완전히 다른 위치)

**근본 원인**: Geometric keypoint 추정 실패 -> 잘못된 초기화 -> local minimum

### 3. SAM Mask 반전 문제 발견 및 수정

`mask_processing.py`의 `extract_mouse_mask()` 함수가 아레나 내부 원형 공간(18.92%)을 선택. 실제 필요한 것은 생쥐 + 배경(81.08%).

**수정**: `silhouette_renderer.py:load_target_mask()`에 mask inversion 추가
```python
mask = 1.0 - mask  # Critical fix
```

**결과**: Target coverage 18.92% -> 82.22%, IoU 0.0000 -> 0.0139

### 4. 2-Stage Fitting 프로토타입

#### Approach 1: From-scratch Initialization -- 실패
- `fit_silhouette_prototype.py`: translation + scale만 최적화 후 pose refinement
- 결과: IoU 0.0001 고착 (neutral pose 초기화가 너무 불량, optimization landscape가 flat)

#### Approach 2: Refinement from Existing Params -- 부분 성공
- `refine_with_silhouette.py`: 기존 keypoint-based fitting 결과를 초기값으로 사용

**Hyperparameters**:
```
ITERATIONS = 300
LR_TRANS = 0.5, LR_SCALE = 0.05, LR_ROTATION = 0.01, LR_POSE = 0.0001
Loss = iou_loss + 0.1 * bce_loss + 0.001 * pose_reg + 0.0001 * bone_reg
```

**결과: 93.2% IoU 개선**:

| Metric | Initial | Refined | 변화 |
|--------|---------|---------|------|
| IoU | 0.0139 | 0.0269 | **+93.2%** |
| BCE Loss | 81.07 | 73.76 | -9.0% |
| Coverage | 1.25% | 2.20% | +76.0% |

**IoU 수렴 과정** (plateau 없이 consistent improvement):
- Iter 50: 0.0152 (+9.4%)
- Iter 150: 0.0184 (+32.4%)
- Iter 300: 0.0268 (+92.8%)

### 5. 근본 원인 분석

**목표 대비 gap**:
- IoU: 0.0269 vs 목표 0.5-0.7 (18-26배 차이)
- Coverage: 2.20% vs target 82.22% (37배 차이)

**Scale parameter 문제**: 1.0 -> 1.1로만 증가 (10%), 37배(3700%) 필요. Pose regularization이 scale 변화 억제.

**Mesh 크기**: target의 2.7% 수준 (30배 작음). 기존 keypoint fitting의 치명적 잘못된 초기화가 refinement 한계를 결정.

## 핵심 발견

- **Differentiable rendering pipeline 작동 확인**: PyTorch3D 기반 end-to-end gradient flow 성공
- **SAM mask 신뢰성**: 100% detection rate, keypoint estimation보다 훨씬 정확한 supervision signal
- **Keypoint-based fitting의 근본적 한계**: Geometric keypoint 실패 -> 잘못된 초기화 -> refinement만으로 극복 불가
- **Learning rate hierarchy 중요**: Translation > Scale > Rotation > Pose 순서로 학습률 차등 적용 필요
- **초기화가 최적화보다 중요**: From-scratch은 완전 실패, 기존 결과 기반 refinement만 부분 성공

## 미해결 / 다음 단계

**단기 제안**:
- Option 1: Aggressive optimization (LR_SCALE 0.05 -> 0.5, ITERATIONS 300 -> 1000)
- Option 2: Multi-scale approach (coarse 0.1x -> medium 0.5x -> fine 1.0x)
- Option 3: Bounding box initialization (SAM mask에서 bbox 추출 -> init_translation/scale)

**중기 제안**:
- Keypoint-free fitting (SAM mask만으로 end-to-end)
- Learning-based initialization (CNN/ViT로 mask -> pose 예측)

**장기 제안**:
- Temporal consistency loss (4D reconstruction)
- Multi-view integration

---
*Sources: 251104_silhouette_fitting_progress.md, 251104_silhouette_fitting_final.md*
