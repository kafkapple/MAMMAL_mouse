# Silhouette-based Mesh Fitting 계획서

**날짜**: 2025-11-04
**목표**: Keypoint 없이 SAM mask만으로 3D mesh를 영상에 정렬

---

## 문제 진단

### 현재 상태
- ✅ **SAM mask**: 고품질 (18.9% coverage, 100% detection)
- ❌ **Geometric keypoint**: 부정확 (중앙에 몰려있음)
- ❌ **Fitting 결과**: 엉뚱한 위치의 mesh

### 근본 원인
Geometric keypoint 추정 알고리즘이 PCA 기반으로 부정확:
- Head/tail 구분 실패
- Limb position 추정 실패
- 결과적으로 optimization이 잘못된 초기값에서 시작

---

## 해결 방법: Silhouette-based Direct Fitting

### 핵심 아이디어
**Keypoint를 우회하고 SAM mask를 직접 loss로 사용**

```
SAM Mask (Ground Truth Silhouette)
        ↓
PyTorch3D Differentiable Renderer
        ↓
Rendered Silhouette
        ↓
Silhouette Loss (IoU / BCE)
        ↓
Backprop → Optimize (pose, trans, scale)
```

### 이론적 배경
- **SMAL/SMALR** (CVPR 2018): 동물 silhouette fitting
- **SMPLify** (ECCV 2016): Human body fitting
- **PyTorch3D Tutorial**: Mesh fitting with silhouette loss

---

## 구현 계획 (2-4시간)

### Phase 1: Silhouette Renderer 구축 (30분)

**목표**: PyTorch3D로 mouse mesh의 silhouette 렌더링

**파일**: `silhouette_renderer.py`

```python
class SilhouetteRenderer:
    def __init__(self, image_size, device):
        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-5,
            faces_per_pixel=10
        )

        # Silhouette shader
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings),
            shader=SoftSilhouetteShader()
        )

    def render(self, meshes, cameras):
        # Returns alpha channel (0-1)
        silhouettes = self.renderer(meshes, cameras=cameras)
        return silhouettes[..., 3]  # Alpha channel
```

**테스트**: 초기 mesh를 렌더링해서 silhouette 확인

---

### Phase 2: Silhouette Loss 구현 (30분)

**목표**: SAM mask와 rendered silhouette 비교

**Metric 1: IoU Loss**
```python
def silhouette_iou_loss(predicted, target):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)
    return 1.0 - iou
```

**Metric 2: BCE Loss**
```python
def silhouette_bce_loss(predicted, target):
    return F.binary_cross_entropy(predicted, target)
```

**Combined Loss**
```python
loss = 0.5 * iou_loss + 0.5 * bce_loss
```

---

### Phase 3: 2-Stage Optimization (1-2시간)

#### Stage 1: Global Alignment (빠른 초기화)
**목표**: Translation + Scale만 최적화 (pose 고정)

```python
# Optimize variables
translation = torch.tensor([0., 0., 500.], requires_grad=True)
scale = torch.tensor([1.0], requires_grad=True)

# Fixed initial pose (neutral)
pose = initial_neutral_pose.clone()

# Optimizer
optimizer = torch.optim.Adam([translation, scale], lr=0.1)

# Iterations: 50-100
for i in range(100):
    mesh = bodymodel.forward(pose, trans=translation, scale=scale)
    rendered_sil = renderer.render(mesh, camera)
    loss = silhouette_loss(rendered_sil, sam_mask)
    loss.backward()
    optimizer.step()
```

**예상 결과**: Mesh가 대략적으로 생쥐 위치로 이동

---

#### Stage 2: Pose Refinement
**목표**: Pose parameters도 최적화 (limb 정렬)

```python
# Optimize variables
translation = stage1_translation.clone().requires_grad_(True)
scale = stage1_scale.clone().requires_grad_(True)
pose = stage1_pose.clone().requires_grad_(True)

# Optimizer with smaller learning rate
optimizer = torch.optim.Adam([
    {'params': [translation], 'lr': 0.01},
    {'params': [scale], 'lr': 0.01},
    {'params': [pose], 'lr': 0.001}  # Smaller for stability
], lr=0.01)

# Regularization
def pose_prior_loss(pose):
    # Keep pose close to neutral
    return ((pose - neutral_pose) ** 2).mean()

# Combined loss
for i in range(200):
    mesh = bodymodel.forward(pose, trans=translation, scale=scale)
    rendered_sil = renderer.render(mesh, camera)

    sil_loss = silhouette_loss(rendered_sil, sam_mask)
    prior_loss = pose_prior_loss(pose)

    total_loss = sil_loss + 0.1 * prior_loss
    total_loss.backward()
    optimizer.step()
```

**예상 결과**: Mesh가 생쥐의 pose에 맞게 변형

---

### Phase 4: 기존 코드 통합 (1시간)

**목표**: `fitter_articulation.py`에 silhouette loss 추가

**변경사항**:

1. **초기화 개선**:
```python
# solve_step0에서 silhouette-based initialization
def initialize_from_silhouette(self, mask, camera):
    """2-stage silhouette fitting"""
    # Stage 1: Global alignment
    trans, scale = self.fit_global_alignment(mask, camera)

    # Stage 2: Pose refinement
    pose = self.fit_pose_refinement(mask, camera, trans, scale)

    return {'trans': trans, 'scale': scale, 'pose': pose}
```

2. **Loss term 추가**:
```python
# 기존 loss에 silhouette term 추가
self.term_weights["silhouette"] = 1.0  # 초기에는 높게

def compute_silhouette_loss(self, params, target_mask, camera):
    mesh = self.bodymodel.forward(...)
    rendered_mask = self.silhouette_renderer.render(mesh, camera)
    return self.silhouette_iou_loss(rendered_mask, target_mask)
```

3. **Optimization loop 수정**:
```python
# Step 0: Silhouette-based initialization
if iter == 0:
    params = self.initialize_from_silhouette(mask, camera)

# All steps: Include silhouette loss
sil_loss = self.compute_silhouette_loss(params, mask, camera)
total_loss += self.term_weights["silhouette"] * sil_loss
```

---

## 예상 결과

### 수치적 개선
| Metric | 현재 (Keypoint) | 예상 (Silhouette) |
|--------|-----------------|-------------------|
| **Mask IoU** | ~0.1 (mesh 엉뚱한 곳) | ~0.7-0.8 |
| **Position Error** | 수백 픽셀 | <50 픽셀 |
| **Convergence** | 거의 없음 | 100-300 iterations |

### 정성적 개선
- ✅ Mesh가 실제 생쥐 위치에 배치
- ✅ 대략적인 body orientation 정렬
- ⚠️ Limb detail은 제한적 (single view limitation)

---

## 대안 방법들 (비교)

### Option A: DeepLabCut SuperAnimal (기각)
- **장점**: SOTA keypoint detection (26 keypoints)
- **단점**:
  - TensorFlow dependency 문제 (tensorpack 누락)
  - 환경 충돌 위험
  - 추가 설치 필요 (1-2시간)
- **결론**: ❌ 시간 대비 불확실성 높음

### Option B: ViTPose (기각)
- **장점**: PyTorch 기반, 80.4 mAP on AP-10K
- **단점**:
  - 완전히 새로 설치
  - 모델 다운로드 + fine-tuning 필요
  - 3-4시간 소요
- **결론**: ❌ 시간 초과

### Option C: Silhouette-based Fitting (채택) ✅
- **장점**:
  - PyTorch3D 이미 설치됨
  - SAM mask 이미 준비됨
  - Keypoint 우회 가능
  - 2-4시간 구현 가능
- **단점**:
  - Single view 한계
  - Limb detail 부족할 수 있음
- **결론**: ✅ **가장 빠르고 확실한 방법**

---

## 단계별 실행 계획

### Step 1: Silhouette Renderer 구현 (30분)
```bash
# 파일 생성
preprocessing_utils/silhouette_renderer.py

# 테스트
python test_silhouette_renderer.py
```

### Step 2: 2-Stage Fitting 프로토타입 (1시간)
```bash
# 독립 스크립트로 먼저 검증
fit_silhouette_prototype.py

# 1 프레임 테스트
python fit_silhouette_prototype.py --frame 0
```

### Step 3: 기존 코드 통합 (1시간)
```bash
# fitter_articulation.py 수정
# solve_step0에 silhouette initialization 추가
# Loss에 silhouette term 추가
```

### Step 4: 검증 (30분)
```bash
# 10 프레임 테스트
export PYOPENGL_PLATFORM=egl && \
conda run -n mammal_stable python fitter_articulation.py \
  fitter.end_frame=10 \
  fitter.with_render=true
```

---

## 성공 기준

### 필수 (Must Have)
- ✅ Mesh가 생쥐 위치에 배치 (중앙이 아닌 실제 위치)
- ✅ Mask IoU > 0.5
- ✅ 10 프레임 피팅 성공

### 선택 (Nice to Have)
- Mask IoU > 0.7
- Limb orientation 대략적 일치
- Temporal consistency across frames

---

## 위험 요소 및 대응

### 위험 1: Camera parameter 불일치
- **증상**: Rendered size가 target과 다름
- **대응**: Camera intrinsics를 new_cam.pkl에서 정확히 로드
- **검증**: 초기 neutral pose 렌더링 크기 확인

### 위험 2: Local minima
- **증상**: Optimization이 수렴하지 않음
- **대응**:
  - Multi-scale optimization (큰 blur → 작은 blur)
  - Multiple random initialization
  - Learning rate scheduling

### 위험 3: Overfitting to silhouette
- **증상**: 내부 구조 무시하고 외형만 맞춤
- **대응**:
  - Pose prior regularization 강화
  - Bone length constraint 추가
  - 기존 keypoint loss와 병행 (가중치 낮게)

---

## 타임라인

| 단계 | 소요 시간 | 누적 시간 |
|------|----------|----------|
| Silhouette Renderer 구현 | 30분 | 0.5시간 |
| 2-Stage Fitting 프로토타입 | 1시간 | 1.5시간 |
| 기존 코드 통합 | 1시간 | 2.5시간 |
| 테스트 및 디버깅 | 30분 | 3시간 |
| **총계** | **3시간** | - |

**여유 포함**: 4시간 이내 완료 목표

---

## 결론

**Silhouette-based fitting이 현재 상황에서 최선의 선택**:
1. 이미 갖춘 요소 활용 (PyTorch3D + SAM)
2. Keypoint 추정 실패 우회
3. 3-4시간 내 구현 가능
4. 즉각적인 품질 개선 기대

**다음 단계**: 사용자 승인 후 Step 1부터 순차 진행

---

**작성**: Claude (Anthropic)
**검토 요청**: 계획 승인 후 즉시 구현 시작
