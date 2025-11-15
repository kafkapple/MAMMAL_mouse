# Silhouette-based Fitting 진행 상황

**날짜**: 2025-11-04
**목표**: Keypoint 우회, SAM mask 기반 직접 fitting

---

## 진행 상황 요약

### ✅ Phase 1 완료: Silhouette Renderer 구현 (1시간)

#### 구현 내용
1. **`preprocessing_utils/silhouette_renderer.py`**
   - `SilhouetteRenderer`: PyTorch3D 기반 differentiable renderer
   - `SilhouetteLoss`: IoU, BCE, Dice loss 구현
   - Helper 함수: mask 로딩, 시각화

2. **테스트 스크립트**
   - `test_silhouette_simple.py`: 기존 fitting 결과 검증
   - PyTorch3D camera setup 완료
   - Mesh rendering 성공

#### 검증 결과

**현재 Fitting 품질** (Frame 0):
```
Predicted silhouette coverage: 1.15%
Target mask coverage: 17.78%
IoU: 0.0000 (완전 실패)
BCE Loss: 18.7151
```

**시각화 분석** (`test_silhouette_comparison.png`):
- 초록색: SAM mask (실제 생쥐 위치 - 왼쪽 위)
- 빨간색: 현재 fitted mesh (중앙, 작고 수직)
- 노란색 (overlap): 없음

**문제 진단**:
- Mesh가 생쥐와 완전히 다른 위치
- Geometric keypoint 추정 실패로 인한 초기화 문제
- Optimization이 local minimum에 빠짐

#### 기술적 해결 사항

1. **Camera Format 변환**
   ```python
   # OpenCV camera (R, T) → PyTorch3D format
   R = torch.from_numpy(R_cam).float().unsqueeze(0)
   T = torch.from_numpy(T_cam).float().squeeze().unsqueeze(0)  # (1, 3)
   ```

2. **Mesh Faces 로딩**
   ```python
   faces = torch.from_numpy(bodymodel.faces_vert_np).long()
   ```

3. **Rasterization Settings**
   ```python
   RasterizationSettings(
       image_size=(480, 640),
       blur_radius=np.log(1. / 1e-4 - 1.) * 1e-5,
       faces_per_pixel=50
   )
   ```

---

## 📋 다음 단계: Phase 2 (진행 예정)

### Step 2: 2-Stage Fitting 프로토타입 (예상 1시간)

#### Stage 1: Global Alignment
**목표**: Translation + Scale만 최적화

```python
# Variables
translation = torch.tensor([0., 0., 500.], requires_grad=True)
scale = torch.tensor([1.0], requires_grad=True)

# Fixed
pose = neutral_pose.clone()

# Optimizer
optimizer = torch.optim.Adam([translation, scale], lr=0.1)

# Loss
for iter in range(100):
    mesh = bodymodel.forward(pose, trans=translation, scale=scale)
    silhouette = renderer.render(mesh, camera)
    loss = silhouette_loss(silhouette, sam_mask)
    loss.backward()
    optimizer.step()
```

**예상 결과**: Mesh가 대략적으로 생쥐 위치로 이동

#### Stage 2: Pose Refinement
**목표**: Pose parameters 추가 최적화

```python
# All variables
translation = stage1_trans.requires_grad_(True)
scale = stage1_scale.requires_grad_(True)
pose = stage1_pose.requires_grad_(True)

# Optimizer with smaller LR
optimizer = torch.optim.Adam([
    {'params': [translation], 'lr': 0.01},
    {'params': [scale], 'lr': 0.01},
    {'params': [pose], 'lr': 0.001}
])

# Combined loss
for iter in range(200):
    mesh = bodymodel.forward(pose, trans, scale, ...)
    silhouette = renderer.render(mesh, camera)

    sil_loss = silhouette_loss(silhouette, sam_mask)
    prior_loss = pose_prior(pose)  # Regularization

    total_loss = sil_loss + 0.1 * prior_loss
    total_loss.backward()
    optimizer.step()
```

**예상 결과**: Mesh가 생쥐 pose에 맞게 변형

---

## 🎯 성공 기준

### 필수 (Must Have)
- ✅ Silhouette renderer 작동 확인
- ⏳ Mesh가 SAM mask 위치로 이동
- ⏳ IoU > 0.5
- ⏳ 시각적으로 생쥐와 mesh 겹침

### 선택 (Nice to Have)
- IoU > 0.7
- Limb orientation 대략적 일치
- Temporal consistency

---

## 📊 예상 개선 효과

| Metric | Before (현재) | After (예상) |
|--------|--------------|-------------|
| **IoU** | 0.000 | 0.5-0.7 |
| **BCE Loss** | 18.7 | <5.0 |
| **Mesh Position** | 중앙 (엉뚱한 곳) | 생쥐 위치 |
| **Coverage** | 1.15% | 15-20% |

---

## 🔧 현재 파일 구조

```
preprocessing_utils/
├── silhouette_renderer.py  ✅ (완료)
│   ├── SilhouetteRenderer
│   ├── SilhouetteLoss
│   └── Helper functions

test_silhouette_simple.py  ✅ (검증 완료)
└── 현재 fitting IoU = 0.0 확인

reports/
├── silhouette_fitting_plan.md  ✅ (계획서)
└── silhouette_fitting_progress_20251104.md  ✅ (이 파일)
```

---

## ⏱️ 타임라인

| 단계 | 예상 시간 | 상태 | 실제 시간 |
|------|----------|------|----------|
| Phase 1: Renderer 구현 | 30분 | ✅ 완료 | ~1시간 |
| Phase 2: 2-Stage Fitting | 1시간 | 🔄 다음 | - |
| Phase 3: 기존 코드 통합 | 1시간 | ⏳ 대기 | - |
| Phase 4: 테스트 & 검증 | 30분 | ⏳ 대기 | - |
| **총계** | **3시간** | - | **~1시간** |

---

## 🐛 이슈 및 해결

### Issue 1: Camera T shape mismatch
- **증상**: `Expected T to have shape (N, 3); got torch.Size([1, 3, 1])`
- **원인**: OpenCV T shape (3, 1) vs PyTorch3D (N, 3)
- **해결**: `.squeeze().unsqueeze(0)` 적용
- **코드**: `T = torch.from_numpy(T_cam).float().squeeze().unsqueeze(0)`

### Issue 2: Bin size overflow warning
- **증상**: "Bin size was too small in coarse rasterization"
- **영향**: 무시 가능 (rendering은 정상 작동)
- **해결**: 향후 `max_faces_per_bin` 증가 고려

### Issue 3: BodyModel attribute 혼동
- **증상**: `num_q`, `faces` attribute 없음
- **원인**: `BodyModelTorch` vs `ArticulationTorch` 혼동
- **해결**: `ArticulationTorch` 사용, `faces_vert_np` 사용

---

## 📝 Ground Truth 정의

**질문**: "실제 생쥐 위치 GT는 어떻게 알고 비교하는가?"

**답변**:
1. **Real GT** (수동 annotation) = 없음
2. **Pseudo-GT** (자동 생성) = **SAM mask**
   - SAM이 실제 생쥐를 정확히 감지 (100% detection)
   - 시각화로 검증 가능 (frame_000020.png)
   - 18.9% coverage = 적절한 생쥐 크기

3. **검증 방법**:
   - SAM visualization으로 육안 확인
   - 실제 영상에서 생쥐 위치와 SAM mask 일치 확인
   - Keypoints가 생쥐 body를 따라 분포

4. **비교 구조**:
   ```
   실제 영상 → SAM → SAM Mask (Pseudo-GT, 초록색)
                          ↓ (목표: 정렬)
   Fitting → Mesh Silhouette (현재, 빨간색)
   ```

---

## 💡 핵심 인사이트

1. **SAM mask의 신뢰성**
   - 100% detection rate
   - 시각적 검증 완료
   - Keypoint estimation보다 훨씬 정확

2. **현재 fitting의 근본 문제**
   - Geometric keypoint 추정 실패
   - 잘못된 초기화 → 잘못된 수렴
   - Keypoint-based loss는 무의미

3. **Silhouette-based 접근의 장점**
   - Keypoint 우회
   - 직접 mask 정렬
   - Differentiable하여 end-to-end 학습 가능

---

## 다음 세션 준비사항

**계속 진행 시**:
1. `fit_silhouette_prototype.py` 작성
2. Stage 1 (Global Alignment) 구현 및 테스트
3. Stage 2 (Pose Refinement) 구현 및 테스트

**필요한 것**:
- 시간: 약 2시간
- 현재 코드베이스 그대로 사용 가능

---

**작성**: Claude (Anthropic)
**상태**: Phase 1 완료, Phase 2 준비 완료
