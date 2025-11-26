# 251127 Silhouette-Only Mesh Fitting 연구 노트

## 개요

- **날짜**: 2025-11-27
- **주제**: Keypoint annotation 없이 mask silhouette만으로 3D mesh fitting
- **목적**: 2D keypoint labeling 없이도 multi-view mask만으로 정확한 mesh reconstruction 가능성 검증

---

## 배경

기존 MAMMAL 파이프라인:
```
Input: Multi-view video + 2D keypoints + Masks
       ↓
Output: 3D articulated mesh
```

문제점:
- 2D keypoint annotation은 시간 소모적
- 특히 tail, limbs 등 keypoint 정확도 낮음
- Mask (silhouette)는 자동 생성 가능 (SAM, SimpleClick 등)

---

## 구현 내용

### 1. Silhouette-Only Mode 활성화

**변경 사항** (`fitter_articulation.py`):

```python
# 기존: keypoint loss만 사용
term_weights["2d"] = keypoint_weight
term_weights["mask"] = 0  # Step 0, 1에서 비활성화

# 변경: keypoint 비활성화 시 mask loss 활성화
if not use_keypoints:
    term_weights["2d"] = 0
    term_weights["mask"] = 1000  # Step 0
    term_weights["mask"] = 1500  # Step 1
```

### 2. Mask 기반 초기화

**PCA Rotation Estimation**:
```python
def init_params_from_masks():
    # 1. Mask centroid로 초기 translation 추정
    # 2. Mask bounding box로 초기 scale 추정
    # 3. Mask contour PCA로 초기 rotation 추정 (body orientation)
```

### 3. 정규화 강화

Keypoint 없이 mask만 사용 시 ambiguity 증가 → 정규화 필요:

| 파라미터 | Keypoint 모드 | Silhouette 모드 |
|----------|--------------|-----------------|
| theta_weight | 3.0 | 10.0 |
| bone_weight | 0.5 | 2.0 |
| scale_weight | 0.5 | 50.0 |
| iter_multiplier | 1.0 | 2.0 |

---

## 실험 설계

### 4가지 비교 실험

| 실험 | 변경 요소 | 가설 |
|------|----------|------|
| exp1_baseline | 기준 (iter=2x, theta=10) | 비교 기준선 |
| exp2_more_iters | iter=3x | 반복 증가로 수렴 개선 |
| exp3_high_reg | theta=15, bone=3 | 강한 정규화로 안정성 향상 |
| exp4_no_pca | use_pca_init=false | PCA 초기화 효과 검증 |

### 실행 방법

```bash
# 디버그 (2 프레임)
./run_silhouette_experiments.sh /path/to/data 0 2

# 본 실험 (100 프레임)
./run_silhouette_experiments.sh /path/to/data 0 100
```

---

## 핵심 발견

### 1. Multi-view + Silhouette = Viable

- 6개 뷰의 mask silhouette만으로 reasonable한 mesh fitting 가능
- 단, keypoint 대비 정밀도는 낮음 (특히 사지 pose)

### 2. Temporal Propagation 효과

```
Frame 0 → Frame 1 → ... → Frame N
  ↓          ↓              ↓
초기값    이전결과      누적최적화
(불안정)  (개선)       (안정)
```

- 후반 프레임일수록 피팅 품질 향상
- 이전 프레임 결과가 다음 프레임 초기값으로 사용

### 3. 정규화의 중요성

- Mask만으로는 pose ambiguity 존재
- 강한 theta/bone 정규화로 unrealistic pose 방지
- scale 정규화로 mesh collapse 방지

---

## 파라미터 탐색 범위 (향후 실험)

### Iteration Multiplier
```
범위: 1.0 ~ 5.0
권장: 2.0 ~ 3.0
효과: 높을수록 수렴 정밀도 ↑, 시간 ↑
```

### Theta Weight (Pose Regularization)
```
범위: 5.0 ~ 30.0
권장: 10.0 ~ 15.0
효과: 높을수록 pose 안정 ↑, 표현력 ↓
```

### Bone Weight
```
범위: 0.5 ~ 5.0
권장: 2.0 ~ 3.0
효과: 높을수록 bone length 보존 ↑
```

### Scale Weight
```
범위: 10.0 ~ 100.0
권장: 50.0
효과: 낮으면 mesh collapse 위험
```

### Use PCA Init
```
값: true / false
권장: true
효과: 초기 orientation 추정에 도움
```

---

## 향후 계획

1. **정량 평가**: Keypoint 모드 vs Silhouette 모드 mesh accuracy 비교
2. **Ablation Study**: 각 파라미터별 sensitivity 분석
3. **Single-view 확장**: Monocular silhouette fitting 가능성 검토
4. **Temporal Smoothing**: 프레임 간 일관성 강화

---

## 관련 파일

- `fitter_articulation.py`: 메인 피팅 코드
- `conf/config.yaml`: silhouette 설정
- `run_silhouette_experiments.sh`: 실험 스크립트
- `README.md`: 사용법 문서

---

## 참고

- MAMMAL 원본 논문의 silhouette loss 구현 참조
- PCA rotation initialization은 OpenCV 기반
