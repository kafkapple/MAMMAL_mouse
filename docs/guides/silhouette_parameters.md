# Silhouette-Only Fitting 파라미터 가이드

## 개요

Keypoint annotation 없이 mask silhouette만으로 mesh fitting 시 사용하는 파라미터들의 상세 가이드입니다.

---

## 파라미터 요약표

| 파라미터 | 기본값 | 권장 범위 | 효과 |
|----------|--------|----------|------|
| `iter_multiplier` | 2.0 | 1.0 ~ 5.0 | 최적화 반복 횟수 배율 |
| `theta_weight` | 10.0 | 5.0 ~ 30.0 | 포즈 정규화 강도 |
| `bone_weight` | 2.0 | 0.5 ~ 5.0 | 뼈대 길이 정규화 |
| `scale_weight` | 50.0 | 10.0 ~ 100.0 | 스케일 정규화 |
| `use_pca_init` | true | true/false | PCA 기반 회전 초기화 |

---

## 상세 설명

### 1. `iter_multiplier` (반복 횟수 배율)

```yaml
silhouette:
  iter_multiplier: 2.0  # 기본값
```

**설명**: 각 최적화 단계의 반복 횟수를 배율로 조절

**영향**:
- 높을수록: 더 정밀한 수렴, 더 긴 실행 시간
- 낮을수록: 빠른 실행, 덜 정밀한 결과

**권장 설정**:
```
디버그/테스트: 1.0 ~ 2.0
일반 사용: 2.0 ~ 3.0
고품질 필요: 3.0 ~ 5.0
```

**실험 범위**:
```bash
# Grid search 예시
for mult in 1.0 1.5 2.0 2.5 3.0 4.0 5.0; do
  python fitter_articulation.py ... silhouette.iter_multiplier=$mult
done
```

---

### 2. `theta_weight` (포즈 정규화)

```yaml
silhouette:
  theta_weight: 10.0  # 기본값 (keypoint 모드는 3.0)
```

**설명**: 관절 각도(theta)의 정규화 강도. 높을수록 기본 포즈에 가깝게 유지.

**영향**:
- 높을수록: 안정적인 포즈, 표현력 감소
- 낮을수록: 자유로운 포즈, 비현실적 포즈 위험

**권장 설정**:
```
일반적인 행동: 10.0 ~ 15.0
정적인 장면: 15.0 ~ 20.0
활발한 움직임: 5.0 ~ 10.0
```

**주의**: Mask만 사용 시 pose ambiguity가 높아 keypoint 모드보다 높은 값 필요

---

### 3. `bone_weight` (뼈대 길이 정규화)

```yaml
silhouette:
  bone_weight: 2.0  # 기본값 (keypoint 모드는 0.5)
```

**설명**: 뼈대 길이가 기본 템플릿에서 벗어나지 않도록 제약

**영향**:
- 높을수록: 일관된 체형 유지
- 낮을수록: 체형 변형 허용

**권장 설정**:
```
일반: 2.0 ~ 3.0
체형 변화가 큰 경우: 0.5 ~ 1.0
체형 고정 필요: 3.0 ~ 5.0
```

---

### 4. `scale_weight` (스케일 정규화)

```yaml
silhouette:
  scale_weight: 50.0  # 기본값 (keypoint 모드는 0.5)
```

**설명**: 전체 mesh 크기가 급격히 변하지 않도록 제약

**영향**:
- 높을수록: 안정적인 크기 유지
- 낮을수록: mesh collapse 위험 (매우 작아지거나 커짐)

**권장 설정**:
```
필수: 30.0 이상 (silhouette 모드에서)
안전: 50.0 ~ 100.0
```

**주의**: Silhouette 모드에서 이 값이 낮으면 mesh가 축소되어 사라지는 현상 발생

---

### 5. `use_pca_init` (PCA 초기화)

```yaml
silhouette:
  use_pca_init: true  # 기본값
```

**설명**: 첫 프레임에서 mask contour의 PCA 분석으로 초기 회전 추정

**영향**:
- true: 마우스 body axis 자동 추정, 빠른 수렴
- false: 기본 orientation 사용, 초기 수렴 느림

**권장 설정**:
```
일반: true
PCA 실패 시: false (수동 초기화 필요할 수 있음)
```

---

## 실험 설계 가이드

### 단일 변수 실험 (Ablation)

각 파라미터의 개별 효과 측정:

```bash
# Baseline
./run_mesh_fitting_default.sh 0 100 -- --keypoints none \
    --input_dir /path/to/data

# iter_multiplier 변화
for mult in 1.0 2.0 3.0 4.0 5.0; do
  ./run_mesh_fitting_default.sh 0 100 -- --keypoints none \
      --input_dir /path/to/data \
      silhouette.iter_multiplier=$mult
done

# theta_weight 변화
for theta in 5.0 10.0 15.0 20.0 30.0; do
  ./run_mesh_fitting_default.sh 0 100 -- --keypoints none \
      --input_dir /path/to/data \
      silhouette.theta_weight=$theta
done
```

### Grid Search 실험

주요 파라미터 조합 탐색:

```bash
# iter_multiplier x theta_weight
for mult in 2.0 3.0; do
  for theta in 10.0 15.0 20.0; do
    echo "Testing mult=$mult, theta=$theta"
    ./run_mesh_fitting_default.sh 0 50 -- --keypoints none \
        --input_dir /path/to/data \
        silhouette.iter_multiplier=$mult \
        silhouette.theta_weight=$theta
  done
done
```

---

## 문제 해결

### Mesh가 보이지 않음 (Collapse)
```bash
# scale_weight 증가
silhouette.scale_weight=100.0
```

### 포즈가 비현실적
```bash
# theta_weight 증가
silhouette.theta_weight=20.0
silhouette.bone_weight=3.0
```

### 수렴이 느림
```bash
# iter_multiplier 증가 + PCA 활성화
silhouette.iter_multiplier=3.0
silhouette.use_pca_init=true
```

### 초기 회전이 틀림
```bash
# PCA 비활성화 후 수동 조정 또는 더 많은 프레임 실행
silhouette.use_pca_init=false
# 또는 프레임 범위 확대 (temporal propagation 활용)
fitter.start_frame=0 fitter.end_frame=100
```

---

## 관련 문서

- [연구 노트: 251127_silhouette_only_fitting.md](../notes/251127_silhouette_only_fitting.md)
- [README.md - Silhouette-Only Fitting](../../README.md#-silhouette-only-fitting-keypoint-없이-마스크만-사용)
