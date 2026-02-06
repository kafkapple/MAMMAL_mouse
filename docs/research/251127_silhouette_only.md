# 251127 연구노트 — Silhouette-Only Mesh Fitting

## 목표
- Keypoint annotation 없이 multi-view mask silhouette만으로 3D mesh fitting 가능성 검증
- Temporal propagation 효과 확인
- 정규화 파라미터 설계 및 실험

## 진행 내용

### 1. Silhouette-Only Mode 구현

기존 keypoint 기반 loss를 비활성화하고, mask loss를 전 단계에서 활성화:

| 파라미터 | Keypoint 모드 | Silhouette 모드 |
|----------|--------------|-----------------|
| `term_weights["2d"]` | 활성화 (0.2) | **0** |
| `term_weights["mask"]` Step 0 | 0 | **1000** |
| `term_weights["mask"]` Step 1 | 0 | **1500** |
| `theta_weight` | 3.0 | **10.0** |
| `bone_weight` | 0.5 | **2.0** |
| `scale_weight` | 0.5 | **50.0** |
| `iter_multiplier` | 1.0 | **2.0** |

**핵심**: Mask loss가 Step 0부터 활성화되어야 초기 피팅 가능

### 2. Mask 기반 초기화

PCA Rotation Estimation으로 body orientation 추정:
```python
def init_params_from_masks():
    # 1. Mask centroid → 초기 translation
    # 2. Mask bounding box → 초기 scale
    # 3. Mask contour PCA → 초기 rotation (body orientation)
```

### 3. Temporal Propagation

```
Frame 0: 기본 초기값 (PCA init) → 불안정
  ↓ 결과 전달
Frame 1: Frame 0 최적화 결과로 시작 → 개선
  ↓
Frame N: 누적된 최적화 이점 → 안정
```

- 후반 프레임일수록 fitting 품질 향상
- 단, 초반 실패 시 오류 누적 위험

### 4. 정규화 강화의 필요성

Keypoint 없이 mask만 사용 시 pose ambiguity 증가:
- 강한 theta/bone 정규화 → unrealistic pose 방지
- **scale_weight 50.0 필수** (낮으면 mesh collapse 발생)

### 5. 4가지 비교 실험 설계

| 실험 | 변경 요소 | 목적 |
|------|----------|------|
| exp1_baseline | 기준 (iter=2x, theta=10) | 비교 기준선 |
| exp2_more_iters | iter=3x | 반복 증가 효과 |
| exp3_high_reg | theta=15, bone=3 | 정규화 강화 효과 |
| exp4_no_pca | use_pca_init=false | PCA 초기화 효과 |

**설계 원칙**: 한 번에 하나의 변수만 변경 (controlled experiment)

### 6. Config 구조

```yaml
# conf/config.yaml
silhouette:
  iter_multiplier: 2.0
  theta_weight: 10.0
  bone_weight: 2.0
  scale_weight: 50.0
  use_pca_init: true
```

### 7. 서버 호환성

```bash
# Python 경로 자동 감지 (run_silhouette_experiments.sh)
if [ -f "${HOME}/miniconda3/envs/mammal_stable/bin/python" ]; then
    PYTHON="${HOME}/miniconda3/envs/mammal_stable/bin/python"
elif [ -f "${HOME}/anaconda3/envs/mammal_stable/bin/python" ]; then
    PYTHON="${HOME}/anaconda3/envs/mammal_stable/bin/python"
fi
export PYOPENGL_PLATFORM=egl  # Headless rendering
```

**Hydra 인자 파싱 주의**: `$extra_args` 문자열 확장 시 파싱 오류 → 배열 방식으로 변경 필요

## 핵심 발견
- **Multi-view (6-view) + Silhouette-only = Viable**: keypoint 없이 reasonable한 mesh fitting 가능
- **단, keypoint 대비 정밀도 낮음**: 특히 사지(limb) pose 불정확
- **Temporal propagation 효과적**: 프레임 누적 시 후반부 품질 향상
- **scale_weight 임계값**: 최소 30.0 이상 필수 (미만 시 mesh collapse)
- **PCA init 도움**: contour 기반 초기 orientation이 수렴 속도 개선

## 파라미터 탐색 범위

| 파라미터 | 범위 | 권장 |
|----------|------|------|
| iter_multiplier | 1.0 ~ 5.0 | 2.0 ~ 3.0 |
| theta_weight | 5.0 ~ 30.0 | 10.0 ~ 15.0 |
| bone_weight | 0.5 ~ 5.0 | 2.0 ~ 3.0 |
| scale_weight | 10.0 ~ 100.0 | 50.0 |
| use_pca_init | true/false | true |

## 미해결 / 다음 단계
- 정량 평가: keypoint 모드 vs silhouette 모드 mesh accuracy 비교
- Ablation study: 각 파라미터별 sensitivity 분석
- Single-view 확장: monocular silhouette fitting 가능성
- Temporal smoothing: 프레임 간 일관성 강화 정규화

---
*Sources: 251127_silhouette_only_fitting.md, 251127_TIL_silhouette_mesh_fitting.md*
