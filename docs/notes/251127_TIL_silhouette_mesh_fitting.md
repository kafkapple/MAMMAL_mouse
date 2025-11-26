# 📄 Silhouette-Only Multi-View Mesh Fitting

**대화 요약**: Keypoint annotation 없이 multi-view mask silhouette만으로 3D mouse mesh fitting을 구현하고, 최적화 파라미터 설계 및 서버 간 호환성을 확보함

**주요 다룬 주제**:

1. Silhouette-only fitting 모드 구현 및 활성화
2. Temporal propagation을 통한 프레임 간 최적화
3. 실험 파라미터 설계 및 서버 호환성

---

## 1. Silhouette-Only Fitting 핵심 개념

### 1.1 기존 방식 vs 새로운 방식

**기존 (Keypoint 기반)**:
```
Input: Multi-view video + 2D keypoints + Masks
Loss: 2D keypoint reprojection loss (주력) + Mask IoU loss (보조)
```

**새로운 (Silhouette 기반)**:
```
Input: Multi-view video + Masks only (keypoint 불필요)
Loss: Mask IoU loss만 사용 (6개 뷰 동시)
```

- **핵심개념**: Keypoint annotation 비용 절감하면서 multi-view geometry 활용
- **작동원리**: 6개 카메라 뷰의 silhouette 일치도(IoU)로 3D mesh 최적화
- **활용예시**: 자동 mask 생성 도구(SAM, SimpleClick)와 연계 가능

### 1.2 구현 변경사항

| 구분 | Keypoint 모드 | Silhouette 모드 |
|------|--------------|-----------------|
| `term_weights["2d"]` | 활성화 | **0** |
| `term_weights["mask"]` Step0 | 0 | **1000** |
| `term_weights["mask"]` Step1 | 0 | **1500** |
| `theta_weight` | 3.0 | **10.0** |
| `scale_weight` | 0.5 | **50.0** |

**핵심 변경**: Mask loss가 Step 0/1에서도 활성화되어야 초기 피팅 가능

---

## 2. Temporal Propagation 효과

### 2.1 프레임 간 초기화 전파

```
Frame 0: 기본 초기값 (PCA init)
   ↓ 결과 전달
Frame 1: Frame 0 최적화 결과로 시작
   ↓
Frame N: 누적된 최적화 이점
```

- **문제상황**: 각 프레임 독립 최적화 시 일관성 부족
- **해결방법**: 이전 프레임 결과를 다음 프레임 초기값으로 사용
- **주의사항**: 초반 피팅 실패 시 오류가 누적될 수 있음

### 2.2 실험적 발견

```bash
# 2프레임만 실행 (디버그)
./run_silhouette_experiments.sh /path/to/data 0 2
# → Frame 0, 1만 피팅 (초기 품질)

# 100프레임 실행 (실제)
./run_silhouette_experiments.sh /path/to/data 0 100
# → Frame 99는 98번의 누적 최적화 이점 보유
```

**결론**: 더 많은 프레임 실행 시 후반 프레임 품질 향상

---

## 3. 실험 파라미터 설계

### 3.1 4가지 비교 실험 구조

| 실험 | 변경 요소 | 목적 |
|------|----------|------|
| exp1_baseline | 기준 | 비교 기준선 |
| exp2_more_iters | `iter_multiplier=3.0` | 반복 횟수 효과 |
| exp3_high_reg | `theta=15, bone=3` | 정규화 강화 효과 |
| exp4_no_pca | `use_pca_init=false` | PCA 초기화 효과 |

**설계 원칙**: 한 번에 하나의 변수만 변경 (controlled experiment)

### 3.2 파라미터 탐색 범위

```yaml
iter_multiplier: 1.0 ~ 5.0  # 권장: 2.0 ~ 3.0
theta_weight: 5.0 ~ 30.0    # 권장: 10.0 ~ 15.0
bone_weight: 0.5 ~ 5.0      # 권장: 2.0 ~ 3.0
scale_weight: 10.0 ~ 100.0  # 권장: 50.0 (필수!)
use_pca_init: true/false    # 권장: true
```

---

## 4. 서버 호환성 (Portability)

### 4.1 자동 처리 항목

```bash
# Python 경로 자동 감지 (run_silhouette_experiments.sh)
if [ -f "${HOME}/miniconda3/envs/mammal_stable/bin/python" ]; then
    PYTHON="${HOME}/miniconda3/envs/mammal_stable/bin/python"
elif [ -f "${HOME}/anaconda3/envs/mammal_stable/bin/python" ]; then
    PYTHON="${HOME}/anaconda3/envs/mammal_stable/bin/python"
fi

# EGL 환경변수 자동 설정
export PYOPENGL_PLATFORM=egl
```

### 4.2 수동 지정 필요 항목

```bash
# 데이터 경로는 서버마다 다름 → 항상 --input_dir 사용
./run_mesh_fitting_default.sh 0 10 -- --keypoints none \
    --input_dir /your/server/specific/path
```

### 4.3 Hydra 인자 파싱 수정

**문제**: `$extra_args` 문자열 확장 시 Hydra 파싱 오류
```
mismatched input '<EOF>' expecting {EQUAL, '~', '+', '@', KEY_SPECIAL, DOT_PATH, ID}
```

**해결**: 배열 방식으로 변경
```bash
# Before (문제)
run_experiment "exp1" "arg1=val1 arg2=val2"

# After (해결)
run_experiment "exp1" arg1=val1 arg2=val2
```

---

## 5. 구현 코드 핵심

### 5.1 Silhouette 모드 활성화 (`fitter_articulation.py`)

```python
# Silhouette 모드 설정 적용
if not getattr(self.cfg.fitter, 'use_keypoints', True):
    self.term_weights["2d"] = 0
    sil_cfg = getattr(self.cfg, 'silhouette', None)
    if sil_cfg:
        self.term_weights["scale"] = getattr(sil_cfg, 'scale_weight', 50.0)
        self.term_weights["theta"] = getattr(sil_cfg, 'theta_weight', 10.0)
        self.silhouette_iter_multiplier = getattr(sil_cfg, 'iter_multiplier', 2.0)
```

### 5.2 Config 구조 (`conf/config.yaml`)

```yaml
silhouette:
  iter_multiplier: 2.0
  theta_weight: 10.0
  bone_weight: 2.0
  scale_weight: 50.0
  use_pca_init: true
```

---

## 💡 대화에서 얻은 핵심 인사이트

1. **Multi-view + Silhouette = Viable**: Keypoint 없이도 6개 뷰의 mask만으로 reasonable한 3D mesh fitting 가능. 단, 정규화 강화 필수.

2. **Temporal Propagation의 힘**: 프레임을 많이 실행할수록 후반 프레임 품질 향상. 디버그는 2프레임, 실제 분석은 100+ 프레임 권장.

3. **Scale Weight의 중요성**: Silhouette 모드에서 scale_weight가 낮으면 mesh collapse 발생. 최소 30.0 이상 필수.

---

## ❓ 미해결 질문 또는 추가 학습 필요 사항

- **정량 평가**: Keypoint 모드 vs Silhouette 모드의 정확도 차이는?
- **Single-view 확장**: Monocular silhouette fitting 가능성?
- **최적 파라미터**: Grid search 실험 후 최적 조합 확인 필요

---

## 🔗 참고 자료 및 키워드

**키워드**:
- Silhouette-based optimization
- Multi-view reconstruction
- Temporal propagation
- Hydra configuration
- EGL headless rendering

**생성된 파일**:
- `run_silhouette_experiments.sh`: 실험 스크립트
- `conf/config.yaml`: silhouette 설정 추가
- `docs/notes/251127_silhouette_only_fitting.md`: 연구 노트
- `docs/guides/silhouette_parameters.md`: 파라미터 가이드

**명령어 Quick Reference**:
```bash
# 디버그 테스트
./run_mesh_fitting_default.sh 0 2 -- --keypoints none --input_dir /path/to/data

# 실험 비교
./run_silhouette_experiments.sh /path/to/data 0 2

# 본 실행
./run_mesh_fitting_default.sh 0 100 -- --keypoints none --input_dir /path/to/data
```
