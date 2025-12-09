# UV Map Generation System

> Multi-view RGB 이미지와 fitted mesh sequence로부터 UV texture map을 생성하는 시스템

---

## 0. 데이터 및 Output 위치

### Input 데이터

| 경로 | 용도 |
|------|------|
| `data/examples/markerless_mouse_1_nerf/` | 예제 데이터 (multi-view 영상, 마스크, 카메라) |
| `data/examples/.../videos_undist/{0-5}.mp4` | 왜곡 보정된 6-view RGB 영상 |
| `data/examples/.../simpleclick_undist/{0-5}.mp4` | segmentation 마스크 영상 |
| `data/examples/.../new_cam.pkl` | 카메라 캘리브레이션 |
| `mouse_model/mouse_txt/` | Mouse body model (mesh, UV coords) |

### Output 위치

| 경로 | 용도 |
|------|------|
| `results/fitting/{experiment}/` | Fitting 결과 |
| `results/fitting/{experiment}/params/` | 프레임별 mesh 파라미터 (.pkl) |
| `results/fitting/{experiment}/uvmap/` | UV map 출력 |
| `results/logs/` | 학습 로그 |
| `wandb_sweep_results/` | WandB sweep 결과 |

### Assets 구조

```
/home/joon/dev/MAMMAL_mouse/
├── colormaps/              ← 실제 사용됨 (코드에서 참조)
│   ├── anliang_paper.txt   ← 주로 사용 (22 keypoint colormap)
│   └── ...
├── assets/
│   ├── colormaps/          ← 중복 (삭제 가능)
│   ├── figs/
│   └── mouse_model/
└── mouse_model/            ← body model 위치
```

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input                                    │
├─────────────────────────────────────────────────────────────────┤
│  • Fitting 결과: results/fitting/{experiment}/params/*.pkl      │
│  • Multi-view 영상: data/.../videos_undist/{0-5}.mp4            │
│  • Segmentation 마스크: data/.../simpleclick_undist/{0-5}.mp4   │
│  • 카메라 파라미터: data/.../new_cam.pkl                         │
│  • Mouse 모델: mouse_model/mouse_txt/                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UV Map Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Texture Sampling (texture_sampler.py)                 │
│    - 각 프레임마다 mesh → multi-view 이미지에 projection        │
│    - vertex별 RGB color 샘플링                                   │
│    - visibility weight 계산                                      │
│                                                                  │
│  Stage 2: Texture Accumulation                                   │
│    - 프레임들 간 weighted average                                │
│    - confidence map 생성                                         │
│                                                                  │
│  Stage 3: UV Rendering (uv_renderer.py)                         │
│    - vertex colors → UV space로 매핑                            │
│    - UV texture map 생성                                         │
│                                                                  │
│  Stage 4: (Optional) Photometric Optimization                    │
│    - differentiable rendering으로 texture 정제                   │
│    - TV regularization으로 smoothness 유지                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Output                                   │
├─────────────────────────────────────────────────────────────────┤
│  • texture_final.png: UV texture map (RGB)                      │
│  • confidence.png: confidence heatmap                           │
│  • uv_mask.png: valid UV region mask                            │
│  • texture.pt: PyTorch tensor 저장                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 최적화 파라미터 (Hyperparameters)

### 2.1 Core Parameters (탐색 대상)

| 파라미터 | 타입 | 범위 | 설명 |
|---------|------|------|------|
| `visibility_threshold` | float | 0.1 ~ 0.7 | visibility weight 기준값. 낮을수록 더 많은 view 포함 |
| `uv_size` | categorical | [256, 512, 1024] | UV map 해상도. 높을수록 디테일 증가, 메모리 사용량 증가 |
| `fusion_method` | categorical | average, visibility_weighted, max_visibility | multi-view 융합 방법 |
| `w_tv` | float (log) | 1e-5 ~ 1e-1 | Total Variation 정규화 가중치. 높을수록 smooth |
| `do_optimization` | bool | True/False | photometric optimization 수행 여부 |
| `opt_iters` | int | 30, 50, 100 | optimization 반복 횟수 |
| `opt_lr` | float (log) | 1e-4 ~ 1e-1 | optimization 학습률 |

### 2.2 Fusion Methods

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| `average` | 단순 평균 | 빠름, 안정적 | 가려진 영역 노이즈 |
| `visibility_weighted` | visibility 가중 평균 | 가려짐 처리 우수 | 파라미터 의존적 |
| `max_visibility` | 최대 visibility view 선택 | 선명한 텍스처 | view 간 불연속 가능 |

### 2.3 Fixed Parameters

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `frame_interval` | 1 | 프레임 샘플링 간격 |
| `max_frames` | 20 | HPO시 사용할 최대 프레임 수 (속도 최적화) |
| `w_photo` | 1.0 | photometric loss 가중치 |
| `w_smooth` | 1e-4 | smoothness 정규화 |

---

## 3. 평가 메트릭 (Evaluation Metrics)

### 3.1 개별 메트릭

| 메트릭 | 범위 | 목표 | 설명 |
|--------|------|------|------|
| `coverage` | 0~100% | maximize | UV 공간 중 유효 픽셀 비율 |
| `mean_confidence` | 0~1 | maximize | 평균 샘플링 confidence |
| `seam_discontinuity` | 0~∞ | minimize | UV seam에서의 색상 불연속성 |

### 3.2 Composite Score 계산 (v2 - Exponential Decay)

```python
def _compute_score(metrics):
    coverage_score = metrics['coverage'] / 100.0      # [0, 1]
    confidence_score = metrics['mean_confidence']      # [0, 1]

    # ===== v2: Exponential Decay (Hard Clipping 대체) =====
    # 기존: max(0, 1 - seam * 10) → seam > 0.1이면 무조건 0점 (Dead Zone)
    # 개선: exp(-k * seam) → 연속적인 기울기로 Optimizer 학습 신호 유지
    #
    # seam=0.0 → score=1.0 (완벽)
    # seam=0.05 → score≈0.47 (양호)
    # seam=0.1 → score≈0.22 (주의)
    # seam=0.2 → score≈0.05 (나쁨, but 기울기 존재!)
    seam_sensitivity = 15.0  # k: 민감도 상수
    seam_score = np.exp(-seam_sensitivity * metrics['seam_discontinuity'])

    score = (
        0.4 * coverage_score +     # w_coverage
        0.3 * confidence_score +   # w_psnr (confidence 사용)
        0.3 * seam_score           # w_seam (Exponential Decay)
    )

    # ===== v2: Coverage Gating =====
    # Coverage < 80%이면 전체 점수에 페널티 (10배 감소)
    # 목적: UV 공간 활용도가 낮은 결과는 다른 지표가 좋아도 탈락
    if metrics['coverage'] < 80.0:
        score *= 0.1

    return score  # Higher is better
```

#### v2 개선사항 요약

| 항목 | v1 (기존) | v2 (개선) | 효과 |
|------|-----------|-----------|------|
| **Seam Score** | `max(0, 1-seam*10)` Hard Clipping | `exp(-15*seam)` Exponential Decay | Dead Zone 제거, Optimizer 학습 효율↑ |
| **Coverage Gate** | 없음 | Coverage<80% → score×0.1 | 저커버리지 결과 조기 탈락 |

### 3.3 메트릭 해석

- **coverage 90%+**: 대부분의 UV 영역에 텍스처 할당됨
- **mean_confidence 0.5+**: 평균적으로 여러 view에서 샘플링됨
- **seam_discontinuity < 0.05**: seam artifact 거의 없음

---

## 4. 최적화 도구

### 4.1 WandB Sweep

**특징:**
- Bayesian optimization (TPE)
- 실시간 대시보드 시각화
- 분산 agent 지원

#### 4.1.1 최적화 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| `method` | `bayes` | Bayesian Optimization (Tree-structured Parzen Estimator) |
| `metric_name` | `score` | 최적화 대상 메트릭 |
| `metric_goal` | `maximize` | **높을수록 좋음** |

#### 4.1.2 Score 계산 상세 (v2)

```python
# wandb_sweep.py: _compute_score() - v2 Exponential Decay
score = (
    w_coverage * coverage_score +      # 0.4 * (coverage / 100)
    w_psnr * confidence_score +        # 0.3 * mean_confidence
    w_seam * seam_score                # 0.3 * exp(-15 * seam)
)

# Coverage Gating
if coverage < 80%:
    score *= 0.1
```

| 구성요소 | 가중치 | 원본값 | 변환 (v2) | 최종 범위 |
|----------|--------|--------|-----------|-----------|
| **coverage_score** | 40% | coverage (0~100%) | `/ 100` | [0, 1] |
| **confidence_score** | 30% | mean_confidence | 그대로 | [0, 1] |
| **seam_score** | 30% | seam_discontinuity | `exp(-15 * seam)` | (0, 1] |
| **coverage_gate** | - | coverage < 80% | `score *= 0.1` | 페널티 |

#### 4.1.3 가중치 설계 근거

| 메트릭 | 가중치 | 근거 |
|--------|--------|------|
| **coverage** | 40% (최대) | UV 공간 활용도 = 텍스처 완성도. 빈 영역이 많으면 렌더링 품질 저하 |
| **confidence** | 30% | 다중 view에서 일관되게 샘플링된 영역 → 색상 신뢰도 높음 |
| **seam** | 30% | UV 경계에서 색상 불연속 = 시각적 artifact. **낮을수록 좋아서 반전** |

#### 4.1.4 Bayesian Optimization 동작

```
Trial 1-5: Random exploration (초기 탐색)
    ↓
Trial 6+: TPE가 좋은 영역 집중 탐색
    - 높은 score를 낸 파라미터 조합 주변 탐색
    - 탐색(exploration) vs 활용(exploitation) 균형
    ↓
Best config 수렴
```

#### 4.1.5 Parameter Search Space

```python
DEFAULT_SWEEP_PARAMS = {
    'visibility_threshold': {'distribution': 'uniform', 'min': 0.1, 'max': 0.7},
    'uv_size': {'values': [256, 512, 1024]},
    'fusion_method': {'values': ['average', 'visibility_weighted', 'max_visibility']},
    'w_tv': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
    'do_optimization': {'values': [False, True]},
    'opt_iters': {'values': [30, 50, 100]},
}
```

#### 4.1.6 Search Space 최적화 (v2)

**uv_size 고정 옵션 (기본 활성화)**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--no_fix_uv_size` | False | uv_size 탐색 활성화 |
| `--uv_size` | 512 | 고정할 해상도 |

**왜 uv_size를 고정하는가?**
- **Resolution Bias 제거**: 해상도에 따라 Metric Scale이 왜곡됨
- **탐색 효율화**: 3개 값 중 하나를 제거하면 탐색 공간 1/3 감소
- **공정한 비교**: 같은 해상도에서 파라미터 효과 평가 가능
- **후처리 Upscaling**: 최종 렌더링 시 1024로 상향 조정

#### 4.1.7 2-Stage Optimization 전략 (v2)

서로 다른 성격의 파라미터를 분리하여 탐색 효율성 향상

| Stage | 목적 | 탐색 파라미터 | 고정 파라미터 |
|-------|------|--------------|--------------|
| **Stage A (Structure)** | 구조적 최적점 탐색 | visibility_threshold, fusion_method | do_optimization=False, uv_size=512 |
| **Stage B (Refinement)** | 미세 조정 | opt_iters, w_tv, opt_lr | Stage A best + do_optimization=True |

**효과:**
- Stage A(20회) + Stage B(20회) ≈ Full(50회) 동등 품질
- 총 탐색 비용 20% 절감

**사용법:**

```bash
# ===== 기본 실행 (권장: uv_size 고정, full stage) =====
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/{experiment} \
    --count 30

# ===== 2-Stage Optimization =====
# Stage A: 구조 파라미터 최적화 (빠른 탐색)
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/{experiment} \
    --stage stage_a \
    --count 20

# Stage B: Stage A 결과 기반 미세 조정
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/{experiment} \
    --stage stage_b \
    --stage_a_config wandb_sweep_results/best_config.json \
    --count 20

# ===== uv_size 탐색 포함 (기존 방식) =====
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/{experiment} \
    --no_fix_uv_size \
    --count 50

# sweep만 생성 (다른 서버에서 agent 실행용)
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/{experiment} \
    --create_only

# 기존 sweep에 agent 추가
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/{experiment} \
    --sweep_id {sweep_id} \
    --count 10
```

**WandB 대시보드 확인 항목:**
- `score`: 최종 최적화 점수
- `coverage`, `mean_confidence`, `seam_discontinuity`: 개별 메트릭
- `uv_texture`: UV 텍스처 이미지
- `confidence_map`: Confidence 히트맵
- `uv_mask`: UV 마스크

### 4.2 Optuna

**특징:**
- TPE/CMA-ES 등 다양한 sampler
- Pruning 지원
- Multi-objective 최적화 가능

**사용법:**

```bash
python -m uvmap.optuna_optimizer \
    --result_dir results/fitting/{experiment} \
    --n_trials 50 \
    --output_dir optuna_results
```

### 4.3 Grid Search (Ablation Study)

**사용법:**

```bash
# Quick test (적은 조합)
python -m uvmap.experiment_runner \
    --result_dir results/fitting/{experiment} \
    --output_dir uvmap_experiments \
    --quick

# Full ablation
python -m uvmap.experiment_runner \
    --result_dir results/fitting/{experiment} \
    --output_dir uvmap_experiments
```

---

## 5. 모듈 구조

```
uvmap/
├── uv_pipeline.py          # UVMapPipeline - 전체 파이프라인 오케스트레이션
│   └── UVPipelineConfig    # 파이프라인 설정
│
├── texture_sampler.py      # Multi-view texture 샘플링
│   ├── TextureSampler      # vertex → image projection & sampling
│   └── TextureAccumulator  # 프레임 간 누적
│
├── uv_renderer.py          # UV space 렌더링
│   └── UVRenderer          # vertex attr → UV map 변환
│
├── texture_optimizer.py    # Photometric optimization
│   ├── TextureOptConfig    # 최적화 설정
│   ├── TextureModel        # learnable texture (direct/residual)
│   └── TextureOptimizer    # differentiable rendering 기반 최적화
│
├── experiment_runner.py    # 실험 및 평가
│   ├── UVMapEvaluator      # coverage, confidence, seam 메트릭 계산
│   └── ExperimentRunner    # grid search 실행
│
├── wandb_sweep.py          # WandB Sweep HPO
│   ├── WandBSweepConfig    # sweep 설정
│   └── WandBSweepOptimizer # sweep 생성 및 agent 실행
│
└── optuna_optimizer.py     # Optuna HPO
    ├── OptimizationConfig  # optuna 설정
    └── UVMapObjective      # objective function
```

---

## 6. 실행 예시

### 6.1 기본 UV Map 생성

```bash
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --uv_size 512 \
    --do_optimization
```

### 6.2 WandB Sweep 최적화

```bash
# Step 1: Sweep 생성 및 실행
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --count 30

# Step 2: 대시보드에서 결과 확인
# https://wandb.ai/{entity}/uvmap-optimization

# Step 3: best config 적용
# wandb_sweep_results/best_config.json 참조
```

### 6.3 최적 파라미터로 최종 UV Map 생성

```python
from uvmap.uv_pipeline import UVMapPipeline, UVPipelineConfig

config = UVPipelineConfig(
    result_dir="results/fitting/{experiment}",
    uv_size=512,
    visibility_threshold=0.35,  # from HPO
    use_visibility_weighting=True,
    do_optimization=True,
    opt_iters=100,
    opt_w_tv=0.001,
)

pipeline = UVMapPipeline(config)
pipeline.setup()
texture = pipeline.run()
```

---

## 7. 트러블슈팅

### 7.1 Score가 NaN으로 표시됨

**원인:** `seam_discontinuity` 계산 시 빈 텐서에 `.mean()` 호출

**해결:** `wandb_sweep.py`의 `_compute_score`에 NaN 체크 추가됨

### 7.2 "Can't call numpy() on Tensor that requires grad"

**원인:** optimization 후 gradient가 있는 텐서를 직접 numpy 변환

**해결:** `.detach()` 추가 (`uv_pipeline.py`의 `_save_results`)

### 7.3 Coverage가 낮음 (< 50%)

**가능한 원인:**
- `visibility_threshold`가 너무 높음 → 낮추기 (0.2~0.3)
- 프레임 수 부족 → `frame_interval` 줄이기
- 카메라 캘리브레이션 오류

### 7.4 Seam artifact 발생

**해결 방법:**
- `w_tv` 증가 (1e-3 → 1e-2)
- `do_optimization=True`로 photometric refinement 수행
- `fusion_method='visibility_weighted'` 사용

---

## 8. References

- Seamless Texture Optimization (CGF 2024)
- Image Quality Assessment: [PMC7817470](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/)
- Optuna TPE: https://optuna.readthedocs.io/
- WandB Sweeps: https://docs.wandb.ai/guides/sweeps
