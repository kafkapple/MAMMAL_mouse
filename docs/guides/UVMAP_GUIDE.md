# UV Map & Texture Guide

> Multi-view RGB 이미지와 fitted mesh sequence로부터 UV texture map을 생성하고, Blender로 내보내는 통합 가이드

---

## 개요

MAMMAL fitting 결과(mesh parameters)와 multi-view 영상으로부터 UV texture map을 생성하는 파이프라인이다.
6개 뷰의 RGB 이미지를 mesh에 projection하여 vertex별 색상을 샘플링하고, 이를 UV 공간으로 매핑하여
512x512 텍스처 이미지를 생성한다. 생성된 텍스처는 Blender OBJ로 내보내어 3D 시각화에 활용할 수 있다.

### Input/Output 경로

**Input:**

| 경로 | 용도 |
|------|------|
| `results/fitting/{experiment}/params/*.pkl` | 프레임별 mesh 파라미터 |
| `results/fitting/{experiment}/obj/*.obj` | 3D 메쉬 |
| `data/examples/.../videos_undist/{0-5}.mp4` | 왜곡 보정된 6-view RGB 영상 |
| `data/examples/.../simpleclick_undist/{0-5}.mp4` | segmentation 마스크 영상 |
| `data/examples/.../new_cam.pkl` | 카메라 캘리브레이션 (K, R, T) |
| `mouse_model/mouse_txt/textures.txt` | Mouse body model UV 좌표 |

**Output:**

| 경로 | 용도 |
|------|------|
| `results/fitting/{experiment}/uvmap/` | UV map 출력 |
| `results/fitting/{experiment}/uvmap/texture_final.png` | UV texture map (RGB, 512x512) |
| `results/fitting/{experiment}/uvmap/confidence.png` | confidence heatmap |
| `results/fitting/{experiment}/uvmap/uv_mask.png` | 유효 UV 영역 마스크 |
| `results/fitting/{experiment}/uvmap/texture.pt` | PyTorch tensor 저장 |

---

## 아키텍처

### 4단계 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input                                    │
│  Fitting params (.pkl) + Multi-view RGB + Masks + Camera         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Texture Sampling (texture_sampler.py)                 │
│    - 각 프레임마다 mesh → multi-view 이미지에 projection        │
│    - vertex별 RGB color 샘플링                                   │
│    - Backface culling으로 visibility weight 계산                 │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Texture Accumulation                                   │
│    - 다중 프레임 가중 평균 (weighted average)                    │
│    - Confidence map 생성                                         │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: UV Rendering (uv_renderer.py)                         │
│    - Vertex color → UV 좌표 공간으로 rasterize                  │
│    - 512x512 텍스처맵 생성                                       │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: (Optional) Photometric Optimization                    │
│    - Differentiable rendering + L1/SSIM loss                     │
│    - TV regularization으로 seam 완화                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: texture_final.png, confidence.png, uv_mask.png         │
└─────────────────────────────────────────────────────────────────┘
```

### 모듈 구조

```
uvmap/
├── uv_pipeline.py          # UVMapPipeline - 전체 파이프라인 오케스트레이션
│   └── UVPipelineConfig    # 파이프라인 설정 dataclass
├── texture_sampler.py      # Multi-view texture 샘플링
│   ├── TextureSampler      # vertex → image projection & sampling
│   └── TextureAccumulator  # 프레임 간 누적
├── uv_renderer.py          # UV space 렌더링
│   └── UVRenderer          # vertex attr → UV map 변환
├── texture_optimizer.py    # Photometric optimization
│   ├── TextureOptConfig    # 최적화 설정
│   ├── TextureModel        # learnable texture (direct/residual)
│   └── TextureOptimizer    # differentiable rendering 기반 최적화
├── experiment_runner.py    # Grid search 실험
│   ├── UVMapEvaluator      # coverage, confidence, seam 메트릭
│   └── ExperimentRunner    # ablation study 실행
├── wandb_sweep.py          # WandB Sweep HPO
│   ├── WandBSweepConfig    # sweep 설정
│   └── WandBSweepOptimizer # sweep 생성 및 agent 실행
└── optuna_optimizer.py     # Optuna HPO
    ├── OptimizationConfig  # optuna 설정
    └── UVMapObjective      # objective function
```

### 하이퍼파라미터

#### Core Parameters (탐색 대상)

| 파라미터 | 타입 | 범위 | 설명 |
|---------|------|------|------|
| `visibility_threshold` | float | 0.1 ~ 0.7 | visibility weight 기준값. 낮을수록 더 많은 view 포함 |
| `uv_size` | categorical | [256, 512, 1024] | UV map 해상도 |
| `fusion_method` | categorical | average, visibility_weighted, max_visibility | multi-view 융합 방법 |
| `w_tv` | float (log) | 1e-5 ~ 1e-1 | Total Variation 정규화 가중치 |
| `do_optimization` | bool | True/False | photometric optimization 수행 여부 |
| `opt_iters` | int | 30, 50, 100 | optimization 반복 횟수 |
| `opt_lr` | float (log) | 1e-4 ~ 1e-1 | optimization 학습률 |

#### Fusion Methods 비교

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| `average` | 단순 평균 | 빠름, 안정적 | 가려진 영역 노이즈 |
| `visibility_weighted` | visibility 가중 평균 | 가려짐 처리 우수 | 파라미터 의존적 |
| `max_visibility` | 최대 visibility view 선택 | 선명한 텍스처 | view 간 불연속 가능 |

#### Fixed Parameters

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `frame_interval` | 1 | 프레임 샘플링 간격 |
| `max_frames` | 20 | HPO 시 사용할 최대 프레임 수 |
| `w_photo` | 1.0 | photometric loss 가중치 |
| `w_smooth` | 1e-4 | smoothness 정규화 |

---

## 실행 방법

### 기본 UV Map 생성

```bash
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
    --uv_size 512
```

### Best Known Config 적용 (wild-sweep-9: Score 0.836)

수치적 최적 (PSNR 역설로 opt=False가 최고 score):

```bash
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
    --uv_size 512 \
    --visibility_threshold 0.346 \
    --fusion_method average
```

시각적 최적 (Blender용, TV smoothing으로 seam 완화):

```bash
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
    --uv_size 512 \
    --visibility_threshold 0.35 \
    --fusion_method average \
    --do_optimization \
    --opt_iters 100 \
    --w_tv 0.005
```

### Python API 사용

```python
from uvmap.uv_pipeline import UVMapPipeline, UVPipelineConfig

config = UVPipelineConfig(
    result_dir="results/fitting/{experiment}",
    uv_size=512,
    visibility_threshold=0.35,
    use_visibility_weighting=True,
    do_optimization=True,
    opt_iters=100,
    opt_w_tv=0.001,
)

pipeline = UVMapPipeline(config)
pipeline.setup()
texture = pipeline.run()
```

### Best Known Config 상세 (wild-sweep-9)

```yaml
visibility_threshold: 0.346
fusion_method: average
do_optimization: false     # PSNR 역설: 원본 색상 유지 → 높은 score
opt_iters: 100             # (미사용, opt=false)
uv_size: 512
w_tv: 0.0049               # (미사용, opt=false)
```

**PSNR 역설**: TV regularization이 텍스처를 부드럽게 하여 시각적으로는 개선되지만,
PSNR은 오히려 낮아질 수 있다. Bayesian optimizer가 opt=False로 수렴하는 경향이 있다.

| 용도 | 권장 설정 | 이유 |
|------|----------|------|
| **수치적 최적** (Score 최고) | `wild-sweep-9` (opt=False) | 원본 색상 유지 → 높은 score |
| **시각적 최적** (Blender용) | opt=True, w_tv=0.005~0.01 | TV smoothing → seam 완화, 노이즈 제거 |
| **seam 최소** (3D 프린팅 등) | `grateful-sweep-37` (seam=0.173) | 최저 seam discontinuity |

---

## WandB Sweep

### Score 함수 (v3)

```
Score = 0.50 * PSNR + 0.15 * SSIM + 0.20 * Coverage + 0.15 * Seam
```

- **Coverage Gating**: Coverage < 80% → Score x 0.1 (패널티)
- **Seam score**: `exp(-15 * seam_discontinuity)` (Exponential Decay)

Score v2 구현 상세 (`wandb_sweep.py`):

```python
def _compute_score(metrics):
    coverage_score = metrics['coverage'] / 100.0
    confidence_score = metrics['mean_confidence']

    # Exponential Decay (Hard Clipping 대체)
    # seam=0.0 → 1.0, seam=0.05 → 0.47, seam=0.1 → 0.22
    seam_sensitivity = 15.0
    seam_score = np.exp(-seam_sensitivity * metrics['seam_discontinuity'])

    score = (
        0.4 * coverage_score +
        0.3 * confidence_score +
        0.3 * seam_score
    )

    # Coverage < 80% → 전체 점수에 페널티
    if metrics['coverage'] < 80.0:
        score *= 0.1

    return score
```

### HPO 파라미터 공간 (6 parameters)

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

### uv_size 고정 옵션

기본적으로 `uv_size=512`로 고정하여 Resolution Bias를 제거하고, 탐색 공간을 1/3 축소한다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--no_fix_uv_size` | False | uv_size 탐색 활성화 |
| `--uv_size` | 512 | 고정할 해상도 |

### 2-Stage Optimization 전략

서로 다른 성격의 파라미터를 분리하여 탐색 효율성을 높인다.

| Stage | 목적 | 탐색 파라미터 | 고정 파라미터 |
|-------|------|--------------|--------------|
| **Stage A (Structure)** | 구조적 최적점 탐색 | visibility_threshold, fusion_method | do_optimization=False, uv_size=512 |
| **Stage B (Refinement)** | 미세 조정 | opt_iters, w_tv, opt_lr | Stage A best + do_optimization=True |

Stage A(20회) + Stage B(20회)로 Full(50회)과 동등 품질, 탐색 비용 20% 절감.

### Sweep 실행 명령어

```bash
# 기본 실행 (권장: uv_size 고정, full stage)
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/{experiment} \
    --count 30

# 2-Stage Optimization
# Stage A: 구조 파라미터 최적화
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

# uv_size 탐색 포함 (기존 방식)
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

### WandB 대시보드 확인 항목

- `score`: 최종 최적화 점수
- `coverage`, `mean_confidence`, `seam_discontinuity`: 개별 메트릭
- `uv_texture`: UV 텍스처 이미지
- `confidence_map`: Confidence 히트맵
- `uv_mask`: UV 마스크

### 기존 Sweep 실험 결과 Top 5

| Rank | Sweep | Run Name | Score | Coverage | Seam | Config |
|------|-------|----------|-------|----------|------|--------|
| **1** | aabflhvs | **wild-sweep-9** | **0.836** | 100% | 0.241 | vis=0.35, avg, opt=F |
| **2** | n7feev3i | elated-sweep-5 | 0.836 | 100% | 0.241 | vis=0.58, avg, opt=F |
| **3** | gm4e24p6 | upbeat-sweep-76 | 0.617 | 97.8% | 0.924 | vis=0.52, avg, opt=T |
| **4** | kpfwmwhj | grateful-sweep-37 | 0.399 | 100% | 0.173 | vis=0.14, max, opt=T |
| **5** | 0heflsc1 | eager-sweep-3 | 0.396 | 100% | 0.192 | vis=0.33, max, opt=T |

> Score 차이: sweep별로 Score 함수 버전(v1/v2/v3)이 다름.

### Optuna / Grid Search

```bash
# Optuna
python -m uvmap.optuna_optimizer \
    --result_dir results/fitting/{experiment} \
    --n_trials 50 \
    --output_dir optuna_results

# Grid Search (Quick test)
python -m uvmap.experiment_runner \
    --result_dir results/fitting/{experiment} \
    --output_dir uvmap_experiments \
    --quick

# Grid Search (Full ablation)
python -m uvmap.experiment_runner \
    --result_dir results/fitting/{experiment} \
    --output_dir uvmap_experiments
```

---

## Blender 내보내기

### 좌표 변환 (MAMMAL -> Blender)

MAMMAL과 Blender는 **up 축**이 다르므로, 변환 없이 임포트하면 메쉬가 90도 옆으로 누워 보인다.

| 좌표계 | Up 축 | Forward 축 | Right 축 |
|--------|--------|-----------|----------|
| **MAMMAL** | **-Y** | +X (head->tail) | +Z |
| **Blender World** | **+Z** | +Y | +X |

변환 공식 (X축 기준 +90도 회전):

```
(x, y, z)_MAMMAL  →  (x, z, -y)_Blender
```

행렬 표현:

```
| x' |   | 1  0  0 | | x |
| y' | = | 0  0  1 | | y |
| z' |   | 0 -1  0 | | z |
```

`export_to_blender.py` 기본 처리:

| 처리 | 기본값 | 설명 | 비활성화 |
|------|--------|------|----------|
| **좌표 변환** | ON | MAMMAL -> Blender World | `--no_transform` |
| **센터링** | ON | 원점 중심으로 이동 | `--no_center` |
| **mm->m 스케일** | ON | MAMMAL은 mm 단위, Blender는 m | `--no_scale` |

**검증 기준** (Blender에서 정상):
- 등(back)이 위(+Z), 배(belly)가 아래(-Z)
- 머리(head)가 앞(+Y 또는 +X 방향)
- 크기가 약 0.1m (실제 마우스 체장 약 10cm)

### 단일 프레임 내보내기

```bash
# 기존 sweep 최고 성능 텍스처 사용
python scripts/export_to_blender.py \
    --mesh results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/obj/step_2_frame_000000.obj \
    --texture results/sweep/run_wild-sweep-9/texture_final.png \
    --output exports/mouse_textured.obj

# 새로 생성한 텍스처 사용
python scripts/export_to_blender.py \
    --mesh results/fitting/<experiment>/obj/step_2_frame_000000.obj \
    --texture results/fitting/<experiment>/uvmap/texture_final.png \
    --output exports/mouse_textured.obj

# 좌표 변환 없이 (MAMMAL 원본 좌표)
python scripts/export_to_blender.py \
    --mesh ... --texture ... --output ... \
    --no_transform --no_center --no_scale
```

출력물:

```
exports/
├── mouse_textured.obj     # 메쉬 + UV 좌표 (v/vt/f)
├── mouse_textured.mtl     # 머티리얼 파일 (텍스처 참조)
└── texture_final.png      # UV 텍스처 이미지
```

### Batch 처리 (전체 프레임 + 6-view 그리드 영상)

```bash
# 전체 프레임 일괄 처리 (OBJ + 영상)
python -m mammal_ext.blender_export.run_all \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --texture results/sweep/run_wild-sweep-9/texture_final.png \
    --output_dir exports/v012345_kp22_20251206/

# OBJ만 (영상 생략)
python -m mammal_ext.blender_export.run_all \
    --result_dir ... --texture ... --output_dir ... --skip_video

# 영상만 (OBJ 생략)
python -m mammal_ext.blender_export.run_all \
    --result_dir ... --texture ... --output_dir ... --skip_obj

# 첫 10프레임만 테스트
python -m mammal_ext.blender_export.run_all \
    --result_dir ... --texture ... --output_dir ... --max_frames 10
```

Batch 출력 구조:

```
exports/v012345_kp22_20251206/
├── obj/                              # OBJ + MTL + texture (Blender용)
│   ├── step_2_frame_000000.obj
│   ├── step_2_frame_000000.mtl
│   ├── texture_final.png
│   └── ...
└── renders/
    ├── *_6view_grid.mp4              # 6-view 그리드 영상
    └── *_6view_sample.png            # 첫 프레임 샘플 이미지
```

> UV 텍스처는 body model 고유의 UV layout이므로, **동일 텍스처를 모든 프레임에 재사용** 가능.

### 개별 모듈 실행

```bash
# OBJ만 일괄 내보내기
python -m mammal_ext.blender_export.batch_export \
    --result_dir results/fitting/<experiment> \
    --texture texture_final.png \
    --output_dir exports/obj/

# 6-view 그리드 영상만
python -m mammal_ext.blender_export.sequence_renderer \
    --result_dir results/fitting/<experiment> \
    --texture texture_final.png \
    --output_dir exports/renders/
```

### Blender에서 임포트

1. **File > Import > Wavefront (.obj)** 에서 `mouse_textured.obj` 선택
2. MTL이 자동으로 텍스처 연결
3. 텍스처가 보이지 않으면 **Z 키** -> Material Preview 또는 Rendered 선택

**텍스처 수동 연결** (MTL 자동 로드 실패 시):
1. Material Properties 탭
2. Base Color 옆 노란 점 클릭
3. **Image Texture** 선택
4. `texture_final.png` 열기

**애니메이션** (다중 프레임):
1. 각 프레임 OBJ를 Stop Motion OBJ 애드온으로 임포트
2. 또는 `scripts/blender_mesh_animation.py` 활용:
   ```bash
   # Blender 내 Python console에서
   exec(open("scripts/blender_mesh_animation.py").read())
   ```

### OBJ 파일 구조 (참고)

```obj
# Vertices
v  x y z

# Texture coordinates (UV, mouse_model/mouse_txt/textures.txt)
vt u v

# Faces (1-indexed, vertex/texture pair)
f v1/vt1 v2/vt2 v3/vt3

# Material reference
mtllib mouse_textured.mtl
usemtl mouse_material
```

---

## 평가 메트릭

### 개별 메트릭

| 메트릭 | 범위 | 목표 | 설명 |
|--------|------|------|------|
| `coverage` | 0~100% | maximize | UV 공간 중 유효 픽셀 비율 |
| `mean_confidence` | 0~1 | maximize | 평균 샘플링 confidence |
| `seam_discontinuity` | 0~inf | minimize | UV seam에서의 색상 불연속성 |

### 메트릭 해석

- **coverage 90%+**: 대부분의 UV 영역에 텍스처 할당됨
- **mean_confidence 0.5+**: 평균적으로 여러 view에서 샘플링됨
- **seam_discontinuity < 0.05**: seam artifact 거의 없음

---

## Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| Score가 NaN | `seam_discontinuity` 계산 시 빈 텐서 `.mean()` | `wandb_sweep.py`에 NaN 체크 추가됨 |
| "Can't call numpy() on Tensor that requires grad" | optimization 후 gradient 텐서 직접 변환 | `.detach()` 추가 |
| Coverage 0~50% | `visibility_threshold` 과도 | 0.2~0.3으로 낮춤 |
| Seam artifact | TV regularization 부족 | `w_tv` 0.001~0.01로 증가, `do_optimization=True` |
| 색상 반전 (R/B 뒤바뀜) | BGR/RGB 불일치 | `cv2.cvtColor` 확인 |
| 메쉬가 1픽셀로 투영 | 카메라 T 단위 불일치 (mm vs m) | `cam['T'] / 1000` |
| 메쉬가 90도 옆으로 누움 | MAMMAL(-Y up) vs Blender(Z up) | `--no_transform` 제거 (기본값이 변환 ON) |
| Blender에서 텍스처 안 보임 | Viewport Shading 모드 | Material Preview(Z) 또는 Rendered로 전환 |
| platformdirs ImportError | wandb 의존성 누락 | `pip install platformdirs` |

---

## 사용 가능한 리소스

### Fitting 실험 (obj+params 보유)

| 실험명 | 날짜 | obj | params |
|--------|------|-----|--------|
| `v012345_kp22_20251213_200852` | 2025-12-13 | O | O |
| `v012345_kp22_20251213_201317` | 2025-12-13 | O | O |
| `v012345_kp22_20260125_174356` | 2026-01-25 | O | O |
| `v012345_kp22_20260125_230540` | 2026-01-25 | O | O |
| `v012345_kp22_20260125_230806` | 2026-01-25 | O | O |
| `v012345_kp22_20260125_231350` | 2026-01-25 | O | O |

### Sweep 텍스처 (바로 사용 가능)

| Run | Score | 텍스처 경로 |
|-----|-------|-----------|
| **wild-sweep-9** | **0.836** | `results/sweep/run_wild-sweep-9/texture_final.png` |
| upbeat-sweep-76 | 0.617 | `results/sweep/run_upbeat-sweep-76/texture_final.png` |
| grateful-sweep-37 | 0.399 | `results/sweep/run_grateful-sweep-37/texture_final.png` |
| eager-sweep-3 | 0.396 | `results/sweep/run_eager-sweep-3/texture_final.png` |

---

## References

- Seamless Texture Optimization (CGF 2024)
- Image Quality Assessment: [PMC7817470](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/)
- Optuna TPE: https://optuna.readthedocs.io/
- WandB Sweeps: https://docs.wandb.ai/guides/sweeps

---

*Merged from: uvmap_system.md, UV_TEXTURE_TO_BLENDER.md*
*Last updated: 2026-02-06*
