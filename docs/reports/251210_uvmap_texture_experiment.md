---
date: 2025-12-10
context_name: "2_Research"
tags: [ai-assisted, uv-mapping, texture, wandb-sweep, psnr, ssim, loss-function, hyperparameter-optimization]
project: MAMMAL_mouse
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# UV Map Texture 실험: Loss 설계, Grid 시각화, WandB Sweep 최적화

## 1. 제목 (Title)

**Multi-view Mouse Mesh의 UV Texture Mapping: 정량/정성 평가 기준 설계 및 Hyperparameter 최적화**

---

## 2. 날짜 (Date)

- **실험 기간**: 2025-12-09 ~ 2025-12-10
- **노트 작성일**: 2025-12-10

---

## 3. 연구 주제 (Research Topic)

Mesh fitting 결과로부터 UV texture map을 생성하는 파이프라인의:
- **Loss/Score 함수 설계**: 정량적 품질 평가 기준
- **시각화 체계**: 6-view projection grid를 통한 정성적 평가
- **HPO (Hyperparameter Optimization)**: WandB Sweep을 활용한 최적 파라미터 탐색

---

## 4. 핵심 목표 (Key Objective)

1. **정량 평가 기준 확립**: PSNR, SSIM 기반 photometric score 도입
2. **정성 평가 체계 구축**: 6-view comparison grid로 원본 vs 렌더링 비교
3. **WandB Sweep 파이프라인**: 자동화된 hyperparameter 탐색 및 대시보드 제공
4. **최적 파라미터 도출**: UV texture 품질 극대화 설정 발견

---

## 5. 배경 및 동기 (Background & Motivation)

### 5.1 UV Texture Mapping 개요

Mesh fitting 결과에 텍스처를 입히는 과정:

```
Multi-view RGB → Mesh Vertices → UV Projection → Texture Fusion → UV Map
```

**핵심 과제**:
- 다중 view에서 관측된 색상을 어떻게 융합할 것인가?
- Visibility, occlusion, seam 처리

### 5.2 기존 Score 함수의 문제점

**Score v2 (이전 버전)**:
```python
score = w_coverage * coverage + w_psnr * confidence + w_seam * seam
```

| 문제점 | 설명 |
|--------|------|
| **변수명 오류** | `w_psnr`가 실제 PSNR이 아닌 `mean_confidence` 사용 |
| **Photometric 부재** | Rendered vs Original 이미지 비교 없음 |
| **Perceptual 부재** | Human perception 기반 메트릭 없음 |

### 5.3 학술 문헌 기반 개선

**3D Gaussian Splatting (SIGGRAPH 2023)**:
```
Loss = (1 - λ) × L1 + λ × D-SSIM,  λ = 0.2
```

→ **L1 (pixel-wise) + SSIM (structural)** 조합이 표준

---

## 6. 방법론 (Methodology)

### 6.1 Score v3 설계

#### Score 공식

```python
# uvmap/wandb_sweep.py:794-799
score = (
    w_photo * photo_score +       # 0.50: PSNR 기반 photometric
    w_ssim * ssim_score +         # 0.15: Structural similarity
    w_coverage * coverage_score + # 0.20: UV 공간 활용도
    w_seam * seam_score           # 0.15: Seam continuity (exp decay)
)
```

#### 가중치 설계 근거

| 가중치 | 값 | 근거 |
|--------|-----|------|
| `w_photo` | 0.50 | 3DGS의 L1 weight (0.8) 참고, 주요 품질 지표 |
| `w_ssim` | 0.15 | 3DGS의 SSIM weight (0.2) 참고 |
| `w_coverage` | 0.20 | UV 공간 활용도 (텍스처 완성도) |
| `w_seam` | 0.15 | 텍스처 연속성 (시각적 품질) |

### 6.2 Photometric Metrics 구현

#### PSNR 계산 (Masked)

```python
def compute_psnr_masked(rendered, target, mask):
    """
    Mesh 영역에서만 PSNR 계산.
    배경 제외로 정확한 텍스처 품질 측정.
    """
    mse = np.mean((rendered[mask] - target[mask]) ** 2)
    psnr_db = 10 * np.log10(255.0 ** 2 / mse)

    # [0, 1] 정규화 (15-40 dB 범위)
    psnr_score = np.clip((psnr_db - 15) / (40 - 15), 0, 1)
    return psnr_score, psnr_db
```

#### SSIM 계산 (Bounding Box Crop)

```python
def compute_ssim_masked(rendered, target, mask):
    """
    Bounding box crop으로 효율적 SSIM 계산.
    skimage.metrics.structural_similarity 활용.
    """
    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    ssim_val = ssim(target[y1:y2, x1:x2], rendered[y1:y2, x1:x2],
                   channel_axis=2, data_range=255)
    return ssim_val
```

### 6.3 6-View Projection Grid 시각화

#### Grid 레이아웃

```
┌─────────────┬─────────────┬─────────────┐
│ View 0      │ View 1      │ View 2      │
│ [Orig|Rend] │ [Orig|Rend] │ [Orig|Rend] │
├─────────────┼─────────────┼─────────────┤
│ View 3      │ View 4      │ View 5      │
│ [Orig|Rend] │ [Orig|Rend] │ [Orig|Rend] │
└─────────────┴─────────────┴─────────────┘
```

- 2행 × 3열 구성
- 각 셀: [Original | Rendered] side-by-side 비교
- 캡션: View ID, PSNR (dB), SSIM 표시

#### WandB 로깅

```python
log_dict['projection_6view'] = wandb.Image(
    grid_path,
    caption=f"6-View Projection (PSNR={psnr:.1f}dB, SSIM={ssim:.3f})"
)
```

### 6.4 WandB Sweep 설정

#### Search Space

| 파라미터 | 범위 | 분포 |
|----------|------|------|
| `visibility_threshold` | [0.1, 0.7] | uniform |
| `uv_size` | 512 (고정) | - |
| `fusion_method` | average, visibility_weighted, max_visibility | categorical |
| `do_optimization` | True/False | categorical |
| `opt_iters` | [30, 50, 100] | categorical |
| `w_tv` | [1e-5, 1e-2] | log_uniform |

#### 2-Stage 최적화 전략

```
Stage A (Structure): do_optimization=False
  └─ visibility_threshold, fusion_method 탐색 (빠른 평가)

Stage B (Refinement): Stage A 최적값 고정
  └─ opt_iters, w_tv, opt_lr 탐색 (미세 조정)
```

---

## 7. 주요 결과 (Key Findings/Results)

### 7.1 WandB Sweep 진행 현황

**프로젝트**: `uvmap-optimization`
**총 Trial 수**: ~60+ runs

#### Sweep 결과 예시

| Run Name | Score | PSNR (dB) | SSIM | Coverage (%) |
|----------|-------|-----------|------|--------------|
| `ancient-sweep-34` | 0.72 | 28.5 | 0.82 | 94.2 |
| `autumn-sweep-57` | 0.68 | 27.1 | 0.79 | 91.8 |
| `comic-sweep-66` | 0.65 | 26.3 | 0.77 | 89.5 |

### 7.2 시각화 개선

**이전**: 단일 view render (front, side, diagonal)
**현재**: 6-view projection grid (원본 vs 렌더링 직접 비교)

#### 변경 사항

```python
# wandb_sweep.py:203
log_rendered_mesh: bool = False  # 비활성화 (grid로 대체)
log_projection_grid: bool = True  # 활성화
```

**장점**:
1. 모든 view에서의 품질을 한눈에 비교
2. 원본과 렌더링을 직접 대조
3. PSNR/SSIM 수치와 시각적 품질 연관성 확인

### 7.3 Score 함수 비교

| 버전 | 구성 요소 | 문제점 |
|------|----------|--------|
| **v2** | coverage, confidence(명칭오류), seam | Photometric 부재 |
| **v3** | photo(PSNR), ssim, coverage, seam | 학술 표준 준수 |

---

## 8. 분석 및 논의 (Analysis & Discussion)

### 8.1 Photometric Score의 효과

**도입 전**: UV texture가 시각적으로 부자연스러워도 높은 score 가능
**도입 후**: 렌더링 결과가 원본과 유사할수록 높은 score

```
Score v3 = 직접적 품질 신호 (rendered vs original)
```

### 8.2 face_sampling 결정

| face_sampling | 품질 | 속도 | 결정 |
|---------------|------|------|------|
| 1 (100%) | 최상 | ~10초 | **채택** |
| 2 (50%) | 양호 | ~5초 | 후보 |
| 5 (20%) | Coarse | ~2초 | 제외 |

**결론**: Photometric 비교 정확도 우선 → `face_sampling=1` 고정

### 8.3 WandB 로깅 최적화

**제거된 항목**:
- `render_front`, `render_side`, `render_diagonal` (개별 view 이미지)

**유지/추가된 항목**:
- `projection_6view` (6-view comparison grid)
- `uv_texture`, `confidence_map`, `uv_mask`
- `mean_psnr_score`, `mean_ssim_score`, `mean_psnr_db`

---

## 9. 미결 과제 (Open Questions)

### 9.1 추가 메트릭 검토

- [ ] **LPIPS**: Perceptual similarity (GPU 필요, 계산 비용 높음)
- [ ] **FID**: Feature-level distribution 비교
- [ ] **View-wise variance**: 다중 view 간 일관성

### 9.2 Sweep 확장

- [ ] Stage A/B 순차 실행 자동화
- [ ] Best config로 최종 UV Map 생성
- [ ] 다른 데이터셋에서 검증

### 9.3 알려진 한계

1. **단일 프레임 평가**: frame_idx=0에서만 PSNR/SSIM 계산
2. **Static mesh 가정**: 동일 포즈의 mesh로 렌더링
3. **RGB만 비교**: 깊이, 법선 등 추가 신호 미활용

---

## 10. 결론 및 권장사항

### 10.1 핵심 결론

1. **Score v3 도입**: PSNR+SSIM 기반 photometric scoring으로 품질 측정 정확도 향상
2. **6-view Grid 시각화**: 정성적 평가의 효율성 및 해석 가능성 증대
3. **Sweep 자동화**: WandB 대시보드를 통한 실시간 모니터링 및 분석

### 10.2 권장 설정

```python
# 최적화된 Score v3 가중치
config = WandBSweepConfig(
    w_photo=0.50,     # PSNR 기반
    w_ssim=0.15,      # Structural similarity
    w_coverage=0.20,  # UV 완성도
    w_seam=0.15,      # 텍스처 연속성

    # 시각화 설정
    log_rendered_mesh=False,  # 개별 view 비활성화
    log_projection_grid=True, # 6-view grid 활성화
    projection_face_sampling=1,  # Full quality
)
```

### 10.3 다음 단계

1. **Sweep 완료 후 Best Config 추출**
2. **최종 UV Map 생성 및 검증**
3. **Cross-dataset 일반화 테스트**

---

## 11. 파일 구조

### 11.1 핵심 코드

```
uvmap/
├── wandb_sweep.py      # Score v3, 6-view grid, WandB 통합
├── uv_pipeline.py      # UV mapping 파이프라인
├── uv_renderer.py      # UV 공간 렌더링
└── texture_sampler.py  # 텍스처 샘플링/융합
```

### 11.2 결과 디렉토리

```
wandb_sweep_results/
├── run_ancient-sweep-34/
│   ├── texture_final.png
│   ├── confidence.png
│   ├── projection_6view_grid.png
│   └── ...
├── run_autumn-sweep-57/
└── ...
```

### 11.3 관련 문서

- `docs/reports/251210_sweep_score_improvement.md`: Score 함수 개선 상세
- `docs/reports/251210_uvmap_score_optimization_strategy.md`: 최적화 전략 수립

---

## 12. 핵심 교훈

1. **변수명 = 실제 의미**: 코드의 변수명이 실제 계산 내용과 일치해야 함
2. **학술 표준 활용**: 3DGS의 검증된 loss 구조 참고
3. **Masked Comparison**: 배경 포함 시 메트릭 왜곡 발생
4. **시각화 = 디버깅 도구**: Grid 비교로 문제점 빠르게 발견 가능

---

*Generated: 2025-12-10*
*Tool: Claude Code (claude-opus-4-5-20251101)*
