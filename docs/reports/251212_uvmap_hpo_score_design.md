# UV Map Hyperparameter Optimization: Score Design & Search Space

- **날짜**: 2025-12-12
- **주제**: UV Map HPO의 파라미터 공간 및 Score 설계 분석
- **목적**: Score 디자인 근거 명시, do_optimization 수렴 현상 분석

---

## 1. Hyperparameter Search Space

### 1.1 주요 파라미터 (6개)

| 파라미터 | 범위 | 분포 | 설명 |
|---------|------|------|------|
| `visibility_threshold` | 0.1 ~ 0.7 | uniform | 텍스처 샘플링 시 visibility 임계값 |
| `uv_size` | 256/512/1024 | categorical | UV 텍스처 해상도 (기본 512 고정) |
| `fusion_method` | average/visibility_weighted/max_visibility | categorical | 다중 뷰 색상 합성 방법 |
| `do_optimization` | True/False | categorical | Photometric 최적화 수행 여부 |
| `opt_iters` | 30/50/100 | categorical | 최적화 반복 횟수 |
| `w_tv` | 1e-5 ~ 1e-2 | log-uniform | Total Variation 정규화 가중치 |

### 1.2 2-Stage Optimization 전략

| Stage | do_optimization | 탐색 파라미터 | 목적 |
|-------|-----------------|---------------|------|
| `stage_a` | False 고정 | visibility_threshold, fusion_method | 구조 파라미터 빠른 탐색 |
| `stage_b` | True 고정 | opt_iters, w_tv, opt_lr | Stage A 결과 기반 미세 조정 |
| `full` | True/False 탐색 | 전체 | 기존 방식 (동시 탐색) |

---

## 2. Score Design (v3 - Photometric-Aware)

### 2.1 Score 수식

```
Score = w_photo × PSNR_score + w_ssim × SSIM_score + w_coverage × Coverage_score + w_seam × Seam_score
```

### 2.2 가중치 설정

| 메트릭 | 가중치 | 범위 | 설명 |
|--------|--------|------|------|
| `w_photo` (PSNR) | **0.50** | [0, 1] | PSNR 15~40dB → [0,1] 정규화 |
| `w_ssim` | 0.15 | [0, 1] | Structural Similarity Index |
| `w_coverage` | 0.20 | [0, 1] | UV 공간 커버리지 (%) / 100 |
| `w_seam` | 0.15 | [0, 1] | exp(-15 × seam_discontinuity) |

**Coverage Gating**: Coverage < 80% → 전체 score × 0.1 (페널티)

### 2.3 설계 근거: 3D Gaussian Splatting (SIGGRAPH 2023)

```
3DGS Loss: L = (1-λ)×L1 + λ×D-SSIM,  λ=0.2
```

| 3DGS 요소 | UV Map Score 대응 | 비고 |
|-----------|-------------------|------|
| L1 loss | PSNR (w=0.50) | 픽셀 단위 reconstruction |
| D-SSIM (λ=0.2) | SSIM (w=0.15) | Perceptual quality |
| - | Coverage (w=0.20) | UV mapping 특화 |
| - | Seam (w=0.15) | UV mapping 특화 |

> **주의**: 현재 가중치 0.50/0.15/0.20/0.15는 **휴리스틱**. 3DGS 원본 λ=0.2와 직접 대응되지 않음.

---

## 3. do_optimization=False 수렴 현상 분석

### 3.1 현상

Sweep 실행 결과 (sweep-gm4e24p6):
```
do_optimization=True:  4개 (27%)
do_optimization=False: 11개 (73%)
```

Bayesian optimizer가 False 쪽으로 수렴.

### 3.2 원인 분석

#### Score 구조
```
Photometric (PSNR + SSIM) = 65%  ← 지배적
UV Quality (Coverage + Seam) = 35%
```

#### do_optimization 파이프라인 비교

```
do_optimization=False:
  Input Images → Visibility Weighted Fusion → UV Texture
                 (단순 가중 평균, 원본 색상 유지)

do_optimization=True:
  Input Images → Visibility Weighted Fusion → Photometric Optimization → UV Texture
                                              (TV regularization 포함)
```

#### TV Regularization의 부작용

| 요인 | False | True |
|------|-------|------|
| TV Regularization | 없음 | 있음 (blurring 효과) |
| 픽셀 값 | 원본 색상 그대로 | 최적화로 변형됨 |
| 노이즈 | 원본 노이즈 유지 | Smoothing으로 제거 |

**PSNR 역설**: TV regularization이 텍스처를 부드럽게 만들어 **시각적으로는 개선**되지만, **PSNR은 오히려 낮아질 수 있음**.

```
예시:
원본 이미지: [120, 122, 119, 121]  (약간의 노이즈)
False 결과:  [120, 122, 119, 121]  (그대로 복사) → PSNR 높음
True 결과:   [120, 120, 120, 120]  (smoothed)   → PSNR 낮음 (MSE 증가)
```

### 3.3 Bayesian Optimizer 수렴 패턴

```
Trial 1: do_opt=True,  score=0.72
Trial 2: do_opt=False, score=0.75  ← 더 높음
Trial 3: do_opt=True,  score=0.71
Trial 4: do_opt=False, score=0.76
...
→ Bayesian: "False가 더 좋네" → False 쪽으로 탐색 집중
```

### 3.4 평가: 이게 문제인가?

| 관점 | False | True |
|------|-------|------|
| 수치적 품질 (PSNR) | 높음 | 낮을 수 있음 |
| 시각적 품질 | 노이즈 있음 | 깨끗함 |
| Seam 품질 | 경계 불연속 있음 | 완화됨 |
| 계산 시간 | 빠름 | 느림 |

**결론**: 현재 Score 설계가 **수치적 PSNR을 과대평가**하고 있을 가능성.

---

## 4. 개선 방향 제안

### 4.1 Option A: SSIM 비중 증가

```python
# 현재
w_photo: 0.50, w_ssim: 0.15  # PSNR 중심

# 개선안
w_photo: 0.35, w_ssim: 0.30  # Perceptual quality 강화
```

**근거**: SSIM은 structural similarity를 측정하여 smoothing에 덜 민감.

### 4.2 Option B: 2-Stage 분리 평가

```bash
# Stage A: 구조 파라미터만 (do_optimization=False)
python -m uvmap.wandb_sweep --stage stage_a --count 20

# Stage B: 미세 조정 (do_optimization=True 고정)
python -m uvmap.wandb_sweep --stage stage_b --stage_a_config best_config.json --count 20
```

**근거**: 서로 다른 성격의 파라미터 분리하여 공정한 비교.

### 4.3 Option C: LPIPS 추가 (Perceptual Loss)

```python
# LPIPS: Learned Perceptual Image Patch Similarity
# 딥러닝 기반 perceptual metric (human perception과 더 일치)

Score = w_psnr × PSNR + w_lpips × (1 - LPIPS) + w_coverage × Coverage + w_seam × Seam
```

**근거**: PSNR/SSIM보다 human perception과 correlation이 높음.

### 4.4 Option D: do_optimization 분리 리포팅

```python
# Sweep 결과를 do_optimization 별로 분리 분석
# Best of False vs Best of True 비교
```

---

## 5. 참고 자료

- **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023)
  - Loss: L = (1-λ)×L1 + λ×D-SSIM, λ=0.2
  - https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **LPIPS** (Zhang et al., CVPR 2018)
  - Learned Perceptual Image Patch Similarity
  - https://github.com/richzhang/PerceptualSimilarity

---

## 6. 관련 파일

- `uvmap/wandb_sweep.py`: HPO 구현 (Score v3)
- `uvmap/wandb_sweep.py:738-817`: `_compute_score()` 함수
- `uvmap/wandb_sweep.py:219-243`: DEFAULT_SWEEP_PARAMS
