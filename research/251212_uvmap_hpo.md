# 251212 연구노트 — UV Map HPO Score 설계 및 TV Regularization 역설

## 목표
- UV Map hyperparameter search space (6 parameters) 정리 및 분석
- Score v3 설계 근거 명시 (3DGS 문헌 기반)
- do_optimization=False 수렴 현상 (TV regularization paradox) 분석

## 진행 내용

### 1. Hyperparameter Search Space (6개)

| 파라미터 | 범위 | 분포 | 설명 |
|---------|------|------|------|
| `visibility_threshold` | 0.1~0.7 | uniform | 텍스처 샘플링 visibility 임계값 |
| `uv_size` | 256/512/1024 | categorical | UV 텍스처 해상도 (기본 512 고정) |
| `fusion_method` | average / visibility_weighted / max_visibility | categorical | 다중 뷰 색상 합성 방법 |
| `do_optimization` | True/False | categorical | Photometric 최적화 수행 여부 |
| `opt_iters` | 30/50/100 | categorical | 최적화 반복 횟수 |
| `w_tv` | 1e-5 ~ 1e-2 | log-uniform | Total Variation 정규화 가중치 |

### 2. 2-Stage Optimization 전략

| Stage | do_optimization | 탐색 파라미터 | 목적 |
|-------|----------------|---------------|------|
| **stage_a** | False 고정 | visibility_threshold, fusion_method | 구조 파라미터 빠른 탐색 |
| **stage_b** | True 고정 | opt_iters, w_tv, opt_lr | Stage A 결과 기반 미세 조정 |
| full | True/False 탐색 | 전체 | 기존 동시 탐색 방식 |

### 3. Score v3 설계 근거

**3DGS (SIGGRAPH 2023) 대응**:

| 3DGS 요소 | UV Score 대응 | 가중치 |
|-----------|---------------|--------|
| L1 loss | PSNR (photo) | **0.50** |
| D-SSIM (lambda=0.2) | SSIM | **0.15** |
| - | Coverage (UV 특화) | **0.20** |
| - | Seam (UV 특화) | **0.15** |

**PSNR 정규화**: 15~40 dB → [0, 1] (clip)
**Seam score**: `exp(-15 * seam_discontinuity)`
**Coverage gating**: coverage < 80% → score x 0.1

> **주의**: 현재 가중치 0.50/0.15/0.20/0.15는 휴리스틱. 3DGS 원본 lambda=0.2와 직접 대응은 아님.

### 4. do_optimization=False 수렴 현상 (TV Regularization Paradox)

#### 현상

Sweep 결과 (sweep-gm4e24p6):
- do_optimization=True: 4개 (27%)
- do_optimization=False: 11개 (73%)

Bayesian optimizer가 **False 쪽으로 수렴**.

#### 원인 분석

**파이프라인 비교**:
```
False: Input Images → Visibility Weighted Fusion → UV Texture
       (원본 색상 그대로 복사)

True:  Input Images → Fusion → Photometric Optimization (TV reg) → UV Texture
       (TV regularization으로 smoothing)
```

**PSNR 역설**:
```
원본:   [120, 122, 119, 121] (약간의 노이즈 포함)
False:  [120, 122, 119, 121] (그대로 복사) → PSNR 높음
True:   [120, 120, 120, 120] (smoothed)   → PSNR 낮음 (MSE 증가)
```

TV regularization이 텍스처를 **시각적으로 개선**하지만, **PSNR은 오히려 낮아짐**.

**Bayesian optimizer 수렴 패턴**:
```
Trial 1: True, score=0.72
Trial 2: False, score=0.75  ← 더 높음
Trial 3: True, score=0.71
Trial 4: False, score=0.76
→ "False가 더 좋네" → False 쪽 탐색 집중
```

#### 평가

| 관점 | False | True |
|------|-------|------|
| 수치적 품질 (PSNR) | **높음** | 낮을 수 있음 |
| 시각적 품질 | 노이즈 있음 | **깨끗함** |
| Seam 품질 | 경계 불연속 | **완화됨** |
| 계산 시간 | **빠름** | 느림 |

**결론**: 현재 Score가 수치적 PSNR을 과대평가하고 있을 가능성

### 5. 개선 방향 4가지

**Option A: SSIM 비중 증가**
```python
# 현재:  w_photo=0.50, w_ssim=0.15
# 개선:  w_photo=0.35, w_ssim=0.30  (perceptual quality 강화)
```
SSIM은 smoothing에 덜 민감하여 perceptual quality 반영 개선

**Option B: 2-Stage 분리 평가**
```bash
python -m uvmap.wandb_sweep --stage stage_a --count 20  # 구조만
python -m uvmap.wandb_sweep --stage stage_b --count 20  # True 고정 미세 조정
```

**Option C: LPIPS 추가**
- LPIPS (Learned Perceptual Image Patch Similarity): human perception과 높은 correlation
- 단점: GPU 필요, 계산 비용 높음

**Option D: do_optimization 분리 리포팅**
- Best of False vs Best of True 별도 비교

## 핵심 발견
- **TV regularization paradox**: smoothing이 시각적 품질은 개선하지만 PSNR은 낮춤
- Bayesian optimizer가 PSNR 높은 do_optimization=False로 수렴하는 것은 score 설계 한계
- **PSNR만으로 텍스처 품질을 측정하면 over-fitting 위험**: perceptual metric (SSIM, LPIPS) 비중 증가 필요
- 2-Stage separation (구조 파라미터 vs optimization 파라미터)이 공정한 비교에 유리

## 미해결 / 다음 단계
- SSIM 비중 증가한 Score v3.1 실험
- LPIPS 도입 가능성 검토 (GPU 비용 vs 정확도)
- do_optimization True/False 별도 best config 비교
- Stage A/B 순차 실행 자동화

---
*Sources: 251212_uvmap_hpo_score_design.md*
