# 251210 연구노트 — View/Keypoint Ablation + UV Texture 실험

## 목표
- View 수 / Keypoint 수에 따른 mesh fitting 품질 정량 비교 (Ablation Study)
- UV Map texture score 함수 개선 (v2 → v3: PSNR + SSIM 기반)
- UV Map vs Gaussian Avatar (MoReMouse) 텍스처 표현 방식 비교 분석

## 진행 내용

### 1. View/Keypoint Ablation Study

**Dataset**: markerless_mouse_1_nerf, 100 frames, Baseline = 6V/22KP

#### A. Keypoint Ablation (6-view 고정)

| 실험명 | Keypoints | 설명 |
|--------|-----------|------|
| baseline_6view_keypoint | 22개 | MAMMAL 원본 |
| sparse_9kp_dlc | 9개 [0,1,2,3,4,5,6,8,12] | DeepLabCut 스타일 |
| sparse_7kp_mars | 7개 [0,1,2,3,5,18,21] | MARS 스타일 |
| sparse_5kp_minimal | 5개 [0,1,2,3,5] | 최소 구성 |

#### B. View Ablation (3 core keypoints 고정: nose, neck, tail_base)

| 실험명 | Views | 카메라 배치 |
|--------|-------|-----------|
| sparse_5view | 5 | [0,1,2,3,4] |
| sparse_4view | 4 | [0,1,2,3] 연속 |
| sparse_3view | 3 | [0,2,4] 120도 간격 |
| sparse_2view | 2 | [0,3] 180도 대칭 |

#### View/KP 감소에 따른 보상 전략

| View 수 | Iteration 증가 | Regularization 강화 | Mask Loss 활성화 |
|---------|----------------|---------------------|------------------|
| 6 | 기본값 | 기본값 | Step 2만 |
| 5 | +20% | +20% | Step 1부터 |
| 3-4 | +50% | +50% | Step 0부터 |
| 2 | +100% | +100% | Step 0부터 강화 |

### 2. 정량적 결과 (V2V 기준 정렬)

**Baseline (Pseudo GT)**: 6V/22KP fitted mesh

| 순위 | Views | Keypoints | V2V Mean (mm) | V2V Max (mm) | V2V Std (mm) | V2V Median (mm) | Chamfer (mm) | Hausdorff (mm) |
|------|-------|-----------|---------------|--------------|-------------|-----------------|--------------|----------------|
| 1 | 6 (0-5) | 9 (DLC) | **1.76** | 10.16 | 2.17 | 1.01 | 0.53 | 5.22 |
| 2 | 6 (0-5) | 7 (MARS) | **1.99** | 8.44 | 1.95 | 1.18 | 0.66 | 5.34 |
| 3 | 5 (0-4) | 3 (core) | **4.16** | 28.93 | 4.29 | 2.79 | 1.19 | 9.49 |
| 4 | 3 (0,2,4) | 3 (core) | **5.14** | 28.62 | 4.51 | 3.96 | 1.45 | 10.07 |
| 5 | 4 (0-3) | 3 (core) | **6.38** | 28.68 | 5.20 | 4.63 | 1.55 | 10.35 |
| 6 | 2 (0,3) | 3 (core) | **6.76** | 31.63 | 5.37 | 4.78 | 2.65 | 14.04 |
| 7 | 6 (0-5) | 5 (minimal) | **17.59** | 61.80 | 12.24 | 15.11 | 10.33 | 54.44 |

#### Efficiency Score (V2V Mean x log(Resources+1))

| Config | Resources (VxKP) | V2V Mean | Efficiency |
|--------|------------------|----------|------------|
| 6V/9KP | 54 | 1.76 | 7.04 |
| 6V/7KP | 42 | 1.99 | 7.49 |
| 5V/3KP | 15 | 4.16 | 11.54 |
| 3V/3KP | 9 | 5.14 | 11.85 |

### 3. Ablation 핵심 결론

**Keypoint Ablation**:
- 22→9 (DLC): V2V 1.76mm — **실질적 동등 품질**
- 22→7 (MARS): V2V 1.99mm — 실용적 대안
- 22→5 (minimal): V2V 17.59mm — **품질 급락, 비권장**
- **최소 7개 keypoint 유지 필요**

**View Ablation**:
- **배치 전략이 수보다 중요**: 3V (120도 간격, 5.14mm) > 4V (연속, 6.38mm)
- 최소 3개 이상 권장, 2개는 flip ambiguity 위험
- 5V (4.16mm)가 효율적 차선

**정성적 관찰**:

| 구성 | 전체 형태 | 사지 위치 | 꼬리 추적 | 시간 일관성 |
|------|-----------|-----------|-----------|-------------|
| 6V/22KP (baseline) | 최상 | 최상 | 최상 | 최상 |
| 6V/9KP (DLC) | 최상 | 상 | 최상 | 최상 |
| 6V/7KP (MARS) | 상 | 상 | 상 | 상 |
| 3V/3KP (120도) | 중 | 중 | 중상 | 중 |
| 2V/3KP | 중하 | 하 | 중 | 중하 |

### 4. UV Map Score 함수 개선 (v2 → v3)

**v2 문제점**: `w_psnr` 변수명이지만 실제 `mean_confidence` 사용, photometric loss 부재

**v3 Score 공식**:
```python
score = (
    w_photo * photo_score +       # 0.50: PSNR 기반 (15-40dB → [0,1])
    w_ssim * ssim_score +         # 0.15: structural similarity
    w_coverage * coverage_score + # 0.20: UV 공간 활용도
    w_seam * seam_score           # 0.15: exp(-15 * seam_discontinuity)
)
```

**Coverage Gating**: Coverage < 80% → 전체 score x 0.1

**설계 근거 — 3D Gaussian Splatting (SIGGRAPH 2023)**:
```
3DGS Loss: L = (1-lambda)*L1 + lambda*D-SSIM,  lambda=0.2
```

**구현된 Helper Functions** (`uvmap/wandb_sweep.py`):
- `compute_psnr_masked()`: mesh 영역만 PSNR 계산, [0,1] 정규화
- `compute_ssim_masked()`: bbox crop 후 SSIM 계산
- `create_mesh_mask()`: 흰색 배경 제외한 mesh 영역 마스크

**6-View Projection Grid**: 2x3 grid, 각 셀 [Original | Rendered] 비교 + PSNR/SSIM overlay

**face_sampling 결정**: `face_sampling=1` (100% faces) 고정 — photometric 비교 정확도 우선

### 5. WandB Sweep HPO 설정

**Search Space** (6 parameters):

| 파라미터 | 범위 | 분포 |
|----------|------|------|
| visibility_threshold | 0.1~0.7 | uniform |
| fusion_method | average / visibility_weighted / max_visibility | categorical |
| do_optimization | True/False | categorical |
| opt_iters | 30/50/100 | categorical |
| w_tv | 1e-5 ~ 1e-2 | log_uniform |
| uv_size | 512 (고정) | - |

**2-Stage 전략**:
- Stage A: do_optimization=False → visibility_threshold, fusion_method 탐색
- Stage B: Stage A 최적값 고정 → opt_iters, w_tv 미세 조정

### 6. UV Map vs Gaussian Avatar (MoReMouse) 비교

| 항목 | MAMMAL UV Map | MoReMouse Gaussian Avatar |
|------|---------------|---------------------------|
| Input | 6-view multi-view | 6-view (800 frames) |
| Output | 2D UV texture (256-1024 px) | Per-vertex Gaussian (~250K params) |
| 방법 | Projection fusion | Differentiable optimization (400K steps) |
| Params | O(UV_size^2) ~256K-1M | ~19 params x 13,059 vertices ~250K |
| 처리 시간 | 수 분 (프레임당 ~1초) | 400K iterations |
| Seam | UV seam artifacts 가능 | Seam-free (3D 공간) |
| Pose 변형 | Pose-independent (단일 텍스처) | LBS로 Gaussian 함께 변형 |
| Loss | TV regularization | L1 + SSIM + LPIPS |
| 용도 | 직접 텍스처 렌더링 | 합성 학습 데이터 → single-image 3D recon |

**공통 데이터**: markerless_mouse_1 (Dunn et al., 2021), 6-view, 18K frames, 100FPS

**상호 보완 가능성**:
- MAMMAL UV Map → 빠른 프로토타이핑
- MoReMouse Gaussian → 고품질 학습 데이터 생성

## 핵심 발견
- **6V/9KP (DLC)가 최적 품질/비용 균형**: V2V 1.76mm, baseline 대비 실질적 동등
- **View 배치 > View 수**: 120도 간격 3V (5.14mm) > 연속 4V (6.38mm)
- **5개 이하 keypoint 비권장**: 품질 급락 (17.59mm)
- **Score v3의 PSNR+SSIM 도입**으로 텍스처 품질 직접 측정 가능
- **do_optimization=False가 PSNR 역설적으로 높음**: TV regularization smoothing이 PSNR을 오히려 낮춤

## 권장 설정 (상황별)

```yaml
production_quality:   # 연구/논문용
  views: [0,1,2,3,4,5]
  keypoints: 9  # DLC
  expected_v2v: "< 2mm"

balanced_default:     # 실용적 기본값
  views: [0,1,2,3,4,5]
  keypoints: 7  # MARS
  expected_v2v: "< 2mm"

minimal_viable:       # 저비용 프로토타입
  views: [0,2,4]      # 120도 간격
  keypoints: 3  # core (nose, neck, tail)
  expected_v2v: "~5mm"
```

## 미해결 / 다음 단계
- MPJPE (Mean Per-Joint Position Error) 추가 평가
- View 배치 최적화 (어떤 각도 조합이 최적인지)
- Cross-dataset validation (다른 마우스 데이터에 적용)
- LPIPS 추가 (perceptual loss, GPU 필요)
- do_optimization True/False 분리 리포팅

## 분석 스크립트 및 자료

| 파일 | 설명 |
|------|------|
| `scripts/compare_mesh_ablation.py` | 정량 비교 스크립트 |
| `ablation_quantitative_results.md` | 상세 정량 분석 |
| `ablation_quantitative_results.json` | JSON 메트릭 데이터 |
| `ablation_comparison.png` | 정량 비교 차트 |
| `qualitative_comparison_grid.png` | 정성 비교 그리드 |
| `uvmap/wandb_sweep.py` | Score v3 + 6-view grid + WandB 통합 |

---
*Sources: 251210_mesh_fit_ablation_study.md, 251210_sweep_score_improvement.md, 251210_uvmap_score_optimization_strategy.md, 251210_uvmap_texture_experiment.md, 251210_uvmap_vs_gaussian_avatar_comparison.md, ablation_quantitative_results.md*
