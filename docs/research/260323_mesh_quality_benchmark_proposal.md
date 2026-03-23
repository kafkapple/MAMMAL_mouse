# Rodent Mesh Fitting Quality: Benchmark Proposal

> Navigation: [← Fitting Methods Survey](260323_fitting_methods_survey.md) | [← EXPERIMENTS](../EXPERIMENTS.md)
> Created: 2026-03-23 | Target: NeurIPS 2026 Datasets & Benchmarks

## 1. Problem Statement

### The Gap

Rodent (mouse/rat) 3D mesh fitting은 행동 신경과학의 핵심 인프라이지만,
**mesh surface quality에 대한 체계적 평가가 전무**하다.

| Paper | Species | Quality Metric | Mesh Eval? |
|-------|---------|---------------|------------|
| MAMMAL (2023) | Pig, Mouse | Keypoint error (2.43mm) | ❌ (pig IoU만, mouse 없음) |
| ArMo (2023) | Mouse (head-fixed) | None quantitative | ❌ |
| MoReMouse (2025) | Mouse (synthetic) | PSNR 22.0 (synthetic only) | ⚠️ Synthetic |
| Pose-Splatter (2025) | Mouse, Rat | IoU 0.76, PSNR 29.0 | ⚠️ (3DGS, not mesh) |
| RatBodyFormer (2024) | Rat | Surface point error | ❌ Mesh surface |

### Why This Matters

1. **Rodent = 연구 동물 #1**: 전세계 동물 실험의 ~95%가 설치류
2. **Mesh > Keypoints**: Behavioral phenotyping, social interaction, contact detection은 dense surface 필요
3. **Downstream impact**: FaceLift Neural Texture, Pose-Splatter novel view 등은 mesh quality에 직접 의존
4. **No guidance**: 연구자들이 어떤 fitting config를 사용해야 하는지 기준 없음

## 2. Research Questions

### Q1: Iteration-Quality-Cost Trade-off
- Fitting iteration 수와 mesh surface quality의 관계는?
- 어디서 diminishing returns가 시작되는가?
- Silhouette loss 활성화의 효과는?

### Q2: Temporal Strategy
- Sequential tracking (paper: temporal init) vs Independent fitting (per-frame optimization)?
- Keyframe fitting + interpolation은 어떤 간격에서 acceptable quality를 유지하는가?
- Interpolation 방법별 비교 (linear, slerp, spline)?

### Q3: Cross-method Comparison
- Optimization (MAMMAL) vs Feed-forward (Pose-Splatter) mesh quality 비교
- 속도-품질 Pareto frontier는?
- Hybrid approach의 실현 가능성?

### Q4: Evaluation Protocol
- Mouse mesh quality 측정에 적합한 metric 조합은?
- Silhouette IoU만으로 충분한가, 아니면 textured PSNR/LPIPS도 필요한가?
- View-dependent vs view-independent metrics?

## 3. Proposed Benchmark Framework

### 3.1 Metrics (Multi-level)

| Level | Metric | Measures | Computation |
|-------|--------|----------|-------------|
| **L1: Silhouette** | IoU, Boundary F1 | Shape alignment | Render silhouette → compare with GT mask |
| **L2: Appearance** | PSNR, SSIM, LPIPS | Textured surface quality | Render textured mesh → compare with GT image |
| **L3: Geometry** | Chamfer distance, Surface normal consistency | 3D accuracy | Compare with pseudo-GT (if available) |
| **L4: Temporal** | Jerk (3rd derivative), Smoothness | Motion naturalness | Parameter trajectory analysis |
| **L5: Downstream** | Novel view PSNR, Action recognition accuracy | Task relevance | End-to-end pipeline evaluation |

### 3.2 Configs to Benchmark

| Config | step1 | step2 | mask | Mode | Speed |
|--------|-------|-------|------|------|-------|
| paper | 5 | 3 | 0 | Sequential | ~2s |
| paper+mask | 5 | 3 | 3000 | Sequential | ~3s |
| fast | 50 | 15 | 3000 | Independent | ~3min |
| default | 100 | 30 | 3000 | Independent | ~7min |
| accurate | 200 | 50 | 3000 | Independent | ~14min |
| ultra | 400 | 100 | 3000 | Independent | ~28min |
| sweep best | TBD | TBD | TBD | Independent | TBD |

### 3.3 Interpolation Benchmark

| Interval (M5) | Time gap | Keyframes/3600 | Fitting cost (4 GPU) |
|---------------|----------|----------------|---------------------|
| 1 (all frames) | 0.05s | 3600 | baseline |
| 2 | 0.10s | 1800 | 50% |
| 4 | 0.20s | 900 | 25% |
| 6 | 0.30s | 600 | 17% |
| 12 | 0.60s | 300 | 8% |
| 24 | 1.20s | 150 | 4% |

Compare: Linear lerp, Slerp (rotations), Cubic spline

### 3.4 Cross-method Comparison

| Method | Type | What we measure |
|--------|------|----------------|
| MAMMAL (configs above) | Optimization | Full config sweep |
| Pose-Splatter | Feed-forward | If code released |
| MAMMAL + PS init (hybrid) | Hybrid | PS output → MAMMAL 2-3 iter refine |

## 4. Data & Assets (Already Available)

| Asset | Description | Status |
|-------|-------------|--------|
| 6-view video | 18K frames, 100fps, 1152×1024 | ✅ |
| GT masks | SimpleClick segmentation, 6 views | ✅ |
| 2D keypoints | 22 joints, 6 views | ✅ |
| 3D keypoints (DANNCE) | 22 joints, 3600 frames | ✅ |
| MAMMAL mouse model | 14,522v, 140 joints, UV texture | ✅ |
| Baseline fast fitting | 3600 frames keypoints | ✅ |
| Accurate fitting (partial) | 23 bad + 200 dense (in progress) | 🔄 |
| Parameter sweep | 9 configs × 5 frames (in progress) | 🔄 |
| Comparison module | IoU + textured overlay + 6-view grid | ✅ |
| Interpolation analysis | 3600-frame keypoint-based | ✅ |

## 5. Expected Contributions

1. **First systematic benchmark** of mouse mesh fitting quality (no prior work)
2. **Config guideline**: Optimal iteration-quality-cost operating point for rodent mesh fitting
3. **Interpolation strategy**: Validated keyframe interval + method recommendation
4. **Evaluation protocol**: Multi-level metric suite for rodent mesh quality
5. **Reproducible code + data**: Open benchmark for community

## 6. Timeline (Preliminary)

| Phase | Duration | Content |
|-------|----------|---------|
| Data collection | 1 week | Complete all fitting experiments (E2-E4 + full config sweep) |
| Analysis | 1 week | Trade-off curves, interpolation validation, cross-method comparison |
| Writing | 2 weeks | NeurIPS format paper |
| **Total** | **4 weeks** | Target: NeurIPS 2026 Datasets & Benchmarks deadline |

## 7. Risks

| Risk | Mitigation |
|------|-----------|
| Pose-Splatter code not released | Use our existing PS results from FaceLift comparison project |
| Single dataset (markerless_mouse_1) | Add Rat7M if time permits |
| "Just an ablation study" criticism | Frame as benchmark + evaluation protocol contribution |
| Compute cost for full sweep | 4 GPU parallel, keyframe + interpolation strategy |

---

*Created: 2026-03-23 | MAMMAL Mesh Quality Benchmark*
