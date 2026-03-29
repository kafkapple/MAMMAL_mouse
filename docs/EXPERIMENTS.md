# MAMMAL Fitting Experiments

> Navigation: [← MOC](MOC.md)
> Updated: 2026-03-23

## Overview

MAMMAL mesh fitting 품질 개선 + 3,600프레임 스케일링을 위한 실험 계획.

## Optimization Configs (SSOT)

4개 config 파일: `conf/optim/{name}.yaml`

| Config | step0 | step1 | step2 | mask_step2 | 용도 | 속도 |
|--------|-------|-------|-------|------------|------|------|
| **paper** | 60 | **5** | **3** | **0** (off) | 논문 원본. Sequential tracking, temporal init 의존 | ~2s/frame |
| **paper_fast** | 60 | **5** | **3** | **0** (off) | = paper + render off | ~1.5s/frame |
| **fast** | 5 | **50** | **15** | 3000 | 독립 프레임 테스트. Silhouette loss 활성 | ~3min/frame |
| **accurate** | 20 | **200** | **50** | 3000 | 고품질 독립 피팅. 4× iterations vs fast | ~14min/frame |
| *(default)* | 10 | 100 | 30 | 3000 | 기본값 (config 미지정 시) | ~7min/frame |

### 핵심 차이

```
paper/paper_fast: 논문 방식. Temporal smoothness로 이전 프레임 결과를 초기값으로 사용.
                  매 프레임 5회 iteration만으로 추적. mask loss OFF → 빠르지만 silhouette 불일치.
                  Sequential 실행 필수 (temporal init 의존).

fast:             독립 프레임 피팅. mask loss ON (3000). 각 프레임 개별 수렴.
                  iteration 부족 → 23% 프레임 IoU < 0.7.

accurate:         독립 프레임 고품질. fast 대비 4배 iteration.
                  IoU 평균 +0.21 개선. 비용: 14min/frame.
```

### 원본 3600프레임 fitting (E1)

**Config: `paper_fast`** (step1=5, mask=0)
- 논문 설정으로 전체 3600프레임 sequential 피팅
- 속도: ~2s/frame × 3600 = ~2시간
- 결과: OBJ/params 삭제됨, `keypoints_22_3d.npz`만 잔존
- 100프레임 PoC subset (step=24)의 IoU: mean=0.795, 11/100 bad

## Dataset

| 항목 | 값 |
|------|---|
| Video | 18,000 frames, 100fps, 6 cameras |
| M5 dataset | 3,600 frames (video interval=5) |
| PoC subset | 100 frames (M5 step=24, video step=120) |
| Body length ref | 66.2mm (nose-to-body) |

## Experiment Registry

### E1: Baseline Fast (completed, 2026-01-26)

| 항목 | 값 |
|------|---|
| Config | `fast` (step1=50, step2=15) |
| Frames | 3,600 (full M5, interval=5) |
| Output | `results/fitting/baseline_fast_3600/` |
| Artifacts | `keypoints_22_3d.npz` only (params/obj deleted) |
| IoU (100 subset) | Mean=0.795, 11/100 bad (<0.7) |
| Worst | 9480(0.548), 9360(0.566), 5520(0.585) |

### E2: Refit Accurate 23 (in progress)

| 항목 | 값 |
|------|---|
| Config | `accurate` (step1=200, step2=50) |
| Frames | 23 bad frames (IoU < 0.7 from good_frames.npy) |
| GPU | 4 (A6000) |
| Output | `results/fitting/refit_accurate_23/` |
| Speed | ~14 min/frame |
| **Result** | **Mean IoU 0.689→0.840 (+0.151), 23/23 pass** |
| Status | ✅ Completed (19,232s total, 836s/frame) |
| Report | `docs/reports/260323_mesh_refit_experiment_report.md` |

### E3: Parameter Sweep ✅

| 항목 | 값 |
|------|---|
| Configs | step1=[100,200,400] × mask_step2=[1000,3000,5000] = 9 |
| Frames | Worst 5 (9480, 9360, 5520, 1320, 8400) |
| Output | `results/fitting/sweep_s1_{N}_m_{M}/`, `results/comparison/sweep/sweep_iou.json` |
| **Best** | **s1_400_m_3000 (mean=0.820)** — only +0.004 vs current accurate |
| **Finding** | **mask_step2=3000 is critical** (m=1000 → -10%p). step1 has diminishing returns |
| Status | ✅ Completed |

### E4: Dense Accurate ✅

| 항목 | 값 |
|------|---|
| Config | `accurate` (step1=200, step2=50) |
| Frames | M5 0-199 (200 consecutive, interval=5) |
| Output | `results/fitting/dense_accurate_0_100/`, `dense_accurate_100_200/` |
| **Finding** | Interpolation error same for fast vs accurate → **dominated by mouse motion nonlinearity** |
| **Optimal interval** | **4 M5 frames (0.2s): 7.1% body, 900 keyframes, ~52h/4GPU** |
| Status | ✅ Completed |

### E5: Production 900-Keyframe Fitting ✅

| 항목 | 값 |
|------|---|
| Config | `accurate` (step1=200, step2=50) |
| Frames | 900 keyframes (video interval=20, M5 interval=4) |
| GPU | 4 (Part1), 5 (Part2), 6 (Part3), 7 (Part4) — 225 keyframes each |
| Output | `results/fitting/production_keyframes_part{1-4}/` → merged `production_900_merged/` |
| **Progress** | P1 ✅ P2 ✅ P3 ✅ P4 ✅ (100%) |
| **Merge** | `results/fitting/production_900_merged/` ✅ |
| **Slerp 3600** | `results/fitting/production_3600_slerp/obj/` — 3600 OBJ ✅ |
| **Videos** | `results/comparison/production_3600_slerp/` — 7 mesh-only views ✅ |
| **GT overlay** | `results/comparison/production_3600_slerp_gt/` — in progress |
| **Status** | ✅ Completed (fitting + slerp). GT overlay rendering. |
| **Next** | UV texture transplant → FaceLift handoff |

### Visualization Outputs

| Output | Path | Content |
|--------|------|---------|
| Refit comparison | `results/comparison/refit_23/` | IoU chart, best/worst 6-view, videos |
| 100-frame sequence | `results/comparison/sequence/` | 6-view grid + per-view videos |
| Interpolated smooth | `results/comparison/sequence_interpolated/` | 397fr vertex lerp |
| Dense smooth | `results/comparison/sequence_dense/` | M5 0-99 at 20fps |

## Analysis Tools

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/baseline_iou_all.py` | 전체 IoU 베이스라인 | `CUDA_VISIBLE_DEVICES=5 python scripts/baseline_iou_all.py` |
| `scripts/compare_refit.py` | A vs B 비교 (silhouette + textured + 6view) | `python scripts/compare_refit.py --views 0 1 2 3 4 5` |
| `scripts/sweep_fitting_params.sh` | 파라미터 sweep | `bash scripts/sweep_fitting_params.sh [GPU]` |
| `scripts/dense_accurate_fitting.sh` | 연속 구간 dense fitting | `bash scripts/dense_accurate_fitting.sh <GPU> <START> <END>` |
| `scripts/refit_bad_frames.sh` | Bad frame 재피팅 | `bash scripts/refit_bad_frames.sh` |

## Comparison Module

`mammal_ext/visualization/mesh_comparison.py`:
- `MeshComparison.compare()` → silhouette IoU + overlay images
- `_render_textured()` → OpenCV camera-matched textured rendering
- `_build_6view_grid()` → 6-view silhouette comparison
- `_build_6view_textured_grid()` → 6-view GT vs textured comparison

## Result Structure

```
results/
├── fitting/
│   ├── baseline_fast_3600/        # E1: keypoints only
│   ├── refit_accurate_23/         # E2: 23 bad frames
│   ├── sweep_s1_*_m_*/            # E3: parameter sweep
│   └── dense_accurate_{S}_{E}/    # E4: interpolation analysis
├── comparison/
│   ├── baseline_iou/              # IoU for all 100 frames
│   ├── fast_vs_accurate/          # Single-view comparison
│   ├── fast_vs_accurate_6view/    # 6-view comparison
│   ├── interpolation/             # E4 interpolation analysis
│   ├── sweep/                     # E3 sweep IoU comparison
│   └── refit_23_final/            # E2 full comparison (6-view + textured)
├── docs/
│   ├── EXPERIMENTS.md             # This file
│   └── reports/260323_mesh_refit_experiment_report.md  # Full report
```

## Interpolation Analysis (E4 accurate data, vertex-level)

| M5 Interval | Time gap | Mean (mm) | P95 (mm) | Mean % body | P95 % body | Keyframes | 4GPU hours |
|-------------|----------|-----------|----------|-------------|------------|-----------|------------|
| 2 | 0.10s | 2.40 | 4.79 | 3.8% | 7.5% | 1800 | 105h |
| **4** | **0.20s** | **4.52** | **9.30** | **7.1%** | **14.6%** | **900** | **52h** |
| **6** | **0.30s** | **10.2%** | **30.4%** | **600** | **35h** |
| 12 | 0.60s | 19.5% | 52.8% | 300 | 17.5h |
| 24 | 1.20s | 34.7% | 86.7% | 150 | 8.8h |

*Error as % of body length (66.2mm). Based on fast fitting — accurate expected ~50-70% of these values.*

**E4 완료 후 accurate 기준 실측으로 업데이트 예정.**

## Key Findings

1. `accurate` config (4× iterations) dramatically improves IoU: +0.21 average on bad frames
2. 100프레임 중 실제 bad (IoU<0.7)은 11개 (핸드오프의 23개와 불일치)
3. 3,600프레임 전체 accurate fitting은 비현실적 (210h/GPU) → keyframe + interpolation 필수
4. MAMMAL parameters (thetas, trans, rotation, scale) 모두 interpolation 가능

## Related Research

> Detailed survey: [research/260323_fitting_methods_survey.md](research/260323_fitting_methods_survey.md)

- MAMMAL paper config (step1=5)는 속도 최적화 → 품질 ablation 미수행
- Pose-Splatter (NeurIPS 2025): feed-forward ~30ms, 우리 비교 프로젝트 진행 중
- Hybrid approach (feed-forward init + LBFGS refine): 장기적 최적 전략

---

*Created: 2026-03-23*
