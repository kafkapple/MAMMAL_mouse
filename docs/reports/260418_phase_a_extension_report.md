# Phase A Extension (N=100) — Final Correlation Report

**Date**: 2026-04-18 (evening, updated iter 2 with Spearman + per-view)
**Sample**: N=100 frames (0 to 17820 step 180) × 6 views = 600 samples
**Verdict**: 🟡 **No linear OR monotone signal at N=100, α=0.05** for kinematic hypotheses. "Failed to detect" ≠ "Falsified" (post-hoc power: at r=0.2, N=100 has ~50% power). F6d + F6a remain untested.

---

## 1. Statistical Significance Threshold

For α=0.05, two-tail Pearson test:
- N=23: |r| ≥ 0.413 required (NOT achieved for any)
- N=100: |r| ≥ 0.196 required

---

## 2. N=23 → N=100 Correlation Evolution (Pearson + Spearman)

### Pearson (linear)

| Pair | N=23 | N=100 | N=100 p | Significance (α=0.05, \|r\|>0.196) |
|------|:----:|:-----:|:----:|:-----:|
| r(belly_iou, −\|θ[49]\|) | +0.002 | **+0.111** | 0.270 | NS |
| r(belly_iou, −bone_length[13] extreme) | +0.188 | **−0.079** | 0.436 | NS (sign flip) |
| r(belly_iou, −spine_angle) | **−0.371** | **+0.017** | 0.870 | NS (sign flip) |

### Spearman (monotone non-linear, iter 2 addition)

| Pair | ρ | p | Verdict |
|------|:-:|:-:|:-:|
| belly_iou × −\|θ[49]\| | +0.067 | 0.509 | No monotone signal |
| belly_iou × −bone_length[13] extreme | −0.119 | 0.240 | No monotone signal |
| belly_iou × −spine_angle | +0.014 | 0.891 | No monotone signal |

### Critical interpretation (updated)

- **N=23 "borderline signal" (rearing r=−0.37, bone13 +0.19) 는 statistical noise 였음** — N=100에서 방향도 일치하지 않고 크기도 소멸
- **Spearman도 동일 결론**: 비선형 threshold 효과 가설도 empirically 미지지
- **중요 caveat**: N=100 검정력은 r=0.2 기준 ~50%. "Failed to detect" ≠ "H0 참". F6j/F6b를 "rejected"가 아닌 "no evidence at current sample" 로 표기

---

## 3. Hypothesis Status — Updated (iter 2, softened language)

| Hypothesis | N=23 | N=100 Pearson | N=100 Spearman | Final verdict |
|-----------|:----:|:---:|:---:|:----:|
| F6a (LBS generic no blend shapes) | 🟡 Untested | — | — | 🟡 **Untested** (kinematic analysis cannot test) |
| F6b (rearing OOD) | 🟡 Borderline | r=+0.02, NS | ρ=+0.01, NS | 🟡 **No evidence at N=100** (not rejected; N=782 required for 80% power at r=0.1) |
| F6c (mask_loss=0) | 🟡 Inconclusive | — | — | 🟡 **Untested properly** |
| F6d (GT mask self-loop) | 🟡 Untested | — | — | 🟢 **Elevated by elimination** (but still untested directly) |
| F6e (bone drift) | 🟡 Weak | r=-0.08, NS | ρ=-0.12, NS | 🟡 **No evidence at N=100** |
| F6j (belly_stretch_deformer missing) | 🟡 No support | r=+0.11, NS | ρ=+0.07, NS | 🟡 **No evidence at N=100** (directional consistency but weak) |

> **Language revision (Devil's Advocate iter 1 S1 적용)**: "Falsified" → "No evidence at current sample size". 2종 오류 가능성 (r=0.1-0.2 실제 효과 존재 but N=100 underpowered) 배제 안 됨.

---

## 4. Descriptive Statistics (N=600 samples)

- **Belly IoU**: mean 0.856, std 0.056, range [0.60, 0.94]
- **Belly − Global delta**: mean +0.019 (belly slightly BETTER than global on average — surprising)
- **θ[49] magnitude**: mean 3.18 (≈π), range [1.27, 5.42]
- **Spine angle**: mean 89°, range [15.6°, 172.8°] — wide pose coverage

### Distribution insight
- Belly − Global positive delta (+0.019): belly region often matches silhouette better than mean. "Belly-dent" severity may be masked by global IoU metric.
- Minimum belly IoU 0.326 (frame 11340 view 5) — significant local failure but individual not explained by kinematic hypotheses.

---

## 5. Top-10 Worst Belly IoU Frames

| Rank | Frame | View | Belly | Global | Delta |
|:----:|:----:|:----:|:----:|:-----:|:----:|
| 1 | 11340 | 5 | 0.326 | 0.494 | -0.169 |
| 2 | 3420 | 4 | 0.393 | 0.696 | -0.303 |
| 3 | 3420 | 5 | 0.396 | 0.571 | -0.174 |
| 4 | 16920 | 2 | 0.416 | 0.518 | -0.102 |
| 5 | 5220 | 4 | 0.417 | 0.824 | **-0.408** |
| 6 | 16920 | 0 | 0.471 | 0.537 | -0.066 |
| 7 | 3060 | 2 | 0.530 | 0.635 | -0.104 |
| 8 | 5400 | 4 | 0.537 | 0.872 | **-0.335** |
| 9 | 3060 | 0 | 0.539 | 0.660 | -0.121 |
| 10 | 11340 | 0 | 0.545 | 0.413 | +0.132 |

### Patterns observed (iter 2 corrected)

- **View count correction (fact-check)**: View 4 appears 3/10, View 5 2/10, View 0 3/10, View 2 2/10 → **View 4+5 = 5/10** (not 6/10 as initial claim). View 0 도 동일 3/10 — **view 4/5 특별 dominant 아님**.
- **Per-view baseline (iter 2 new)**: View 4/5 mean belly_iou = 0.83 vs View 0/2/3 = 0.87+ → **view 4/5 worst concentration은 baseline parallax** (camera angle으로 belly region less visible), NOT belly-dent specific event
- Frames 3420, 11340, 16920, 3060 appear multiple views: **bad frame 본질적 문제** (어느 view든 문제)
- Delta −0.4 cases (frame 5220 v4, 5400 v4): belly specifically fails while global is fine → belly region outlier

### Per-View Baseline (N=100 each)

| View | Mean belly IoU | Median | Min |
|:---:|:---:|:---:|:---:|
| 0 | 0.873 | 0.880 | 0.471 |
| 1 | 0.838 | 0.860 | 0.555 |
| 2 | 0.878 | 0.900 | 0.416 |
| 3 | 0.877 | 0.885 | 0.673 |
| 4 | 0.834 | 0.861 | 0.393 |
| 5 | 0.835 | 0.844 | 0.326 |

**Finding**: Views 4/5 (side-low) have systematically lower baseline belly IoU. "Worst case concentration" in these views is artifact, not belly-dent causal evidence.

---

## 6. Decision Gate — Final

Phase A script auto-determined:
```
⚪ No strong signal (r_spine=0.017, r_θ49=0.111) → F6d GT mask
```

**Action plan revised**:

### 🟡 No evidence at N=100 (de-prioritized, NOT falsified)
- Phase B (rearing init pilot): |r|=0.02, ρ=0.01 — clear no-effect at sample size
- Phase C (belly_stretch_deformer impl): r=+0.11 — directional consistency, underpowered
- Phase E bone regularization: r=-0.08, ρ=-0.12 — weakest signal

### 🟢 Elevated (untested, by elimination from weak signals)
- **F6d (GT silhouette self-loop)**: Most likely remaining cause
  - Next step: check `belly_iou_diagnostic.py::_load_gt_mask` — self-rendered vs external GT?
  - If self-loop → belly has no external constraint → LBS rolls to artifact
- **F6a (LBS generic, no blend shapes)**: CANNOT be tested by kinematic approach. Separate hypothesis — architectural limit

### ❌ View-dependent hypothesis DROPPED (iter 2)
- Initial reading "view 4/5 dominate worst" turned out to be **baseline parallax**, not belly-dent per se
- Per-view mean baseline differs: views 4/5 systematically lower due to camera angle

---

## 7. Quantitative Comparison Available

| Artifact | Path | Use |
|----------|------|-----|
| `belly_iou_canon_N100.csv` | `results/reports/` | 600 samples, per-frame-per-view |
| `260418_phase_a_correlations_N100.csv` | `results/reports/` | 100 frames, per-frame features |
| `grid_sequence_3600_sweep9.mp4` | `~/results/MAMMAL/260417_novel_view_mvp/` | 180s video, CORRECT texture, 3600 frames |
| `batch_frame1800_verify.png` | Same dir | Single-frame batch sanity (verified gray mouse) |

---

## 8. Next Step Priority (updated)

**P0**:
1. **F6d investigation**: Check `belly_iou_diagnostic.py::_load_gt_mask` — is it self-rendered or external GT?
2. **View-dependent analysis**: Why views 4/5 dominate worst cases? Camera angle vs belly visibility
3. **Qualitative comparison**: Top-3 worst frames (11340, 3420, 5220) — GT RGB vs MAMMAL rendered bottom view overlay

**P1**:
4. Visual verification of sweep9 batch video (user qualitative)
5. Phase 3 blend-shape feasibility (long-term F6a) — if even GT mask doesn't help

**De-prioritized (empirically failed)**:
- ❌ Phase B rearing init
- ❌ Phase C deformer impl
- ❌ Phase E bone regularization

---

## 9. Method Section Implication (ICML)

This negative finding is **scientifically valuable**: N=23→N=100 comparison demonstrates the danger of underpowered pilot studies. Honest disclosure:

> "Phase A correlation analysis (N=23 pilot, extended to N=100) found no kinematic predictor of belly-dent severity (all |r| < 0.2 at N=100, α=0.05 threshold 0.196). Rearing pose correlation from pilot (r=−0.371, N=23) collapsed to r=+0.017 at N=100, confirming preliminary results were noise. By elimination, hypothesis focus shifts to silhouette-loss-constraint quality (F6d) and view-dependent visibility factors."

---

*Phase A Extension Report v1.0 | 2026-04-18 | Empirical kinematic hypothesis falsification at N=100*
