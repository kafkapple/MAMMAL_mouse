# MAMMAL Mesh Quality — Unified Failure Mode Taxonomy (SSOT)

**Date**: 2026-04-17
**Status**: v1.0 Canonical
**Purpose**: Integrate pop (temporal) + belly-dent (spatial) failure modes into single authoritative reference.

> **Why this doc**: Prior analysis (260417_pop_root_cause_analysis.md) focused on pop. /audit --fact --devil (2026-04-17) found belly-dent 이슈가 별개 mode로 미커버. 본 문서는 **pop + shape quality** 통합 taxonomy.

> **Related**:
> - `260417_pop_root_cause_analysis.md` — Pop 상세 (F1-F5) + F6 요약
> - `260323_mesh_refit_experiment_report.md` — Accurate refit 결과 (silhouette IoU 관점)
> - `260327_lbs_skinning_analysis.md` — LBS 모델 한계 (blend shapes, rearing)

---

## 1. Two-Axis Failure Mode Framework

MAMMAL fitting 실패는 **두 직교 축**에서 발생:

```
              Spatial failure (single-frame shape)
                       ↑
                       |
         Belly-dent ───┼─── Both
          (F6a-e)      |    (rare)
                       |
    ←──────────────────┼──────────────────→
                       |        Temporal failure (frame-to-frame)
           None        |    Pop (F1-F5)
                       |
                       ↓
```

| Axis | Failure Mode | Root Causes | Fix Track |
|------|-------------|-------------|-----------|
| **Temporal** (frame-to-frame) | Pop / jitter / discontinuity | F1 (AA overflow), F2 (slerp flip), F3 (trans jump), F4 (BAD_KF), F5 (scale drift) | T1-T6 (interpolation + fitter regularization) |
| **Spatial** (single-frame shape) | Belly-dent / candy-wrapper / extremity twist | F6a (no blend shapes), F6b (rearing OOD), F6c (mask_loss=0), F6d (no pre-mask), F6e (bone drift) | H-B1~B5 (LBS upgrade + pose init + mask regularization) |

**핵심 원칙**: 두 mode는 원인·fix 메커니즘이 **직교**. Pop을 fix해도 belly-dent는 남을 수 있고, 반대도 마찬가지.

---

## 2. Pop Failure Modes (Temporal Axis)

### F1 — Axis-angle magnitude overflow (|θ|>π)
- **Scope**: 100% keyframes, 3.82% joint-theta entries
- **Mechanism**: MAMMAL fitter `|θ|≤π` constraint 없음 → non-canonical representation
- **Fix**: `canonicalize_axis_angle()` in `mammal_ext/fitting/interpolation.py` (완료, T1)
- **Validation**: rotmat err 1e-7 float noise

### F2 — Slerp hemisphere flip
- **Scope**: 70.1% intervals (post-canon), 3.7% pre-canon (undercounted by F1 masking)
- **Mechanism**: quaternion dot 부호 반대 → long-path slerp → 중간 프레임 180° 튐
- **Fix**: hemisphere-safe quaternion slerp (완료, F1 패치에 병합)

### F3 — Translation discontinuity
- **Scope**: max 81mm / 0.2s (405 mm/s — 불가능)
- **Mechanism**: fitter translation 제약 없음 → keyframe 간 점프
- **Fix**: T4 (fitter-level) + temporal smoothness penalty (pending)

### F4 — BAD keyframe fits
- **Scope**: 11 KF (1.2%), 3 cluster + 1 isolated
- **Mechanism**: fitter local minimum + AA overflow (F1과 correlated 11/11)
- **Fix**: T3 (re-fit with accurate + tighter regularization) — T4 선행 필요

### F5 — Scale & bone-length drift
- **Scope**: scale ±10%, bone length ±4.67mm per-KF
- **Mechanism**: Single animal이므로 상수여야 하나 per-frame 최적화
- **Fix**: T4 (pin scale + bone_lengths after KF 0)

---

## 3. Spatial Shape Quality Modes (New Axis)

### F6a — LBS no blend shapes
- **Scope**: 모든 pose 영향 (근본적 모델 한계)
- **Mechanism**: MAMMAL = pure LBS (14522 vert × 140 joint skinning weights only). SMPL은 207 pose blend shapes + 10 shape PCA. MAMMAL은 shape space β 없음 → candy-wrapper at extreme bends.
- **Evidence**: 260327 §2.3 비교표
- **Fix track**: **Phase 3 (1-2 months)** — pose-dependent correctives 학습 (delta_v = v_obs - v_lbs(θ) 회귀)
- **Difficulty**: High — 아키텍처 변경

> **F6j 가설 업데이트 (최신: [260418_phase_a_extension_report.md](260418_phase_a_extension_report.md))**:
> 1. bone_length[13] 0/900 saturation, std 0.22 actively used (260417 phase 0)
> 2. Upstream MAMMAL도 deformer 없음 — "forgotten feature" 아닌 design choice (260417 belly_deformer_investigation)
> 3. **Phase A N=100 Pearson+Spearman**: r(belly_iou, \|θ[49]\|) = +0.11 Pearson (p=0.27, NS), ρ=+0.07 Spearman (p=0.51, NS). Sign flips between N=23→N=100 confirmed as noise
> 4. **Status**: "No evidence at N=100" (not "falsified" — N=782 required for 80% power at r=0.1)
> 5. **Phase C (deformer impl) de-prioritized** — resume only if N≥500 kinematic sweep shows effect, OR F6d/F6a elevated fix provides context
>
> **F6b (rearing) update**: N=100 r=+0.017 (spine angle), ρ=+0.014. Initial N=23 r=-0.37 was noise. De-prioritized.
> **F6e (bone drift) update**: N=100 r=-0.08, ρ=-0.12. De-prioritized.

### F6b — Rearing / extreme pose OOD
- **Scope**: Rearing frame (뒷다리 서기) + grooming + extreme bend. 예시: frame 10080, 9840 (100.8s, 98.4s)
- **Mechanism**: T-pose init → 사족보행 기준 → rearing 수직 자세에서 local min. Extreme foreshortening in some views.
- **Evidence**: 260327 §3.2 — accurate Δ=0.04 (vs 일반 Δ=0.15)
- **Fix track**:
  - **Short-term** (testable next session): Rearing init template (manual pose) + behavior detection heuristic (spine vector)
  - **Mid-term**: Multi-hypothesis fitting (best-of-N init)
- **Difficulty**: Medium

### F6c — paper_fast mask_loss=0 under-fit
- **Scope**: production_3600_slerp 전체 (3600 frames)
- **Mechanism**: `paper_fast` config: `mask_step2=0` → silhouette 정렬 loss 비활성 → belly 같은 정밀 영역 under-fit
- **Evidence**: 260323 §3.2 (config 비교표). Accurate의 +0.15 IoU 개선에 mask_loss=3000 기여.
- **Fix track**:
  - **Immediate validation** (H-B5): refit_accurate_23 (mask_loss=3000) vs production_3600_slerp (mask_loss=0) 동일 frame belly IoU 직접 비교
  - **Remediation if confirmed**: paper_fast default에 mask_step2 3000 (속도 trade-off, 원본 ~2s/frame → ~??s/frame)
- **Difficulty**: Low (config change)

### F6d — Rendered silhouette mask self-loop
- **Scope**: paper_fast pipeline (pre-computed mask 없음)
- **Mechanism**: mask = rendered mesh silhouette → optim target 자기순환. GT mask 없음.
- **Evidence**: 260327 §3.2
- **Fix track**: SAM-based GT mask pre-compute (H-B4). SAM을 6 view × 18000 frame에 pre-run.
- **Difficulty**: Medium (SAM installation + pipeline integration)

### F6e — Bone length per-frame variation (F5와 교차)
- **Scope**: 4.67mm max per-KF delta
- **Mechanism**: Anatomical constant여야 함 but per-frame optim 허용
- **Fix track**: T4 (pin after KF 0) — F5 fix와 공동
- **Difficulty**: Low

---

## 4. Cross-Matrix — Root Cause × Fix Track

| Root Cause | F1 Canon | T3 Refit | T4 Fitter | H-B2 Rearing | H-B4 SAM | H-B5 mask_loss | LBS Blend |
|------------|:--------:|:--------:|:---------:|:------------:|:--------:|:--------------:|:---------:|
| F1 AA overflow | ✅ | 🟡 | ✅ | — | — | — | — |
| F2 slerp flip | ✅ | — | — | — | — | — | — |
| F3 trans jump | — | 🟡 | ✅ | — | — | — | — |
| F4 BAD_KF | — | ✅ | ✅ | 🟡 | — | — | — |
| F5 scale/bone drift | — | — | ✅ | — | — | — | — |
| F6a no blend shapes | — | — | — | — | — | — | ✅ |
| F6b rearing OOD | — | — | — | ✅ | 🟡 | — | 🟡 |
| F6c mask_loss=0 | — | ✅ | 🟡 | — | — | 📊 (measure) | — |
| F6d no pre-mask | — | — | — | — | ✅ | — | — |
| F6e bone drift | — | — | ✅ | — | — | — | — |

✅ Direct fix | 🟡 Partial | 📊 Measurement only | — Not applicable

---

## 5. Investigation Priority (2026-04-17)

### Tonight (overnight compute)
1. **T2** canon re-interp (F1/F2 validation)
2. **H-B5** refit_accurate vs paper_fast belly IoU (F6c validation)
3. **T5** belly IoU on canon output (F6c partial)
4. **G4** SG baseline (F1/F2 null hypothesis)
5. **T6** side-by-side grid video (both axes visual)

### Next session (after overnight data)
- If H-B5 confirms F6c: paper_fast config upgrade (mask_step2 default 3000)
- If G4 competitive with T2: vertex smoothing fallback viable, T4 priority ↓
- T4 design (fitter-level): |θ|≤π + scale/bone pin + mask_loss↑ integrated
- H-B2 rearing init template (3 rearing frames pilot)

### Deferred (weeks-months)
- F6a LBS blend shapes (Phase 3): 900 keyframe delta_v residual learning
- H-B4 SAM pre-mask: pipeline 통합 필요
- Representation upgrade (6D rotation): orthogonal long-term

---

## 6. Measurement Gaps (pre-investigation state)

| Gap | Impact | Remediation |
|-----|--------|-------------|
| `belly_iou_diagnostic.py` v1 is 2D bbox proxy | Belly 특정 shape 왜곡을 정밀 측정 불가 | **v2**: vertex-group belly IoU (manual belly vertex annotation, 1-2h work) |
| Rearing frame identification manual | F6b scope 모름 | Spine vector heuristic (260327 §6.3) 자동 실행 → rearing frame list 생성 |
| mask_loss sensitivity not measured | F6c 가설 정량 근거 부재 | H-B5 실행 (tonight) |
| Joint 51 identity unconfirmed | F4 interpretation ambiguous (tail vs spine) | `bone_names.txt` mapping read (10min) |

---

## 7. Decision Criteria for Session Close

**Pop track decisive test** (per Sonnet L2 in 260417 deliberation):
- T6 side-by-side grid video에서 user가 pop 감소 확인 (qualitative)
- T5 belly IoU metric stable (quantitative)

**Belly-dent track decisive test**:
- H-B5 결과가 F6c 지지 (refit_accurate_23 belly IoU > paper_fast by ≥0.10)
- → paper_fast config 업그레이드 경로 열림
- 반대 경우 (belly IoU 차이 미미): F6a (LBS 한계) 또는 F6b (rearing) 우선, F6c는 false hypothesis

---

## 8. Empirical Results (2026-04-17 session close)

See: `results/reports/260417_canon_vs_paper_validation.md` (SSOT empirical)

### Pop Track ✅ VALIDATED

**H-B5-interp** (pop-prone interpolated frames test):

| | Paper_fast slerp | Canon slerp | Δ |
|---|:---:|:---:|:---:|
| Pop frame Global IoU (N=48) | 0.24 | **0.79** | **+0.54** |
| Pop frame Belly IoU (N=48) | 0.41 | **0.83** | **+0.42** |
| Canon wins | — | **48/48** | — |
| Non-pop frame Δ (N=18) | — | — | +0.01 |

**Verdict**:
- F1 (AA overflow) + F2 (slerp hemisphere flip) 가설 **완전 empirically 검증**
- T2 canon re-interp이 **pop 완전 해결** at pop frames
- T4 (fitter-level fix) → **pop용 불필요** (interpolation은 canon이 해결)
- **T4 잔여 scope**: F3 trans jump + F5 scale/bone drift + F6c mask_loss (keyframe 수준)

### Belly-Dent Track 🟡 PARTIAL

- **At pop frames**: canon이 side-effect로 belly 개선 (+0.42 IoU)
- **At non-pop frames**: canon 효과 없음 (±0.01 변동) — **F6 track 별도 해결 필요**
- **F6c (mask_loss=0)**: refit_accurate_23 vs paper_fast @ keyframes 차이 미미 (-0.01) — **F6c 가설 약화** (keyframe 재피팅이 belly를 유의미하게 개선하지 않음). 단 keyframe과 interpolation 모두에서 동일 결과이므로 upstream 원인 (F6a/F6b) 가능성 ↑

### G4 (SG vertex smoothing baseline)

| Severity stratum | N | Reduction |
|------------------|:-:|:---------:|
| Extreme pop (>100) | 35 | **77.3%** |
| Severe pop (50-100) | 51 | 65.2% |
| Moderate (20-50) | 176 | 53.0% |
| Total pop-like (>20) | 262 | avg ~60% |

**Verdict**:
- Post-hoc SG smoothing이 표면 symptom은 효과적으로 제거 but representation level fix 아님
- Canon (upstream) + SG (fallback) 선택 시 canon이 root cause에 작용하므로 우선
- SG는 F6b rearing 같은 fitter-level OOD에서 fallback 옵션으로 유지

### Open Questions (next session)

1. **Non-pop frames의 belly-dent 원인 규명** — H-B2 rearing init (F6b) / F6a LBS blend shapes 실험
2. **Production baseline switch** — `production_3600_canon/` → downstream (FaceLift, pose-splatter) propagation 필요 여부
3. **F3 trans jump 81mm/0.2s** — keyframe-level 별도 fix, T4 scope 축소됨
4. **86 pop frames (G4 accel>50) 전체에서 canon 효과 균일한가** — 8 frame sample → 전체 검증 필요

---

*Mesh Quality Failure Modes | v1.1 | 2026-04-17 | Empirical validation added*
