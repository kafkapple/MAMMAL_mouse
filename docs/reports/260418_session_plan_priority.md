# Session Plan + Priority — MAMMAL Next Steps

**Date**: 2026-04-18
**Context**: Post-pop-fix + novel-view MVP. Identifies (1) texture issue resolution, (2) belly-dent implementation plan, (3) prioritized pending items across handoffs.

---

## 1. Texture Issue — Resolved

### Discovery
- **Wrong texture used in 260417 batch render**: `exports/sequence/texture_final.png` is **incomplete/broken** (near-white with dark artifact smudges)
- **Correct texture exists**: `results/sweep/run_wild-sweep-9/texture_final.png` — WandB Bayesian HPO optimized (gray mouse with realistic features: ears, paws, tail tip detail)
- **Evidence**: Side-by-side `grid_2x3_textured.png` (broken) vs `grid_2x3_sweep9_texture.png` (correct) in `~/results/MAMMAL/260417_novel_view_mvp/`

### Root cause
`novel_view_render.py` initial default hardcoded `exports/sequence/texture_final.png` without checking if other textures exist. UVMAP_GUIDE.md documents full HPO pipeline but texture canonical path is not declared in a single place.

### Fix applied
- **`scripts/novel_view_render.py`**: default texture path changed to `results/sweep/run_wild-sweep-9/texture_final.png` with docstring explaining why
- **TODO**: Batch re-render `canon_3600` with corrected texture (40min GPU)
- **TODO**: `docs/guides/UVMAP_GUIDE.md` §Output에 "Canonical production texture = `results/sweep/run_wild-sweep-9/texture_final.png`" 명시

### Texture optimization 추가 가능성 (HPO re-run)

기존 sweep (251212) 파라미터:
- visibility_threshold, fusion_method, do_optimization, opt_iters, w_tv (TV reg)
- Score v3: 0.5·PSNR + 0.15·SSIM + 0.2·coverage + 0.15·seam

| Improvement path | 기대 효과 | 비용 | 장단점 |
|-----------------|:--------:|:----:|--------|
| A) 기존 sweep-9 그대로 사용 | — | 0 | ✅ 현재 결과 합리적 / ❌ 260417 canon mesh 기반 최적화 아님 |
| B) canon mesh 기반 texture 재최적화 | +1-3 dB PSNR 추정 | 5-10h (sweep 30+ trials) | ✅ 현 mesh와 최적 정합 / ❌ 큰 GPU 비용, 시각 개선 marginal |
| C) SSIM 가중치 상향 (0.15→0.3) + TV↓ | 시각적 개선 (seam 감소) | 2-3h | ✅ 빠른 tune / ❌ PSNR 약간 감소 |
| D) Neural texture (learned MLP) | 큰 개선 가능 | 2주+ | ✅ GS-LRM 수준 품질 / ❌ 구현 복잡, 학습 필요 |

**추천 (MVP-first)**: **A** 유지, 필요 시 **C** (low-cost fine-tune). **B**는 downstream 요구 시. **D**는 long-term Phase 3.

---

## 2. Belly-Dent Implementation Plan

### 기존 코드/문서 상태 (확인 결과)

| Item | Status | 위치 |
|------|:------:|------|
| `belly_stretch_deformer` parameter | ❌ **미구현** — docstring에만 존재 | `articulation_th.py:385` (function body에 없음) |
| `chest_deformer` parameter | ✅ 구현 | `articulation_th.py:243-253, 400-411` |
| Bone_length[13] (belly_stretch) 1D scale | ✅ 사용 중 | std=0.223, range [0.55-1.46] active |
| Rearing init template | ❌ 미구현 | 260327 §4.3 제안만 |
| SAM-based GT mask | ❌ 미구현 | H-B4 후보 |

### Phase A — Low-cost empirical probes (Day 0, 3-4h)

**실험 가설**:
- H-A1: Belly-dent severity ↔ joint 49 (belly_stretch) theta magnitude 상관관계
- H-A2: Belly-dent severity ↔ bone_length[13] extreme value (< 0.6 or > 1.4) 상관관계
- H-A3: Rearing frame (spine heuristic) ↔ belly-dent frame 겹침 비율

**실행**:
1. **Spine vector heuristic 배포** (30min): `V = kp3d['neck'] - kp3d['pelvis']; rearing = V[Y_up] > 0.7` → 전체 3600 frames에 rearing score 계산
2. **Belly IoU batch** (1h GPU): `belly_iou_diagnostic.py --all-100` 또는 샘플 200 frames → `belly_iou_3600.csv`
3. **Correlation 분석** (1h): belly_iou vs spine_angle / theta[49] / bone_length[13] → Pearson r, Spearman ρ
4. **Visual verification** (30min): Top 10 belly-dent frames에서 rearing 맞는지 시각 확인

**결정 gate**:
- r(belly_iou, rearing) > 0.6 → **F6b 강력한 근거** → Phase B 진행
- r(belly_iou, theta[49]) > 0.5 → **F6a 구조적 근거** → Phase C
- 모두 r < 0.3 → **미식별 원인** → F6d (GT mask) 탐색

### Phase B — Rearing init pilot (Day 1, 3-4h)

**조건**: Phase A가 F6b 지지

**실행**:
1. 3-5 rearing frames 선택 (H-A3 top)
2. Manual rearing pose init 스크립트 (pelvis tilt + spine rotation)
3. `accurate` config로 3 frame × 14min = 42min GPU
4. Belly IoU 전/후 비교

**결정 gate**:
- Belly IoU +0.10 이상 → **init heuristic 자동화** 단계 (Day 3)
- 개선 없음 → F6a 구조적 한계 확증 → Phase C

### Phase C — F6a conditional (Week 1-4)

**조건**: Phase A/B 모두 F6a 증거 축적

**실행**:
1. **belly_stretch_deformer 구현** (2-3일)
   - `articulation_th.py::forward()` parameter 추가
   - chest_deformer와 동일한 3x3 matrix (belly region vertices에 적용)
   - `fitter_articulation.py` optimization variable + L2 penalty
2. **Pilot fit** (하루): 3 rearing frames에 새 deformer로 refit
3. **Decision gate**: belly IoU +0.05 이상 → full 900 keyframe refit

**Long-term (Phase D, 월 단위, 조건부)**:
- Pose-dependent blend shapes (SMPL-style 207 correctives)
- Data requirement: 50+ annotated belly-shape targets (수동 편집)
- Difficulty: HIGH

### Phase E — SAM GT mask (parallel track, Week 1-2)

**독립 path** (F6a/b와 직교):
- SAM2로 6 view × 18000 frames mask pre-compute (~8h GPU)
- silhouette loss를 self-loop가 아닌 GT mask 기반으로 전환
- 기대: fitter의 belly region optimization에 올바른 gradient 제공

---

## 3. Pending Handoffs + Priority

### Handoff 상태 scan (기존 + 신규)

| Handoff | Status | Relevance | Priority |
|---------|:------:|:---------:|:--------:|
| `260417_1130_mammal_canon_validated.md` | **open (current)** | Primary | 🔴 P0 |
| `260416_1637_mammal_quality_gate_pipeline.md` | ✅ superseded, archived | — | — |
| `260417_0030_mammal_pop_validation.md` | superseded | — | — |
| `260417_0200_facelift_ssot_icml_integration.md` | open | FaceLift side | 🟡 P1 |
| `260413_1930.md` (BehaviorSplatter) | open (5 days) | BS coord blocking | 🟡 P1 |
| `260416_1700_facelift_rat7m_p1_pivot.md` | open | Different track | 🟢 P2 |
| `260416_1805.md` (ICML submission) | open | Cross-project | 🟢 P2 |

### Consolidated Priority List

**🔴 P0 (이번 세션 or 다음)**:
1. **Texture fix batch re-render** — 40min GPU. canon_3600 novel view video 올바른 색상으로 재생성
2. **Phase A: Belly empirical probes** — 3-4h, low-cost, 전체 F6 track 방향 결정
3. **Commit + sync** — 10+ uncommitted files (2 repos)

**🟡 P1 (이번 주)**:
4. **Phase B: Rearing init pilot** — 3-4h GPU, conditional on Phase A
5. **Production switchover decision** — `production_3600_canon` deploy + FaceLift/PS resync
6. **ICML workshop integration** — novel view mesh feature → sparse/dense chain motivation

**🟢 P2 (다음 주+)**:
7. **F6a / F6j 구현** (Phase C) — conditional, architectural
8. **Cross-project symlink SSOT** — BehaviorSplatter novel view integration
9. **M5_SCENE_CENTER 8+ hard-copy 정리** — pip-installable `mammal_facelift_bridge` package
10. **SAM GT mask pipeline** (Phase E) — independent track

**🔵 P3 (backlog)**:
11. 26 pop frame 전체 canon 효과 확증 (D1 audit 우려)
12. 86 G4 pop frames (accel>50) 전수 검증
13. 다른 handoff follow-ups (FaceLift Rat 7M, ICML submission 등)

---

## 4. Minimum Viable Session Plan (권장)

**Goal**: 다음 세션 3-4h 내 완결

1. **Texture batch re-render** (40min GPU, background) — canon_3600 재생성 + fetch
2. **Phase A empirical probes** (3-4h 병렬):
   - Spine vector heuristic 배포
   - Belly IoU batch (200 frames sample)
   - Correlation 분석
3. **Phase A 결과 기반 Phase B/C 분기 결정** + handoff 작성
4. **Commit + sync** (15min)

**DEFERRED 명시**: Phase D (blend shapes), neural texture (Phase D), M5 shared package, SAM mask — 구조적, 다음 세션 이상 필요

---

*Session Plan | 2026-04-18 | Post-pop-fix + novel-view MVP*
