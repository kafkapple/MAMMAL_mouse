# MAMMAL 3600-Frame Slerp Production: Status Audit Report

| Field | Value |
|-------|-------|
| Date | 2026-04-16 |
| Audit Level | Lv.3 (`/audit --fact --devil`) |
| Scope | Production fitting status, frame-rate claims, FaceLift alignment |
| Related Reports | [260323 refit experiment](260323_mesh_refit_experiment_report.md), [260327 LBS analysis](260327_lbs_skinning_analysis.md) |

---

## 1. Executive Summary

"3600 프레임 (20fps) 피팅 완료" 라는 단순한 표현은 **절반만 맞습니다**. 실제 파이프라인은:

```
900 keyframes (accurate config, 직접 피팅)
      │
      ▼ params slerp (θ quaternion slerp + shape lerp)
3600 frames (20fps equivalent, M5 interval=5 alignment)
      │
      ▼ [NOT YET DONE]
18000 frames (100fps, raw video rate)
```

- **20fps (3600 frames) 최종 산출물 ✅ 완성** — 단, 900 직접 피팅 + 2700 slerp 보간
- **100fps (18000 frames) ❌ 미완성** — 추가 보간 단계 필요
- 현재 gpu03 에서 돌고 있는 MAMMAL 피팅 프로세스 없음 (2026-04-16 확인)
- "원래 계획이었던 3600 dense accurate 피팅" 은 210h/GPU 비현실적 → **keyframe+slerp 로 전략 피벗** (E4 실험 결론 기반)

## 2. Fact-Check Matrix

| # | 주장 | 검증 | 근거 |
|---|------|:---:|------|
| 1 | 전체 raw frames = 18000 | ✅ | `conf/frames/aligned_posesplatter.yaml:3,13`, CLAUDE.md Dataset Specs |
| 2 | FPS = 100 | ✅ | 동일 |
| 3 | interval=5 → 3600 M5 frames = 20fps equivalent | ✅ | 18000/5=3600, 100/5=20 |
| 4 | "3600 프레임 모두 직접 피팅" | ❌ **거짓** | 실제 피팅된 것은 **900 keyframes** (M5 interval=4). `docs/EXPERIMENTS.md:106` "Frames: 900 keyframes" |
| 5 | "20fps 완성" | 🟡 조건부 ✅ | `production_3600_slerp/obj/` 3600 OBJ 존재. 900 fitted + 2700 slerp. "완성" 의미에 따라 다름 |
| 6 | "slerp 하면 100fps 완성" | 🟡 **추가 작업 필요** | 현재 slerp 는 900→3600 단계에서 이미 쓰였음. 100fps 까지 가려면 **3600→18000 추가 보간** (script 존재 확인 필요) |
| 7 | "FaceLift 에서 이렇게 하고 있어서" | 🟡 해석 주의 | FaceLift handoff spec 은 3600 OBJ 기준 (`EXPERIMENTS.md:125`). FaceLift 는 20fps 입력 소비자이지 100fps 요구자 아님 |

## 3. Pipeline 실제 구조 (E5 Production)

| Stage | Config | Output | Frame Count | FPS (video-time) |
|-------|--------|--------|:-----------:|:----:|
| Keyframe fitting | `accurate` (step0=20, step1=200, step2=50) | `production_keyframes_part{1-4}/` | 900 (225×4 GPU) | 5 |
| Merge | - | `production_900_merged/` | 900 | 5 |
| Params slerp | `scripts/interpolate_keyframes.py` (θ quaternion slerp, β lerp) | `production_3600_slerp/obj/` | 3600 | 20 |
| UV transplant | `scripts/uv_transplant_refit.py` | `production_3600_slerp/obj_textured/` | 3600 | 20 |
| Blender verify | headless render (frame 0/9000/17995) | `results/comparison/texture_verify/` | 3 samples | - |

**Keyframe 선정**: video interval=20 (= M5 interval=4), 0.2 초 간격. 파일 `exports/keyframe_indices.txt` 에 900 indices.

**Compute 비교**:
- Dense accurate 3600 전체 → ~210 h/GPU (비현실적)
- 900 keyframes × 4 GPU 병렬 → ~52 h (실제 수행 비용)
- 절약 비율: 4× 감소 + 보간 손실 7.1% body error (E4 측정)

## 4. Data Locations (gpu03: `/home/joon/dev/MAMMAL_mouse/`)

### 4.1 Fitted Geometry

| Artifact | Path | Count | Note |
|----------|------|:-----:|------|
| 900 keyframes (split) | `results/fitting/production_keyframes_part{1,2,3,4}/` | 225×4 | per-GPU split |
| 900 merged | `results/fitting/production_900_merged/` | 900 | post-merge canonical |
| 3600 slerp OBJ | `results/fitting/production_3600_slerp/obj/step_2_frame_{000000..017995}.obj` | 3600 | geometry-only |
| 3600 textured OBJ | `results/fitting/production_3600_slerp/obj_textured/` | 3600 | + `.mtl` + texture ref |
| Keyframe indices | `exports/keyframe_indices.txt` | 900 lines | video frame numbers |
| Static texture | `exports/texture_final.png` | 1 | 모든 프레임에 동일 적용 |

### 4.2 Visualizations

| Output | Path | Content |
|--------|------|---------|
| **Grid still** (mesh only) | `results/comparison/production_3600_slerp/grid_frame01800.png` | frame 1800, 6-view grid |
| **Grid still** (GT overlay) | `results/comparison/production_3600_slerp_gt/grid_frame01800_gt_overlay.png` | frame 1800, 6-view GT\|Mesh overlay |
| 6-view 결합 video | `results/comparison/production_3600_slerp/sequence_6view.mp4` | mesh only, 3×2 grid |
| Per-view mesh videos | `results/comparison/production_3600_slerp/sequence_v{0..5}.mp4` | each camera, mesh only |
| **GT overlay 3×2 video** | `results/comparison/production_3600_slerp_gt/grid_3x2.mp4` | **6-view GT vs Mesh 비교 (메인 산출물)** |
| Per-view GT overlay videos | `results/comparison/production_3600_slerp_gt/interpolated_v{0..5}.mp4` | each camera, GT + mesh overlay |
| Texture verify | `results/comparison/texture_verify/step_2_frame_{000000,009000,017995}_textured.png` | Blender headless render, 3 samples |

### 4.3 빠른 확인 커맨드

```bash
# GT 비교 메인 영상
ssh gpu03 "ls -la /home/joon/dev/MAMMAL_mouse/results/comparison/production_3600_slerp_gt/grid_3x2.mp4"

# 로컬로 복사 (macOS 에서 재생)
scp gpu03:/home/joon/dev/MAMMAL_mouse/results/comparison/production_3600_slerp_gt/grid_3x2.mp4 ~/Downloads/
scp gpu03:/home/joon/dev/MAMMAL_mouse/results/comparison/production_3600_slerp_gt/grid_frame01800_gt_overlay.png ~/Downloads/

# Or SSOT symlink: ~/results/MAMMAL_mouse/ (repo convention)
```

## 5. Devil's Advocate (남은 리스크)

### 5.1 "완성" 라벨의 모호성
`docs/EXPERIMENTS.md:116` "✅ Completed" 는 **E5 production 범위 한정**. "MAMMAL mesh fitting 프로젝트 전체 완성" 과 동일하지 않음.

구체적으로 **미완성 항목**:
- 100fps (18000 frames) 커버리지 — 현재 20fps (3600) 에서 정지
- Fast-motion (rearing, quick turn) 구간의 **self-intersection** — 최근 커밋 `07f5d0c` 에서 warning 만 추가, 해결 안 됨
- Body error 7.1% (E4 측정) — 정량 리포트 아직 GT overlay 정성 검토 단계

### 5.2 "FaceLift 가 이렇게 한다" 주장의 왜곡 위험
`docs/EXPERIMENTS.md:124` FaceLift handoff 는 **20fps (3600) 입력 수신** 을 가정함. FaceLift 쪽이 "100fps 필요하다" 고 요구한 적은 없음 (handoff checklist 체크 완료됨 ✅). 따라서 "100fps 로 가야 FaceLift 와 맞는다" 는 전제가 있다면 재확인 필요:
- FaceLift 가 요구하는 temporal density 명세는?
- 현재 3600 frame 1:1 대응으로 합의 완료된 것 아닌가?

### 5.3 Slerp 자체의 한계
`260327_lbs_skinning_analysis.md:47,109` — quaternion slerp 는 joint-local rotation 이 선형 궤적인 경우에만 안전. 마우스의 빠른 rearing/grooming 은 keyframe 사이에 **비선형 curvature** 를 가짐. E4 측정 7.1% body error 는 평균이고, fast-motion frames 는 outlier 로 자가 교차 발생 가능성 보고됨.

→ 대응: **keyframe 밀도 국소 상승** (motion-aware adaptive interval) 또는 **neural interp**. 현재 코드베이스에 미구현.

### 5.4 감사 #1(이전 리포트) 정정
`/audit --fact --devil` 1차에서 필자가 "3600 obj = 20fps 완성, interval=5 직접 피팅" 으로 기술했던 부분은 **사실상 부정확**. 정확히는:
- interval=5 는 **alignment grid** 이지 피팅 granularity 가 아님
- 실제 피팅 granularity 는 900 keyframes (interval=20 in video frames, interval=4 in M5 frames)
- 3600 OBJ 파일명이 `frame_000000, 000005, 000010, ...` 으로 interval=5 처럼 보이지만, 이는 **slerp 결과물의 출력 스케줄** (M5 grid 에 맞춘 naming) 일 뿐

## 6. Verdict

| 질문 | 답 |
|------|------|
| 현재 돌고 있는 피팅? | **없음** (gpu03 `ps aux` 에 MAMMAL 프로세스 0, 2026-04-16) |
| 완료? 중단? | **E5 production 완료** (20fps 3600 OBJ + UV + GT overlay + Blender verify ✅). "원래 계획 dense accurate 3600" 은 E4 단계에서 **의도적 피벗** (비현실적 비용) |
| 3600 전부 피팅? | ❌ **아님**. 900 keyframes 직접 피팅 + 2700 slerp 보간 |
| 현재 20fps 완성? | ✅ OBJ 산출물 기준 (900+slerp). 품질은 7.1% body error |
| slerp → 100fps? | ❌ **추가 보간 안 됨**. 3600→18000 까지 가려면 동일 slerp 로직 재적용 필요하나 self-intersection 리스크 증폭 가능 |
| 데이터·시각화 위치? | §4 참조. 메인: `results/comparison/production_3600_slerp_gt/grid_3x2.mp4` |

## 7. Action Items (우선순위)

1. 🔴 **사실 정정 공유**: "3600 모두 피팅" 표현은 사내 커뮤니케이션/핸드오프 문서에서 "900 keyframe + slerp 3600" 으로 고정.
2. 🟡 **100fps 필요성 재확인**: FaceLift (또는 pose-splatter) 측 요구 사양 재검토. 필요 없다면 현재 20fps 로 종결.
3. 🟡 **7.1% body error 정량 분해**: per-frame distribution 로드해 fast-motion 구간 error spike 구체화. `docs/reports/` 에 후속 리포트.
4. 🟢 **Self-intersection warning 실제 측정**: 영향 프레임 리스트 추출, keyframe 밀도 상승 ROI 평가.

---

*Engram v1.0 | MAMMAL 3600 Slerp Status Audit | 2026-04-16*
