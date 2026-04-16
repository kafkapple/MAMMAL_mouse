# paper_fast vs production_3600_slerp: Quantitative + Qualitative Comparison

| Field | Value |
|-------|-------|
| Date | 2026-04-16 |
| Scope | 논문 기본 설정(paper_fast, 전체 3600 직접 피팅) vs 현재 방법(900 keyframe accurate + slerp) |
| Related | [260416 status audit](260416_mammal_3600_slerp_status_audit.md), [260323 refit report](260323_mesh_refit_experiment_report.md), [PRODUCTION_3600_PIPELINE](../guides/PRODUCTION_3600_PIPELINE.md) |

---

## 1. TL;DR

| 축 | paper_fast (3600 dense) | production_3600_slerp (900 kf + slerp) | 승 |
|---|---|---|:---:|
| Compute | ~5 h (1 GPU) | ~52 h (4 GPU 병렬) | **paper** |
| Mean IoU (100-sample) | **0.7945** (measured) | — (미측정, 전수 필요) | - |
| Bad frame 비율 (IoU<0.7) | **11%** (11/100, measured) | — (미측정) | - |
| 최악 프레임 IoU | 0.5483 (frame 9480, measured) | — (미측정) | - |
| Accurate 재피팅 효과 (worst 23) | +0.1509 IoU (0.6894→0.8402) | keyframe 은 accurate 로 피팅됨 | **production** |
| Body error (slerp 구간) | 0 (직접 피팅) | 7.1% (E4, interval=4 측정) | **paper** |
| Fast-motion 강건성 | ❌ 5000~10000 구간 IoU 0.55~0.67 | 🟡 slerp self-intersection 경고 | **draw** |
| 실용성 (비용 대비) | 품질 낮음, 빠름 | 품질 개선, 병렬 필요 | **production** |

**결론**: paper_fast 는 **fast-motion 프레임에서 명백히 부족**(bad rate 11%, 최악 0.55). production 은 keyframe 에서 +15% IoU 개선 검증되었으나 **slerp 보간 구간은 7.1% body error + self-intersection 리스크** 를 가져옴. Direct head-to-head IoU (전체 3600 측정) 는 **아직 수행 안 됨** — 이것이 최우선 빈 구멍.

## 2. 정량 비교 (Quantitative)

### 2.1 Baseline (paper_fast, 3600 전체 dense) — IoU 샘플

**측정**: view 3, 100-frame 샘플링 (video frame 0, 120, 240, …, 11880 = 1.2 s 간격)
**출처**: `results/comparison/baseline_iou/baseline_iou_report.txt`

| 지표 | 값 |
|------|---|
| N | 100 frames |
| Mean IoU | 0.7945 |
| Min IoU | 0.5483 (frame 9480) |
| Max IoU | 0.9213 (frame 8040) |
| Bad (IoU<0.7) | **11 frames (11%)** |
| 최악 top-5 | f9480(.548), f9360(.566), f5520(.585), f1320(.612), f8400(.645) |

분포 특징 — bad frames 는 **5000-10000 video frame 구간에 집중** (9개/11개). 이는 마우스의 rearing/fast-motion 구간과 일치 ([260323 report](260323_mesh_refit_experiment_report.md:#motion-analysis)).

### 2.2 Refit accurate vs fast (worst 23 frames)

**측정**: paper_fast baseline 의 worst 23 (IoU<0.7) 을 accurate config 로 재피팅.
**출처**: `results/comparison/refit_23/iou_report.txt`

| 지표 | paper_fast | accurate | Δ |
|------|:---:|:---:|:---:|
| Mean IoU | 0.6894 | **0.8402** | **+0.1509** |
| Pass (≥0.7) | 0/23 | **23/23** | +100 pp |
| 최대 개선 | — | f5520 +0.2871 | — |
| 최소 개선 | — | f10080 +0.0633 | — |

→ **accurate config 는 hard frames 에 강력**. production 의 900 keyframe 이 accurate 로 피팅되었다는 점이 핵심 품질 근거.

### 2.3 Interpolation error (E4)

**측정**: dense accurate (interval=1, ground truth) vs keyframe+interpolation (interval=4)
**출처**: `docs/EXPERIMENTS.md:97-98`, `260323_mesh_refit_experiment_report.md:155`

| 항목 | 값 |
|------|---|
| Body error (interval=4) | **7.1%** |
| Keyframes | 900 |
| Compute | ~52 h / 4 GPU |

7.1% body error 의 의미: slerp 보간 프레임의 관절 위치가 dense accurate 대비 평균 7.1% 체장 오차. 이는 paper_fast 의 IoU 와 직접 변환되는 지표는 아님 — **보간 오차 ≠ 피팅 오차**.

### 2.4 빠진 측정 (Gap)

| 필요 측정 | 상태 | 대안 |
|-----------|:---:|------|
| production_3600_slerp 전체 3600 IoU | ❌ | 같은 100-sample grid 재측정 권장 |
| production vs baseline 동일 프레임 head-to-head | ❌ | 위 기반 계산 가능 |
| 보간 vs keyframe 기여도 분해 | ❌ | keyframe-only IoU + interp-only IoU 분리 측정 |
| Self-intersection 발생 프레임 리스트 | ❌ | trimesh.intersections 스크립트 필요 |

## 3. 정성 비교 (Qualitative)

### 3.1 available 산출물

| 축 | paper_fast | production_3600_slerp |
|---|---|---|
| 전체 시퀀스 video | `results/fitting/baseline_fast_3600` (OBJ 소실, 로그만 남음) | `results/comparison/production_3600_slerp/sequence_6view.mp4` ✅ |
| GT overlay 전체 | ❌ 없음 | `results/comparison/production_3600_slerp_gt/grid_3x2.mp4` ✅ |
| Worst frame still | refit_23 의 `worst_frame_010080_6view.jpg` (accurate 재피팅 후) | `grid_frame01800.png` (중간 프레임만) |
| Silhouette video | `refit_23/video_silhouette_6view.mp4` | ❌ 별도 생성 안 됨 |

### 3.2 관찰 가능한 특징

**로컬 미러 `~/results/MAMMAL/260329_production_3600_slerp/`**:
- `gt_overlay/grid_3x2.mp4` (3600 프레임 GT vs mesh 6-view 겹치기) — **정성 확인 1차 소스**
- `texture_verify/step_2_frame_{000000,009000,017995}_textured.png` — 프레임 0/중간/끝 정적 렌더
- `metrics/iou_chart.jpg` — refit_23 IoU 개선 막대차트
- `metrics/worst_frame_010080_6view.jpg` vs `best_frame_001920_6view.jpg` — accurate refit 의 극단 케이스

**paper_fast baseline 의 OBJ 가 남아있지 않은 상황** (`/home/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/obj/` 비어 있음). 직접 재렌더 불가. 대안:
1. 로그(`fitting_paper_fast_20260126_025245.log`, 6.7MB) 는 남음 — params 추출 가능할 수도
2. 또는 paper_fast 재실행 (~5h, 1 GPU) 후 동일 측정

## 4. 해석 (Interpretation)

### 4.1 왜 production 이 선택되었는가 (재확인)
- paper_fast 는 **평균은 괜찮지만 꼬리가 나쁨** (mean 0.79, 11% 미만 기준치). 학술/리서치 산출물로 11% bad frame 은 허용 범위 밖.
- accurate 3600 dense 는 210 h/GPU 로 **현실적으로 재현 불가**.
- 따라서 "keyframe 만 accurate 로 확실히 + 사이는 저비용 보간" 이 유일 실용 경로. E4 에서 interval=4 를 선택한 것도 7.1% body error 를 **허용 상한** 으로 본 것.

### 4.2 head-to-head 가 없는 상태에서의 추정
keyframe (25%) 는 accurate 품질 → paper_fast 동프레임 대비 +15% IoU 기대. 보간 프레임 (75%) 의 품질은:
- paper_fast 의 보간 구간 평균 IoU 와 slerp 의 7.1% body error 기반 IoU 를 직접 비교 불가
- 그러나 **slerp 는 keyframe 양 끝이 accurate**(높은 품질) 인 상태에서 선형 보간이므로, 중간 프레임이 paper_fast 에서 bad (0.55-0.67) 였던 구간은 **slerp 가 오히려 더 부드럽고 정확할 가능성** 이 높음 (motion 이 크지 않다면)
- 반대로 keyframe 사이에 **빠른 비선형 모션** (rearing 시작 등) 이 있다면 slerp 가 실제 궤적을 놓쳐 **paper_fast 보다 더 나쁠** 수 있음 — self-intersection warning 의 근거

### 4.3 실용 결론
- **FaceLift/PoseSplatter downstream**: production 사용 권장 (현재 합의 상태). keyframe 품질 개선 + 보간 부드러움이 downstream 의 prior 로 더 유리.
- **논문/정확한 벤치마크**: full 3600 direct IoU 측정 선행 필요. 측정 전까지는 "production > paper_fast" 주장 정량 근거 약함.

## 5. Action Items

1. 🔴 **production_3600_slerp 전수 IoU 측정** — baseline 과 동일 grid (100 frames, view 3) 우선, 이후 전체 3600. 스크립트: `scripts/measure_iou.py` (존재한다면 재사용, 없으면 신규).
2. 🟡 **paper_fast OBJ 재생성 또는 params 복원** — 직접 비교 위해 필요. 재실행 비용 5h 로 저렴.
3. 🟡 **Self-intersection 프레임 리스트업** — trimesh.intersections.mesh_multiplane or pyembree 기반 전수 스캔. 영향 프레임만 keyframe 추가 재피팅.
4. 🟢 **이 보고서를 MOC.md research 섹션에 등록** (`260416_paper_vs_production_comparison`).

## 6. Data Pointers

### Local (macOS)
- 본 보고서 & 상태 감사: `~/dev/MAMMAL_mouse/docs/reports/260416_*.md`
- 시각화 + 메트릭 미러: **`~/results/MAMMAL/260329_production_3600_slerp/`**
  - `gt_overlay/grid_3x2.mp4` (메인 정성)
  - `mesh_only/sequence_6view.mp4`
  - `metrics/baseline_iou.json` + `iou_report.json` (refit 23)
  - `metrics/iou_chart.jpg` + `best/worst_frame_*.jpg`

### Remote (gpu03)
- 원본 OBJ: `/home/joon/dev/MAMMAL_mouse/results/fitting/production_3600_slerp/obj{,_textured}/`
- 시각화 소스: `results/comparison/production_3600_slerp{,_gt}/`, `refit_23/`, `baseline_iou/`
- paper_fast OBJ: **없음** (소실). 로그는 `logs/fitting_paper_fast_20260126_025245.log`

---

*Engram v1.0 | paper_fast vs production Comparison | 2026-04-16*
