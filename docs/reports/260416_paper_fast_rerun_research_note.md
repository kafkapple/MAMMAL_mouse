# Research Note: paper_fast 3600 Rerun — Pre-Execution Analysis

**Date**: 2026-04-16
**Author**: joon + Claude
**Purpose**: 실험 실행 승인 게이트 문서. 연구 맥락 / 가설 / 조건 / 우선순위 조사.

---

## 1. Prior Research State

### Existing fittings (gpu03 `/home/joon/dev/MAMMAL_mouse/results/fitting/`)

| Dir | Frames | Source | Role |
|-----|--------|--------|------|
| `production_3600_slerp/obj/` | 3600 (stride=5, 0-17995) | 900 keyframes + θ-slerp ×4 | 현재 production output |
| `production_900_merged/` | 900 keyframes | `paper_fast` 4-part split merge | slerp base |
| `baseline_fast_3600/` | - (only `keypoints_22_3d.npz`) | OBJ 없음 (파일 유실 or 이동?) | 불명 |
| `dense_accurate_0_100`, `dense_accurate_100_200` | 200 | `accurate` config | 실험 비교용 |
| `refit_accurate_23` | 23 samples | 키포인트 refit | quality spot-check |
| `rearing_test_exp_{a,b}` | - | rearing event 분석 | 별개 분석 |

### Reported 0.7945 baseline (handoff 기준)

- **Source dir**: `/home/joon/data/synthetic/textured_obj/` ← `baseline_iou_all.py` default
- **Coverage**: 100 frames at raw_frame_id ∈ {0, 120, 240, ..., 11880} (stride 120, range 0-11880 = 66% of video)
- **Filename**: `step_2_frame_{fid:06d}.obj` (= after paper_fast articulation step 2)
- **Computed**: `results/comparison/baseline_iou/baseline_iou.json`, mean IoU=0.7945, bad-rate 11%

### ⚠️ 발견된 gap

1. **`textured_obj/` 의 lineage 불명** — 어느 run 에서 왔는지 명시된 문서 없음. `baseline_fast_3600/` 의 obj 가 이동된 흔적 (디렉터리에 `keypoints_22_3d.npz` 만 남음)
2. **reported 베이스라인의 coverage = 66% of video** (0-11880 only). full range 평가 아님
3. **determinism 미검증** — paper_fast 재현성 한 번도 확인된 적 없음 (seed/stepwise 난수 의존성 불명)

---

## 2. Current Proposed Experiment

### Variant
- **Config**: `baseline_6view_keypoint` + `frames=aligned_posesplatter` + `optim=paper_fast`
- **Output**: `results/fitting/baseline_paper_fast_260417/obj/` (3600 OBJ, stride=5)
- **Wall-time**: ~5h (background nohup on gpu03)
- **Resource**: GPU 0 (현재 empty, 97GB free), ~70-80GB VRAM 예상

### Hypothesis (가설)

**H1 (main)**: `paper_fast` 는 deterministic 이어서, 재실행 시 subset {0, 120, ..., 11880} IoU 평균이 기존 0.7945 와 <0.5% 범위 내 일치한다.

**H2 (secondary)**: 재실행은 `baseline_fast_3600/` 의 "obj 유실" lineage gap 을 해소하고, 비교 가능한 full-range (0-17995) paper_fast 베이스라인을 확보한다.

### Success criteria (verify_metrics.py gate)

- mean_iou_drift < 0.5% AND bad-rate drift < 2pp → downstream 비교 진행 허가
- 둘 중 하나라도 위반 → 비교 파이프라인 자체 감사 선행

---

## 3. Why This, Why Now — Priority Argument

### 상위 목표 (from handoff)
1. mesh popping 원인 진단 및 수정 (slerp)
2. belly 왜곡 진단 및 수정 (β_delta probe)
3. paper_fast vs production slerp 비교 문서화

### 5h rerun 이 상위 목표에 기여하는가?

| 목표 | Rerun 필수? | 대안 |
|------|:---:|------|
| (1) slerp popping | **❌ 불필요** | `diagnostic_slerp.py` (keyframe pair 만 필요, 1h 이내) |
| (2) belly 왜곡 | **❌ 불필요** | `belly_iou_diagnostic.py` (100 frame grid, 1-2h) |
| (3) paper vs production 비교 | ✅ 단, lineage 보증용 | 기존 0.7945 사용 시 "textured_obj 출처 불명" 리스크 존재 |

### Priority re-assessment

현재 제안된 순서 (핸드오프): **1. rerun (5h BG) → 2. 병렬 진단 → 3. 검증**
- 병렬 진단이 slerp fix 효과를 이미 보이면, rerun 은 retrospective validation 용
- 병렬 진단이 5h 안에 끝나면, rerun 결과 도착 시 추가 인사이트 없음

대안 순서: **1. 진단 먼저 (1-2h) → 2. 발견에 따라 rerun 여부 결정**
- 만약 slerp WRONG_HEMISPHERE = 0 → slerp 자체는 정상 → 다른 원인 탐색 (keyframe fit 실패 등)
- 이 경우 5h rerun 을 아낄 수 있음

### 판단

**Rerun 은 필요하다** (H2 — lineage gap 해소), 단 **진단과 **동시 병렬** 실행이 효율적. 핸드오프의 순서는 타당.

Caveat: 5h 완료 전에 진단 결과로 원인이 확정되면 rerun 은 "문서 증거" 역할로 격하. 그래도 감사·재현성 측면에서 가치 있음.

---

## 4. Conditions / Data / Env

| 항목 | 값 | 검증 |
|------|-----|------|
| Dataset | `data/examples/markerless_mouse_1_nerf/` | .claude/CLAUDE.md 명시 |
| Frames | 3600 (interval=5, 0-17995 raw) | `conf/frames/aligned_posesplatter.yaml` |
| Optim | step0=60, step1=5, step2=3, mask=0 | `conf/optim/paper_fast.yaml` ✓ 논문 정합 |
| GPU | 0 (empty, 97GB) | `nvidia-smi` 확인 ✓ |
| CUDA_VISIBLE_DEVICES | 0 | shared_server.md job naming 필요 |
| Job name | `claude:joon:paper_fast_rerun_260417` | 누락 → 명시해야 함 |
| Output dir | `results/fitting/baseline_paper_fast_260417/` | 신규, 충돌 없음 ✓ |
| Log path | `logs/paper_fast_rerun_260417.log` | NFS (`/home/joon/...`) — **고빈도 append → `/node_data` redirect 권장** |
| Wall-time | ~5h | shared_server.md approval gate 해당 (>1h) |
| PID 기록 | `logs/paper_fast_rerun.pid` | monitoring 용 |

### ⚠️ 주의사항

1. **NFS write issue** (CLAUDE.md 2026-04-15 인시던트): stdout redirect 를 `/node_data/joon/logs_MAMMAL/` 로 해야 안전. 기존 `logs/` 심볼릭링크 확인 필요.
2. **Shared-server approval gate**: 5h × 1 GPU (single node, single rank) — 승인 필수.
3. **Seed/determinism**: `conf/` 에 명시된 seed 확인 필요. `paper_fast.yaml` 에 seed 미포함 → 기본값 의존.

---

## 5. What Could Go Wrong / Missing

1. **NFS log write**: 현재 `logs/` 가 `/node_data` 심볼릭링크인지 미검증 → 5h × stdout append 시 NFS hang 재발 리스크
2. **OBJ 생성 실패**: 중간 GPU OOM / step size mismatch → 3600/3600 중 일부 누락 가능 → `wc -l` 로 완주 확인 필요
3. **determinism 위반 시 downstream**: `verify_metrics.py` exit 1 로 멈추지만, **이후 어떤 액션** 을 할지는 명시 안 됨 (handoff Open Q2 참조)
4. **기존 reported lineage**: 만약 `textured_obj/` 가 `paper` (not `paper_fast`) 설정에서 생성된 것이면 drift 발생이 config 차이 때문일 수 있음. 문서 재조사 필요.
5. **병렬 실행 경합**: 진단이 동일 GPU 0 를 점유하면 rerun 에 영향. 진단은 다른 GPU (1,2,3,6) 할당 필요.
6. **Checkpoint resume**: 중간 중단 시 재시작 기능 유무 불명 (`./run_experiment.sh` 체크 필요).

---

## 6. Alternatives Considered

| Alt | 내용 | 채택? | 이유 |
|-----|------|:---:|------|
| **A** | 핸드오프 순서 그대로: rerun BG + 병렬 진단 | ✅ (조건부) | lineage gap + 병렬 효율 |
| B | 진단 먼저 → 결과 보고 rerun 결정 | ❌ | 순차 = 8-9h, 5h 절약 불확실 |
| C | rerun skip, 기존 0.7945 사용 | ❌ | lineage gap 리스크, 사용자 원칙 위배 (측정 gate) |
| D | short rerun (100 frames @ step=120) 으로 determinism 만 먼저 확인 | ⚠️ 대안 | 1-2h 로 H1 검증, H2 포기. 토의 가치 |

**D 는 매력적**: determinism 만 확인 목적이면 3600 전체 불필요. 100 frame rerun (stride=120, same as reported baseline) = ~1h 이면 충분. H2 (full-range lineage) 는 별도 쿼터로.

---

## 7. Proposed Execution Order (revised)

### Option A (handoff 그대로, 추천 조건부)
```
Day 1 (8h window):
  t=0:00  Rerun 3600 start (GPU 0, BG)
  t=0:05  Diagnostic slerp start (CPU + GPU 1, FG, ~1h)
  t=1:00  Diagnostic belly start (GPU 2, FG, ~1-2h)
  t=2:30  Diagnostic 결과 해석 → slerp/belly 원인 추정
  t=5:00  Rerun 완료 → verify_metrics.py
  t=5:15  side-by-side 비교 비디오 렌더 시작
```

### Option D (determinism-first, 더 빠름)
```
Day 1 (4-5h window):
  t=0:00  Short rerun 100 frames (GPU 0, ~1h)
  t=0:05  Diagnostic slerp + belly 병렬 (GPU 1,2)
  t=1:00  Short rerun 완료 → verify_metrics.py → H1 pass/fail
  t=1:15  H1 pass: 비교 문서화 완결, full rerun 불필요
           H1 fail: config 감사 → 재판단
  t=2:30  Diagnostic 종합 → fix 적용
```

**제안**: 사용자 승인에 따라 A 또는 D 결정.

---

## 8. Fact-Check Summary

| 주장 | 근거 | 상태 |
|------|------|:---:|
| reported mean IoU = 0.7945 | `baseline_iou.json` 100 frames, computed just now | ✅ (tool_use: Read + Python eval) |
| reported bad-rate = 11% | 동일 JSON, threshold=0.7 | ✅ |
| `production_3600_slerp/obj/` has 3600 OBJ | `ls | wc -l = 3600` | ✅ |
| `baseline_fast_3600/` has no OBJ | `ls | wc -l = 0` | ✅ |
| GPU 0 empty (97GB free) | `nvidia-smi` csv output | ✅ |
| `aligned_posesplatter` = 3600 frames | `conf/frames/aligned_posesplatter.yaml` | ✅ |
| paper_fast determinism 미검증 | 검증 artifacts 부재, handoff 명시 | ⚠️ HYPOTHETICAL |
| `textured_obj/` lineage 불명 | README/doc 없음, baseline_fast_3600 empty | ⚠️ 추정 |
| 5h wall-time | handoff 명시, 이전 run 기록 없음 | ⚠️ HYPOTHETICAL |

---

*Pre-execution research note v1.0 | 2026-04-16 | Pending /deliberate --audit --devil*
