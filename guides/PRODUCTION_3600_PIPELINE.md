# Production 3600 Pipeline (Keyframe + Slerp)

> 20fps (3600 frames) 메쉬 시퀀스를 얻는 **현재 프로덕션 방법**. "3600 직접 피팅" 이 아니라 **900 keyframe 피팅 + slerp 4× 보간** 전략이다.
>
> 관련 문서: [EXPERIMENTS.md](../EXPERIMENTS.md) (실험 이력), [reference/PAPER.md](../reference/PAPER.md) (논문 baseline), [../reports/260416_mammal_3600_slerp_status_audit.md](../reports/260416_mammal_3600_slerp_status_audit.md) (상태 감사), [../reports/260327_lbs_skinning_analysis.md](../reports/260327_lbs_skinning_analysis.md) (slerp 이론)

---

## 1. Why — 왜 keyframe + slerp 인가

| 옵션 | 설정 | 예상 비용 | 품질 (100프레임 샘플 IoU) | 채택 |
|------|------|-----------|:---:|:----:|
| paper_fast (A) | step1=5, step2=3 | ~5h | mean 0.7945, min 0.5483, bad=11% | 1차 (baseline) |
| accurate dense 3600 (B) | step1=200, step2=50 | **~210 h/GPU** | - (미측정, 불가) | ❌ 비현실적 |
| **900 keyframe + slerp (C)** | accurate 900 keyframes (4 GPU 병렬) + θ slerp | **~52 h / 4 GPU** | 7.1% body error (E4) | ✅ **현재 방법** |

핵심 관찰 (E4 실험, [260323 refit report](../reports/260323_mesh_refit_experiment_report.md)):
- paper_fast 결과의 **bad frames (IoU<0.7) 는 fast-motion 구간에 집중** (5000-10000 frame)
- 동일 프레임에 accurate config 재피팅시 IoU +0.1509 개선 (0.6894 → 0.8402)
- 그러나 전체 3600 accurate dense 는 계산 비용 불가 → **keyframe 밀도** 결정 최적화로 방향 전환
- 간격 스윕 결과 M5 interval=4 (video interval=20, 0.2s) 가 body error 7.1% 로 sweet spot

## 2. How — 파이프라인 단계

```
Stage 1: Keyframe Fitting        (accurate config, 4 GPU split)
  ├─ part1: keyframes[0..224]    → GPU 4
  ├─ part2: keyframes[225..449]  → GPU 5
  ├─ part3: keyframes[450..674]  → GPU 6
  └─ part4: keyframes[675..899]  → GPU 7
                ↓ merge
Stage 2: 900 merged              (canonical params .pkl)
                ↓ θ quaternion slerp + β lerp
Stage 3: 3600 slerp-interpolated OBJ (20fps grid)
                ↓ uv_transplant_refit.py
Stage 4: 3600 textured OBJ       (geometry + static texture)
                ↓ Blender headless verify
Stage 5: 3 sample renders        (frame 0 / 9000 / 17995)
```

### 2.1 Keyframe 선정 규칙

- Raw video: 100 fps, 18000 frames
- M5 dataset alignment (pose-splatter 호환): interval=5 → 3600 M5 frames
- **Keyframe**: M5 interval=**4** → 900 keyframes = video interval=**20** = 5 fps 간격 (0.2 s)
- 파일: `exports/keyframe_indices.txt` (video frame numbers, step=20)

### 2.2 Slerp 구현

[`scripts/interpolate_keyframes.py`](../../scripts/interpolate_keyframes.py) 참조. 논리는 [`reports/260327_lbs_skinning_analysis.md`](../reports/260327_lbs_skinning_analysis.md#55-params-slerp-보간-구현):

```python
# Joint rotation: quaternion slerp (axis-angle 직접 slerp 은 불안정)
q_interp = quaternion_slerp(q_A, q_B, t)

# Shape (β): linear interpolation
β_interp = (1-t) * β_A + t * β_B

# Body model forward → vertices
vertices = body_model(θ_interp, β_interp)
```

## 3. What — 산출물

### 3.1 Data Locations (gpu03)

| Artifact | Path | Count |
|----------|------|:-----:|
| 900 fitted (split) | `results/fitting/production_keyframes_part{1..4}/` | 225×4 |
| 900 merged | `results/fitting/production_900_merged/` | 900 |
| **3600 slerp geometry** | `results/fitting/production_3600_slerp/obj/` | **3600** |
| **3600 slerp textured** | `results/fitting/production_3600_slerp/obj_textured/` | **3600** |
| Keyframe indices | `exports/keyframe_indices.txt` | 900 |
| Static texture | `exports/texture_final.png` | 1 |

### 3.2 Visualizations (gpu03 origin → 로컬 미러)

| Output | gpu03 path | Local mirror |
|--------|-----------|--------------|
| **GT overlay 3×2 grid video** | `results/comparison/production_3600_slerp_gt/grid_3x2.mp4` | `~/results/MAMMAL/260329_production_3600_slerp/gt_overlay/grid_3x2.mp4` |
| GT overlay still (frame 1800) | `.../grid_frame01800_gt_overlay.png` | `.../gt_overlay/grid_frame01800_gt_overlay.png` |
| GT overlay per-view (6) | `.../interpolated_v{0..5}.mp4` | `.../gt_overlay/interpolated_v*.mp4` |
| Mesh-only 6-view grid | `results/comparison/production_3600_slerp/sequence_6view.mp4` | `.../mesh_only/sequence_6view.mp4` |
| Mesh-only per-view (6) | `.../sequence_v{0..5}.mp4` | `.../mesh_only/sequence_v*.mp4` |
| Mesh-only still (frame 1800) | `.../grid_frame01800.png` | `.../mesh_only/grid_frame01800.png` |
| Blender texture verify (3) | `results/comparison/texture_verify/step_2_frame_{000000,009000,017995}_textured.png` | `.../texture_verify/*.png` |

### 3.3 Metrics (로컬 미러에 동반 저장)

| File | Content |
|------|---------|
| `metrics/baseline_iou_report.txt` | paper_fast 100-frame IoU 샘플 |
| `metrics/baseline_iou.json` | 동일 JSON |
| `metrics/iou_report.txt` | refit_accurate_23 비교 (fast vs accurate) |
| `metrics/iou_report.json` | 동일 JSON |
| `metrics/iou_chart.jpg` | 시각화 차트 |
| `metrics/best_frame_001920_6view.jpg`, `worst_frame_010080_6view.jpg` | accurate refit 대표 프레임 |

## 4. Reproducing

```bash
# Stage 1: keyframe fitting (per-GPU, on gpu03)
CUDA_VISIBLE_DEVICES=4 ./run_experiment.sh baseline_6view_keypoint \
    frames=keyframes_part1 optim=accurate \
    hydra.run.dir='results/fitting/production_keyframes_part1'
# ... repeat for part2/3/4 on GPU 5/6/7

# Stage 2-3: merge + slerp
python scripts/merge_keyframes.py \
    --parts results/fitting/production_keyframes_part{1,2,3,4}/ \
    --output results/fitting/production_900_merged/
python scripts/interpolate_keyframes.py \
    --keyframes results/fitting/production_900_merged/ \
    --indices exports/keyframe_indices.txt \
    --output results/fitting/production_3600_slerp/obj/

# Stage 4: UV transplant
python scripts/uv_transplant_refit.py \
    --src results/fitting/production_3600_slerp/obj/ \
    --dst results/fitting/production_3600_slerp/obj_textured/ \
    --texture exports/texture_final.png

# Stage 5: Blender verify
blender --background --python scripts/blender_texture_verify.py -- \
    --obj-dir results/fitting/production_3600_slerp/obj_textured \
    --out results/comparison/texture_verify
```

## 5. Known Limits & Next Steps

- **Self-intersection** on fast-motion frames (rearing, quick turn) — keyframe 간격 0.2 s 내에서 비선형 곡률 발생 시 slerp 직선 경로가 몸체 교차. 구체 프레임 리스트화 미수행.
- **Static texture**: 프레임별 조명/털 변화 반영 불가. FaceLift 측에 "initialization 용이지 강한 prior 아님" 명시 합의.
- **100 fps (18000 OBJ)** 까지 가려면 3600→18000 추가 slerp 단계 필요. 현재 FaceLift handoff 는 20fps 로 닫힘.
- **per-frame IoU / body error** 전수 측정 미수행 (baseline 100-frame 샘플만 존재). full 3600 IoU chart 생성은 [후속 작업](#).

---

*Engram v1.0 | Production 3600 Slerp Pipeline | 2026-04-16*
