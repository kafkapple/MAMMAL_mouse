# Visualization Guide

> 메쉬 시각화 및 비디오 생성

---

## Quick Commands

```bash
# 샘플 프레임 렌더링 (첫/마지막)
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --start_frame 0 --end_frame 1 \
    --save_video --no_rrd

# 전체 시퀀스 비디오
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --view_modes orbit fixed \
    --save_video --save_rrd
```

---

## mesh_visualizer 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--result_dir` | (필수) | 피팅 결과 디렉토리 |
| `--view_modes` | orbit fixed | 카메라 모드 |
| `--start_frame` | 0 | 시작 프레임 |
| `--end_frame` | -1 | 끝 프레임 (-1 = 전체) |
| `--frame_interval` | 1 | 프레임 간격 |
| `--save_video` | True | MP4 저장 |
| `--save_rrd` | True | Rerun RRD 저장 |
| `--no_video` | - | 비디오 비활성화 |
| `--no_rrd` | - | RRD 비활성화 |
| `--fps` | 30 | 비디오 FPS |
| `--image_size` | 1024 1024 | 렌더 크기 |
| `--show_keypoints` | - | 3D 키포인트 표시 |
| `--show_skeleton` | - | 스켈레톤 표시 |

### View Modes

| Mode | 설명 |
|------|------|
| orbit | 360° 회전 |
| fixed | 고정 카메라 (6개 뷰) |
| novel | 새로운 시점 |

---

## 실험 비교

```bash
# HTML 보고서 생성
python scripts/compare_experiments.py \
    "results/fitting/*sparse3*" \
    "results/fitting/*sparse5*" \
    --output comparison.html

# 특정 실험들
python scripts/compare_experiments.py \
    results/fitting/exp1_20251210 \
    results/fitting/exp2_20251211 \
    --output ab_test.html
```

---

## Blender Export

```bash
# 단일 프레임 (텍스처 포함)
python scripts/export_to_blender.py \
    --mesh results/fitting/exp/obj/step_2_frame_000000.obj \
    --texture results/uvmap/texture_final.png \
    --output exports/mouse_textured.obj

# 배치 (여러 프레임)
for i in $(seq -f "%06g" 0 9); do
    python scripts/export_to_blender.py \
        --mesh results/fitting/exp/obj/step_2_frame_${i}.obj \
        --texture results/uvmap/texture_final.png \
        --output exports/mouse_frame${i}.obj
done
```

---

## Rerun Export

```bash
# RRD 파일 생성
python scripts/export_to_rerun.py \
    --result_dir results/fitting/exp \
    --texture results/uvmap/texture_final.png \
    --output exports/sequence.rrd

# Rerun 뷰어로 열기
rerun exports/sequence.rrd
```

---

## Output

```
results/fitting/<exp_dir>/
├── visualization/           # mesh_visualizer 출력
│   ├── orbit.mp4           # 360° 회전 비디오
│   ├── fixed.mp4           # 고정 뷰 비디오
│   └── sequence.rrd        # Rerun 파일
│
└── render_samples/          # 샘플 렌더링
    └── step_2_frame_*.png
```

---

## Related Documents

- [EXPERIMENTS.md](EXPERIMENTS.md) - 실험 실행
- [../reference/OUTPUT_FORMAT.md](../reference/OUTPUT_FORMAT.md) - 출력 형식

---

*Last updated: 2026-01-25*
