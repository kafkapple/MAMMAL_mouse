# Visualization Guide

> 메쉬 시각화, 비디오 생성, 실험 비교, Blender/Rerun 연동 가이드

---

## Quick Reference

가장 자주 사용하는 `mesh_visualizer` 핵심 명령어 모음.

```bash
# 샘플 프레임 렌더링 (첫 프레임만, 빠른 확인)
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --start_frame 0 --end_frame 1 \
    --save_video --no_rrd

# 전체 시퀀스 비디오 (orbit + fixed 뷰)
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --view_modes orbit fixed \
    --save_video --save_rrd

# 키포인트 + 스켈레톤 포함 렌더링
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --view_modes orbit \
    --show_keypoints --show_skeleton \
    --save_video --no_rrd
```

---

## mesh_visualizer 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--result_dir` | (필수) | 피팅 결과 디렉토리 |
| `--view_modes` | orbit fixed | 카메라 모드 (복수 가능) |
| `--start_frame` | 0 | 시작 프레임 |
| `--end_frame` | -1 | 끝 프레임 (-1 = 전체) |
| `--frame_interval` | 1 | 프레임 간격 |
| `--save_video` | True | MP4 저장 |
| `--save_rrd` | True | Rerun RRD 저장 |
| `--no_video` | - | 비디오 비활성화 |
| `--no_rrd` | - | RRD 비활성화 |
| `--fps` | 30 | 비디오 FPS |
| `--image_size` | 1024 1024 | 렌더 크기 (width height) |
| `--show_keypoints` | - | 3D 키포인트 표시 |
| `--show_skeleton` | - | 스켈레톤 표시 |

---

## 뷰 모드

### Orbit

360도 회전 뷰. 메쉬 주위를 한 바퀴 돌며 렌더링한다.
전체적인 형상과 텍스처 품질을 확인하는 데 적합하다.

### Fixed

고정 카메라 뷰 (6개 방향). 특정 각도에서의 정적 렌더링.
front, back, left, right, top, bottom 6개 시점에서 촬영한다.

### Novel Views

새로운 시점에서의 렌더링. 학습에 사용되지 않은 각도에서 메쉬를 평가한다.

```bash
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --view_modes novel \
    --save_video
```

---

## 시퀀스 시각화 & 비디오 생성

### 전체 시퀀스

전체 프레임을 순서대로 렌더링하여 비디오를 생성한다.

```bash
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --view_modes orbit fixed \
    --save_video --save_rrd
```

### 프레임 범위 지정

특정 구간만 렌더링하여 빠르게 확인한다.

```bash
# 0~9번 프레임만
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --start_frame 0 --end_frame 10 \
    --save_video --no_rrd

# 매 5프레임마다 (빠른 미리보기)
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --frame_interval 5 \
    --save_video --no_rrd
```

---

## 비교 스크립트

실험 간 결과를 비교하는 HTML 보고서를 생성한다.

```bash
# glob 패턴으로 여러 실험 비교
python scripts/compare_experiments.py \
    "results/fitting/*sparse3*" \
    "results/fitting/*sparse5*" \
    --output comparison.html

# 특정 실험 지정
python scripts/compare_experiments.py \
    results/fitting/exp1_20251210 \
    results/fitting/exp2_20251211 \
    --output ab_test.html
```

생성된 HTML 파일을 브라우저에서 열면, 실험별 메트릭과 렌더링 결과를 나란히 비교할 수 있다.

---

## Blender / Rerun 연동

### Blender Export

텍스처가 적용된 OBJ 파일을 생성하여 Blender에서 활용한다.
상세 내용은 [UVMAP_GUIDE.md](UVMAP_GUIDE.md)의 "Blender 내보내기" 섹션을 참조한다.

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

### Rerun Export

Rerun 뷰어에서 인터랙티브하게 3D 시퀀스를 탐색할 수 있다.

```bash
# RRD 파일 생성
python scripts/export_to_rerun.py \
    --result_dir results/fitting/exp \
    --texture results/uvmap/texture_final.png \
    --output exports/sequence.rrd

# Rerun 뷰어로 열기 (로컬)
rerun exports/sequence.rrd

# 웹 뷰어로 열기 (원격 서버)
rerun --web-viewer --port 9090 exports/sequence.rrd
```

Rerun 뷰어에서는 타임라인 스크러빙, 카메라 자유 이동, 키포인트/스켈레톤 on/off 등이 가능하다.

---

## 출력 구조

`mesh_visualizer` 및 관련 도구의 출력물이 저장되는 위치.

```
results/fitting/<exp_dir>/
├── visualization/           # mesh_visualizer 출력
│   ├── orbit.mp4           # 360도 회전 비디오
│   ├── fixed.mp4           # 고정 뷰 비디오
│   └── sequence.rrd        # Rerun 파일
│
├── render_samples/          # 샘플 렌더링
│   └── step_2_frame_*.png
│
├── obj/                     # 3D 메쉬 (피팅 결과)
│   └── step_2_frame_*.obj
│
└── uvmap/                   # UV 텍스처 (별도 생성)
    ├── texture_final.png
    ├── confidence.png
    └── uv_mask.png
```

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [UVMAP_GUIDE.md](UVMAP_GUIDE.md) | UV 텍스처 생성 및 Blender 내보내기 |
| [EXPERIMENTS.md](../reference/EXPERIMENTS.md) | 실험 실행 가이드 |
| [OUTPUT_FORMAT.md](../reference/OUTPUT_FORMAT.md) | 출력 형식 상세 |

---

*Source: VISUALIZATION.md*
*Last updated: 2026-02-06*
