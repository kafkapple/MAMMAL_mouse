# Project: MAMMAL_mouse

## Overview

Markerless 3D mouse pose estimation and mesh reconstruction from multi-view video.
Based on MAMMAL (An et al., Nature Communications 2023).

## Repository

| 항목 | URL |
|------|-----|
| **원본** | https://github.com/anl13/MAMMAL_mouse |
| **수정본** | https://github.com/kafkapple/MAMMAL_mouse |

**수정 원칙**: 원본 코드 수정 최소화. 모든 확장/수정은 `mammal_ext/` 하위에 모듈화하여 구현. 원본 파일 직접 수정 금지.

## Tech Stack

- **Language**: Python 3.10
- **Framework**: PyTorch, Hydra
- **3D**: PyTorch3D, Trimesh, Pyrender
- **Tracking**: WandB

## Key Paths

| 항목 | 경로 |
|------|------|
| 데이터 | `data/examples/markerless_mouse_1_nerf/` |
| 결과 | `results/fitting/` |
| 로그 | `logs/` (nohup), `results/logs/` (hydra) |
| Config | `conf/` |
| 확장 모듈 | `mammal_ext/` (GPU, loss, keypoint config) |
| 문서 | `docs/MOC.md` (진입점) |

## Commands

```bash
# 전체 피팅 (논문 설정, ~10시간)
./run_experiment.sh baseline_6view_keypoint frames=aligned_posesplatter optim=paper_fast

# 테스트 (100프레임)
./run_experiment.sh baseline_6view_keypoint frames=aligned_test_100 optim=paper_fast

# 디버그 (5프레임)
./run_experiment.sh baseline_6view_keypoint --debug

# 피팅 후 시각화
python -m visualization.mesh_visualizer --result_dir results/fitting/<exp> --save_video
```

## Conventions

### Config Override (Hydra)
```bash
./run_experiment.sh <experiment> frames=<frame_config> optim=<optim_config>
```

### Frame Configs
- `aligned_posesplatter`: 3600 프레임 (전체, pose-splatter 정렬)
- `aligned_test_100`: 100 프레임 (테스트)

### Optim Configs
- `paper_fast`: 논문 설정 + 렌더링 비활성화 (최고속)
- `paper`: 논문 설정 + 렌더링

## Paper Settings Reference

> An et al., "MAMMAL", Nature Communications 2023

| 항목 | 값 | 근거 |
|------|-----|------|
| step0_iters | 60 | 첫 프레임 초기화 |
| step1_iters | 5 | "3-5 iterations" |
| step2_iters | 3 | "3 iterations" |
| mask_loss | 0.0 | "wsil=0" |

## Dataset Specs

| 항목 | 값 |
|------|-----|
| 카메라 | 6 views |
| FPS | 100 |
| 총 프레임 | 18,000 |
| frame_jump=5 | 3,600 샘플 |
| 키포인트 | 22개 |

## Related Projects

- **pose-splatter**: 3D Gaussian Splatting for pose (downstream)
  - MAMMAL 메쉬를 3D prior로 사용
  - Frame alignment: interval=5 일치 필수

## Documentation

- **진입점**: `docs/MOC.md`
- **실험**: `docs/practical/EXPERIMENTS.md`
- **시각화**: `docs/practical/VISUALIZATION.md`
- **논문 설정**: `docs/reference/MAMMAL_PAPER.md`
- **코드 분석**: `docs/reports/CODEBASE_ANALYSIS.md`
- **리팩토링**: `docs/reports/REFACTORING_PLAN.md`
- **좌표계**: `docs/coordinates/coordinate_systems_reference.md`
- **UV→Blender**: `docs/practical/UV_TEXTURE_TO_BLENDER.md`

## Architecture (after refactoring)

```
MAMMAL_mouse/
├── [ORIGINAL] fitter_articulation.py, articulation_th.py, bodymodel_*.py
├── [EXTENSION] mammal_ext/           # 확장 모듈 (원본 수정 최소화)
│   ├── config/                       # GPU, loss, keypoint 설정
│   ├── fitting/                      # Debug grid, fitting utilities
│   ├── visualization/                # Mesh viz, video gen, Rerun export
│   ├── preprocessing/                # Mask, keypoint, SAM inference
│   ├── uvmap/                        # UV texture mapping pipeline
│   └── blender_export/              # Blender OBJ export, coord transform, 6-view grid
├── [COMPAT] visualization/, preprocessing_utils/, uvmap/  # Backward compat wrappers
├── [RESULTS] results/                # 결과 폴더
│   ├── fitting/                      # 피팅 결과
│   └── logs/                         # Hydra 로그
├── [LOGS] logs/                      # nohup 런타임 로그
├── conf/                             # Hydra 설정
└── tests/                            # 유닛 테스트
```

## Module Import Guide

```python
# Recommended (new)
from mammal_ext.visualization import MeshVisualizer
from mammal_ext.preprocessing.keypoint_estimation import estimate_mammal_keypoints
from mammal_ext.uvmap import UVPipeline
from mammal_ext.config import configure_gpu

# Deprecated (backward compatible, shows warning)
from visualization import MeshVisualizer
from preprocessing_utils.keypoint_estimation import estimate_mammal_keypoints

# Blender export
from mammal_ext.blender_export import transform_vertices, export_obj_with_uv
from mammal_ext.blender_export.batch_export import batch_export
from mammal_ext.blender_export.sequence_renderer import render_sequence
```

## Coordinate Systems (Quick Reference)

> 상세: `docs/coordinates/coordinate_systems_reference.md`

### 좌표계 정의

| 좌표계 | Up | Forward | Right | 대표 |
|--------|-----|---------|-------|------|
| **MAMMAL** | **-Y** | +X (head) | +Z | MAMMAL fitting |
| **Blender World** | +Z | +Y | +X | Blender |
| **OpenCV** | -Y (down) | +Z | +X | OpenCV, COLMAP |
| **OpenGL** | +Y | -Z | +X | Blender camera |

### MAMMAL 메쉬 특성

- 단위: **mm** (체장 ~115mm, 높이 ~53mm, 폭 ~41mm)
- X=body length, Y=height (+Y=back), Z=width

### 좌표 변환

| From → To | 공식 | 행렬 |
|-----------|------|------|
| **MAMMAL → Blender World** | `(x, z, -y)` | Rx(+90°) |
| Blender World → MAMMAL | `(x, -z, y)` | Rx(-90°) |
| OpenGL → Blender World | `(x, -z, y)` | Rx(-90°) |

```python
# MAMMAL → Blender World
from mammal_ext.blender_export import transform_vertices, MAMMAL_TO_BLENDER
vertices_blender = transform_vertices(vertices_mammal)  # center + mm→m + Rx(+90°)
```

### Blender 임포트 검증

정상 기준: 등(back)이 +Z(위), 배(belly)가 -Z(아래), 크기 ~0.1m

## Modularization Rules

**원본 코드 수정 시 반드시 준수:**

1. **원본 파일 직접 수정 금지** — `mammal_ext/` 하위에 모듈 생성
2. **모듈 구조**: `mammal_ext/{기능명}/` (예: `blender_export/`, `visualization/`)
3. **Backward compat wrapper**: 기존 import 경로 유지 필요 시 루트에 래퍼 파일 배치
4. **scripts/**: CLI 진입점은 thin wrapper로, 실제 로직은 `mammal_ext/`에
