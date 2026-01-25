# Project: MAMMAL_mouse

## Overview

Markerless 3D mouse pose estimation and mesh reconstruction from multi-view video.
Based on MAMMAL (An et al., Nature Communications 2023).

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
| 로그 | `results/logs/` (hydra/, runtime/) |
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

## Architecture (after refactoring)

```
MAMMAL_mouse/
├── [ORIGINAL] fitter_articulation.py, articulation_th.py, bodymodel_*.py
├── [EXTENSION] mammal_ext/           # 확장 모듈 (원본 수정 최소화)
│   └── config/                       # GPU, loss, keypoint 설정
├── [RESULTS] results/                # 통합된 결과 폴더
│   ├── fitting/                      # 피팅 결과
│   ├── logs/hydra/                   # Hydra 로그
│   └── logs/runtime/                 # 런타임 로그
├── conf/                             # Hydra 설정
└── tests/                            # 유닛 테스트
```
