# MAMMAL_EXT Architecture Reference

`mammal_ext/` 확장 모듈의 전체 아키텍처와 통합 패턴을 설명합니다.

---

## 개요

**mammal_ext**는 MAMMAL 프로젝트의 핵심 기능을 확장하는 독립적인 Python 패키지입니다.

| 항목 | 내용 |
|------|------|
| **총 코드량** | 11,719 lines, 36 files |
| **버전** | v0.2.0 |
| **통합 방식** | Delegation pattern (monkey patching 아님) |
| **원본 수정** | 3 files, ~8 lines만 수정 |
| **의존성** | Optional dependencies (try/except fallback) |

---

## 디렉토리 구조

```
mammal_ext/
├── __init__.py                    # v0.2.0 버전 정의
├── model_loader.py                # ⭐ Centralized ArticulationTorch loader
│
├── config/           (336L)       # GPU 자동 감지, loss/keypoint weights
│   ├── env_config.py              # hostname 기반 GPU/conda 자동 설정
│   └── loss_config.py             # loss_weights, keypoint_weights 중앙 관리
│
├── fitting/          (274L)       # 피팅 확장
│   └── debug_collector.py         # DebugGridCollector (이미지 압축)
│
├── preprocessing/    (2,465L)     # 데이터 전처리
│   ├── sam_inference.py           # SAM segmentation
│   ├── superanimal_preprocess.py  # SuperAnimal keypoint estimation
│   ├── yolo_detector.py           # YOLO bounding box detection
│   └── keypoint_estimator.py      # DeepLabCut integration
│
├── uvmap/            (4,988L)     # UV texture pipeline
│   ├── uv_pipeline.py             # Main UV mapping CLI
│   ├── texture_processor.py       # Texture projection/optimization
│   ├── wandb_sweep.py             # WandB hyperparameter sweep
│   └── optuna_hpo.py              # Optuna HPO integration
│
├── visualization/    (2,777L)     # 시각화
│   ├── mesh_visualizer.py         # Mesh + texture visualization
│   ├── rerun_logger.py            # Rerun SDK integration
│   └── video_generator.py         # Turntable/multi-view video
│
└── blender_export/   (815L)       # Blender 연동
    ├── batch_export.py            # OBJ sequence export
    └── coordinate_transform.py    # Y-up ↔ Z-up 변환
```

---

## 핵심 통합 패턴

### Delegation Pattern (Not Monkey Patching)

**원본 MAMMAL 코드 수정 최소화**:
- 수정 파일: 3개 (`fitter_articulation.py`, `articulation_th.py`, `fit_monocular.py`)
- 수정 라인: ~8 lines
- 수정 내용: `mammal_ext.model_loader` import 추가

**예시 (fitter_articulation.py)**:
```python
# Original
self.artic = ArticulationTorch()

# Modified (delegation)
try:
    from mammal_ext.model_loader import load_articulation_model
    self.artic = load_articulation_model()
except ImportError:
    self.artic = ArticulationTorch()  # Fallback
```

### Reverse Dependency 중앙화

**`model_loader.py`**: Reverse dependency (mammal_ext → MAMMAL)를 한 곳에 집중
```python
from third_party.articulation.articulation_th import ArticulationTorch

def load_articulation_model():
    # mammal_ext 설정 적용 후 ArticulationTorch 반환
    return ArticulationTorch()
```

**장점**:
- 원본 코드는 mammal_ext를 import하지 않음
- mammal_ext는 MAMMAL을 import하지만 단일 파일에 격리
- Optional dependency로 설치 없이도 MAMMAL 단독 실행 가능

---

## 모듈 의존성 그래프

```
┌─────────────────────────────────────────────────────────────┐
│                   Original MAMMAL                           │
│          (ArticulationTorch, FitterArticulation)            │
└────────────────────────┬────────────────────────────────────┘
                         │ imports (try/except)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              mammal_ext (Tier 1: Integration)               │
│     config/  fitting/  preprocessing/  model_loader.py      │
│              ← Original 코드에서 사용                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ imports (reverse dependency)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            mammal_ext (Tier 2: Standalone)                  │
│       uvmap/  visualization/  blender_export/               │
│              ← 독립적 CLI, MAMMAL 코드 import               │
└─────────────────────────────────────────────────────────────┘
```

**Tier 1 (Integration)**: Original 코드가 import (가벼운 유틸리티)
**Tier 2 (Standalone)**: 독립 실행 CLI (무거운 dependency)

---

## 주요 CLI 명령어

### UV Mapping
```bash
python -m mammal_ext.uvmap.uv_pipeline \
    --result_dir /path/to/results/fitting/exp_name \
    --output_dir /path/to/uvmap_output
```

### Mesh Visualization
```bash
python -m mammal_ext.visualization.mesh_visualizer \
    --result_dir /path/to/results/fitting/exp_name \
    --output_dir /path/to/visualizations
```

### Blender Export
```bash
python -m mammal_ext.blender_export.batch_export \
    --result_dir /path/to/results/fitting/exp_name \
    --output_dir /path/to/blender_objs
```

### SAM Segmentation
```bash
python -m mammal_ext.preprocessing.sam_inference \
    --input /path/to/images \
    --output /path/to/masks
```

---

## 의존성 관리

### Core Dependencies (필수)
```
torch, numpy, omegaconf, tqdm
```

### Optional Dependencies (모듈별)
| 모듈 | 의존성 | 설치 여부 확인 |
|------|--------|--------------|
| `uvmap/` | pytorch3d, wandb, optuna | try/except |
| `visualization/` | rerun-sdk, matplotlib | try/except |
| `preprocessing/` | segment-anything, deeplabcut | try/except |
| `blender_export/` | trimesh (optional) | try/except |

**Fallback 전략**: Optional dependency 없으면 해당 기능만 비활성화, 전체 패키지 import는 성공

---

## 핵심 기능 요약

### 1. config/ - 환경 자동 감지
- **env_config.py**: hostname 기반 GPU/conda 경로 자동 설정 (gpu03 vs joon)
- **loss_config.py**: loss_weights, keypoint_weights 중앙 관리

### 2. fitting/ - 디버깅 강화
- **debug_collector.py**: DebugGridCollector (per-frame debug images → grid 압축)

### 3. preprocessing/ - 데이터 준비
- **SAM**: Zero-shot segmentation
- **SuperAnimal**: Cross-species keypoint estimation
- **YOLO**: Bounding box detection

### 4. uvmap/ - UV Texture Pipeline
- **uv_pipeline.py**: Multi-view image → UV texture map
- **wandb_sweep.py**: Hyperparameter sweep (projection weights, blur)
- **optuna_hpo.py**: Automatic hyperparameter optimization

### 5. visualization/ - 시각화
- **mesh_visualizer.py**: Mesh + texture + skeleton 렌더링
- **rerun_logger.py**: Rerun SDK로 3D 시각화
- **video_generator.py**: Turntable/multi-view 비디오 생성

### 6. blender_export/ - Blender 연동
- **batch_export.py**: OBJ sequence export (per-frame mesh)
- **coordinate_transform.py**: MAMMAL (Y-up) ↔ Blender (Z-up) 변환

---

## 설치 및 사용

### 설치 (Editable Mode)
```bash
cd /home/joon/dev/MAMMAL_mouse
pip install -e .
```

### Import 패턴
```python
# Tier 1 (Integration) - Original 코드에서 사용
from mammal_ext.config.env_config import get_gpu_id
from mammal_ext.model_loader import load_articulation_model

# Tier 2 (Standalone) - CLI로 실행
python -m mammal_ext.uvmap.uv_pipeline --result_dir ...
```

---

## 버전 히스토리

| 버전 | 날짜 | 변경 사항 |
|------|------|-----------|
| **v0.1.0** | 2026-01-19 | Initial release (config, fitting, preprocessing) |
| **v0.2.0** | 2026-01-26 | uvmap, visualization, blender_export 추가 |

---

## 관련 문서

| 문서 | 위치 | 설명 |
|------|------|------|
| **Architecture** | `docs/reference/ARCHITECTURE.md` | MAMMAL 전체 아키텍처 |
| **Fitting Guide** | `docs/guides/FITTING_GUIDE.md` | 피팅 워크플로우 |
| **UVMap Guide** | `docs/guides/UVMAP_GUIDE.md` | UV texture mapping 가이드 |
| **Visualization** | `docs/guides/VISUALIZATION_GUIDE.md` | 시각화 도구 사용법 |

---

*MAMMAL_mouse Extension | v0.2.0 | 2026-02-07*
