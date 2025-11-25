# Keypoint Estimation Pipeline 확장 계획

**날짜**: 2025-11-25
**목적**: Keypoint 없는 데이터셋에 대한 자동 추정 및 웹 기반 어노테이션 파이프라인 구축

---

## 1. 현황 분석

### 1.1 기존 구현 현황

| 모듈 | 파일 | 상태 | 설명 |
|------|------|------|------|
| Geometric (PCA) | `keypoint_estimation.py` | ✅ 완료 | Mask → 22 keypoints (no training) |
| SuperAnimal/DLC | `superanimal_detector.py` | ⚠️ 부분 | DLC 모델 로딩, 27 keypoints |
| YOLO-Pose | `yolo_keypoint_detector.py` | ⚠️ 부분 | YOLOv8-Pose wrapper |
| DANNCE→YOLO | `dannce_to_yolo.py` | ✅ 완료 | 데이터 변환 유틸 |

### 1.2 문제점

1. **SuperAnimal/YOLO 모델 로딩 경로 하드코딩**
2. **통합 인터페이스 부재**: 각 detector가 독립적
3. **웹 기반 어노테이션 미지원**
4. **fine-tuning 파이프라인 없음**

---

## 2. 사용 가능한 Pre-trained 모델 조사

### 2.1 SuperAnimal (DeepLabCut)

**출처**: [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48792-2)

| 모델 | Keypoints | 특징 |
|------|-----------|------|
| SuperAnimal-TopViewMouse | 27 | Top-view 마우스 특화, C57 bias |
| SuperAnimal-Quadruped | 39 | 45+ 포유류 지원 |

**장점**:
- Zero-shot 성능 우수
- Fine-tuning 시 10-100x 데이터 효율적
- [HuggingFace 모델](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)

**단점**:
- DLC 환경 의존성 복잡
- 27 keypoints → MAMMAL 22 keypoints 매핑 필요

### 2.2 YOLO-Pose (Ultralytics)

**출처**: [Ultralytics Docs](https://docs.ultralytics.com/tasks/pose/)

| 모델 | 속도 | 정확도 | 용도 |
|------|------|--------|------|
| YOLOv8n-pose | 매우 빠름 | 낮음 | 실시간 |
| YOLOv8s-pose | 빠름 | 중간 | 균형 |
| YOLOv8m-pose | 중간 | 높음 | 권장 |
| YOLOv8x-pose | 느림 | 최고 | 정밀 분석 |

**장점**:
- 설치 간단 (`pip install ultralytics`)
- Custom 학습 용이
- 실시간 inference

**단점**:
- 마우스 pre-trained 모델 없음 (학습 필요)

### 2.3 STPoseNet (YOLOv8 기반)

**출처**: [iScience 2024](https://www.cell.com/iscience/fulltext/S2589-0042(24)00994-5)

- YOLOv8 + Temporal tracking + Kalman filter
- 마우스 특화 (nose, ears, spine×5, hind legs, tail)
- DLC, SLEAP 대비 우수한 성능

---

## 3. Web 기반 어노테이션 도구 비교

### 3.1 CVAT vs Label Studio

| 기능 | [CVAT](https://www.cvat.ai/) | [Label Studio](https://labelstud.io/) |
|------|-----|-------------|
| **Pose/Keypoint** | ⭐⭐⭐⭐⭐ 최적화 | ⭐⭐⭐ 지원 |
| **Video 지원** | ⭐⭐⭐⭐⭐ Interpolation | ⭐⭐⭐ 기본 |
| **설치** | Docker 권장 | pip 가능 |
| **확장성** | API 지원 | SDK/API 풍부 |
| **ML 연동** | 제한적 | 강력 (Active Learning) |
| **Multi-modal** | Vision only | Text, Audio 지원 |

**권장**: **CVAT** (Pose estimation 특화)

### 3.2 DeepLabCut GUI

- Built-in labeling interface
- napari 기반
- DLC 생태계와 완전 통합

---

## 4. 제안 아키텍처

### 4.1 모듈 구조

```
mammal-keypoint-pipeline/
├── mammal_keypoints/           # Python package (pip installable)
│   ├── __init__.py
│   ├── detectors/              # Keypoint detection backends
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class
│   │   ├── geometric.py        # PCA-based (existing)
│   │   ├── superanimal.py      # DLC SuperAnimal (existing)
│   │   ├── yolo_pose.py        # YOLO-Pose (existing)
│   │   └── ensemble.py         # Multi-detector ensemble
│   ├── converters/             # Format conversions
│   │   ├── dlc_to_mammal.py    # 27 → 22 keypoints
│   │   ├── yolo_to_mammal.py   # Custom → 22 keypoints
│   │   └── mammal_to_coco.py   # Export to COCO format
│   ├── annotators/             # Annotation interfaces
│   │   ├── gradio_app.py       # Web UI (Gradio)
│   │   ├── cvat_integration.py # CVAT API wrapper
│   │   └── dlc_napari.py       # DLC napari integration
│   ├── training/               # Fine-tuning utilities
│   │   ├── yolo_trainer.py     # YOLO custom training
│   │   └── dlc_trainer.py      # DLC fine-tuning
│   └── cli.py                  # Command-line interface
├── configs/                    # YAML configurations
│   ├── detectors/
│   │   ├── superanimal.yaml
│   │   ├── yolo_pose.yaml
│   │   └── geometric.yaml
│   └── keypoint_mappings/
│       ├── superanimal_to_mammal.yaml
│       └── yolo_to_mammal.yaml
├── models/                     # Pre-trained model weights
│   └── README.md               # Download instructions
├── scripts/
│   ├── download_models.sh      # Auto model download
│   ├── run_detector.py         # CLI detector runner
│   └── start_annotation_server.py
├── docker/
│   ├── Dockerfile              # Full environment
│   └── docker-compose.yml      # With CVAT integration
├── pyproject.toml              # Modern Python packaging
└── README.md
```

### 4.2 통합 인터페이스

```python
from mammal_keypoints import KeypointDetector

# Factory pattern으로 detector 선택
detector = KeypointDetector.create(
    backend='superanimal',  # 'geometric', 'yolo', 'ensemble'
    config='configs/detectors/superanimal.yaml'
)

# 통일된 출력 (MAMMAL 22 keypoints)
keypoints = detector.detect(image, mask=mask)
# Returns: (22, 3) array [x, y, confidence]

# Batch processing
results = detector.process_directory(
    input_dir='data/images/',
    output_dir='data/keypoints/',
    visualize=True
)
```

### 4.3 Web 어노테이션 서버

```python
# Gradio 기반 간단 구현
import gradio as gr
from mammal_keypoints.annotators import GradioAnnotator

app = GradioAnnotator(
    detector='superanimal',
    output_format='mammal22'
)
app.launch(server_name='0.0.0.0', server_port=7860, share=True)
```

**기능**:
- 이미지/비디오 업로드
- Auto-detection + 수동 수정
- MAMMAL 22 keypoints 표시
- Export (JSON, COCO, YOLO format)

---

## 5. Keypoint 매핑 전략

### 5.1 SuperAnimal (27) → MAMMAL (22)

```yaml
# configs/keypoint_mappings/superanimal_to_mammal.yaml
mapping:
  # MAMMAL Head (0-5)
  0: nose                    # SA: nose
  1: left_ear               # SA: left_ear
  2: right_ear              # SA: right_ear
  3: left_eye               # SA: left_eye (estimated)
  4: right_eye              # SA: right_eye (estimated)
  5: head_center            # SA: average(nose, ears)

  # MAMMAL Spine (6-13)
  6-13: spine_1 to spine_8  # SA: spine points (interpolate if needed)

  # MAMMAL Limbs (14-17)
  14: left_front_paw        # SA: left_front_paw
  15: right_front_paw       # SA: right_front_paw
  16: left_hind_paw         # SA: left_hind_paw
  17: right_hind_paw        # SA: right_hind_paw

  # MAMMAL Tail (18-20)
  18: tail_base             # SA: tail_base
  19: tail_mid              # SA: interpolate
  20: tail_tip              # SA: tail_tip

  # MAMMAL Centroid (21)
  21: centroid              # SA: computed from all points
```

### 5.2 YOLO Custom (12) → MAMMAL (22)

```yaml
# Custom YOLO 학습 시 MAMMAL 22 직접 사용 권장
# 또는 핵심 12개 학습 후 나머지 보간
core_keypoints:
  - nose, left_ear, right_ear
  - spine_start, spine_mid, spine_end
  - 4 paws
  - tail_base, tail_tip

interpolation:
  - eyes: from head geometry
  - spine_2-7: linear interpolation
  - tail_mid: from base/tip
```

---

## 6. 구현 우선순위

### Phase 1: Core Pipeline (1-2주)

1. **통합 Detector 인터페이스** (`detectors/base.py`)
2. **기존 코드 리팩토링** (geometric, superanimal, yolo)
3. **Keypoint 매핑 유틸** (converters/)
4. **CLI 도구** (`python -m mammal_keypoints detect ...`)

### Phase 2: Web Annotation (1주)

1. **Gradio 기반 Web UI**
   - Auto-detection preview
   - Manual correction
   - Batch export
2. **외부 접속 설정** (ngrok 또는 port forwarding)

### Phase 3: Training Pipeline (2주)

1. **YOLO fine-tuning** (MAMMAL 22 keypoints)
2. **DLC fine-tuning** wrapper
3. **Active learning** (불확실한 샘플 우선 어노테이션)

### Phase 4: Packaging (1주)

1. **pip installable** (`pip install mammal-keypoints`)
2. **Docker image**
3. **Documentation**

---

## 7. 기술적 고려사항

### 7.1 의존성 관리

```toml
# pyproject.toml
[project]
dependencies = [
    "numpy>=1.20",
    "opencv-python>=4.5",
    "torch>=2.0",
]

[project.optional-dependencies]
superanimal = ["deeplabcut>=2.3"]
yolo = ["ultralytics>=8.0"]
web = ["gradio>=4.0"]
all = ["mammal-keypoints[superanimal,yolo,web]"]
```

### 7.2 모델 다운로드 자동화

```bash
# scripts/download_models.sh
#!/bin/bash

# SuperAnimal from HuggingFace
huggingface-cli download mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse \
    --local-dir models/superanimal_topviewmouse

# YOLO-Pose base model
wget -P models/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt
```

### 7.3 서버 외부 접속

```python
# Option 1: Gradio share
app.launch(share=True)  # 자동 public URL 생성

# Option 2: ngrok
# ngrok http 7860

# Option 3: SSH tunnel
# ssh -R 80:localhost:7860 serveo.net
```

---

## 8. 예상 결과물

### 8.1 사용 시나리오

```bash
# 1. 새 데이터셋에 keypoint 자동 추정
mammal-keypoints detect \
    --input data/new_mouse_videos/ \
    --output data/keypoints/ \
    --detector superanimal \
    --visualize

# 2. Web UI로 수동 수정
mammal-keypoints annotate \
    --input data/keypoints/ \
    --port 7860 \
    --share

# 3. YOLO 모델 fine-tuning
mammal-keypoints train \
    --data data/annotated/ \
    --model yolov8m-pose \
    --epochs 100

# 4. MAMMAL mesh fitting에 활용
python fit_monocular.py \
    --input_dir data/images/ \
    --keypoints data/keypoints/ \
    --detector precomputed
```

### 8.2 디렉토리 구조 예시

```
data/
├── raw_video.mp4
├── frames/
│   ├── 000000.png
│   └── ...
├── masks/
│   ├── 000000_mask.png
│   └── ...
├── keypoints/                  # 자동 생성
│   ├── 000000_keypoints.json
│   ├── 000000_keypoints.png    # 시각화
│   └── ...
├── keypoints_corrected/        # 수동 수정 후
│   └── ...
└── mesh_fitting_results/
    └── ...
```

---

## 9. 피드백 요청 사항

1. **우선순위**: Phase 1-4 중 가장 급한 기능은?
2. **Web UI**: Gradio vs CVAT 통합 vs 별도 개발?
3. **Keypoint 수**: MAMMAL 22개 유지 vs 축소 (핵심 12개)?
4. **배포 방식**: pip package vs Docker vs 둘 다?
5. **추가 기능**: Active learning, temporal smoothing 필요?

---

## Sources

- [DeepLabCut SuperAnimal](https://www.nature.com/articles/s41467-024-48792-2)
- [SuperAnimal-TopViewMouse on HuggingFace](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/)
- [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- [STPoseNet for Mouse Pose](https://www.cell.com/iscience/fulltext/S2589-0042(24)00994-5)
- [Animal Pose with YOLOv8](https://learnopencv.com/animal-pose-estimation/)
