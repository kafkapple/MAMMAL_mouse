# Annotation Guide

MAMMAL_mouse 프로젝트의 어노테이션 관련 모든 정보를 하나로 통합한 가이드.

---

## 목차

1. [Quick Start](#quick-start)
2. [도구 비교](#도구-비교)
3. [22 키포인트 정의](#22-키포인트-정의)
4. [키포인트 어노테이션](#키포인트-어노테이션)
5. [SAM 마스크 생성](#sam-마스크-생성)
6. [Unified Annotator](#unified-annotator)
7. [워크플로우: 프레임 선택 → 내보내기](#워크플로우)
8. [Manual Labeling → YOLOv8 파이프라인](#manual-labeling--yolov8-파이프라인)
9. [MAMMAL 포맷 변환 및 Mesh Fitting](#mammal-포맷-변환-및-mesh-fitting)
10. [팁 & Best Practices](#팁--best-practices)
11. [문제 해결](#문제-해결)

---

## Quick Start

**5분 안에 시작하는 최단 경로**

### 목적별 빠른 선택

| 필요 작업 | 사용 도구 | 명령어 |
|-----------|-----------|--------|
| **Keypoint만** | `keypoint_annotator_v2.py` | `python keypoint_annotator_v2.py data/frames` |
| **Mask만** | `run_sam_gui.py` | `python run_sam_gui.py --frames-dir data/frames` |
| **Mask + Keypoint 둘 다** | `unified_annotator.py` | `./run_unified_annotator.sh data/frames data/annotations both` |
| **20개 라벨 → YOLOv8 학습** | Roboflow (웹) | https://roboflow.com/ |

### Keypoint 어노테이션 (최소 경로)

```bash
# 1. 환경 활성화
conda activate mammal_stable
cd /home/joon/dev/MAMMAL_mouse

# 2. Keypoint Annotator 실행
python keypoint_annotator_v2.py data/100-KO-male-56-20200615_cropped \
  --output data/keypoints_manual.json

# 3. 브라우저 접속
#    로컬: http://localhost:7861
#    원격: ssh -L 7861:localhost:7861 joon@bori → http://localhost:7861

# 4. MAMMAL 포맷 변환
python convert_keypoints_to_mammal.py \
  --input data/keypoints_manual.json \
  --output data/100-KO-male-56-20200615_cropped/keypoints2d_undist/result_view_0.pkl \
  --num-frames 20

# 5. Mesh fitting 실행
python fitter_articulation.py dataset=custom_cropped
```

### SAM 마스크 어노테이션 (최소 경로)

```bash
conda activate mammal_stable
cd /home/joon/dev/MAMMAL_mouse

python run_sam_gui.py \
  --frames-dir data/100-KO-male-56-20200615_frames \
  --port 7860

# 브라우저: http://localhost:7860
```

---

## 도구 비교

### 기능 비교표

| 기능 | Unified Annotator | Keypoint V2 | SAM (run_sam_gui) |
|------|-------------------|-------------|-------------------|
| **Mask Annotation** | O | X | O |
| **Keypoint Annotation** | O | O | X |
| **SAM2 지원** | O | X | O |
| **Zoom 지원** | X | O | X |
| **Visibility 제어** | O | O (3단계) | X |
| **통합 저장 (JSON)** | O | X | X |
| **MAMMAL 호환** | O | O | O |
| **의존성** | 높음 (SAM2) | 낮음 | 중간 (SAM2) |
| **메모리 사용** | 높음 | 낮음 | 중간 |

### Roboflow (웹 기반 라벨링)

- **장점**: 설치 불필요, 직관적 UI, YOLO 포맷 직접 내보내기, 무료 티어 충분
- **단점**: 인터넷 필요, 클라우드에 이미지 업로드
- **적합**: 20개 이미지로 YOLOv8 fine-tuning할 때

### Label Studio

- **장점**: 모던 UI, Python 친화적, 로컬 실행
- **단점**: 설치 필요
- **설치**: `pip install label-studio && label-studio start` → http://localhost:8080

### CVAT

- **장점**: 웹 기반 인터페이스, 팀 작업 지원, YOLO 포맷 내보내기
- **단점**: Docker 필요
- **설치**: `docker run -p 8080:8080 cvat/server`

### 선택 가이드

| 상황 | 권장 도구 | 이유 |
|------|-----------|------|
| 첫 사용자 | `keypoint_annotator_v2.py` | 가장 간단, 가벼움 |
| 전체 파이프라인 | `unified_annotator.py` | Mask + Keypoint 통합 |
| Zoom 필요 | `keypoint_annotator_v2.py` | 유일하게 Zoom 지원 |
| Mask 특화 | `run_sam_gui.py` | SAM 전용, 최적화 |
| 메모리 제약 | `keypoint_annotator_v2.py` | SAM 불필요 |
| YOLOv8 학습 데이터 | Roboflow | 웹에서 빠르게 완료 |

---

## 22 키포인트 정의

### 전체 매핑 (MAMMAL 표준)

```
Index  Name              Body Part    설명
-----  ----              ---------    ----
  0    nose              Head         코 끝 (snout tip)
  1    left_ear          Head         왼쪽 귀 base
  2    right_ear         Head         오른쪽 귀 base
  3    left_eye          Head         왼쪽 눈 중심
  4    right_eye         Head         오른쪽 눈 중심
  5    head_center       Head         양 귀 사이 중간점
  6    spine_1           Spine        목 시작 (neck)
  7    spine_2           Spine        상부 등
  8    spine_3           Spine        중상부 등
  9    spine_4           Spine        등 중앙
 10    spine_5           Spine        중하부 등
 11    spine_6           Spine        하부 등
 12    spine_7           Spine        꼬리 base 직전
 13    spine_8           Spine        꼬리 base 연결부
 14    left_front_paw    Limbs        왼쪽 앞발 관절 (elbow/wrist)
 15    right_front_paw   Limbs        오른쪽 앞발 관절
 16    left_rear_paw     Limbs        왼쪽 뒷발 관절 (knee/ankle)
 17    right_rear_paw    Limbs        오른쪽 뒷발 관절
 18    tail_base         Tail         꼬리 시작점 (= spine_8)
 19    tail_mid          Tail         꼬리 중간점
 20    tail_tip          Tail         꼬리 끝
 21    centroid          Body         몸통 중심 (body center)
```

### 부위별 그룹

**Head (0-5)**:
- `nose` (0): 코 끝
- `left_ear` (1), `right_ear` (2): 귀 base (마우스를 향해 봤을 때 좌/우)
- `left_eye` (3), `right_eye` (4): 눈 중심 (보이는 경우만)
- `head_center` (5): 양 귀 사이 중간

**Spine (6-13)**: 목부터 꼬리 base까지 8등분하여 균등 배치
- 팁: 등을 8개 구간으로 나눈다고 상상

**Limbs (14-17)**:
- 발끝(paw tip)이 아닌 **관절 중심**에 표시
- 보이지 않으면 `not_visible`로 처리

**Tail (18-20)**:
- `tail_base` (18)은 `spine_8` (13)과 동일 위치
- 꼬리의 자연스러운 곡선을 따라 배치

**Body (21)**: `centroid` = 몸통 기하학적 중심

### Core 7 Keypoints (간이 어노테이션용)

`keypoint_annotator_v2.py`와 `unified_annotator.py`에서 사용하는 7개 핵심 키포인트:

| Index | Name | 설명 |
|-------|------|------|
| 0 | `nose` | 코 끝 |
| 1 | `neck` | 목 base (= spine_1) |
| 2 | `spine_mid` | 척추 중간 (= spine_4) |
| 3 | `hip` | 골반 |
| 4 | `tail_base` | 꼬리 시작점 |
| 5 | `left_ear` | 왼쪽 귀 |
| 6 | `right_ear` | 오른쪽 귀 |

**최소 권장**: `nose`, `spine_mid`, `hip`, `tail_base` (4개)

### Visibility 레벨

| 값 | 의미 | 사용 시기 | 표시 방식 |
|----|------|-----------|-----------|
| **1.0** (visible) | 명확하게 보임 | 경계가 명확, 정확한 위치 확인 가능 | 채워진 원 |
| **0.5** (occluded) | 부분적으로 가려짐 | 대략적 위치만 추정 가능 | 빈 원 (hollow) |
| **0.0** (not_visible) | 전혀 보이지 않음 | 완전히 가려짐, 이미지 밖 | 표시 안됨 |

**MAMMAL Confidence 처리**:
- `confidence >= 0.25`: 최적화에 사용
- `confidence < 0.25`: 자동 무시
- `confidence = 0.0`: 미 어노테이션 (loss 기여 0)

### 키포인트 개수별 Fitting 품질

| 개수 | 예상 결과 |
|------|-----------|
| 1-2개 | 위치만 맞춤 (coarse alignment) |
| 3-4개 | 기본적인 body orientation |
| 5-7개 (core) | 전체 body pose 추정 가능 |
| 10개+ | 세밀한 fitting |

---

## 키포인트 어노테이션

### Keypoint Annotator V2 사용법

```bash
python keypoint_annotator_v2.py data/100-KO-male-56-20200615_cropped \
  --output data/keypoints_manual.json
```

**V2 주요 기능**:
- **Zoom**: 1.0x ~ 4.0x (권장 2.0x ~ 3.0x)
- **Point Size**: 1 ~ 8 픽셀 (기본 3)
- **Visibility**: 3단계 (visible / occluded / not_visible)
- "Mark Current as NOT VISIBLE" 버튼으로 빠르게 표시 가능

**권장 설정 조합**:

| Zoom | Point Size | 용도 |
|------|-----------|------|
| 1.0x-1.5x | 3-4 | 전체 구조 파악 |
| 2.0x-3.0x | 2-3 | 정확한 배치 (권장) |
| 3.5x-4.0x | 2 | 매우 작은 디테일 |

**작업 순서 (프레임 단위)**:
1. 프레임 로드 (슬라이더 또는 Prev/Next)
2. Zoom 및 Point Size 조정
3. 키포인트 선택 → Visibility 설정 → 이미지 클릭
4. 모든 키포인트 반복: nose → neck → spine_mid → hip → tail_base → ears
5. "Save Keypoints" 클릭
6. 다음 프레임

**효율적 작업 팁**:
- **한 키포인트씩 전 프레임 완료**: nose를 20 프레임 모두 → neck을 20 프레임 모두 (프레임별 전체보다 빠름)
- 비슷한 자세 프레임은 이전 위치 참고
- 첫 프레임에서 설정 확정 후, 이후 동일 설정 유지

**출력 형식** (`keypoints_manual.json`):
```json
{
  "frame_000000": {
    "nose": {"x": 125.3, "y": 89.2, "visibility": 1.0},
    "neck": {"x": 118.5, "y": 102.1, "visibility": 1.0},
    "spine_mid": {"x": 110.2, "y": 115.3, "visibility": 0.5},
    "hip": {"x": 95.1, "y": 128.4, "visibility": 1.0},
    "tail_base": {"x": 82.3, "y": 135.7, "visibility": 1.0},
    "left_ear": {"x": 130.0, "y": 85.0, "visibility": 0.0}
  }
}
```

- `x`, `y`: 원본 이미지 좌표 (Zoom 적용 전 좌표로 자동 변환)
- `visibility`: 0.0 / 0.5 / 1.0

**예상 소요 시간**:
- 프레임당 1.5-2분 (7 키포인트)
- 20 프레임: 30-40분
- 첫 프레임은 느림 (학습), 이후 빨라짐

---

## SAM 마스크 생성

### 실행 방법

**권장: `run_sam_gui.py`** (Hydra 충돌 없음, 어디서든 실행 가능)

```bash
conda activate mammal_stable
cd /home/joon/dev/MAMMAL_mouse

# 포그라운드 실행
python run_sam_gui.py \
  --frames-dir data/100-KO-male-56-20200615_frames \
  --port 7860

# 백그라운드 실행 (터미널 닫아도 유지)
nohup python run_sam_gui.py \
  --frames-dir data/100-KO-male-56-20200615_frames \
  --port 7860 \
  > sam_annotator.log 2>&1 &
```

**원격 접속 (SSH 터널)**:
```bash
# 로컬 PC 터미널
ssh -L 7860:localhost:7860 joon@bori
# 브라우저: http://localhost:7860
```

**참고: `conda run` + Hydra 충돌 문제**

`conda run -n mammal_stable python -m sam_annotator ...` 형태는 Hydra GlobalHydra 초기화 오류가 발생한다. `run_sam_gui.py`는 OmegaConf만 사용하여 이 문제를 우회한다. 다른 프로젝트에서도 `run_sam_gui.py`를 복사하여 사용할 수 있다.

### SAM Annotator 어노테이션 워크플로우

1. **프레임 로드**: 슬라이더로 선택 → "Load Frame" 클릭
2. **Foreground 포인트**: "Foreground" 선택 → 생쥐 위 3-5곳 클릭 (초록 점)
3. **Background 포인트**: "Background" 선택 → 배경 1-2곳 클릭 (빨간 점)
4. **마스크 생성**: "Generate Mask" 클릭 → 오른쪽에서 확인
5. **저장 또는 재시도**: 괜찮으면 "Save Annotation", 이상하면 "Clear" → 재시도
6. **다음 프레임**: 슬라이더 이동 → 반복

### 포인트 배치 전략

**Foreground (생쥐)**: 권장 3-5개, 최소 2개

```
배치 위치:
1. 머리/코 영역
2. 등/허리 중앙
3. 엉덩이/꼬리 시작 부분
4. (선택) 다리나 꼬리 끝
```

**Background (배경)**: 권장 1-2개, 없어도 가능

```
배치 위치:
1. 바닥 (아래쪽) 명확한 영역
2. (선택) 벽 (위쪽이나 옆)
```

**프레임마다 포인트 개수가 달라도 무관** - SAM은 각 프레임을 독립적으로 처리.

**어려운 상황별 대응**:

| 상황 | Foreground | Background |
|------|-----------|-----------|
| 일반적 | 3-4개 | 1개 |
| 벽에 가까이 | 5-6개 (골고루) | 2-3개 (벽, 바닥) |
| 그림자 많음 | 4-5개 | 2-3개 (그림자 영역 포함) |
| 생쥐가 작음 | 3-4개 (핵심만) | 1-2개 |

### 포인트 배치 원칙

**Good Practice**:
- 확실한 영역에 클릭 (명확한 털 / 명확한 바닥)
- 경계선에서 5-10픽셀 이상 안쪽
- 생쥐 전체에 골고루 분산 (머리-몸통-꼬리)

**Bad Practice**:
- 경계선 근처 클릭 (SAM 혼란)
- 그림자인지 생쥐인지 애매한 곳
- 한 곳에 집중 (머리에만 5개 등)

### 마스크 품질 기준

| 판정 | 기준 | 대응 |
|------|------|------|
| 바로 저장 | 생쥐 전체 포함, 배경 거의 없음, 경계 자연스러움 | Save |
| 저장 OK | 90%+ 포함, 배경 소량 포함, 꼬리 끝 약간 잘림 | Save |
| 재작업 | 몸통 일부 빠짐, 배경 큰 영역 포함, 여러 조각 | Clear → 재시도 |

### 저장 결과

```
data/100-KO-male-56-20200615_frames/annotations/
├── frame_000000_annotation.json   # SAM 포인트 + 메타데이터
├── frame_000000_mask.png          # Binary mask (0=배경, 255=전경)
├── frame_000001_annotation.json
└── frame_000001_mask.png
```

### 마스크 어노테이션 후 크롭 프레임 생성

```bash
python process_annotated_frames.py \
  data/100-KO-male-56-20200615_frames/annotations \
  --output-dir data/100-KO-male-56-20200615_cropped \
  --padding 50
```

---

## Unified Annotator

Mask + Keypoint를 하나의 인터페이스에서 처리하는 통합 도구.

### 설치

```bash
# SAM2 (mask mode 필요 시)
cd ~/dev
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
  -O checkpoints/sam2_hiera_large.pt

# 의존성
pip install gradio opencv-python numpy
```

### 실행

```bash
python unified_annotator.py \
  -i data/frames \
  -o data/annotations \
  --mode both \
  --sam-checkpoint ~/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt

# 또는 쉘 스크립트
./run_unified_annotator.sh data/frames data/annotations both
```

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--input, -i` | 입력 프레임 디렉토리 | (필수) |
| `--output, -o` | 출력 어노테이션 디렉토리 | (필수) |
| `--mode, -m` | `mask`, `keypoint`, `both` | `both` |
| `--sam-checkpoint` | SAM2 체크포인트 경로 | None |
| `--port, -p` | 서버 포트 | 7860 |

### UI 구조

```
+---------------------------+---------------------------+
|                           |    Frame Navigation       |
|                           |  [Slider] [Load] [Save]   |
|     Frame Display         |                           |
|   (클릭하여 어노테이션)     |  +-- Mask Mode Tab --+    |
|                           |  | Point Type (FG/BG) |    |
|                           |  | Generate / Clear    |    |
|                           |  +--------------------+    |
|    Status Message         |  +-- Keypoint Mode --+    |
|                           |  | Keypoint 선택      |    |
|                           |  | Visibility         |    |
|                           |  | Mark/Remove        |    |
+---------------------------+---------------------------+
```

### Mask Mode 워크플로우

1. **Mask Mode** 탭 선택
2. **Foreground** 선택 → 생쥐 위 클릭 (초록 점)
3. **Background** 선택 → 배경 클릭 (빨간 점)
4. **Generate Mask** → 마스크 확인
5. 불만족 시: 포인트 추가/수정 → 재생성
6. **Save Annotation**

### Keypoint Mode 워크플로우

1. **Keypoint Mode** 탭 선택
2. 드롭다운에서 키포인트 선택 (nose, neck, spine_mid, ...)
3. Visibility 설정 (visible / occluded / not_visible)
4. 이미지 클릭하여 배치
5. 모든 키포인트 반복
6. **Save Annotation**

### 출력 형식

```json
{
  "frame": "/path/to/frame_0000.png",
  "frame_idx": 0,
  "mask": {
    "points": [[100, 200], [150, 250]],
    "labels": [1, 0],
    "has_mask": true,
    "confidence": 0.95,
    "mask_area_pct": 25.5
  },
  "keypoints": {
    "nose": {"x": 120.0, "y": 80.0, "visibility": 1.0},
    "neck": {"x": 140.0, "y": 100.0, "visibility": 0.5}
  }
}
```

### 커스텀 키포인트 추가

```python
config = AnnotationConfig(
    input_dir="data/frames",
    output_dir="data/annotations",
    mode=AnnotationMode.BOTH,
    keypoint_names=[
        'nose', 'neck', 'spine_mid', 'hip', 'tail_base',
        'left_ear', 'right_ear',
        'left_paw', 'right_paw'  # 추가 키포인트
    ]
)
```

---

## 워크플로우

### 전체 파이프라인

```
프레임 선택 → 어노테이션 → 검증 → 포맷 변환 → 내보내기/학습
```

### 1. 프레임 선택

```python
import random
from pathlib import Path
import shutil

# 균등 분포로 샘플링
imgs = sorted(Path('data/fauna_mouse').rglob('*_rgb.png'))
n_samples = 20
indices = [int(i * len(imgs) / n_samples) for i in range(n_samples)]
sampled = [imgs[i] for i in indices]

out_dir = Path('data/manual_labeling/images')
out_dir.mkdir(parents=True, exist_ok=True)
for i, img in enumerate(sampled):
    shutil.copy(img, out_dir / f'sample_{i:03d}.png')
```

**선택 기준**: 다양한 자세 (서기, 걷기, 회전), 다양한 각도, 다양한 조명, 모든 신체 부위 가시성

### 2. 어노테이션

도구 선택에 따라 위 섹션 참조.

### 3. Confidence 기반 필터링

MAMMAL fitter는 자동으로 confidence 필터링 수행:

```python
# fitter_articulation.py:214
diff = (J2d_projected - target_2d) * confidence
# confidence=0 -> diff=0 -> loss 기여 0
```

- `confidence >= 0.25`: 최적화에 사용
- `confidence < 0.25`: 자동 무시
- Missing 키포인트는 `confidence=0.0`으로 설정하면 자동으로 필터링

### 4. 검증

```python
import pickle
with open('result_view_0.pkl', 'rb') as f:
    kpts = pickle.load(f)

print(f"Shape: {kpts.shape}")  # (num_frames, 22, 3)

for i, (x, y, conf) in enumerate(kpts[0]):
    if conf > 0:
        print(f"Keypoint {i}: ({x:.1f}, {y:.1f}) conf={conf:.2f}")
```

시각적 검증:
```bash
python preprocessing_utils/visualize_yolo_labels.py \
  --images data/manual_labeling/images \
  --labels data/manual_labeling/labels \
  --output data/manual_labeling/viz \
  --max_images 5
```

### 5. 내보내기

COCO/DLC 포맷으로도 내보내기 가능:
```bash
python export_to_coco.py --input data/annotations
python export_to_dlc.py --input data/annotations
```

---

## Manual Labeling → YOLOv8 파이프라인

20개 수동 라벨 → YOLOv8-Pose fine-tuning으로 키포인트 탐지 정확도를 대폭 향상.

### 예상 개선

| 항목 | Before (Geometric) | After (Fine-tuned) |
|------|--------------------|--------------------|
| mAP | ~0 | 0.6-0.8 |
| Paw detection | 0% | 70-80% |
| Confidence | 0.4-0.6 | 0.85+ |

### Roboflow 단계별 가이드

**1. 계정 및 프로젝트 생성** (5분)
1. https://roboflow.com/ 가입
2. "Create New Project" → **Keypoint Detection** 선택
3. Project Name: `MAMMAL_Mouse_Keypoints`

**2. 22 키포인트 정의** (순서 중요!)

위의 [22 키포인트 정의](#22-키포인트-정의) 섹션의 정확한 순서대로 등록.

**3. 이미지 업로드 및 라벨링** (2-3시간)
- `data/manual_labeling/images/`에서 20개 이미지 업로드
- 이미지당 5-10분, 총 ~2.5시간
- 30분마다 5분 휴식 권장

**4. 라벨링 가이드**

**Head (0-5)**:
- `nose` (0): 코 끝 (snout tip)
- `left_ear` (1): 왼쪽 귀 base
- `right_ear` (2): 오른쪽 귀 base
- `left_eye` (3): 왼쪽 눈 중심 (보이는 경우)
- `right_eye` (4): 오른쪽 눈 중심 (보이는 경우)
- `head_center` (5): 양 귀 사이 중간

**Spine (6-13)**: 목부터 꼬리 base까지 8등분하여 균등 배치

**Paws (14-17)**: 발끝이 아닌 관절 중심 (elbow/wrist, knee/ankle). 보이지 않으면 "not visible" 처리.

**Tail (18-20)**: tail_base (18)은 spine_8 (13)과 동일 위치. 꼬리 자연스러운 곡선 따라 배치.

**Visibility flag**: visible=2, occluded=1, not_visible=0

**5. 내보내기**
- Generate → Export → **YOLO v8** → ZIP 다운로드

```bash
unzip roboflow.zip -d ~/dev/MAMMAL_mouse/data/manual_labeling/roboflow_export
cp -r data/manual_labeling/roboflow_export/train/labels/* data/manual_labeling/labels/
```

### 데이터셋 병합 및 학습

```bash
# 수동 라벨(20) + geometric 라벨(50) 병합
python preprocessing_utils/merge_datasets.py \
  --manual data/manual_labeling \
  --geometric data/yolo_mouse_pose \
  --output data/yolo_mouse_pose_enhanced \
  --train_split 0.8

# YOLOv8 fine-tuning
python scripts/train_yolo_pose.py \
  --data data/yolo_mouse_pose_enhanced/data.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 256 \
  --weights yolov8n-pose.pt \
  --name mammal_mouse_finetuned
```

### YOLO 라벨 포맷

각 `.txt` 파일:
```
<class_id> <x_center> <y_center> <width> <height> <kpt1_x> <kpt1_y> <kpt1_v> ... <kpt22_x> <kpt22_y> <kpt22_v>
```

- 모든 좌표 [0, 1]로 정규화 (x = pixel_x / image_width)
- 총 값 수: 1 + 4 + 66 = 71 (class + bbox + 22x3 keypoints)

### 프로덕션 적용

```bash
# 최적 모델 복사
cp runs/pose/mammal_mouse_finetuned/weights/best.pt models/yolo_mouse_pose_finetuned.pt

# fit_monocular.py에서 사용
python fit_monocular.py \
  --input_dir data/images \
  --detector yolo \
  --yolo_weights models/yolo_mouse_pose_finetuned.pt
```

### 시간 투자 요약

| 작업 | 소요 시간 | 누적 |
|------|-----------|------|
| 이미지 선택 | 10분 | 10분 |
| 라벨링 (20개) | 2.5시간 | 2시간 40분 |
| 품질 체크 | 15분 | 2시간 55분 |
| 학습 | 30분 | 3시간 25분 |
| 검증 | 15분 | 3시간 40분 |
| **합계** | **~3.5-4시간** | |

---

## MAMMAL 포맷 변환 및 Mesh Fitting

### JSON → MAMMAL PKL 변환

```bash
python convert_keypoints_to_mammal.py \
  --input data/annotations/keypoints.json \
  --output data/100-KO-male-56-20200615_cropped/keypoints2d_undist/result_view_0.pkl \
  --num-frames 20 \
  --visualize 0
```

**변환 과정**:
1. JSON 어노테이션 로드
2. MAMMAL 22-키포인트 포맷으로 매핑
3. Visibility → Confidence 변환
4. 미 어노테이션 키포인트는 confidence=0.0
5. NumPy 배열 `(num_frames, 22, 3)` 형태로 pickle 저장

**Core 7 → MAMMAL 22 키포인트 매핑**:
```python
{
    'nose': 0,
    'neck': 1,
    'spine_mid': 2,
    'hip': 3,
    'tail_base': 4,
    'left_ear': 5,
    'right_ear': 6,
    # Indices 7-21: Reserved (confidence=0.0)
}
```

### MAMMAL 3단계 최적화

| 단계 | 최적화 대상 | Keypoint Weight | Mask Weight | 목적 |
|------|------------|-----------------|-------------|------|
| **Step 0** | rotation, translation, scale | Normal | 0 | Coarse alignment |
| **Step 1** | + thetas, bone_lengths | Normal | 0 | Pose fitting |
| **Step 2** | + chest_deformer | Foot x10 | 3000 | Silhouette refinement |

**Loss 구성**:
```python
total_loss =
  + keypoint_2d_loss * 0.2    # 주요 signal
  + theta_regularization * 3
  + bone_length * 0.5
  + scale * 0.5
  + mask_loss * 3000          # Step 2에서만
```

### 데이터셋 디렉토리 구조

```
data/100-KO-male-56-20200615_cropped/
├── frame_000000_cropped.png
├── frame_000000_mask.png
├── keypoints2d_undist/
│   └── result_view_0.pkl       # 변환된 키포인트
├── simpleclick_undist/
│   └── 0.mp4                   # Mask 비디오 (선택)
└── videos_undist/
    └── 0.mp4                   # RGB 비디오 (선택)
```

### 데이터셋 Config

```yaml
# conf/dataset/custom_cropped.yaml
# @package _global_

data:
  data_dir: data/100-KO-male-56-20200615_cropped/
  views_to_use: [0]

fitter:
  start_frame: 0
  end_frame: 19
  interval: 1
  render_cameras: [0]
  with_render: true
  keypoint_num: 22
```

### Fitting 실행

```bash
# 기본 fitting
python fitter_articulation.py dataset=custom_cropped

# Mask 없이 키포인트만으로 fitting
python fitter_articulation.py dataset=custom_cropped fitter.term_weights.mask=0

# Mask 활성화 (Step 2 refinement)
python fitter_articulation.py dataset=custom_cropped fitter.term_weights.mask=3000
```

### Unified Annotator → MAMMAL 변환

```bash
# 1. Unified annotator 출력에서 키포인트 추출
python extract_unified_keypoints.py \
  -i data/annotations \
  -o keypoints.json

# 2. MAMMAL 포맷 변환
python convert_keypoints_to_mammal.py \
  -i keypoints.json \
  -o data/.../result_view_0.pkl \
  -n 20
```

**예상 개선치 (Keypoint fitting 추가)**:
- IoU: 46% → 60-75%
- Pose 정확도: 크게 향상
- 수렴 속도: 더 빠름

---

## 팁 & Best Practices

### 키포인트 어노테이션

1. **어노테이션 순서**: spine부터 (nose → neck → spine_mid → hip → tail_base), 그 다음 ears
2. **일관성이 정확도보다 중요**: 프레임 간 동일한 기준으로 배치
3. **Visibility를 적극 활용**: 불확실하면 occluded (0.5), 정말 안 보이면 not_visible (0.0)
4. **브라우저 Zoom 활용**: Ctrl/Cmd + Plus/Minus로 전체 UI 확대 가능
5. **최소 5개 키포인트**: nose, neck, spine_mid, hip, tail_base

### 마스크 어노테이션

1. **Foreground부터 시작**: 생쥐 중심부에 2-3개 클릭
2. **Background는 보조**: 마스크에 불필요한 영역 포함 시 추가
3. **반복 정제**: Generate → 확인 → 포인트 추가 → 재생성
4. **경계선 멀리**: 생쥐/배경 경계에서 멀리 클릭
5. **완벽할 필요 없음**: 90% 정도면 충분 (mesh fitting에서 추가 정제)

### 라벨링 공통

1. **저장 습관**: 프레임마다 즉시 저장
2. **진행 상황 확인**: Summary 패널로 완료된 프레임 확인
3. **프레임당 목표 시간**: 마스크 30초-1분, 키포인트 1.5-2분
4. **Left/Right 구분 주의**: 마우스를 향해 봤을 때의 좌/우 기준
5. **Spine 균등 배치**: 등을 8등분하여 자연스러운 곡선

### 이미지 선택 기준

- 다양한 자세 (서기, 걷기, 회전, 웅크림)
- 다양한 카메라 각도
- 다양한 조명 조건
- 모든 신체 부위 가시성

### Quality Checklist (이미지당)

- [ ] 모든 키포인트 배치 (또는 not_visible로 표시)
- [ ] Left/Right 뒤바뀌지 않음
- [ ] Spine이 자연스러운 곡선
- [ ] Paw가 관절 중심 (발끝 아님)
- [ ] Tail이 자연스러운 경로

### 방법별 비교

| 방법 | 정확도 | 소요 시간 | Spine | Limbs |
|------|--------|-----------|-------|-------|
| Heuristic | 낮음 | 0분 | ~50% | X |
| Manual (7 keypoints) | 높음 | 30분 | ~95% | O |
| Manual (22 keypoints) | 매우 높음 | 2.5시간 | ~98% | O |
| DeepLabCut | 매우 높음 | 2시간 셋업 | ~98% | O |

---

## 문제 해결

### SAM 관련

**"SAM not available!"**
```bash
# SAM2 설치 확인
ls ~/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt

# 없으면 설치
cd ~/dev
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
  -O checkpoints/sam2_hiera_large.pt
```

**포트 충돌**
```bash
lsof -ti :7860 | xargs kill -9
# 또는 다른 포트 사용
python run_sam_gui.py --frames-dir ... --port 8080
```

**GPU 메모리 부족**: 더 작은 SAM 모델 사용 또는 CPU 모드

### Keypoint 관련

**키포인트가 보이지 않음**: Point Size 증가 (3 → 5), Zoom 증가

**잘못된 위치 클릭**: 같은 키포인트 재선택 후 올바른 위치 클릭 (덮어쓰기)

**이미지가 화면 초과**: Zoom 낮춤 (2.0x → 1.5x)

**Visibility 잘못 설정**: 같은 키포인트를 올바른 Visibility로 다시 클릭

### 포맷 변환 관련

**"No keypoints found"**: JSON 프레임 이름 확인 (예: `frame_000000`)

**"Keypoint index out of range"**: `--num-frames` 값이 실제 프레임 수와 일치하는지 확인

**Fitting 결과 나쁨**: 최소 5개 core 키포인트 어노테이션 확인, spine landmark 정확도 점검

### Roboflow 관련

**업로드 실패**: 5개씩 나눠서 업로드, PNG 포맷 확인, 파일 크기 <10MB

**키포인트 순서 오류**: 프로젝트 설정에서 MAMMAL 정의와 동일 순서 확인

**내보내기 포맷 오류**: YOLO v8 선택 (v5/v7 아님), 각 줄 = 1 + 4 + 66 값

### 서버 관리

```bash
# 상태 확인
curl -s http://localhost:7860 > /dev/null && echo "Running" || echo "Not running"

# 프로세스 종료
lsof -ti :7860 | xargs kill -9

# SAM 관련 모든 프로세스 종료
pkill -f "run_sam_gui"
```

---

## 주요 스크립트 참조

| 스크립트 | 용도 |
|----------|------|
| `keypoint_annotator_v2.py` | Keypoint 어노테이션 (zoom, visibility) |
| `unified_annotator.py` | Mask + Keypoint 통합 어노테이션 |
| `run_sam_gui.py` | SAM 마스크 어노테이션 (권장 런처) |
| `convert_keypoints_to_mammal.py` | JSON → MAMMAL PKL 변환 |
| `extract_unified_keypoints.py` | Unified 출력 → Keypoint JSON 추출 |
| `extract_sam_keypoints.py` | SAM 클릭 → Keypoint (비추천) |
| `fitter_articulation.py` | MAMMAL Mesh Fitting |
| `preprocessing_utils/visualize_yolo_labels.py` | YOLO 라벨 시각화 |
| `preprocessing_utils/merge_datasets.py` | 데이터셋 병합 |
| `scripts/train_yolo_pose.py` | YOLOv8 Pose 학습 |

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [KEYPOINTS](../reference/KEYPOINTS.md) | 22 키포인트 정의 및 Skeleton |
| [PREPROCESSING_GUIDE](PREPROCESSING_GUIDE.md) | SAM 마스크 전처리 파이프라인 |
| [DATASET](../reference/DATASET.md) | 데이터셋 스펙 (6cam, 18K frames) |

---

*Merged from 14 source files: UNIFIED_ANNOTATOR_GUIDE.md, START_ANNOTATION.md, ANNOTATOR_COMPARISON.md, KEYPOINT_ANNOTATION_GUIDE.md, KEYPOINT_ANNOTATOR_V2_GUIDE.md, KEYPOINT_QUICK_START.md, KEYPOINT_WORKFLOW.md, MANUAL_LABELING_GUIDE.md, ROBOFLOW_LABELING_GUIDE.md, RUN_SAM.md, SAM_LAUNCHER_EXPLAINED.md, ANNOTATION_TIPS.md, ANNOTATION_TOOLS_README.md, QUICK_START_LABELING.md*

*Created: 2026-02-06*
