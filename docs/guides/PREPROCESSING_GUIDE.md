# Preprocessing Guide

MAMMAL Mouse 프로젝트의 데이터 전처리 통합 가이드입니다.
비디오 프레임 추출, SAM 마스크 생성, 크로핑, 피팅 준비까지의 전체 파이프라인을 다룹니다.

---

## Quick Start (5-Step Pipeline)

```bash
cd /home/joon/dev/MAMMAL_mouse
conda activate mammal_stable

# Step 1: 비디오에서 프레임 추출
python extract_video_frames.py \
    /home/joon/dev/data/100-KO-male-56-20200615.avi \
    --output-dir data/my_mouse/frames \
    --num-frames 20

# Step 2: (Optional) 프레임 미리보기
python visualize_extracted_frames.py data/my_mouse/frames

# Step 3: SAM annotation (Web UI)
conda activate mammal_stable
python run_sam_gui.py \
    --frames-dir data/my_mouse/frames \
    --port 7860

# Step 4: Annotation 처리 및 크로핑
python process_annotated_frames.py \
    data/my_mouse/frames/annotations \
    --output-dir data/my_mouse/cropped \
    --padding 50

# Step 5: Mesh fitting 실행
python fit_cropped_frames.py \
    data/my_mouse/cropped \
    --output-dir results/my_mouse_fitting
```

---

## 비디오 처리

### 프레임 추출

```bash
VIDEO_PATH="/home/joon/dev/data/100-KO-male-56-20200615.avi"
OUTPUT_DIR="data/100-KO-male-56-20200615_frames"

# 20개 균등 간격 프레임 추출
python extract_video_frames.py \
    "$VIDEO_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames 20
```

**추출 옵션**:

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--num-frames N` | N개 균등 간격 프레임 | `--num-frames 20` |
| `--fps-sample 1.0` | 초당 1 프레임 샘플링 | `--fps-sample 2.0` |
| `--frame-indices 0 100 500` | 특정 프레임 인덱스 | `--frame-indices 0 100 500` |
| `--all` | 모든 프레임 추출 | `--all` |

**출력**:
- `frame_XXXXXX.png`: 추출된 프레임
- `extraction_metadata.json`: timestamp, frame index 메타데이터

### 디렉토리 구조

전체 파이프라인 완료 후 디렉토리 구조:

```
data/my_mouse/
├── frames/                          # Step 1: 추출된 프레임
│   ├── frame_000000.png
│   ├── ...
│   ├── extraction_metadata.json
│   ├── frames_preview.png           # Step 2: 미리보기
│   └── annotations/                 # Step 3: SAM annotation
│       ├── frame_000000_annotation.json
│       ├── frame_000000_mask.png
│       └── ...
└── cropped/                         # Step 4: 크롭된 프레임
    ├── frame_000000_cropped.png
    ├── frame_000000_mask.png
    ├── frame_000000_crop_info.json
    ├── processing_summary.json
    └── processing_visualization.png
```

### 비디오 문제 해결

```bash
# 비디오 코덱 확인
ffmpeg -i your_video.avi

# 코덱 변환 (필요 시)
ffmpeg -i input.avi -c:v libx264 output.mp4

# 해상도 업스케일 (저해상도 비디오)
ffmpeg -i input.avi -vf scale=1280:960 output.avi
```

---

## SAM 마스크 생성

### SAM 설치 및 체크포인트

```bash
# SAM 설치
pip install segment-anything

# 체크포인트 다운로드
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/

# SAM 2 (대안)
cd ~/dev
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
cd checkpoints && ./download_ckpts.sh
```

**모델 종류**:
- `vit_h` (ViT-Huge): 최고 품질, 가장 느림 (현재 사용 중)
- `vit_l` (ViT-Large): 중간 품질, 중간 속도
- `vit_b` (ViT-Base): 빠른 속도, 낮은 품질

### Web UI를 이용한 Annotation

```bash
# conda 환경 먼저 활성화 (conda run은 Hydra와 충돌)
conda activate mammal_stable

python run_sam_gui.py \
    --frames-dir data/my_mouse/frames \
    --port 7860
```

**접속 방법**:
- 로컬: `http://localhost:7860`
- 원격 (SSH tunnel): `ssh -L 7860:localhost:7860 joon@server` 후 `http://localhost:7860`

**Annotation 워크플로우**:
1. **Load Frame**: 슬라이더로 프레임 선택 -> "Load Frame" 클릭
2. **Foreground 포인트**: "Foreground" 선택 -> 마우스 body 위에 3-5개 클릭 (초록 점)
3. **Background 포인트**: "Background" 선택 -> 배경 위에 1-2개 클릭 (빨간 점)
4. **Generate Mask**: "Generate Mask" 클릭 -> 마스크 확인
5. **Save**: 만족스러우면 "Save Annotation" 클릭
6. 다음 프레임으로 이동하여 반복

**출력** (`annotations/` 하위 디렉토리):
- `frame_XXXXXX_annotation.json`: 포인트 좌표 및 라벨
- `frame_XXXXXX_mask.png`: Binary segmentation mask

### 자동 마스크 (Automatic Mask Generation)

SAM이 자동으로 이미지의 모든 객체를 감지합니다. 수동 입력 불필요하지만 후처리가 필요합니다.

```python
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
sam.to(device="cuda")

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100,
)

frame_rgb = cv2.cvtColor(cv2.imread("frame.png"), cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(frame_rgb)
```

**마스크 결과 구조**:
- `segmentation`: (H, W) binary mask
- `area`: 마스크 면적 (픽셀 수)
- `bbox`: [x, y, w, h] 바운딩 박스
- `predicted_iou`: 예측 IoU
- `stability_score`: 안정성 점수

### 마스크 선택 전략

마우스 마스크를 자동으로 선택하는 방법:

```python
# 전략 1: 크기 필터링 (마우스 = 이미지의 5-15%)
size_filtered = [m for m in masks if 0.05 < m['area']/total_area < 0.20]
mouse_mask = max(size_filtered, key=lambda x: x['predicted_iou'])['segmentation']

# 전략 2: 가장 큰 마스크 (아레나가 아닐 때)
largest_mask = max(masks, key=lambda x: x['area'])['segmentation']

# 전략 3: IoU 기반
best_mask = max(masks, key=lambda x: x['predicted_iou'])['segmentation']
```

### 포인트 프롬프트 (Point Prompt)

마우스 위치를 알 때 빠르게 마스크를 생성하는 방법입니다 (프레임당 1-2초).

```python
from segment_anything import SamPredictor

predictor = SamPredictor(sam)
predictor.set_image(frame_rgb)

# 마우스 중심 좌표
mouse_point = np.array([[W // 2, H // 2]])
point_label = np.array([1])  # 1 = foreground

masks, scores, logits = predictor.predict(
    point_coords=mouse_point,
    point_labels=point_label,
    multimask_output=True,
)
best_mask = masks[np.argmax(scores)]
```

### Negative 프롬프트 (배경 제외)

아레나 전체가 선택되는 문제를 해결합니다:

```python
# Positive (마우스) + Negative (아레나 배경)
input_points = np.array([
    [W // 2, H // 2],        # 마우스 중심 (foreground)
    [W // 2, H // 2 - 100],  # 위쪽 배경 (background)
    [W // 2, H // 2 + 100],  # 아래쪽 배경 (background)
    [W // 2 - 150, H // 2],  # 왼쪽 배경 (background)
    [W // 2 + 150, H // 2],  # 오른쪽 배경 (background)
])
input_labels = np.array([1, 0, 0, 0, 0])

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)
```

### 방법별 비교

| 방법 | 속도 | 정확도 | 자동화 | 추천 상황 |
|------|------|--------|--------|----------|
| **Automatic** | 느림 (5-10s) | 높음 | 완전 자동 | 배치 처리, 탐색 |
| **Point Prompt** | 빠름 (1-2s) | 높음 | 반자동 | 실시간, 인터랙티브 |
| **Negative Prompt** | 보통 | 최고 | 반자동 | 정밀 세그멘테이션 |

---

## 프레임 크로핑 & 전처리

Annotation 완료 후 마스크 기반으로 프레임을 크로핑합니다:

```bash
python process_annotated_frames.py \
    data/my_mouse/frames/annotations \
    --output-dir data/my_mouse/cropped \
    --padding 50
```

**옵션**:
- `--padding N`: 감지 영역 주변 패딩 (기본: 50px)
- `--no-visualize`: 시각화 건너뛰기

**출력**:
- `frame_XXXXXX_cropped.png`: 크롭된 프레임
- `frame_XXXXXX_mask.png`: 크롭된 마스크
- `frame_XXXXXX_crop_info.json`: 크롭 메타데이터
- `processing_summary.json`: 처리 통계
- `processing_visualization.png`: before/after 비교

**Crop Info JSON 형식**:
```json
{
    "original_shape": [480, 640],
    "bbox": [365, 251, 217, 196],
    "crop_coords": [365, 251, 582, 447],
    "cropped_shape": [196, 217],
    "mask_area": 2929,
    "frame_idx": 6
}
```

---

## 파이프라인 아키텍처 설계

전체 전처리 파이프라인은 4개의 독립 모듈로 구성됩니다:

### Module 1: SAM Inference (`sam_inference.py`)
- SAM 모델 로딩 및 캐싱
- 배치 처리로 효율성 향상
- GPU 메모리 관리

### Module 2: Mask Processing (`mask_processing.py`)
- 아레나 감지 및 제거
- 마우스 마스크 추출 (크기/형상/위치 필터링)
- Temporal consistency 필터링
- Noise 감소 (morphological operations)

**마우스 마스크 추출 로직**:
```python
def extract_mouse_mask(sam_masks, frame_shape):
    # Stage 1: 크기 필터 (마우스 = 5-20%)
    size_filtered = [m for m in sam_masks
                     if 0.05 < mask_area(m)/total_area < 0.20]
    # Stage 2: 형상 분석 (아레나 = 높은 원형도)
    shape_filtered = [m for m in size_filtered
                      if circularity(m) < 0.8]
    # Stage 3: 위치 필터 (아레나 내부)
    position_filtered = [m for m in shape_filtered
                        if is_inside_arena(m)]
    # Stage 4: 최적 후보 선택
    return max(position_filtered, key=mask_area)
```

### Module 3: Keypoint Estimation (`keypoint_estimation.py`)
- Contour 추출 및 PCA orientation
- MAMMAL 22-point keypoint 생성
- Confidence score 할당
- Temporal smoothing (moving average)

**MAMMAL 22 Keypoint Layout**:
```
Head (0-5): nose, left ear, right ear, left eye, right eye, head center
Spine (6-13): 8 points along backbone (neck to tail base)
Limbs (14-17): LF paw, RF paw, LR paw, RR paw
Tail (18-20): tail base, mid, tip
Body (21): centroid
```

### Module 4: Visualization (`visualization.py`)
- Side-by-side 비교
- Quality metrics 오버레이
- 디버그 프레임 생성

### 출력 형식

**Keypoints 파일** (`result_view_0.pkl`):
```python
# Shape: (N_frames, 22, 3) - [x, y, confidence]
keypoints = np.array([
    [[x0, y0, conf0], [x1, y1, conf1], ..., [x21, y21, conf21]],  # Frame 0
    ...
])
```

**Quality Report** (`quality_report.json`):
```json
{
    "processed_frames": 27000,
    "mask_quality": {"detection_rate": 0.97, "mean_iou": 0.88},
    "keypoint_quality": {"on_body_ratio": 0.94, "mean_confidence": 0.68}
}
```

---

## Troubleshooting & 성능 팁

### SAM Annotator가 시작되지 않음

```bash
# 포트 사용 중 확인
lsof -i :7860
kill -9 <PID>

# 다른 포트 사용
bash run_sam_annotator.sh 8080
```

### Web UI 원격 접속 불가

```bash
# SSH tunnel 설정
ssh -L 7860:localhost:7860 joon@server
# 브라우저에서 http://localhost:7860
```

### CUDA Out of Memory (SAM)

```bash
# 작은 모델 사용
# run_sam_annotator.sh에서 변경:
# sam2.1_hiera_large -> sam2.1_hiera_small
```

### SAM이 아레나 전체를 선택하는 문제

**해결 1**: Negative prompt 사용 (배경 포인트 추가)
**해결 2**: Color 기반 pre-filtering
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
dark_mask = gray < 100  # 어두운 영역 (마우스)
```

### 마스크 반전 문제
```python
mask = 255 - mask  # 반전
```

### 마스크 조각 문제 (Morphological fix)
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

### 빈 크롭 결과
- `--padding` 값 증가
- SAM annotator에서 마스크 품질 재확인
- 문제 프레임 재annotation

### 성능 최적화 팁

| 팁 | 효과 |
|-----|------|
| `sam2.1_hiera_small` 사용 | ~2x 속도 향상 |
| `--num-frames 10`으로 테스트 | 초기 검증 빠르게 |
| 매 2-3번째 프레임만 annotation | 작업량 감소 |
| 배치 처리 (batch_size=4) | ~1.5-2x 속도 향상 |
| 해상도 축소 (target_size=1024) | ~1.5x 속도 향상 |
| Chunked processing (1000 프레임씩) | OOM 방지 |

### 배치 비디오 처리

```bash
for video in /path/to/videos/*.avi; do
    basename=$(basename "$video" .avi)
    python extract_video_frames.py "$video" \
        --output-dir "data/${basename}/frames" \
        --num-frames 20
done
```

### Best Practices

**프레임 선택**:
- 10-20개로 시작하여 테스트
- 다양한 포즈/위치의 프레임 선택
- `--num-frames`로 시간적 커버리지 확보

**SAM Annotation**:
- Foreground 3-5개: 마우스 body (머리, 등, 꼬리)
- Background 1-2개: 아레나 바닥/벽
- 가장자리 피하고 명확한 영역에 포인트 배치
- `processing_visualization.png`으로 결과 확인

**크로핑**:
- Padding 50-100px 권장
- 모든 프레임에 동일한 padding 사용
- 크롭 결과 시각화로 품질 검증

---

## 관련 스크립트

| 스크립트 | 용도 |
|---------|------|
| `extract_video_frames.py` | 비디오에서 프레임 추출 |
| `visualize_extracted_frames.py` | 추출 프레임 미리보기 |
| `run_sam_gui.py` | SAM annotation Web UI |
| `run_sam_annotator.sh` | SAM annotator 실행 스크립트 |
| `process_annotated_frames.py` | Annotation 크로핑 |
| `sam_point_prompt.py` | 포인트 프롬프트 테스트 |
| `sam_mouse_only.py` | Negative 프롬프트 테스트 |
| `preprocessing_utils/sam_inference.py` | SAM 래퍼 클래스 |
| `preprocessing_utils/keypoint_estimation.py` | Geometric keypoint 추정 |

---

*Merged from: VIDEO_PROCESSING_GUIDE.md, QUICK_START_VIDEO.md, SAM_MASK_ACQUISITION_MANUAL.md, sam_preprocessing_plan.md*
*Created: 2026-02-06*
