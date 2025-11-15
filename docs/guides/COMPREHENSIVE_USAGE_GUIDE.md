# MAMMAL Mouse - Comprehensive Usage Guide

**최종 업데이트**: 2025-11-15
**프로젝트 버전**: v2.0 (ML Keypoint Detection + Monocular Fitting)

---

## 📖 목차

1. [개요](#개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [환경 설정](#환경-설정)
4. [사용 시나리오별 가이드](#사용-시나리오별-가이드)
5. [모든 기능 상세 설명](#모든-기능-상세-설명)
6. [고급 사용법](#고급-사용법)
7. [문제 해결](#문제-해결)
8. [참고 자료](#참고-자료)

---

## 개요

MAMMAL Mouse는 마우스의 3D 자세 추정 및 메쉬 재구성을 위한 통합 프레임워크입니다.

### 주요 기능

| 기능 | 설명 | 상태 |
|------|------|------|
| **Multi-view Fitting** | 다중 카메라 동기화 영상에서 3D 피팅 | ✅ 안정 |
| **Monocular Fitting** | 단일 카메라 영상에서 직접 3D 피팅 | 🆕 신규 |
| **ML Keypoint Detection** | YOLOv8, SuperAnimal 기반 키포인트 검출 | 🆕 신규 |
| **Geometric Baseline** | PCA 기반 기하학적 키포인트 추정 | ✅ 안정 |
| **Manual Labeling Workflow** | Roboflow 기반 수동 라벨링 파이프라인 | 🆕 신규 |
| **Hydra Configuration** | 유연한 실험 관리 시스템 | ✅ 안정 |

### 지원하는 입력 형식

- **Multi-view**: 동기화된 다중 카메라 영상 + 2D 키포인트 + 실루엣 마스크
- **Monocular**: 단일 카메라 영상 프레임 (PNG, JPG)
- **Preprocessing**: 원본 비디오 (MP4, AVI 등)

---

## 프로젝트 구조

### 최종 정리된 구조 (2025-11-15)

```
MAMMAL_mouse/
├── README.md                    # 프로젝트 개요
├── requirements.txt             # Python 의존성
│
├── 📁 conf/                     # Hydra 설정 파일
│   ├── config.yaml              # 메인 설정
│   ├── dataset/                 # 데이터셋별 설정
│   ├── preprocess/              # 전처리 방법 설정
│   └── optim/                   # 최적화 설정
│
├── 🐍 Python Scripts (루트)    # 실행 가능한 메인 스크립트
│   ├── fitter_articulation.py   # 메인 피팅 스크립트
│   ├── fit_monocular.py         # 🆕 모노큘러 피팅
│   ├── preprocess.py            # 전처리 파이프라인
│   ├── train_yolo_pose.py       # 🆕 YOLO 학습
│   ├── articulation_th.py       # 관절 모델
│   ├── bodymodel_th.py          # 바디 모델
│   └── utils.py                 # 유틸리티 함수
│
├── 📦 preprocessing_utils/      # 전처리 모듈
│   ├── keypoint_estimation.py   # 기하학적 키포인트 추정
│   ├── mask_processing.py       # 마스크 처리
│   ├── yolo_keypoint_detector.py    # 🆕 YOLO 검출기
│   ├── superanimal_detector.py      # 🆕 SuperAnimal 검출기
│   ├── dannce_to_yolo.py            # 🆕 데이터셋 변환
│   └── visualize_yolo_labels.py     # 🆕 라벨 시각화
│
├── 💾 data/                     # 데이터셋
│   ├── raw/                     # 원본 데이터
│   ├── preprocessed/            # 전처리 결과
│   ├── training/                # ML 학습 데이터
│   │   ├── yolo_mouse_pose/     # YOLO 데이터셋
│   │   └── manual_labeling/     # 수동 라벨링 (진행 중)
│   └── examples/                # 예제 데이터
│
├── 🤖 models/                   # 모델 가중치
│   ├── pretrained/              # 사전학습 모델
│   │   ├── superanimal_topviewmouse/  # SuperAnimal
│   │   ├── sam/                 # SAM (Segment Anything)
│   │   ├── yolov8n-pose.pt      # YOLOv8 기본
│   │   └── yolo11n.pt           # YOLO11 기본
│   └── trained/                 # 학습된 모델
│       └── yolo/                # YOLO 학습 결과
│
├── 📊 results/                  # 최신 실험 결과
│   ├── monocular/               # 모노큘러 피팅 결과
│   └── preprocessing/           # 전처리 결과
│
├── 📁 outputs/                  # Hydra 자동 생성
│   └── archives/                # 오래된 실험 아카이브
│
├── 📚 docs/                     # 문서
│   ├── guides/                  # 사용 가이드 (7개)
│   │   ├── MONOCULAR_FITTING_GUIDE.md
│   │   ├── QUICK_START_LABELING.md
│   │   ├── ROBOFLOW_LABELING_GUIDE.md
│   │   ├── SAM_MASK_ACQUISITION_MANUAL.md
│   │   └── MAMMAL_ARCHITECTURE_MANUAL.md
│   └── reports/                 # 연구 보고서 (11개)
│       ├── 251114_ml_keypoint_detection_integration.md
│       ├── 251115_comprehensive_ml_keypoint_summary.md
│       └── ... (기타 세션 보고서)
│
├── 🎨 assets/                   # 정적 리소스
│   ├── colormaps/               # 시각화 컬러맵
│   ├── figs/                    # README 이미지
│   └── mouse_model/             # 3D 마우스 모델
│       ├── mouse_reduced_face_*.obj
│       └── mouse_txt/           # 모델 파라미터
│
└── 🧪 tests/                    # 테스트 스크립트
    ├── test_sam.py
    ├── test_superanimal.py
    └── ... (기타 테스트)
```

---

## 환경 설정

### 1회 설정 (처음 사용 시)

```bash
# 1. 리포지토리 클론
git clone <repository_url>
cd MAMMAL_mouse

# 2. 환경 설정 스크립트 실행
bash setup.sh
```

### 환경 활성화

```bash
conda activate mammal_stable
```

### 의존성 확인

```bash
# Python 환경 확인
python --version  # Python 3.10

# PyTorch 및 CUDA 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 주요 패키지 확인
python -c "import pytorch3d, hydra, ultralytics; print('All packages OK')"
```

---

## 사용 시나리오별 가이드

### 시나리오 1: 단일 영상에서 빠른 3D 피팅 (NEW! ⭐ 추천)

**목적**: 단일 카메라 영상에서 빠르게 3D 자세 추정

**입력**:
- 영상 프레임 (PNG, JPG)
- 또는 전처리된 마스크 + 키포인트

**예상 시간**: ~30초/프레임

**실행 방법**:

```bash
# Step 1: 영상 프레임 준비
# - 프레임 추출이 필요한 경우:
ffmpeg -i video.mp4 -vf fps=10 frames/frame_%04d.png

# Step 2: Monocular fitting 실행
python fit_monocular.py \
  --input_dir frames/ \
  --output_dir results/monocular/my_experiment \
  --detector geometric \
  --max_images 10

# 옵션:
# --detector geometric|yolo|superanimal
# --max_images N (처리할 프레임 수 제한)
# --yolo_weights path/to/weights.pt (YOLO 사용 시)
```

**결과 확인**:
```bash
ls results/monocular/my_experiment/
# - frame_0000_mesh.obj: 3D 메쉬
# - frame_0000_keypoints.png: 키포인트 시각화
# - ... (각 프레임별 결과)
```

**상세 가이드**: `docs/guides/MONOCULAR_FITTING_GUIDE.md`

---

### 시나리오 2: ML 기반 고품질 키포인트 검출 (NEW! 🎓)

**목적**: YOLOv8을 수동 라벨링하여 고품질 키포인트 검출기 구축

**예상 개선**: Confidence 2×, Loss 10-20×, mAP 0→0.6-0.8

**전체 워크플로우** (~3-4시간):

#### Step 1: 이미지 샘플링 (완료됨)
```bash
# 이미 20개 이미지 준비됨
ls data/training/manual_labeling/images/
# sample_000.png ~ sample_019.png
```

#### Step 2: Roboflow에서 라벨링 (2-3시간)
1. https://roboflow.com/ 접속 및 가입
2. 프로젝트 생성: "MAMMAL_Mouse_Keypoints" (Keypoint Detection)
3. 22개 keypoints 정의 (정확한 순서!):
   ```
   0: nose, 1: left_ear, 2: right_ear, 3: left_eye, 4: right_eye,
   5: head_center, 6-13: spine_1 to spine_8,
   14-17: paws (left/right, front/rear),
   18-20: tail (base/mid/tip), 21: centroid
   ```
4. 20개 이미지 업로드 및 라벨링
5. YOLO v8 format으로 export

#### Step 3: 라벨 검증 (5분)
```bash
# Roboflow export 압축 해제
cd ~/Downloads
unzip roboflow.zip -d ~/dev/MAMMAL_mouse/data/training/manual_labeling/roboflow_export

# 라벨 복사
cp -r data/training/manual_labeling/roboflow_export/train/labels/* \
      data/training/manual_labeling/labels/

# 시각화 검증
python preprocessing_utils/visualize_yolo_labels.py \
  --images data/training/manual_labeling/images \
  --labels data/training/manual_labeling/labels \
  --output data/training/manual_labeling/viz \
  --max_images 5

# 결과 확인
ls data/training/manual_labeling/viz/
```

#### Step 4: 데이터셋 병합 (2분)
```bash
# Manual (20) + Geometric (50) = 70 images
python preprocessing_utils/merge_datasets.py \
  --manual data/training/manual_labeling \
  --geometric data/training/yolo_mouse_pose \
  --output data/training/yolo_mouse_pose_enhanced \
  --train_split 0.8
```

#### Step 5: YOLOv8 학습 (30분)
```bash
python scripts/train_yolo_pose.py \
  --data data/training/yolo_mouse_pose_enhanced/data.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 256 \
  --weights models/pretrained/yolov8n-pose.pt \
  --name mammal_mouse_finetuned

# 학습 모니터링 (다른 터미널)
tail -f /tmp/yolo_train.log
```

#### Step 6: 평가 (10분)
```bash
# Validation 평가
python -c "
from ultralytics import YOLO
model = YOLO('models/trained/yolo/mammal_mouse_finetuned/weights/best.pt')
metrics = model.val(data='data/training/yolo_mouse_pose_enhanced/data.yaml')
print(f'mAP50: {metrics.box.map50:.3f}')
print(f'mAP50-95: {metrics.box.map:.3f}')
"

# 시각적 비교
python fit_monocular.py \
  --input_dir data/training/manual_labeling/images \
  --output_dir results/yolo_comparison \
  --detector yolo \
  --yolo_weights models/trained/yolo/mammal_mouse_finetuned/weights/best.pt \
  --max_images 5
```

#### Step 7: Production 통합 (5분)
```bash
# Best model 복사
mkdir -p models/production
cp models/trained/yolo/mammal_mouse_finetuned/weights/best.pt \
   models/production/yolo_mouse_pose_finetuned.pt

# 이후 사용
python fit_monocular.py \
  --detector yolo \
  --yolo_weights models/production/yolo_mouse_pose_finetuned.pt
```

**상세 가이드**:
- `docs/guides/QUICK_START_LABELING.md`
- `docs/guides/ROBOFLOW_LABELING_GUIDE.md`
- `docs/reports/251115_comprehensive_ml_keypoint_summary.md`

---

### 시나리오 3: 다중 카메라 영상에서 3D 피팅 (기본)

**목적**: 동기화된 다중 카메라 영상에서 정밀한 3D 자세 추정

**입력**:
- 다중 카메라 동기화 영상
- 2D 키포인트 (각 view)
- 실루엣 마스크 (각 view)
- 카메라 파라미터

**실행 방법**:

```bash
# 예제 데이터셋 사용 (markerless_mouse_1)
python fitter_articulation.py \
  dataset=markerless \
  optim=fast \
  fitter.end_frame=10

# 커스텀 데이터셋
python fitter_articulation.py \
  dataset=custom \
  data.data_dir=data/preprocessed/my_dataset/ \
  fitter.end_frame=100
```

**결과 확인**:
```bash
ls outputs/YYYY-MM-DD/HH-MM-SS/
# Hydra가 자동으로 타임스탬프 폴더 생성
```

---

### 시나리오 4: 원본 비디오 전처리 (단일 카메라)

**목적**: 원본 비디오를 MAMMAL 입력 형식으로 변환

**입력**: 비디오 파일 (MP4, AVI 등)

**출력**:
- `videos_undist/0.mp4`: 원본 비디오
- `simpleclick_undist/0.mp4`: 마스크 비디오
- `keypoints2d_undist/result_view_0.pkl`: 2D 키포인트
- `new_cam.pkl`: 카메라 파라미터

**실행 방법**:

```bash
# Step 1: 설정 파일 준비
# conf/dataset/my_video.yaml 생성:
cat > conf/dataset/my_video.yaml << 'EOF'
# @package _global_

data:
  data_dir: data/preprocessed/my_video/
  views_to_use: [0]

preprocess:
  input_video_path: data/raw/my_video.mp4
  output_data_dir: data/preprocessed/my_video/

fitter:
  start_frame: 0
  end_frame: 100
  render_cameras: [0]
EOF

# Step 2: 전처리 실행
python scripts/preprocess.py \
  dataset=my_video \
  mode=single_view_preprocess

# Step 3: 결과 확인
ls data/preprocessed/my_video/
```

**전처리 후 피팅**:
```bash
python fitter_articulation.py \
  dataset=my_video \
  mode=multi_view
```

---

### 시나리오 5: 기하학적 베이스라인 사용

**목적**: 빠른 프로토타이핑 및 베이스라인 비교

**장점**:
- 모델 학습 불필요
- 빠른 실행
- 전처리 단계 없음

**단점**:
- 정확도 낮음 (특히 paw detection)
- Confidence 낮음 (0.4-0.6)

**사용 방법**:

```bash
# Monocular fitting with geometric detector
python fit_monocular.py \
  --input_dir frames/ \
  --output_dir results/geometric_baseline \
  --detector geometric

# 또는 전처리 단계에서
python scripts/preprocess.py \
  dataset=my_video \
  preprocess.method=opencv  # 기하학적 방법
```

**예상 성능**:
- Detected keypoints: 15/22
- Confidence: 0.4-0.6
- Loss: ~300K

---

## 모든 기능 상세 설명

### 1. Keypoint Detector 옵션

#### Geometric (기하학적)
- **파일**: `preprocessing_utils/keypoint_estimation.py`
- **방법**: PCA 기반 contour 분석
- **장점**: 빠름, 학습 불필요
- **단점**: 정확도 낮음
- **사용 시기**: 빠른 프로토타이핑

#### YOLOv8-Pose
- **파일**: `preprocessing_utils/yolo_keypoint_detector.py`
- **방법**: CNN 기반 키포인트 검출
- **장점**: 빠르고 정확, GPU 가속
- **단점**: 학습 데이터 필요
- **사용 시기**: Production, 실시간 처리

**사용법**:
```bash
python fit_monocular.py \
  --detector yolo \
  --yolo_weights models/production/yolo_mouse_pose_finetuned.pt
```

#### SuperAnimal-TopViewMouse
- **파일**: `preprocessing_utils/superanimal_detector.py`
- **방법**: DeepLabCut pretrained model (27 keypoints → 22 mapping)
- **장점**: 사전학습, 해부학적 정확도
- **단점**: DLC API 제약, 느림
- **상태**: Geometric fallback 사용 중 (DLC 3.0 대기)

**사용법**:
```bash
python fit_monocular.py \
  --detector superanimal \
  --superanimal_model models/pretrained/superanimal_topviewmouse
```

### 2. Hydra Configuration 시스템

#### 설정 파일 구조
```
conf/
├── config.yaml          # 메인 설정 (defaults)
├── dataset/             # 데이터셋별 설정
│   ├── markerless.yaml  # 예: 6-view multi-camera
│   ├── shank3.yaml      # 예: single-view
│   └── custom.yaml      # 템플릿
├── preprocess/          # 전처리 방법
│   ├── opencv.yaml      # Geometric baseline
│   └── sam.yaml         # SAM (향후)
└── optim/               # 최적화 설정
    ├── fast.yaml        # 빠른 테스트 (적은 iteration)
    └── accurate.yaml    # 정밀 결과 (많은 iteration)
```

#### 설정 조합 예시

**빠른 테스트**:
```bash
python fitter_articulation.py \
  dataset=markerless \
  optim=fast \
  fitter.end_frame=5
```

**정밀 피팅**:
```bash
python fitter_articulation.py \
  dataset=custom \
  optim=accurate \
  fitter.with_render=true
```

**파라미터 오버라이드**:
```bash
python fitter_articulation.py \
  dataset=shank3 \
  fitter.start_frame=10 \
  fitter.end_frame=50 \
  optim.solve_step1_iters=200
```

### 3. 3D Fitting Pipeline

#### 3단계 최적화

**Step 0: 초기 자세 추정** (10 iterations)
- Global translation/rotation 초기화
- 대략적인 자세 맞춤

**Step 1: 키포인트 기반 피팅** (100 iterations)
- 2D 키포인트와 3D 모델 정렬
- 관절 각도 최적화
- Loss terms: 2D reprojection, 3D keypoint, bone length

**Step 2: Silhouette 기반 정밀화** (30 iterations)
- PyTorch3D를 사용한 실루엣 매칭
- 메쉬 표면 최적화
- Loss terms: silhouette IoU, smoothness

#### Loss Terms 가중치

`fitter_articulation.py` 라인 ~82:
```python
self.term_weights = {
    "theta": 3,           # 관절 정규화
    "3d": 2.5,            # 3D 키포인트 loss
    "2d": 0.2,            # 2D 재투영 loss
    "bone": 0.5,          # 뼈 길이 제약
    "scale": 0.5,         # 스케일 정규화
    "mask": 0,            # 실루엣 loss (기본 비활성화)
    "chest_deformer": 0.1,  # 가슴 변형 정규화
    "stretch": 1,         # 늘어남 페널티
    "temp": 0.25,         # 시간적 부드러움
    "temp_d": 0.2         # 시간 미분 부드러움
}
```

### 4. 시각화 및 출력

#### 출력 파일 구조

**Monocular Fitting**:
```
results/monocular/my_experiment/
├── frame_0000_mesh.obj           # 3D 메쉬
├── frame_0000_keypoints.png      # 키포인트 오버레이
├── frame_0000_params.pkl         # 피팅 파라미터
└── ...
```

**Multi-view Fitting**:
```
outputs/YYYY-MM-DD/HH-MM-SS/
└── (Hydra 자동 생성)

결과는 실제로 저장됨:
results/obj/                      # 3D 메쉬
results/params/                   # 피팅 파라미터
results/render/                   # 시각화
```

#### 비디오 생성

```bash
# 렌더링 이미지에서 비디오 생성
ffmpeg -framerate 10 \
  -i results/monocular/my_experiment/frame_%04d_keypoints.png \
  -c:v libx264 -pix_fmt yuv420p -y output.mp4
```

---

## 고급 사용법

### 1. 커스텀 키포인트 가중치 조정

`fitter_articulation.py` 라인 ~65:
```python
self.keypoint_weight = np.ones(22)

# 신뢰도 낮은 키포인트 가중치 감소
self.keypoint_weight[4] = 0.4   # right_ear
self.keypoint_weight[11] = 0.9  # left_hip
self.keypoint_weight[15] = 0.9  # left_foot

# 또는 특정 키포인트 무시
self.keypoint_weight[14:18] = 0  # paws (geometric이 잘 못잡을 때)
```

### 2. SAM을 이용한 고품질 마스크 생성

```bash
# SAM으로 마스크 생성 (향후 기능)
python tests/sam_point_prompt.py \
  --image frame_0001.png \
  --output mask_0001.png

# 또는 batch processing
python preprocessing_utils/sam_inference.py \
  --input_dir frames/ \
  --output_dir masks/
```

**상세 가이드**: `docs/guides/SAM_MASK_ACQUISITION_MANUAL.md`

### 3. 배치 처리

```bash
# 여러 데이터셋 순차 처리
for dataset in mouse1 mouse2 mouse3; do
  python fit_monocular.py \
    --input_dir data/${dataset}/frames \
    --output_dir results/monocular/${dataset} \
    --detector yolo \
    --max_images 100
done
```

### 4. GPU 메모리 최적화

```bash
# Batch size 감소
python scripts/train_yolo_pose.py --batch 4  # default: 8

# 이미지 크기 감소
python scripts/train_yolo_pose.py --imgsz 192  # default: 256

# Fitting시 렌더링 비활성화
python fitter_articulation.py fitter.with_render=false
```

---

## 문제 해결

### 환경 관련

**Q: `ModuleNotFoundError: No module named 'torch'`**
```bash
# 해결: 환경 재설치
bash setup.sh
conda activate mammal_stable
```

**Q: `CUDA out of memory`**
```bash
# 해결 1: Batch size 감소
python scripts/train_yolo_pose.py --batch 2

# 해결 2: 프레임 수 제한
python fit_monocular.py --max_images 10

# 해결 3: GPU 메모리 확인
nvidia-smi
```

### Keypoint Detection 관련

**Q: Geometric detector가 paws를 못 찾음**
- **답변**: 정상입니다. Geometric 방법은 PCA 기반이라 사지(paws)검출 불가능
- **해결**: YOLO fine-tuning 또는 SuperAnimal 사용

**Q: YOLO 학습 결과가 mAP ~0**
- **답변**: Geometric labels로 학습하면 발생
- **해결**: Manual labeling (20개) 수행 후 재학습

**Q: SuperAnimal이 작동하지 않음**
- **답변**: DLC 2.3.11 TensorFlow API 제약
- **해결**: Geometric fallback 사용 중, DLC 3.0 릴리스 대기

### Fitting 관련

**Q: Fitting이 이상한 자세로 수렴**
```bash
# 해결 1: 더 많은 iteration
python fitter_articulation.py optim=accurate

# 해결 2: 시작 프레임 변경
python fitter_articulation.py fitter.start_frame=10

# 해결 3: 키포인트 품질 확인
python preprocessing_utils/visualize_yolo_labels.py ...
```

**Q: `FileNotFoundError: new_cam.pkl not found`**
```bash
# 해결: 전처리 먼저 실행
python scripts/preprocess.py dataset=my_video mode=single_view_preprocess
```

### Rendering 관련

**Q: `NoSuchDisplayException: Cannot connect to "None"`**
- **답변**: 이미 처리됨 (`export PYOPENGL_PLATFORM=egl`)
- **확인**: `ldconfig -p | grep EGL`

**Q: 렌더링 이미지가 검은색**
```bash
# 해결: 렌더링 비활성화하고 디버깅
python fitter_articulation.py fitter.with_render=false
```

---

## 참고 자료

### 문서 (docs/guides/)
1. **MONOCULAR_FITTING_GUIDE.md** - 모노큘러 피팅 상세 가이드
2. **QUICK_START_LABELING.md** - 수동 라벨링 빠른 시작
3. **ROBOFLOW_LABELING_GUIDE.md** - Roboflow 단계별 가이드
4. **SAM_MASK_ACQUISITION_MANUAL.md** - SAM 마스크 획득 방법
5. **MAMMAL_ARCHITECTURE_MANUAL.md** - 전체 아키텍처 상세 설명

### 연구 보고서 (docs/reports/)
1. **251114_ml_keypoint_detection_integration.md** - ML 통합 기술 보고서
2. **251115_comprehensive_ml_keypoint_summary.md** - 종합 ML 워크플로우
3. **251103_success_report.md** - 전처리 개선 보고서
4. **251104_silhouette_fitting_final.md** - Silhouette 피팅 보고서

### 외부 리소스
- **MAMMAL 논문**: [Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL]
- **DANNCE**: https://github.com/spoonsso/dannce
- **PyTorch3D**: https://pytorch3d.org/
- **Ultralytics YOLOv8**: https://docs.ultralytics.com/
- **DeepLabCut**: https://deeplabcut.github.io/
- **Roboflow**: https://roboflow.com/

### 프로젝트 히스토리
- **v1.0** (2025-11-03): 기본 multi-view fitting, Hydra 설정
- **v2.0** (2025-11-14~15):
  - Monocular fitting 추가
  - ML keypoint detection 통합
  - Manual labeling 워크플로우
  - 프로젝트 구조 대폭 정리

---

## 요약: 추천 워크플로우

### 초보자 (빠른 시작)
```bash
# 1. 환경 설정
bash setup.sh
conda activate mammal_stable

# 2. Monocular fitting 시도 (geometric)
python fit_monocular.py \
  --input_dir frames/ \
  --output_dir results/test \
  --detector geometric \
  --max_images 5

# 3. 결과 확인
ls results/test/
```

### 중급자 (품질 향상)
```bash
# 1. 20개 이미지 수동 라벨링 (Roboflow)
# 2. YOLO 학습
python scripts/train_yolo_pose.py --data data/...
# 3. Fine-tuned 모델로 피팅
python fit_monocular.py --detector yolo --yolo_weights ...
```

### 고급자 (Production)
```bash
# 1. 커스텀 데이터셋 설정
# 2. Hyperparameter 튜닝
# 3. Multi-view + ML keypoint 결합
# 4. Batch processing 파이프라인 구축
```

---

**최종 업데이트**: 2025-11-15
**작성자**: MAMMAL Mouse Team
**문의**: docs/README.md 참조
