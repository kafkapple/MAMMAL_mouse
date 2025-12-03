# MAMMAL_mouse

Three-dimensional surface motion capture of mice using the MAMMAL framework. This project enables markerless 3D pose estimation and mesh reconstruction for behavioral analysis by fitting an articulated 3D mouse model to video data.

![mouse_model](assets/figs/mouse_1.png)

---

## 🚀 Shell 스크립트 사용법 (권장)

> **권장**: 쉘 스크립트 사용 시 `PYOPENGL_PLATFORM=egl` 자동 설정, 환경 검증 포함

### Multi-View Fitting (`run_mesh_fitting_default.sh`)

```bash
# 🎯 Experiment 기반 실행 (권장)
./run_mesh_fitting_default.sh quick_test           # conf/experiment/quick_test.yaml 사용
./run_mesh_fitting_default.sh quick_test 0 5       # experiment + frame override

# 기본 사용 (frame 0-10, experiment 없이)
./run_mesh_fitting_default.sh - 0 10               # "-"는 experiment 생략

# keypoint 없이 (silhouette only)
./run_mesh_fitting_default.sh - 0 10 -- --keypoints none

# 다른 input_dir 지정 + keypoint 없이
./run_mesh_fitting_default.sh - 0 10 -- --input_dir /home/joon/data/my_data --keypoints none
```

**Experiment configs** (`conf/experiment/`):

| Config | Views | Keypoints | 설명 |
|--------|-------|-----------|------|
| `quick_test` | 6 | ✅ | 5 frames, 최소 iterations (디버깅) |
| `views_6` | 6 | ✅ | Full baseline (100 samples) |
| `views_5` | 5 | ✅ | [0,1,2,3,4] |
| `views_4` | 4 | ✅ | [0,1,2,3] |
| `views_3_diagonal` | 3 | ✅ | [0,2,4] 대각선 배치 |
| `views_3_consecutive` | 3 | ✅ | [0,1,2] 연속 배치 |
| `views_2_opposite` | 2 | ✅ | [0,3] 반대편 |
| `views_1_single` | 1 | ✅ | [0] 단일뷰 |
| `silhouette_only_6views` | 6 | ❌ | Mask만 사용 (keypoint 없음) |
| `silhouette_only_4views` | 4 | ❌ | |
| `silhouette_only_3views` | 3 | ❌ | |
| `silhouette_only_1view` | 1 | ❌ | |
| `accurate_6views` | 6 | ✅ | 고정밀 (iterations 증가) |

**데이터셋 정보** (`data/examples/markerless_mouse_1_nerf/`):
| 항목 | 값 |
|------|-----|
| 카메라 | 6개 (0~5) |
| 총 프레임 | 18,000 frames |
| 기본 샘플링 | end_frame=1000, interval=10 → 100 samples |

### Monocular Fitting (`run_mesh_fitting_monocular.sh`)

```bash
# 기본 사용 (전체 이미지)
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output

# 처음 10개만
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output 10

# keypoint 없이 (silhouette only)
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output - -- --keypoints none

# 처음 5개 + silhouette only
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output 5 -- --keypoints none
```

**4x Upsampled 데이터셋** (`data/100-KO-male-56-20200615_4x/`):
```bash
# Cropped 이미지로 silhouette-only fitting (권장)
./run_mesh_fitting_monocular.sh \
    data/100-KO-male-56-20200615_4x/cropped/ \
    results/monocular/shank3_4x/ \
    - -- --keypoints none

# 처음 5개만 테스트
./run_mesh_fitting_monocular.sh \
    data/100-KO-male-56-20200615_4x/cropped/ \
    results/monocular/shank3_4x_test/ \
    5 -- --keypoints none
```

| 항목 | 값 |
|------|-----|
| 경로 | `data/100-KO-male-56-20200615_4x/cropped/` |
| 파일 패턴 | `*_cropped.png` + `*_mask.png` |
| 프레임 수 | 20개 |
| 해상도 | ~516×556 (4x upsampled) |

> **Note**: `--` 뒤에 추가 인자를 전달하면 Python 스크립트로 그대로 전달됩니다. EGL 환경변수는 자동 설정됩니다.

### 🆕 Silhouette-Only Fitting (keypoint 없이 마스크만 사용)

Keypoint annotation 없이 **mask silhouette만으로** 메시 피팅을 수행합니다.

#### 워크플로우

```
1. 디버그 테스트 (2 프레임) - 오류 확인
   └─ 성공? → 2. 실험 스크립트로 설정 비교
              └─ 최적 설정 확인
                  └─ 3. 전체 프레임 실행
```

#### Step 1: 디버그 테스트 (필수!)

```bash
# ⚠️ 중요: --input_dir로 실제 데이터 경로 지정 (서버마다 다름!)
./run_mesh_fitting_default.sh 0 2 -- --keypoints none \
    --input_dir /home/joon/MAMMAL_mouse/data/markerless_mouse_1_nerf

# 데이터 위치 확인
ls /home/joon/MAMMAL_mouse/data/
```

#### Step 2: 실험 비교 (디버그 성공 후)

```bash
# 4가지 설정으로 순차 실험
./run_silhouette_experiments.sh /path/to/your/data 0 2
```

#### Step 3: 전체 실행

```bash
# 최적 설정으로 전체 프레임 실행
./run_mesh_fitting_default.sh 0 100 -- --keypoints none \
    --input_dir /path/to/data \
    silhouette.iter_multiplier=3.0 silhouette.theta_weight=15.0
```

**Silhouette 모드 설정 옵션** (`conf/config.yaml` 또는 CLI):

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `silhouette.iter_multiplier` | 2.0 | 반복 횟수 배율 (높을수록 정밀) |
| `silhouette.theta_weight` | 10.0 | 포즈 정규화 (높을수록 안정적) |
| `silhouette.bone_weight` | 2.0 | 뼈대 길이 정규화 |
| `silhouette.scale_weight` | 50.0 | 스케일 정규화 |
| `silhouette.use_pca_init` | true | PCA 기반 회전 초기화 |

### 서버 간 호환성 (Portability)

스크립트와 config는 다양한 서버 환경에서 동작하도록 설계되었습니다:

**자동 처리 항목:**
- Python 경로: `miniconda3` / `anaconda3` 자동 감지
- EGL 환경변수: 스크립트에서 자동 설정

**수동 지정 필요:**
```bash
# 데이터 경로는 서버마다 다르므로 항상 --input_dir 사용
./run_mesh_fitting_default.sh 0 10 -- --input_dir /your/server/data/path

# 또는 Hydra 방식
python fitter_articulation.py data.data_dir=/your/server/data/path
```

**Config 파일의 경로:**
- 모든 config는 상대 경로 사용 (`data/examples/...`)
- 절대 경로는 CLI에서 override 권장
- 서버별 데이터 위치 확인: `ls /home/$USER/*/data/`

---

## ⚡ Quick Start (5분 안에 실행)

### 📁 데이터 준비

```bash
# 데이터 폴더 구조 (예시)
data/
├── my_video/                    # Monocular용 (단일 카메라)
│   ├── 000000_rgb.png           # RGB 이미지
│   ├── 000000_mask.png          # 바이너리 마스크
│   └── ...
└── examples/markerless_mouse_1_nerf/   # Multi-view용 (다중 카메라)
    ├── videos_undist/           # 6개 뷰 비디오
    ├── simpleclick_undist/      # 마스크
    ├── keypoints2d_undist/      # 2D keypoints
    └── new_cam.pkl              # 카메라 파라미터
```

### 🎯 Monocular Fitting (단일 이미지/비디오)

```bash
# 환경 활성화
conda activate mammal_stable

# 1. 기본 실행 (keypoint 기반)
python fit_monocular.py \
    --input_dir data/my_video/ \
    --output_dir results/monocular/test/

# 2. Keypoint 선택 (부정확한 부분 제외)
python fit_monocular.py \
    --input_dir data/my_video/ \
    --output_dir results/monocular/test/ \
    --keypoints spine,head      # head, spine, limbs, tail, centroid

# 3. Silhouette 기반 (keypoint 없이 mask만 사용)
python fit_monocular.py \
    --input_dir data/my_video/ \
    --output_dir results/monocular/test/ \
    --keypoints none            # mask IoU loss로 fitting
```

**출력 파일**:
```
results/monocular/test/
├── *_mesh.obj          # 3D 메시 (Blender 호환)
├── *_comparison.png    # RGB | Mask | Rendered | Overlay
├── *_keypoints.png     # Keypoint 시각화
├── *_rendered.png      # 렌더링된 mesh
└── *_params.pkl        # MAMMAL 파라미터
```

### 🎥 Multi-View Fitting (다중 카메라)

```bash
# 환경 활성화 (headless 서버용)
conda activate mammal_stable
export PYOPENGL_PLATFORM=egl  # ⚠️ 필수! 직접 python 실행 시 반드시 설정

# 1. 기본 실행 (Hydra 방식)
python fitter_articulation.py \
    dataset=default_markerless \
    fitter.start_frame=0 \
    fitter.end_frame=10 

# 2. argparse 방식 (fit_monocular.py와 동일한 CLI)
python fitter_articulation.py \
    --input_dir /path/to/data \
    --start_frame 0 \
    --end_frame 10 \
    --with_render

# 3. Keypoint 없이 Silhouette만 사용
python fitter_articulation.py \
    dataset=default_markerless \
    --keypoints none           # 또는 fitter.use_keypoints=false

# 4. 혼합 사용 (Hydra + argparse)
python fitter_articulation.py \
    dataset=default_markerless \
    --keypoints none \
    --with_render
```

> **CLI 호환성**: `fitter_articulation.py`는 Hydra 방식(`key=value`)과 argparse 방식(`--key value`) 모두 지원합니다.

**Config 설정** (`conf/dataset/default_markerless.yaml`):
```yaml
video_dir: data/examples/markerless_mouse_1_nerf/videos_undist/
mask_dir: data/examples/markerless_mouse_1_nerf/simpleclick_undist/
keypoint_dir: data/examples/markerless_mouse_1_nerf/keypoints2d_undist/
cam_pkl: data/examples/markerless_mouse_1_nerf/new_cam.pkl
```

**출력 결과**:
```
results/fitting/{dataset}_{timestamp}/
├── fitting_keypoints_*.png     # 6뷰 keypoint overlay
├── render/fitting_*.png        # 6뷰 mesh rendering
├── obj/*.obj                   # Frame별 3D mesh
└── params/*.pkl                # Frame별 파라미터
```

### 📊 결과 시각화

```bash
# Cropped fitting 결과 + GT RGB 비교
python scripts/utils/visualize_fitting_comparison.py \
    --results results/cropped_fitting_final \
    --gt_dir data/cropped_images \
    --output results/gallery.png
```

---

## 📁 결과 출력물 가이드 (Output Structure)

실험 완료 후 `results/fitting/{experiment_name}/` 폴더에 다음 파일들이 생성됩니다:

### 폴더 구조
```
results/fitting/{dataset}_{views}_{keypoints}_{timestamp}/
├── config.yaml                              # 실험 설정 (재현용)
├── loss_history.json                        # 학습 로스 기록
├── render/
│   ├── step_1_frame_000000.png              # Step1 결과 + 키포인트 오버레이
│   ├── step_2_frame_000000.png              # Step2 최종 결과
│   ├── step_summary_frame_000000.png        # 3단계 비교 (첫 프레임)
│   ├── debug/                               # 중간 iteration 결과
│   │   ├── step_0_frame_000000_iter_00000.png
│   │   ├── step_1_frame_000000_iter_00000.png
│   │   └── ...
│   └── keypoints/                           # GT vs Predicted 비교
│       ├── step_1_frame_000000_keypoints.png
│       ├── step_1_frame_000000_keypoints_gt.png
│       └── step_1_frame_000000_keypoints_compare.png
├── params/                                  # 모델 파라미터 (pickle)
│   ├── step_1_frame_000000.pkl
│   └── step_2_frame_000000.pkl
└── obj/                                     # 3D 메시 파일
    └── step_2_frame_000000.obj
```

### Fitting 3단계 설명

| Step | 이름 | 최적화 대상 | 설명 |
|------|------|------------|------|
| **Step 0** | Global Positioning | `trans`, `rotation`, `scale` | 초기 위치/크기/방향 설정 (관절각 고정) |
| **Step 1** | Articulation Fitting | `thetas`, `bone_lengths` | 관절 각도와 뼈 길이 최적화 (포즈 피팅 핵심) |
| **Step 2** | Silhouette Refinement | 전체 파라미터 | Mask loss 활성화, 실루엣 정교화 |

### 🎯 3D Geometric Prior로 사용하기

다른 프로젝트(예: NeRF, 3D Gaussian Splatting)에서 프레임별 3D mesh를 geometric prior로 활용하려면:

#### 1. OBJ 파일 (권장 - 가장 간단)
```python
# 프레임별 3D 메시 직접 로드
import trimesh

mesh = trimesh.load("results/fitting/.../obj/step_2_frame_000000.obj")
vertices = mesh.vertices  # (N_verts, 3) - 3D 좌표
faces = mesh.faces        # (N_faces, 3) - 삼각형 인덱스

# 모든 프레임 로드
import glob
obj_files = sorted(glob.glob("results/fitting/.../obj/step_2_frame_*.obj"))
meshes = [trimesh.load(f) for f in obj_files]
```

#### 2. PKL 파일 (파라미터 재사용 - 고급)
```python
import pickle
import torch

# 파라미터 로드
with open("results/fitting/.../params/step_2_frame_000000.pkl", "rb") as f:
    params = pickle.load(f)

# params 구조:
# {
#     "thetas": (1, 20, 3),       # 관절 회전 (axis-angle)
#     "bone_lengths": (1, 20),    # 뼈 길이 오프셋
#     "trans": (1, 3),            # 3D 위치 (mm)
#     "rotation": (1, 3),         # 전역 회전 (axis-angle)
#     "scale": (1, 1),            # 스케일 팩터
#     "chest_deformer": (1, 1),   # 가슴 변형
# }

# BodyModel로 메시 재생성
from bodymodel_th import BodyModelTorch
bodymodel = BodyModelTorch(device='cuda')
V, J = bodymodel.forward(
    params["thetas"], params["bone_lengths"],
    params["rotation"], params["trans"], params["scale"],
    params["chest_deformer"]
)
vertices = V[0].cpu().numpy()  # (N_verts, 3)
```

#### 3. 멀티뷰 데이터 구조

```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/
│   ├── 0.mp4          # View 0 (카메라 0)
│   ├── 1.mp4          # View 1 (카메라 1)
│   ├── 2.mp4          # View 2 (카메라 2)
│   ├── 3.mp4          # View 3 (카메라 3)
│   ├── 4.mp4          # View 4 (카메라 4)
│   └── 5.mp4          # View 5 (카메라 5)
├── new_cam.pkl        # 카메라 파라미터 (6개 카메라)
└── keypoints2d_undist/
    ├── result_view_0.pkl  # View 0 2D 키포인트
    └── ...
```

**뷰 식별 방법:**
- 비디오 파일명 = 카메라 ID (예: `0.mp4` → Camera 0)
- 동일 프레임 인덱스 = 동일 시점 (모든 카메라 동기화됨)
- `new_cam.pkl`: 리스트 형태, `cams[i]`가 Camera i의 파라미터

**카메라 파라미터 구조:**
```python
import pickle
with open("new_cam.pkl", "rb") as f:
    cams = pickle.load(f)  # List[Dict]

# cams[i] 구조:
# {
#     'K': (3, 3),    # Intrinsic matrix
#     'R': (3, 3),    # Rotation matrix
#     'T': (3, 1),    # Translation vector
#     'mapx': ...,    # Undistortion map x
#     'mapy': ...,    # Undistortion map y
# }
```

**멀티뷰 동기화:**
- 동기화 기준: 프레임 인덱스 (파일명의 `frame_XXXXXX`)
- 가정: 모든 카메라가 동기화된 녹화 (동일 FPS, 동일 시작점)
- Fitting 시 동일 frame index로 모든 뷰 동시 접근

### 활용 예시

```python
# 예: 4D-GS에서 프레임별 mesh를 deformation prior로 사용
for frame_idx in range(num_frames):
    mesh = trimesh.load(f"obj/step_2_frame_{frame_idx:06d}.obj")

    # Mesh vertices를 Gaussian 초기화에 사용
    init_positions = mesh.vertices
    init_normals = mesh.vertex_normals

    # 또는 mesh surface에서 샘플링
    points, face_indices = trimesh.sample.sample_surface(mesh, count=10000)
```

### 3D 메시 시퀀스 시각화 및 영상 저장

```bash
# PKL에서 메시 재구성하여 영상 저장 (BodyModel 필요)
python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh_sequence.mp4

# OBJ 파일 직접 사용 (BodyModel 없이 독립 실행)
python scripts/visualize_mesh_sequence.py results/fitting/xxx --use-obj --output mesh.mp4

# 특정 뷰포인트에서 렌더링
python scripts/visualize_mesh_sequence.py results/fitting/xxx \
    --azimuth 45 --elevation 30 --output side_view.mp4

# 360° 회전 뷰 생성
python scripts/visualize_mesh_sequence.py results/fitting/xxx --rotating --output rotating.mp4

# Pyrender 사용 (더 고품질, EGL 필요)
python scripts/visualize_mesh_sequence.py results/fitting/xxx --use-pyrender --output hq.mp4
```

**옵션:**
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--use-obj` | OBJ 파일 직접 로드 (BodyModel 불필요) | False |
| `--azimuth` | 카메라 방위각 (도) | 45 |
| `--elevation` | 카메라 고도각 (도) | 30 |
| `--rotating` | 프레임별 360° 회전 | False |
| `--fps` | 출력 영상 FPS | 30 |
| `--use-pyrender` | Pyrender 렌더러 사용 | False |

---

## ✨ Features

- **Multi-view 3D fitting**: Fit 3D mouse model to synchronized multi-camera videos
- **Single-view (monocular) fitting**: Process single videos with ML-based keypoint detection
- **ML keypoint detection**: YOLOv8-Pose and SuperAnimal support for anatomically accurate keypoints
- **🆕 Flexible keypoint annotation**: Manual annotation tool + automatic format conversion (1-22 keypoints)
- **Confidence-based filtering**: Missing keypoints automatically ignored (no need for all 22!)
- **Hydra configuration**: Flexible experiment management with dataset-specific configs
- **Modular pipeline**: Separate preprocessing and fitting stages for easy customization

---

## 🚀 Quick Start

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/MAMMAL_mouse.git
cd MAMMAL_mouse

# Create conda environment and install dependencies (one-time setup)
bash scripts/setup/setup.sh
```

**What this installs**:
- Python 3.10 environment named `mammal_stable`
- PyTorch 2.0.0 + CUDA 11.8
- PyTorch3D 0.7.5
- All required dependencies (opencv, hydra, ultralytics, etc.)

**Requirements**:
- Anaconda/Miniconda
- NVIDIA GPU with CUDA 11.8
- ~10GB disk space for dependencies

### Step 2: Download Data and Models

#### Option A: Example Dataset (Recommended for First-Time Users)

Download the example multi-view dataset:

```bash
# Download from Google Drive
# https://drive.google.com/file/d/1NbaIFOvpvQ_WLOabUtMrVHS7vVBq-8zD/view?usp=sharing

# Extract to the correct location
unzip markerless_mouse_1_nerf.zip
mv markerless_mouse_1_nerf/ data/examples/
```

**Dataset structure**:
```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/           # 6 camera views
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
├── simpleclick_undist/      # Binary masks
│   ├── 0.mp4
│   └── ...
├── keypoints2d_undist/      # 2D keypoints
│   ├── result_view_0.pkl
│   └── ...
├── new_cam.pkl              # Camera parameters
└── new_params.pkl           # Model parameters
```

#### Option B: Your Own Video

If you have your own single-view video:

```bash
# 1. Place your video
mkdir -p data/raw/my_experiment/
cp /path/to/your/video.mp4 data/raw/my_experiment/

# 2. Extract frames (optional, for monocular fitting)
mkdir -p data/raw/my_experiment/frames/
ffmpeg -i data/raw/my_experiment/video.mp4 \
  -vf "fps=30" data/raw/my_experiment/frames/%06d.png
```

#### Optional: Download Pretrained Models

For ML-based keypoint detection:

```bash
# YOLOv8-Pose pretrained model (auto-downloaded on first use)
# Will be saved to: models/pretrained/yolov8n-pose.pt

# SuperAnimal-TopViewMouse model (optional, 245MB)
python scripts/setup/download_superanimal.py
# Saved to: models/pretrained/superanimal_topviewmouse/
```

### Step 3: Run Your First Experiment

#### Scenario 1: Multi-View Fitting (Example Dataset) ⭐ Recommended for Testing

Process the example multi-view dataset:

```bash
# Activate environment
conda activate mammal_stable

# Run fitting on first 10 frames (using shell script)
./run_mesh_fitting_default.sh 0 10

# OR run directly with Python
python fitter_articulation.py \
  dataset=default_markerless \
  fitter.start_frame=0 \
  fitter.end_frame=10 \
  fitter.with_render=true
```

**What happens**:
1. Loads 6-camera preprocessed data from `data/examples/markerless_mouse_1_nerf/`
2. Fits 3D mouse model to frames 0-10
3. Saves results to `results/fitting/{dataset}_{timestamp}/`

**Expected output**:
```
results/fitting/markerless_mouse_1_nerf_20251125_143000/
├── obj/                     # 3D mesh files (.obj)
│   ├── mesh_000000.obj
│   ├── mesh_000002.obj
│   └── ...
├── params/                  # Fitting parameters (.pkl)
│   ├── param0.pkl
│   ├── param0_sil.pkl
│   └── ...
└── render/                  # Visualization overlays (.png)
    ├── fitting_0.png
    ├── fitting_0_sil.png
    └── debug/               # Optimization debug images
```

**Processing time**: ~5-10 minutes (RTX 3090)

#### Scenario 2: Monocular Fitting (Single Video) 🆕

Process a single-view video with ML keypoint detection:

```bash
conda activate mammal_stable

# Using geometric keypoint detection (baseline)
python fit_monocular.py \
  --input_dir data/raw/my_experiment/frames/ \
  --output_dir results/monocular/my_experiment \
  --detector geometric \
  --max_images 10

# OR using YOLOv8-Pose (better quality, requires GPU)
python fit_monocular.py \
  --input_dir data/raw/my_experiment/frames/ \
  --output_dir results/monocular/my_experiment \
  --detector yolo \
  --max_images 10
```

**What happens**:
1. Detects 22 keypoints per frame using chosen detector
2. Estimates camera parameters from first frame
3. Fits 3D mouse model frame-by-frame
4. Saves meshes, parameters, and visualizations

**Expected output**:
```
results/monocular/my_experiment/
├── obj/                     # 3D mesh files
│   ├── frame_000001.obj
│   └── ...
├── params/                  # Fitting parameters
│   ├── frame_000001.pkl
│   └── ...
├── keypoints_2d/            # Detected 2D keypoints
│   ├── frame_000001.pkl
│   └── ...
├── camera_params.pkl        # Estimated camera
└── visualizations/          # Overlays (if enabled)
```

**Processing time**: ~30 seconds/frame (geometric), ~1 minute/frame (YOLO)

#### Scenario 3: Traditional Preprocessing + Fitting

Full pipeline for single-view video:

```bash
conda activate mammal_stable

# 1. Preprocess video (extract frames, masks, keypoints)
python scripts/preprocess.py \
  dataset=custom \
  mode=single_view_preprocess \
  preprocess.input_video_path="data/raw/my_experiment/video.mp4" \
  preprocess.output_data_dir="data/preprocessed/my_experiment/"

# 2. Fit 3D model to preprocessed data
python fitter_articulation.py \
  dataset=custom \
  data.data_dir="data/preprocessed/my_experiment/" \
  fitter.end_frame=100 \
  fitter.with_render=false
```

**Preprocessing outputs**:
```
data/preprocessed/my_experiment/
├── videos_undist/
│   └── 0.mp4                # Original video
├── simpleclick_undist/
│   └── 0.mp4                # Binary mask video
├── keypoints2d_undist/
│   └── result_view_0.pkl    # 22 keypoints per frame
└── new_cam.pkl              # Camera parameters
```

---

## 📖 Usage Scenarios

### 1️⃣ Quick Test with Example Data (5 minutes)

**Goal**: Verify installation and see multi-view fitting results

```bash
conda activate mammal_stable

# Using shell script (recommended)
./run_mesh_fitting_default.sh 0 5

# OR using Python directly
python fitter_articulation.py \
  dataset=default_markerless \
  optim=fast \
  fitter.end_frame=5
```

**Results**: `results/fitting/{dataset}_{timestamp}/obj/` contains 3D meshes

### 2️⃣ Process Your Single Video (30 minutes)

**Goal**: Get 3D pose from your own video

```bash
conda activate mammal_stable

# Extract frames from video
mkdir -p data/raw/my_video/frames/
ffmpeg -i your_video.mp4 data/raw/my_video/frames/%06d.png

# Using shell script (recommended)
./run_mesh_fitting_monocular.sh data/raw/my_video/frames/ results/monocular/my_video yolo

# OR using Python directly
python fit_monocular.py \
  --input_dir data/raw/my_video/frames/ \
  --output_dir results/monocular/my_video \
  --detector yolo \
  --max_images 50
```

**Results**: `results/monocular/my_video/obj/` contains 3D meshes

### 3️⃣ Train Custom ML Detector (1 day)

**Goal**: Improve keypoint detection for your specific setup

**Step 1: Sample images** (5 min)
```bash
conda activate mammal_stable

python scripts/setup/sample_images_for_labeling.py \
  --input_dir data/raw/my_video/frames/ \
  --output_dir data/training/manual_labeling/images/ \
  --num_samples 20
```

**Step 2: Label on Roboflow** (2-3 hours)
1. Create account at https://roboflow.com
2. Create new "Keypoint Detection" project
3. Define 22 keypoints (see `docs/guides/ROBOFLOW_LABELING_GUIDE.md`)
4. Upload 20 images and label all keypoints
5. Export as "YOLOv8 Pose" format

**Step 3: Train YOLOv8** (30 min)
```bash
conda activate mammal_stable

# Merge manual labels with geometric labels
python preprocessing_utils/merge_datasets.py \
  --manual data/training/manual_labeling/ \
  --geometric data/training/yolo_mouse_pose/ \
  --output data/training/yolo_enhanced/

# Train YOLOv8
python scripts/train_yolo_pose.py \
  --data data/training/yolo_enhanced/data.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 256 \
  --name my_custom_detector
```

**Step 4: Use trained model**
```bash
python fit_monocular.py \
  --detector yolo \
  --yolo_weights models/trained/yolo/my_custom_detector/weights/best.pt
```

**Expected improvements**:
- Confidence: 0.5 → 0.85+ (2×)
- Loss: ~300K → 15-30K (10-20×)
- Paw detection: 0% → 70-80%

### 4️⃣ Batch Process Multiple Videos (customizable)

Process a directory of videos:

```bash
conda activate mammal_stable

# Create batch processing script
cat > batch_process.sh << 'EOF'
#!/bin/bash
for video in data/raw/batch/*.mp4; do
  name=$(basename "$video" .mp4)
  echo "Processing $name..."

  # Extract frames
  mkdir -p "data/raw/batch/${name}_frames/"
  ffmpeg -i "$video" "data/raw/batch/${name}_frames/%06d.png"

  # Run monocular fitting
  python fit_monocular.py \
    --input_dir "data/raw/batch/${name}_frames/" \
    --output_dir "results/monocular/${name}/" \
    --detector yolo \
    --max_images 100
done
EOF

chmod +x batch_process.sh
./batch_process.sh
```

---

## 📊 Understanding the Output

### Multi-View Fitting Output

After running `fitter_articulation.py`, outputs are in `results/fitting/{dataset}_{timestamp}/`:

```
results/fitting/markerless_mouse_1_nerf_20251125_143000/
├── obj/                           # 3D mesh files (can open in Blender/MeshLab)
│   ├── mesh_000000.obj            # Mesh for frame 0
│   ├── mesh_000002.obj
│   └── ...
│
├── params/                        # Fitting parameters (Python pickle)
│   ├── param0.pkl                 # Contains: body_pose, global_orient, betas, etc.
│   ├── param0_sil.pkl             # After silhouette refinement
│   └── ...
│
├── render/                        # Visualization overlays (if with_render=true)
│   ├── fitting_0.png              # Fitted model overlaid on all views
│   ├── fitting_0_sil.png          # After silhouette refinement
│   └── debug/                     # Optimization debug images
│
└── .hydra/                        # Hydra config snapshots
    └── config.yaml                # Exact config used for this run
```

**How to visualize**:
```bash
# View 3D mesh in Blender
blender results/fitting/*/obj/mesh_000000.obj

# View 3D mesh in MeshLab
meshlab results/fitting/*/obj/mesh_000000.obj

# View overlays
eog results/fitting/*/render/fitting_0.png
```

### Monocular Fitting Output

After running `fit_monocular.py`, outputs are in specified `--output_dir`:

```
results/monocular/my_experiment/
├── obj/                           # 3D mesh files
│   ├── frame_000001.obj
│   ├── frame_000002.obj
│   └── ...
│
├── params/                        # Fitting parameters
│   ├── frame_000001.pkl
│   └── ...
│
├── keypoints_2d/                  # Detected 2D keypoints
│   ├── frame_000001.pkl           # 22 keypoints [x, y, conf]
│   └── ...
│
├── camera_params.pkl              # Estimated camera intrinsics
│
└── visualizations/                # Overlays (if --visualize)
    ├── frame_000001.png           # Keypoints overlaid on image
    └── ...
```

**How to inspect keypoints**:
```python
import pickle
import numpy as np

# Load keypoints for frame 1
with open('results/monocular/my_experiment/keypoints_2d/frame_000001.pkl', 'rb') as f:
    kpts = pickle.load(f)

print(kpts.shape)  # (22, 3) -> [x, y, confidence]
# Note: GT annotation에서 nose는 index 2 (mouse_22_defs.py 기준)
# Model output에서 nose는 index 0 (keypoint22_mapper.json 기준)
print(f"Nose (model idx 0): x={kpts[0,0]:.1f}, y={kpts[0,1]:.1f}, conf={kpts[0,2]:.2f}")
```

**How to inspect fitted parameters**:
```python
import pickle

# Load fitted parameters
with open('results/monocular/my_experiment/params/frame_000001.pkl', 'rb') as f:
    params = pickle.load(f)

print(params.keys())
# dict_keys(['body_pose', 'global_orient', 'betas', 'transl'])

print(f"Body pose shape: {params['body_pose'].shape}")  # Limb rotations
print(f"Global orient shape: {params['global_orient'].shape}")  # Root orientation
print(f"Translation: {params['transl']}")  # 3D position
```

### Training Output

After running `scripts/train_yolo_pose.py`, training results are in `models/trained/yolo/`:

```
models/trained/yolo/my_custom_detector/
├── weights/
│   ├── best.pt                    # Best model checkpoint (use this!)
│   └── last.pt                    # Last epoch checkpoint
│
├── results.png                    # Training curves (loss, mAP, etc.)
├── confusion_matrix.png           # Confusion matrix
├── PR_curve.png                   # Precision-Recall curve
├── results.csv                    # Metrics per epoch
└── args.yaml                      # Training arguments
```

**How to evaluate**:
```bash
# View training curves
eog models/trained/yolo/my_custom_detector/results.png

# Check final metrics
tail -1 models/trained/yolo/my_custom_detector/results.csv
```

---

## ⚙️ Configuration Guide

### Hydra Configuration System

This project uses [Hydra](https://hydra.cc/) for flexible configuration. Config files are in `conf/`:

```
conf/
├── config.yaml              # Main config (don't edit directly)
├── dataset/                 # Dataset-specific configs
│   ├── markerless.yaml      # Multi-view (6 cameras)
│   ├── shank3.yaml          # Single-view
│   └── custom.yaml          # Template for your data
├── preprocess/              # Preprocessing configs
│   ├── opencv.yaml          # Current: geometric keypoints
│   └── sam.yaml             # Future: SAM-based masking
└── optim/                   # Optimization configs
    ├── fast.yaml            # Quick test (fewer iterations)
    └── accurate.yaml        # High quality (more iterations)
```

### Key Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `dataset` | Which dataset config to use | `shank3` | `dataset=markerless` |
| `optim` | Optimization settings | `fast` | `optim=accurate` |
| `mode` | Processing mode | `multi_view` | `mode=single_view_preprocess` |
| `data.data_dir` | Input data path | varies | `data.data_dir="data/preprocessed/custom/"` |
| `fitter.start_frame` | First frame | `0` | `fitter.start_frame=10` |
| `fitter.end_frame` | Last frame | `2` | `fitter.end_frame=100` |
| `fitter.with_render` | Enable rendering | `false` | `fitter.with_render=true` |
| `fitter.use_keypoints` | Enable keypoint loss | `true` | `fitter.use_keypoints=false` |
| `optim.solve_step0_iters` | Step 0 iterations | `10` | `optim.solve_step0_iters=20` |
| `optim.solve_step1_iters` | Step 1 iterations | `100` | `optim.solve_step1_iters=200` |
| `optim.solve_step2_iters` | Step 2 iterations | `30` | `optim.solve_step2_iters=50` |

### CLI 인자 ↔ Hydra 매핑

`fitter_articulation.py`는 argparse 스타일 인자를 자동으로 Hydra 형식으로 변환합니다:

| argparse 스타일 | Hydra 형식 |
|----------------|-----------|
| `--keypoints none` | `fitter.use_keypoints=false` |
| `--input_dir /path` | `data.data_dir=/path` |
| `--output_dir /path` | `result_folder=/path` |
| `--start_frame N` | `fitter.start_frame=N` |
| `--end_frame N` | `fitter.end_frame=N` |
| `--with_render` | `fitter.with_render=true` |

### 🆕 Manual Keypoint Annotation Workflow

For detailed mesh fitting with custom keypoint annotations:

**Quick workflow**:
```bash
# 1. Annotate keypoints (Gradio UI)
python keypoint_annotator_v2.py data/100-KO-male-56-20200615_cropped

# 2. Convert to MAMMAL format
python convert_keypoints_to_mammal.py \
  --input keypoints.json \
  --output data/.../keypoints2d_undist/result_view_0.pkl \
  --num-frames 20

# 3. Run mesh fitting
python fitter_articulation.py dataset=custom_cropped
```

**Key features**:
- ✅ **Flexible keypoint count**: 1-22 keypoints (recomm 5-7)
- ✅ **Auto-filtering**: Missing keypoints ignored automatically
- ✅ **Interactive UI**: Zoom, visibility control, progress tracking

📖 **Full guide**: [`KEYPOINT_QUICK_START.md`](KEYPOINT_QUICK_START.md) | [`docs/KEYPOINT_WORKFLOW.md`](docs/KEYPOINT_WORKFLOW.md)

---

### Usage Examples

```bash
# Use markerless dataset with accurate optimization
python fitter_articulation.py \
  dataset=markerless \
  optim=accurate

# Process frames 50-100 with rendering
python fitter_articulation.py \
  dataset=markerless \
  fitter.start_frame=50 \
  fitter.end_frame=100 \
  fitter.with_render=true

# Quick test on first 5 frames
python fitter_articulation.py \
  dataset=markerless \
  optim=fast \
  fitter.end_frame=5

# Override data directory
python fitter_articulation.py \
  data.data_dir="data/preprocessed/custom/" \
  fitter.end_frame=10
```

### Creating Custom Dataset Config

1. Copy template:
```bash
cp conf/dataset/custom.yaml conf/dataset/my_dataset.yaml
```

2. Edit `conf/dataset/my_dataset.yaml`:
```yaml
# @package _global_

data:
  data_dir: "data/preprocessed/my_dataset/"
  num_views: 1  # Single camera

fitter:
  start_frame: 0
  end_frame: 100

preprocess:
  input_video_path: "data/raw/my_dataset/video.mp4"
  output_data_dir: "data/preprocessed/my_dataset/"
```

3. Use it:
```bash
python fitter_articulation.py dataset=my_dataset
```

---

## 🔬 Advanced Usage

### Custom Keypoint Order

The default keypoint order is defined in `mouse_22_defs.py`:

```python
# 22 anatomical keypoints
KEYPOINT_NAMES = [
    'nose', 'left_ear', 'right_ear', 'left_eye', 'right_eye',
    'head_center', 'spine_1', 'spine_2', 'spine_3', 'spine_4',
    'spine_5', 'spine_6', 'spine_7', 'spine_8',
    'left_paw_front', 'right_paw_front',
    'left_paw_rear', 'right_paw_rear',
    'tail_base', 'tail_mid', 'tail_tip', 'centroid'
]
```

To use different keypoints, modify `mouse_22_defs.py` and update detection accordingly.

### Batch Processing with GNU Parallel

Process multiple datasets in parallel:

```bash
# Install GNU parallel
sudo apt-get install parallel

# Create experiment list
cat > experiments.txt << EOF
markerless 0 10
markerless 10 20
markerless 20 30
EOF

# Run in parallel (4 jobs)
parallel -j 4 --colsep ' ' \
  python fitter_articulation.py \
  dataset={1} \
  fitter.start_frame={2} \
  fitter.end_frame={3} \
  :::: experiments.txt
```

### Exporting Results

Convert 3D meshes to different formats:

```bash
# Convert OBJ to PLY
conda activate mammal_stable
pip install trimesh

python << EOF
import trimesh
import glob

for obj_file in glob.glob('results/fitting/*/obj/*.obj'):
    mesh = trimesh.load(obj_file)
    ply_file = obj_file.replace('.obj', '.ply')
    mesh.export(ply_file)
    print(f"Converted {obj_file} -> {ply_file}")
EOF
```

### Visualization with PyVista

Interactive 3D visualization:

```bash
conda activate mammal_stable
pip install pyvista

python << EOF
import pyvista as pv
import glob

# Load all meshes
meshes = []
for obj_file in sorted(glob.glob('results/fitting/*/obj/mesh_*.obj')):
    meshes.append(pv.read(obj_file))

# Create animation
plotter = pv.Plotter()
for mesh in meshes:
    plotter.add_mesh(mesh, color='tan')
    plotter.show(auto_close=False)
    plotter.clear()
EOF
```

---

## 🔧 Troubleshooting

### Installation Issues

**Problem**: `bash scripts/setup/setup.sh` fails
```bash
# Solution 1: Check conda is installed
conda --version

# Solution 2: Update conda
conda update -n base -c defaults conda

# Solution 3: Manual installation
conda create -n mammal_stable python=3.10 -y
conda activate mammal_stable
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

**Problem**: `CUDA out of memory`
```bash
# Solution: Reduce batch size or process fewer frames
python fitter_articulation.py fitter.end_frame=5  # Instead of 10
python fit_monocular.py --max_images 5  # Instead of 10
```

**Problem**: `ModuleNotFoundError: No module named 'pytorch3d'`
```bash
# Solution: Reinstall pytorch3d
conda activate mammal_stable
pip uninstall pytorch3d -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Data Issues

**Problem**: `FileNotFoundError: new_cam.pkl not found`
```bash
# Solution: Run preprocessing first
python scripts/preprocess.py dataset=custom mode=single_view_preprocess

# Or use monocular fitting (no preprocessing needed)
python fit_monocular.py --input_dir frames/
```

**Problem**: Poor keypoint quality
```bash
# Solution 1: Use YOLO instead of geometric
python fit_monocular.py --detector yolo

# Solution 2: Train custom detector (see Usage Scenario 3)
# Solution 3: Manually inspect and fix
python preprocessing_utils/visualize_yolo_labels.py --images frames/ --labels labels/
```

**Problem**: Camera calibration fails
```bash
# Solution: Provide known camera parameters
# Edit conf/dataset/my_dataset.yaml:
camera:
  fx: 1000.0  # Focal length X
  fy: 1000.0  # Focal length Y
  cx: 640.0   # Principal point X
  cy: 360.0   # Principal point Y
```

### Fitting Issues

**Problem**: Model converges to wrong pose
```bash
# Solution 1: Use more iterations
python fitter_articulation.py optim=accurate

# Solution 2: Start from clearer frame
python fitter_articulation.py fitter.start_frame=10

# Solution 3: Check keypoint quality
# Inspect: data/preprocessed/*/keypoints2d_undist/result_view_0.pkl
```

**Problem**: Rendering produces black images
```bash
# Solution 1: Disable rendering during debugging
python fitter_articulation.py fitter.with_render=false

# Solution 2: Check EGL libraries
ldconfig -p | grep EGL

# Solution 3: Use CPU rendering (slower)
export PYOPENGL_PLATFORM=osmesa
```

**Problem**: Very slow processing
```bash
# Solution 1: Disable rendering
python fitter_articulation.py fitter.with_render=false

# Solution 2: Use fast optimization
python fitter_articulation.py optim=fast

# Solution 3: Process fewer frames
python fitter_articulation.py fitter.end_frame=10
```

### ML Training Issues

**Problem**: YOLOv8 training fails
```bash
# Check dataset format
python << EOF
import yaml
with open('data/training/yolo_enhanced/data.yaml') as f:
    config = yaml.safe_load(f)
    print(config)
# Should contain: train, val, nc (22), names (list of 22 keypoints)
EOF

# Verify images and labels match
ls data/training/yolo_enhanced/train/images/ | wc -l
ls data/training/yolo_enhanced/train/labels/ | wc -l
# Should be equal
```

**Problem**: Low mAP after training
```bash
# Solution 1: More labeled data (add 10-20 more images)
# Solution 2: More training epochs
python scripts/train_yolo_pose.py --epochs 200

# Solution 3: Data augmentation
python scripts/train_yolo_pose.py --augment
```

---

## 📈 Performance Benchmarks

### Processing Time (NVIDIA RTX 3090)

| Task | Frames | Time | FPS |
|------|--------|------|-----|
| Monocular fitting (geometric) | 10 | 5 min | 0.033 |
| Monocular fitting (YOLO) | 10 | 10 min | 0.017 |
| Multi-view fitting (no render) | 10 | 25 min | 0.007 |
| Multi-view fitting (with render) | 10 | 70 min | 0.002 |
| YOLOv8 training (100 epochs) | - | 30 min | - |
| Preprocessing (OpenCV) | 100 | 1 min | 1.67 |

### Memory Usage

| Task | GPU Memory | RAM |
|------|------------|-----|
| Monocular fitting | 3-4 GB | 8 GB |
| Multi-view fitting | 4-6 GB | 16 GB |
| YOLOv8 training | 4-5 GB | 8 GB |
| Preprocessing | 2 GB | 4 GB |

### Recommendations

- **Quick testing**: Use `optim=fast`, `fitter.end_frame=5`, `fitter.with_render=false`
- **Production quality**: Use `optim=accurate`, trained YOLO detector, `fitter.with_render=true`
- **Long videos**: Process in batches of 100 frames
- **Limited GPU**: Reduce batch size, use geometric detector

---

## 🎯 Mesh Fitting with Multiple Datasets

This project supports flexible mesh fitting across different dataset formats. See the comprehensive guide for details.

### Quick Reference

**Run with default dataset (multi-view):**
```bash
./run_mesh_fitting_default.sh 0 50     # frames 0-50
./run_mesh_fitting_default.sh 0 10 1 true  # with render
```

**Run with monocular fitting (single-view):**
```bash
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output --keypoints none  # silhouette only
```

**Quick test (3 frames):**
```bash
./run_mesh_fitting_default.sh 0 3      # Multi-view test
python fit_monocular.py --input_dir data/test/ --output_dir results/test/ --max_images 3
```

### Supported Dataset Types

| Dataset | Location | Has Masks | Has Keypoints | Best Script |
|---------|----------|-----------|---------------|-------------|
| **Default Markerless** | `data/examples/markerless_mouse_1_nerf/` | ✅ | ✅ | `fitter_articulation.py` |
| **Single Images** | Any RGB+mask folder | ✅ | Optional | `fit_monocular.py` |
| **Cropped Frames** | `data/.../cropped/` | ✅ | Optional | `fit_monocular.py --keypoints none` |
| **Custom** | User-defined | Varies | Varies | Configurable |

### Configuration System

The project uses Hydra for hierarchical configuration. Available dataset configs:

- `default_markerless` - Reference multi-view dataset with 6 cameras
- `cropped` - Cropped frames with masks (single-view)
- `upsampled` - Upsampled frames (requires mask generation)
- `shank3` - Shank3 experiment dataset
- `custom` - Template for your custom data

**Override configuration from command line:**
```bash
python fitter_articulation.py \
  dataset=cropped \
  data.data_dir=/path/to/data \
  fitter.start_frame=0 \
  fitter.end_frame=100 \
  fitter.with_render=true
```

### Output Structure

```
results/fitting/{dataset}_{timestamp}/
├── obj/
│   ├── mesh_000000.obj           # 3D mesh per frame
│   └── ...
├── params/
│   ├── param0.pkl                # Fitted parameters
│   ├── param0_sil.pkl            # After silhouette refinement
│   └── ...
├── render/                       # (if with_render=true)
│   ├── fitting_0.png             # Visualization overlay
│   └── debug/                    # Optimization debug images
└── .hydra/
    └── config.yaml               # Configuration used
```

**Hydra logs** are stored in: `results/logs/YYYY-MM-DD/HH-MM-SS/`

### Documentation

- **[Mesh Fitting Guide](docs/MESH_FITTING_GUIDE.md)** - Complete workflow and troubleshooting
- **[Quick Cheatsheet](MESH_FITTING_CHEATSHEET.md)** - Command reference

---

## 📚 Documentation

### Complete Guides
- **[Mesh Fitting Guide](docs/MESH_FITTING_GUIDE.md)** - Multi-dataset mesh fitting workflows
- **[Monocular Fitting Guide](docs/guides/MONOCULAR_FITTING_GUIDE.md)** - Detailed single-view workflow
- **[Comprehensive Usage Guide](docs/guides/COMPREHENSIVE_USAGE_GUIDE.md)** - All usage scenarios
- **[Roboflow Labeling Guide](docs/ROBOFLOW_LABELING_GUIDE.md)** - Manual labeling tutorial
- **[SAM Mask Acquisition](docs/guides/SAM_MASK_ACQUISITION_MANUAL.md)** - High-quality masks

### Quick Reference
- **[Mesh Fitting Cheatsheet](MESH_FITTING_CHEATSHEET.md)** - Command quick reference

### Technical Reports
- **[ML Keypoint Detection](docs/reports/251115_comprehensive_ml_keypoint_summary.md)** - Complete ML workflow
- **[Implementation Summary](docs/reports/251114_ml_keypoint_detection_integration.md)** - Technical details
- **[All Reports](docs/reports/)** - Research session summaries

---

## 🎓 Key Concepts

### Three-Step Optimization

The fitting uses a progressive optimization strategy:

1. **Step 0: Global Initialization** (10 iters)
   - Objective: Find initial 3D pose
   - Uses: 2D keypoint reprojection only
   - Fast and robust to initialization

2. **Step 1: Joint Optimization** (100 iters)
   - Objective: Refine pose with all views
   - Uses: 2D keypoints + temporal smoothness
   - Main fitting stage

3. **Step 2: Silhouette Refinement** (30 iters)
   - Objective: Fine-tune surface details
   - Uses: Silhouette masks + PyTorch3D rendering
   - Highest quality but slower

### Keypoint Detectors

**Geometric** (Baseline):
- Extracts keypoints from silhouette contours
- Fast but low accuracy (~50% confidence)
- Good for: Quick testing, clean backgrounds

**YOLOv8-Pose** (Recommended):
- Pretrained on COCO, fine-tunable on your data
- Medium accuracy (~70-80% with fine-tuning)
- Good for: Most use cases

**SuperAnimal-TopViewMouse** (Future):
- Pretrained on 5K+ mice, highest accuracy
- Currently limited by API constraints
- Good for: Research applications when available

### Camera Models

**Single-View (Monocular)**:
- Estimates intrinsics from first frame
- Assumes: Known mouse size (~3cm body length)
- Limitations: Scale ambiguity, depth uncertainty

**Multi-View**:
- Uses calibrated camera parameters
- Assumes: Synchronized cameras, known calibration
- Advantages: Full 3D reconstruction, no scale ambiguity

---

## 🆕 Recent Updates

### 2025-11-26: CLI 일관성 개선
- ✅ `fitter_articulation.py`에 argparse 스타일 CLI 호환성 추가
- ✅ `--keypoints none`, `--input_dir`, `--output_dir` 등 fit_monocular.py와 동일한 인터페이스
- ✅ `fitter.use_keypoints` 설정 옵션 추가 (keypoint loss 비활성화)
- ✅ `fit_cropped_frames.py` deprecated로 이동 (fit_monocular.py로 통합)
- ✅ README 업데이트: CLI 매핑 테이블, 사용법 통일

### 2025-11-25: Folder Organization and Monocular Pipeline
- ✅ Consolidated result folders to unified `results/` structure
- ✅ Added monocular fitting shell script (`run_mesh_fitting_monocular.sh`)
- ✅ Created monocular config (`conf/monocular.yaml`)
- ✅ Enhanced visualization with keypoint overlay
- ✅ Added keypoint selection by groups (head, spine, limbs, tail)
- ✅ Cleaned up git-tracked large files (2.4GB → 6.5MB)
- ✅ Updated all output paths in codebase

### 2025-11-15: Major Cleanup and Documentation
- ✅ Reorganized project structure (36 → 21 root items)
- ✅ Created comprehensive README with step-by-step examples
- ✅ Moved all scripts to `scripts/` directory
- ✅ Cleaned 410MB of archived outputs
- ✅ Updated all documentation paths

### 2025-11-14: ML Integration
- ✅ Monocular fitting pipeline (`fit_monocular.py`)
- ✅ YOLOv8-Pose integration
- ✅ SuperAnimal-TopViewMouse support
- ✅ Manual labeling workflow

### 2025-11-03: Preprocessing Improvements
- ✅ OpenCV-based preprocessing
- ✅ Geometric keypoint estimation
- ✅ SAM mask acquisition (experimental)

---

## 📊 Comparison with DANNCE

![comparison](assets/figs/mouse_2.png)

Results comparing DANNCE-T (temporal version) with MAMMAL_mouse on `markerless_mouse_1` sequence.

**MAMMAL_mouse advantages**:
- Full 3D mesh reconstruction (not just keypoints)
- Articulated model enforces anatomical constraints
- Compatible with single-view videos

**DANNCE advantages**:
- Faster processing
- Simpler setup (no model fitting)
- More robust to occlusions

---

## 📁 Project Structure

```
MAMMAL_mouse/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── # Core Python Files
├── fitter_articulation.py         # Main multi-view mesh fitter (Hydra + argparse 지원)
├── fit_monocular.py               # Single-view monocular fitting (argparse)
├── articulation_th.py             # Articulation model (PyTorch)
├── bodymodel_th.py                # Body model (PyTorch)
├── bodymodel_np.py                # Body model (NumPy)
├── mouse_22_defs.py               # 22 keypoint definitions
├── utils.py                       # Utility functions
│
├── # Shell Scripts (Quick Start)
├── run_mesh_fitting_default.sh    # Multi-view fitting
├── run_mesh_fitting_monocular.sh  # Monocular fitting
├── run_unified_annotator.sh       # Launch annotation tool
│
├── # Configuration
├── conf/                          # Hydra configs
│   ├── config.yaml                # Main config
│   ├── monocular.yaml             # Monocular fitting config
│   └── dataset/                   # Dataset-specific configs
│       ├── default_markerless.yaml
│       ├── cropped.yaml
│       └── custom.yaml
│
├── # Scripts (Organized)
├── scripts/
│   ├── annotators/                # Annotation tools
│   │   ├── unified_annotator.py   # Mask + Keypoint tool (Gradio)
│   │   └── keypoint_annotator_v2.py
│   ├── preprocessing/             # Video preprocessing
│   │   └── extract_video_frames.py
│   ├── setup/                     # Installation scripts
│   │   ├── setup.sh
│   │   ├── download_superanimal.py
│   │   └── sample_images_for_labeling.py
│   ├── utils/                     # Utility scripts
│   │   ├── convert_keypoints_to_mammal.py
│   │   └── process_video_with_sam.py
│   ├── tests/                     # Test scripts
│   ├── deprecated/                # Old/replaced scripts
│   ├── preprocess.py
│   ├── evaluate.py
│   └── train_yolo_pose.py
│
├── # Preprocessing Utilities
├── preprocessing_utils/
│   ├── keypoint_estimation.py     # Geometric keypoint detector
│   ├── yolo_keypoint_detector.py  # YOLO-Pose detector
│   ├── superanimal_detector.py    # SuperAnimal detector
│   ├── mask_processing.py         # Mask utilities
│   ├── sam_inference.py           # SAM integration
│   └── silhouette_renderer.py     # PyTorch3D rendering
│
├── # Assets (tracked)
├── mouse_model/                   # MAMMAL parametric model
│   ├── mouse.pkl                  # Main model file
│   └── mouse_txt/                 # Auxiliary files
│
├── # Documentation
├── docs/
│   ├── guides/                    # Usage guides
│   └── reports/                   # Research notes (YYMMDD_*.md)
│
├── # Models (git-ignored, download separately)
├── models/
│   ├── README.md                  # Download instructions
│   ├── pretrained/                # SAM, YOLO base models
│   └── trained/                   # Fine-tuned models
│
├── # Data (git-ignored)
├── data/                          # Input datasets
│
└── # Results (git-ignored)
└── results/
    ├── fitting/                   # Mesh fitting outputs
    ├── monocular/                 # Monocular fitting outputs
    └── logs/                      # Hydra logs
```

---

## 📧 Support

### Getting Help

1. **Check documentation**:
   - This README
   - `docs/guides/` for detailed tutorials
   - `docs/reports/` for technical details

2. **Common issues**:
   - See Troubleshooting section above
   - Check existing GitHub issues

3. **Report bugs**:
   - Open GitHub issue with:
     - Error message and full traceback
     - Your environment: `conda list > environment.txt`
     - Config file used
     - Minimal reproducible example

### Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests if applicable
4. Update documentation
5. Submit pull request

---

## 🙏 Acknowledgments

- **MAMMAL framework**: An et al. (2023)
- **Virtual mouse model**: Bolanos et al. (2021)
- **DANNCE dataset**: Dunn et al. (2021)
- **PyTorch3D**: Meta AI Research
- **YOLOv8**: Ultralytics
- **SuperAnimal**: Mathis Lab

---

## 🧪 Ablation Study Experiments

체계적인 ablation study를 위한 실험 가이드입니다.

### 실험 스크립트 사용법

```bash
# 사용 가능한 실험 목록 보기
./run_experiment.sh

# 디버그 모드 (2 frames, 빠른 테스트)
./run_experiment.sh <experiment_name> --debug

# 전체 실행
./run_experiment.sh <experiment_name>

# 커스텀 프레임 수
./run_experiment.sh <experiment_name> --frames 50
```

### 실험 그룹

#### Group 1: Baseline (Paper Reference)

| Experiment | Views | Keypoints | Description |
|------------|-------|-----------|-------------|
| `baseline_6view_keypoint` | 6 | 22 (full) | MAMMAL 논문 기본 설정 |

```bash
./run_experiment.sh baseline_6view_keypoint --debug   # 테스트
./run_experiment.sh baseline_6view_keypoint           # 전체 실행
```

#### Group 2: Keypoint Ablation (6-view 고정)

| Experiment | Views | Keypoints | Description |
|------------|-------|-----------|-------------|
| `baseline_6view_keypoint` | 6 | 22 | Full keypoints |
| `sixview_sparse_keypoint` | 6 | 3 | Sparse (nose, neck, tail) |
| `sixview_no_keypoint` | 6 | 0 | Silhouette only |

```bash
# 전체 그룹 실행
for exp in baseline_6view_keypoint sixview_sparse_keypoint sixview_no_keypoint; do
    ./run_experiment.sh $exp --debug
done
```

#### Group 3: Viewpoint Ablation (Sparse 3 keypoints 고정)

| Experiment | Views | Cameras | Description |
|------------|-------|---------|-------------|
| `sixview_sparse_keypoint` | 6 | 0,1,2,3,4,5 | Reference |
| `sparse_5view` | 5 | 0,1,2,3,4 | Drop camera 5 |
| `sparse_4view` | 4 | 0,1,2,3 | 4 consecutive |
| `sparse_3view` | 3 | 0,2,4 | Diagonal (better coverage) |
| `sparse_2view` | 2 | 0,3 | Opposite (stereo-like) |

```bash
# 전체 그룹 실행
for exp in sixview_sparse_keypoint sparse_5view sparse_4view sparse_3view sparse_2view; do
    ./run_experiment.sh $exp --debug
done
```

### Sparse Keypoint 설정

**중요**: GT annotation과 Model definition의 Head keypoint 순서가 다릅니다!

| Index | GT (mouse_22_defs.py) | Model (keypoint22_mapper.json) |
|-------|----------------------|-------------------------------|
| 0 | left_ear | nose |
| 1 | right_ear | left_ear |
| 2 | **nose** | right_ear |
| 3+ | 동일 | 동일 |

실제 데이터는 GT 정의를 따르므로 sparse indices는 `[2, 5, 3]` (nose, tail_root, neck)입니다.

자세한 keypoint 정보는 `docs/KEYPOINT_REFERENCE.md` 참조.

### 결과 비교

```bash
# 결과 디렉토리 구조
results/fitting/
├── markerless_mouse_1_nerf_v012345_kp22_*/    # Baseline
├── markerless_mouse_1_nerf_v012345_sparse3_*/ # 6view sparse
├── markerless_mouse_1_nerf_v012345_noKP_*/    # 6view no keypoint
├── markerless_mouse_1_nerf_v01234_sparse3_*/  # 5view sparse
└── ...

# 결과 시각화 비교
ls results/fitting/*/render/fitting_*.png
```

### Debug vs Full 실행 비교

| Mode | Frames | Step0 | Step1 | Step2 | 예상 시간 |
|------|--------|-------|-------|-------|----------|
| Debug | 2 | 5 | 20 | 10 | ~1분 |
| Full | 100 | 10-20 | 100-180 | 30-50 | ~30분 |

---

## 📄 License

[Specify your license here]

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{MAMMAL,
    author = {An, Liang and Ren, Jilong and Yu, Tao and Hai, Tang and Jia, Yichang and Liu, Yebin},
    title = {Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
    journal = {},
    year = {2023}
}

@article{bolanos2021three,
  title={A three-dimensional virtual mouse generates synthetic training data for behavioral analysis},
  author={Bola{\~n}os, Luis A and Xiao, Dongsheng and Ford, Nancy L and LeDue, Jeff M and Gupta, Pankaj K and Doebeli, Carlos and Hu, Hao and Rhodin, Helge and Murphy, Timothy H},
  journal={Nature methods},
  volume={18},
  number={4},
  pages={378--381},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```
