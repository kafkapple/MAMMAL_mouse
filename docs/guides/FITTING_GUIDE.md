# Mesh Fitting Guide

MAMMAL Mouse 프로젝트의 3D mesh fitting 통합 가이드입니다.
Multi-view, monocular, silhouette-only 세 가지 방식의 피팅을 다룹니다.

---

## Quick Start

### 1. Multi-view Fitting (6카메라, keypoints + masks)

```bash
cd /home/joon/dev/MAMMAL_mouse
conda activate mammal_stable

python fitter_articulation.py \
    dataset=default_markerless \
    fitter.start_frame=0 \
    fitter.end_frame=3 \
    fitter.with_render=true
```

### 2. Monocular Fitting (단일 RGB + mask -> 3D mesh)

```bash
python fit_monocular.py \
    --input_dir /path/to/images \
    --output_dir outputs/monocular_result \
    --max_images 10 \
    --device cuda
```

### 3. Single-view Silhouette Fitting (크롭 이미지 + mask)

```bash
python fit_cropped_frames.py \
    data/100-KO-male-56-20200615_cropped \
    --output-dir results/cropped_fitting \
    --max-frames 10
```

---

## Multi-view Fitting

### 데이터셋 구조

**기본 데이터셋 위치**: `data/examples/markerless_mouse_1_nerf/`

```
markerless_mouse_1_nerf/
├── videos_undist/           # 6개 카메라 비디오
│   ├── 0.mp4
│   └── ...
├── keypoints2d_undist/      # 2D keypoints (pkl)
│   ├── result_view_0.pkl
│   └── ...
├── simpleclick_undist/      # segmentation mask 비디오
│   └── *.mp4
├── new_cam.pkl              # 6개 카메라 캘리브레이션
└── add_labels_3d_8keypoints.pkl  # 3D GT (optional)
```

**특징**:
- 6개 동기화된 카메라 뷰
- Multi-view 카메라 캘리브레이션
- 뷰별 2D keypoint annotation
- 3D keypoint annotation
- Segmentation masks

### Hydra 설정 시스템

프로젝트는 Hydra를 사용한 계층적 설정 관리 방식을 사용합니다.

**Config 디렉토리 구조**:
```
conf/
├── config.yaml                  # 메인 설정
├── dataset/
│   ├── default_markerless.yaml  # 기본 multi-view 데이터셋
│   ├── cropped.yaml             # 크롭 프레임 (mask 포함)
│   ├── upsampled.yaml           # 업샘플링 프레임
│   ├── shank3.yaml              # Shank3 데이터셋
│   └── custom.yaml              # 사용자 정의 템플릿
└── optim/
    ├── default.yaml             # 기본 (느림)
    ├── fast.yaml                # 빠른 테스트
    ├── paper.yaml               # 논문 설정 + 렌더링
    └── paper_fast.yaml          # 논문 설정 + 최고속
```

**주요 설정 파라미터**:

| 파라미터 | 설명 | 기본값 | 예시 |
|----------|------|--------|------|
| `data.data_dir` | 데이터셋 루트 경로 | (varies) | `/path/to/data` |
| `fitter.start_frame` | 처리 시작 프레임 | 0 | 0 |
| `fitter.end_frame` | 처리 종료 프레임 | (varies) | 100 |
| `fitter.interval` | 프레임 간격 | 1 | 5 (매 5번째) |
| `fitter.with_render` | 시각화 활성화 | false | true |
| `fitter.keypoint_num` | keypoint 수 | 22 | 22 |
| `fitter.resume` | 체크포인트에서 재개 | false | true |
| `result_folder` | 출력 디렉토리 | `mouse_fitting_result/results/` | `results/exp1/` |

**커맨드라인 오버라이드**:
```bash
# 단일 파라미터 오버라이드
python fitter_articulation.py data.data_dir=/path/to/data

# 다중 오버라이드
python fitter_articulation.py \
    dataset=cropped \
    data.data_dir=/custom/path \
    fitter.with_render=true \
    optim=paper_fast \
    result_folder=results/my_experiment/
```

### 실행 명령어

```bash
# Shell 스크립트 (인자: start_frame, end_frame, interval, with_render)
./run_mesh_fitting_default.sh 0 50 1 true

# 빠른 테스트 (3 프레임)
./run_quick_test.sh default_markerless

# 논문 설정 + 전체 처리 (~10시간) - 권장
nohup ./run_experiment.sh baseline_6view_keypoint \
    frames=aligned_posesplatter optim=paper_fast \
    > logs/fitting_paper_fast_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 테스트용 (100 프레임)
./run_experiment.sh baseline_6view_keypoint \
    frames=aligned_test_100 optim=paper_fast
```

**커스텀 데이터셋 설정 파일 작성** (`conf/dataset/my_mouse.yaml`):
```yaml
# @package _global_

data:
  data_dir: /home/joon/dev/MAMMAL_mouse/data/my_mouse_experiment/
  views_to_use: [0]

fitter:
  start_frame: 0
  end_frame: 500
  interval: 5
  render_cameras: [0]
  with_render: true
```

```bash
python fitter_articulation.py dataset=my_mouse
```

### 출력 구조

**Hydra 출력 디렉토리** (기본):
```
outputs/YYYY-MM-DD/HH-MM-SS/
├── .hydra/
│   ├── config.yaml       # 사용된 설정
│   ├── overrides.yaml    # 커맨드라인 오버라이드
│   └── hydra.yaml
├── fitter_articulation.log
└── results/
```

**Multi-view fitting 결과**:
```
mouse_fitting_result/results_markerless_mouse_1_nerf_YYYYMMDD_HHMMSS/
├── obj/
│   ├── mesh_000000.obj      # 3D mesh (Blender/MeshLab 로드 가능)
│   └── ...
├── params/
│   ├── param0.pkl           # 최적화 파라미터
│   └── param0_sil.pkl       # Silhouette 단계 파라미터
└── render/
    ├── debug/               # 최적화 과정 시각화
    │   └── fitting_0_global_iter_*.png
    └── fitting_keypoints_*.png  # keypoint 비교
```

**파라미터 형식** (param0.pkl):
- `thetas`: 관절 각도 (1, 140, 3)
- `bone_lengths`: 뼈대 길이 (1, 28)
- `rotation`: 전역 회전 (1, 3)
- `trans`: 전역 이동 (1, 3)
- `scale`: 전역 스케일 (1, 1)
- `chest_deformer`: 흉부 변형 (1, 1)

### 피팅 재시작 (Resume)

중단된 작업을 이어서 처리할 수 있습니다:

```bash
./run_experiment.sh <experiment> --resume_from results/fitting/<result_folder>
```

**동작 방식**:
1. `params/step_2_frame_*.pkl` 스캔하여 마지막 완료 프레임 감지
2. 다음 프레임부터 자동 시작 + 이전 파라미터 로드 (`resume=True`)

---

## Monocular Fitting

### 개요

단일 RGB 이미지와 binary mask로부터 3D mouse mesh를 재구성합니다.

| 항목 | 값 |
|------|-----|
| 처리 시간 | ~21초/이미지 (GPU) |
| 출력 mesh | 14,522 vertices, 28,800 faces |
| Optimizer | Adam (lr=0.01, 50 iterations) |
| 상태 | PoC 완료, Production Ready |

### 시스템 아키텍처

```
Input: RGB Image + Binary Mask
    |
[Geometric Keypoint Estimation] (PCA 기반, 22 keypoints)
    |
[MAMMAL Parameter Initialization] (T-pose, default scale)
    |
[Optimization Loop] (50 iterations, Adam optimizer)
    |- Forward: ArticulationTorch -> 3D mesh + 22 keypoints
    |- Loss: 2D reprojection + pose regularization
    |- Backward: Gradient descent on thetas, T, s
    |
Output: 3D Mesh (.obj) + Parameters (.pkl) + Visualization (.png)
```

### 파이프라인 상세

**Step 1: Keypoint Estimation (PCA 기반)**

Binary mask에서 contour를 추출하고, PCA로 body axis를 결정한 후 22개의 anatomical keypoints를 추정합니다.

- Head (0-5): nose, ears, eyes, head center
- Spine (6-13): 8 points along backbone
- Limbs (14-17): 4 paws
- Tail (18-20): tail base, mid, tip
- Centroid (21): body center

Confidence 점수는 geometric reliability에 기반 (0.35~0.95).

**Step 2: Parameter Initialization**
- `thetas`: Zero (T-pose)
- `bone_lengths`: Zero (기본 뼈대 구조)
- `T`: Keypoint centroid 중심
- `s`: Keypoint 분포에서 추정

**Step 3: Optimization**
```python
loss = loss_2d + loss_pose_reg
# loss_2d: weighted L2 distance (predicted vs target 2D keypoints)
# loss_pose_reg: L2 regularization on joint angles (weight=0.001)
```

### 입력 요구사항

파일 명명 규칙:
- `<frame_id>_rgb.png` - RGB 이미지
- `<frame_id>_mask.png` - Binary mask (0=background, 255=mouse)

### 실행 방법

```bash
# 단일 이미지 처리
python fit_monocular.py \
    --input_dir /path/to/images \
    --output_dir outputs/monocular_result \
    --max_images 1 \
    --device cuda

# 배치 처리
python fit_monocular.py \
    --input_dir /home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view/train/000000_00000 \
    --output_dir outputs/monocular_poc_batch \
    --device cuda
```

**Python API 사용**:
```python
from fit_monocular import MonocularMAMMALFitter

fitter = MonocularMAMMALFitter(device="cuda")
results = fitter.fit_single_image(rgb_path="image.png", mask_path="mask.png")
mesh = results["mesh"]  # trimesh object
mesh.export("output.obj")
```

### 출력 파일

| 파일 | 설명 | 크기 |
|------|------|------|
| `<frame_id>_mesh.obj` | 3D mesh (14,522 vertices) | ~1.1 MB |
| `<frame_id>_params.pkl` | MAMMAL 파라미터 | ~2.5 KB |
| `<frame_id>_keypoints.png` | Keypoint 시각화 | ~84 KB |

### 한계점

1. **Keypoint 정확도**: Geometric estimation은 근사치 (learned prior 없음)
2. **Single-view ambiguity**: 깊이 정보 부재, 좌우 대칭 모호성
3. **T-pose bias**: Regularization으로 T-pose 근처에 수렴하는 경향
4. **처리 속도**: 실시간 아님 (21초/이미지)

---

## Silhouette-Only Fitting

Keypoint annotation 없이 mask silhouette만으로 mesh fitting을 수행합니다.

### 파라미터 튜닝

| 파라미터 | 기본값 | 권장 범위 | 효과 |
|----------|--------|----------|------|
| `iter_multiplier` | 2.0 | 1.0~5.0 | 최적화 반복 횟수 배율 |
| `theta_weight` | 10.0 | 5.0~30.0 | 포즈 정규화 강도 (keypoint 모드: 3.0) |
| `bone_weight` | 2.0 | 0.5~5.0 | 뼈대 길이 정규화 (keypoint 모드: 0.5) |
| `scale_weight` | 50.0 | 10.0~100.0 | 스케일 정규화 (keypoint 모드: 0.5) |
| `use_pca_init` | true | true/false | PCA 기반 회전 초기화 |

**`iter_multiplier`** -- 디버그: 1.0~2.0, 일반: 2.0~3.0, 고품질: 3.0~5.0

**`theta_weight`** -- 일반 행동: 10.0~15.0, 정적: 15.0~20.0, 활발: 5.0~10.0. Mask만 사용 시 keypoint 모드보다 높은 값 필요.

**`scale_weight`** -- Silhouette 모드에서 30.0 이상 필수. 낮으면 mesh collapse 발생.

**`use_pca_init`** -- 첫 프레임 mask contour의 PCA로 초기 회전 추정.

### Ablation 설계

```bash
# 단일 변수
for mult in 1.0 2.0 3.0 4.0 5.0; do
    python fitter_articulation.py ... silhouette.iter_multiplier=$mult
done

# Grid search
for mult in 2.0 3.0; do
    for theta in 10.0 15.0 20.0; do
        python fitter_articulation.py \
            fitter.end_frame=50 \
            silhouette.iter_multiplier=$mult \
            silhouette.theta_weight=$theta
    done
done
```

---

## 최적화 설정

### 논문 핵심 설정

> An, L., et al. (2023). "Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL." *Nature Communications*, 14(1), 7727.

| 단계 | 논문 권장 | 기본값 | 비고 |
|------|----------|--------|------|
| step0 (초기화) | 60 | 10 | 첫 프레임만 |
| step1 (tracking) | **3-5** | 100 | T > 0 |
| step2 (refinement) | **3** | 30 | |

속도: Detection 50ms, Matching 0.15ms, **Mesh Fitting 1.2~2초/frame** (GPU)

### Config 프로필 비교

| Config | step0 | step1 | step2 | render | 용도 |
|--------|-------|-------|-------|--------|------|
| `default` | 10 | 100 | 30 | Yes | 기본 (느림) |
| `fast` | 10 | 50 | 15 | Yes | 빠른 테스트 |
| `paper` | 60 | 5 | 3 | Yes | 논문 설정 + 렌더링 |
| **`paper_fast`** | **60** | **5** | **3** | **No** | **논문 설정 + 최고속** |

### 예상 처리 시간

| Config | 100 프레임 | 3,600 프레임 |
|--------|-----------|-------------|
| `default` | ~20시간 | ~31일 |
| `paper` | ~1.4시간 | ~2일 |
| **`paper_fast`** | **~15분** | **~10시간** |

### Loss Weight 커스터마이징

```python
self.term_weights = {
    "theta": 3, "3d": 2.5, "2d": 0.2, "bone": 0.5,
    "scale": 0.5, "mask": 0, "chest_deformer": 0.1,
    "stretch": 1, "temp": 0.25, "temp_d": 0.2
}
```

### 피팅 완료 후 시각화

```bash
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --start_frame 0 --end_frame 1 \
    --save_video --no_rrd
```

---

## 스크립트 비교

| 항목 | `fitter_articulation.py` | `fit_cropped_frames.py` |
|------|--------------------------|-------------------------|
| **용도** | Multi-view 3D fitting | Single-view silhouette fitting |
| **입력** | 6카메라 비디오 + keypoints | SAM 크롭 이미지 + mask |
| **설정** | Hydra config | CLI arguments |
| **출력** | `mouse_fitting_result/` | `results/` (지정 가능) |
| **카메라** | 캘리브레이션 필요 (`new_cam.pkl`) | 불필요 |

### 결과 확인

```python
import trimesh, pickle
mesh = trimesh.load("mouse_fitting_result/*/obj/mesh_000000.obj")
print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

with open("mouse_fitting_result/*/params/param0.pkl", "rb") as f:
    params = pickle.load(f)
print("Keys:", params.keys())
```

---

## 데이터셋 Quick Reference

| 데이터셋 | 위치 | Masks | Keypoints | 스크립트 | Config |
|---------|------|-------|-----------|---------|--------|
| **Default Markerless** | `data/examples/markerless_mouse_1_nerf/` | Yes | Yes | `fitter_articulation.py` | `default_markerless` |
| **Cropped** | `data/100-KO-male-56-20200615_cropped/` | Yes | Optional | `fit_cropped_frames.py` | `cropped` |
| **Upsampled** | `data/100-KO-male-56-20200615_upsampled/` | No | No | 전처리 필요 | `upsampled` |
| **Fauna (Monocular)** | (외부 데이터) | Yes | No | `fit_monocular.py` | - |

### 방법론 비교

| 방법 | 입력 | 속도 | 품질 |
|------|------|------|------|
| **Multi-view (MAMMAL)** | Multi-view 비디오 | 1.2-2s/frame | Best |
| **Monocular (MAMMAL)** | 단일 RGB+mask | 21s/image | Good |
| **Silhouette Fitting** | 크롭+mask | (varies) | Good |
| **DANNCE + MAMMAL** | Multi-view | Medium | Best |

---

## Troubleshooting

### No Masks Found
Mask 파일이 같은 디렉토리에 `frame_{idx:06d}_mask.png` 형식으로 있는지 확인.

### CUDA Out of Memory
`--max-frames 10` / `fitter.end_frame=10` / `fitter.with_render=false` / `--device cpu`

### Camera Calibration Not Found
Multi-view 전용 파일. Single-view는 `dataset=cropped` 사용.

### Configuration Override Not Working
```bash
python fitter_articulation.py data.data_dir=/path/to/data  # 올바름 (dot notation)
python fitter_articulation.py data_dir=/path/to/data        # 잘못됨
```

### ModuleNotFoundError
```bash
cp scripts/analysis/data_seaker_video_new.py .
ln -s assets/colormaps colormaps
```

### Mesh Collapse (Silhouette)
`silhouette.scale_weight=100.0`

### 비현실적 포즈
`silhouette.theta_weight=20.0` + `silhouette.bone_weight=3.0`

### 느린 수렴
`silhouette.iter_multiplier=3.0` + `silhouette.use_pca_init=true`

---

## 기존 피팅 결과 현황

| 날짜 | interval | 완료 프레임 | 상태 |
|------|----------|-----------|------|
| 20251206 | 1 | 100 | interval 불일치 (pose-splatter=5) |
| 20251213 | 5 | 3 | 중단 |
| 20260118 | 5 | 7 | 중단 |

> **주의**: 20251206 결과는 interval=1로 피팅되어 pose-splatter (interval=5)와 프레임 인덱스 불일치.

---

## 관련 스크립트

| 스크립트 | 용도 |
|---------|------|
| `fitter_articulation.py` | 메인 fitting (multi-view) |
| `fit_cropped_frames.py` | Silhouette 기반 fitting (single-view) |
| `fit_monocular.py` | Monocular MAMMAL fitting |
| `extract_video_frames.py` | 비디오에서 프레임 추출 |
| `run_sam_gui.py` | SAM annotation GUI |
| `process_annotated_frames.py` | Annotation에서 크롭 프레임 생성 |
| `keypoint_annotator_v2.py` | 수동 keypoint annotation |

---

## 참고 문헌

1. **MAMMAL**: An, L., et al. (2023). *Nature Communications*, 14(1), 7727.
2. **Virtual Mouse**: Bolanos, L. A., et al. (2021). *Nature Methods*.
3. **DANNCE**: Dunn, T. W., et al. (2023). *Nature Methods*.

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [KEYPOINTS](../reference/KEYPOINTS.md) | 22 키포인트 정의 및 가중치 |
| [EXPERIMENTS](../reference/EXPERIMENTS.md) | 실험 config 및 batch 실행 |
| [OUTPUT_FORMAT](../reference/OUTPUT_FORMAT.md) | 피팅 결과 파일 형식 |
| [ARCHITECTURE](../reference/ARCHITECTURE.md) | 시스템 아키텍처 및 Hydra config |
| [QUICK_REFERENCE](../QUICK_REFERENCE.md) | 자주 쓰는 명령어 |

---

*Merged from: MESH_FITTING_GUIDE.md, MESH_FITTING_CHEATSHEET.md, FITTING_SCRIPTS_COMPARISON.md, MONOCULAR_FITTING_GUIDE.md, README_MONOCULAR.md, OPTIMIZATION_GUIDE.md, silhouette_parameters.md*
*Created: 2026-02-06*
