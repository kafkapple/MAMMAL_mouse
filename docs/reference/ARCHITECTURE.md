# MAMMAL Architecture Reference

> 시스템 아키텍처, Hydra config, 최적화 파이프라인, 시나리오별 사용법 통합 레퍼런스

---

## 시스템 개요

MAMMAL_mouse는 다중/단일 시점 비디오에서 생쥐의 3D 자세 추정 및 메쉬 재구성을 수행하는 최적화 기반 프레임워크이다.

### 핵심 컴포넌트

| 컴포넌트 | 설명 |
|----------|------|
| **MouseFitter** | 핵심 최적화 클래스 (L-BFGS optimizer) |
| **ArticulationTorch** | 관절 모델 (140 joints) |
| **BodyModelTorch** | 바디 모델 (14,500 vertices) |
| **MeshRenderer** | PyTorch3D 기반 미분 가능 렌더러 |
| **DataSeakerDet** | 데이터 로더 (multi-view 영상, keypoints, masks) |

### 전체 파이프라인

```
Input Data --> [Data Loader] --> [Fitter] --> Output Files

Input Data:  Multi-view 영상 + 2D Keypoints + Masks + Camera Params
Data Loader: DataSeakerDet (입력 데이터를 파이프라인 형식으로 가공)
Fitter:      MouseFitter (3단계 최적화)
Output:      .pkl (파라미터) + .obj (메쉬) + .png (렌더링)
```

---

## 프로젝트 구조

```
MAMMAL_mouse/
├── fitter_articulation.py        # 메인 multi-view fitter (Hydra + argparse)
├── fit_monocular.py              # 단일 뷰 monocular fitting
├── articulation_th.py            # 관절 모델 (PyTorch)
├── bodymodel_th.py               # 바디 모델 (PyTorch)
├── mouse_22_defs.py              # 22 keypoint 정의
├── utils.py                      # 유틸리티
│
├── conf/                         # Hydra 설정
│   ├── config.yaml               # 메인 설정
│   ├── dataset/                  # 데이터셋별 설정
│   ├── experiment/               # 실험별 설정
│   ├── optim/                    # 최적화 설정
│   └── frames/                   # 프레임 범위 설정
│
├── preprocessing_utils/          # 전처리 모듈
│   ├── keypoint_estimation.py    # 기하학적 키포인트 추정
│   ├── yolo_keypoint_detector.py # YOLO 검출기
│   ├── superanimal_detector.py   # SuperAnimal 검출기
│   ├── mask_processing.py        # 마스크 처리
│   └── sam_inference.py          # SAM 통합
│
├── scripts/                      # 유틸리티 스크립트
├── assets/mouse_model/           # 3D 마우스 모델
├── data/                         # 데이터셋
├── results/                      # 실험 결과
└── docs/                         # 문서
```

---

## Hydra Config 시스템

### 설정 파일 구조

```
conf/
├── config.yaml              # 메인 (defaults 목록)
├── dataset/                 # 데이터셋별 설정
│   ├── default_markerless.yaml  # 6-view 기본
│   ├── cropped.yaml             # Cropped 단일 뷰
│   ├── custom.yaml              # 템플릿
│   └── ...
├── experiment/              # 실험 프리셋
│   ├── quick_test.yaml          # 디버그 (5 frames, 최소 iters)
│   ├── views_6.yaml             # 6뷰 baseline
│   ├── views_4.yaml             # 4뷰
│   ├── silhouette_only_6views.yaml  # Mask만 사용
│   └── ...
├── optim/                   # 최적화 설정
│   ├── fast.yaml            # 빠른 테스트
│   ├── paper_fast.yaml      # 논문 설정 (wsil=0)
│   └── accurate.yaml        # 정밀 피팅
└── frames/                  # 프레임 범위
    ├── aligned_posesplatter.yaml   # 3,600 frames
    ├── aligned_test_100.yaml       # 100 frames
    └── quick_test_30.yaml          # 30 frames
```

### 설정 조합 예시

```bash
# 빠른 테스트
python fitter_articulation.py dataset=default_markerless optim=fast fitter.end_frame=5

# 논문 설정 정밀 피팅
python fitter_articulation.py dataset=default_markerless optim=paper_fast frames=aligned_posesplatter

# 커맨드라인 오버라이드
python fitter_articulation.py dataset=default_markerless \
    fitter.start_frame=10 fitter.end_frame=50 \
    optim.solve_step1_iters=200
```

### CLI 인자 vs Hydra 매핑

`fitter_articulation.py`는 argparse/Hydra 모두 지원한다:

| argparse 스타일 | Hydra 형식 |
|----------------|-----------|
| `--keypoints none` | `fitter.use_keypoints=false` |
| `--input_dir /path` | `data.data_dir=/path` |
| `--output_dir /path` | `result_folder=/path` |
| `--start_frame N` | `fitter.start_frame=N` |
| `--end_frame N` | `fitter.end_frame=N` |
| `--with_render` | `fitter.with_render=true` |

### Experiment Configs

| Config | Views | Keypoints | 설명 |
|--------|-------|-----------|------|
| `quick_test` | 6 | 22 | 5 frames, 최소 iterations (디버깅) |
| `views_6` | 6 | 22 | Full baseline (100 samples) |
| `views_5` | 5 | 22 | [0,1,2,3,4] |
| `views_4` | 4 | 22 | [0,1,2,3] |
| `views_3_diagonal` | 3 | 22 | [0,2,4] 대각선 배치 |
| `views_2_opposite` | 2 | 22 | [0,3] 반대편 |
| `views_1_single` | 1 | 22 | [0] 단일뷰 |
| `silhouette_only_6views` | 6 | 0 | Mask만 (keypoint 없음) |
| `silhouette_only_1view` | 1 | 0 | 단일뷰 mask only |
| `accurate_6views` | 6 | 22 | 고정밀 (iterations 증가) |

---

## 3단계 최적화 (Step 0/1/2)

MouseFitter는 L-BFGS 옵티마이저로 프레임별 3단계 최적화를 수행한다.

### Step 0: Global Alignment (초기 위치)

- **목표**: 전체적인 위치, 회전, 크기 맞춤
- **활성 파라미터**: `trans`, `rotation`, `scale`
- **비활성 파라미터**: `thetas`, `bone_lengths`, `chest_deformer`
- **주요 Loss**: `2d` (keypoint reprojection)
- **Iterations**: 10 (default) / 60 (논문)

### Step 1: Skeletal Fitting (관절 최적화)

- **목표**: 관절 각도와 뼈 길이를 최적화하여 포즈 맞춤
- **활성 파라미터**: `trans`, `rotation`, `scale`, `thetas`, `bone_lengths`
- **비활성 파라미터**: `chest_deformer`
- **주요 Loss**: `2d`, `theta` (정규화), `bone` (정규화), `temp` (시간적 부드러움)
- **Iterations**: 100 (default) / 5 (논문 tracking)

### Step 2: Silhouette Refinement (실루엣 정밀화)

- **목표**: PyTorch3D 실루엣 렌더링으로 메쉬 표면 정밀 맞춤
- **활성 파라미터**: 전체 (`chest_deformer` 포함)
- **주요 Loss**: `mask` (실루엣 IoU), `2d`, `theta`, `bone`, `temp`
- **Iterations**: 30 (default) / 3 (논문)

### Loss Terms 및 가중치

| Loss | Default Weight | 설명 |
|------|---------------|------|
| `theta` | 3.0 | 관절 정규화 (초기값에 가깝게) |
| `2d` | 0.2 | 2D keypoint reprojection (pixels^2) |
| `bone` | 0.5 | 뼈 길이 제약 |
| `scale` | 0.5 | 스케일 정규화 (목표: 115mm) |
| `mask` | 0 (Step0,1) / 3000 (Step2) | 실루엣 IoU |
| `chest_deformer` | 0.1 | 가슴 변형 정규화 |
| `stretch` | 1.0 | 뼈 stretch 페널티 |
| `temp` | 0.25 | 시간적 부드러움 |
| `temp_d` | 0.2 | 시간 미분 부드러움 |

---

## 모델 파라미터 (body_param)

최적화 대상 파라미터들이다. `torch.Tensor`로 변환, `requires_grad_(True)` 설정.

| 파라미터 | Shape | 설명 |
|----------|-------|------|
| `thetas` | `(1, 140, 3)` | 140개 관절 회전 (axis-angle) |
| `trans` | `(1, 3)` | 전역 이동 (x, y, z) mm |
| `scale` | `(1, 1)` | 전역 크기 |
| `rotation` | `(1, 3)` | 전역 회전 (root joint) |
| `bone_lengths` | `(1, 20)` | 주요 20개 뼈의 길이 변화량 |
| `chest_deformer` | `(1, 1)` | 가슴 메쉬 변형 파라미터 |

### 메쉬 스펙

| 항목 | 값 |
|------|-----|
| Vertices | ~14,500 |
| Joints | 140 |
| Keypoints | 22 |
| Body length (X축) | ~115mm |
| Height (Y축) | ~53mm |
| Width (Z축) | ~41mm |

---

## Shell Script 래퍼

### Multi-View Fitting (`run_mesh_fitting_default.sh`)

```bash
# Experiment 기반 실행 (권장)
./run_mesh_fitting_default.sh quick_test           # conf/experiment/quick_test.yaml
./run_mesh_fitting_default.sh quick_test 0 5       # experiment + frame override

# experiment 없이
./run_mesh_fitting_default.sh - 0 10               # "-"는 experiment 생략

# 추가 인자 전달
./run_mesh_fitting_default.sh - 0 10 -- --keypoints none
./run_mesh_fitting_default.sh - 0 10 -- --input_dir /path/to/data --keypoints none
```

스크립트 자동 처리: `PYOPENGL_PLATFORM=egl` 설정, conda 경로 자동 감지

### Monocular Fitting (`run_mesh_fitting_monocular.sh`)

```bash
# 기본 사용
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output

# 프레임 수 제한
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output 10

# Silhouette only (keypoint 없이)
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output - -- --keypoints none
```

### Experiment Runner (`run_experiment.sh`)

```bash
# 사용 가능한 실험 목록
./run_experiment.sh

# 디버그 모드 (2 frames)
./run_experiment.sh baseline_6view_keypoint --debug

# 전체 실행
./run_experiment.sh baseline_6view_keypoint

# 자동 재시작 (--resume_from)
./run_experiment.sh <experiment> --resume_from results/fitting/<result_folder>
```

---

## 시나리오별 사용법

### 시나리오 1: Multi-View Fitting (다중 카메라, 기본)

**입력**: 동기화된 다중 카메라 영상 + 2D keypoints + masks + camera params

```bash
# 쉘 스크립트 (권장)
./run_mesh_fitting_default.sh quick_test

# Python 직접 실행
export PYOPENGL_PLATFORM=egl
python fitter_articulation.py \
    dataset=default_markerless \
    optim=fast \
    fitter.end_frame=10
```

### 시나리오 2: Monocular Fitting (단일 카메라)

**입력**: 단일 카메라 영상 프레임 (RGB + mask)

```bash
# 쉘 스크립트
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output

# Python 직접 실행
python fit_monocular.py \
    --input_dir data/frames/ \
    --output_dir results/monocular/output \
    --detector geometric \
    --max_images 10
```

**Detector 옵션**:

| Detector | 방법 | 장점 | 단점 |
|----------|------|------|------|
| `geometric` | PCA contour 분석 | 빠름, 학습 불필요 | 정확도 낮음 (paw 불가) |
| `yolo` | YOLOv8-Pose CNN | 빠르고 정확, GPU 가속 | 학습 데이터 필요 |
| `superanimal` | DLC pretrained | 사전학습, 해부학적 | DLC API 제약, 느림 |

### 시나리오 3: Silhouette-Only (keypoint 없이)

**입력**: Mask만 사용 (keypoint annotation 불필요)

```bash
# Multi-view
./run_mesh_fitting_default.sh - 0 10 -- --keypoints none

# Silhouette 모드 설정
./run_mesh_fitting_default.sh - 0 10 -- --keypoints none \
    silhouette.iter_multiplier=3.0 silhouette.theta_weight=15.0
```

| Silhouette 옵션 | 기본값 | 설명 |
|-----------------|--------|------|
| `iter_multiplier` | 2.0 | 반복 횟수 배율 |
| `theta_weight` | 10.0 | 포즈 정규화 (높을수록 안정적) |
| `bone_weight` | 2.0 | 뼈대 길이 정규화 |
| `scale_weight` | 50.0 | 스케일 정규화 |
| `use_pca_init` | true | PCA 기반 회전 초기화 |

### 시나리오 4: 원본 비디오 전처리

**입력**: 원본 비디오 (MP4)

```bash
# 전처리
python scripts/preprocess.py dataset=my_video mode=single_view_preprocess

# 전처리 후 피팅
python fitter_articulation.py dataset=my_video
```

---

## 입력 데이터 명세

### Multi-View 데이터 구조

```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/               # 왜곡 보정된 비디오
│   ├── 0.mp4 ~ 5.mp4           # 6개 뷰
├── simpleclick_undist/          # 마스크 비디오
│   ├── 0.mp4 ~ 5.mp4           # 흰색=마우스, 검정=배경
├── keypoints2d_undist/          # 2D 키포인트
│   └── result_view_*.pkl        # Shape: (18000, 22, 3) [x, y, conf]
└── new_cam.pkl                  # 카메라 파라미터
```

### 카메라 파라미터 (`new_cam.pkl`)

```python
import pickle
with open("new_cam.pkl", "rb") as f:
    cams = pickle.load(f)  # List[Dict]

# cams[i] = {
#     'K': (3, 3),    # Intrinsic matrix (np.float64)
#     'R': (3, 3),    # Rotation matrix
#     'T': (3, 1),    # Translation vector
# }
```

---

## 성능 벤치마크 (RTX 3090)

| Task | Frames | 소요 시간 |
|------|--------|----------|
| Multi-view fitting (no render) | 10 | ~25분 |
| Multi-view fitting (with render) | 10 | ~70분 |
| Monocular fitting (geometric) | 10 | ~5분 |
| 논문 설정 (paper_fast) | 100 | ~15분 |
| 전체 시퀀스 (paper_fast) | 3,600 | ~10시간 |

---

## 관련 문서

- [KEYPOINTS.md](KEYPOINTS.md) - 22 키포인트 정의
- [DATASET.md](DATASET.md) - 데이터셋 스펙
- [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md) - 출력 형식
- [COORDINATES.md](COORDINATES.md) - 좌표계
- [PAPER.md](PAPER.md) - 논문 설정
- [EXPERIMENTS.md](EXPERIMENTS.md) - 실험 명령어

---

*Last updated: 2026-02-06*
