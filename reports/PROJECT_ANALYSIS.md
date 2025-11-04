# MAMMAL_mouse 프로젝트 종합 분석

**분석 일자**: 2025-11-03
**분석자**: Claude Code

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [기본 주요 기능](#2-기본-주요-기능)
3. [새로 구현된 기능](#3-새로-구현된-기능)
4. [현재 문제점 및 이슈](#4-현재-문제점-및-이슈)
5. [환경 설정 분석](#5-환경-설정-분석)
6. [구현 계획](#6-구현-계획)

---

## 1. 프로젝트 개요

### 프로젝트 목적
MAMMAL (Multi-Animal Multi-Modal Articulated Locomotion) 프레임워크의 마우스 서브프로젝트로, **다중 뷰 영상에서 마커리스 3D 마우스 모델 피팅**을 수행합니다.

### 핵심 기술
- **입력**: 다중/단일 뷰 비디오, 2D 키포인트, 실루엣 마스크
- **처리**: 관절형 3D 모델 피팅 (Articulated Model Fitting)
- **출력**: 3D 메시 (.obj), 피팅 파라미터 (.pkl), 시각화 결과 (.png)

### 기반 모델
- C57BL6_Female_V1.2 (블렌더 파일 기반)
- 22개 키포인트 (MAMMAL 표준)

---

## 2. 기본 주요 기능

### 2.1 다중 뷰 3D 피팅 (원본 기능)
- **데이터셋**: `markerless_mouse_1` (DANNCE 프로젝트 제공)
- **입력 요구사항**:
  - 사전 보정된 다중 카메라 파라미터 (`new_cam.pkl`)
  - 수동으로 어노테이션된 2D 키포인트
  - SimpleClick으로 생성된 실루엣 마스크
- **실행**: `bash run.sh` (전체 환경 설정 + 피팅 실행)

### 2.2 3단계 최적화 프로세스
1. **Step 0**: 초기 파라미터 추정 (Global Pose Initialization)
2. **Step 1**: 중간 피팅 (Joint Optimization with 2D Keypoints)
3. **Step 2**: 정밀 피팅 (Silhouette-based Refinement with PyTorch3D)

### 2.3 피팅 결과물
- `mouse_fitting_result/results/obj/` - 3D 메시 파일
- `mouse_fitting_result/results/params/` - 피팅 파라미터
- `mouse_fitting_result/results/render/` - 렌더링 결과 이미지
- `mouse_fitting_result/results/fitting_keypoints_*.png` - 키포인트 시각화

---

## 3. 새로 구현된 기능

### 3.1 Hydra 기반 설정 관리 ✅
**구현 완료 (2025-10-30)**

- **설정 파일**: `conf/config.yaml`
- **모드 전환**:
  - `mode: multi_view` - 피팅 실행
  - `mode: single_view_preprocess` - 전처리 실행
- **장점**:
  - 중앙화된 파라미터 관리
  - 실험 재현성 향상
  - 다양한 데이터셋 지원 용이

**주요 설정 항목**:
```yaml
data:
  data_dir: data/preprocessed_shank3/
  views_to_use: [0]

preprocess:
  input_video_path: data/shank3/video.avi
  output_data_dir: data/preprocessed_shank3/

fitter:
  start_frame: 0
  end_frame: 2
  with_render: false
  keypoint_num: 22

optim:
  solve_step0_iters: 10
  solve_step1_iters: 100
  solve_step2_iters: 30
```

### 3.2 단일 뷰 비디오 전처리 자동화 ✅
**구현 완료 (2025-10-30)**

**파일**: `preprocess.py`

**기능**:
1. **자동 마스크 생성** (OpenCV 배경 차분)
   - `BackgroundSubtractorMOG2` 사용
   - 형태학적 연산으로 노이즈 제거

2. **자동 키포인트 추정** (기하학적 매핑)
   - 컨투어 분석으로 경계 박스 추출
   - 중심점, 극점 기반으로 22개 키포인트 생성
   - **한계**: 해부학적 정확도 부족

3. **더미 카메라 파라미터 생성**
   - 단일 뷰용 기본 내재 파라미터 (K, R, T)

**출력**:
- `videos_undist/0.mp4` - 원본 비디오
- `simpleclick_undist/0.mp4` - 마스크 비디오
- `keypoints2d_undist/result_view_0.pkl` - 2D 키포인트
- `new_cam.pkl` - 카메라 파라미터

### 3.3 다중 데이터셋 지원
**실험 완료 (2025-10-30)**

- `markerless_mouse_1` (다중 뷰, 기존 데이터셋) ✅
- `shank3` (단일 뷰, 새로운 커스텀 데이터) ✅

---

## 4. 현재 문제점 및 이슈

### 4.1 환경 의존성 충돌 🔴 **Critical**

#### 문제 1: PyTorch/NumPy 버전 불일치
**증상** (2025-10-31):
```
AttributeError: module 'distutils' has no attribute 'version'
ModuleNotFoundError: No module named 'numpy._core'
```

**근본 원인**:
- PyTorch 1.10.2 + NumPy 1.23.5 조합의 불안정성
- `tensorboard` 설치 시 setuptools 버전 충돌
- NumPy 2.x와 PyTorch 1.x 비호환성

**시도된 해결책**:
1. ❌ setuptools 다운그레이드 → 추가 충돌 발생
2. ❌ NumPy 재설치 → `.pkl` 파일 호환성 문제
3. ✅ **완전 재설치** (mammal_stable 환경) → 성공

#### 문제 2: 환경 설정 불일치
**현재 상황**:
- `run.sh`: `mouse` 환경 사용 (PyTorch 1.10.2)
- 보고서 (2025-11-02): `mammal_stable` 환경 권장 (PyTorch 2.0.0)
- **불일치 상태** → 혼란 초래

### 4.2 카메라 투영 수학 오류 🔴 **Critical**

#### 오류 위치: `fitter_articulation.py:174`
**증상** (2025-11-02):
```
RuntimeError: The size of tensor a (22) must match the size of tensor b (3) at non-singleton dimension 1
```

**근본 원인**:
`calc_2d_keypoint_loss` 함수에서 **행렬 곱셈 순서와 브로드캐스팅 오류**

**기존 코드** (잘못됨):
```python
J2d = (J3d@self.Rs[camid].transpose(1,2) + self.Ts[camid].transpose(0,1)) @ self.Ks[camid].transpose(1,2)
```
- `J3d` shape: `(1, 22, 3)`
- `Rs[camid]` shape: `(1, 3, 3)`
- `Ts[camid]` shape: `(1, 3, 1)` 또는 `(3, 1)`
- **문제**: T 벡터 브로드캐스팅 불가

**해결 방법** (보고서에 명시):
```python
def calc_2d_keypoint_loss(self, J3d, x2):
    loss = 0
    for camid in range(self.camN):
        # 올바른 카메라 투영 수학
        J3d_t = J3d.transpose(1, 2)  # (1, 3, 22)
        rotated = self.Rs[camid] @ J3d_t  # (1, 3, 3) @ (1, 3, 22) = (1, 3, 22)

        # T 벡터 브로드캐스팅 수정
        T_vec = self.Ts[camid]  # (1, 3, 1) or (1, 3)
        if T_vec.dim() == 2:
            T_vec = T_vec.unsqueeze(2)  # (1, 3, 1)

        J3d_cam = rotated + T_vec  # (1, 3, 22) + (1, 3, 1) = (1, 3, 22)
        J2d = self.Ks[camid] @ J3d_cam  # (1, 3, 3) @ (1, 3, 22) = (1, 3, 22)
        J2d = J2d.transpose(1, 2)  # (1, 22, 3)
        J2d = J2d / J2d[:,:,2:3]  # 정규화
        J2d = J2d[:,:,0:2]  # (1, 22, 2)

        diff = (J2d - x2[:,camid,:,0:2]) * x2[:,camid,:,2:]
        weighted_diff = diff * self.keypoint_weight[..., [0,0]]
        loss += torch.mean(torch.norm(weighted_diff, dim=-1))
    return loss
```

### 4.3 PyTorch3D T 벡터 Shape 불일치 🔴 **Critical**

#### 오류 위치: `solve_step2` 함수 (PyTorch3D 카메라 생성)
**증상** (2025-11-02):
```
ValueError: Expected T to have shape (N, 3); got 'torch.Size([1, 3, 1])'
```

**근본 원인**:
PyTorch3D의 `cameras_from_opencv_projection`이 T 벡터를 `(N, 3)` 형태로 기대

**해결 방법** (보고서에 명시):
```python
def fix_camera_T_shape(self):
    """PyTorch3D 호환을 위한 T 벡터 shape 수정"""
    for camid in range(self.camN):
        T = self.Ts[camid]
        if T.shape == (1, 3, 1):
            self.Ts[camid] = T.squeeze(-1)  # (1, 3, 1) -> (1, 3)
        elif T.shape == (3, 1):
            self.Ts[camid] = T.T  # (3, 1) -> (1, 3)

# solve_step2 함수 시작 부분에 추가
def solve_step2(self, ...):
    self.fix_camera_T_shape()  # 추가
    # 기존 코드 계속...
```

### 4.4 렌더링 환경 문제 ⚠️ **Resolved**

**증상** (2025-10-31):
```
pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"
```

**해결책** (이미 적용됨):
```bash
export PYOPENGL_PLATFORM=egl
```
- `run.sh`에 이미 설정되어 있음

### 4.5 전처리 정확도 제한 ⚠️ **Enhancement Needed**

**현재 방식**: OpenCV 기하학적 접근
- **장점**: 빠른 처리, 외부 의존성 없음
- **단점**:
  - 키포인트 해부학적 정확도 부족
  - 배경 변화에 민감한 마스크
  - 복잡한 자세에서 실패 가능

**개선 계획** (보고서에 명시):
1. **SAM (Segment Anything Model)** - 고품질 마스크
2. **DeepLabCut** - 마우스 특화 키포인트
3. **YOLOv8 Pose** - 실시간 처리

---

## 5. 환경 설정 분석

### 5.1 현재 환경 (run.sh 기준)

```bash
conda create -n mouse python=3.9
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3
pip install numpy==1.23.5
conda install pytorch3d==0.6.2
```

**문제점**:
- ❌ 버전 조합 불안정
- ❌ tensorboard 충돌 가능
- ❌ NumPy 2.x 호환성 없음

### 5.2 권장 환경 (보고서 기준)

```bash
conda create -n mammal_stable python=3.10
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0" tensorboard==2.13.0
pip install opencv-python omegaconf hydra-core tqdm trimesh pyrender scipy matplotlib
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html
```

**장점**:
- ✅ 검증된 버전 조합
- ✅ tensorboard 안정적 동작
- ✅ PyTorch3D 최신 버전 지원

### 5.3 requirements.txt 현황

**현재 내용**:
```
glfw
pyGLM
freetype-py
pyrender
matplotlib
scipy
scikit-learn
opencv-python
tqdm
ipython
trimesh
plotly
imageio
videoio
scikit-image
```

**문제점**:
- ❌ 버전 명시 없음 → 재현성 부족
- ❌ PyTorch/PyTorch3D 누락
- ❌ Hydra 관련 패키지 누락

---

## 6. 구현 계획

### Phase 1: 환경 및 인프라 안정화 (우선순위: 높음)

#### 1.1 환경 일원화 및 문서화
**목표**: mouse → mammal_stable 환경 전환

**작업 항목**:
1. `requirements.txt` 업데이트
   - 모든 패키지 버전 명시
   - PyTorch, PyTorch3D, Hydra 포함

2. `setup.sh` 생성 (환경 설정 전용)
   ```bash
   # 환경 생성 + 의존성 설치만 수행
   # 피팅 실행은 별도 스크립트로 분리
   ```

3. `run_fitting.sh` 생성 (피팅 실행 전용)
   ```bash
   # 환경 활성화 + fitter_articulation.py 실행
   # 전처리는 별도 스크립트로 분리
   ```

4. `run_preprocess.sh` 생성 (전처리 실행 전용)
   ```bash
   # 환경 활성화 + preprocess.py 실행
   ```

#### 1.2 README 업데이트
**목표**: 신규 사용자가 즉시 실행 가능한 문서

**포함 내용**:
- 환경 설정 방법 (setup.sh)
- 기본 데이터셋 실행 방법
- 커스텀 데이터 처리 방법
- 트러블슈팅 가이드

### Phase 2: 버그 수정 (우선순위: 높음)

#### 2.1 카메라 투영 수학 오류 수정
**파일**: `fitter_articulation.py`

**수정 위치**:
1. `calc_2d_keypoint_loss` 함수 (Line ~174)
2. `render` 함수 내 T 벡터 처리

#### 2.2 PyTorch3D Shape 호환성 수정
**파일**: `fitter_articulation.py`

**수정 방법**:
- `fix_camera_T_shape()` 메서드 추가
- `solve_step2` 함수 시작 부분에서 호출

#### 2.3 통합 테스트
**데이터셋**:
1. `markerless_mouse_1` (다중 뷰) - 회귀 테스트
2. `shank3` (단일 뷰) - 새 기능 검증

### Phase 3: Hydra 설정 개선 (우선순위: 중간)

#### 3.1 데이터셋별 설정 프로파일
**목표**: 다양한 데이터셋 쉽게 전환

**구조**:
```
conf/
├── config.yaml          # 기본 설정
├── dataset/
│   ├── markerless.yaml  # 다중 뷰 기본 데이터
│   ├── shank3.yaml      # 단일 뷰 커스텀 데이터
│   └── custom.yaml      # 사용자 정의 템플릿
├── preprocess/
│   ├── opencv.yaml      # 현재 방식
│   ├── sam.yaml         # SAM 기반 (향후)
│   └── dlc.yaml         # DeepLabCut 기반 (향후)
└── optim/
    ├── fast.yaml        # 빠른 테스트용
    └── accurate.yaml    # 고품질 결과용
```

**사용 예시**:
```bash
# 다중 뷰 데이터 피팅
python fitter_articulation.py dataset=markerless optim=accurate

# 단일 뷰 전처리
python preprocess.py dataset=custom preprocess=opencv

# 커스텀 설정 오버라이드
python fitter_articulation.py dataset=shank3 fitter.end_frame=100
```

#### 3.2 실험 로깅 개선
**목표**: Hydra의 출력 디렉토리 활용

**기능**:
- 각 실행마다 타임스탬프 기반 폴더 생성
- 설정 파일 자동 저장 (`.hydra/config.yaml`)
- 결과물 체계적 정리

### Phase 4: 전처리 정확도 개선 (우선순위: 낮음)

#### 4.1 SAM 통합 (Phase 4-1)
**목표**: 고품질 마스크 생성

**구현 파일**: `sam_preprocess.py`

**예상 작업 기간**: 1-2주

#### 4.2 DeepLabCut/YOLO 통합 (Phase 4-2)
**목표**: 해부학적으로 정확한 키포인트

**구현 파일**: `dlc_preprocess.py`, `yolo_preprocess.py`

**예상 작업 기간**: 2-3주

#### 4.3 통합 전처리 시스템 (Phase 4-3)
**목표**: 사용자가 전처리 방법 선택 가능

**구현 파일**: `unified_preprocess.py`

**예상 작업 기간**: 1주

---

## 요약

### ✅ 구현 완료
1. Hydra 기반 설정 관리
2. 단일 뷰 전처리 자동화
3. 다중 데이터셋 지원 (markerless, shank3)

### 🔴 즉시 해결 필요
1. 환경 의존성 통일 (mouse → mammal_stable)
2. 카메라 투영 수학 오류 수정
3. PyTorch3D T 벡터 Shape 수정
4. requirements.txt 업데이트

### 📋 개선 계획
1. setup.sh / run_fitting.sh 분리
2. Hydra 데이터셋 프로파일 구축
3. 전처리 정확도 향상 (SAM, DLC, YOLO)

### 📊 최종 목표
**일반화된 마커리스 3D 마우스 피팅 시스템**
- ✅ 다중/단일 뷰 지원
- ✅ 자동 전처리 (마스크 + 키포인트)
- ✅ 다양한 데이터셋 손쉬운 적용
- 🔄 고품질 결과물 (AI 모델 통합)
