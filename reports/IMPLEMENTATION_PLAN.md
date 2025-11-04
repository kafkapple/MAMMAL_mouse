# MAMMAL_mouse 구현 계획 및 실행 로드맵

**작성일**: 2025-11-03
**목표**: 일반화된 마커리스 3D 마우스 피팅 시스템 구축

---

## 전체 목표

다양한 영상 데이터(다중 뷰, 단일 뷰)에 자동으로 대응 가능한 **범용 마우스 3D 피팅 파이프라인** 구축

### 핵심 요구사항
1. ✅ 기존 다중 뷰 데이터 지원 (`markerless_mouse_1`)
2. ✅ 신규 단일 뷰 데이터 자동 전처리
3. ✅ Hydra 기반 유연한 설정 관리
4. 🔴 안정적인 환경 및 버그 수정
5. 🔄 고품질 전처리 (AI 모델 통합)

---

## Phase 1: 환경 및 인프라 안정화 (즉시 실행)

**우선순위**: 🔴 Critical
**예상 기간**: 1-2일
**담당**: 개발팀

### 1.1 환경 통일화 ✅ 완료

**목표**: `mouse` 환경을 `mammal_stable`로 대체

**완료 항목**:
- [x] `requirements.txt` 버전 명시 및 업데이트
- [x] `setup.sh` 환경 설정 스크립트 생성
- [x] `run_preprocess.sh` 전처리 실행 스크립트 생성
- [x] `run_fitting.sh` 피팅 실행 스크립트 생성
- [x] 스크립트 실행 권한 부여

**새로운 환경 스펙**:
```bash
Python: 3.10
PyTorch: 2.0.0 + CUDA 11.8
PyTorch3D: 0.7.5
NumPy: <2.0
TensorBoard: 2.13.0
```

**검증 방법**:
```bash
bash setup.sh
conda activate mammal_stable
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import pytorch3d; print(pytorch3d.__version__)"
```

### 1.2 Hydra 설정 개선 ✅ 완료

**목표**: 데이터셋별 설정 프로파일 구축

**완료 항목**:
- [x] `conf/dataset/` 디렉토리 생성
  - [x] `markerless.yaml` - 다중 뷰 데이터
  - [x] `shank3.yaml` - 단일 뷰 데이터
  - [x] `custom.yaml` - 사용자 정의 템플릿
- [x] `conf/preprocess/` 디렉토리 생성
  - [x] `opencv.yaml` - 현재 기하학적 접근
  - [x] `sam.yaml` - SAM 기반 (향후)
- [x] `conf/optim/` 디렉토리 생성
  - [x] `fast.yaml` - 빠른 테스트
  - [x] `accurate.yaml` - 고품질 결과
- [x] `config.yaml` defaults 섹션 업데이트

**사용 예시**:
```bash
# 다중 뷰 데이터 + 정확한 최적화
python fitter_articulation.py dataset=markerless optim=accurate

# 단일 뷰 전처리 + 빠른 피팅
python preprocess.py dataset=custom mode=single_view_preprocess
python fitter_articulation.py dataset=custom optim=fast
```

### 1.3 문서화 ✅ 완료

**목표**: 신규 사용자를 위한 명확한 가이드

**완료 항목**:
- [x] `README.md` 전면 재작성
  - [x] Quick Start 섹션
  - [x] Hydra 설정 가이드
  - [x] 상세 워크플로우
  - [x] 트러블슈팅 섹션
- [x] `PROJECT_ANALYSIS.md` 작성
  - [x] 기능 분석
  - [x] 문제점 정리
  - [x] 구현 계획
- [x] `IMPLEMENTATION_PLAN.md` (이 문서)

---

## Phase 2: 버그 수정 (즉시 실행)

**우선순위**: 🔴 Critical
**예상 기간**: 2-3일
**전제조건**: Phase 1 완료

### 2.1 카메라 투영 수학 오류 수정

**파일**: `fitter_articulation.py`
**위치**: Line ~174, `calc_2d_keypoint_loss` 함수

**현재 코드** (잘못됨):
```python
J2d = (J3d@self.Rs[camid].transpose(1,2) + self.Ts[camid].transpose(0,1)) @ self.Ks[camid].transpose(1,2)
```

**수정 코드**:
```python
def calc_2d_keypoint_loss(self, J3d, x2):
    """
    Calculate 2D keypoint reprojection loss with correct camera math

    Args:
        J3d: 3D keypoints (B, N, 3) - e.g., (1, 22, 3)
        x2: 2D keypoints (B, C, N, 3) - e.g., (1, 6, 22, 3) with confidence

    Returns:
        loss: Scalar tensor
    """
    loss = 0
    for camid in range(self.camN):
        # Correct camera projection math
        J3d_t = J3d.transpose(1, 2)  # (B, 3, N) = (1, 3, 22)
        rotated = self.Rs[camid] @ J3d_t  # (1, 3, 3) @ (1, 3, 22) = (1, 3, 22)

        # T vector broadcasting fix
        T_vec = self.Ts[camid]  # (1, 3, 1) or (1, 3)
        if T_vec.dim() == 2:
            T_vec = T_vec.unsqueeze(2)  # Ensure (1, 3, 1)

        J3d_cam = rotated + T_vec  # (1, 3, 22) + (1, 3, 1) = (1, 3, 22)
        J2d = self.Ks[camid] @ J3d_cam  # (1, 3, 3) @ (1, 3, 22) = (1, 3, 22)
        J2d = J2d.transpose(1, 2)  # (1, 22, 3)

        # Perspective division
        J2d = J2d / J2d[:,:,2:3]  # (1, 22, 3) / (1, 22, 1) = (1, 22, 3)
        J2d = J2d[:,:,0:2]  # (1, 22, 2)

        # Weighted loss
        diff = (J2d - x2[:,camid,:,0:2]) * x2[:,camid,:,2:]  # Apply confidence
        weighted_diff = diff * self.keypoint_weight[..., [0,0]]
        loss += torch.mean(torch.norm(weighted_diff, dim=-1))

    return loss
```

**테스트 방법**:
```bash
conda activate mammal_stable
python fitter_articulation.py dataset=shank3 fitter.end_frame=1 optim=fast
# 오류 없이 Step 1까지 완료되어야 함
```

### 2.2 PyTorch3D T 벡터 Shape 수정

**파일**: `fitter_articulation.py`
**위치**: `solve_step2` 함수, `render` 함수

**추가 메서드**:
```python
def fix_camera_T_shape(self):
    """
    Fix T vector shape for PyTorch3D compatibility
    PyTorch3D expects T in (N, 3) format
    """
    for camid in range(self.camN):
        T = self.Ts[camid]

        if T.shape == (1, 3, 1):
            self.Ts[camid] = T.squeeze(-1)  # (1, 3, 1) -> (1, 3)
        elif T.shape == (3, 1):
            self.Ts[camid] = T.T  # (3, 1) -> (1, 3)
        elif T.shape == (3,):
            self.Ts[camid] = T.unsqueeze(0)  # (3,) -> (1, 3)

        # Verify final shape
        assert self.Ts[camid].shape == (1, 3), \
            f"T shape must be (1, 3), got {self.Ts[camid].shape}"
```

**solve_step2 수정**:
```python
def solve_step2(self, ...):
    # Add at the beginning
    self.fix_camera_T_shape()

    # Rest of the function...
```

**render 함수 수정** (Line ~100):
```python
def render(self, ...):
    for view in views:
        K, R, T = cam_param['K'].T, cam_param['R'].T, cam_param['T'] / 1000

        # T shape normalization
        if T.shape == (1, 3):
            T = T.T  # (1, 3) -> (3, 1)
        elif T.shape == (3,):
            T = T.reshape(3, 1)
        elif T.shape == (1, 3, 1):
            T = T.squeeze().reshape(3, 1)
        elif T.shape == (3, 1, 1):
            T = T.squeeze()

        # Ensure T is (3, 1) for pyrender
        assert T.shape == (3, 1), f"T must be (3, 1) for pyrender, got {T.shape}"

        camera_pose[:3, 3:4] = np.dot(-R.T, T)
```

**테스트 방법**:
```bash
python fitter_articulation.py dataset=shank3 fitter.end_frame=1 fitter.with_render=true
# Step 2까지 완료되고 렌더링 성공해야 함
```

### 2.3 통합 테스트

**테스트 시나리오**:

1. **기존 다중 뷰 데이터 회귀 테스트**:
```bash
# markerless_mouse_1 데이터로 기존 기능 확인
python fitter_articulation.py dataset=markerless fitter.end_frame=1
```

2. **새로운 단일 뷰 전체 파이프라인**:
```bash
# 전처리
python preprocess.py dataset=shank3 mode=single_view_preprocess

# 피팅
python fitter_articulation.py dataset=shank3 fitter.end_frame=5

# 비디오 생성
ffmpeg -framerate 10 -i mouse_fitting_result/results/render/fitting_%d.png \
       -c:v libx264 -pix_fmt yuv420p -y test_output.mp4
```

3. **다양한 설정 조합 테스트**:
```bash
# 빠른 최적화
python fitter_articulation.py dataset=shank3 optim=fast

# 정확한 최적화
python fitter_articulation.py dataset=shank3 optim=accurate fitter.end_frame=3

# 파라미터 오버라이드
python fitter_articulation.py dataset=shank3 optim.solve_step1_iters=50
```

**성공 기준**:
- [ ] 모든 테스트 시나리오에서 오류 없이 완료
- [ ] 출력 파일들이 올바르게 생성됨 (.obj, .pkl, .png)
- [ ] 렌더링 결과가 시각적으로 합리적
- [ ] Hydra 설정 오버라이드가 정상 작동

---

## Phase 3: 전처리 정확도 개선 (선택적)

**우선순위**: ⚠️ Medium
**예상 기간**: 2-4주
**전제조건**: Phase 1-2 완료

### 3.1 SAM (Segment Anything Model) 통합

**목표**: 고품질 마스크 생성

**작업 항목**:
1. SAM 모델 다운로드 및 설치
   ```bash
   pip install segment-anything
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        -O models/sam_vit_h_4b8939.pth
   ```

2. `sam_preprocess.py` 구현:
   ```python
   import torch
   from segment_anything import sam_model_registry, SamPredictor

   class SAMPreprocessor:
       def __init__(self, checkpoint_path, device="cuda"):
           sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
           sam.to(device=device)
           self.predictor = SamPredictor(sam)

       def generate_mask(self, frame):
           self.predictor.set_image(frame)

           # Use center point as prompt
           h, w = frame.shape[:2]
           input_point = np.array([[w//2, h//2]])
           input_label = np.array([1])

           masks, scores, _ = self.predictor.predict(
               point_coords=input_point,
               point_labels=input_label,
               multimask_output=True,
           )

           # Select best mask
           best_mask = masks[np.argmax(scores)]
           return (best_mask * 255).astype(np.uint8)
   ```

3. `preprocess.py`에 통합:
   ```python
   @hydra.main(config_path="./conf", config_name="config")
   def preprocess_video(cfg: DictConfig):
       if cfg.preprocess.mask_method == "sam":
           from sam_preprocess import SAMPreprocessor
           mask_generator = SAMPreprocessor(
               checkpoint_path=cfg.preprocess.sam.checkpoint,
               device=cfg.preprocess.sam.device
           )
       else:  # opencv
           mask_generator = OpenCVMaskGenerator()

       # Use mask_generator in video loop
   ```

4. 설정 파일 활성화:
   ```yaml
   # conf/preprocess/sam.yaml already exists
   # Just need to implement the code
   ```

**테스트**:
```bash
python preprocess.py dataset=custom preprocess=sam mode=single_view_preprocess
```

**예상 효과**:
- 배경 변화에 강인한 마스크
- 복잡한 환경에서도 정확한 세그멘테이션
- 수동 어노테이션 불필요

### 3.2 DeepLabCut 키포인트 추정 통합

**목표**: 해부학적으로 정확한 키포인트

**작업 항목**:
1. DeepLabCut 설치:
   ```bash
   pip install deeplabcut
   ```

2. 마우스 특화 사전 훈련 모델 다운로드:
   - Model Zoo에서 마우스 모델 검색
   - 또는 직접 훈련 (시간 소요)

3. `dlc_preprocess.py` 구현:
   ```python
   import deeplabcut

   class DLCPreprocessor:
       def __init__(self, config_path):
           self.config_path = config_path

       def extract_keypoints(self, video_path, output_dir):
           # Run DLC analysis
           deeplabcut.analyze_videos(
               self.config_path,
               [video_path],
               save_as_csv=True,
               destfolder=output_dir
           )

           # Convert to MAMMAL format
           dlc_results = pd.read_csv(f"{output_dir}/results.csv")
           mammal_keypoints = self.convert_dlc_to_mammal(dlc_results)
           return mammal_keypoints

       def convert_dlc_to_mammal(self, dlc_data):
           # Map DLC keypoints to MAMMAL 22-point format
           # This requires understanding both schemas
           pass
   ```

4. `preprocess.py`에 통합:
   ```python
   if cfg.preprocess.keypoint_method == "dlc":
       keypoint_extractor = DLCPreprocessor(cfg.preprocess.dlc.config_path)
   else:  # opencv geometric
       keypoint_extractor = OpenCVKeypointExtractor()
   ```

**테스트**:
```bash
python preprocess.py dataset=custom \
    preprocess.keypoint_method=dlc \
    preprocess.dlc.config_path=models/mouse_dlc_config.yaml
```

**예상 효과**:
- 해부학적으로 정확한 22개 키포인트
- 프레임별 일관성 향상
- 피팅 수렴 속도 개선

### 3.3 통합 전처리 시스템

**목표**: 사용자가 전처리 방법 자유롭게 선택

**작업 항목**:
1. `unified_preprocess.py` 생성:
   ```python
   from typing import Protocol

   class MaskGenerator(Protocol):
       def generate_mask(self, frame: np.ndarray) -> np.ndarray: ...

   class KeypointExtractor(Protocol):
       def extract_keypoints(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

   class UnifiedPreprocessor:
       def __init__(self, cfg: DictConfig):
           # Initialize mask generator
           if cfg.preprocess.mask_method == "sam":
               self.mask_gen = SAMPreprocessor(...)
           else:  # opencv
               self.mask_gen = OpenCVMaskGenerator(...)

           # Initialize keypoint extractor
           if cfg.preprocess.keypoint_method == "dlc":
               self.kpt_ext = DLCPreprocessor(...)
           elif cfg.preprocess.keypoint_method == "yolo":
               self.kpt_ext = YOLOPreprocessor(...)
           else:  # opencv
               self.kpt_ext = OpenCVKeypointExtractor(...)

       def process_video(self, video_path):
           cap = cv2.VideoCapture(video_path)

           all_masks = []
           all_keypoints = []

           while True:
               ret, frame = cap.read()
               if not ret:
                   break

               # Generate mask
               mask = self.mask_gen.generate_mask(frame)
               all_masks.append(mask)

               # Extract keypoints
               keypoints = self.kpt_ext.extract_keypoints(frame, mask)
               all_keypoints.append(keypoints)

           return all_masks, all_keypoints
   ```

2. 설정 조합 테스트:
   ```bash
   # SAM mask + OpenCV keypoints
   python preprocess.py preprocess.mask_method=sam preprocess.keypoint_method=opencv

   # OpenCV mask + DLC keypoints
   python preprocess.py preprocess.mask_method=opencv preprocess.keypoint_method=dlc

   # SAM mask + DLC keypoints (최고 품질)
   python preprocess.py preprocess.mask_method=sam preprocess.keypoint_method=dlc
   ```

**성공 기준**:
- [ ] 모든 조합이 오류 없이 동작
- [ ] 출력 형식이 일관됨
- [ ] 품질 향상 검증 (수동 평가)

---

## Phase 4: 품질 보증 및 최적화 (선택적)

**우선순위**: ⚠️ Low
**예상 기간**: 1-2주
**전제조건**: Phase 1-3 완료

### 4.1 유닛 테스트 구축

**파일**: `tests/test_*.py`

**테스트 커버리지**:
1. 카메라 투영 수학:
   ```python
   def test_camera_projection():
       # Test correct projection math
       J3d = torch.randn(1, 22, 3)
       K = torch.eye(3).unsqueeze(0)
       R = torch.eye(3).unsqueeze(0)
       T = torch.zeros(1, 3, 1)

       # Should not raise dimension errors
       J2d = project_3d_to_2d(J3d, K, R, T)
       assert J2d.shape == (1, 22, 2)
   ```

2. 데이터 로더:
   ```python
   def test_data_loader():
       cfg = load_config("conf/config.yaml")
       loader = DataSeakerDet(cfg)

       batch = loader[0]
       assert "keypoints_2d" in batch
       assert batch["keypoints_2d"].shape[-1] == 22
   ```

3. Hydra 설정 검증:
   ```python
   def test_hydra_configs():
       configs = ["markerless", "shank3", "custom"]
       for cfg_name in configs:
           cfg = load_config(f"dataset={cfg_name}")
           assert "data" in cfg
           assert "fitter" in cfg
   ```

### 4.2 성능 벤치마킹

**메트릭**:
- 전처리 속도 (frames/sec)
- 피팅 속도 (iterations/sec)
- 메모리 사용량
- GPU 활용률

**벤치마크 스크립트**:
```python
# benchmark.py
import time
import torch
import psutil

def benchmark_preprocessing(video_path):
    start = time.time()
    preprocess_video(video_path)
    duration = time.time() - start

    fps = total_frames / duration
    print(f"Preprocessing: {fps:.2f} FPS")

def benchmark_fitting(data_dir, num_frames):
    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    run_fitting(data_dir, end_frame=num_frames)
    duration = time.time() - start

    max_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Fitting: {duration/num_frames:.2f} sec/frame")
    print(f"Peak GPU memory: {max_memory:.2f} GB")
```

### 4.3 코드 리팩토링

**목표**: 코드 품질 및 유지보수성 향상

**작업 항목**:
1. Type hints 추가:
   ```python
   def calc_2d_keypoint_loss(
       self,
       J3d: torch.Tensor,  # (B, N, 3)
       x2: torch.Tensor    # (B, C, N, 3)
   ) -> torch.Tensor:      # scalar
       ...
   ```

2. Docstrings 작성:
   ```python
   def solve_step1(self, ...):
       """
       Perform joint optimization with 2D keypoints.

       This step optimizes both global pose and joint angles
       using 2D keypoint reprojection loss.

       Args:
           ...

       Returns:
           Optimized parameters dict
       """
   ```

3. 매직 넘버 제거:
   ```python
   # Before
   self.keypoint_weight[4] = 0.4

   # After
   RIGHT_EAR_IDX = 4
   RIGHT_EAR_CONFIDENCE = 0.4
   self.keypoint_weight[RIGHT_EAR_IDX] = RIGHT_EAR_CONFIDENCE
   ```

---

## 실행 우선순위 및 타임라인

### 🔴 즉시 실행 (1주 이내)

**Phase 1: 인프라 안정화** ✅ 완료
- [x] 환경 통일화
- [x] Hydra 설정 개선
- [x] 문서화

**Phase 2: 버그 수정** ⬅️ **현재 작업**
- [ ] 카메라 투영 수학 오류 수정
- [ ] PyTorch3D T 벡터 Shape 수정
- [ ] 통합 테스트

### ⚠️ 중기 목표 (2-4주)

**Phase 3: 전처리 개선**
- [ ] SAM 통합 (1-2주)
- [ ] DeepLabCut 통합 (1-2주)
- [ ] 통합 전처리 시스템 (1주)

### ⬇️ 장기 목표 (1-3개월)

**Phase 4: 품질 보증**
- [ ] 유닛 테스트 구축
- [ ] 성능 벤치마킹
- [ ] 코드 리팩토링

**향후 확장**:
- [ ] 실시간 처리 파이프라인
- [ ] 다중 동물 추적
- [ ] 시간적 일관성 개선
- [ ] Interactive 어노테이션 도구

---

## 검증 및 테스트 전략

### 단위 테스트
- 각 함수별 입출력 검증
- Edge case 처리 확인
- 차원 불일치 오류 방지

### 통합 테스트
- 전체 파이프라인 E2E 실행
- 다양한 데이터셋 조합
- 설정 오버라이드 검증

### 회귀 테스트
- 기존 markerless_mouse_1 결과 비교
- 성능 저하 모니터링
- 시각적 품질 평가

### 사용자 수용 테스트
- 신규 사용자 온보딩 시뮬레이션
- 문서만으로 실행 가능한지 검증
- 일반적인 오류 시나리오 테스트

---

## 성공 지표

### Phase 1-2 (즉시)
- ✅ 환경 설정이 단일 스크립트로 완료
- ✅ Hydra 설정으로 다양한 데이터셋 처리
- [ ] 모든 알려진 버그 수정
- [ ] 회귀 테스트 100% 통과

### Phase 3 (중기)
- [ ] SAM 마스크가 OpenCV 대비 IoU 10%↑
- [ ] DLC 키포인트가 RMSE 30%↓
- [ ] 전처리 시간 2배 이내 유지

### Phase 4 (장기)
- [ ] 코드 커버리지 80% 이상
- [ ] GPU 메모리 사용량 20% 감소
- [ ] 피팅 속도 30% 향상

---

## 리스크 및 대응 방안

### 리스크 1: SAM 모델이 너무 무거움
**영향**: 전처리 시간 급증
**대응**:
- SAM-Light 버전 사용 (vit_b)
- 배치 처리로 효율성 향상
- 선택적 사용 (Hydra config로 on/off)

### 리스크 2: DLC 키포인트 매핑 불일치
**영향**: 피팅 품질 저하
**대응**:
- MAMMAL 22-point와 DLC schema 상세 매핑 테이블 작성
- 수동 검증 단계 추가
- Fallback to OpenCV

### 리스크 3: 환경 설정 여전히 실패
**영향**: 사용자 온보딩 어려움
**대응**:
- Docker 이미지 제공
- conda-lock 파일로 정확한 버전 고정
- CI/CD에서 자동 검증

---

## 다음 단계 (Action Items)

### 이번 주 (Day 1-3)
1. [ ] Phase 2.1 실행: 카메라 투영 수학 오류 수정
2. [ ] Phase 2.2 실행: PyTorch3D T 벡터 수정
3. [ ] Phase 2.3 실행: 통합 테스트

### 다음 주 (Day 4-7)
1. [ ] Phase 2 완료 검증
2. [ ] Phase 3.1 시작: SAM 환경 구축
3. [ ] 중간 보고서 작성

### 향후 계획
- **2주차**: SAM 통합 완료
- **3-4주차**: DLC 통합
- **5주차**: 통합 전처리 시스템 및 테스트
- **6주차 이후**: Phase 4 품질 보증

---

## 참고 문서

- `PROJECT_ANALYSIS.md` - 전체 프로젝트 분석
- `README.md` - 사용자 가이드
- `reports/` - 과거 작업 보고서
- `conf/` - Hydra 설정 예제

---

**작성자**: Claude Code
**최종 업데이트**: 2025-11-03
**버전**: 1.0
