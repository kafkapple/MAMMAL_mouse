# 251103 연구노트 — SAM 기반 전처리 전환

## 목표
- shank3 데이터셋에서 OpenCV 전처리 실패 원인 규명 및 해결
- SAM (Segment Anything Model) 기반 전처리 파이프라인 구축
- mammal_stable conda 환경 안정화

## 진행 내용

### 1. OpenCV 전처리 실패 진단

기존 OpenCV BackgroundSubtractorMOG2가 **흰색 arena 배경에서 흰색 마우스를 구분하지 못함**:
- Mask 상태: 완전히 검은색 (0.0% coverage)
- Keypoint 상태: 84.6% (502,650 / 594,000)가 (0, 0, 0)
- Mean confidence: 0.097 (사실상 랜덤)
- Fitting 결과: mesh가 실제 마우스 위치와 완전히 무관

### 2. 환경 안정화 (mammal_stable)

**환경 스펙**:
- Python 3.10, PyTorch 2.0.0 + CUDA 11.8, PyTorch3D 0.7.5, NumPy < 2.0
- 1-스크립트 설치: `setup.sh`, `run_preprocess.sh`, `run_fitting.sh`
- Hydra 설정 시스템 구축: `conf/dataset/`, `conf/preprocess/`, `conf/optim/`

### 3. 버그 수정 (3건)

**3.1 카메라 투영 수학 오류** (`fitter_articulation.py:192-217`):
- 행렬 차원 불일치 문제. 올바른 순서: J3d -> R @ J3d_t + T -> K @ J3d_cam -> 정규화
- 검증 완료

**3.2 PyTorch3D T 벡터 shape** (`fitter_articulation.py:138-162`):
- `cameras_from_opencv_projection`이 (N, 3) 기대. T shape 변환 로직 추가
- (3, 1) -> (1, 3), (1, 3, 1) -> (1, 3) 등 다양한 입력 처리

**3.3 Render 함수 T 벡터** (`fitter_articulation.py:483-491`):
- pyrender용 (3, 1)과 PyTorch3D용 (1, 3) 분리 처리

### 4. SAM 통합 및 전처리 파이프라인

**SAM 아키텍처**: ViT-H (2.4GB checkpoint)

```
Input Video -> SAM Inference -> Multi-stage Mouse Detection -> Mask Refinement -> PCA Keypoint Estimation -> Output
```

**Multi-stage Mouse Detection**:
- Size filtering: 3-25% coverage (arena 50-70% 제거)
- Shape analysis: circularity < 0.85 (arena ~0.95 제거)
- Position filtering: arena bounds 내부

**PCA 기반 Keypoint Estimation**:
- Body orientation 검출 (major axis)
- Head/tail 구분 (narrower end = head)
- MAMMAL 22 keypoint layout 매핑 (head 0-5, spine 6-13, limbs 14-17, tail 18-20, centroid 21)
- Temporal filtering (IoU 기반)

### 5. SAM 전처리 검증 결과

**테스트 설정**: shank3 50 frames, 처리 시간 9분 10초 (~11 sec/frame)

| Metric | OpenCV (이전) | SAM (이후) | 개선 |
|--------|--------------|-----------|------|
| Detection Rate | 0% | 100% | - |
| Mean Confidence | 0.097 | 0.605 | **+525%** |
| Keypoints at Zero | 84.6% | 0% | -100% |
| Mask Coverage | 0.0% | 18.9% | - |
| Mean Mask Area | 0 px | 56,804 px | - |

- Fallback 발생: 1 frame (2%, frame 13) -- second-largest mask 사용으로 성공
- Primary strategy 성공률: 98%

### 6. Hydra 설정 시스템

```bash
# shank3 데이터 + 빠른 최적화
python fitter_articulation.py dataset=shank3 optim=fast fitter.end_frame=10

# markerless 데이터 + 정확한 최적화
python fitter_articulation.py dataset=markerless optim=accurate
```

### 7. shank3 피팅 검증

- 2 frames (디버그): 성공, ~1분
- 10 frames (interval=2, 5 frames 실제 처리): 성공, ~5분
- 출력: mesh_000000.obj ~ mesh_000008.obj (각 962KB), param*.pkl (각 3.6KB)
- GPU 메모리: ~4-5GB, 프레임당 ~35초

**Mask shape mismatch 경고**: 렌더링 해상도(1024x1152) vs 입력(480x640) 불일치. mask loss 자동 skip, 다른 loss term으로 수렴

## 핵심 발견

- **OpenCV 전처리 완전 실패**: 흰색 배경 + 흰색 마우스 조합에서 BackgroundSubtractorMOG2는 작동 불가
- **SAM 전환으로 525% confidence 개선**: 0.097 -> 0.605, 100% detection rate 달성
- **처리 속도 제한**: ~11 sec/frame으로 27,000 frames 처리 시 ~82시간 예상 (batch/parallel 최적화 필요)
- **Geometric keypoint의 한계**: Confidence ~0.6 수준으로 ML 기반 방법(SuperAnimal 등)보다 낮음
- **DeepLabCut SuperAnimal**: TensorFlow vs PyTorch 환경 충돌, NumPy 버전 문제로 빠른 통합 어려움

## 미해결 / 다음 단계
- Fitting mesh 위치 정확도 추가 조사 필요 (별도 이슈)
- 전체 27,000 frames 처리 여부 결정 (처리 시간 vs 품질 trade-off)
- DeepLabCut SuperAnimal Phase 2 통합 (별도 conda 환경 또는 DLC 3.0 대기)
- 렌더링 해상도 자동 조정 (mask shape mismatch 해결)

---
*Sources: 251103_success_report.md, 251103_sam_preprocessing_validation.md, 251103_preprocessing_improvement.md*
