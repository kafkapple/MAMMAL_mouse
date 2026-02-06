# 251114 연구노트 — ML Keypoint Detection + Monocular Fitting PoC

## 목표
- Monocular RGB 이미지에서 MAMMAL 3D mouse mesh 생성 PoC
- ML 기반 keypoint detection으로 geometric baseline 대체 (YOLOv8-Pose, SuperAnimal)
- Fauna의 mouse 한계 (sub-voxel problem) 극복

## 진행 내용

### 1. Monocular MAMMAL Fitting PoC

**배경**: Fauna는 mouse-scale에서 이론적으로 불가 (mouse leg 5mm < DMTet voxel 11.7mm)

**파이프라인**:
```
Monocular RGB + Mask -> Geometric Keypoint Estimation (22 kpts) -> MAMMAL Parameter Optimization -> 3D Mesh (14,522 vertices)
```

**MAMMAL 모델 파라미터**: thetas (1,140,3), bone_lengths (1,28), R (1,3), T (1,3), s (1,1), chest_deformer (1,1)

**Optimization**: Adam, lr=0.01, 50 iterations
- Loss = weighted L2 (2D keypoints) + 0.001 * L2 regularization (joint angles)
- 최적화 변수: thetas, T, s / 고정: bone_lengths, R, chest_deformer

**PoC 결과 (5 test images)**:

| Metric | Value |
|--------|-------|
| Processing time | 21 sec/image |
| Optimization iterations | 50 |
| Final loss (typical) | 280K - 340K |
| Mesh vertices | 14,522 |
| Mesh faces | 28,800 |
| Mean keypoint confidence | 0.605 |
| Success rate | 5/5 (100%) |

**Loss 수렴 패턴**: smooth, consistent ~1-3% reduction. Best frame (0000696): 285K -> 283K

**정성적 평가**: mesh topology 정확 (mouse-like), keypoints 해부학적으로 합리적. 다만 fine details 제한 (paws, ears), T-pose bias (regularization 영향)

### 2. Fauna vs MAMMAL 비교

| Aspect | Fauna | MAMMAL Monocular |
|--------|-------|------------------|
| Representation | DMTet grid (discrete) | Parametric mesh (continuous) |
| Resolution | Voxel-limited | Vertex-level (14K points) |
| Mouse support | 불가능 (sub-voxel) | Native 지원 |
| Prior | Diffusion (generic) | Anatomical (mouse-specific) |
| Speed | Hours (training) | Seconds (optimization) |

### 3. YOLOv8-Pose Infrastructure (Phase 1)

**DANNCE -> YOLO 변환**: `preprocessing_utils/dannce_to_yolo.py` (329 lines)
- BBox clipping 필수! (clipping 전 26/50 이미지 rejected, 후 50/50 accepted)
- flip_idx format: flat list (pairs 아님)
- 결과: 50 train + 10 val images

**YOLOv8-Pose 학습**:
- Model: yolov8n-pose (3.4M params), Transfer: 361/397 weights (91%)
- 10 epochs test run, ~15분
- **결과: mAP ~0 (완전 실패)**
- **원인: Geometric labels의 낮은 품질 (confidence 0.4-0.6)**

### 4. SuperAnimal-TopViewMouse 통합 시도 (Phase 2)

**모델 정보**: HuggingFace, 245 MB, TensorFlow checkpoint, 27 keypoints (5K+ mice 학습)

**SuperAnimal -> MAMMAL Keypoint Mapping (27 -> 22)**:
- Direct mapping: 10/22 (45%) -- nose, ears, eyes, head_center, tail_base, tail_tip
- Interpolation: 9/22 (41%) -- spine 8개 (arc-length parameterization), tail_mid
- Estimation: 3/22 (14%) -- paws (shoulder/hip 기준 perpendicular 추정)

**DLC API 한계 발견**:
- `video_inference_superanimal()`은 비디오 전용, 단일 이미지 미지원
- h5 결과 파일 미생성
- DLC 3.0 PyTorch API (`superanimal_analyze_images()`) 아직 미릴리스

**NumPy 충돌 해결**: TF 2.12 requires <1.24, 환경에 2.2.6 설치됨
- 해결: `~/miniconda3/envs/mammal_stable/bin/pip install "numpy<1.24,>=1.22"` (직접 pip path 사용)

**Geometric fallback**: 15/22 keypoints 검출 (conf=0.5) -- 예상보다 양호

### 5. fit_monocular.py 통합

- `--detector geometric|superanimal` 옵션 추가
- `--superanimal_model` path 옵션
- Device mismatch 수정: ArticulationTorch가 CUDA 하드코딩 -> override 처리

### 6. Keypoint Confidence 분석 (Geometric Baseline)

| 영역 | Keypoints | Confidence | 검출 |
|------|-----------|------------|------|
| Head | 6/6 | 0.70-0.95 | 100% |
| Spine | 8/8 | 0.65-0.80 | 100% |
| Paws | 0/4 | - | **0%** |
| Tail | 3/3 | 0.50-0.65 | 100% |
| 전체 | 15/22 | 0.40-0.70 | 68% |

## 핵심 발견

- **Data Quality > Algorithm**: Geometric keypoints로 YOLOv8 학습 시 mAP ~0. ML 모델은 학습 데이터 품질에 절대적으로 의존. **20개 perfect labels > 500개 noisy labels**
- **Monocular PoC 성공**: 21초/image로 14K vertex mesh 생성 가능. Fauna 대비 mouse에 실용적 대안
- **Pretrained model API != Usability**: SuperAnimal은 우수하나 DLC 2.3.11 API 제약. 안정적 릴리스 확인 필수
- **Progressive research workflow**: Geometric baseline -> YOLOv8 실패 -> SuperAnimal API 제한 -> Manual labeling으로 자연스럽게 수렴. 각 실험이 다음 방향을 제시
- **Transfer learning 필수**: YOLOv8 COCO pretrained에서 91% weight transfer. Scratch training은 비현실적

## 미해결 / 다음 단계
- Manual labeling (20 images, Roboflow, 2-3시간) -> YOLOv8 fine-tuning
- 예상 개선: confidence 0.5 -> 0.85+, loss 300K -> 15-30K (10-20x), paw detection 0% -> 70-80%
- fit_monocular.py에 `--detector yolo` 통합
- Multi-view triangulation 구현 (DANNCE 6-view 데이터 활용)

---
*Sources: 251114_ml_keypoint_detection_integration.md, 251114_monocular_mammal_fitting_poc.md, 251114_session_summary.md*
