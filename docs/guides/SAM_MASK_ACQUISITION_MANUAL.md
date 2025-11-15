# SAM (Segment Anything Model) Mask Acquisition Manual

## 개요

이 매뉴얼은 MAMMAL Mouse 프로젝트에서 SAM을 사용하여 마우스 세그멘테이션 마스크를 획득하는 다양한 방법들을 체계적으로 정리합니다.

**작성일**: 2025-11-04
**프로젝트**: MAMMAL Mouse 3D Pose Estimation  
**SAM 버전**: ViT-H (sam_vit_h_4b8939.pth)

---

## 목차

1. [SAM 기본 개념](#1-sam-기본-개념)
2. [SAM Mask 획득 방법들](#2-sam-mask-획득-방법들)
3. [각 방법의 장단점 비교](#3-각-방법의-장단점-비교)
4. [코드베이스 파일 가이드](#4-코드베이스-파일-가이드)
5. [실전 사용 가이드](#5-실전-사용-가이드)
6. [문제 해결 (Troubleshooting)](#6-문제-해결-troubleshooting)

---

## 1. SAM 기본 개념

### 1.1 SAM이란?

**SAM (Segment Anything Model)**은 Meta AI에서 개발한 강력한 세그멘테이션 모델입니다.

**주요 특징**:
- Zero-shot 세그멘테이션 (학습 없이 바로 사용 가능)
- 다양한 프롬프트 지원 (점, 박스, 마스크)
- 고품질 마스크 생성

**모델 종류**:
- \`vit_h\` (ViT-Huge): 가장 높은 품질, 가장 느림 ⭐ **현재 사용 중**
- \`vit_l\` (ViT-Large): 중간 품질, 중간 속도
- \`vit_b\` (ViT-Base): 빠른 속도, 낮은 품질

### 1.2 SAM 설치 및 체크포인트

\`\`\`bash
# SAM 설치
pip install segment-anything

# 체크포인트 다운로드 위치
checkpoints/sam_vit_h_4b8939.pth
\`\`\`

**체크포인트 다운로드**:
\`\`\`bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
\`\`\`

---

## 2. SAM Mask 획득 방법들

### 2.1 방법 1: Automatic Mask Generation (자동 마스크 생성)

**설명**: SAM이 자동으로 이미지에서 모든 객체를 감지하고 마스크를 생성합니다.

**사용 시기**:
- 마우스 위치를 모를 때
- 전체 이미지에서 모든 객체를 찾고 싶을 때
- 배치 처리 (여러 프레임 일괄 처리)

**관련 파일**:
- \`preprocessing_utils/sam_inference.py\` (SAM 래퍼 클래스)
- \`test/test_sam.py\` (테스트 스크립트)

**코드 예제**:

\`\`\`python
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. SAM 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)

# 2. 자동 마스크 생성기 설정
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,           # 그리드 포인트 수 (높을수록 정밀)
    pred_iou_thresh=0.86,         # IoU 임계값 (높을수록 품질 높은 마스크만)
    stability_score_thresh=0.92,  # 안정성 임계값
    min_mask_region_area=100,     # 최소 마스크 크기 (픽셀)
)

# 3. 이미지 로드 및 마스크 생성
frame = cv2.imread("frame.png")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(frame_rgb)

# 4. 결과 확인
print(f"감지된 마스크 개수: {len(masks)}")
for i, mask in enumerate(masks):
    print(f"Mask {i}: Area={mask['area']}, IoU={mask['predicted_iou']:.3f}")
\`\`\`

**마스크 결과 구조**:
\`\`\`python
{
    'segmentation': np.ndarray,  # (H, W) binary mask
    'area': int,                 # 마스크 면적 (픽셀 수)
    'bbox': [x, y, w, h],       # 바운딩 박스
    'predicted_iou': float,      # 예측 IoU
    'stability_score': float,    # 안정성 점수
}
\`\`\`

**마우스 마스크 선택 전략**:
\`\`\`python
# 전략 1: 가장 큰 마스크 선택
largest_mask = max(masks, key=lambda x: x['area'])['segmentation']

# 전략 2: 중간 크기 마스크 (너무 크면 아레나, 너무 작으면 노이즈)
size_filtered = [m for m in masks if 0.05 < m['area']/frame.size < 0.30]
mouse_mask = max(size_filtered, key=lambda x: x['predicted_iou'])['segmentation']

# 전략 3: IoU 기반 선택
best_mask = max(masks, key=lambda x: x['predicted_iou'])['segmentation']
\`\`\`

**장점**:
- ✅ 완전 자동 - 수동 입력 불필요
- ✅ 여러 객체 동시 감지
- ✅ 배치 처리 가능

**단점**:
- ❌ 느림 (프레임당 5-10초)
- ❌ 마우스/아레나 구분 어려움
- ❌ 후처리 필요 (가장 큰 마스크가 아레나일 수 있음)

---

### 2.2 방법 2: Point Prompt (포인트 프롬프트)

**설명**: 마우스 중심에 점을 직접 지정하여 해당 영역의 마스크를 생성합니다.

**사용 시기**:
- 마우스 대략적 위치를 알 때
- 빠른 마스크 생성이 필요할 때
- 특정 객체만 선택하고 싶을 때

**관련 파일**:
- \`sam_point_prompt.py\` (포인트 프롬프트 구현)

**코드 예제**:

\`\`\`python
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# 1. SAM Predictor 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

# 2. 이미지 설정
frame = cv2.imread("frame.png")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
predictor.set_image(frame_rgb)

# 3. 포인트 프롬프트 정의
# 마우스 중심 좌표 (x, y)
H, W = frame.shape[:2]
mouse_point = np.array([[W // 2, H // 2]])  # 화면 중심
point_label = np.array([1])  # 1 = foreground (전경)

# 4. 마스크 생성
masks, scores, logits = predictor.predict(
    point_coords=mouse_point,
    point_labels=point_label,
    multimask_output=True,  # 3개의 후보 마스크 생성
)

# 5. 가장 좋은 마스크 선택
best_idx = np.argmax(scores)
best_mask = masks[best_idx]

print(f"생성된 마스크 개수: {len(masks)}")
print(f"최고 점수: {scores[best_idx]:.3f}")
print(f"마스크 커버리지: {best_mask.sum() / best_mask.size * 100:.2f}%")
\`\`\`

**장점**:
- ✅ 빠름 (프레임당 1-2초)
- ✅ 원하는 객체만 정확히 선택
- ✅ 간단한 인터페이스

**단점**:
- ❌ 수동 포인트 지정 필요 (또는 추정 필요)
- ❌ 마우스가 아레나 내부에 있을 때 아레나 전체를 선택할 수 있음

---

### 2.3 방법 3: Negative Prompt (네거티브 프롬프트)

**설명**: Positive 포인트 (마우스)와 Negative 포인트 (아레나 배경)를 동시에 지정하여 마우스만 정확히 선택합니다.

**사용 시기**:
- 마우스가 큰 배경 (아레나) 안에 있을 때
- 포인트 프롬프트만으로 아레나 전체가 선택될 때
- 더 정밀한 세그멘테이션이 필요할 때

**관련 파일**:
- \`sam_mouse_only.py\` (네거티브 프롬프트 구현)

**코드 예제**:

\`\`\`python
import numpy as np

# 1. Positive + Negative 포인트 정의
mouse_center = (W // 2, H // 2)  # 마우스 중심

# 아레나 경계 포인트들 (배경으로 지정)
arena_negatives = [
    (W // 2, H // 2 - 100),  # 위
    (W // 2, H // 2 + 100),  # 아래
    (W // 2 - 150, H // 2),  # 왼쪽
    (W // 2 + 150, H // 2),  # 오른쪽
]

# 2. 포인트 배열 생성
input_points = np.array([mouse_center] + arena_negatives)
input_labels = np.array([1] + [0] * len(arena_negatives))
# 1 = foreground (마우스), 0 = background (아레나)

# 3. 마스크 생성
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

# 4. 가장 작은 마스크 선택 (마우스만 포함)
coverages = [mask.sum() / mask.size for mask in masks]
best_idx = None
for i, (mask, score, coverage) in enumerate(zip(masks, scores, coverages)):
    if 0.05 < coverage < 0.15:  # 5-15% 범위 (마우스 크기)
        if best_idx is None or scores[i] > scores[best_idx]:
            best_idx = i
\`\`\`

**장점**:
- ✅ 배경 제외 가능
- ✅ 마우스만 정확히 선택
- ✅ 포인트 프롬프트보다 정밀

**단점**:
- ❌ 네거티브 포인트 위치 결정 필요
- ❌ 여전히 수동 입력 필요

---

## 3. 각 방법의 장단점 비교

| 방법 | 속도 | 정확도 | 자동화 | 사용 난이도 | 추천 상황 |
|------|------|--------|--------|-------------|-----------|
| **Automatic** | ⭐ (느림) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 쉬움 | 배치 처리, 탐색 |
| **Point Prompt** | ⭐⭐⭐⭐ (빠름) | ⭐⭐⭐⭐ | ⭐⭐ | 보통 | 실시간, 인터랙티브 |
| **Negative Prompt** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 어려움 | 정밀 세그멘테이션 |

---

## 4. 코드베이스 파일 가이드

### 4.1 핵심 모듈

| 파일 | 설명 | 사용 목적 |
|------|------|-----------|
| \`preprocessing_utils/sam_inference.py\` | SAM 래퍼 클래스 | 재사용 가능한 SAM 인터페이스 |
| \`preprocessing_utils/silhouette_renderer.py\` | Silhouette 렌더링 | PyTorch3D 기반 렌더링 |

### 4.2 실험/테스트 스크립트

| 파일 | 설명 | 언제 사용 |
|------|------|-----------|
| \`test/test_sam.py\` | SAM 기본 테스트 | SAM 작동 확인 |
| \`test/preprocess_sam_improved.py\` | SAM 전처리 (개선 버전) | 전체 파이프라인 테스트 |
| \`sam_point_prompt.py\` | 포인트 프롬프트 테스트 | 포인트 기반 마스크 생성 |
| \`sam_mouse_only.py\` | 네거티브 프롬프트 테스트 | 배경 제외 마스크 생성 |
| \`test/visualize_sam_mouse_detection.py\` | SAM 결과 시각화 | 마스크 품질 확인 |

---

## 5. 실전 사용 가이드

### 5.1 첫 프레임 마스크 생성 (빠른 시작)

\`\`\`bash
# 1. Point Prompt 방법 (가장 빠름)
python sam_point_prompt.py

# 출력:
# - sam_point_prompt_results.png (시각화)
# - sam_point_prompt_best_mask.png (마스크)
# - sam_point_prompt_mask.npy (numpy 배열)
\`\`\`

### 5.2 마스크 품질 확인

\`\`\`bash
# SAM 마스크 시각화
python test/visualize_sam_mouse_detection.py

# 출력:
# - sam_mouse_detection_visualization.png
# - sam_mouse_bbox.png
\`\`\`

---

## 6. 문제 해결 (Troubleshooting)

### 6.1 SAM이 아레나 전체를 선택하는 문제

**증상**: Point prompt 사용 시 38.9% coverage (아레나 전체)

**해결책 1 - Negative Prompt**:
\`\`\`python
arena_negatives = [
    (W // 2, H // 2 - 100),  # 위
    (W // 2, H // 2 + 100),  # 아래
]
\`\`\`

**해결책 2 - Color-based Pre-filtering**:
\`\`\`python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
dark_mask = gray < 100  # 어두운 영역
\`\`\`

### 6.2 마스크 반전 (Inversion) 문제

**증상**: SAM 마스크가 하얀색으로 아레나, 검은색으로 마우스 표시

**해결책**:
\`\`\`python
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
mask = 255 - mask  # 반전
\`\`\`

### 6.3 마스크가 조각조각 흩어지는 문제

**해결책 - Morphological Operations**:
\`\`\`python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
\`\`\`

---

**작성자**: Claude Code  
**프로젝트**: MAMMAL Mouse
