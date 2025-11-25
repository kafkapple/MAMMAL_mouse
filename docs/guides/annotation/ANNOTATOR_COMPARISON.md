# Annotator Comparison & Usage Guide

**어떤 annotator를 사용할까?**

---

## TL;DR

| Need | Use | Command |
|------|-----|---------|
| **Mask + Keypoints 둘 다** | `unified_annotator.py` | `./run_unified_annotator.sh` |
| **Keypoints만** | `keypoint_annotator_v2.py` | `python keypoint_annotator_v2.py data/frames` |
| **Masks만 (정밀)** | SAM annotator (mouse-SR) | `python -m sam_annotator` (mouse-SR) |

---

## 도구 비교

### 1. Unified Annotator (NEW!)

**파일**: `unified_annotator.py`

**특징**:
- ✅ Mask + Keypoint 통합 인터페이스
- ✅ 하나의 JSON에 모든 annotation 저장
- ✅ 프레임 navigation 공유
- ✅ MAMMAL 포맷 호환

**사용 시기**:
- Mask와 keypoint 둘 다 필요
- 통합 workflow 선호
- 데이터 관리 단순화

**실행**:
```bash
# Quick start
./run_unified_annotator.sh data/frames data/annotations both

# Or manually
python unified_annotator.py \
  -i data/frames \
  -o data/annotations \
  --sam-checkpoint ~/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt
```

**장점**:
- 한 번에 모든 annotation
- 일관된 인터페이스
- 관리 편리

**단점**:
- SAM2 설치 필요 (mask mode)
- 메모리 사용량 높음
- UI 복잡

---

### 2. Keypoint Annotator V2

**파일**: `keypoint_annotator_v2.py`

**특징**:
- ✅ Keypoint 전용
- ✅ Zoom 지원
- ✅ Visibility control (1.0, 0.5, 0.0)
- ✅ 가벼움 (SAM 불필요)

**사용 시기**:
- Keypoint만 필요
- Zoom 기능 필요
- 가벼운 도구 선호

**실행**:
```bash
python keypoint_annotator_v2.py data/100-KO-male-56-20200615_cropped \
  --output keypoints.json
```

**장점**:
- Zoom 기능
- 빠른 속도
- 낮은 메모리
- SAM 불필요

**단점**:
- Keypoint만 가능
- Mask annotation 불가

---

### 3. SAM Annotator (mouse-SR)

**파일**: `/home/joon/dev/mouse-super-resolution/sam_annotator/`

**특징**:
- ✅ SAM 전용
- ✅ Foreground/Background point 기반
- ✅ Interactive mask 생성
- ✅ Hydra config 지원

**사용 시기**:
- Mask만 필요
- SAM 고급 기능 사용
- mouse-SR 프로젝트와 통합

**실행**:
```bash
cd /home/joon/dev/mouse-super-resolution
python -m sam_annotator \
  data.input_dir=~/data/frames \
  data.output_dir=~/data/masks
```

**장점**:
- SAM 특화
- Hydra 설정
- mouse-SR 통합

**단점**:
- Mask만 가능
- Keypoint annotation 불가

---

## 기능 비교표

| Feature | Unified | Keypoint V2 | SAM (mouse-SR) |
|---------|---------|-------------|----------------|
| **Mask Annotation** | ✅ | ❌ | ✅ |
| **Keypoint Annotation** | ✅ | ✅ | ❌ |
| **SAM2 Support** | ✅ | ❌ | ✅ |
| **Zoom** | ❌ | ✅ | ❌ |
| **Visibility Control** | ✅ | ✅ | ❌ |
| **Unified Storage** | ✅ | ❌ | ❌ |
| **MAMMAL Compatible** | ✅ | ✅ | ✅ |
| **Dependencies** | High | Low | Medium |
| **Memory Usage** | High | Low | Medium |

---

## Workflow 선택 가이드

### Scenario 1: Full MAMMAL Pipeline

**목표**: Mask + Keypoint → MAMMAL mesh fitting

**Workflow**:
```bash
# 1. Annotate with unified tool
./run_unified_annotator.sh data/frames data/annotations both

# 2. Extract keypoints
python extract_unified_keypoints.py \
  -i data/annotations \
  -o keypoints.json

# 3. Convert to MAMMAL
python convert_keypoints_to_mammal.py \
  -i keypoints.json \
  -o data/.../result_view_0.pkl \
  -n 20

# 4. Run fitting
python fitter_articulation.py dataset=custom
```

**도구**: `unified_annotator.py`

---

### Scenario 2: Keypoint Only

**목표**: Keypoint annotation → MAMMAL fitting (Step 0-1)

**Workflow**:
```bash
# 1. Annotate keypoints
python keypoint_annotator_v2.py data/frames

# 2. Convert to MAMMAL
python convert_keypoints_to_mammal.py \
  -i keypoints.json \
  -o result_view_0.pkl \
  -n 20

# 3. Run fitting (without mask)
python fitter_articulation.py \
  dataset=custom \
  fitter.term_weights.mask=0
```

**도구**: `keypoint_annotator_v2.py`

---

### Scenario 3: Mask Only (Segmentation)

**목표**: Mask annotation → Training data / Validation

**Workflow**:
```bash
# Using mouse-SR SAM annotator
cd /home/joon/dev/mouse-super-resolution
python -m sam_annotator \
  data.input_dir=~/data/frames \
  data.output_dir=~/data/masks
```

**도구**: SAM annotator (mouse-SR)

---

## 출력 포맷 비교

### Unified Annotator

**파일 구조**:
```
data/annotations/
├── frame_0000_annotation.json  # Mask + Keypoints
├── frame_0000_mask.png          # Binary mask
├── frame_0001_annotation.json
└── frame_0001_mask.png
```

**JSON 포맷**:
```json
{
  "mask": {
    "points": [[100, 200]],
    "labels": [1],
    "confidence": 0.95
  },
  "keypoints": {
    "nose": {"x": 120, "y": 80, "visibility": 1.0}
  }
}
```

### Keypoint Annotator V2

**파일 구조**:
```
keypoints.json
```

**JSON 포맷**:
```json
{
  "frame_000000": {
    "nose": {"x": 120, "y": 80, "visibility": 1.0}
  }
}
```

### SAM Annotator (mouse-SR)

**파일 구조**:
```
data/annotations/
├── frame_0000_annotation.json  # SAM points only
└── frame_0000_mask.png
```

**JSON 포맷**:
```json
{
  "points": [[100, 200]],
  "labels": [1],
  "confidence": 0.95,
  "has_mask": true
}
```

---

## 통합 예제

### Example: 두 도구 조합 사용

**Scenario**: Keypoint V2 (zoom 필요) + SAM (mask 필요)

```bash
# 1. Annotate keypoints with zoom
python keypoint_annotator_v2.py data/frames -o keypoints.json

# 2. Annotate masks with SAM
cd /home/joon/dev/mouse-super-resolution
python -m sam_annotator data.input_dir=~/data/frames data.output_dir=~/data/masks

# 3. Merge annotations (manual script needed)
# TODO: Create merge script

# 4. Convert and use with MAMMAL
```

---

## 추천 사항

### 일반 사용자

**추천**: `unified_annotator.py`
- 모든 기능 포함
- 한 번에 처리
- 데이터 관리 편리

### 성능 중시

**추천**: `keypoint_annotator_v2.py`
- 가벼움
- 빠른 속도
- 필요시 SAM 별도 사용

### 프로젝트 통합

**MAMMAL 프로젝트**:
- `unified_annotator.py` 또는 `keypoint_annotator_v2.py`

**mouse-SR 프로젝트**:
- SAM annotator (mouse-SR)

---

## 설치 가이드

### Unified Annotator

```bash
# 1. SAM2 설치 (mask mode용)
cd ~/dev
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
  -O checkpoints/sam2_hiera_large.pt

# 2. 의존성
pip install gradio opencv-python numpy
```

### Keypoint V2

```bash
# 의존성만
pip install gradio opencv-python numpy
```

### SAM Annotator (mouse-SR)

```bash
# mouse-SR 프로젝트 설정 참조
cd /home/joon/dev/mouse-super-resolution
# (기존 설정 사용)
```

---

## Summary

| 상황 | 도구 | 이유 |
|------|------|------|
| 빠른 시작 | Keypoint V2 | 가장 간단 |
| 완전한 pipeline | Unified | 모든 기능 |
| Mask 특화 | SAM (mouse-SR) | 고급 기능 |
| Zoom 필요 | Keypoint V2 | Zoom 지원 |
| 메모리 제약 | Keypoint V2 | 가벼움 |

**최종 추천**:
- 첫 사용자: `keypoint_annotator_v2.py` (간단)
- 전체 workflow: `unified_annotator.py` (통합)
