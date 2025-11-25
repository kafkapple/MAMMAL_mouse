# Annotation Tools Overview

**MAMMAL_mouse 프로젝트의 annotation 도구들**

---

## 사용 가능한 도구

### 1. 🎯 Unified Annotator (통합 도구)
- **파일**: `unified_annotator.py`
- **기능**: Mask + Keypoint 통합
- **실행**: `./run_unified_annotator.sh`
- **가이드**: [UNIFIED_ANNOTATOR_GUIDE.md](UNIFIED_ANNOTATOR_GUIDE.md)

### 2. 📍 Keypoint Annotator V2
- **파일**: `keypoint_annotator_v2.py`
- **기능**: Keypoint 전용 (zoom 지원)
- **실행**: `python keypoint_annotator_v2.py data/frames`
- **가이드**: [KEYPOINT_ANNOTATOR_V2_GUIDE.md](KEYPOINT_ANNOTATOR_V2_GUIDE.md)

### 3. 🔄 Format Converters
- **Keypoint JSON → MAMMAL PKL**: `convert_keypoints_to_mammal.py`
- **Unified → Keypoint JSON**: `extract_unified_keypoints.py`

---

## Quick Start

### Keypoint만 필요 (가장 간단)
```bash
python keypoint_annotator_v2.py data/100-KO-male-56-20200615_cropped
```

### Mask + Keypoint 둘 다 필요
```bash
./run_unified_annotator.sh data/frames data/annotations both
```

---

## 전체 Workflow

```bash
# 1. Annotation
python keypoint_annotator_v2.py data/frames

# 2. Convert to MAMMAL format
python convert_keypoints_to_mammal.py \
  -i keypoints.json \
  -o result_view_0.pkl \
  -n 20

# 3. Mesh fitting
python fitter_articulation.py dataset=custom
```

---

## 문서

- 📖 [Quick Start](KEYPOINT_QUICK_START.md) - 빠른 시작 가이드
- 📖 [Workflow](docs/KEYPOINT_WORKFLOW.md) - 상세 워크플로우
- 📖 [Unified Guide](UNIFIED_ANNOTATOR_GUIDE.md) - 통합 annotator 가이드
- 📖 [Comparison](ANNOTATOR_COMPARISON.md) - 도구 비교

---

## 도구 선택

| Need | Tool |
|------|------|
| Keypoints only | `keypoint_annotator_v2.py` |
| Mask + Keypoints | `unified_annotator.py` |
| Zoom support | `keypoint_annotator_v2.py` |
| Lightweight | `keypoint_annotator_v2.py` |

자세한 비교는 [ANNOTATOR_COMPARISON.md](ANNOTATOR_COMPARISON.md) 참조.
