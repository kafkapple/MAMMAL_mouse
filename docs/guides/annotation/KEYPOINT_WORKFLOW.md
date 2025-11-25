# Keypoint Annotation and Mesh Fitting Workflow

**Date**: 2025-11-18
**Purpose**: Complete workflow for manual keypoint annotation and MAMMAL mesh fitting

---

## Overview

This document describes the complete workflow for annotating keypoints manually and using them with MAMMAL's mesh fitting system.

### Key Features

- ✅ **Flexible keypoint count**: 1-22 keypoints (not all required)
- ✅ **Confidence-based filtering**: Missing keypoints automatically ignored
- ✅ **MAMMAL-compatible format**: Direct integration with existing fitter
- ✅ **Manual annotation tool**: Gradio-based interactive annotator

---

## MAMMAL Mesh Fitting Mechanism

### Data Requirements

MAMMAL's mesh fitter uses:
1. **Keypoints** (primary): 2D keypoint coordinates with confidence
2. **Masks** (optional): Silhouette masks for refinement

### Keypoint Format

**MAMMAL PKL format**:
```python
# Shape: (num_frames, 22, 3)
# [:, :, 0] = x coordinate
# [:, :, 1] = y coordinate
# [:, :, 2] = confidence (0.0 - 1.0)
```

**Confidence handling**:
- `confidence >= 0.25`: Keypoint used in optimization
- `confidence < 0.25`: Keypoint ignored (automatic filtering)
- `confidence = 0.0`: Missing keypoint (not annotated)

### How MAMMAL Uses Keypoints

**Loss computation** (`fitter_articulation.py:214`):
```python
# Reprojection error weighted by confidence
diff = (J2d_projected - target_2d) * confidence
loss = mean(norm(diff * keypoint_weight))
```

**Result**: Missing keypoints (conf=0) contribute 0 to the loss → automatically filtered out!

### 3-Stage Optimization

| Stage | Optimized Params | Keypoint Weight | Mask Weight | Purpose |
|-------|------------------|-----------------|-------------|---------|
| **Step 0** | rotation, translation, scale | Normal | 0 | Coarse alignment |
| **Step 1** | + thetas, bone_lengths | Normal | 0 | Pose fitting |
| **Step 2** | + chest_deformer | Foot x10 | 3000 | Silhouette refinement |

**Key points**:
- Step 0: Only global transform (position/scale)
- Step 1: Full pose optimization with keypoint loss
- Step 2: Add mask loss for fine details (especially feet)

---

## Workflow

### 1. Manual Keypoint Annotation

**Launch annotator**:
```bash
python keypoint_annotator_v2.py \
    data/100-KO-male-56-20200615_cropped \
    --output data/annotations/keypoints.json
```

**Available keypoints** (7 core keypoints):
- `nose`: Tip of nose
- `neck`: Base of neck
- `spine_mid`: Middle of spine
- `hip`: Hip/pelvis region
- `tail_base`: Base of tail
- `left_ear`: Left ear
- `right_ear`: Right ear

**Visibility levels**:
- `visible` (1.0): Clear and unambiguous
- `occluded` (0.5): Partially visible or uncertain
- `not_visible` (0.0): Cannot see (will be ignored)

**Output format** (JSON):
```json
{
  "frame_000000": {
    "nose": {"x": 50.0, "y": 30.0, "visibility": 1.0},
    "neck": {"x": 60.0, "y": 40.0, "visibility": 0.5},
    "spine_mid": {"x": 0, "y": 0, "visibility": 0.0}
  }
}
```

### 2. Convert to MAMMAL Format

**Run converter**:
```bash
python convert_keypoints_to_mammal.py \
    --input data/annotations/keypoints.json \
    --output data/100-KO-male-56-20200615_cropped/keypoints2d_undist/result_view_0.pkl \
    --num-frames 20 \
    --visualize 0
```

**What it does**:
1. Loads JSON annotations
2. Maps to MAMMAL 22-keypoint format
3. Sets confidence from visibility
4. Fills missing keypoints with conf=0.0
5. Saves as pickle (NumPy array)

**Keypoint mapping**:
```python
{
    'nose': 0,
    'neck': 1,
    'spine_mid': 2,
    'hip': 3,
    'tail_base': 4,
    'left_ear': 5,
    'right_ear': 6,
    # Indices 7-21: Reserved (confidence=0.0)
}
```

### 3. Set Up Dataset Config

**Create dataset config** (`conf/dataset/custom_cropped.yaml`):
```yaml
# @package _global_

data:
  data_dir: data/100-KO-male-56-20200615_cropped/
  views_to_use: [0]  # Single camera view

fitter:
  start_frame: 0
  end_frame: 19
  interval: 1
  render_cameras: [0]
  with_render: true
  keypoint_num: 22  # MAMMAL expects 22 keypoints
```

**Directory structure**:
```
data/100-KO-male-56-20200615_cropped/
├── frame_000000_cropped.png
├── frame_000000_mask.png
├── keypoints2d_undist/
│   └── result_view_0.pkl       # ← Converted keypoints
├── simpleclick_undist/
│   └── 0.mp4                   # Mask video (optional)
└── videos_undist/
    └── 0.mp4                   # RGB video (optional)
```

### 4. Run Mesh Fitting

**Using MAMMAL fitter**:
```bash
python fitter_articulation.py \
    dataset=custom_cropped \
    fitter.start_frame=0 \
    fitter.end_frame=19
```

**Fitting process**:
1. Loads keypoints from PKL file
2. Filters keypoints by confidence (≥ 0.25)
3. Runs 3-stage optimization
4. Saves fitted parameters and visualizations

---

## Keypoint Requirements

### Minimum Requirements

**For basic fitting**:
- ✅ **1-3 keypoints** sufficient for coarse alignment
- ✅ Recommended: `nose`, `hip`, `tail_base` (spine landmarks)

**For good quality**:
- ✅ **5-7 keypoints** recommended
- ✅ All 7 core keypoints ideally

**For best results**:
- ✅ **10+ keypoints** if available
- ✅ Include limb landmarks (ears, paws)

### Keypoint Quality vs Quantity

| Scenario | Num Keypoints | Expected Result |
|----------|---------------|-----------------|
| 1-2 keypoints | Poor | Position only, no pose |
| 3-4 keypoints | Fair | Basic body orientation |
| 5-7 keypoints (core) | Good | Full body pose |
| 10+ keypoints | Excellent | Fine details |

---

## Alternative: Extract from SAM Annotations

**If you used SAM annotator**:
```bash
# Extract SAM click points (not semantic keypoints!)
python extract_sam_keypoints.py \
    data/100-KO-male-56-20200615_cropped \
    --output keypoints_from_sam.json
```

**⚠️ Warning**: SAM clicks are generic points (foreground/background), not semantic keypoints. They will have low confidence and may not align with anatomical landmarks.

**Recommended**: Use manual annotation (`keypoint_annotator_v2.py`) for better results.

---

## Testing

**Run full pipeline test**:
```bash
./test_keypoint_conversion.sh
```

**Manual verification**:
```python
import pickle
import numpy as np

# Load converted data
with open('data/.../result_view_0.pkl', 'rb') as f:
    kpts = pickle.load(f)

print(f"Shape: {kpts.shape}")  # (num_frames, 22, 3)

# Check frame 0
frame0 = kpts[0]
for i, (x, y, conf) in enumerate(frame0):
    if conf > 0:
        print(f"Keypoint {i}: ({x:.1f}, {y:.1f}) conf={conf:.2f}")
```

---

## Troubleshooting

### "No keypoints found"
- Check JSON format matches expected structure
- Verify frame names match (e.g., `frame_000000`)

### "Keypoint index out of range"
- Ensure `--num-frames` matches actual frame count
- Check frame indices in JSON are < num_frames

### "Mesh fitting fails / poor results"
- Annotate at least 5 core keypoints
- Check keypoint accuracy (especially spine landmarks)
- Increase visibility/confidence for ambiguous points

### "Mask shape mismatch"
- Ensure mask images match cropped image size
- Check mask file naming (frame_*_mask.png)

---

## Summary

### Key Scripts

1. **`keypoint_annotator_v2.py`**: Manual annotation tool
2. **`convert_keypoints_to_mammal.py`**: JSON → PKL converter
3. **`extract_sam_keypoints.py`**: Extract SAM clicks (optional)
4. **`test_keypoint_conversion.sh`**: Full pipeline test

### Data Flow

```
Manual Annotation → JSON → Converter → PKL → MAMMAL Fitter → 3D Mesh
(Gradio UI)                            (22 keypoints)
```

### MAMMAL Compatibility

- ✅ **Format**: NumPy array (num_frames, 22, 3)
- ✅ **Confidence filtering**: Automatic (threshold 0.25)
- ✅ **Missing keypoints**: Set confidence=0.0
- ✅ **Partial annotations**: 1-22 keypoints supported

---

## References

- MAMMAL fitter: `fitter_articulation.py`
- Data loader: `scripts/analysis/data_seaker_video_new.py`
- Example dataset: `data/examples/markerless_mouse_1_nerf/`
