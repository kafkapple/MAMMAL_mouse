# Unified Annotator Guide

**통합 Annotation 도구: Mask + Keypoint**

---

## Overview

`unified_annotator.py`는 SAM 기반 mask annotation과 semantic keypoint annotation을 하나의 인터페이스로 통합한 도구입니다.

### Key Features

✅ **Two Annotation Modes**:
- **Mask Mode**: SAM2 기반 foreground/background segmentation
- **Keypoint Mode**: Semantic keypoint annotation with visibility

✅ **Unified Interface**:
- Single Gradio UI
- Shared frame navigation
- Combined annotation storage

✅ **Modular Design**:
- Each mode can be used independently
- Compatible output formats
- Easy extension

---

## Installation

### Prerequisites

1. **SAM2** (for mask mode):
```bash
# Clone SAM2
cd ~/dev
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2

# Install
pip install -e .

# Download checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
  -O checkpoints/sam2_hiera_large.pt
```

2. **Dependencies**:
```bash
pip install gradio opencv-python numpy
```

---

## Usage

### Basic Command

```bash
python unified_annotator.py \
  --input data/frames \
  --output data/annotations \
  --mode both \
  --sam-checkpoint ~/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt
```

### Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--input, -i` | Input frame directory | Required | `data/frames` |
| `--output, -o` | Output annotation directory | Required | `data/annotations` |
| `--mode, -m` | Annotation mode | `both` | `mask`, `keypoint`, `both` |
| `--sam-checkpoint` | SAM2 checkpoint path | None | `~/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt` |
| `--port, -p` | Server port | `7860` | `7860` |

---

## Annotation Modes

### 1. Mask Mode (SAM-based)

**Purpose**: Segment foreground object (mouse) from background

**Workflow**:
1. Click **Mask Mode** tab
2. Select **Foreground** or **Background**
3. Click on image to add points
   - Green: Foreground (mouse body)
   - Red: Background (environment)
4. Click **Generate Mask** to run SAM
5. Review mask in mask display
6. Click **Save Annotation** if satisfied

**Use Case**:
- Silhouette extraction for mesh fitting (Step 2)
- Training data for segmentation models
- Object detection validation

### 2. Keypoint Mode (Semantic)

**Purpose**: Annotate anatomical landmarks

**Workflow**:
1. Click **Keypoint Mode** tab
2. Select keypoint from dropdown
   - nose, neck, spine_mid, hip, tail_base, left_ear, right_ear
3. Select visibility level
   - **visible**: Clear and unambiguous (1.0)
   - **occluded**: Partially visible (0.5)
   - **not_visible**: Cannot see (0.0)
4. Click on image to place keypoint
5. Repeat for all keypoints
6. Click **Save Annotation**

**Use Case**:
- MAMMAL mesh fitting (primary signal)
- Pose estimation ground truth
- Behavior analysis

---

## UI Guide

### Main Interface

```
+---------------------------+---------------------------+
|                           |    Frame Navigation       |
|                           |  [Slider] [Load] [Save]   |
|                           |                           |
|     Frame Display         |  +---------------------+  |
|   (Click to annotate)     |  |  🎯 Mask Mode       |  |
|                           |  | - Point Type        |  |
|                           |  | - Generate Mask     |  |
|                           |  | - Clear Mask        |  |
|                           |  +---------------------+  |
|                           |                           |
|                           |  +---------------------+  |
|    Status Message         |  |  📍 Keypoint Mode   |  |
|                           |  | - Keypoint selector |  |
|                           |  | - Visibility        |  |
|                           |  | - Mark Invisible    |  |
+---------------------------+---------------------------+
```

### Controls

**Frame Navigation**:
- **Slider**: Navigate frames
- **Load Frame**: Load selected frame
- **Save Annotation**: Save current annotations

**Mask Mode Tab**:
- **Point Type**: Foreground/Background radio
- **Generate Mask**: Run SAM inference
- **Clear Mask**: Remove all mask points and mask

**Keypoint Mode Tab**:
- **Keypoint**: Select which keypoint to annotate
- **Visibility**: visible/occluded/not_visible
- **Mark Not Visible**: Quick mark as invisible
- **Remove Keypoint**: Remove selected keypoint
- **Clear All Keypoints**: Remove all keypoints

**Summary**:
- **Update Summary**: Refresh annotation statistics

---

## Output Format

### Annotation JSON

**Location**: `{output_dir}/frame_XXXX_annotation.json`

**Format**:
```json
{
  "frame": "/path/to/frame_0000.png",
  "frame_idx": 0,

  "mask": {
    "points": [[100, 200], [150, 250]],
    "labels": [1, 0],
    "has_mask": true,
    "confidence": 0.95,
    "mask_area_pct": 25.5
  },

  "keypoints": {
    "nose": {"x": 120.0, "y": 80.0, "visibility": 1.0},
    "neck": {"x": 140.0, "y": 100.0, "visibility": 0.5},
    "spine_mid": {"x": 160.0, "y": 120.0, "visibility": 1.0}
  }
}
```

### Mask Image

**Location**: `{output_dir}/frame_XXXX_mask.png`

**Format**: Binary mask (0=background, 255=foreground)

---

## Integration with MAMMAL

### Convert Annotations to MAMMAL Format

**1. Extract keypoints**:
```bash
# Unified annotator output → keypoints.json
python extract_unified_keypoints.py \
  --input data/annotations \
  --output keypoints.json
```

**2. Convert to MAMMAL format**:
```bash
python convert_keypoints_to_mammal.py \
  --input keypoints.json \
  --output data/.../keypoints2d_undist/result_view_0.pkl \
  --num-frames 20
```

**3. Run mesh fitting**:
```bash
python fitter_articulation.py dataset=custom_cropped
```

### Use Masks for Step 2 Refinement

Masks from unified annotator can be used directly in MAMMAL's Step 2 optimization:

```yaml
# conf/dataset/custom.yaml
fitter:
  term_weights:
    mask: 3000  # Enable mask loss
```

---

## Comparison: Unified vs Separate Tools

### Unified Annotator

**Pros**:
- ✅ Single interface for both tasks
- ✅ Shared frame navigation
- ✅ Combined output (easier to manage)
- ✅ Consistent workflow

**Cons**:
- ⚠️ Requires SAM2 installation
- ⚠️ More complex UI
- ⚠️ Larger memory footprint

### Separate Tools

**Keypoint Annotator V2** (`keypoint_annotator_v2.py`):
- Specialized for keypoints only
- Zoom support
- Lighter weight
- No SAM dependency

**SAM Annotator** (mouse-SR):
- Specialized for masks only
- Advanced SAM features
- Optimized for segmentation

**Recommendation**:
- **Both needed**: Use unified annotator
- **Keypoints only**: Use `keypoint_annotator_v2.py`
- **Masks only**: Use SAM annotator from mouse-SR

---

## Tips & Best Practices

### Mask Annotation

1. **Start with foreground points**:
   - Click center of mouse body
   - Add 2-3 foreground points initially

2. **Add background for refinement**:
   - Click background if mask includes unwanted areas
   - Use background points for fine boundaries

3. **Iterative refinement**:
   - Generate mask → Review → Add more points → Regenerate

### Keypoint Annotation

1. **Annotation order** (recommended):
   - Start with spine: nose → neck → spine_mid → hip → tail_base
   - Then ears: left_ear, right_ear

2. **Use visibility levels wisely**:
   - **visible (1.0)**: Only if very confident
   - **occluded (0.5)**: When uncertain (still used in fitting!)
   - **not_visible (0.0)**: Only if truly cannot see

3. **Consistency across frames**:
   - Maintain similar annotation style
   - Use same criteria for visibility

### General

1. **Save frequently**:
   - Click Save after each frame
   - Annotations are cached per frame

2. **Check summary**:
   - Update summary to see progress
   - Verify all keypoints annotated

3. **Use navigation efficiently**:
   - Use slider for quick jumps
   - Use Load to refresh frame

---

## Troubleshooting

### "SAM not available!"

**Cause**: SAM2 not installed or checkpoint missing

**Solution**:
```bash
# Check SAM2 installation
ls ~/dev/segment-anything-2

# Check checkpoint
ls ~/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt

# If missing, follow installation steps above
```

### "Load a frame first!"

**Cause**: Trying to annotate before loading frame

**Solution**: Click "Load Frame" button first

### Annotations not saving

**Cause**: Output directory permissions

**Solution**:
```bash
# Check directory exists and writable
mkdir -p data/annotations
chmod 755 data/annotations
```

### Keypoints not visible

**Cause**: Point size too small or zoom level

**Solution**:
- Increase point size in config
- Check if visibility = 0.0 (invisible)

---

## Advanced Usage

### Custom Keypoint Names

Edit `UnifiedAnnotator` class:

```python
config = AnnotationConfig(
    input_dir="data/frames",
    output_dir="data/annotations",
    mode=AnnotationMode.BOTH,
    keypoint_names=[
        'nose', 'neck', 'spine_mid', 'hip', 'tail_base',
        'left_ear', 'right_ear',
        'left_paw', 'right_paw'  # ← Add custom keypoints
    ]
)
```

### Batch Processing

Annotate sequentially:

```python
# Auto-advance to next frame after save
# (Feature to be implemented)
```

### Export to Other Formats

```python
# Export to COCO format
python export_to_coco.py --input data/annotations

# Export to DeepLabCut format
python export_to_dlc.py --input data/annotations
```

---

## Summary

### Quick Start

```bash
# 1. Launch annotator
python unified_annotator.py \
  -i data/frames \
  -o data/annotations \
  --sam-checkpoint ~/dev/segment-anything-2/checkpoints/sam2_hiera_large.pt

# 2. Open browser: http://localhost:7860

# 3. Annotate frames:
#    - Load frame
#    - Mask Mode: Add points → Generate mask
#    - Keypoint Mode: Click keypoints
#    - Save annotation

# 4. Convert to MAMMAL format
python convert_keypoints_to_mammal.py -i keypoints.json -o result_view_0.pkl -n 20
```

### When to Use What

| Task | Tool | Why |
|------|------|-----|
| Both mask + keypoints | `unified_annotator.py` | Single interface |
| Keypoints only | `keypoint_annotator_v2.py` | Lighter, zoom support |
| Masks only | SAM annotator (mouse-SR) | Specialized features |

---

## References

- MAMMAL mesh fitting: `fitter_articulation.py`
- Keypoint converter: `convert_keypoints_to_mammal.py`
- SAM annotator (mouse-SR): `/home/joon/dev/mouse-super-resolution/sam_annotator`
- Keypoint annotator V2: `keypoint_annotator_v2.py`
