# Manual Keypoint Labeling Guide for MAMMAL Mouse

## Overview

This guide explains how to manually label 10-20 mouse images for YOLOv8-Pose fine-tuning.

## Goal

Improve keypoint detection accuracy, especially for:
- Paws (front and rear)
- Eyes
- More precise spine/tail positions

## MAMMAL 22 Keypoints Definition

```
0:  nose
1:  left_ear
2:  right_ear
3:  left_eye
4:  right_eye
5:  head_center
6:  spine_1  (neck)
7:  spine_2
8:  spine_3
9:  spine_4
10: spine_5
11: spine_6
12: spine_7
13: spine_8  (tail base)
14: left_front_paw
15: right_front_paw
16: left_rear_paw
17: right_rear_paw
18: tail_base
19: tail_mid
20: tail_tip
21: centroid (body center)
```

## Labeling Tools Options

### Option 1: CVAT (Computer Vision Annotation Tool)
**Recommended for team work**
- Web-based interface
- Supports keypoint annotation
- Can export to YOLO format
- Setup: `docker run -p 8080:8080 cvat/server`

### Option 2: Label Studio
- Modern UI
- Python-friendly
- Good for solo work

### Option 3: Roboflow
- Cloud-based
- Automatic format conversion
- Free tier available

### Option 4: DeepLabCut GUI
- Built-in to DLC
- But we're not using DLC for training, so less ideal

## Recommended Workflow

### Step 1: Image Selection (5-10 minutes)

Select 10-20 diverse images from dataset:
```bash
# Sample diverse frames
python -c "
import random
from pathlib import Path

# Get all training images
imgs = sorted(Path('data/fauna_mouse').rglob('*_rgb.png'))
print(f'Total images: {len(imgs)}')

# Sample evenly distributed
n_samples = 20
indices = [int(i * len(imgs) / n_samples) for i in range(n_samples)]
sampled = [imgs[i] for i in indices]

# Copy to labeling directory
import shutil
out_dir = Path('data/manual_labeling/images')
out_dir.mkdir(parents=True, exist_ok=True)

for i, img in enumerate(sampled):
    shutil.copy(img, out_dir / f'sample_{i:03d}.png')
    print(f'Copied: {img.name}')
"
```

**Selection criteria:**
- Different poses (standing, walking, turning)
- Different viewing angles
- Different lighting conditions
- Clear visibility of all body parts

### Step 2: Labeling (2-3 hours)

**Estimated time:**
- 5-10 minutes per image
- 20 images × 7 min = ~2.3 hours

**Tips for accurate labeling:**
1. **Start with obvious points** (nose, ears, tail tip)
2. **Use spine geometric progression** (8 evenly spaced points)
3. **Mark paws at joint center**, not paw tip
4. **Use visibility flags**:
   - 2 = visible and labeled
   - 1 = occluded but estimated
   - 0 = not visible at all

### Step 3: Quality Check (15 minutes)

Review labeled data:
```python
# Visualize labels
python preprocessing_utils/visualize_yolo_labels.py \
    --images data/manual_labeling/images \
    --labels data/manual_labeling/labels \
    --output data/manual_labeling/viz
```

**Check for:**
- All 22 keypoints present
- Consistent left/right labeling
- Spine progression makes sense
- No outlier points

### Step 4: Convert to YOLO Format

If using non-YOLO tool:
```bash
python preprocessing_utils/convert_to_yolo_pose.py \
    --input data/manual_labeling/labels.json \
    --output data/manual_labeling/yolo \
    --format cvat  # or labelstudio, roboflow
```

### Step 5: Merge with Existing Dataset

```bash
# Combine manual labels with geometric labels
python preprocessing_utils/merge_datasets.py \
    --manual data/manual_labeling/yolo \
    --geometric data/yolo_mouse_pose \
    --output data/yolo_mouse_pose_enhanced
```

## YOLO Label Format

Each label file: `<image_name>.txt`
```
<class_id> <x_center> <y_center> <width> <height> <kpt1_x> <kpt1_y> <kpt1_v> ... <kpt22_x> <kpt22_y> <kpt22_v>
```

Example:
```
0 0.5 0.5 0.3 0.6 0.45 0.25 2 0.42 0.23 2 0.48 0.23 2 ...
```

**Normalization:**
- All coordinates normalized to [0, 1]
- x = pixel_x / image_width
- y = pixel_y / image_height

## Training After Labeling

```bash
# Fine-tune YOLOv8 with enhanced dataset
python train_yolo_pose.py \
    --data data/yolo_mouse_pose_enhanced/data.yaml \
    --epochs 100 \
    --batch 8 \
    --imgsz 256 \
    --weights yolov8n-pose.pt \
    --name mammal_mouse_finetuned
```

**Expected improvements:**
- mAP: 0 → 0.6-0.8 (with quality labels)
- Paw detection: 0% → 70-80%
- Overall confidence: 0.5 → 0.85+

## Next Steps After Training

1. Test on validation set
2. Compare with geometric baseline
3. Integrate into fit_monocular.py
4. Run full pipeline on 100+ images

## Time Investment Summary

| Task | Time | Cumulative |
|------|------|------------|
| Image selection | 10 min | 10 min |
| Labeling (20 imgs) | 2.5 hrs | 2h 40m |
| Quality check | 15 min | 2h 55m |
| Training | 30 min | 3h 25m |
| Validation | 15 min | 3h 40m |
| **Total** | **~3.5-4 hours** | |

**Return on investment:**
- 4 hours work → 10-20× quality improvement
- Paw detection: impossible → reliable
- Foundation for future improvements

## References

- YOLOv8 Pose: https://docs.ultralytics.com/tasks/pose/
- CVAT: https://github.com/opencv/cvat
- Our existing converter: `preprocessing_utils/dannce_to_yolo.py`
