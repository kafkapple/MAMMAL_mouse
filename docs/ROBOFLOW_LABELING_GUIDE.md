# Roboflow Labeling Guide for MAMMAL Mouse

## Why Roboflow?

- ✅ Web-based (no installation)
- ✅ Free tier sufficient for 20 images
- ✅ Direct YOLO export
- ✅ Fast and intuitive UI
- ✅ Automatic format conversion

## Setup (5 minutes)

### Step 1: Create Account
1. Go to https://roboflow.com/
2. Sign up with email or GitHub
3. Create new workspace (free tier)

### Step 2: Create Project
1. Click "Create New Project"
2. Project Type: **Object Detection** → then select **Keypoint Detection**
3. Project Name: `MAMMAL_Mouse_Keypoints`
4. License: Private (or your choice)

### Step 3: Define Keypoints
Add 22 keypoints in this exact order:

```
0:  nose
1:  left_ear
2:  right_ear
3:  left_eye
4:  right_eye
5:  head_center
6:  spine_1
7:  spine_2
8:  spine_3
9:  spine_4
10: spine_5
11: spine_6
12: spine_7
13: spine_8
14: left_front_paw
15: right_front_paw
16: left_rear_paw
17: right_rear_paw
18: tail_base
19: tail_mid
20: tail_tip
21: centroid
```

**Important**: Order matters for YOLO format!

## Upload Images (2 minutes)

1. Click "Upload" → "Upload Images"
2. Select all 20 images from `data/manual_labeling/images/`
3. Wait for upload to complete

## Labeling (2-3 hours)

### General Tips

**Speed:**
- Use keyboard shortcuts
- Label in batches of 5
- Take breaks every 30 minutes

**Accuracy:**
- Zoom in for precise placement
- Use mask as reference (if needed)
- Mark visibility correctly

### Keypoint Placement Guide

**Head (0-5):**
```
0: nose        - Tip of snout
1: left_ear    - Base of left ear (your left when facing mouse)
2: right_ear   - Base of right ear
3: left_eye    - Center of left eye (if visible)
4: right_eye   - Center of right eye (if visible)
5: head_center - Midpoint between ears
```

**Spine (6-13):**
```
Evenly distribute 8 points from neck to tail base:

6:  spine_1 - Just behind head (neck start)
7:  spine_2 - Upper back
8:  spine_3 - Mid-upper back
9:  spine_4 - Middle back
10: spine_5 - Mid-lower back
11: spine_6 - Lower back
12: spine_7 - Just before tail base
13: spine_8 - Tail base connection
```

**Tip**: Imagine dividing back into 8 equal segments

**Paws (14-17):**
```
14: left_front_paw  - Front left paw joint (elbow/wrist)
15: right_front_paw - Front right paw joint
16: left_rear_paw   - Rear left paw joint (knee/ankle)
17: right_rear_paw  - Rear right paw joint
```

**Note**: Mark as "not visible" if paws not clearly visible

**Tail (18-20):**
```
18: tail_base - Where tail starts (same as spine_8)
19: tail_mid  - Midpoint of tail
20: tail_tip  - End of tail
```

### Visibility Flags

In Roboflow:
- ✅ Visible and marked: Click and place
- ⚠️ Occluded but estimated: Place + mark as "occluded"
- ❌ Not visible: Skip or mark as "not visible"

### Quality Check Per Image

Before moving to next image, verify:
- [ ] All 22 keypoints placed (or marked not visible)
- [ ] Left/right not swapped
- [ ] Spine progression looks natural
- [ ] Tail follows curved path
- [ ] Paws at joint centers, not tips

## Export (2 minutes)

### Step 1: Generate Dataset Version
1. After labeling all images, click "Generate"
2. Preprocessing: None (keep original)
3. Augmentation: None (we'll do in training)
4. Generate

### Step 2: Export
1. Click "Export"
2. Format: **YOLO v8**
3. Download ZIP

### Step 3: Extract
```bash
cd ~/Downloads
unzip roboflow.zip -d ~/dev/MAMMAL_mouse/data/manual_labeling/roboflow_export
```

### Step 4: Copy to Project
```bash
# Copy labels to correct location
cp -r data/manual_labeling/roboflow_export/train/labels/* \
      data/manual_labeling/labels/

# Verify
ls -l data/manual_labeling/labels/
# Should see sample_000.txt, sample_001.txt, etc.
```

## Validate Labels (5 minutes)

```bash
# Visualize first 5 labeled images
python preprocessing_utils/visualize_yolo_labels.py \
    --images data/manual_labeling/images \
    --labels data/manual_labeling/labels \
    --output data/manual_labeling/viz \
    --max_images 5
```

Check visualization:
- Keypoints in correct positions?
- Left/right correctly labeled?
- Spine smooth and natural?

## Merge with Dataset

```bash
# Create enhanced dataset
python preprocessing_utils/merge_datasets.py \
    --manual data/manual_labeling \
    --geometric data/yolo_mouse_pose \
    --output data/yolo_mouse_pose_enhanced \
    --train_split 0.8
```

## Train YOLOv8

```bash
# Fine-tune with quality labels
python train_yolo_pose.py \
    --data data/yolo_mouse_pose_enhanced/data.yaml \
    --epochs 100 \
    --batch 8 \
    --imgsz 256 \
    --weights yolov8n-pose.pt \
    --name mammal_mouse_finetuned
```

## Expected Results

**Before (Geometric labels):**
- mAP: ~0
- Paw detection: 0%
- Confidence: 0.4-0.6

**After (Manual labels):**
- mAP: 0.6-0.8
- Paw detection: 70-80%
- Confidence: 0.85+

## Troubleshooting

**Roboflow upload fails:**
- Try smaller batches (5 images at a time)
- Check image format (PNG supported)
- Check file size (<10MB per image)

**Keypoint order wrong:**
- Re-define in project settings
- Order MUST match MAMMAL definition
- Export and check .txt files

**Export format wrong:**
- Make sure to select YOLO v8, not v5/v7
- Each line should have: class_id + bbox + 22×3 values

## Time Estimate

| Task | Time |
|------|------|
| Setup account & project | 5 min |
| Upload images | 2 min |
| Label 20 images (5 min each) | 1h 40m |
| Breaks | 20 min |
| Quality check | 15 min |
| Export & setup | 5 min |
| **Total** | **~2.5 hours** |

## Alternative: Label Studio

If Roboflow doesn't work, use Label Studio:

```bash
pip install label-studio
label-studio start

# Access at http://localhost:8080
# Import images and define same 22 keypoints
# Export as YOLO format
```

## References

- Roboflow: https://roboflow.com/
- YOLO Pose: https://docs.ultralytics.com/tasks/pose/
- Our keypoint definition: `mouse_model/keypoint22_mapper.json`
