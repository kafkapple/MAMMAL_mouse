# Quick Start: Manual Labeling → YOLOv8 Fine-tuning

Complete guide to improve keypoint detection from geometric baseline to high-quality ML model.

## 🎯 Goal

Improve keypoint detection accuracy by manually labeling 20 images and fine-tuning YOLOv8.

**Expected improvement:**
- mAP: 0 → 0.6-0.8
- Paw detection: 0% → 70-80%
- Overall confidence: 0.5 → 0.85+

## ⏱️ Time Required

**Total: ~3-4 hours**
- Setup: 10 minutes
- Labeling: 2-3 hours (20 images)
- Training: 30 minutes
- Validation: 15 minutes

## 📋 Prerequisites

✅ Already prepared:
- 20 sampled images in `data/manual_labeling/images/`
- Masks in `data/manual_labeling/masks/`
- YOLOv8 training pipeline ready
- Visualization tools ready

## 🚀 Step-by-Step Workflow

### Step 1: Choose Labeling Tool (2 minutes)

**Option A: Roboflow** ⭐ RECOMMENDED
- Web-based, no installation
- Free tier sufficient
- Direct YOLO export
- Guide: `docs/ROBOFLOW_LABELING_GUIDE.md`

**Option B: Label Studio**
```bash
pip install label-studio
label-studio start
# Access at http://localhost:8080
```

**Option C: CVAT**
```bash
docker run -p 8080:8080 cvat/server
```

### Step 2: Define 22 Keypoints

**CRITICAL: Use exact order!**

```
Head (0-5):
  0: nose, 1: left_ear, 2: right_ear, 3: left_eye,
  4: right_eye, 5: head_center

Spine (6-13):
  6-13: spine_1 to spine_8 (evenly distributed)

Paws (14-17):
  14: left_front_paw, 15: right_front_paw,
  16: left_rear_paw, 17: right_rear_paw

Tail (18-20):
  18: tail_base, 19: tail_mid, 20: tail_tip

Body (21):
  21: centroid
```

### Step 3: Label Images (2-3 hours)

**Labeling tips:**
- Zoom in for precision
- Use mask as reference
- Mark visibility correctly
- Take 5-min break every 30 minutes

**Quality checklist per image:**
- [ ] All 22 keypoints placed
- [ ] Left/right not swapped
- [ ] Spine smooth and evenly spaced
- [ ] Paws at joint centers
- [ ] Tail follows natural curve

### Step 4: Export Labels (2 minutes)

**Roboflow:**
1. Generate dataset version
2. Export as "YOLO v8"
3. Download ZIP

**Extract:**
```bash
cd ~/Downloads
unzip roboflow.zip -d ~/dev/MAMMAL_mouse/data/manual_labeling/roboflow_export

# Copy labels
cp -r data/manual_labeling/roboflow_export/train/labels/* \
      data/manual_labeling/labels/
```

### Step 5: Validate Labels (5 minutes)

```bash
# Visualize first 5 images
~/miniconda3/envs/mammal_stable/bin/python \
  preprocessing_utils/visualize_yolo_labels.py \
  --images data/manual_labeling/images \
  --labels data/manual_labeling/labels \
  --output data/manual_labeling/viz \
  --max_images 5

# Check output
ls data/manual_labeling/viz/
```

**Verify:**
- Keypoints in correct positions?
- Skeleton connections make sense?
- No obvious errors?

### Step 6: Merge Datasets (2 minutes)

```bash
# Combine manual (20) + geometric (50) = 70 total images
python preprocessing_utils/merge_datasets.py \
  --manual data/manual_labeling \
  --geometric data/yolo_mouse_pose \
  --output data/yolo_mouse_pose_enhanced \
  --train_split 0.8

# Result: 56 train + 14 val
```

### Step 7: Train YOLOv8 (30 minutes)

```bash
# Fine-tune on enhanced dataset
~/miniconda3/envs/mammal_stable/bin/python scripts/train_yolo_pose.py \
  --data data/yolo_mouse_pose_enhanced/data.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 256 \
  --weights yolov8n-pose.pt \
  --name mammal_mouse_finetuned
```

**Monitor training:**
```bash
# In another terminal
tail -f /tmp/yolo_train.log

# Or use TensorBoard
tensorboard --logdir runs/pose/mammal_mouse_finetuned
```

### Step 8: Evaluate Results (10 minutes)

```bash
# Test on validation set
~/miniconda3/envs/mammal_stable/bin/python -c "
from ultralytics import YOLO

model = YOLO('runs/pose/mammal_mouse_finetuned/weights/best.pt')
metrics = model.val(data='data/yolo_mouse_pose_enhanced/data.yaml')

print(f'mAP50: {metrics.box.map50:.3f}')
print(f'mAP50-95: {metrics.box.map:.3f}')
"
```

### Step 9: Compare with Baseline (5 minutes)

```bash
# Geometric baseline
~/miniconda3/envs/mammal_stable/bin/python fit_monocular.py \
  --input_dir data/manual_labeling/images \
  --output_dir results/geometric \
  --detector geometric \
  --max_images 5

# YOLO fine-tuned
~/miniconda3/envs/mammal_stable/bin/python fit_monocular.py \
  --input_dir data/manual_labeling/images \
  --output_dir results/yolo_finetuned \
  --detector yolo \
  --yolo_weights runs/pose/mammal_mouse_finetuned/weights/best.pt \
  --max_images 5

# Compare visualizations
ls results/geometric/
ls results/yolo_finetuned/
```

### Step 10: Integrate into Production (5 minutes)

```bash
# Copy best model to models directory
cp runs/pose/mammal_mouse_finetuned/weights/best.pt \
   models/yolo_mouse_pose_finetuned.pt

# Update fit_monocular.py default
# Change --detector default to 'yolo'
# Change --yolo_weights default to 'models/yolo_mouse_pose_finetuned.pt'
```

## 📊 Expected Results

### Before (Geometric)
```
Detected: 15/22 keypoints
Confidence: 0.40-0.60
Paw detection: 0%
mAP: ~0
```

### After (Fine-tuned YOLO)
```
Detected: 20-22/22 keypoints
Confidence: 0.80-0.95
Paw detection: 70-80%
mAP: 0.6-0.8
```

## 🔧 Troubleshooting

**Labeling tool won't start:**
- Roboflow: Check internet connection
- Label Studio: `pip install --upgrade label-studio`
- CVAT: `docker pull cvat/server:latest`

**Export format wrong:**
- Make sure to select "YOLO v8" not v5/v7
- Each .txt should have: 1 + 4 + 66 values (class + bbox + 22×3 keypoints)

**Training fails:**
- Check CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size: `--batch 4`
- Check data.yaml paths correct

**Low mAP after training:**
- Label quality issue - review annotations
- Need more epochs: `--epochs 200`
- Need more data: Label 10 more images

## 📁 Directory Structure After Completion

```
MAMMAL_mouse/
├── data/
│   ├── manual_labeling/
│   │   ├── images/           # 20 images
│   │   ├── masks/            # 20 masks
│   │   ├── labels/           # 20 YOLO labels (manually created)
│   │   └── viz/              # Visualizations
│   │
│   ├── yolo_mouse_pose/      # Original geometric (50)
│   │
│   └── yolo_mouse_pose_enhanced/  # Combined (70)
│       ├── train/            # 56 images + labels
│       ├── val/              # 14 images + labels
│       └── data.yaml
│
├── runs/pose/mammal_mouse_finetuned/
│   ├── weights/
│   │   ├── best.pt           # Best model
│   │   └── last.pt           # Last checkpoint
│   └── results.png           # Training curves
│
└── models/
    └── yolo_mouse_pose_finetuned.pt  # Production model
```

## ✅ Success Checklist

- [ ] 20 images labeled with 22 keypoints each
- [ ] Labels validated visually
- [ ] Training completed (100 epochs)
- [ ] mAP > 0.6 achieved
- [ ] Paw detection working
- [ ] Integrated into fit_monocular.py
- [ ] Tested on new images

## 🚀 Next Steps After Success

1. **Label 30 more images** → Even better accuracy
2. **Fine-tune hyperparameters** → Optimize for your data
3. **Add augmentation** → Rotation, flip, scale
4. **Export to ONNX** → Faster inference
5. **Create ensemble** → Geometric + YOLO for robustness

## 📚 References

- Roboflow Guide: `docs/ROBOFLOW_LABELING_GUIDE.md`
- Manual Labeling Guide: `docs/MANUAL_LABELING_GUIDE.md`
- YOLO Pose Docs: https://docs.ultralytics.com/tasks/pose/
- Keypoint Definition: `mouse_model/keypoint22_mapper.json`

---

**Ready to start?** → Follow Step 1 and begin labeling! 🎯
