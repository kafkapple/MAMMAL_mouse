# Mouse Mesh Fitting Guide

Complete guide for 3D mesh fitting on mouse video data with multiple dataset configurations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Types](#dataset-types)
3. [Configuration System](#configuration-system)
4. [Usage Examples](#usage-examples)
5. [Output Structure](#output-structure)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Default Dataset (Markerless Mouse)

Fit mesh to the reference multi-view dataset:

```bash
# Using default configuration
python fitter_articulation.py dataset=default_markerless

# With custom settings
python fitter_articulation.py \
  dataset=default_markerless \
  fitter.start_frame=0 \
  fitter.end_frame=50 \
  fitter.with_render=true
```

### Custom Mouse Video Dataset

Fit mesh to your own processed mouse video:

```bash
# Using cropped frames (with masks)
python fit_cropped_frames.py \
  data/100-KO-male-56-20200615_cropped \
  --output-dir results/cropped_fitting \
  --max-frames 10
```

---

## Dataset Types

### 1. Default Markerless Dataset (Multi-View)

**Location:** `/home/joon/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/`

**Structure:**
```
markerless_mouse_1_nerf/
├── keypoints2d_undist/
│   ├── result_view_0.pkl  # 2D keypoints for view 0
│   └── result_view_1.pkl  # 2D keypoints for view 1
├── simpleclick_undist/    # Segmentation masks
├── videos_undist/         # Video frames
├── new_cam.pkl           # Camera calibration (6 cameras)
└── add_labels_3d_8keypoints.pkl  # 3D keypoint annotations
```

**Features:**
- 6 synchronized camera views
- Multi-view camera calibration
- 2D keypoint annotations per view
- 3D keypoint annotations
- Segmentation masks

**Use Case:** Reference dataset with complete annotations for full 3D reconstruction

**Configuration:** `dataset=default_markerless`

---

### 2. Cropped Frames Dataset (Single-View with Masks)

**Location:** `/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_cropped/`

**Structure:**
```
100-KO-male-56-20200615_cropped/
├── frame_000000_cropped.png      # Cropped mouse image
├── frame_000000_mask.png          # Binary segmentation mask
├── frame_000000_crop_info.json   # Crop metadata
├── frame_000001_cropped.png
├── frame_000001_mask.png
├── frame_000001_crop_info.json
└── processing_summary.json       # Processing summary
```

**Crop Info JSON Format:**
```json
{
  "original_shape": [480, 640],
  "bbox": [365, 251, 217, 196],
  "crop_coords": [365, 251, 582, 447],
  "cropped_shape": [196, 217],
  "mask_area": 2929,
  "frame_idx": 6,
  "original_frame": "data/..../frame_000006.png",
  "cropped_frame": "data/..../frame_000006_cropped.png",
  "cropped_mask": "data/..../frame_000006_mask.png"
}
```

**Features:**
- Tightly cropped around mouse
- Binary segmentation masks included
- Variable image sizes (based on mouse size)
- Crop metadata for reconstruction
- Optional keypoint annotations

**Use Case:** Single-view silhouette-based fitting with accurate masks

**Script:** `fit_cropped_frames.py`

**Configuration:** `dataset=cropped`

---

### 3. Upsampled Frames Dataset (Single-View, No Masks)

**Location:** `/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_upsampled/`

**Structure:**
```
100-KO-male-56-20200615_upsampled/
├── frame_000000_upsampled.png
├── frame_000001_upsampled.png
├── frame_000002_upsampled.png
└── ...
```

**Features:**
- High-resolution upsampled frames
- Variable image sizes
- No masks or annotations
- Raw preprocessed frames

**Use Case:** Requires preprocessing (SAM/segmentation) before fitting

**Configuration:** `dataset=upsampled`

**⚠️ Note:** This dataset requires mask generation before mesh fitting. See [Preprocessing Guide](#preprocessing-guide).

---

### 4. Custom Dataset

**Location:** User-defined

**Structure:** Flexible, depends on your data format

**Configuration:** `dataset=custom` (template to modify)

---

## Configuration System

The project uses Hydra for hierarchical configuration management.

### Configuration Files

```
conf/
├── config.yaml              # Main config
├── dataset/
│   ├── default_markerless.yaml  # Reference multi-view dataset
│   ├── cropped.yaml             # Cropped frames with masks
│   ├── upsampled.yaml           # Upsampled frames
│   ├── shank3.yaml              # Shank3 dataset
│   └── custom.yaml              # Template for custom data
├── preprocess/
│   └── opencv.yaml
└── optim/
    ├── fast.yaml
    └── accurate.yaml
```

### Configuration Parameters

#### Data Settings

```yaml
data:
  data_dir: '/path/to/dataset'     # Dataset root directory
  views_to_use: [0]                 # Camera views (0-indexed)
```

#### Fitter Settings

```yaml
fitter:
  start_frame: 0                    # First frame to process
  end_frame: 10                     # Last frame to process
  interval: 1                       # Frame interval (1 = every frame)
  resume: false                     # Resume from checkpoint
  with_render: false                # Enable visualization rendering
  keypoint_num: 22                  # Number of keypoints
  render_cameras: [0]               # Cameras to render from
```

#### Optimization Settings

```yaml
optim:
  solve_step0_iters: 10            # Coarse alignment iterations
  solve_step1_iters: 100           # Fine fitting iterations
  solve_step2_iters: 30            # Refinement iterations
```

### Overriding Configuration

You can override any configuration parameter from the command line:

```bash
# Override dataset path
python fitter_articulation.py \
  data.data_dir=/path/to/my/dataset

# Override frame range
python fitter_articulation.py \
  fitter.start_frame=0 \
  fitter.end_frame=100 \
  fitter.interval=5

# Override multiple settings
python fitter_articulation.py \
  dataset=cropped \
  data.data_dir=/custom/path \
  fitter.with_render=true \
  optim=accurate
```

---

## Usage Examples

### Example 1: Default Dataset (Full Pipeline)

Process the reference markerless dataset with full rendering:

```bash
python fitter_articulation.py \
  dataset=default_markerless \
  fitter.start_frame=0 \
  fitter.end_frame=50 \
  fitter.interval=1 \
  fitter.with_render=true \
  result_folder=results/markerless_full/
```

**Expected Output:**
- Multi-view fitted meshes
- Rendered visualizations from 6 cameras
- Optimized parameters (thetas, bone_lengths, etc.)
- Loss curves and metrics

---

### Example 2: Cropped Frames with Masks (Silhouette Fitting)

Fit mesh to cropped frames using silhouette-based optimization:

```bash
# Process all available frames
python fit_cropped_frames.py \
  /home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_cropped \
  --output-dir results/cropped_fitting_full

# Process first 5 frames only (for testing)
python fit_cropped_frames.py \
  data/100-KO-male-56-20200615_cropped \
  --output-dir results/cropped_fitting_test \
  --max-frames 5
```

**Expected Output:**
```
results/cropped_fitting_test/
├── fitting_summary.json
├── frame_000000/
│   ├── params.json          # Optimized parameters
│   └── comparison.png       # Target vs rendered silhouette
├── frame_000001/
│   ├── params.json
│   └── comparison.png
└── ...
```

---

### Example 3: Custom Dataset Path

Process custom dataset with modified paths:

```bash
python fitter_articulation.py \
  dataset=custom \
  data.data_dir=/home/joon/dev/MAMMAL_mouse/data/my_mouse_video \
  fitter.start_frame=0 \
  fitter.end_frame=200 \
  fitter.interval=10 \
  fitter.with_render=false
```

---

### Example 4: Quick Test Run

Run a quick test on a few frames:

```bash
python fitter_articulation.py \
  dataset=default_markerless \
  fitter.start_frame=0 \
  fitter.end_frame=2 \
  fitter.interval=1 \
  fitter.with_render=false
```

---

### Example 5: Multiple Dataset Configurations

Create custom configuration for your specific dataset:

**conf/dataset/my_mouse.yaml:**
```yaml
# @package _global_

data:
  data_dir: /home/joon/dev/MAMMAL_mouse/data/my_mouse_experiment/
  views_to_use: [0]

fitter:
  start_frame: 0
  end_frame: 500
  interval: 5
  render_cameras: [0]
  with_render: true
```

**Run with custom config:**
```bash
python fitter_articulation.py dataset=my_mouse
```

---

## Output Structure

### Hydra Output Directory

By default, Hydra creates output directories with timestamps:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/
        │   ├── config.yaml       # Used configuration
        │   ├── overrides.yaml    # Command-line overrides
        │   └── hydra.yaml
        ├── fitter_articulation.log
        └── results/              # Mesh fitting results
```

### Mesh Fitting Results

```
results/
├── frame_0000/
│   ├── mesh.obj              # Fitted 3D mesh
│   ├── params.json           # Optimized parameters
│   ├── render_view_0.png     # Rendered view 0 (if enabled)
│   ├── render_view_1.png     # Rendered view 1 (if enabled)
│   └── metrics.json          # Fitting metrics (loss, IoU, etc.)
├── frame_0001/
│   └── ...
└── summary.json              # Overall fitting summary
```

### Parameter JSON Format

```json
{
  "thetas": [...],              // Joint angles (1, 140, 3)
  "bone_lengths": [...],        // Bone lengths (1, 28)
  "rotation": [...],            // Global rotation (1, 3)
  "translation": [...],         // Global translation (1, 3)
  "scale": [...],               // Global scale (1, 1)
  "chest_deformer": [...],      // Chest deformation (1, 1)
  "loss": 0.0234                // Final loss value
}
```

---

## Preprocessing Guide

### For Upsampled Frames (No Masks)

If you have upsampled frames without masks, you need to generate masks first:

#### Option 1: Using SAM (Segment Anything Model)

```bash
# Run SAM-based annotation GUI
python run_sam_gui.py

# Process frames with SAM
python process_video_with_sam.py \
  --input data/100-KO-male-56-20200615_upsampled \
  --output data/100-KO-male-56-20200615_sam_processed
```

#### Option 2: Using Manual Annotation

```bash
# Launch manual keypoint annotator
python keypoint_annotator_v2.py \
  --frames-dir data/100-KO-male-56-20200615_upsampled \
  --output-dir data/100-KO-male-56-20200615_annotated
```

#### Option 3: Process Cropped Frames

```bash
# Process annotated frames to generate crops
python process_annotated_frames.py \
  --annotations-dir data/100-KO-male-56-20200615_frames/annotations \
  --output-dir data/100-KO-male-56-20200615_cropped \
  --padding 50
```

---

## Workflow Recommendations

### Workflow 1: Full Pipeline (Raw Video → Fitted Mesh)

```bash
# 1. Extract frames from video
python extract_video_frames.py \
  --video data/raw/my_mouse.avi \
  --output data/my_mouse_frames \
  --fps 30

# 2. Annotate with SAM
python run_sam_gui.py
# (manually annotate frames)

# 3. Process annotated frames
python process_annotated_frames.py \
  --annotations-dir data/my_mouse_frames/annotations \
  --output-dir data/my_mouse_cropped

# 4. Fit mesh
python fit_cropped_frames.py \
  data/my_mouse_cropped \
  --output-dir results/my_mouse_fitting
```

---

### Workflow 2: Quick Testing (Existing Cropped Data)

```bash
# Test with few frames
python fit_cropped_frames.py \
  data/100-KO-male-56-20200615_cropped \
  --output-dir results/quick_test \
  --max-frames 3

# Check results
ls results/quick_test/
```

---

### Workflow 3: Using Hydra Configuration

```bash
# Create custom dataset config
# conf/dataset/my_experiment.yaml

# Run with custom config
python fitter_articulation.py \
  dataset=my_experiment \
  fitter.with_render=true
```

---

## Troubleshooting

### Issue 1: No Masks Found

**Error:**
```
Mask not found: /path/to/frame_000000_mask.png
```

**Solution:**
- Ensure masks are in the same directory as cropped frames
- Mask filename should match: `frame_{idx:06d}_mask.png`
- Generate masks using SAM or other segmentation tools

---

### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size
- Process fewer frames at once
- Use `--max-frames` to limit processing
- Reduce image resolution

---

### Issue 3: Image Size Mismatch

**Error:**
```
RuntimeError: Expected image size (H, W), got (H', W')
```

**Solution:**
- Ensure all frames in the dataset have consistent dimensions
- Or use `fit_cropped_frames.py` which handles variable sizes

---

### Issue 4: Camera Calibration Not Found

**Error:**
```
FileNotFoundError: new_cam.pkl not found
```

**Solution:**
- This file is required for multi-view datasets
- For single-view datasets, use `dataset=cropped` or `dataset=upsampled`
- Ensure you're using the correct dataset configuration

---

### Issue 5: Configuration Override Not Working

**Problem:** Command-line override not taking effect

**Solution:**
```bash
# Correct syntax (note the dot notation)
python fitter_articulation.py data.data_dir=/path/to/data

# Incorrect syntax
python fitter_articulation.py data_dir=/path/to/data
```

---

## Advanced Configuration

### Custom Loss Weights

Modify loss term weights in `fitter_articulation.py`:

```python
self.term_weights = {
    "theta": 3,              # Joint angle regularization
    "3d": 2.5,               # 3D keypoint loss
    "2d": 0.2,               # 2D projection loss
    "bone": 0.5,             # Bone length regularization
    "scale": 0.5,            # Scale regularization
    "mask": 0,               # Mask/silhouette loss
    "chest_deformer": 0.1,   # Chest deformation regularization
    "stretch": 1,            # Stretch regularization
    "temp": 0.25,            # Temporal smoothness
    "temp_d": 0.2            # Temporal derivative smoothness
}
```

---

### Visualization Options

Enable detailed rendering:

```bash
python fitter_articulation.py \
  dataset=default_markerless \
  fitter.with_render=true \
  fitter.render_cameras=[0,1,2,3,4,5]
```

---

## Summary

### Dataset Quick Reference

| Dataset Type | Location | Has Masks | Has Keypoints | Use Case |
|-------------|----------|-----------|---------------|----------|
| Default Markerless | `data/examples/markerless_mouse_1_nerf/` | ✅ | ✅ | Reference multi-view |
| Cropped | `data/100-KO-male-56-20200615_cropped/` | ✅ | Optional | Single-view fitting |
| Upsampled | `data/100-KO-male-56-20200615_upsampled/` | ❌ | ❌ | Requires preprocessing |
| Custom | User-defined | Varies | Varies | Flexible |

### Command Quick Reference

```bash
# Default dataset
python fitter_articulation.py dataset=default_markerless

# Cropped frames
python fit_cropped_frames.py data/100-KO-male-56-20200615_cropped

# Custom configuration
python fitter_articulation.py dataset=custom data.data_dir=/path/to/data

# Override parameters
python fitter_articulation.py \
  dataset=cropped \
  fitter.start_frame=0 \
  fitter.end_frame=50 \
  fitter.with_render=true
```

---

## Next Steps

1. **Test with default dataset**: Verify setup with reference data
2. **Process your video**: Extract frames and generate masks
3. **Run mesh fitting**: Use appropriate configuration for your data
4. **Visualize results**: Enable rendering to inspect fitted meshes
5. **Iterate and refine**: Adjust loss weights and parameters as needed

For more details, see:
- `README.md`: Project overview
- `docs/reports/`: Experimental reports
- `conf/`: Configuration files
- Individual script help: `python script.py --help`
