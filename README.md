# MAMMAL_mouse

Three-dimensional surface motion capture of mice using the MAMMAL framework. This project enables markerless 3D pose estimation and mesh reconstruction for behavioral analysis by fitting an articulated 3D mouse model to video data.

![mouse_model](assets/figs/mouse_1.png)

## ✨ Features

- **Multi-view 3D fitting**: Fit 3D mouse model to synchronized multi-camera videos
- **Single-view (monocular) fitting**: Process single videos with ML-based keypoint detection
- **ML keypoint detection**: YOLOv8-Pose and SuperAnimal support for anatomically accurate keypoints
- **🆕 Flexible keypoint annotation**: Manual annotation tool + automatic format conversion (1-22 keypoints)
- **Confidence-based filtering**: Missing keypoints automatically ignored (no need for all 22!)
- **Hydra configuration**: Flexible experiment management with dataset-specific configs
- **Modular pipeline**: Separate preprocessing and fitting stages for easy customization

---

## 🚀 Quick Start

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/MAMMAL_mouse.git
cd MAMMAL_mouse

# Create conda environment and install dependencies (one-time setup)
bash scripts/setup/setup.sh
```

**What this installs**:
- Python 3.10 environment named `mammal_stable`
- PyTorch 2.0.0 + CUDA 11.8
- PyTorch3D 0.7.5
- All required dependencies (opencv, hydra, ultralytics, etc.)

**Requirements**:
- Anaconda/Miniconda
- NVIDIA GPU with CUDA 11.8
- ~10GB disk space for dependencies

### Step 2: Download Data and Models

#### Option A: Example Dataset (Recommended for First-Time Users)

Download the example multi-view dataset:

```bash
# Download from Google Drive
# https://drive.google.com/file/d/1NbaIFOvpvQ_WLOabUtMrVHS7vVBq-8zD/view?usp=sharing

# Extract to the correct location
unzip markerless_mouse_1_nerf.zip
mv markerless_mouse_1_nerf/ data/examples/
```

**Dataset structure**:
```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/           # 6 camera views
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
├── simpleclick_undist/      # Binary masks
│   ├── 0.mp4
│   └── ...
├── keypoints2d_undist/      # 2D keypoints
│   ├── result_view_0.pkl
│   └── ...
├── new_cam.pkl              # Camera parameters
└── new_params.pkl           # Model parameters
```

#### Option B: Your Own Video

If you have your own single-view video:

```bash
# 1. Place your video
mkdir -p data/raw/my_experiment/
cp /path/to/your/video.mp4 data/raw/my_experiment/

# 2. Extract frames (optional, for monocular fitting)
mkdir -p data/raw/my_experiment/frames/
ffmpeg -i data/raw/my_experiment/video.mp4 \
  -vf "fps=30" data/raw/my_experiment/frames/%06d.png
```

#### Optional: Download Pretrained Models

For ML-based keypoint detection:

```bash
# YOLOv8-Pose pretrained model (auto-downloaded on first use)
# Will be saved to: models/pretrained/yolov8n-pose.pt

# SuperAnimal-TopViewMouse model (optional, 245MB)
python scripts/setup/download_superanimal.py
# Saved to: models/pretrained/superanimal_topviewmouse/
```

### Step 3: Run Your First Experiment

#### Scenario 1: Multi-View Fitting (Example Dataset) ⭐ Recommended for Testing

Process the example multi-view dataset:

```bash
# Activate environment
conda activate mammal_stable

# Run fitting on first 10 frames (using shell script)
./run_mesh_fitting_default.sh 0 10

# OR run directly with Python
python fitter_articulation.py \
  dataset=default_markerless \
  fitter.start_frame=0 \
  fitter.end_frame=10 \
  fitter.with_render=true
```

**What happens**:
1. Loads 6-camera preprocessed data from `data/examples/markerless_mouse_1_nerf/`
2. Fits 3D mouse model to frames 0-10
3. Saves results to `results/fitting/{dataset}_{timestamp}/`

**Expected output**:
```
results/fitting/markerless_mouse_1_nerf_20251125_143000/
├── obj/                     # 3D mesh files (.obj)
│   ├── mesh_000000.obj
│   ├── mesh_000002.obj
│   └── ...
├── params/                  # Fitting parameters (.pkl)
│   ├── param0.pkl
│   ├── param0_sil.pkl
│   └── ...
└── render/                  # Visualization overlays (.png)
    ├── fitting_0.png
    ├── fitting_0_sil.png
    └── debug/               # Optimization debug images
```

**Processing time**: ~5-10 minutes (RTX 3090)

#### Scenario 2: Monocular Fitting (Single Video) 🆕

Process a single-view video with ML keypoint detection:

```bash
conda activate mammal_stable

# Using geometric keypoint detection (baseline)
python fit_monocular.py \
  --input_dir data/raw/my_experiment/frames/ \
  --output_dir results/monocular/my_experiment \
  --detector geometric \
  --max_images 10

# OR using YOLOv8-Pose (better quality, requires GPU)
python fit_monocular.py \
  --input_dir data/raw/my_experiment/frames/ \
  --output_dir results/monocular/my_experiment \
  --detector yolo \
  --max_images 10
```

**What happens**:
1. Detects 22 keypoints per frame using chosen detector
2. Estimates camera parameters from first frame
3. Fits 3D mouse model frame-by-frame
4. Saves meshes, parameters, and visualizations

**Expected output**:
```
results/monocular/my_experiment/
├── obj/                     # 3D mesh files
│   ├── frame_000001.obj
│   └── ...
├── params/                  # Fitting parameters
│   ├── frame_000001.pkl
│   └── ...
├── keypoints_2d/            # Detected 2D keypoints
│   ├── frame_000001.pkl
│   └── ...
├── camera_params.pkl        # Estimated camera
└── visualizations/          # Overlays (if enabled)
```

**Processing time**: ~30 seconds/frame (geometric), ~1 minute/frame (YOLO)

#### Scenario 3: Traditional Preprocessing + Fitting

Full pipeline for single-view video:

```bash
conda activate mammal_stable

# 1. Preprocess video (extract frames, masks, keypoints)
python scripts/preprocess.py \
  dataset=custom \
  mode=single_view_preprocess \
  preprocess.input_video_path="data/raw/my_experiment/video.mp4" \
  preprocess.output_data_dir="data/preprocessed/my_experiment/"

# 2. Fit 3D model to preprocessed data
python fitter_articulation.py \
  dataset=custom \
  data.data_dir="data/preprocessed/my_experiment/" \
  fitter.end_frame=100 \
  fitter.with_render=false
```

**Preprocessing outputs**:
```
data/preprocessed/my_experiment/
├── videos_undist/
│   └── 0.mp4                # Original video
├── simpleclick_undist/
│   └── 0.mp4                # Binary mask video
├── keypoints2d_undist/
│   └── result_view_0.pkl    # 22 keypoints per frame
└── new_cam.pkl              # Camera parameters
```

---

## 📖 Usage Scenarios

### 1️⃣ Quick Test with Example Data (5 minutes)

**Goal**: Verify installation and see multi-view fitting results

```bash
conda activate mammal_stable

# Using shell script (recommended)
./run_mesh_fitting_default.sh 0 5

# OR using Python directly
python fitter_articulation.py \
  dataset=default_markerless \
  optim=fast \
  fitter.end_frame=5
```

**Results**: `results/fitting/{dataset}_{timestamp}/obj/` contains 3D meshes

### 2️⃣ Process Your Single Video (30 minutes)

**Goal**: Get 3D pose from your own video

```bash
conda activate mammal_stable

# Extract frames from video
mkdir -p data/raw/my_video/frames/
ffmpeg -i your_video.mp4 data/raw/my_video/frames/%06d.png

# Using shell script (recommended)
./run_mesh_fitting_monocular.sh data/raw/my_video/frames/ results/monocular/my_video yolo

# OR using Python directly
python fit_monocular.py \
  --input_dir data/raw/my_video/frames/ \
  --output_dir results/monocular/my_video \
  --detector yolo \
  --max_images 50
```

**Results**: `results/monocular/my_video/obj/` contains 3D meshes

### 3️⃣ Train Custom ML Detector (1 day)

**Goal**: Improve keypoint detection for your specific setup

**Step 1: Sample images** (5 min)
```bash
conda activate mammal_stable

python scripts/setup/sample_images_for_labeling.py \
  --input_dir data/raw/my_video/frames/ \
  --output_dir data/training/manual_labeling/images/ \
  --num_samples 20
```

**Step 2: Label on Roboflow** (2-3 hours)
1. Create account at https://roboflow.com
2. Create new "Keypoint Detection" project
3. Define 22 keypoints (see `docs/guides/ROBOFLOW_LABELING_GUIDE.md`)
4. Upload 20 images and label all keypoints
5. Export as "YOLOv8 Pose" format

**Step 3: Train YOLOv8** (30 min)
```bash
conda activate mammal_stable

# Merge manual labels with geometric labels
python preprocessing_utils/merge_datasets.py \
  --manual data/training/manual_labeling/ \
  --geometric data/training/yolo_mouse_pose/ \
  --output data/training/yolo_enhanced/

# Train YOLOv8
python scripts/train_yolo_pose.py \
  --data data/training/yolo_enhanced/data.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 256 \
  --name my_custom_detector
```

**Step 4: Use trained model**
```bash
python fit_monocular.py \
  --detector yolo \
  --yolo_weights models/trained/yolo/my_custom_detector/weights/best.pt
```

**Expected improvements**:
- Confidence: 0.5 → 0.85+ (2×)
- Loss: ~300K → 15-30K (10-20×)
- Paw detection: 0% → 70-80%

### 4️⃣ Batch Process Multiple Videos (customizable)

Process a directory of videos:

```bash
conda activate mammal_stable

# Create batch processing script
cat > batch_process.sh << 'EOF'
#!/bin/bash
for video in data/raw/batch/*.mp4; do
  name=$(basename "$video" .mp4)
  echo "Processing $name..."

  # Extract frames
  mkdir -p "data/raw/batch/${name}_frames/"
  ffmpeg -i "$video" "data/raw/batch/${name}_frames/%06d.png"

  # Run monocular fitting
  python fit_monocular.py \
    --input_dir "data/raw/batch/${name}_frames/" \
    --output_dir "results/monocular/${name}/" \
    --detector yolo \
    --max_images 100
done
EOF

chmod +x batch_process.sh
./batch_process.sh
```

---

## 📊 Understanding the Output

### Multi-View Fitting Output

After running `fitter_articulation.py`, outputs are in `results/fitting/{dataset}_{timestamp}/`:

```
results/fitting/markerless_mouse_1_nerf_20251125_143000/
├── obj/                           # 3D mesh files (can open in Blender/MeshLab)
│   ├── mesh_000000.obj            # Mesh for frame 0
│   ├── mesh_000002.obj
│   └── ...
│
├── params/                        # Fitting parameters (Python pickle)
│   ├── param0.pkl                 # Contains: body_pose, global_orient, betas, etc.
│   ├── param0_sil.pkl             # After silhouette refinement
│   └── ...
│
├── render/                        # Visualization overlays (if with_render=true)
│   ├── fitting_0.png              # Fitted model overlaid on all views
│   ├── fitting_0_sil.png          # After silhouette refinement
│   └── debug/                     # Optimization debug images
│
└── .hydra/                        # Hydra config snapshots
    └── config.yaml                # Exact config used for this run
```

**How to visualize**:
```bash
# View 3D mesh in Blender
blender results/fitting/*/obj/mesh_000000.obj

# View 3D mesh in MeshLab
meshlab results/fitting/*/obj/mesh_000000.obj

# View overlays
eog results/fitting/*/render/fitting_0.png
```

### Monocular Fitting Output

After running `fit_monocular.py`, outputs are in specified `--output_dir`:

```
results/monocular/my_experiment/
├── obj/                           # 3D mesh files
│   ├── frame_000001.obj
│   ├── frame_000002.obj
│   └── ...
│
├── params/                        # Fitting parameters
│   ├── frame_000001.pkl
│   └── ...
│
├── keypoints_2d/                  # Detected 2D keypoints
│   ├── frame_000001.pkl           # 22 keypoints [x, y, conf]
│   └── ...
│
├── camera_params.pkl              # Estimated camera intrinsics
│
└── visualizations/                # Overlays (if --visualize)
    ├── frame_000001.png           # Keypoints overlaid on image
    └── ...
```

**How to inspect keypoints**:
```python
import pickle
import numpy as np

# Load keypoints for frame 1
with open('results/monocular/my_experiment/keypoints_2d/frame_000001.pkl', 'rb') as f:
    kpts = pickle.load(f)

print(kpts.shape)  # (22, 3) -> [x, y, confidence]
print(f"Nose: x={kpts[0,0]:.1f}, y={kpts[0,1]:.1f}, conf={kpts[0,2]:.2f}")
```

**How to inspect fitted parameters**:
```python
import pickle

# Load fitted parameters
with open('results/monocular/my_experiment/params/frame_000001.pkl', 'rb') as f:
    params = pickle.load(f)

print(params.keys())
# dict_keys(['body_pose', 'global_orient', 'betas', 'transl'])

print(f"Body pose shape: {params['body_pose'].shape}")  # Limb rotations
print(f"Global orient shape: {params['global_orient'].shape}")  # Root orientation
print(f"Translation: {params['transl']}")  # 3D position
```

### Training Output

After running `scripts/train_yolo_pose.py`, training results are in `models/trained/yolo/`:

```
models/trained/yolo/my_custom_detector/
├── weights/
│   ├── best.pt                    # Best model checkpoint (use this!)
│   └── last.pt                    # Last epoch checkpoint
│
├── results.png                    # Training curves (loss, mAP, etc.)
├── confusion_matrix.png           # Confusion matrix
├── PR_curve.png                   # Precision-Recall curve
├── results.csv                    # Metrics per epoch
└── args.yaml                      # Training arguments
```

**How to evaluate**:
```bash
# View training curves
eog models/trained/yolo/my_custom_detector/results.png

# Check final metrics
tail -1 models/trained/yolo/my_custom_detector/results.csv
```

---

## ⚙️ Configuration Guide

### Hydra Configuration System

This project uses [Hydra](https://hydra.cc/) for flexible configuration. Config files are in `conf/`:

```
conf/
├── config.yaml              # Main config (don't edit directly)
├── dataset/                 # Dataset-specific configs
│   ├── markerless.yaml      # Multi-view (6 cameras)
│   ├── shank3.yaml          # Single-view
│   └── custom.yaml          # Template for your data
├── preprocess/              # Preprocessing configs
│   ├── opencv.yaml          # Current: geometric keypoints
│   └── sam.yaml             # Future: SAM-based masking
└── optim/                   # Optimization configs
    ├── fast.yaml            # Quick test (fewer iterations)
    └── accurate.yaml        # High quality (more iterations)
```

### Key Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `dataset` | Which dataset config to use | `shank3` | `dataset=markerless` |
| `optim` | Optimization settings | `fast` | `optim=accurate` |
| `mode` | Processing mode | `multi_view` | `mode=single_view_preprocess` |
| `data.data_dir` | Input data path | varies | `data.data_dir="data/preprocessed/custom/"` |
| `fitter.start_frame` | First frame | `0` | `fitter.start_frame=10` |
| `fitter.end_frame` | Last frame | `2` | `fitter.end_frame=100` |
| `fitter.with_render` | Enable rendering | `false` | `fitter.with_render=true` |
| `optim.solve_step0_iters` | Step 0 iterations | `10` | `optim.solve_step0_iters=20` |
| `optim.solve_step1_iters` | Step 1 iterations | `100` | `optim.solve_step1_iters=200` |
| `optim.solve_step2_iters` | Step 2 iterations | `30` | `optim.solve_step2_iters=50` |

### 🆕 Manual Keypoint Annotation Workflow

For detailed mesh fitting with custom keypoint annotations:

**Quick workflow**:
```bash
# 1. Annotate keypoints (Gradio UI)
python keypoint_annotator_v2.py data/100-KO-male-56-20200615_cropped

# 2. Convert to MAMMAL format
python convert_keypoints_to_mammal.py \
  --input keypoints.json \
  --output data/.../keypoints2d_undist/result_view_0.pkl \
  --num-frames 20

# 3. Run mesh fitting
python fitter_articulation.py dataset=custom_cropped
```

**Key features**:
- ✅ **Flexible keypoint count**: 1-22 keypoints (recomm 5-7)
- ✅ **Auto-filtering**: Missing keypoints ignored automatically
- ✅ **Interactive UI**: Zoom, visibility control, progress tracking

📖 **Full guide**: [`KEYPOINT_QUICK_START.md`](KEYPOINT_QUICK_START.md) | [`docs/KEYPOINT_WORKFLOW.md`](docs/KEYPOINT_WORKFLOW.md)

---

### Usage Examples

```bash
# Use markerless dataset with accurate optimization
python fitter_articulation.py \
  dataset=markerless \
  optim=accurate

# Process frames 50-100 with rendering
python fitter_articulation.py \
  dataset=markerless \
  fitter.start_frame=50 \
  fitter.end_frame=100 \
  fitter.with_render=true

# Quick test on first 5 frames
python fitter_articulation.py \
  dataset=markerless \
  optim=fast \
  fitter.end_frame=5

# Override data directory
python fitter_articulation.py \
  data.data_dir="data/preprocessed/custom/" \
  fitter.end_frame=10
```

### Creating Custom Dataset Config

1. Copy template:
```bash
cp conf/dataset/custom.yaml conf/dataset/my_dataset.yaml
```

2. Edit `conf/dataset/my_dataset.yaml`:
```yaml
# @package _global_

data:
  data_dir: "data/preprocessed/my_dataset/"
  num_views: 1  # Single camera

fitter:
  start_frame: 0
  end_frame: 100

preprocess:
  input_video_path: "data/raw/my_dataset/video.mp4"
  output_data_dir: "data/preprocessed/my_dataset/"
```

3. Use it:
```bash
python fitter_articulation.py dataset=my_dataset
```

---

## 🔬 Advanced Usage

### Custom Keypoint Order

The default keypoint order is defined in `mouse_22_defs.py`:

```python
# 22 anatomical keypoints
KEYPOINT_NAMES = [
    'nose', 'left_ear', 'right_ear', 'left_eye', 'right_eye',
    'head_center', 'spine_1', 'spine_2', 'spine_3', 'spine_4',
    'spine_5', 'spine_6', 'spine_7', 'spine_8',
    'left_paw_front', 'right_paw_front',
    'left_paw_rear', 'right_paw_rear',
    'tail_base', 'tail_mid', 'tail_tip', 'centroid'
]
```

To use different keypoints, modify `mouse_22_defs.py` and update detection accordingly.

### Batch Processing with GNU Parallel

Process multiple datasets in parallel:

```bash
# Install GNU parallel
sudo apt-get install parallel

# Create experiment list
cat > experiments.txt << EOF
markerless 0 10
markerless 10 20
markerless 20 30
EOF

# Run in parallel (4 jobs)
parallel -j 4 --colsep ' ' \
  python fitter_articulation.py \
  dataset={1} \
  fitter.start_frame={2} \
  fitter.end_frame={3} \
  :::: experiments.txt
```

### Exporting Results

Convert 3D meshes to different formats:

```bash
# Convert OBJ to PLY
conda activate mammal_stable
pip install trimesh

python << EOF
import trimesh
import glob

for obj_file in glob.glob('results/fitting/*/obj/*.obj'):
    mesh = trimesh.load(obj_file)
    ply_file = obj_file.replace('.obj', '.ply')
    mesh.export(ply_file)
    print(f"Converted {obj_file} -> {ply_file}")
EOF
```

### Visualization with PyVista

Interactive 3D visualization:

```bash
conda activate mammal_stable
pip install pyvista

python << EOF
import pyvista as pv
import glob

# Load all meshes
meshes = []
for obj_file in sorted(glob.glob('results/fitting/*/obj/mesh_*.obj')):
    meshes.append(pv.read(obj_file))

# Create animation
plotter = pv.Plotter()
for mesh in meshes:
    plotter.add_mesh(mesh, color='tan')
    plotter.show(auto_close=False)
    plotter.clear()
EOF
```

---

## 🔧 Troubleshooting

### Installation Issues

**Problem**: `bash scripts/setup/setup.sh` fails
```bash
# Solution 1: Check conda is installed
conda --version

# Solution 2: Update conda
conda update -n base -c defaults conda

# Solution 3: Manual installation
conda create -n mammal_stable python=3.10 -y
conda activate mammal_stable
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

**Problem**: `CUDA out of memory`
```bash
# Solution: Reduce batch size or process fewer frames
python fitter_articulation.py fitter.end_frame=5  # Instead of 10
python fit_monocular.py --max_images 5  # Instead of 10
```

**Problem**: `ModuleNotFoundError: No module named 'pytorch3d'`
```bash
# Solution: Reinstall pytorch3d
conda activate mammal_stable
pip uninstall pytorch3d -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Data Issues

**Problem**: `FileNotFoundError: new_cam.pkl not found`
```bash
# Solution: Run preprocessing first
python scripts/preprocess.py dataset=custom mode=single_view_preprocess

# Or use monocular fitting (no preprocessing needed)
python fit_monocular.py --input_dir frames/
```

**Problem**: Poor keypoint quality
```bash
# Solution 1: Use YOLO instead of geometric
python fit_monocular.py --detector yolo

# Solution 2: Train custom detector (see Usage Scenario 3)
# Solution 3: Manually inspect and fix
python preprocessing_utils/visualize_yolo_labels.py --images frames/ --labels labels/
```

**Problem**: Camera calibration fails
```bash
# Solution: Provide known camera parameters
# Edit conf/dataset/my_dataset.yaml:
camera:
  fx: 1000.0  # Focal length X
  fy: 1000.0  # Focal length Y
  cx: 640.0   # Principal point X
  cy: 360.0   # Principal point Y
```

### Fitting Issues

**Problem**: Model converges to wrong pose
```bash
# Solution 1: Use more iterations
python fitter_articulation.py optim=accurate

# Solution 2: Start from clearer frame
python fitter_articulation.py fitter.start_frame=10

# Solution 3: Check keypoint quality
# Inspect: data/preprocessed/*/keypoints2d_undist/result_view_0.pkl
```

**Problem**: Rendering produces black images
```bash
# Solution 1: Disable rendering during debugging
python fitter_articulation.py fitter.with_render=false

# Solution 2: Check EGL libraries
ldconfig -p | grep EGL

# Solution 3: Use CPU rendering (slower)
export PYOPENGL_PLATFORM=osmesa
```

**Problem**: Very slow processing
```bash
# Solution 1: Disable rendering
python fitter_articulation.py fitter.with_render=false

# Solution 2: Use fast optimization
python fitter_articulation.py optim=fast

# Solution 3: Process fewer frames
python fitter_articulation.py fitter.end_frame=10
```

### ML Training Issues

**Problem**: YOLOv8 training fails
```bash
# Check dataset format
python << EOF
import yaml
with open('data/training/yolo_enhanced/data.yaml') as f:
    config = yaml.safe_load(f)
    print(config)
# Should contain: train, val, nc (22), names (list of 22 keypoints)
EOF

# Verify images and labels match
ls data/training/yolo_enhanced/train/images/ | wc -l
ls data/training/yolo_enhanced/train/labels/ | wc -l
# Should be equal
```

**Problem**: Low mAP after training
```bash
# Solution 1: More labeled data (add 10-20 more images)
# Solution 2: More training epochs
python scripts/train_yolo_pose.py --epochs 200

# Solution 3: Data augmentation
python scripts/train_yolo_pose.py --augment
```

---

## 📈 Performance Benchmarks

### Processing Time (NVIDIA RTX 3090)

| Task | Frames | Time | FPS |
|------|--------|------|-----|
| Monocular fitting (geometric) | 10 | 5 min | 0.033 |
| Monocular fitting (YOLO) | 10 | 10 min | 0.017 |
| Multi-view fitting (no render) | 10 | 25 min | 0.007 |
| Multi-view fitting (with render) | 10 | 70 min | 0.002 |
| YOLOv8 training (100 epochs) | - | 30 min | - |
| Preprocessing (OpenCV) | 100 | 1 min | 1.67 |

### Memory Usage

| Task | GPU Memory | RAM |
|------|------------|-----|
| Monocular fitting | 3-4 GB | 8 GB |
| Multi-view fitting | 4-6 GB | 16 GB |
| YOLOv8 training | 4-5 GB | 8 GB |
| Preprocessing | 2 GB | 4 GB |

### Recommendations

- **Quick testing**: Use `optim=fast`, `fitter.end_frame=5`, `fitter.with_render=false`
- **Production quality**: Use `optim=accurate`, trained YOLO detector, `fitter.with_render=true`
- **Long videos**: Process in batches of 100 frames
- **Limited GPU**: Reduce batch size, use geometric detector

---

## 🎯 Mesh Fitting with Multiple Datasets

This project supports flexible mesh fitting across different dataset formats. See the comprehensive guide for details.

### Quick Reference

**Run with default dataset (multi-view):**
```bash
./run_mesh_fitting_default.sh 0 50     # frames 0-50
./run_mesh_fitting_default.sh 0 10 1 true  # with render
```

**Run with monocular fitting (single-view):**
```bash
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output yolo
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output geometric
```

**Run with cropped frames (single-view with masks):**
```bash
./run_mesh_fitting_cropped.sh data/100-KO-male-56-20200615_cropped
```

**Quick test (3 frames):**
```bash
./run_mesh_fitting_default.sh 0 3      # Multi-view test
./run_mesh_fitting_monocular.sh data/test/ results/test/ geometric 3  # Monocular test
```

### Supported Dataset Types

| Dataset | Location | Has Masks | Has Keypoints | Best Script |
|---------|----------|-----------|---------------|-------------|
| **Default Markerless** | `data/examples/markerless_mouse_1_nerf/` | ✅ | ✅ | `fitter_articulation.py` |
| **Cropped Frames** | `data/100-KO-male-56-20200615_cropped/` | ✅ | Optional | `fit_cropped_frames.py` |
| **Upsampled Frames** | `data/100-KO-male-56-20200615_upsampled/` | ❌ | ❌ | Needs preprocessing |
| **Custom** | User-defined | Varies | Varies | Configurable |

### Configuration System

The project uses Hydra for hierarchical configuration. Available dataset configs:

- `default_markerless` - Reference multi-view dataset with 6 cameras
- `cropped` - Cropped frames with masks (single-view)
- `upsampled` - Upsampled frames (requires mask generation)
- `shank3` - Shank3 experiment dataset
- `custom` - Template for your custom data

**Override configuration from command line:**
```bash
python fitter_articulation.py \
  dataset=cropped \
  data.data_dir=/path/to/data \
  fitter.start_frame=0 \
  fitter.end_frame=100 \
  fitter.with_render=true
```

### Output Structure

```
results/fitting/{dataset}_{timestamp}/
├── obj/
│   ├── mesh_000000.obj           # 3D mesh per frame
│   └── ...
├── params/
│   ├── param0.pkl                # Fitted parameters
│   ├── param0_sil.pkl            # After silhouette refinement
│   └── ...
├── render/                       # (if with_render=true)
│   ├── fitting_0.png             # Visualization overlay
│   └── debug/                    # Optimization debug images
└── .hydra/
    └── config.yaml               # Configuration used
```

**Hydra logs** are stored in: `results/logs/YYYY-MM-DD/HH-MM-SS/`

### Documentation

- **[Mesh Fitting Guide](docs/MESH_FITTING_GUIDE.md)** - Complete workflow and troubleshooting
- **[Quick Cheatsheet](MESH_FITTING_CHEATSHEET.md)** - Command reference

---

## 📚 Documentation

### Complete Guides
- **[Mesh Fitting Guide](docs/MESH_FITTING_GUIDE.md)** - Multi-dataset mesh fitting workflows
- **[Monocular Fitting Guide](docs/guides/MONOCULAR_FITTING_GUIDE.md)** - Detailed single-view workflow
- **[Comprehensive Usage Guide](docs/guides/COMPREHENSIVE_USAGE_GUIDE.md)** - All usage scenarios
- **[Roboflow Labeling Guide](docs/ROBOFLOW_LABELING_GUIDE.md)** - Manual labeling tutorial
- **[SAM Mask Acquisition](docs/guides/SAM_MASK_ACQUISITION_MANUAL.md)** - High-quality masks

### Quick Reference
- **[Mesh Fitting Cheatsheet](MESH_FITTING_CHEATSHEET.md)** - Command quick reference

### Technical Reports
- **[ML Keypoint Detection](docs/reports/251115_comprehensive_ml_keypoint_summary.md)** - Complete ML workflow
- **[Implementation Summary](docs/reports/251114_ml_keypoint_detection_integration.md)** - Technical details
- **[All Reports](docs/reports/)** - Research session summaries

---

## 🎓 Key Concepts

### Three-Step Optimization

The fitting uses a progressive optimization strategy:

1. **Step 0: Global Initialization** (10 iters)
   - Objective: Find initial 3D pose
   - Uses: 2D keypoint reprojection only
   - Fast and robust to initialization

2. **Step 1: Joint Optimization** (100 iters)
   - Objective: Refine pose with all views
   - Uses: 2D keypoints + temporal smoothness
   - Main fitting stage

3. **Step 2: Silhouette Refinement** (30 iters)
   - Objective: Fine-tune surface details
   - Uses: Silhouette masks + PyTorch3D rendering
   - Highest quality but slower

### Keypoint Detectors

**Geometric** (Baseline):
- Extracts keypoints from silhouette contours
- Fast but low accuracy (~50% confidence)
- Good for: Quick testing, clean backgrounds

**YOLOv8-Pose** (Recommended):
- Pretrained on COCO, fine-tunable on your data
- Medium accuracy (~70-80% with fine-tuning)
- Good for: Most use cases

**SuperAnimal-TopViewMouse** (Future):
- Pretrained on 5K+ mice, highest accuracy
- Currently limited by API constraints
- Good for: Research applications when available

### Camera Models

**Single-View (Monocular)**:
- Estimates intrinsics from first frame
- Assumes: Known mouse size (~3cm body length)
- Limitations: Scale ambiguity, depth uncertainty

**Multi-View**:
- Uses calibrated camera parameters
- Assumes: Synchronized cameras, known calibration
- Advantages: Full 3D reconstruction, no scale ambiguity

---

## 🆕 Recent Updates

### 2025-11-25: Folder Organization and Monocular Pipeline
- ✅ Consolidated result folders to unified `results/` structure
- ✅ Added monocular fitting shell script (`run_mesh_fitting_monocular.sh`)
- ✅ Created monocular config (`conf/monocular.yaml`)
- ✅ Enhanced visualization with keypoint overlay
- ✅ Added keypoint selection by groups (head, spine, limbs, tail)
- ✅ Cleaned up git-tracked large files (2.4GB → 6.5MB)
- ✅ Updated all output paths in codebase

### 2025-11-15: Major Cleanup and Documentation
- ✅ Reorganized project structure (36 → 21 root items)
- ✅ Created comprehensive README with step-by-step examples
- ✅ Moved all scripts to `scripts/` directory
- ✅ Cleaned 410MB of archived outputs
- ✅ Updated all documentation paths

### 2025-11-14: ML Integration
- ✅ Monocular fitting pipeline (`fit_monocular.py`)
- ✅ YOLOv8-Pose integration
- ✅ SuperAnimal-TopViewMouse support
- ✅ Manual labeling workflow

### 2025-11-03: Preprocessing Improvements
- ✅ OpenCV-based preprocessing
- ✅ Geometric keypoint estimation
- ✅ SAM mask acquisition (experimental)

---

## 📊 Comparison with DANNCE

![comparison](assets/figs/mouse_2.png)

Results comparing DANNCE-T (temporal version) with MAMMAL_mouse on `markerless_mouse_1` sequence.

**MAMMAL_mouse advantages**:
- Full 3D mesh reconstruction (not just keypoints)
- Articulated model enforces anatomical constraints
- Compatible with single-view videos

**DANNCE advantages**:
- Faster processing
- Simpler setup (no model fitting)
- More robust to occlusions

---

## 📁 Project Structure

```
MAMMAL_mouse/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── # Core Python Files
├── fitter_articulation.py         # Main multi-view mesh fitter
├── fit_monocular.py               # Single-view monocular fitting
├── fit_cropped_frames.py          # Cropped frame fitting
├── articulation_th.py             # Articulation model (PyTorch)
├── bodymodel_th.py                # Body model (PyTorch)
├── bodymodel_np.py                # Body model (NumPy)
├── mouse_22_defs.py               # 22 keypoint definitions
├── utils.py                       # Utility functions
│
├── # Shell Scripts (Quick Start)
├── run_mesh_fitting_default.sh    # Multi-view fitting
├── run_mesh_fitting_monocular.sh  # Monocular fitting
├── run_mesh_fitting_cropped.sh    # Cropped frames fitting
├── run_unified_annotator.sh       # Launch annotation tool
│
├── # Configuration
├── conf/                          # Hydra configs
│   ├── config.yaml                # Main config
│   ├── monocular.yaml             # Monocular fitting config
│   └── dataset/                   # Dataset-specific configs
│       ├── default_markerless.yaml
│       ├── cropped.yaml
│       └── custom.yaml
│
├── # Scripts (Organized)
├── scripts/
│   ├── annotators/                # Annotation tools
│   │   ├── unified_annotator.py   # Mask + Keypoint tool (Gradio)
│   │   └── keypoint_annotator_v2.py
│   ├── preprocessing/             # Video preprocessing
│   │   └── extract_video_frames.py
│   ├── setup/                     # Installation scripts
│   │   ├── setup.sh
│   │   ├── download_superanimal.py
│   │   └── sample_images_for_labeling.py
│   ├── utils/                     # Utility scripts
│   │   ├── convert_keypoints_to_mammal.py
│   │   └── process_video_with_sam.py
│   ├── tests/                     # Test scripts
│   ├── deprecated/                # Old/replaced scripts
│   ├── preprocess.py
│   ├── evaluate.py
│   └── train_yolo_pose.py
│
├── # Preprocessing Utilities
├── preprocessing_utils/
│   ├── keypoint_estimation.py     # Geometric keypoint detector
│   ├── yolo_keypoint_detector.py  # YOLO-Pose detector
│   ├── superanimal_detector.py    # SuperAnimal detector
│   ├── mask_processing.py         # Mask utilities
│   ├── sam_inference.py           # SAM integration
│   └── silhouette_renderer.py     # PyTorch3D rendering
│
├── # Assets (tracked)
├── mouse_model/                   # MAMMAL parametric model
│   ├── mouse.pkl                  # Main model file
│   └── mouse_txt/                 # Auxiliary files
│
├── # Documentation
├── docs/
│   ├── guides/                    # Usage guides
│   └── reports/                   # Research notes (YYMMDD_*.md)
│
├── # Models (git-ignored, download separately)
├── models/
│   ├── README.md                  # Download instructions
│   ├── pretrained/                # SAM, YOLO base models
│   └── trained/                   # Fine-tuned models
│
├── # Data (git-ignored)
├── data/                          # Input datasets
│
└── # Results (git-ignored)
└── results/
    ├── fitting/                   # Mesh fitting outputs
    ├── monocular/                 # Monocular fitting outputs
    └── logs/                      # Hydra logs
```

---

## 📧 Support

### Getting Help

1. **Check documentation**:
   - This README
   - `docs/guides/` for detailed tutorials
   - `docs/reports/` for technical details

2. **Common issues**:
   - See Troubleshooting section above
   - Check existing GitHub issues

3. **Report bugs**:
   - Open GitHub issue with:
     - Error message and full traceback
     - Your environment: `conda list > environment.txt`
     - Config file used
     - Minimal reproducible example

### Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests if applicable
4. Update documentation
5. Submit pull request

---

## 🙏 Acknowledgments

- **MAMMAL framework**: An et al. (2023)
- **Virtual mouse model**: Bolanos et al. (2021)
- **DANNCE dataset**: Dunn et al. (2021)
- **PyTorch3D**: Meta AI Research
- **YOLOv8**: Ultralytics
- **SuperAnimal**: Mathis Lab

---

## 📄 License

[Specify your license here]

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{MAMMAL,
    author = {An, Liang and Ren, Jilong and Yu, Tao and Hai, Tang and Jia, Yichang and Liu, Yebin},
    title = {Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
    journal = {},
    year = {2023}
}

@article{bolanos2021three,
  title={A three-dimensional virtual mouse generates synthetic training data for behavioral analysis},
  author={Bola{\~n}os, Luis A and Xiao, Dongsheng and Ford, Nancy L and LeDue, Jeff M and Gupta, Pankaj K and Doebeli, Carlos and Hu, Hao and Rhodin, Helge and Murphy, Timothy H},
  journal={Nature methods},
  volume={18},
  number={4},
  pages={378--381},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```
