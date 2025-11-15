# MAMMAL_mouse

Three-dimensional surface motion capture of mice using the MAMMAL framework. This is a sub-project of the manuscript _Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL_.

By fitting an articulated 3D mouse model to video data, this project enables markerless 3D pose estimation and mesh reconstruction for behavioral analysis.

![mouse_model](assets/figs/mouse_1.png)

## ✨ Features

- **Multi-view 3D fitting**: Fit 3D mouse model to synchronized multi-camera videos
- **Single-view (monocular) fitting**: NEW! Process single videos with ML-based keypoint detection
- **ML keypoint detection**: YOLOv8-Pose and SuperAnimal support for anatomically accurate keypoints
- **Hydra configuration**: Flexible experiment management with dataset-specific configs
- **Modular pipeline**: Separate preprocessing and fitting stages for easy customization

## 📊 Comparison with DANNCE

![mouse_model2](assets/figs/mouse_2.png)

The results above compare DANNCE-T (temporal version) with MAMMAL_mouse on the `markerless_mouse_1` sequence.

---

## 🚀 Quick Start

### 1. Environment Setup

**Requirements**:
- Anaconda/Miniconda
- NVIDIA GPU with CUDA 11.8
- ~10GB disk space for dependencies

**Installation**:
```bash
# Clone the repository
git clone <repository_url>
cd MAMMAL_mouse

# Run the setup script (one-time setup)
bash setup.sh
```

This will create a `mammal_stable` conda environment with:
- Python 3.10
- PyTorch 2.0.0 + CUDA 11.8
- PyTorch3D 0.7.5
- All required dependencies

### 2. Basic Usage

#### Option A: Monocular Fitting (Single Video) - NEW! ⭐

Process a single video with ML-based keypoint detection:

```bash
# Activate environment
conda activate mammal_stable

# Run monocular fitting
python fit_monocular.py \
  --input_dir path/to/your/video_frames/ \
  --output_dir results/monocular/your_dataset \
  --detector geometric

# OR with fine-tuned YOLO (after manual labeling)
python fit_monocular.py \
  --input_dir path/to/your/video_frames/ \
  --output_dir results/monocular/your_dataset \
  --detector yolo \
  --yolo_weights models/trained/yolo/mammal_mouse_finetuned/weights/best.pt
```

See `docs/guides/MONOCULAR_FITTING_GUIDE.md` for detailed monocular fitting instructions.

#### Option B: Multi-View Fitting (Multiple Cameras)

If you have the `markerless_mouse_1` dataset:

```bash
# Download data from Google Drive
# https://drive.google.com/file/d/1NbaIFOvpvQ_WLOabUtMrVHS7vVBq-8zD/view?usp=sharing
# Extract to data/examples/markerless_mouse_1_nerf/

# Activate environment
conda activate mammal_stable

# Run fitting (using markerless dataset config)
python fitter_articulation.py dataset=markerless optim=fast fitter.end_frame=10
```

#### Option C: Process Your Own Single Video (Traditional Pipeline)

```bash
# 1. Activate environment
conda activate mammal_stable

# 2. Update config for your video
# Edit conf/config.yaml or conf/dataset/custom.yaml:
#   preprocess.input_video_path: "path/to/your/video.mp4"
#   preprocess.output_data_dir: "data/preprocessed/custom/"

# 3. Run preprocessing
python preprocess.py dataset=custom mode=single_view_preprocess

# 4. Run fitting on preprocessed data
python fitter_articulation.py dataset=custom mode=multi_view fitter.end_frame=100
```

---

## 📖 Documentation

### User Guides
- **[Monocular Fitting Guide](docs/guides/MONOCULAR_FITTING_GUIDE.md)** - NEW! Complete guide for single-view fitting
- **[Quick Start Labeling](docs/guides/QUICK_START_LABELING.md)** - Manual labeling workflow for ML training
- **[Roboflow Labeling Guide](docs/guides/ROBOFLOW_LABELING_GUIDE.md)** - Step-by-step Roboflow tutorial
- **[SAM Mask Acquisition](docs/guides/SAM_MASK_ACQUISITION_MANUAL.md)** - Using SAM for high-quality masks
- **[MAMMAL Architecture Manual](docs/guides/MAMMAL_ARCHITECTURE_MANUAL.md)** - Detailed architecture explanation

### Technical Reports
- **[ML Keypoint Detection Integration](docs/reports/251114_ml_keypoint_detection_integration.md)** - Technical report on ML integration
- **[Comprehensive ML Summary](docs/reports/251115_comprehensive_ml_keypoint_summary.md)** - Complete ML workflow documentation
- **[Session Reports](docs/reports/)** - All research session summaries and experiment reports

### Implementation Plans
- **[Implementation Plan](docs/guides/implementation_plan.md)** - Overall implementation roadmap
- **[SAM Preprocessing Plan](docs/guides/sam_preprocessing_plan.md)** - SAM integration plan

---

## 📁 Project Structure

```
MAMMAL_mouse/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── conf/                        # Hydra configuration files
│   ├── config.yaml              # Main config
│   ├── dataset/                 # Dataset configs
│   ├── preprocess/              # Preprocessing configs
│   └── optim/                   # Optimization configs
│
├── preprocessing_utils/         # Preprocessing modules
│   ├── keypoint_estimation.py   # Geometric keypoint estimation
│   ├── mask_processing.py       # Mask processing utilities
│   ├── yolo_keypoint_detector.py    # YOLOv8-Pose detector
│   ├── superanimal_detector.py      # SuperAnimal detector
│   ├── dannce_to_yolo.py            # Dataset conversion
│   └── visualize_yolo_labels.py     # Label visualization
│
├── Core scripts/                # Main execution scripts
│   ├── fit_monocular.py         # NEW! Monocular fitting
│   ├── fitter_articulation.py   # Main multi-view fitting
│   ├── preprocess.py            # Preprocessing pipeline
│   ├── train_yolo_pose.py       # YOLOv8 training
│   ├── articulation_th.py       # Articulation model
│   ├── bodymodel_th.py          # Body model
│   └── bodymodel_np.py          # NumPy body model
│
├── data/                        # Data directory
│   ├── raw/                     # Raw datasets
│   ├── preprocessed/            # Preprocessed outputs
│   ├── training/                # Training data for ML
│   │   ├── yolo_mouse_pose/     # YOLO dataset (geometric)
│   │   └── manual_labeling/     # Manual labels (in progress)
│   └── examples/                # Example datasets
│       └── markerless_mouse_1_nerf/
│
├── models/                      # Model weights
│   ├── pretrained/              # Pretrained models
│   │   └── superanimal_topviewmouse/
│   └── trained/                 # Your trained models
│       └── yolo/                # YOLO training runs
│
├── results/                     # Latest experiment results
│   ├── monocular/               # Monocular fitting results
│   ├── preprocessing/           # Preprocessing results
│   └── training/                # Training results
│
├── outputs/                     # Hydra outputs
│   └── archives/                # Archived old experiments
│
├── docs/                        # Documentation
│   ├── guides/                  # User guides
│   └── reports/                 # Technical reports
│
├── assets/                      # Static resources
│   ├── colormaps/               # Visualization colormaps
│   ├── figs/                    # README images
│   └── mouse_model/             # 3D mouse model files
│       ├── mouse.pkl            # Model definition
│       └── mouse_txt/           # Model parameters
│
├── tests/                       # Test scripts
│   ├── unit/
│   ├── integration/
│   └── outputs/                 # Test outputs
│
└── deprecated/                  # Deprecated files (for reference)
```

---

## 🔬 Processing Pipeline

### Stage 1: Preprocessing (Single-View Only)

Converts raw video into required inputs:
- **Input**: Single video file (`.mp4`, `.avi`, etc.)
- **Outputs**:
  - `videos_undist/0.mp4` - Original video
  - `simpleclick_undist/0.mp4` - Binary mask video
  - `keypoints2d_undist/result_view_0.pkl` - 2D keypoints (22 points)
  - `new_cam.pkl` - Camera parameters

**Available Methods**:
1. **OpenCV-based** (Current baseline)
   - Background subtraction for masks (BackgroundSubtractorMOG2)
   - Geometric keypoint estimation from contours
   - Automatic camera parameter generation

2. **Monocular Fitting** (NEW!)
   - ML-based keypoint detection (YOLOv8, SuperAnimal)
   - Direct 3D fitting from single-view images
   - No preprocessing stage required

**Future Enhancements**:
- SAM (Segment Anything Model) for better masks
- DeepLabCut integration

### Stage 2: 3D Fitting

Fits articulated 3D mouse model to preprocessed data:

**Three-Step Optimization**:
1. **Step 0**: Global pose initialization (10 iterations)
2. **Step 1**: Joint optimization with 2D keypoints (100 iterations)
3. **Step 2**: Silhouette-based refinement with PyTorch3D (30 iterations)

**Outputs**:
- `results/*/obj/` - 3D mesh files (`.obj`)
- `results/*/params/` - Fitting parameters (`.pkl`)
- `results/*/render/` - Visualization overlays (`.png`)
- `results/*/fitting_keypoints_*.png` - Keypoint comparisons

---

## 🎓 ML Keypoint Detection

### Quick Start: Manual Labeling + Fine-tuning

To improve keypoint detection quality (10-20× improvement expected):

```bash
# 1. Sample images for labeling (already done)
ls data/training/manual_labeling/images/  # 20 images ready

# 2. Follow Roboflow guide to label images
# See: docs/guides/ROBOFLOW_LABELING_GUIDE.md
# Time: ~2-3 hours for 20 images

# 3. Fine-tune YOLOv8 (after labeling)
python train_yolo_pose.py \
  --data data/training/yolo_mouse_pose_enhanced/data.yaml \
  --epochs 100 --batch 8 --imgsz 256 \
  --weights yolov8n-pose.pt \
  --name mammal_mouse_finetuned

# 4. Use fine-tuned model in monocular fitting
python fit_monocular.py \
  --detector yolo \
  --yolo_weights models/trained/yolo/mammal_mouse_finetuned/weights/best.pt
```

**Expected Improvements**:
- Confidence: 0.5 → 0.85+ (2× improvement)
- Loss: ~300K → 15-30K (10-20× improvement)
- Paw detection: 0% → 70-80%
- mAP: 0 → 0.6-0.8

See **[Comprehensive ML Summary](docs/reports/251115_comprehensive_ml_keypoint_summary.md)** for complete workflow.

---

## ⚙️ Configuration Management

This project uses [Hydra](https://hydra.cc/) for flexible configuration management.

### Config Structure

```
conf/
├── config.yaml          # Main config with defaults
├── dataset/             # Dataset-specific configs
│   ├── markerless.yaml  # Multi-view dataset (6 cameras)
│   ├── shank3.yaml      # Single-view dataset
│   └── custom.yaml      # Template for your data
├── preprocess/          # Preprocessing method configs
│   ├── opencv.yaml      # Current geometric approach
│   └── sam.yaml         # Future: SAM-based masking
└── optim/               # Optimization settings
    ├── fast.yaml        # Quick testing (fewer iterations)
    └── accurate.yaml    # High-quality results
```

### Usage Examples

```bash
# Use markerless dataset with accurate optimization
python fitter_articulation.py dataset=markerless optim=accurate

# Override specific parameters
python fitter_articulation.py dataset=shank3 fitter.end_frame=50 fitter.with_render=true

# Run preprocessing with custom config
python preprocess.py dataset=custom mode=single_view_preprocess
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mode` | Processing mode | `multi_view` |
| `dataset` | Dataset config to use | `shank3` |
| `data.data_dir` | Input data directory | `data/preprocessed/shank3_opencv/` |
| `fitter.start_frame` | First frame to process | `0` |
| `fitter.end_frame` | Last frame to process | `2` |
| `fitter.with_render` | Enable visualization rendering | `false` |
| `optim.solve_step1_iters` | Optimization iterations (step 1) | `100` |

---

## 🔧 Troubleshooting

### Environment Issues

**Problem**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Run `bash setup.sh` to create environment, then `conda activate mammal_stable`

**Problem**: `AttributeError: module 'distutils' has no attribute 'version'`
- **Solution**: This indicates environment corruption. Re-run `bash setup.sh`

**Problem**: `CUDA out of memory`
- **Solution**:
  - Reduce `fitter.end_frame` to process fewer frames
  - Use `optim=fast` for fewer iterations
  - Process frames in batches

### Rendering Issues

**Problem**: `NoSuchDisplayException: Cannot connect to "None"`
- **Solution**: Already handled by `export PYOPENGL_PLATFORM=egl` in scripts
- If issue persists, verify EGL libraries: `ldconfig -p | grep EGL`

**Problem**: Rendering produces black images
- **Solution**: Set `fitter.with_render=false` to skip rendering during debugging

### Data Issues

**Problem**: `FileNotFoundError: new_cam.pkl not found`
- **Solution**: Run preprocessing first: `bash run_preprocess.sh`

**Problem**: Poor keypoint quality in preprocessing
- **Solution**: Use monocular fitting with ML-based keypoint detection (see ML section above)

**Problem**: Fitting converges to incorrect pose
- **Solution**:
  - Use `optim=accurate` for more iterations
  - Verify 2D keypoint quality in `keypoints2d_undist/result_view_0.pkl`
  - Try adjusting `fitter.start_frame` to begin with clearer pose

---

## 🎯 Model Information

**Base Model**: C57BL6_Female_V1.2
- Source: _A three-dimensional virtual mouse generates synthetic training data for behavioral analysis_
- Original format: Blender file (`C57BL6_Female_V1.2_opensource-file.blend`)
- Keypoints: 22 anatomical landmarks following MAMMAL standard

**Keypoint Definitions**:
```
0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
9:left_paw, 10:right_paw, 11:left_hip, 12:right_hip,
13:left_knee, 14:right_knee, 15:left_foot, 16:right_foot,
17:neck, 18:tail_base, 19:wither, 20:center, 21:tail_middle
```

---

## 📈 Performance Notes

**Processing Time** (NVIDIA RTX 3090):
- Monocular fitting: ~30 seconds per frame (ML inference + 3D fitting)
- Traditional preprocessing: ~5-10 seconds per 100 frames
- Multi-view fitting (with render): ~7 minutes per frame
- Multi-view fitting (without render): ~2-3 minutes per frame

**Memory Usage**:
- Monocular fitting: ~3-4GB GPU memory
- Traditional preprocessing: ~2GB GPU memory
- Multi-view fitting: ~4-6GB GPU memory

**Recommendations**:
- Start with `optim=fast` and small `end_frame` for testing
- Use `fitter.with_render=false` for faster iteration
- Process long videos in frame batches
- For monocular fitting, use `--max_images` to limit batch size

---

## 🆕 Recent Updates (2025-11-15)

### NEW Features
- ✅ **Monocular Fitting Pipeline**: Direct 3D fitting from single-view images
- ✅ **ML Keypoint Detection**: YOLOv8-Pose and SuperAnimal support
- ✅ **Manual Labeling Workflow**: Complete end-to-end pipeline for training custom detectors
- ✅ **Comprehensive Documentation**: 10+ guides and technical reports

### Codebase Cleanup
- ✅ **Organized Project Structure**: Clear separation of docs/, data/, models/, results/
- ✅ **Standardized Naming**: YYMMDD_ prefix for all reports
- ✅ **Modular Architecture**: preprocessing_utils/ with clear responsibilities

### Next Steps
- ⏳ **Manual Labeling**: 20 images ready, labeling in progress
- ⏳ **YOLOv8 Fine-tuning**: Expected mAP 0.6-0.8 after labeling
- 📋 **SAM Integration**: Planned for high-quality mask generation

---

## 📚 Citation

If you found this project useful, please cite:

```BibTeX
@article{MAMMAL,
    author = {An, Liang and Ren, Jilong and Yu, Tao and Hai, Tang and Jia, Yichang and Liu, Yebin},
    title = {Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
    journal = {},
    year = {2023}
}
```

Mouse model citation:
```BibTeX
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

---

## 📧 Contact

If you encounter any problems using this code, please:
1. Check the troubleshooting section above
2. Review documentation in `docs/guides/` and `docs/reports/`
3. Open an issue on GitHub with:
   - Error message and full traceback
   - Your environment details (`conda list`)
   - Config file you're using
   - Sample data if possible

For general questions about the MAMMAL framework, please refer to the main paper.

---

## 🙏 Acknowledgments

- Original MAMMAL framework by An et al.
- DANNCE dataset by Dunn et al.
- Virtual mouse model by Bolanos et al.
- PyTorch3D by Meta AI Research
- Ultralytics YOLOv8 by Ultralytics
- DeepLabCut SuperAnimal by Mathis Lab

---

## 📄 License

[Specify your license here]
