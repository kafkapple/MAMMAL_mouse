# MAMMAL_mouse

Three-dimensional surface motion capture of mice using the MAMMAL framework. This is a sub-project of the manuscript _Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL_.

By fitting an articulated 3D mouse model to video data, this project enables markerless 3D pose estimation and mesh reconstruction for behavioral analysis.

![mouse_model](figs/mouse_1.png)

## Features

- **Multi-view 3D fitting**: Fit 3D mouse model to synchronized multi-camera videos
- **Single-view preprocessing**: Automatically process single videos without manual annotation
- **Hydra configuration**: Flexible experiment management with dataset-specific configs
- **Modular pipeline**: Separate preprocessing and fitting stages for easy customization

## Comparison with DANNCE

![mouse_model2](figs/mouse_2.png)

The results above compare DANNCE-T (temporal version) with MAMMAL_mouse on the `markerless_mouse_1` sequence.

---

## Quick Start

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

#### Option A: Use Existing Multi-View Data

If you have the `markerless_mouse_1` dataset:

```bash
# Download data from Google Drive
# https://drive.google.com/file/d/1NbaIFOvpvQ_WLOabUtMrVHS7vVBq-8zD/view?usp=sharing
# Extract to data/markerless_mouse_1_nerf/

# Activate environment
conda activate mammal_stable

# Run fitting (using markerless dataset config)
python fitter_articulation.py dataset=markerless optim=fast fitter.end_frame=10
```

#### Option B: Process Your Own Single Video

```bash
# 1. Activate environment
conda activate mammal_stable

# 2. Update config for your video
# Edit conf/config.yaml or conf/dataset/custom.yaml:
#   preprocess.input_video_path: "path/to/your/video.mp4"
#   preprocess.output_data_dir: "data/preprocessed_custom/"

# 3. Run preprocessing
python preprocess.py dataset=custom mode=single_view_preprocess

# 4. Run fitting on preprocessed data
python fitter_articulation.py dataset=custom mode=multi_view fitter.end_frame=100
```

#### Using Shell Scripts

For convenience, use the provided scripts:

```bash
# Preprocessing
bash run_preprocess.sh

# Fitting
bash run_fitting.sh
```

---

## Configuration Management

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
| `data.data_dir` | Input data directory | `data/preprocessed_shank3/` |
| `fitter.start_frame` | First frame to process | `0` |
| `fitter.end_frame` | Last frame to process | `2` |
| `fitter.with_render` | Enable visualization rendering | `false` |
| `optim.solve_step1_iters` | Optimization iterations (step 1) | `100` |

---

## Processing Pipeline

### Stage 1: Preprocessing (Single-View Only)

Converts raw video into required inputs:
- **Input**: Single video file (`.mp4`, `.avi`, etc.)
- **Outputs**:
  - `videos_undist/0.mp4` - Original video
  - `simpleclick_undist/0.mp4` - Binary mask video
  - `keypoints2d_undist/result_view_0.pkl` - 2D keypoints (22 points)
  - `new_cam.pkl` - Camera parameters

**Current Method**: OpenCV-based
- Background subtraction for masks (BackgroundSubtractorMOG2)
- Geometric keypoint estimation from contours
- Automatic camera parameter generation

**Limitations**:
- Keypoint anatomical accuracy depends on geometric heuristics
- Mask quality sensitive to background changes

**Future Enhancements**:
- SAM (Segment Anything Model) for better masks
- DeepLabCut for anatomically accurate keypoints
- YOLO Pose for real-time processing

### Stage 2: 3D Fitting

Fits articulated 3D mouse model to preprocessed data:

**Three-Step Optimization**:
1. **Step 0**: Global pose initialization (10 iterations)
2. **Step 1**: Joint optimization with 2D keypoints (100 iterations)
3. **Step 2**: Silhouette-based refinement with PyTorch3D (30 iterations)

**Outputs**:
- `mouse_fitting_result/results/obj/` - 3D mesh files (`.obj`)
- `mouse_fitting_result/results/params/` - Fitting parameters (`.pkl`)
- `mouse_fitting_result/results/render/` - Visualization overlays (`.png`)
- `mouse_fitting_result/results/fitting_keypoints_*.png` - Keypoint comparisons

### Stage 3: Video Generation

Combine output images into video:

```bash
ffmpeg -framerate 10 -i mouse_fitting_result/results/render/fitting_%d.png \
       -c:v libx264 -pix_fmt yuv420p -y output.mp4
```

---

## Detailed Workflow

### For Custom Single-View Data

**1. Prepare Your Video**
- Place video file in `data/your_dataset/`
- Recommended: Static background, clear mouse visibility

**2. Create Dataset Config**
```yaml
# conf/dataset/your_dataset.yaml
# @package _global_

data:
  data_dir: data/preprocessed_your_dataset/
  views_to_use: [0]

preprocess:
  input_video_path: data/your_dataset/video.mp4
  output_data_dir: data/preprocessed_your_dataset/

fitter:
  start_frame: 0
  end_frame: 100  # Adjust to your video length
  render_cameras: [0]
```

**3. Run Preprocessing**
```bash
conda activate mammal_stable
python preprocess.py dataset=your_dataset mode=single_view_preprocess
```

**4. Verify Preprocessing Outputs**
```bash
ls data/preprocessed_your_dataset/
# Should contain: videos_undist/, simpleclick_undist/, keypoints2d_undist/, new_cam.pkl
```

**5. Run Fitting**
```bash
python fitter_articulation.py dataset=your_dataset mode=multi_view
```

**6. Create Output Video**
```bash
ffmpeg -framerate 10 -i mouse_fitting_result/results/render/fitting_%d.png \
       -c:v libx264 -pix_fmt yuv420p -y results_your_dataset.mp4
```

---

## Troubleshooting

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
- **Solution**: Current geometric method has limitations. Future work will integrate ML-based keypoint detectors

**Problem**: Fitting converges to incorrect pose
- **Solution**:
  - Use `optim=accurate` for more iterations
  - Verify 2D keypoint quality in `keypoints2d_undist/result_view_0.pkl`
  - Try adjusting `fitter.start_frame` to begin with clearer pose

---

## Advanced Configuration

### Loss Term Weights

Edit these in `fitter_articulation.py` (line ~82):

```python
self.term_weights = {
    "theta": 3,       # Articulation regularization
    "3d": 2.5,        # 3D keypoint loss
    "2d": 0.2,        # 2D reprojection loss
    "bone": 0.5,      # Bone length constraint
    "scale": 0.5,     # Scale regularization
    "mask": 0,        # Silhouette loss (disabled by default)
    "chest_deformer": 0.1,  # Chest deformation regularization
    "stretch": 1,     # Stretching penalty
    "temp": 0.25,     # Temporal smoothness
    "temp_d": 0.2     # Temporal derivative smoothness
}
```

### Keypoint Weights

Edit these in `fitter_articulation.py` (line ~65):

```python
self.keypoint_weight = np.ones(22)
self.keypoint_weight[4] = 0.4   # Right ear (lower confidence)
self.keypoint_weight[11] = 0.9  # Left hip (higher weight)
self.keypoint_weight[15] = 0.9  # Left foot (higher weight)
# ... adjust based on your data quality
```

---

## Project Structure

```
MAMMAL_mouse/
├── conf/                    # Hydra configuration files
│   ├── config.yaml          # Main config
│   ├── dataset/             # Dataset configs
│   ├── preprocess/          # Preprocessing configs
│   └── optim/               # Optimization configs
├── mouse_model/             # 3D mouse model files
│   ├── mouse.pkl            # Model definition
│   └── reg_weights.txt      # Regularization weights
├── data/                    # Data directory (gitignored)
│   ├── markerless_mouse_1_nerf/  # Original multi-view dataset
│   └── preprocessed_*/      # Preprocessed outputs
├── mouse_fitting_result/    # Fitting results (gitignored)
│   └── results/
│       ├── obj/             # 3D meshes
│       ├── params/          # Parameters
│       └── render/          # Visualizations
├── outputs/                 # Hydra outputs (gitignored)
├── reports/                 # Analysis reports
├── fitter_articulation.py   # Main fitting script
├── preprocess.py            # Preprocessing script
├── articulation_th.py       # Articulation model
├── bodymodel_th.py          # Body model
├── data_seaker_video_new.py # Data loader
├── setup.sh                 # Environment setup script
├── run_preprocess.sh        # Preprocessing runner
├── run_fitting.sh           # Fitting runner
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Known Issues

### Critical Issues (Being Addressed)

1. **Camera projection math error** (Line ~174 in `fitter_articulation.py`)
   - Matrix dimension mismatch in `calc_2d_keypoint_loss`
   - Fix in progress (see `reports/shank3_workflow_final_report.md`)

2. **PyTorch3D T vector shape incompatibility**
   - PyTorch3D expects T in `(N, 3)` format
   - Current code passes `(1, 3, 1)` or `(3, 1)`
   - Workaround: Add shape correction before PyTorch3D calls

### Limitations

- **Preprocessing accuracy**: Geometric keypoint estimation is approximate
- **Single-view ambiguity**: 3D reconstruction from single view is underconstrained
- **Background dependency**: Mask quality depends on static background

---

## Future Enhancements

See `PROJECT_ANALYSIS.md` for detailed implementation roadmap.

### Short-term (Phase 1-2)
- [x] Hydra configuration system
- [x] Single-view preprocessing
- [ ] Fix camera projection bugs
- [ ] Update environment to `mammal_stable`

### Medium-term (Phase 3-4)
- [ ] SAM integration for high-quality masks
- [ ] DeepLabCut/YOLO for accurate keypoints
- [ ] Multi-method preprocessing system
- [ ] Comprehensive unit tests

### Long-term
- [ ] Real-time processing pipeline
- [ ] Multi-animal tracking
- [ ] Temporal consistency improvements
- [ ] Interactive annotation tools

---

## Model Information

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

## Performance Notes

**Processing Time** (NVIDIA RTX 3090):
- Preprocessing: ~5-10 seconds per 100 frames
- Fitting (with render): ~7 minutes per frame
- Fitting (without render): ~2-3 minutes per frame

**Memory Usage**:
- Preprocessing: ~2GB GPU memory
- Fitting: ~4-6GB GPU memory

**Recommendations**:
- Start with `optim=fast` and small `end_frame` for testing
- Use `fitter.with_render=false` for faster iteration
- Process long videos in frame batches

---

## Citation

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

## Contact

If you encounter any problems using this code, please:
1. Check the troubleshooting section above
2. Review `PROJECT_ANALYSIS.md` for known issues
3. Open an issue on GitHub with:
   - Error message and full traceback
   - Your environment details (`conda list`)
   - Config file you're using
   - Sample data if possible

For general questions about the MAMMAL framework, please refer to the main paper.

---

## License

[Specify your license here]

---

## Acknowledgments

- Original MAMMAL framework by An et al.
- DANNCE dataset by Dunn et al.
- Virtual mouse model by Bolanos et al.
- PyTorch3D by Meta AI Research
