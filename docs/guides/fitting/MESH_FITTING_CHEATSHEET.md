# Mesh Fitting Cheatsheet

Quick reference guide for mouse 3D mesh fitting.

---

## 🚀 Quick Start Commands

### 1. Default Dataset (Multi-View Reference)

```bash
# Full run with rendering
./run_mesh_fitting_default.sh 0 50 1 true

# Quick test (3 frames, no render)
./run_quick_test.sh default_markerless
```

### 2. Cropped Frames (With Masks)

```bash
# All frames
./run_mesh_fitting_cropped.sh

# First 10 frames only
./run_mesh_fitting_cropped.sh \
  data/100-KO-male-56-20200615_cropped \
  results/test_fitting \
  10
```

### 3. Custom Dataset

```bash
# Using existing config
./run_mesh_fitting_custom.sh cropped /path/to/data 0 100 false

# Direct Python call
python fitter_articulation.py \
  dataset=custom \
  data.data_dir=/path/to/data \
  fitter.start_frame=0 \
  fitter.end_frame=50
```

---

## 📁 Dataset Quick Reference

| Dataset | Location | Has Masks | Script | Config |
|---------|----------|-----------|--------|--------|
| **Default Markerless** | `data/examples/markerless_mouse_1_nerf/` | ✅ | `fitter_articulation.py` | `default_markerless` |
| **Cropped** | `data/100-KO-male-56-20200615_cropped/` | ✅ | `fit_cropped_frames.py` | `cropped` |
| **Upsampled** | `data/100-KO-male-56-20200615_upsampled/` | ❌ | Needs preprocessing | `upsampled` |

---

## 🔧 Configuration Options

### Command Line Overrides

```bash
# Change dataset path
python fitter_articulation.py data.data_dir=/new/path

# Change frame range
python fitter_articulation.py \
  fitter.start_frame=0 \
  fitter.end_frame=100 \
  fitter.interval=5

# Enable rendering
python fitter_articulation.py fitter.with_render=true

# Multiple overrides
python fitter_articulation.py \
  dataset=cropped \
  data.data_dir=/path/to/data \
  fitter.with_render=true \
  result_folder=results/my_experiment/
```

### Available Dataset Configs

- `default_markerless` - Reference multi-view dataset
- `cropped` - Cropped frames with masks
- `upsampled` - Upsampled frames (needs masks)
- `shank3` - Shank3 experiment dataset
- `custom` - Template for custom datasets

---

## 📊 Output Structure

```
outputs/YYYY-MM-DD/HH-MM-SS/
├── .hydra/
│   └── config.yaml              # Configuration used
├── fitter_articulation.log      # Execution log
└── results/
    ├── frame_0000/
    │   ├── mesh.obj             # Fitted mesh
    │   ├── params.json          # Parameters
    │   ├── render_view_0.png    # Visualization (if enabled)
    │   └── metrics.json         # Metrics
    └── summary.json             # Overall summary

results/cropped_fitting/         # fit_cropped_frames.py output
├── fitting_summary.json
└── frame_000000/
    ├── params.json
    └── comparison.png           # Target vs rendered
```

---

## 🔍 Common Tasks

### Test on Few Frames

```bash
# Quick test with default dataset
./run_quick_test.sh default_markerless

# Quick test with cropped dataset
./run_quick_test.sh cropped
```

### Process Full Dataset

```bash
# Default dataset (frames 0-500)
./run_mesh_fitting_default.sh 0 500 1 true

# Cropped dataset (all frames)
./run_mesh_fitting_cropped.sh \
  data/100-KO-male-56-20200615_cropped \
  results/full_fitting
```

### Custom Configuration

**1. Create config file:** `conf/dataset/my_dataset.yaml`

```yaml
# @package _global_
data:
  data_dir: /path/to/my/data
  views_to_use: [0]

fitter:
  start_frame: 0
  end_frame: 100
  with_render: true
```

**2. Run with config:**

```bash
python fitter_articulation.py dataset=my_dataset
```

---

## 🛠️ Preprocessing Workflow

### From Raw Video to Fitted Mesh

```bash
# 1. Extract frames
python extract_video_frames.py \
  --video data/raw/mouse_video.avi \
  --output data/mouse_frames

# 2. Annotate (SAM GUI)
python run_sam_gui.py
# Select frames directory: data/mouse_frames

# 3. Process annotations (generate crops)
python process_annotated_frames.py \
  --annotations-dir data/mouse_frames/annotations \
  --output-dir data/mouse_cropped

# 4. Fit mesh
python fit_cropped_frames.py \
  data/mouse_cropped \
  --output-dir results/mouse_fitting
```

---

## 🐛 Troubleshooting

### Problem: No masks found

```bash
# Check mask files exist
ls data/100-KO-male-56-20200615_cropped/*_mask.png

# Generate masks using SAM
python run_sam_gui.py
```

### Problem: CUDA out of memory

```bash
# Process fewer frames
python fit_cropped_frames.py data/cropped --max-frames 10

# Or reduce frame range
python fitter_articulation.py \
  fitter.start_frame=0 \
  fitter.end_frame=10
```

### Problem: Configuration not found

```bash
# List available configs
ls conf/dataset/

# Use correct config name (without .yaml)
python fitter_articulation.py dataset=cropped
```

---

## 📝 Key Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `data.data_dir` | Dataset root path | varies | `/path/to/data` |
| `fitter.start_frame` | First frame to process | 0 | 0 |
| `fitter.end_frame` | Last frame to process | varies | 100 |
| `fitter.interval` | Frame skip interval | 1 | 5 (every 5th) |
| `fitter.with_render` | Enable visualization | false | true |
| `fitter.keypoint_num` | Number of keypoints | 22 | 22 |
| `result_folder` | Output directory | `mouse_fitting_result/results/` | `results/exp1/` |

---

## 📖 Documentation

- **Full Guide:** `docs/MESH_FITTING_GUIDE.md`
- **Project README:** `README.md`
- **Configuration Files:** `conf/`
- **Report Archive:** `docs/reports/`

---

## 🎯 Best Practices

1. **Always test first:** Use `run_quick_test.sh` before full runs
2. **Use appropriate dataset config:** Match your data format
3. **Enable rendering for visualization:** Set `with_render=true`
4. **Process in batches:** Use frame intervals for large datasets
5. **Check output logs:** Review `.hydra/config.yaml` for used settings

---

## 💡 Tips

- Use `--max-frames` for quick testing with cropped frames
- Enable `with_render=true` to visualize fitting quality
- Use `interval=5` or `interval=10` for faster processing of long videos
- Check `fitting_summary.json` for per-frame loss values
- Create custom configs for repeated experiments

---

## 🔗 Related Scripts

| Script | Purpose |
|--------|---------|
| `fitter_articulation.py` | Main fitting script (multi-view) |
| `fit_cropped_frames.py` | Silhouette-based fitting (single-view) |
| `extract_video_frames.py` | Extract frames from video |
| `run_sam_gui.py` | SAM annotation GUI |
| `process_annotated_frames.py` | Generate cropped frames from annotations |
| `keypoint_annotator_v2.py` | Manual keypoint annotation |

---

**Last Updated:** 2025-11-17

For detailed documentation, see `docs/MESH_FITTING_GUIDE.md`.
