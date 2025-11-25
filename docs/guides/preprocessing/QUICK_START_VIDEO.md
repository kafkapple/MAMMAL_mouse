# Quick Start: Video Processing with SAM

**Goal**: Extract frames from video, annotate mouse regions with SAM, and prepare for mesh fitting.

## TL;DR - Complete Workflow

```bash
# 1. Extract frames (20 frames, evenly spaced)
conda run -n mammal_stable python extract_video_frames.py \
    /home/joon/dev/data/100-KO-male-56-20200615.avi \
    --output-dir data/100-KO-male-56/frames \
    --num-frames 20

# 2. Preview frames
conda run -n mammal_stable python visualize_extracted_frames.py \
    data/100-KO-male-56/frames

# 3. Run SAM annotator (web UI)
# NOTE: Must activate conda first (conda run conflicts with Hydra)
conda activate mammal_stable
python run_sam_gui.py \
    --frames-dir data/100-KO-male-56/frames \
    --port 7860

# → Annotate frames in browser (http://localhost:7860 or via SSH tunnel)
# → Add foreground/background points, generate masks, save

# 4. Process annotations and crop frames
conda run -n mammal_stable python process_annotated_frames.py \
    data/100-KO-male-56/frames/annotations \
    --output-dir data/100-KO-male-56/cropped \
    --padding 50

# 5. Run mesh fitting (TODO: integrate with existing fitter)
# python fitter_articulation.py --frames-dir data/100-KO-male-56/frames ...
# python fitter_articulation.py --frames-dir data/100-KO-male-56/cropped ...
```

## Step-by-Step Instructions

### 1. Extract Frames

```bash
VIDEO="/home/joon/dev/data/100-KO-male-56-20200615.avi"
OUTPUT="data/100-KO-male-56/frames"

conda run -n mammal_stable python extract_video_frames.py \
    "$VIDEO" \
    --output-dir "$OUTPUT" \
    --num-frames 20
```

**Output**: 20 PNG frames in `data/100-KO-male-56/frames/`

### 2. Preview Frames (Optional)

```bash
conda run -n mammal_stable python visualize_extracted_frames.py \
    data/100-KO-male-56/frames
```

**Output**: `frames_preview.png` showing grid of all extracted frames

### 3. Annotate with SAM

```bash
# Must activate conda environment first
conda activate mammal_stable

python run_sam_gui.py \
    --frames-dir data/100-KO-male-56/frames \
    --port 7860
```

**Important**: Don't use `conda run` - it conflicts with Hydra. Activate environment first.

**Access**:
- Local: http://localhost:7860
- Remote: `ssh -L 7860:localhost:7860 joon@server`, then http://localhost:7860

**Annotation Steps**:
1. Load Frame → Select "Foreground" → Click on mouse (3-5 points)
2. Select "Background" → Click on background (1-2 points)
3. Click "Generate Mask" → Review mask
4. If good, click "Save Annotation"
5. Move to next frame, repeat

**Output**: Annotations and masks in `data/100-KO-male-56/frames/annotations/`

### 4. Process & Crop

```bash
conda run -n mammal_stable python process_annotated_frames.py \
    data/100-KO-male-56/frames/annotations \
    --output-dir data/100-KO-male-56/cropped \
    --padding 50
```

**Output**: Cropped frames and visualization in `data/100-KO-male-56/cropped/`

### 5. Mesh Fitting

**Option A: Full frames**
```bash
# TODO: Add mesh fitting command for full frames
```

**Option B: Cropped frames**
```bash
# TODO: Add mesh fitting command for cropped frames
```

## File Organization

```
data/100-KO-male-56/
├── frames/                          # Step 1: Extracted frames
│   ├── frame_000000.png
│   ├── ...
│   ├── extraction_metadata.json
│   ├── frames_preview.png           # Step 2: Preview
│   └── annotations/                 # Step 3: SAM annotations
│       ├── frame_000000_annotation.json
│       ├── frame_000000_mask.png
│       └── ...
└── cropped/                         # Step 4: Cropped frames
    ├── frame_000000_cropped.png
    ├── frame_000000_mask.png
    ├── processing_summary.json
    └── processing_visualization.png
```

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `extract_video_frames.py` | Extract frames from video | `python extract_video_frames.py video.avi --output-dir frames --num-frames 20` |
| `visualize_extracted_frames.py` | Preview extracted frames | `python visualize_extracted_frames.py frames` |
| `run_sam_annotator.sh` | Launch SAM annotation UI | `bash run_sam_annotator.sh 7860` |
| `process_annotated_frames.py` | Crop frames using masks | `python process_annotated_frames.py annotations --output-dir cropped` |

## Common Issues

### SAM Annotator Won't Start

**Port already in use:**
```bash
lsof -i :7860
kill -9 <PID>
# Or use different port:
bash run_sam_annotator.sh 8080
```

**SAM checkpoint not found:**
```bash
cd ~/dev/segment-anything-2/checkpoints
./download_ckpts.sh
```

### Can't Access Web UI

**From remote server:**
```bash
# On local machine:
ssh -L 7860:localhost:7860 joon@server
# Then open: http://localhost:7860
```

### Poor Mask Quality

- Add more foreground points (3-5 on mouse body)
- Add background points (1-2 on floor/walls)
- Click "Clear" and try different point placement
- Use smaller SAM model if running out of memory

## Tips

### For Best Results
- **Frame selection**: Start with 10-20 frames for testing
- **Point placement**: Foreground on mouse center, background on clear areas
- **Padding**: 50-100 pixels recommended for cropping
- **Verification**: Always check `processing_visualization.png`

### Speed Up
- Use `--num-frames 10` for quick tests
- Use `sam2.1_hiera_small` model in `run_sam_annotator.sh`
- Annotate every 2nd frame if many frames

### Quality Check
```bash
# Preview frames before annotation
python visualize_extracted_frames.py frames

# Check cropping results
open cropped/processing_visualization.png
```

## Next Steps

After cropping frames:
1. Run mesh fitting on both full and cropped frames
2. Compare results (accuracy, speed)
3. Visualize mesh overlays on frames
4. Export 3D meshes for analysis

## Full Documentation

See `VIDEO_PROCESSING_GUIDE.md` for complete details.

---

**Current Video**: `100-KO-male-56-20200615.avi` (640x480, 30fps, 15min)
**Frames Extracted**: 20 frames (evenly spaced)
**Next**: Annotate frames with SAM UI
