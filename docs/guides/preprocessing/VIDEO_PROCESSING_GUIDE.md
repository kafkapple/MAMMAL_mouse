# Video Processing with SAM Annotation & Mesh Fitting Guide

Complete workflow for processing mouse videos with SAM-based annotation and mesh fitting.

## Overview

This pipeline allows you to:
1. Extract frames from a video
2. Annotate mouse regions using SAM (Segment Anything Model) via web UI
3. Crop frames based on annotations
4. Run mesh fitting on both original and cropped frames
5. Visualize and compare results

## Prerequisites

### Environment Setup

```bash
# Activate conda environment
conda activate mammal_stable

# Install required packages (if not already installed)
pip install gradio opencv-python matplotlib tqdm
```

### SAM 2 Setup

```bash
# Clone SAM 2 repository
cd ~/dev
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Download checkpoints
cd checkpoints
./download_ckpts.sh
```

## Workflow

### Step 1: Extract Frames from Video

Extract frames from your video file:

```bash
# Example video: /home/joon/dev/data/100-KO-male-56-20200615.avi
VIDEO_PATH="/home/joon/dev/data/100-KO-male-56-20200615.avi"
OUTPUT_DIR="/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_frames"

# Extract 20 evenly-spaced frames
conda run -n mammal_stable python extract_video_frames.py \
    "$VIDEO_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames 20
```

**Options:**
- `--num-frames N`: Extract N evenly-spaced frames
- `--fps-sample 1.0`: Sample 1 frame per second
- `--frame-indices 0 100 500`: Extract specific frame indices
- `--all`: Extract all frames

**Output:**
- `frame_XXXXXX.png`: Extracted frames
- `extraction_metadata.json`: Metadata with timestamps and frame indices

### Step 2: Annotate Frames with SAM

Run the SAM annotation web interface:

```bash
# Run SAM annotator
bash run_sam_annotator.sh 7860
```

**Access the web UI:**
- **Local**: http://localhost:7860
- **Remote (SSH tunnel)**:
  ```bash
  # On your local machine:
  ssh -L 7860:localhost:7860 joon@server
  # Then open: http://localhost:7860
  ```

**Annotation Workflow:**

1. **Load Frame**: Use slider to select frame → Click "📂 Load Frame"
2. **Add Points**:
   - Select "Foreground" → Click on the mouse (green points)
   - Select "Background" → Click on the background (red points)
   - Add 3-5 points for best results
3. **Generate Mask**: Click "🎯 Generate Mask"
4. **Review**: Check the mask overlay and binary mask
5. **Save**: If satisfied, click "💾 Save Annotation"
6. **Next Frame**: Move to next frame and repeat

**Tips:**
- Add at least 2-3 foreground points on the mouse body
- Add 1-2 background points for better segmentation
- You can click "🗑️ Clear" and retry if not satisfied

**Output** (in `annotations/` subdirectory):
- `frame_XXXXXX_annotation.json`: Point coordinates and labels
- `frame_XXXXXX_mask.png`: Binary segmentation mask

### Step 3: Process Annotations & Crop Frames

After annotation, crop frames based on SAM masks:

```bash
ANNOTATIONS_DIR="/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_frames/annotations"
CROPPED_DIR="/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_cropped"

conda run -n mammal_stable python process_annotated_frames.py \
    "$ANNOTATIONS_DIR" \
    --output-dir "$CROPPED_DIR" \
    --padding 50
```

**Options:**
- `--padding N`: Padding around detected region (default: 50 pixels)
- `--no-visualize`: Skip visualization

**Output:**
- `frame_XXXXXX_cropped.png`: Cropped frames
- `frame_XXXXXX_mask.png`: Cropped masks
- `frame_XXXXXX_crop_info.json`: Crop metadata
- `processing_summary.json`: Processing statistics
- `processing_visualization.png`: Before/after comparison

### Step 4: Run Mesh Fitting

#### Option A: Full Frames (Original Resolution)

Run mesh fitting on original extracted frames:

```bash
# TODO: Update with actual mesh fitting command
# Example:
# python fitter_articulation.py \
#     --frames-dir "$OUTPUT_DIR" \
#     --output-dir results/full_frames
```

#### Option B: Cropped Frames (Mouse Region Only)

Run mesh fitting on cropped frames:

```bash
# TODO: Update with actual mesh fitting command
# Example:
# python fitter_articulation.py \
#     --frames-dir "$CROPPED_DIR" \
#     --output-dir results/cropped_frames
```

**Benefits of Cropped Frames:**
- Smaller resolution → Faster processing
- Focus on mouse region → Better accuracy
- Reduced background noise

### Step 5: Visualize Results

Compare mesh fitting results:

```bash
# TODO: Add visualization script
# Example:
# python visualize_mesh_fitting.py \
#     --full-frames results/full_frames \
#     --cropped-frames results/cropped_frames
```

## Complete Example

```bash
# 1. Extract frames
VIDEO="/home/joon/dev/data/100-KO-male-56-20200615.avi"
PROJECT_DIR="/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615"

conda run -n mammal_stable python extract_video_frames.py \
    "$VIDEO" \
    --output-dir "${PROJECT_DIR}/frames" \
    --num-frames 20

# 2. Annotate (web UI - manual step)
bash run_sam_annotator.sh 7860
# → Annotate frames in browser
# → Press Ctrl+C when done

# 3. Process annotations
conda run -n mammal_stable python process_annotated_frames.py \
    "${PROJECT_DIR}/frames/annotations" \
    --output-dir "${PROJECT_DIR}/cropped" \
    --padding 50

# 4. Run mesh fitting (both versions)
# Full frames
# python fitter_articulation.py --frames-dir "${PROJECT_DIR}/frames" ...

# Cropped frames
# python fitter_articulation.py --frames-dir "${PROJECT_DIR}/cropped" ...
```

## Directory Structure

After running the complete pipeline:

```
data/100-KO-male-56-20200615/
├── frames/                                 # Step 1 output
│   ├── frame_000000.png
│   ├── frame_000001.png
│   ├── ...
│   ├── extraction_metadata.json
│   └── annotations/                        # Step 2 output
│       ├── frame_000000_annotation.json
│       ├── frame_000000_mask.png
│       └── ...
├── cropped/                                # Step 3 output
│   ├── frame_000000_cropped.png
│   ├── frame_000000_mask.png
│   ├── frame_000000_crop_info.json
│   ├── processing_summary.json
│   └── processing_visualization.png
└── results/                                # Step 4 output
    ├── full_frames/
    │   └── mesh_fitting_results...
    └── cropped_frames/
        └── mesh_fitting_results...
```

## Troubleshooting

### SAM Annotator Issues

**Port already in use:**
```bash
# Check what's using the port
lsof -i :7860

# Kill the process
kill -9 <PID>

# Or use a different port
bash run_sam_annotator.sh 8080
```

**CUDA out of memory:**
```bash
# Use smaller SAM model in run_sam_annotator.sh:
# Change: model.name="sam2.1_hiera_large"
# To:     model.name="sam2.1_hiera_small"
```

**Can't access web UI remotely:**
```bash
# Make sure you're using SSH tunnel:
ssh -L 7860:localhost:7860 joon@server

# Or enable Gradio share (public URL):
# Edit run_sam_annotator.sh:
# ui.share=true
```

### Frame Extraction Issues

**Video won't open:**
```bash
# Check video codec
ffmpeg -i your_video.avi

# Convert if needed
ffmpeg -i input.avi -c:v libx264 output.mp4
```

**Low resolution frames:**
- Video is already 640x480 (low resolution)
- Consider upscaling before processing:
  ```bash
  ffmpeg -i input.avi -vf scale=1280:960 output.avi
  ```

### Annotation Processing Issues

**No masks found:**
- Make sure to annotate frames in SAM annotator
- Check that `frame_*_mask.png` files exist
- Verify annotations with `has_mask: true` in JSON

**Empty crops:**
- Increase `--padding` value
- Check mask quality in SAM annotator
- Re-annotate problematic frames

## Best Practices

### Frame Selection
- **Start small**: 10-20 frames for initial testing
- **Diverse samples**: Select frames with different poses/positions
- **Even spacing**: Use `--num-frames` for temporal coverage
- **Key moments**: Use `--frame-indices` for specific events

### SAM Annotation
- **Foreground points**: 3-5 points on mouse body (head, back, tail)
- **Background points**: 1-2 points on arena floor/walls
- **Point placement**: Avoid edges, stay in clear regions
- **Quality check**: Ensure mask covers entire mouse, minimal background

### Cropping
- **Padding**: 50-100 pixels recommended
- **Consistency**: Use same padding for all frames
- **Verification**: Check `processing_visualization.png`

### Mesh Fitting
- **Start with cropped**: Faster, better convergence
- **Compare both**: Full vs cropped for quality assessment
- **Batch processing**: Process multiple frames together

## Performance Tips

### Speed Up Annotation
- Use `sam2.1_hiera_small` model instead of `large`
- Annotate every 2nd or 3rd frame initially
- Save frequently to avoid re-work

### Speed Up Processing
- Process in parallel if multiple GPUs available
- Use cropped frames for faster mesh fitting
- Reduce frame count for initial experiments

## Advanced Usage

### Custom Frame Selection

Extract specific time ranges:
```python
# Extract frames from 5min to 10min
python extract_video_frames.py video.avi \
    --fps-sample 2.0 \
    --frame-indices $(seq 9000 60 18000)
```

### Batch Video Processing

Process multiple videos:
```bash
for video in /path/to/videos/*.avi; do
    basename=$(basename "$video" .avi)
    python extract_video_frames.py "$video" \
        --output-dir "data/${basename}/frames" \
        --num-frames 20
done
```

### Automated Pipeline

Skip manual annotation (use automatic SAM):
```bash
# Extract frames
python extract_video_frames.py video.avi --output-dir frames --num-frames 10

# Auto-annotate (if SAM works well automatically)
# python auto_annotate_sam.py frames --output-dir frames/annotations

# Process & crop
python process_annotated_frames.py frames/annotations --output-dir cropped
```

## References

- SAM 2: https://github.com/facebookresearch/segment-anything-2
- Mouse-Super-Resolution: `/home/joon/dev/mouse-super-resolution`
- MAMMAL Project: Current repository

## Support

For issues or questions:
1. Check this guide
2. Review error logs
3. Check SAM 2 documentation
4. Contact: joon@example.com

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
