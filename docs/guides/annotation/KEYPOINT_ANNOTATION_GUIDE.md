# Keypoint Annotation Guide

## Quick Start

### 1. Launch Annotator

```bash
conda activate mammal_stable
bash run_keypoint_annotator.sh
```

Or with custom settings:
```bash
bash run_keypoint_annotator.sh \
    data/100-KO-male-56-20200615_cropped \
    data/keypoints_manual.json \
    7861
```

### 2. Access Web Interface

Open in browser: `http://localhost:7861`

For remote server (SSH tunnel):
```bash
# On local machine
ssh -N -L 7861:localhost:7861 user@server
```

Then open: `http://localhost:7861`

## Annotation Workflow

### Essential Keypoints (7 points)

Click on image to mark these locations:

1. **🔴 nose**: Tip of the nose/snout
2. **🟠 neck**: Base of skull / start of neck
3. **🟡 spine_mid**: Middle of the spine/back
4. **🟢 hip**: Hip/pelvis region
5. **🔵 tail_base**: Where tail starts (base of tail)
6. **🟣 left_ear**: Left ear tip (optional)
7. **🟣 right_ear**: Right ear tip (optional)

### Step-by-Step

1. **Load Frame**:
   - Use slider or click "Load Frame"
   - Frame 0 loads automatically

2. **Select Keypoint**:
   - Choose from radio buttons (e.g., "nose")

3. **Click on Image**:
   - Click precise location on mouse body
   - Colored dot appears immediately
   - Label shows keypoint name

4. **Repeat** for all keypoints:
   - Select next keypoint from radio buttons
   - Click location
   - Continue until all 7 are marked

5. **Save**:
   - Click "💾 Save Keypoints"
   - Progress shown in summary

6. **Next Frame**:
   - Click "Next ➡️" or use slider
   - Repeat process

### Keyboard Shortcuts

- **Arrow Keys**: Navigate frames (when slider focused)
- **Click**: Add keypoint

### Tips

**Accuracy**:
- Zoom browser if needed (Ctrl/Cmd + Plus)
- Click center of body part
- Use anatomical landmarks

**Speed**:
- Do all frames for one keypoint first (nose on all frames)
- Then move to next keypoint
- Faster than all points per frame

**Quality**:
- Minimum 5 keypoints: nose, neck, spine_mid, hip, tail_base
- Ears optional (often occluded)
- Consistent placement more important than perfect accuracy

## Output Format

Annotations saved to `keypoints_manual.json`:

```json
{
  "frame_000000": {
    "nose": {"x": 125.3, "y": 89.2, "visibility": 1.0},
    "neck": {"x": 118.5, "y": 102.1, "visibility": 1.0},
    "spine_mid": {"x": 110.2, "y": 115.3, "visibility": 1.0},
    "hip": {"x": 95.1, "y": 128.4, "visibility": 1.0},
    "tail_base": {"x": 82.3, "y": 135.7, "visibility": 1.0}
  },
  "frame_000001": {
    ...
  }
}
```

## Features

### Navigation
- **Slider**: Jump to any frame
- **Prev/Next**: Step through frames
- **Auto-load**: Existing annotations loaded automatically

### Editing
- **Add**: Click to add/update keypoint
- **Remove**: Select keypoint and click "Remove"
- **Overwrite**: Click again to update position

### Progress Tracking
- **Summary Panel**: Shows completed keypoints
- **✅/❌ Indicators**: Quick visual feedback
- **Frame Counter**: X/20 frames completed

### Auto-save
- Saves to JSON after each frame
- Resume anytime - progress preserved
- No data loss on browser close

## Estimated Time

- **Per Frame**: 1-2 minutes (7 keypoints)
- **20 Frames**: 20-40 minutes total
- **First frame**: Slower (learning)
- **Later frames**: Faster (muscle memory)

## After Annotation

Once complete (all 20 frames), use keypoints for fitting:

```bash
python fit_with_keypoints.py \
    data/100-KO-male-56-20200615_cropped \
    --keypoints data/keypoints_manual.json \
    --output results/keypoint_fitting
```

Expected improvements with keypoints:
- IoU: 46% → 60-75%
- Pose accuracy: Much better
- Converges faster

## Troubleshooting

### Port Already in Use
```bash
# Use different port
bash run_keypoint_annotator.sh . . 7862
```

### Can't See Image
- Check frames directory path
- Ensure cropped frames exist
- Look for `*_cropped.png` files

### Keypoints Not Saving
- Check file permissions
- Ensure directory writable
- Check console for errors

### Wrong Keypoint Placed
- Select keypoint from dropdown
- Click "Remove Selected"
- Or just click again to update

## Comparison: Heuristic vs Manual

| Method | Accuracy | Time | Spine | Limbs |
|--------|----------|------|-------|-------|
| Heuristic | Low | 0 min | ~50% | ❌ |
| Manual | High | 30 min | ~95% | ✅ |
| DeepLabCut | Very High | 2 hours setup | ~98% | ✅ |

**Recommendation**: Manual annotation for 20 frames (best time/quality trade-off)

## Example Session

```
Frame 0:
  - Click nose → ✅
  - Click neck → ✅
  - Click spine_mid → ✅
  - Click hip → ✅
  - Click tail_base → ✅
  - Save → "✅ Saved 5 keypoints for frame_000000"

Frame 1:
  - Next →
  - Repeat ...

... continue for 20 frames ...

Final: 20/20 frames ✅
Output: data/keypoints_manual.json
```

## Next Steps

1. **Annotate 20 frames** (~30 minutes)
2. **Verify output** (`cat data/keypoints_manual.json | head -50`)
3. **Run fitting** with keypoints
4. **Compare results** to silhouette-only fitting

Your IoU should improve from ~46% to ~65-75%!
