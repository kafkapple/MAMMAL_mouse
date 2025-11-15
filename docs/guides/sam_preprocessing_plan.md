# SAM 기반 전처리 구현 계획서
**Date**: 2025-11-03
**Author**: Claude Code
**Project**: MAMMAL Mouse - Improved Preprocessing with SAM

---

## 1. 목표 (Objectives)

### 1.1 주요 목표
- SAM을 활용한 고품질 마우스 마스크 생성
- MAMMAL 22 keypoints에 맞춘 개선된 geometric keypoint estimation
- 기존 fitting pipeline과의 완벽한 호환성
- 전체 데이터셋 처리 가능한 확장성

### 1.2 성공 기준
- ✅ 마스크 검출률 > 90% (기존 0% → 90%+)
- ✅ Keypoints가 마우스 body에 위치 (기존 전부 0 → 실제 위치)
- ✅ Fitting loss 감소 확인
- ✅ 렌더링된 mesh와 비디오 정렬

---

## 2. 시스템 아키텍처

### 2.1 파이프라인 구조

```
Input Video (0.mp4)
    ↓
[SAM Inference]
    ↓
[Mask Post-processing]
    ├─ Remove arena
    ├─ Extract mouse mask
    └─ Clean noise
    ↓
[Keypoint Estimation]
    ├─ Contour analysis
    ├─ PCA orientation
    ├─ Anatomical constraints
    └─ MAMMAL 22-point mapping
    ↓
[Output Files]
    ├─ masks/0.mp4 (mask video)
    ├─ videos_undist/0.mp4 (copy)
    └─ keypoints2d_undist/result_view_0.pkl
```

### 2.2 모듈 설계

**Module 1: SAM Inference** (`sam_inference.py`)
- SAM model loading and caching
- Batch processing for efficiency
- GPU memory management

**Module 2: Mask Processing** (`mask_processing.py`)
- Arena detection and removal
- Mouse mask extraction
- Temporal consistency filtering
- Noise reduction

**Module 3: Keypoint Estimation** (`keypoint_estimation.py`)
- Contour extraction
- Skeleton fitting
- 22-point MAMMAL keypoint generation
- Confidence score assignment

**Module 4: Visualization** (`visualization.py`)
- Side-by-side comparisons
- Quality metrics overlay
- Debug frame generation

---

## 3. 상세 구현 계획

### 3.1 Mouse Mask Extraction

**Challenge**: SAM detects arena as largest mask

**Solution Strategy**:

```python
def extract_mouse_mask(sam_masks, frame_shape):
    """
    Multi-stage mouse extraction
    """
    # Stage 1: Filter by size
    # Arena: 50-70% of image
    # Mouse: 5-15% of image
    size_filtered = [m for m in sam_masks
                     if 0.05 < mask_area(m)/total_area < 0.20]

    # Stage 2: Shape analysis
    # Arena: high circularity
    # Mouse: irregular shape
    shape_filtered = [m for m in size_filtered
                      if circularity(m) < 0.8]

    # Stage 3: Position filtering
    # Mouse is inside arena (not at edges)
    # Check centroid distance from image center
    position_filtered = [m for m in shape_filtered
                        if is_inside_arena(m, arena_center)]

    # Stage 4: Select best candidate
    if len(position_filtered) > 0:
        # Largest among filtered = mouse
        return max(position_filtered, key=mask_area)
    else:
        # Fallback: return second largest overall
        sorted_masks = sorted(sam_masks, key=mask_area, reverse=True)
        return sorted_masks[1] if len(sorted_masks) > 1 else None
```

**Validation**:
- Visual inspection on 50 sample frames
- Manual annotation of 10 frames for IoU measurement
- Target IoU > 0.85

### 3.2 MAMMAL 22 Keypoint Estimation

**Keypoint Layout**:
```
Head Region (0-5):
  0: Nose tip
  1: Left ear
  2: Right ear
  3: Left eye
  4: Right eye
  5: Head center

Spine (6-13):
  6-13: 8 points along backbone (neck to tail base)

Limbs (14-17):
  14: Left front paw
  15: Right front paw
  16: Left rear paw
  17: Right rear paw

Tail (18-20):
  18: Tail base
  19: Tail mid
  20: Tail tip

Body (21):
  21: Body centroid
```

**Estimation Algorithm**:

```python
def estimate_mammal_keypoints(mouse_mask):
    """
    Advanced geometric keypoint estimation
    """
    # Step 1: Extract contour and skeleton
    contour = get_largest_contour(mouse_mask)
    skeleton = skeletonize(mouse_mask)

    # Step 2: Determine body orientation via PCA
    points = np.argwhere(mouse_mask)
    pca = PCA(n_components=2)
    pca.fit(points)
    major_axis = pca.components_[0]  # Body direction

    # Step 3: Find extrema points
    head_region = find_head_candidate(contour, major_axis)
    tail_region = find_tail_candidate(contour, major_axis)

    # Step 4: Estimate each keypoint group
    keypoints = np.zeros((22, 3))  # (x, y, confidence)

    # Head keypoints (0-5)
    keypoints[0:6] = estimate_head_keypoints(
        head_region, contour, major_axis
    )

    # Spine keypoints (6-13)
    keypoints[6:14] = estimate_spine_keypoints(
        skeleton, head_region, tail_region
    )

    # Limb keypoints (14-17)
    keypoints[14:18] = estimate_limb_keypoints(
        contour, skeleton, major_axis
    )

    # Tail keypoints (18-20)
    keypoints[18:21] = estimate_tail_keypoints(
        tail_region, skeleton
    )

    # Body centroid (21)
    keypoints[21] = estimate_centroid(mouse_mask)

    return keypoints
```

**Confidence Assignment**:
```python
def assign_confidence(keypoint_type, mask_quality):
    """
    Realistic confidence scores for geometric estimation
    """
    base_confidence = {
        'centroid': 0.95,      # Most reliable
        'spine': 0.70,          # Skeleton-based, fairly reliable
        'head_center': 0.75,    # Contour-based, good
        'tail_base': 0.70,      # Contour-based
        'extrema': 0.60,        # Limbs, ears - less reliable
        'tail_tip': 0.50,       # Hardest to estimate
    }

    # Adjust by mask quality
    return base_confidence[keypoint_type] * mask_quality
```

### 3.3 Temporal Consistency

**Problem**: Frame-to-frame jitter in keypoint positions

**Solution**:
```python
class TemporalSmoothing:
    """
    Smooth keypoints across time
    """
    def __init__(self, window_size=5):
        self.window = window_size
        self.history = []

    def smooth(self, keypoints):
        """
        Apply moving average filter
        """
        self.history.append(keypoints)
        if len(self.history) > self.window:
            self.history.pop(0)

        # Weighted average (recent frames weighted more)
        weights = np.linspace(0.5, 1.0, len(self.history))
        weights /= weights.sum()

        smoothed = np.zeros_like(keypoints)
        for w, kpts in zip(weights, self.history):
            smoothed += w * kpts

        return smoothed
```

---

## 4. 구현 단계 (Implementation Phases)

### Phase 1: Core Implementation (Day 1)

**Files to create**:
1. `preprocess_sam_improved.py` - Main script
2. `utils/sam_inference.py` - SAM wrapper
3. `utils/mask_processing.py` - Mask extraction
4. `utils/keypoint_estimation.py` - Keypoint generation

**Tasks**:
- [ ] Set up SAM model loading
- [ ] Implement mouse mask extraction
- [ ] Implement 22-point keypoint estimation
- [ ] Add visualization for debugging

**Deliverable**: Working script that processes single video

### Phase 2: Testing & Validation (Day 1-2)

**Tasks**:
- [ ] Process 50 sample frames
- [ ] Visual quality inspection
- [ ] Compare with ground truth (manual annotation of 10 frames)
- [ ] Measure mask IoU and keypoint error
- [ ] Adjust algorithms based on results

**Deliverable**: Validated preprocessing quality

### Phase 3: Integration & Optimization (Day 2)

**Tasks**:
- [ ] Integrate with Hydra config system
- [ ] Add batch processing
- [ ] Optimize GPU memory usage
- [ ] Add progress tracking and logging
- [ ] Create comparison visualizations

**Deliverable**: Production-ready preprocessing pipeline

### Phase 4: Full Processing (Day 2-3)

**Tasks**:
- [ ] Process first 100 frames as final test
- [ ] If successful, process full 27,000 frames
- [ ] Generate quality report
- [ ] Save all outputs

**Deliverable**: Fully preprocessed shank3 dataset

### Phase 5: Fitting Validation (Day 3)

**Tasks**:
- [ ] Run fitting on 10 frames
- [ ] Compare with old preprocessing results
- [ ] Verify mesh alignment
- [ ] Measure fitting loss improvement
- [ ] Generate before/after comparison

**Deliverable**: Validated fitting improvement

---

## 5. 성능 최적화 전략

### 5.1 Processing Speed

**Current bottleneck**: SAM inference (~2-3 sec/frame)

**Optimization strategies**:

```python
# Strategy 1: Batch processing
def process_batch(frames, batch_size=4):
    """
    Process multiple frames in parallel
    """
    # SAM supports batch inference
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        masks_batch = sam_predictor.generate_batch(batch)
        results.extend(masks_batch)
    return results

# Strategy 2: Resolution reduction
def resize_for_inference(frame, target_size=1024):
    """
    SAM works well at multiple resolutions
    """
    h, w = frame.shape[:2]
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame, scale

# Strategy 3: Frame skipping (as per config)
# Process every 2nd frame, interpolate for skipped frames
```

**Expected improvement**:
- Batch processing: 1.5-2x speedup
- Resolution optimization: 1.5x speedup
- Total: ~3x faster → 5-7 hours for full dataset

### 5.2 Memory Management

```python
def process_with_memory_management(video_path, chunk_size=1000):
    """
    Process in chunks to avoid OOM
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for start_idx in range(0, total_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, total_frames)

        # Process chunk
        chunk_results = process_chunk(cap, start_idx, end_idx)

        # Save intermediate results
        save_chunk_results(chunk_results, start_idx)

        # Clear GPU cache
        torch.cuda.empty_cache()
```

---

## 6. 품질 검증 프로토콜

### 6.1 Mask Quality Metrics

```python
def evaluate_mask_quality(predicted_mask, reference_masks=None):
    """
    Comprehensive mask quality evaluation
    """
    metrics = {}

    # 1. Detection rate
    metrics['detection_rate'] = is_mouse_detected(predicted_mask)

    # 2. Mask size reasonableness
    mask_area = np.sum(predicted_mask)
    total_area = predicted_mask.size
    metrics['coverage_ratio'] = mask_area / total_area
    metrics['size_reasonable'] = 0.05 < metrics['coverage_ratio'] < 0.20

    # 3. Shape metrics
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        metrics['solidity'] = cv2.contourArea(largest) / cv2.contourArea(cv2.convexHull(largest))
        metrics['circularity'] = calculate_circularity(largest)

    # 4. If reference available, compute IoU
    if reference_masks is not None:
        metrics['iou'] = compute_iou(predicted_mask, reference_masks)

    return metrics
```

### 6.2 Keypoint Quality Metrics

```python
def evaluate_keypoint_quality(keypoints, mask, reference_kpts=None):
    """
    Keypoint estimation quality
    """
    metrics = {}

    # 1. Percentage on mouse body
    on_body = np.sum([is_on_mask(kpt, mask) for kpt in keypoints])
    metrics['on_body_ratio'] = on_body / len(keypoints)

    # 2. Confidence distribution
    confidences = keypoints[:, 2]
    metrics['mean_confidence'] = np.mean(confidences)
    metrics['high_conf_ratio'] = np.sum(confidences > 0.7) / len(confidences)

    # 3. Anatomical plausibility
    metrics['skeleton_valid'] = check_skeleton_constraints(keypoints)

    # 4. If reference available, compute RMSE
    if reference_kpts is not None:
        metrics['rmse'] = np.sqrt(np.mean((keypoints[:, :2] - reference_kpts[:, :2])**2))

    return metrics
```

### 6.3 Validation Frames

**Manual annotation target**: 10 representative frames
- Frame 0 (start)
- Frame 5000 (early)
- Frame 10000 (middle)
- Frame 15000 (middle-late)
- Frame 20000 (late)
- Frame 25000 (near end)
- + 4 random frames with different mouse positions

---

## 7. 출력 형식 (Output Format)

### 7.1 Directory Structure

```
data/preprocessed_shank3_sam/
├── videos_undist/
│   └── 0.mp4                    # Original video (copy)
├── sam_masks/
│   └── 0.mp4                    # Binary mask video
├── keypoints2d_undist/
│   └── result_view_0.pkl        # (N, 22, 3) numpy array
├── visualizations/
│   ├── sample_000000.png
│   ├── sample_005000.png
│   └── ...
└── quality_report.json
```

### 7.2 Output File Formats

**Keypoints file** (`result_view_0.pkl`):
```python
# Pickle format, same as original MAMMAL
keypoints = np.array([
    # Frame 0
    [[x0, y0, conf0],   # Keypoint 0
     [x1, y1, conf1],   # Keypoint 1
     ...
     [x21, y21, conf21]],  # Keypoint 21
    # Frame 1
    [...],
    ...
])  # Shape: (N_frames, 22, 3)
```

**Mask video** (`0.mp4`):
```python
# Binary mask video (0=background, 255=mouse)
# Same resolution as input video
# Same FPS as input video
# Codec: mp4v
```

**Quality report** (`quality_report.json`):
```json
{
  "dataset": "shank3",
  "preprocessing_method": "SAM_improved",
  "processed_frames": 27000,
  "processing_time_hours": 6.5,
  "mask_quality": {
    "detection_rate": 0.97,
    "mean_coverage": 0.12,
    "mean_iou": 0.88
  },
  "keypoint_quality": {
    "on_body_ratio": 0.94,
    "mean_confidence": 0.68,
    "high_conf_ratio": 0.55
  },
  "sample_frames": [0, 5000, 10000, 15000, 20000, 25000]
}
```

---

## 8. 에러 처리 및 복구

### 8.1 Common Failure Modes

**Failure 1: No mouse detected**
```python
def handle_no_detection(frame_idx, prev_masks):
    """
    Use previous frame's mask as fallback
    """
    if len(prev_masks) > 0:
        logging.warning(f"Frame {frame_idx}: No mouse detected, using previous mask")
        return prev_masks[-1]
    else:
        logging.error(f"Frame {frame_idx}: No mouse detected and no previous mask")
        return np.zeros(frame_shape, dtype=np.uint8)
```

**Failure 2: Multiple mouse candidates**
```python
def handle_multiple_candidates(candidates, prev_position):
    """
    Choose closest to previous position
    """
    if prev_position is not None:
        distances = [np.linalg.norm(get_centroid(c) - prev_position)
                    for c in candidates]
        return candidates[np.argmin(distances)]
    else:
        # Choose largest
        return max(candidates, key=mask_area)
```

**Failure 3: GPU OOM**
```python
def handle_oom():
    """
    Reduce batch size and retry
    """
    torch.cuda.empty_cache()
    global BATCH_SIZE
    BATCH_SIZE = max(1, BATCH_SIZE // 2)
    logging.warning(f"OOM detected, reducing batch size to {BATCH_SIZE}")
```

### 8.2 Checkpoint and Resume

```python
class CheckpointManager:
    """
    Save progress periodically for resumption
    """
    def __init__(self, checkpoint_dir, interval=1000):
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval

    def save(self, frame_idx, results):
        """Save checkpoint"""
        checkpoint = {
            'frame_idx': frame_idx,
            'results': results,
            'timestamp': time.time()
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_{frame_idx}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_latest(self):
        """Load most recent checkpoint"""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.pkl'))
        if not checkpoints:
            return None
        latest = max(checkpoints, key=os.path.getmtime)
        with open(latest, 'rb') as f:
            return pickle.load(f)
```

---

## 9. 비교 및 검증

### 9.1 Before/After Comparison

```python
def generate_comparison_report(old_dir, new_dir, sample_frames):
    """
    Comprehensive before/after comparison
    """
    report = {
        'old_preprocessing': analyze_preprocessing(old_dir),
        'new_preprocessing': analyze_preprocessing(new_dir),
        'improvements': {}
    }

    # Mask quality
    report['improvements']['mask_detection'] = {
        'before': 0.0,  # 0% detection
        'after': report['new_preprocessing']['mask_quality']['detection_rate'],
        'improvement': '∞'
    }

    # Keypoint quality
    report['improvements']['keypoint_detection'] = {
        'before': 0.0,  # All zeros
        'after': report['new_preprocessing']['keypoint_quality']['on_body_ratio'],
        'improvement': '∞'
    }

    # Generate visual comparisons
    for frame_idx in sample_frames:
        create_side_by_side(
            old_dir, new_dir, frame_idx,
            output_path=f'comparison_frame_{frame_idx}.png'
        )

    return report
```

### 9.2 Fitting Performance Comparison

```python
def compare_fitting_results(old_results_dir, new_results_dir):
    """
    Compare fitting quality
    """
    comparison = {}

    # Load fitting results
    old_params = load_fitting_params(old_results_dir)
    new_params = load_fitting_params(new_results_dir)

    # Compare final losses
    comparison['loss_improvement'] = {
        'old_loss': old_params['final_loss'],
        'new_loss': new_params['final_loss'],
        'reduction': (old_params['final_loss'] - new_params['final_loss']) / old_params['final_loss']
    }

    # Compare convergence
    comparison['convergence'] = {
        'old_iterations': len(old_params['loss_history']),
        'new_iterations': len(new_params['loss_history']),
        'old_converged': check_convergence(old_params['loss_history']),
        'new_converged': check_convergence(new_params['loss_history'])
    }

    # Visual alignment assessment
    comparison['visual_quality'] = {
        'old_alignment': assess_alignment(old_results_dir),
        'new_alignment': assess_alignment(new_results_dir)
    }

    return comparison
```

---

## 10. 타임라인 및 리소스

### 10.1 예상 작업 시간

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Core implementation | 2-3 hours |
| 2 | Testing & validation | 1-2 hours |
| 3 | Integration & optimization | 1-2 hours |
| 4 | Full processing (27K frames) | 6-8 hours |
| 5 | Fitting validation | 1-2 hours |
| **Total** | | **11-17 hours** |

### 10.2 하드웨어 요구사항

**Minimum**:
- GPU: 8GB VRAM
- RAM: 16GB
- Storage: 50GB free space

**Recommended**:
- GPU: 16GB+ VRAM (RTX 3090/4090, A6000)
- RAM: 32GB
- Storage: 100GB SSD

**Current setup**: ✅ Sufficient (CUDA available, SAM tested successfully)

### 10.3 Disk Space Requirements

```
SAM model checkpoint:        2.4 GB
Preprocessed videos:       ~150 MB
SAM mask videos:          ~150 MB
Visualizations:            ~50 MB
Checkpoints (temp):       ~500 MB
Total:                    ~3.2 GB
```

---

## 11. 위험 요소 및 완화 방안

### 11.1 High-Risk Items

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| SAM too slow | High | Medium | Batch processing, resolution reduction |
| Mouse detection fails | High | Low | Manual review, multiple strategies |
| Keypoints still poor | Medium | Medium | Quick pivot to DeepLabCut Phase 2 |
| OOM errors | Medium | Low | Chunked processing, checkpointing |
| Fitting doesn't improve | High | Low | Deep dive into fitting code |

### 11.2 Contingency Plans

**If mask extraction fails** (>10% failure rate):
1. Add manual annotation for difficult frames
2. Implement temporal tracking
3. Use optical flow for interpolation

**If keypoints inadequate** (fitting doesn't converge):
1. Immediately pivot to DeepLabCut Phase 2
2. Use separate conda environment
3. Pre-compute all keypoints offline

**If processing too slow** (>24 hours):
1. Reduce video resolution
2. Process on multiple machines
3. Sample frames (interval=3 or 4)

---

## 12. 후속 작업 (Future Work)

### 12.1 Phase 2: DeepLabCut Integration

**When**: After Phase 1 validation

**Setup**:
```bash
# Create separate environment
conda create -n dlc_env python=3.10
conda activate dlc_env
pip install deeplabcut[modelzoo] tensorflow

# Pre-compute keypoints
python preprocess_deeplabcut.py \
    --input data/preprocessed_shank3_sam/videos_undist/0.mp4 \
    --output data/preprocessed_shank3_sam/keypoints2d_superanimal/

# Use in fitting
python fitter_articulation.py \
    dataset.keypoint_dir=data/preprocessed_shank3_sam/keypoints2d_superanimal/
```

### 12.2 Potential Enhancements

1. **Multi-animal tracking** (if multiple mice)
2. **3D keypoint lifting** (if multi-view available)
3. **Action recognition** (behavior analysis)
4. **Real-time processing** (for live experiments)

---

## 13. 참고 문헌 및 리소스

### 13.1 Papers
- SAM: Kirillov et al., "Segment Anything", ICCV 2023
- SuperAnimal: Ye et al., "SuperAnimal pretrained pose estimation models for behavioral analysis", Nature Communications 2024
- DeepLabCut: Mathis et al., "DeepLabCut: markerless pose estimation of user-defined body parts with deep learning", Nature Neuroscience 2018

### 13.2 Code References
- SAM GitHub: https://github.com/facebookresearch/segment-anything
- DeepLabCut GitHub: https://github.com/DeepLabCut/DeepLabCut
- MAMMAL paper: (reference needed)

### 13.3 Useful Tools
- Labelme: Manual annotation tool
- CVAT: Computer Vision Annotation Tool
- Napari: Multi-dimensional image viewer

---

## 14. 체크리스트

### Pre-implementation Checklist
- [x] SAM installed and tested
- [x] Test results validated (sam_test_results/)
- [x] Existing preprocessing analyzed (preprocessing_debug/)
- [x] Implementation plan documented
- [ ] Hydra config prepared
- [ ] Visualization tools ready
- [ ] Backup created

### Implementation Checklist
- [ ] Core modules implemented
- [ ] Unit tests written
- [ ] Integration tests passed
- [ ] Sample frames processed
- [ ] Quality metrics measured
- [ ] Visualizations generated

### Validation Checklist
- [ ] Mask quality >90% detection
- [ ] Keypoints on mouse body >90%
- [ ] Fitting converges
- [ ] Mesh aligns with video
- [ ] Before/after comparison documented

### Deployment Checklist
- [ ] Full dataset processed
- [ ] Quality report generated
- [ ] Results backed up
- [ ] Documentation updated
- [ ] Next steps identified

---

**Document Status**: 📋 Planning Phase
**Last Updated**: 2025-11-03
**Next Review**: After Phase 1 implementation
