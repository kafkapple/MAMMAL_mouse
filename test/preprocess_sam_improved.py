"""
Improved Preprocessing with SAM and Enhanced Keypoint Estimation

This script replaces the failing OpenCV-based preprocessing with:
- SAM (Segment Anything Model) for high-quality mouse masks
- Improved geometric keypoint estimation based on accurate masks

Usage:
    python preprocess_sam_improved.py --video data/shank3/0.mp4 --output data/preprocessed_shank3_sam --num_frames 50
"""
import cv2
import numpy as np
import pickle
import os
import argparse
import logging
from tqdm import tqdm
import sys

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing_utils.sam_inference import SAMInference
from preprocessing_utils.mask_processing import (
    extract_mouse_mask, clean_mask, smooth_mask,
    TemporalMaskFilter
)
from preprocessing_utils.keypoint_estimation import (
    estimate_mammal_keypoints, TemporalKeypointSmoother,
    validate_keypoints, visualize_keypoints_on_frame
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_video(video_path, output_dir, num_frames=None, visualize_interval=100):
    """
    Process video with SAM-based preprocessing

    Args:
        video_path: Path to input video
        output_dir: Output directory
        num_frames: Number of frames to process (None = all)
        visualize_interval: Save visualization every N frames
    """
    logger.info("="*60)
    logger.info("SAM-based Preprocessing")
    logger.info("="*60)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "sam_masks")
    keypoint_dir = os.path.join(output_dir, "keypoints2d_undist")
    vis_dir = os.path.join(output_dir, "visualizations")
    video_out_dir = os.path.join(output_dir, "videos_undist")

    for d in [mask_dir, keypoint_dir, vis_dir, video_out_dir]:
        os.makedirs(d, exist_ok=True)

    # Initialize SAM
    logger.info("Initializing SAM...")
    sam_inference = SAMInference(
        checkpoint_path="checkpoints/sam_vit_h_4b8939.pth",
        model_type="vit_h"
    )

    # Initialize temporal filters
    mask_filter = TemporalMaskFilter(window_size=5, iou_threshold=0.5)
    keypoint_smoother = TemporalKeypointSmoother(window_size=5, alpha=0.6)

    # Open video
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames is not None:
        total_frames = min(total_frames, num_frames)

    logger.info(f"Video info: {frame_width}x{frame_height}, {fps} fps")
    logger.info(f"Processing {total_frames} frames")

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        os.path.join(video_out_dir, "0.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )
    mask_writer = cv2.VideoWriter(
        os.path.join(mask_dir, "0.mp4"),
        fourcc, fps, (frame_width, frame_height), isColor=False
    )

    # Process frames
    all_keypoints = []
    frame_idx = 0

    # Quality statistics
    stats = {
        'masks_detected': 0,
        'masks_failed': 0,
        'mean_mask_area': [],
        'mean_keypoint_conf': [],
    }

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for SAM
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Generate SAM masks
            sam_masks = sam_inference.generate_masks(frame_rgb)

            # Extract mouse mask
            mouse_mask = extract_mouse_mask(
                sam_masks,
                frame_shape=(frame_height, frame_width),
                strategy='multi_stage'
            )

            if mouse_mask is not None and np.sum(mouse_mask) > 0:
                # Clean and smooth mask
                mouse_mask = clean_mask(mouse_mask, min_size=100)
                mouse_mask = smooth_mask(mouse_mask, kernel_size=5)

                # Apply temporal filtering
                mouse_mask = mask_filter.filter(mouse_mask)

                stats['masks_detected'] += 1
                stats['mean_mask_area'].append(np.sum(mouse_mask > 0))
            else:
                logger.warning(f"Frame {frame_idx}: No mouse mask detected")
                stats['masks_failed'] += 1
                # Create empty mask
                mouse_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Estimate keypoints
            keypoints = estimate_mammal_keypoints(mouse_mask, frame_idx=frame_idx)

            # Apply temporal smoothing
            keypoints = keypoint_smoother.smooth(keypoints)

            # Validate keypoints
            keypoints = validate_keypoints(keypoints, (frame_height, frame_width))

            # Track statistics
            stats['mean_keypoint_conf'].append(np.mean(keypoints[:, 2]))

            all_keypoints.append(keypoints)

            # Write outputs
            video_writer.write(frame)
            mask_writer.write(mouse_mask)

            # Visualization
            if frame_idx % visualize_interval == 0:
                vis_frame = visualize_keypoints_on_frame(frame, keypoints)

                # Add mask overlay
                mask_overlay = cv2.cvtColor(mouse_mask, cv2.COLOR_GRAY2BGR)
                mask_overlay[:, :, 1] = mouse_mask  # Green channel
                vis_frame = cv2.addWeighted(vis_frame, 0.7, mask_overlay, 0.3, 0)

                # Save visualization
                vis_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(vis_path, vis_frame)

                logger.info(f"Saved visualization: {vis_path}")

        except Exception as e:
            logger.error(f"Frame {frame_idx} processing failed: {e}")
            # Use zero keypoints as fallback
            keypoints = np.zeros((22, 3), dtype=np.float32)
            all_keypoints.append(keypoints)
            mouse_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            video_writer.write(frame)
            mask_writer.write(mouse_mask)
            stats['masks_failed'] += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    video_writer.release()
    mask_writer.release()

    # Save keypoints
    all_keypoints = np.array(all_keypoints)
    keypoint_path = os.path.join(keypoint_dir, "result_view_0.pkl")
    with open(keypoint_path, 'wb') as f:
        pickle.dump(all_keypoints, f)

    # Calculate and save statistics
    stats['detection_rate'] = stats['masks_detected'] / frame_idx if frame_idx > 0 else 0
    stats['mean_mask_area'] = np.mean(stats['mean_mask_area']) if stats['mean_mask_area'] else 0
    stats['mean_keypoint_conf'] = np.mean(stats['mean_keypoint_conf']) if stats['mean_keypoint_conf'] else 0

    # Save quality report
    import json
    report = {
        'dataset': os.path.basename(os.path.dirname(video_path)),
        'preprocessing_method': 'SAM_improved',
        'processed_frames': frame_idx,
        'video_info': {
            'width': frame_width,
            'height': frame_height,
            'fps': fps,
        },
        'quality_metrics': {
            'detection_rate': float(stats['detection_rate']),
            'masks_detected': stats['masks_detected'],
            'masks_failed': stats['masks_failed'],
            'mean_mask_area': float(stats['mean_mask_area']),
            'mean_keypoint_confidence': float(stats['mean_keypoint_conf']),
        },
        'output_files': {
            'video': os.path.join(video_out_dir, "0.mp4"),
            'masks': os.path.join(mask_dir, "0.mp4"),
            'keypoints': keypoint_path,
            'visualizations': vis_dir,
        }
    }

    report_path = os.path.join(output_dir, "quality_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info("="*60)
    logger.info("Preprocessing Complete!")
    logger.info("="*60)
    logger.info(f"Processed frames: {frame_idx}")
    logger.info(f"Detection rate: {stats['detection_rate']*100:.1f}%")
    logger.info(f"Mean keypoint confidence: {stats['mean_keypoint_conf']:.3f}")
    logger.info(f"Keypoints shape: {all_keypoints.shape}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quality report: {report_path}")
    logger.info("="*60)

    return all_keypoints, report


def main():
    parser = argparse.ArgumentParser(description="SAM-based preprocessing for MAMMAL")
    parser.add_argument('--video', type=str,
                       default='data/preprocessed_shank3/videos_undist/0.mp4',
                       help='Input video path')
    parser.add_argument('--output', type=str,
                       default='data/preprocessed_shank3_sam',
                       help='Output directory')
    parser.add_argument('--num_frames', type=int, default=None,
                       help='Number of frames to process (None = all)')
    parser.add_argument('--visualize_interval', type=int, default=100,
                       help='Save visualization every N frames')

    args = parser.parse_args()

    # Process video
    keypoints, report = process_video(
        args.video,
        args.output,
        num_frames=args.num_frames,
        visualize_interval=args.visualize_interval
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
