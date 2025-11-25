"""
Video Processing with SAM-based Cropping and Mesh Fitting
Processes a video file with optional SAM-based mouse region cropping
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json
from tqdm import tqdm
import os

# SAM imports (optional)
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM not available. Install with: pip install segment-anything")

# Add mouse-super-resolution path for SAM annotator
sys.path.insert(0, str(Path.home() / 'dev/mouse-super-resolution'))


def get_largest_mask(masks):
    """Get the largest mask from SAM output (likely the mouse)"""
    if len(masks) == 0:
        return None
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    return masks_sorted[0]


def crop_frame_with_sam(frame, sam_generator, padding=50):
    """
    Crop frame to mouse region using SAM

    Args:
        frame: Input frame (BGR)
        sam_generator: SAM mask generator
        padding: Padding pixels around detected region

    Returns:
        cropped_frame: Cropped frame
        crop_info: Dictionary with crop coordinates and mask
    """
    # Convert to RGB for SAM
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = sam_generator.generate(frame_rgb)

    if len(masks) == 0:
        print("Warning: No masks detected, using full frame")
        return frame, None

    # Get largest mask (mouse)
    largest_mask = get_largest_mask(masks)
    mask = largest_mask['segmentation']
    bbox = largest_mask['bbox']  # [x, y, w, h]

    # Add padding
    x, y, w, h = bbox
    x_min = max(0, x - padding)
    y_min = max(0, y - padding)
    x_max = min(frame.shape[1], x + w + padding)
    y_max = min(frame.shape[0], y + h + padding)

    # Crop frame
    cropped_frame = frame[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max].copy()

    crop_info = {
        'bbox': bbox,
        'crop_coords': [x_min, y_min, x_max, y_max],
        'mask': cropped_mask,
        'original_shape': frame.shape[:2],
        'cropped_shape': cropped_frame.shape[:2],
        'mask_area': int(largest_mask['area']),
        'confidence': float(largest_mask.get('predicted_iou', 0.0))
    }

    return cropped_frame, crop_info


def extract_frames(video_path, output_dir, num_frames=10, use_sam=False,
                   sam_checkpoint=None, crop_padding=50):
    """
    Extract frames from video with optional SAM-based cropping

    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        num_frames: Number of frames to extract
        use_sam: Whether to use SAM for cropping
        sam_checkpoint: Path to SAM checkpoint
        crop_padding: Padding around detected mouse region

    Returns:
        frame_info: List of dictionaries with frame information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")

    # Initialize SAM if needed
    sam_generator = None
    if use_sam:
        if not SAM_AVAILABLE:
            raise ImportError("SAM not available. Install segment-anything first.")

        if sam_checkpoint is None:
            sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"

        print(f"\nInitializing SAM...")
        print(f"  Checkpoint: {sam_checkpoint}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=device)

        sam_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        print("SAM loaded successfully!")

    # Select frame indices
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        # Evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    print(f"\nExtracting {len(frame_indices)} frames...")

    frame_info = []

    for idx, frame_num in enumerate(tqdm(frame_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Failed to read frame {frame_num}")
            continue

        info = {
            'frame_num': int(frame_num),
            'timestamp': frame_num / fps,
            'original_shape': frame.shape[:2]
        }

        # Process frame
        if use_sam and sam_generator is not None:
            # Crop with SAM
            cropped_frame, crop_info = crop_frame_with_sam(
                frame, sam_generator, padding=crop_padding
            )

            # Save both original and cropped
            original_path = output_dir / f"frame_{idx:04d}_original.png"
            cropped_path = output_dir / f"frame_{idx:04d}_cropped.png"
            mask_path = output_dir / f"frame_{idx:04d}_mask.png"

            cv2.imwrite(str(original_path), frame)
            cv2.imwrite(str(cropped_path), cropped_frame)

            if crop_info is not None:
                cv2.imwrite(str(mask_path), crop_info['mask'].astype(np.uint8) * 255)
                info.update({
                    'crop_info': {
                        'bbox': crop_info['bbox'],
                        'crop_coords': crop_info['crop_coords'],
                        'cropped_shape': crop_info['cropped_shape'],
                        'mask_area': crop_info['mask_area'],
                        'confidence': crop_info['confidence']
                    },
                    'cropped_path': str(cropped_path),
                    'mask_path': str(mask_path)
                })

            info['original_path'] = str(original_path)

        else:
            # Save original frame only
            frame_path = output_dir / f"frame_{idx:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            info['frame_path'] = str(frame_path)

        frame_info.append(info)

    cap.release()

    # Save metadata
    metadata = {
        'video_path': str(video_path),
        'total_frames': total_frames,
        'fps': fps,
        'resolution': [width, height],
        'extracted_frames': len(frame_info),
        'use_sam': use_sam,
        'crop_padding': crop_padding if use_sam else None,
        'frames': frame_info
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtracted {len(frame_info)} frames to {output_dir}")
    print(f"Metadata saved to {metadata_path}")

    return frame_info


def visualize_extraction_results(output_dir, max_frames=6):
    """Visualize extracted frames"""
    output_dir = Path(output_dir)

    # Load metadata
    metadata_path = output_dir / 'metadata.json'
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {output_dir}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    use_sam = metadata.get('use_sam', False)
    frame_info = metadata['frames'][:max_frames]

    if use_sam:
        # Show original, cropped, and mask
        fig, axes = plt.subplots(len(frame_info), 3, figsize=(15, 5 * len(frame_info)))
        if len(frame_info) == 1:
            axes = axes.reshape(1, -1)

        for i, info in enumerate(frame_info):
            # Original
            original = cv2.imread(info['original_path'])
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(original_rgb)
            axes[i, 0].set_title(f"Frame {info['frame_num']} - Original")
            axes[i, 0].axis('off')

            # Cropped
            if 'cropped_path' in info:
                cropped = cv2.imread(info['cropped_path'])
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                axes[i, 1].imshow(cropped_rgb)
                crop_info = info['crop_info']
                title = f"Cropped ({crop_info['cropped_shape'][1]}x{crop_info['cropped_shape'][0]})"
                axes[i, 1].set_title(title)
                axes[i, 1].axis('off')

                # Mask
                mask = cv2.imread(info['mask_path'], cv2.IMREAD_GRAYSCALE)
                axes[i, 2].imshow(mask, cmap='gray')
                axes[i, 2].set_title(f"Mask (conf: {crop_info['confidence']:.3f})")
                axes[i, 2].axis('off')
    else:
        # Show original frames only
        cols = min(3, len(frame_info))
        rows = (len(frame_info) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).flatten()

        for i, info in enumerate(frame_info):
            frame = cv2.imread(info['frame_path'])
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame_rgb)
            axes[i].set_title(f"Frame {info['frame_num']}")
            axes[i].axis('off')

        # Hide extra subplots
        for i in range(len(frame_info), len(axes)):
            axes[i].axis('off')

    plt.tight_layout()
    viz_path = output_dir / 'extraction_results.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {viz_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video with optional SAM cropping")
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output-dir', type=str, default='extracted_frames',
                       help='Output directory for frames')
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Number of frames to extract')
    parser.add_argument('--use-sam', action='store_true',
                       help='Use SAM to crop frames to mouse region')
    parser.add_argument('--sam-checkpoint', type=str, default='checkpoints/sam_vit_h_4b8939.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--crop-padding', type=int, default=50,
                       help='Padding around detected mouse region (pixels)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of extraction results')

    args = parser.parse_args()

    print("="*80)
    print("Video Frame Extraction with Optional SAM Cropping")
    print("="*80)
    print(f"Video: {args.video_path}")
    print(f"Output: {args.output_dir}")
    print(f"Use SAM: {args.use_sam}")
    print("="*80)

    # Extract frames
    frame_info = extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        use_sam=args.use_sam,
        sam_checkpoint=args.sam_checkpoint if args.use_sam else None,
        crop_padding=args.crop_padding
    )

    # Visualize if requested
    if args.visualize:
        print("\nCreating visualization...")
        visualize_extraction_results(args.output_dir)

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
