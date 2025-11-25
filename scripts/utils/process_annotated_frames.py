"""
Process SAM-annotated frames and prepare for mesh fitting
Crops frames based on SAM annotations and creates visualization
"""
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_sam_annotation(annotation_file):
    """Load SAM annotation JSON"""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def crop_frame_from_annotation(frame_path, mask_path, padding=50):
    """
    Crop frame using SAM mask

    Args:
        frame_path: Path to original frame
        mask_path: Path to SAM mask
        padding: Padding around mask bbox

    Returns:
        cropped_frame: Cropped frame
        crop_info: Crop metadata
    """
    # Load frame and mask
    frame = cv2.imread(str(frame_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if frame is None or mask is None:
        return None, None

    # Find mask bounding box
    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) == 0:
        print(f"Warning: Empty mask for {frame_path.name}")
        return None, None

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Add padding
    h, w = frame.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Crop
    cropped_frame = frame[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max].copy()

    crop_info = {
        'original_shape': [h, w],
        'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
        'crop_coords': [int(x_min), int(y_min), int(x_max), int(y_max)],
        'cropped_shape': [int(y_max - y_min), int(x_max - x_min)],
        'mask_area': int(np.sum(mask > 0))
    }

    return (cropped_frame, cropped_mask), crop_info


def process_all_annotations(annotations_dir, output_dir, padding=50,
                           visualize=True):
    """
    Process all SAM annotations in a directory

    Args:
        annotations_dir: Directory containing SAM annotations
        output_dir: Output directory for cropped frames
        padding: Padding around detected region
        visualize: Create visualization

    Returns:
        processing_info: Dictionary with processing results
    """
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all annotation files
    annotation_files = sorted(annotations_dir.glob('frame_*_annotation.json'))

    if len(annotation_files) == 0:
        print(f"No annotation files found in {annotations_dir}")
        return None

    print(f"Found {len(annotation_files)} annotated frames")

    # Process each annotation
    processed_frames = []
    failed_frames = []

    for ann_file in tqdm(annotation_files):
        # Load annotation
        annotation = load_sam_annotation(ann_file)

        # Check if mask exists
        if not annotation.get('has_mask', False):
            failed_frames.append({
                'annotation_file': str(ann_file),
                'reason': 'no_mask'
            })
            continue

        # Get frame and mask paths
        frame_path = Path(annotation['frame'])
        mask_file = ann_file.parent / ann_file.name.replace('_annotation.json', '_mask.png')

        if not frame_path.exists() or not mask_file.exists():
            failed_frames.append({
                'annotation_file': str(ann_file),
                'reason': 'missing_files'
            })
            continue

        # Crop frame
        result = crop_frame_from_annotation(frame_path, mask_file, padding=padding)

        if result[0] is None:
            failed_frames.append({
                'annotation_file': str(ann_file),
                'reason': 'crop_failed'
            })
            continue

        (cropped_frame, cropped_mask), crop_info = result

        # Save cropped frame and mask
        frame_idx = annotation['frame_idx']
        cropped_frame_path = output_dir / f"frame_{frame_idx:06d}_cropped.png"
        cropped_mask_path = output_dir / f"frame_{frame_idx:06d}_mask.png"

        cv2.imwrite(str(cropped_frame_path), cropped_frame)
        cv2.imwrite(str(cropped_mask_path), cropped_mask)

        # Save crop info
        crop_info_path = output_dir / f"frame_{frame_idx:06d}_crop_info.json"
        crop_info['frame_idx'] = frame_idx
        crop_info['original_frame'] = str(frame_path)
        crop_info['cropped_frame'] = str(cropped_frame_path)
        crop_info['cropped_mask'] = str(cropped_mask_path)
        crop_info['annotation'] = annotation

        with open(crop_info_path, 'w') as f:
            json.dump(crop_info, f, indent=2)

        processed_frames.append(crop_info)

    # Create processing summary
    processing_info = {
        'annotations_dir': str(annotations_dir),
        'output_dir': str(output_dir),
        'total_annotations': len(annotation_files),
        'processed': len(processed_frames),
        'failed': len(failed_frames),
        'padding': padding,
        'failed_frames': failed_frames,
        'processed_frames': processed_frames
    }

    # Save summary
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(processing_info, f, indent=2)

    print(f"\nProcessing complete:")
    print(f"  Processed: {len(processed_frames)}")
    print(f"  Failed: {len(failed_frames)}")
    print(f"  Output: {output_dir}")

    # Visualize if requested
    if visualize and len(processed_frames) > 0:
        create_processing_visualization(output_dir, processed_frames[:6])

    return processing_info


def create_processing_visualization(output_dir, frame_info, max_frames=6):
    """Create visualization showing original, mask, and cropped frames"""
    output_dir = Path(output_dir)

    num_frames = min(len(frame_info), max_frames)
    fig, axes = plt.subplots(num_frames, 3, figsize=(15, 5 * num_frames))

    if num_frames == 1:
        axes = axes.reshape(1, -1)

    for i, info in enumerate(frame_info[:num_frames]):
        # Original
        original = cv2.imread(info['original_frame'])
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Draw bbox on original
        x, y, w, h = info['bbox']
        cv2.rectangle(original_rgb, (x, y), (x+w, y+h), (255, 255, 0), 2)

        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title(f"Frame {info['frame_idx']} - Original\n"
                            f"{info['original_shape'][1]}x{info['original_shape'][0]}")
        axes[i, 0].axis('off')

        # Cropped
        cropped = cv2.imread(info['cropped_frame'])
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        axes[i, 1].imshow(cropped_rgb)
        axes[i, 1].set_title(f"Cropped\n"
                            f"{info['cropped_shape'][1]}x{info['cropped_shape'][0]}")
        axes[i, 1].axis('off')

        # Mask
        mask = cv2.imread(info['cropped_mask'], cv2.IMREAD_GRAYSCALE)
        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title(f"Mask\n"
                            f"Area: {info['mask_area']} px")
        axes[i, 2].axis('off')

    plt.tight_layout()
    viz_path = output_dir / 'processing_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {viz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process SAM-annotated frames for mesh fitting"
    )
    parser.add_argument('annotations_dir', type=str,
                       help='Directory containing SAM annotations')
    parser.add_argument('--output-dir', type=str, default='cropped_frames',
                       help='Output directory for cropped frames')
    parser.add_argument('--padding', type=int, default=50,
                       help='Padding around detected region (pixels)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization')

    args = parser.parse_args()

    print("="*80)
    print("Processing SAM-Annotated Frames")
    print("="*80)

    # Process annotations
    info = process_all_annotations(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        padding=args.padding,
        visualize=not args.no_visualize
    )

    if info is None:
        print("No annotations to process.")
        return

    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total annotations: {info['total_annotations']}")
    print(f"Successfully processed: {info['processed']}")
    print(f"Failed: {info['failed']}")

    if info['failed'] > 0:
        print("\nFailed frames:")
        for fail in info['failed_frames']:
            print(f"  {fail['annotation_file']}: {fail['reason']}")

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("\n1. Run mesh fitting on original frames:")
    print(f"   python fitter_articulation.py --frames-dir <frames_dir>")
    print("\n2. Run mesh fitting on cropped frames:")
    print(f"   python fitter_articulation.py --frames-dir {args.output_dir}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
