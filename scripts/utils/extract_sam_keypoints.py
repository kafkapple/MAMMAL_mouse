"""
Extract keypoint annotations from SAM crop_info.json files

This script extracts the manually annotated points from SAM's crop_info.json
and converts them to the manual annotation format for further processing.

Input: data/xxx_cropped/frame_*_crop_info.json
Output: keypoints.json (manual annotation format)
"""
import json
from pathlib import Path
import argparse
from typing import Dict, List


# SAM annotation labels mapping (from video_sam_annotator.py)
# label=1: foreground (mouse body)
# label=0: background
SAM_LABEL_NAMES = {
    1: "foreground",
    0: "background"
}


def extract_keypoints_from_crop_info(crop_info: Dict) -> Dict:
    """
    Extract annotated points from SAM crop_info

    Args:
        crop_info: Dict loaded from frame_*_crop_info.json

    Returns:
        Dict of {keypoint_name: {x, y, visibility}}
        Returns empty dict if no useful keypoints found
    """
    if 'annotation' not in crop_info:
        return {}

    annotation = crop_info['annotation']

    if 'points' not in annotation or 'labels' not in annotation:
        return {}

    points = annotation['points']
    labels = annotation['labels']

    # Crop coordinates (to convert original coordinates to cropped)
    crop_coords = crop_info.get('crop_coords', None)
    if crop_coords is None:
        return {}

    x_min, y_min, x_max, y_max = crop_coords

    # Extract foreground points (label=1) and convert to cropped coordinates
    keypoints = {}
    foreground_points = []

    for i, (point, label) in enumerate(zip(points, labels)):
        x_orig, y_orig = point

        # Convert to cropped coordinates
        x_crop = x_orig - x_min
        y_crop = y_orig - y_min

        if label == 1:  # Foreground point
            foreground_points.append((x_crop, y_crop))

    # If we have foreground points, use heuristics to assign names
    # For now, we'll just use generic names
    for i, (x, y) in enumerate(foreground_points):
        keypoint_name = f"point_{i}"
        keypoints[keypoint_name] = {
            'x': float(x),
            'y': float(y),
            'visibility': 0.5  # Medium confidence (from SAM click)
        }

    return keypoints


def process_crop_info_directory(
    crop_dir: Path,
    output_json: Path,
    max_frames: int = None
) -> Dict:
    """
    Process all crop_info.json files in a directory

    Args:
        crop_dir: Directory containing frame_*_crop_info.json files
        output_json: Path to save output keypoints.json
        max_frames: Maximum number of frames to process (None = all)

    Returns:
        Dict of {frame_name: {keypoint_name: {x, y, visibility}}}
    """
    crop_info_files = sorted(crop_dir.glob('*_crop_info.json'))

    if max_frames:
        crop_info_files = crop_info_files[:max_frames]

    print(f"Found {len(crop_info_files)} crop_info files")

    all_annotations = {}

    for crop_info_file in crop_info_files:
        # Load crop info
        with open(crop_info_file, 'r') as f:
            crop_info = json.load(f)

        # Extract frame name (e.g., "frame_000000")
        # from "frame_000000_crop_info.json"
        frame_name = crop_info_file.stem.replace('_crop_info', '')

        # Extract keypoints
        keypoints = extract_keypoints_from_crop_info(crop_info)

        if keypoints:
            all_annotations[frame_name] = keypoints
            print(f"  {frame_name}: {len(keypoints)} points")

    # Save to JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n✅ Saved {len(all_annotations)} annotated frames to {output_json}")

    return all_annotations


def print_statistics(annotations: Dict):
    """Print extraction statistics"""
    print("\n" + "="*60)
    print("Extraction Statistics")
    print("="*60)

    num_frames = len(annotations)
    print(f"\nAnnotated frames: {num_frames}")

    if num_frames == 0:
        print("No annotations found!")
        return

    # Count keypoints per frame
    num_kpts_per_frame = [len(kpts) for kpts in annotations.values()]

    print(f"\nKeypoints per frame:")
    print(f"  Min: {min(num_kpts_per_frame)}")
    print(f"  Max: {max(num_kpts_per_frame)}")
    print(f"  Mean: {sum(num_kpts_per_frame) / len(num_kpts_per_frame):.1f}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract keypoints from SAM crop_info.json files"
    )
    parser.add_argument(
        'crop_dir',
        type=str,
        help='Directory containing crop_info.json files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='keypoints_from_sam.json',
        help='Output JSON file (default: keypoints_from_sam.json)'
    )
    parser.add_argument(
        '--max-frames', '-n',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )

    args = parser.parse_args()

    crop_dir = Path(args.crop_dir)
    output_json = Path(args.output)

    print("="*60)
    print("SAM Keypoint Extractor")
    print("="*60)
    print(f"Input:  {crop_dir}")
    print(f"Output: {output_json}")

    if not crop_dir.exists():
        print(f"Error: Directory not found: {crop_dir}")
        return

    # Process directory
    annotations = process_crop_info_directory(
        crop_dir,
        output_json,
        max_frames=args.max_frames
    )

    # Print statistics
    print_statistics(annotations)

    print("\n⚠️  Note: SAM clicks are generic points, not semantic keypoints!")
    print("   You should manually annotate keypoints using keypoint_annotator_v2.py")
    print("   or refine these points with proper keypoint names.\n")


if __name__ == "__main__":
    main()
