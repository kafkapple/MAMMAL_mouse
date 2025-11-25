"""
Extract keypoints from unified annotator output

Converts unified annotator JSON format to keypoint-only JSON format
for compatibility with convert_keypoints_to_mammal.py

Usage:
    python extract_unified_keypoints.py \
        --input data/annotations \
        --output keypoints.json
"""
import json
from pathlib import Path
import argparse


def extract_keypoints_from_unified(annotations_dir: Path, output_file: Path):
    """
    Extract keypoints from unified annotator output

    Args:
        annotations_dir: Directory containing unified annotations
        output_file: Output JSON file for keypoints only
    """
    annotations_dir = Path(annotations_dir)
    annotation_files = sorted(annotations_dir.glob('*_annotation.json'))

    print(f"Found {len(annotation_files)} annotation files")

    keypoints_data = {}

    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)

        # Extract frame name
        frame_name = annotation_file.stem.replace('_annotation', '')

        # Extract keypoints if present
        if 'keypoints' in annotation:
            keypoints_data[frame_name] = annotation['keypoints']
            num_kps = len([k for k, v in annotation['keypoints'].items()
                          if v.get('visibility', 0) > 0])
            print(f"  {frame_name}: {num_kps} keypoints")

    # Save to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(keypoints_data, f, indent=2)

    print(f"\n✅ Extracted keypoints from {len(keypoints_data)} frames")
    print(f"   Saved to: {output_file}")

    return keypoints_data


def print_statistics(keypoints_data: dict):
    """Print extraction statistics"""
    print("\n" + "="*60)
    print("Keypoint Statistics")
    print("="*60)

    total_frames = len(keypoints_data)
    print(f"\nAnnotated frames: {total_frames}")

    if total_frames == 0:
        return

    # Count keypoints per frame
    kp_counts = []
    for frame_name, kps in keypoints_data.items():
        visible_kps = [k for k, v in kps.items() if v.get('visibility', 0) > 0]
        kp_counts.append(len(visible_kps))

    print(f"\nKeypoints per frame:")
    print(f"  Min: {min(kp_counts)}")
    print(f"  Max: {max(kp_counts)}")
    print(f"  Mean: {sum(kp_counts) / len(kp_counts):.1f}")

    # Keypoint usage
    keypoint_usage = {}
    for frame_name, kps in keypoints_data.items():
        for kp_name, kp_data in kps.items():
            if kp_data.get('visibility', 0) > 0:
                if kp_name not in keypoint_usage:
                    keypoint_usage[kp_name] = 0
                keypoint_usage[kp_name] += 1

    print(f"\nKeypoint usage:")
    for kp_name, count in sorted(keypoint_usage.items()):
        pct = count / total_frames * 100
        print(f"  {kp_name:15s}: {count:3d} frames ({pct:5.1f}%)")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract keypoints from unified annotator output"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing unified annotations'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='keypoints.json',
        help='Output JSON file (default: keypoints.json)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print statistics'
    )

    args = parser.parse_args()

    annotations_dir = Path(args.input)
    output_file = Path(args.output)

    print("="*60)
    print("Extract Keypoints from Unified Annotator")
    print("="*60)
    print(f"Input:  {annotations_dir}")
    print(f"Output: {output_file}")
    print()

    # Extract keypoints
    keypoints_data = extract_keypoints_from_unified(annotations_dir, output_file)

    # Print statistics
    if args.stats or len(keypoints_data) > 0:
        print_statistics(keypoints_data)

    print("\n✅ Done!")
    print("\nNext step:")
    print(f"  python convert_keypoints_to_mammal.py \\")
    print(f"    --input {output_file} \\")
    print(f"    --output data/.../keypoints2d_undist/result_view_0.pkl \\")
    print(f"    --num-frames <N>")


if __name__ == "__main__":
    main()
