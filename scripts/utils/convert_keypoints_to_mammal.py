"""
Convert manual keypoint annotations (JSON) to MAMMAL format (PKL)

This script converts keypoint annotations from the manual annotation tool
to the MAMMAL-compatible pickle format for mesh fitting.

Input: keypoints.json (per-frame dict format)
Output: result_view_0.pkl (NumPy array, MAMMAL format)

Usage:
    python convert_keypoints_to_mammal.py \
        --input data/annotations/keypoints.json \
        --output data/100-KO-male-56-20200615_cropped/keypoints2d_undist/result_view_0.pkl \
        --num-frames 20
"""
import numpy as np
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional


# Mapping: Manual annotation keypoint names → MAMMAL 22-keypoint indices
KEYPOINT_MAPPING = {
    # Core body keypoints (0-4)
    'nose': 0,
    'neck': 1,
    'spine_mid': 2,
    'hip': 3,
    'tail_base': 4,

    # Ears (5-6)
    'left_ear': 5,
    'right_ear': 6,

    # Additional keypoints (if annotated in the future)
    'left_shoulder': 7,
    'right_shoulder': 8,
    'left_front_paw': 9,
    'right_front_paw': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_hind_paw': 13,
    'right_hind_paw': 14,
    'tail_mid': 15,
    'tail_end': 16,

    # Indices 17-21: Reserved for future use
}


def load_manual_annotations(json_path: Path) -> Dict:
    """
    Load manual annotations from JSON file

    Format:
        {
            "frame_000000": {
                "nose": {"x": 50.0, "y": 30.0, "visibility": 1.0},
                "neck": {"x": 60.0, "y": 40.0, "visibility": 0.5},
                ...
            },
            ...
        }
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def convert_to_mammal_format(
    annotations: Dict,
    num_frames: int,
    num_keypoints: int = 22,
    default_confidence: float = 0.0
) -> np.ndarray:
    """
    Convert manual annotations to MAMMAL format

    Args:
        annotations: Dict of {frame_name: {kp_name: {x, y, visibility}}}
        num_frames: Total number of frames to generate
        num_keypoints: Number of keypoints in MAMMAL format (default: 22)
        default_confidence: Confidence for missing keypoints (default: 0.0)

    Returns:
        NumPy array of shape (num_frames, num_keypoints, 3)
        where [:, :, 0] = x, [:, :, 1] = y, [:, :, 2] = confidence
    """
    # Initialize array with zeros (missing keypoints)
    keypoints_array = np.zeros((num_frames, num_keypoints, 3), dtype=np.float32)

    # Set default confidence for missing keypoints
    keypoints_array[:, :, 2] = default_confidence

    # Fill in annotated keypoints
    for frame_name, frame_kpts in annotations.items():
        # Extract frame index from name (e.g., "frame_000000" -> 0)
        if frame_name.startswith('frame_'):
            frame_idx = int(frame_name.split('_')[-1])
        else:
            print(f"Warning: Cannot parse frame index from '{frame_name}', skipping")
            continue

        if frame_idx >= num_frames:
            print(f"Warning: Frame index {frame_idx} exceeds num_frames {num_frames}, skipping")
            continue

        # Fill in each annotated keypoint
        for kp_name, kp_data in frame_kpts.items():
            if kp_name not in KEYPOINT_MAPPING:
                print(f"Warning: Unknown keypoint '{kp_name}', skipping")
                continue

            kp_idx = KEYPOINT_MAPPING[kp_name]

            x = float(kp_data['x'])
            y = float(kp_data['y'])
            visibility = float(kp_data.get('visibility', 1.0))

            # MAMMAL uses confidence, our manual tool uses visibility
            # visibility: 1.0 = visible, 0.5 = occluded, 0.0 = not visible
            # We map visibility directly to confidence
            confidence = visibility

            keypoints_array[frame_idx, kp_idx, 0] = x
            keypoints_array[frame_idx, kp_idx, 1] = y
            keypoints_array[frame_idx, kp_idx, 2] = confidence

    return keypoints_array


def visualize_conversion(
    keypoints_array: np.ndarray,
    frame_idx: int = 0,
    save_path: Optional[Path] = None
):
    """
    Visualize converted keypoints for debugging

    Args:
        keypoints_array: (num_frames, num_keypoints, 3)
        frame_idx: Frame to visualize
        save_path: Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt

    kpts = keypoints_array[frame_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot keypoints
    for i, (x, y, conf) in enumerate(kpts):
        if conf > 0.0:  # Only plot visible keypoints
            # Color based on confidence
            if conf >= 0.8:
                color = 'green'
            elif conf >= 0.4:
                color = 'orange'
            else:
                color = 'red'

            ax.plot(x, y, 'o', color=color, markersize=10)
            ax.text(x + 2, y + 2, f'{i}', fontsize=8)

    ax.set_title(f'Frame {frame_idx} - Converted Keypoints')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()  # Image coordinates (y down)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_statistics(keypoints_array: np.ndarray, annotations: Dict):
    """Print conversion statistics"""
    num_frames, num_keypoints, _ = keypoints_array.shape

    print("\n" + "="*60)
    print("Conversion Statistics")
    print("="*60)

    # Frame statistics
    print(f"\nFrames:")
    print(f"  Total frames in array: {num_frames}")
    print(f"  Annotated frames: {len(annotations)}")

    # Keypoint statistics
    visible_kpts = keypoints_array[:, :, 2] > 0.0
    num_visible_per_frame = visible_kpts.sum(axis=1)

    print(f"\nKeypoints per frame:")
    print(f"  Min: {num_visible_per_frame.min():.0f}")
    print(f"  Max: {num_visible_per_frame.max():.0f}")
    print(f"  Mean: {num_visible_per_frame.mean():.1f}")

    # Keypoint type statistics
    print(f"\nKeypoint usage (across all frames):")
    for kp_name, kp_idx in sorted(KEYPOINT_MAPPING.items(), key=lambda x: x[1]):
        num_visible = (keypoints_array[:, kp_idx, 2] > 0.0).sum()
        if num_visible > 0:
            avg_conf = keypoints_array[keypoints_array[:, kp_idx, 2] > 0, kp_idx, 2].mean()
            print(f"  [{kp_idx:2d}] {kp_name:20s}: {num_visible:3d} frames (avg conf: {avg_conf:.2f})")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert manual keypoint annotations to MAMMAL format"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSON file with manual annotations'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output PKL file (MAMMAL format)'
    )
    parser.add_argument(
        '--num-frames', '-n',
        type=int,
        required=True,
        help='Total number of frames to generate'
    )
    parser.add_argument(
        '--num-keypoints', '-k',
        type=int,
        default=22,
        help='Number of keypoints in MAMMAL format (default: 22)'
    )
    parser.add_argument(
        '--visualize',
        type=int,
        default=None,
        help='Visualize a specific frame (frame index)'
    )
    parser.add_argument(
        '--viz-output',
        type=str,
        default=None,
        help='Path to save visualization'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("="*60)
    print("MAMMAL Keypoint Converter")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Frames: {args.num_frames}")
    print(f"Keypoints: {args.num_keypoints}")

    # Load annotations
    print("\nLoading annotations...")
    annotations = load_manual_annotations(input_path)
    print(f"Loaded {len(annotations)} annotated frames")

    # Convert to MAMMAL format
    print("\nConverting to MAMMAL format...")
    keypoints_array = convert_to_mammal_format(
        annotations,
        num_frames=args.num_frames,
        num_keypoints=args.num_keypoints
    )

    # Print statistics
    print_statistics(keypoints_array, annotations)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(keypoints_array, f)

    print(f"✅ Conversion complete!")
    print(f"   Output shape: {keypoints_array.shape}")

    # Visualize if requested
    if args.visualize is not None:
        viz_path = Path(args.viz_output) if args.viz_output else None
        visualize_conversion(keypoints_array, args.visualize, viz_path)


if __name__ == "__main__":
    main()
