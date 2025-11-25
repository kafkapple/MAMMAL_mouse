"""
Visualize extracted video frames
Quick preview of extracted frames before annotation
"""
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import numpy as np


def visualize_frames(frames_dir, max_frames=12):
    """
    Create visualization grid of extracted frames

    Args:
        frames_dir: Directory containing extracted frames
        max_frames: Maximum number of frames to show
    """
    frames_dir = Path(frames_dir)

    # Load metadata
    metadata_path = frames_dir / 'extraction_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Video: {metadata['video_path']}")
        print(f"Total frames extracted: {metadata['extraction_info']['num_frames_extracted']}")
    else:
        metadata = None
        print(f"No metadata found, showing first {max_frames} frames")

    # Find frame files
    frame_files = sorted(frames_dir.glob('frame_*.png'))[:max_frames]

    if len(frame_files) == 0:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Visualizing {len(frame_files)} frames...")

    # Create grid
    cols = 4
    rows = (len(frame_files) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()

    for i, frame_file in enumerate(frame_files):
        # Load frame
        frame = cv2.imread(str(frame_file))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get metadata for this frame if available
        if metadata:
            frame_info = next((f for f in metadata['frames']
                             if f['filename'] == frame_file.name), None)
            if frame_info:
                title = f"Frame {frame_info['extracted_idx']}\n"
                title += f"Original: {frame_info['original_frame_idx']}\n"
                title += f"Time: {frame_info['timestamp']:.1f}s"
            else:
                title = frame_file.name
        else:
            title = frame_file.name

        # Plot
        axes[i].imshow(frame_rgb)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(frame_files), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save
    output_path = frames_dir / 'frames_preview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize extracted frames")
    parser.add_argument('frames_dir', type=str,
                       help='Directory containing extracted frames')
    parser.add_argument('--max-frames', type=int, default=12,
                       help='Maximum number of frames to visualize')

    args = parser.parse_args()

    print("="*80)
    print("Frame Visualization")
    print("="*80)

    visualize_frames(args.frames_dir, max_frames=args.max_frames)

    print("="*80)


if __name__ == "__main__":
    main()
