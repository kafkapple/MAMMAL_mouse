"""
Extract frames from video for SAM annotation
Simple script to prepare video frames for the existing sam_annotator
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, num_frames=None,
                              frame_indices=None, fps_sample=None):
    """
    Extract frames from video file

    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        num_frames: Number of frames to extract (evenly spaced)
        frame_indices: Specific frame indices to extract (overrides num_frames)
        fps_sample: Sample every N frames based on FPS (e.g., 1.0 = 1 frame/sec)

    Returns:
        metadata: Dictionary with extraction metadata
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path.name}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration:.2f}s")

    # Determine which frames to extract
    if frame_indices is not None:
        # Use specific frame indices
        indices = [idx for idx in frame_indices if 0 <= idx < total_frames]
    elif fps_sample is not None:
        # Sample based on FPS (e.g., 1.0 = 1 frame per second)
        frame_interval = int(fps / fps_sample)
        indices = list(range(0, total_frames, frame_interval))
    elif num_frames is not None:
        # Evenly spaced frames
        if num_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    else:
        # Default: extract all frames
        indices = list(range(total_frames))

    print(f"\nExtracting {len(indices)} frames...")

    # Extract frames
    extracted_frames = []
    for i, frame_idx in enumerate(tqdm(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Failed to read frame {frame_idx}")
            continue

        # Save frame
        frame_filename = f"frame_{i:06d}.png"
        frame_path = output_dir / frame_filename

        cv2.imwrite(str(frame_path), frame)

        # Store metadata
        extracted_frames.append({
            'filename': frame_filename,
            'original_frame_idx': int(frame_idx),
            'timestamp': frame_idx / fps,
            'extracted_idx': i
        })

    cap.release()

    # Save metadata
    metadata = {
        'video_path': str(video_path.absolute()),
        'video_info': {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration
        },
        'extraction_info': {
            'method': 'fps_sample' if fps_sample else ('indices' if frame_indices else 'evenly_spaced'),
            'num_frames_extracted': len(extracted_frames),
            'fps_sample': fps_sample,
            'num_frames_requested': num_frames
        },
        'frames': extracted_frames
    }

    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtracted {len(extracted_frames)} frames to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video for SAM annotation"
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for extracted frames')

    # Frame selection methods (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--num-frames', type=int,
                      help='Number of frames to extract (evenly spaced)')
    group.add_argument('--fps-sample', type=float,
                      help='Sample rate in frames per second (e.g., 1.0 = 1 frame/sec)')
    group.add_argument('--frame-indices', type=int, nargs='+',
                      help='Specific frame indices to extract (space-separated)')

    parser.add_argument('--all', action='store_true',
                       help='Extract all frames')

    args = parser.parse_args()

    print("="*80)
    print("Video Frame Extraction for SAM Annotation")
    print("="*80)

    # Extract frames
    metadata = extract_frames_from_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        frame_indices=args.frame_indices,
        fps_sample=args.fps_sample
    )

    print("\n" + "="*80)
    print("Next Step: Run SAM Annotator")
    print("="*80)
    print("\nOption 1: Using existing sam_annotator (recommended)")
    print("  cd /home/joon/dev/mouse-super-resolution")
    print(f"  python -m sam_annotator \\")
    print(f"      data.input_dir={args.output_dir} \\")
    print(f"      data.output_dir={args.output_dir}/annotations \\")
    print(f"      ui.server_port=7860")
    print("\nOption 2: Simple CLI")
    print(f"  sam-annotator-simple \\")
    print(f"      --input {args.output_dir} \\")
    print(f"      --output {args.output_dir}/annotations \\")
    print(f"      --port 7860")
    print("\nAccess via browser:")
    print("  Local: http://localhost:7860")
    print("  SSH tunnel: ssh -L 7860:localhost:7860 user@server")
    print("="*80)


if __name__ == "__main__":
    main()
