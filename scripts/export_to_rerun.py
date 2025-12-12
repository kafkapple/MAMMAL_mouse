#!/usr/bin/env python3
"""
Export mesh sequence to Rerun RRD format for interactive visualization.

Usage:
    # Single experiment
    python scripts/export_to_rerun.py \
        --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_* \
        --texture wandb_sweep_results/run_autumn-sweep-57/texture_final.png \
        --output exports/mouse_sequence.rrd

    # With frame range
    python scripts/export_to_rerun.py \
        --result_dir results/fitting/xxx \
        --texture texture.png \
        --start_frame 0 --end_frame 50 \
        --output exports/mouse_sequence.rrd

    # View result
    rerun exports/mouse_sequence.rrd
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.rerun_exporter import RerunExporter, MOUSE_BONES, get_keypoint_colors


def load_obj_vertices(obj_path: str) -> np.ndarray:
    """Load vertices from OBJ file."""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)


def load_faces(model_dir: str = "mouse_model/mouse_txt") -> np.ndarray:
    """Load face indices from model definition."""
    faces_path = os.path.join(model_dir, "faces_vert.txt")
    return np.loadtxt(faces_path, dtype=np.int32)


def load_texture_colors(texture_path: str, uv_coords: np.ndarray) -> np.ndarray:
    """
    Sample vertex colors from UV texture map.

    Args:
        texture_path: Path to texture PNG
        uv_coords: (N, 2) UV coordinates

    Returns:
        (N, 3) RGB colors in uint8
    """
    import cv2

    texture = cv2.imread(texture_path)
    if texture is None:
        print(f"Warning: Could not load texture {texture_path}")
        return np.full((len(uv_coords), 3), 128, dtype=np.uint8)

    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    h, w = texture.shape[:2]

    # UV to pixel coordinates
    u = np.clip(uv_coords[:, 0], 0, 1) * (w - 1)
    v = np.clip(1 - uv_coords[:, 1], 0, 1) * (h - 1)  # Flip V

    # Sample colors (nearest neighbor)
    px = np.round(u).astype(np.int32)
    py = np.round(v).astype(np.int32)

    colors = texture[py, px]
    return colors.astype(np.uint8)


def load_params(pkl_path: str) -> dict:
    """Load fitting parameters from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Export mesh sequence to Rerun RRD format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python scripts/export_to_rerun.py \\
      --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \\
      --texture wandb_sweep_results/run_autumn-sweep-57/texture_final.png \\
      --output exports/mouse_sequence.rrd

  # Export specific frame range
  python scripts/export_to_rerun.py \\
      --result_dir results/fitting/xxx \\
      --texture texture.png \\
      --start_frame 0 --end_frame 20 \\
      --output exports/short_sequence.rrd

  # View in Rerun
  rerun exports/mouse_sequence.rrd
        """
    )

    parser.add_argument('--result_dir', type=str, required=True,
                       help='Fitting result directory')
    parser.add_argument('--texture', type=str, required=True,
                       help='UV texture PNG file')
    parser.add_argument('--output', type=str, default='exports/mouse_sequence.rrd',
                       help='Output RRD file path')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Start frame index')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='End frame index (default: all)')
    parser.add_argument('--model_dir', type=str, default='mouse_model/mouse_txt',
                       help='Model directory with UV definitions')
    parser.add_argument('--include_keypoints', action='store_true',
                       help='Include keypoint visualization')

    args = parser.parse_args()

    # Resolve glob patterns
    result_dirs = glob(args.result_dir)
    if not result_dirs:
        print(f"Error: No matching directories for {args.result_dir}")
        return
    result_dir = result_dirs[0]
    print(f"Using result directory: {result_dir}")

    # Find OBJ files
    obj_dir = os.path.join(result_dir, "obj")
    obj_files = sorted(glob(os.path.join(obj_dir, "step_2_frame_*.obj")))

    if not obj_files:
        print(f"Error: No OBJ files found in {obj_dir}")
        return

    print(f"Found {len(obj_files)} OBJ files")

    # Apply frame range
    if args.end_frame is not None:
        obj_files = obj_files[args.start_frame:args.end_frame]
    else:
        obj_files = obj_files[args.start_frame:]

    print(f"Exporting frames {args.start_frame} to {args.start_frame + len(obj_files) - 1}")

    # Load model data
    print("Loading model data...")
    faces = load_faces(args.model_dir)
    uv_path = os.path.join(args.model_dir, "textures.txt")
    uv_coords = np.loadtxt(uv_path)

    # Load texture colors
    print(f"Loading texture: {args.texture}")
    vertex_colors = load_texture_colors(args.texture, uv_coords)

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Initialize Rerun exporter
    try:
        exporter = RerunExporter(args.output)
        exporter.init()
    except ImportError as e:
        print(f"Error: {e}")
        print("Install rerun-sdk: pip install rerun-sdk")
        return

    # Export each frame
    print("Exporting frames...")
    keypoint_colors = get_keypoint_colors() if args.include_keypoints else None

    for i, obj_path in enumerate(tqdm(obj_files, desc="Exporting")):
        frame_idx = args.start_frame + i

        # Load vertices
        vertices = load_obj_vertices(obj_path)

        # Log mesh
        exporter.log_mesh(frame_idx, vertices, faces, vertex_colors)

        # Optionally load and log keypoints
        if args.include_keypoints:
            # Try to load params for keypoints
            pkl_path = obj_path.replace('/obj/', '/params/').replace('.obj', '.pkl')
            if os.path.exists(pkl_path):
                params = load_params(pkl_path)
                # Extract keypoints from params if available
                # (Implementation depends on param structure)

        # Log frame info
        exporter.log_text(frame_idx, f"**Frame {frame_idx}**\n\nFile: `{os.path.basename(obj_path)}`")

    # Save RRD file
    exporter.save()

    print(f"\n{'='*50}")
    print("Export complete!")
    print(f"{'='*50}")
    print(f"\nOutput: {args.output}")
    print(f"Frames: {len(obj_files)}")
    print(f"\nView with: rerun {args.output}")


if __name__ == '__main__':
    main()
