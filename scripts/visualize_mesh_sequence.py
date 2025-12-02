#!/usr/bin/env python
"""
Visualize mesh sequence from fitting results.

Usage:
    python scripts/visualize_mesh_sequence.py <results_folder> [--output video.mp4]

Examples:
    # Interactive 3D viewer (matplotlib)
    python scripts/visualize_mesh_sequence.py results/fitting/markerless_mouse_1_nerf_v024_20251202_130053

    # Save as video
    python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh_sequence.mp4

    # Save individual frames
    python scripts/visualize_mesh_sequence.py results/fitting/xxx --output-frames frames/
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from articulation_th import ArticulationTorch


def load_params_from_pkl(pkl_path):
    """Load MAMMAL parameters from pkl file."""
    with open(pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def params_to_tensors(params, device='cuda'):
    """Convert numpy params to torch tensors."""
    # Handle both old format (torch tensors) and new format (numpy arrays)
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=device)

    return {
        'thetas': to_tensor(params['thetas']),
        'bone_lengths': to_tensor(params['bone_lengths']),
        'R': to_tensor(params['R']),
        'T': to_tensor(params['T']),
        's': to_tensor(params['s']),
        'chest_deformer': to_tensor(params['chest_deformer']),
    }


def generate_mesh_from_params(model, params_tensors):
    """Generate mesh vertices from MAMMAL parameters."""
    with torch.no_grad():
        vertices, joints = model(
            params_tensors['thetas'],
            params_tensors['bone_lengths'],
            params_tensors['R'],
            params_tensors['T'],
            params_tensors['s'],
            params_tensors['chest_deformer']
        )
        keypoints_3d = model.forward_keypoints22()

    return {
        'vertices': vertices[0].cpu().numpy(),  # (14522, 3)
        'joints': joints[0].cpu().numpy(),      # (140, 3)
        'keypoints': keypoints_3d[0].cpu().numpy(),  # (22, 3)
        'faces': model.faces_vert_np
    }


def find_pkl_files(results_folder):
    """Find all parameter pkl files in results folder."""
    results_path = Path(results_folder)

    # Try different patterns
    patterns = [
        'params/param*_sil.pkl',  # Multi-view fitting (step2 results)
        'params/param*.pkl',       # Multi-view fitting
        '*_params.pkl',            # Monocular fitting
    ]

    pkl_files = []
    for pattern in patterns:
        files = sorted(results_path.glob(pattern))
        if files:
            pkl_files = files
            break

    return pkl_files


def visualize_mesh_matplotlib(mesh_data, ax=None, show_keypoints=True):
    """Visualize single mesh using matplotlib."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()

    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    keypoints = mesh_data['keypoints']

    # Plot mesh (simplified - show subset of faces)
    step = max(1, len(faces) // 1000)  # Limit number of faces for speed
    mesh_collection = Poly3DCollection(
        vertices[faces[::step]],
        alpha=0.3,
        facecolor='lightblue',
        edgecolor='gray',
        linewidth=0.1
    )
    ax.add_collection3d(mesh_collection)

    # Plot keypoints
    if show_keypoints:
        ax.scatter(
            keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
            c='red', s=50, marker='o', label='Keypoints'
        )

        # Label key points
        key_labels = {0: 'Nose', 21: 'Body', 18: 'Tail'}
        for idx, label in key_labels.items():
            ax.text(keypoints[idx, 0], keypoints[idx, 1], keypoints[idx, 2],
                   label, fontsize=8)

    # Set axis limits
    center = vertices.mean(axis=0)
    max_range = np.abs(vertices - center).max()
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def main():
    parser = argparse.ArgumentParser(description='Visualize mesh sequence from fitting results')
    parser.add_argument('results_folder', type=str, help='Path to results folder')
    parser.add_argument('--output', type=str, default=None, help='Output video path (mp4)')
    parser.add_argument('--output-frames', type=str, default=None, help='Output frames folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode (step through frames)')
    args = parser.parse_args()

    # Find pkl files
    pkl_files = find_pkl_files(args.results_folder)

    if not pkl_files:
        print(f"No parameter files found in {args.results_folder}")
        print("Looking for: params/param*.pkl or *_params.pkl")
        return

    print(f"Found {len(pkl_files)} parameter files")
    for f in pkl_files[:5]:
        print(f"  {f.name}")
    if len(pkl_files) > 5:
        print(f"  ... and {len(pkl_files) - 5} more")

    # Initialize MAMMAL model (once)
    print("\nLoading MAMMAL model...")
    model = ArticulationTorch()
    model.init_params(batch_size=1)
    model.to(args.device)
    print(f"Model loaded: {model.vertexnum} vertices, {model.jointnum} joints")

    # Generate meshes
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if args.output_frames:
        output_frames_path = Path(args.output_frames)
        output_frames_path.mkdir(parents=True, exist_ok=True)

    frames_for_video = []

    for i, pkl_path in enumerate(pkl_files):
        print(f"\rProcessing frame {i+1}/{len(pkl_files)}: {pkl_path.name}", end='')

        # Load and convert parameters
        params = load_params_from_pkl(pkl_path)
        params_tensors = params_to_tensors(params, args.device)

        # Generate mesh
        mesh_data = generate_mesh_from_params(model, params_tensors)

        # Visualize
        ax = visualize_mesh_matplotlib(mesh_data, ax=ax)
        ax.set_title(f'Frame {i+1}/{len(pkl_files)}: {pkl_path.stem}')

        if args.output_frames:
            frame_path = output_frames_path / f'frame_{i:05d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')

        if args.output:
            # Save frame for video
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames_for_video.append(frame)

        if args.interactive:
            plt.pause(0.5)
        elif not args.output and not args.output_frames:
            plt.pause(0.1)

    print("\n")

    # Save video
    if args.output and frames_for_video:
        print(f"Saving video to {args.output}...")
        import cv2
        h, w = frames_for_video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, 10, (w, h))
        for frame in frames_for_video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved: {args.output}")

    if not args.output and not args.output_frames:
        plt.show()

    print("Done!")


if __name__ == '__main__':
    main()
