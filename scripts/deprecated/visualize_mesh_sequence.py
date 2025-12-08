#!/usr/bin/env python
"""
Visualize mesh sequence from fitting results.

This script can:
1. Reconstruct 3D mesh from PKL parameters using BodyModel
2. Load mesh directly from OBJ files (standalone)
3. Render from arbitrary viewpoints (azimuth, elevation)
4. Export as video file

Usage:
    python scripts/visualize_mesh_sequence.py <results_folder> [--output video.mp4]

Examples:
    # Interactive 3D viewer (matplotlib)
    python scripts/visualize_mesh_sequence.py results/fitting/markerless_mouse_1_nerf_v024_20251202_130053

    # Save as video from custom viewpoint
    python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh_sequence.mp4 --azimuth 45

    # Use OBJ files directly (no BodyModel needed)
    python scripts/visualize_mesh_sequence.py results/fitting/xxx --use-obj --output mesh.mp4

    # Rotating 360° view
    python scripts/visualize_mesh_sequence.py results/fitting/xxx --rotating --output rotating.mp4

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

    # Try different patterns (new naming convention first)
    patterns = [
        'params/step_2_frame_*.pkl',  # New naming: step_2_frame_000000.pkl
        'params/param*_sil.pkl',       # Old naming: param0_sil.pkl
        'params/param*.pkl',           # Old naming: param0.pkl
        '*_params.pkl',                # Monocular fitting
    ]

    pkl_files = []
    for pattern in patterns:
        files = sorted(results_path.glob(pattern))
        if files:
            pkl_files = files
            break

    return pkl_files


def find_obj_files(results_folder):
    """Find all OBJ mesh files in results folder."""
    results_path = Path(results_folder)

    # Try different patterns
    patterns = [
        'obj/step_2_frame_*.obj',  # New naming
        'obj/mesh_*.obj',          # Old naming
        '*.obj',
    ]

    obj_files = []
    for pattern in patterns:
        files = sorted(results_path.glob(pattern))
        if files:
            obj_files = files
            break

    return obj_files


def load_mesh_from_obj(obj_path):
    """Load mesh from OBJ file (standalone, no BodyModel needed)."""
    vertices = []
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # OBJ faces are 1-indexed
                face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                faces.append(face)

    return np.array(vertices), np.array(faces)


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


def render_mesh_pyrender(vertices, faces, azimuth=0, elevation=30, distance=400,
                         img_size=(800, 800), color=(0.8, 0.6, 0.4)):
    """Render mesh using pyrender from specified viewpoint."""
    try:
        import pyrender
        import trimesh
        from pyrender.constants import RenderFlags
    except ImportError:
        return None

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                           vertex_colors=np.tile(color, (len(vertices), 1)))

    # Create scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # Add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light)

    # Camera pose from azimuth/elevation
    center = vertices.mean(axis=0)

    # Convert angles to radians
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)

    # Camera position in spherical coordinates
    cam_x = center[0] + distance * np.cos(el_rad) * np.sin(az_rad)
    cam_y = center[1] + distance * np.sin(el_rad)
    cam_z = center[2] + distance * np.cos(el_rad) * np.cos(az_rad)
    cam_pos = np.array([cam_x, cam_y, cam_z])

    # Look-at matrix
    forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 1, 0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = cam_pos

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(camera, pose=camera_pose)

    # Render
    try:
        renderer = pyrender.OffscreenRenderer(img_size[0], img_size[1])
        color_img, _ = renderer.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
        renderer.delete()
        return color_img
    except Exception as e:
        print(f"Pyrender error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Visualize mesh sequence from fitting results')
    parser.add_argument('results_folder', type=str, help='Path to results folder')
    parser.add_argument('--output', type=str, default=None, help='Output video path (mp4)')
    parser.add_argument('--output-frames', type=str, default=None, help='Output frames folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode (step through frames)')
    parser.add_argument('--use-obj', action='store_true', help='Use OBJ files instead of PKL (no BodyModel needed)')
    parser.add_argument('--azimuth', type=float, default=45, help='Camera azimuth angle (degrees)')
    parser.add_argument('--elevation', type=float, default=30, help='Camera elevation angle (degrees)')
    parser.add_argument('--distance', type=float, default=400, help='Camera distance from mesh center')
    parser.add_argument('--rotating', action='store_true', help='Create 360° rotating view')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS')
    parser.add_argument('--use-pyrender', action='store_true', help='Use pyrender instead of matplotlib')
    args = parser.parse_args()

    # Find mesh files
    if args.use_obj:
        mesh_files = find_obj_files(args.results_folder)
        file_type = "OBJ"
    else:
        mesh_files = find_pkl_files(args.results_folder)
        file_type = "PKL"

    if not mesh_files:
        print(f"No {file_type} files found in {args.results_folder}")
        if args.use_obj:
            print("Looking for: obj/step_2_frame_*.obj or obj/mesh_*.obj")
        else:
            print("Looking for: params/step_2_frame_*.pkl or params/param*.pkl")
        return

    print(f"Found {len(mesh_files)} {file_type} files")
    for f in mesh_files[:5]:
        print(f"  {f.name}")
    if len(mesh_files) > 5:
        print(f"  ... and {len(mesh_files) - 5} more")

    # Initialize BodyModel only if using PKL files
    model = None
    if not args.use_obj:
        print("\nLoading BodyModel...")
        from bodymodel_th import BodyModelTorch
        model = BodyModelTorch(device=args.device)
        print(f"Model loaded: {model.vertexnum} vertices")

    # Setup matplotlib if needed
    if not args.use_pyrender:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

    if args.output_frames:
        output_frames_path = Path(args.output_frames)
        output_frames_path.mkdir(parents=True, exist_ok=True)

    frames_for_video = []
    total_frames = len(mesh_files)

    for i, mesh_file in enumerate(mesh_files):
        print(f"\rProcessing frame {i+1}/{total_frames}: {mesh_file.name}", end='')

        # Load mesh
        if args.use_obj:
            vertices, faces = load_mesh_from_obj(mesh_file)
            keypoints = None
        else:
            params = load_params_from_pkl(mesh_file)
            params_tensors = params_to_tensors(params, args.device)
            mesh_data = generate_mesh_from_params(model, params_tensors)
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            keypoints = mesh_data['keypoints']

        # Calculate azimuth for rotating view
        if args.rotating:
            azimuth = (args.azimuth + i * (360 / total_frames)) % 360
        else:
            azimuth = args.azimuth

        # Render
        if args.use_pyrender:
            frame = render_mesh_pyrender(vertices, faces,
                                         azimuth=azimuth,
                                         elevation=args.elevation,
                                         distance=args.distance)
            if frame is None:
                print("\nPyrender failed, falling back to matplotlib")
                args.use_pyrender = False
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

        if not args.use_pyrender:
            mesh_data_vis = {'vertices': vertices, 'faces': faces,
                            'keypoints': keypoints if keypoints is not None else np.zeros((22, 3))}
            ax = visualize_mesh_matplotlib(mesh_data_vis, ax=ax, show_keypoints=keypoints is not None)
            ax.view_init(elev=args.elevation, azim=azimuth)
            ax.set_title(f'Frame {i+1}/{total_frames}: {mesh_file.stem}')

            # Save frame for video
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Add frame info text
        import cv2
        frame_with_info = frame.copy()
        text = f"Frame {i:06d}/{total_frames}"
        cv2.putText(frame_with_info, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 0), 2)

        frames_for_video.append(frame_with_info)

        if args.output_frames:
            frame_path = output_frames_path / f'frame_{i:06d}.png'
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_with_info, cv2.COLOR_RGB2BGR))

        if args.interactive and not args.use_pyrender:
            plt.pause(0.3)

    print("\n")

    # Save video
    if args.output and frames_for_video:
        print(f"Saving video to {args.output}...")
        import cv2
        h, w = frames_for_video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for frame in frames_for_video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved: {args.output} ({total_frames} frames, {args.fps} fps)")

    if not args.output and not args.output_frames and not args.use_pyrender:
        plt.show()

    print("Done!")


if __name__ == '__main__':
    main()
