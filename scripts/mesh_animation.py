#!/usr/bin/env python
"""
Mesh Animation Video Generator

Creates animation videos from fitted mesh sequences (OBJ files).
Supports multiple rendering backends and camera views.

Usage:
    # Basic usage (auto-detect backend)
    python scripts/mesh_animation.py --result_dir results/fitting/experiment_name

    # With specific options
    python scripts/mesh_animation.py \
        --result_dir results/fitting/experiment_name \
        --output_video output.mp4 \
        --fps 30 \
        --resolution 1920 1080 \
        --camera orbit  # orbit, front, side, top

    # Using PyVista backend (interactive preview available)
    python scripts/mesh_animation.py --result_dir ... --backend pyvista --preview
"""

import os
import sys
import glob
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mesh_sequence(obj_dir: str, pattern: str = "step_2_frame_*.obj") -> List[str]:
    """Load OBJ file paths in sorted order."""
    obj_files = sorted(glob.glob(os.path.join(obj_dir, pattern)))
    if not obj_files:
        # Try alternative patterns
        for alt_pattern in ["step_1_frame_*.obj", "frame_*.obj", "*.obj"]:
            obj_files = sorted(glob.glob(os.path.join(obj_dir, alt_pattern)))
            if obj_files:
                break
    return obj_files


def load_params_sequence(params_dir: str, pattern: str = "step_*.pkl") -> List[Dict]:
    """Load parameter files in sorted order."""
    pkl_files = sorted(glob.glob(os.path.join(params_dir, pattern)))
    params_list = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            params_list.append(pickle.load(f))
    return params_list


def create_animation_trimesh(
    obj_files: List[str],
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    camera_type: str = "orbit",
    show_wireframe: bool = False,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> str:
    """
    Create animation video using trimesh + pyrender backend.
    Works in headless environments with EGL.
    """
    import trimesh
    import pyrender
    from pyrender.constants import RenderFlags
    import cv2
    from tqdm import tqdm

    # Set up offscreen rendering
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    width, height = resolution
    frames = []

    # Load first mesh to get bounds
    first_mesh = trimesh.load(obj_files[0])
    center = first_mesh.centroid
    scale = first_mesh.extents.max()

    # Camera distance
    camera_distance = scale * 2.5

    print(f"Rendering {len(obj_files)} frames...")

    for frame_idx, obj_file in enumerate(tqdm(obj_files)):
        # Load mesh
        mesh = trimesh.load(obj_file)

        # Create pyrender mesh
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        # Create scene
        scene = pyrender.Scene(bg_color=np.array(background_color) / 255.0)
        scene.add(mesh_pyrender)

        # Add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))

        # Additional fill light
        fill_light = pyrender.DirectionalLight(color=[0.7, 0.7, 0.8], intensity=1.5)
        fill_pose = np.eye(4)
        fill_pose[:3, :3] = trimesh.transformations.rotation_matrix(
            np.pi/3, [0, 1, 0])[:3, :3]
        scene.add(fill_light, pose=fill_pose)

        # Camera setup based on type
        # Mouse mesh: X-axis is body length, Y is height, Z is width
        # Default "side" view shows the full body profile
        if camera_type == "orbit":
            angle = (frame_idx / len(obj_files)) * 2 * np.pi
            cam_x = center[0] + camera_distance * np.sin(angle)
            cam_y = center[1] + camera_distance * 0.5
            cam_z = center[2] + camera_distance * np.cos(angle)
        elif camera_type == "front":
            # View from +X direction (looking at mouse face)
            cam_x = center[0] + camera_distance
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2]
        elif camera_type == "back":
            # View from -X direction (looking at mouse tail)
            cam_x = center[0] - camera_distance
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2]
        elif camera_type == "side":
            # View from +Z direction (side profile, default recommended)
            cam_x = center[0]
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2] + camera_distance
        elif camera_type == "top":
            # View from +Y direction (bird's eye view)
            cam_x = center[0]
            cam_y = center[1] + camera_distance
            cam_z = center[2] + camera_distance * 0.1
        elif camera_type == "diagonal":
            # 45-degree diagonal view (good for 3D perception)
            cam_x = center[0] + camera_distance * 0.7
            cam_y = center[1] + camera_distance * 0.5
            cam_z = center[2] + camera_distance * 0.7
        else:
            # Default: side view
            cam_x = center[0]
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2] + camera_distance

        # Create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

        # Camera pose (look at center)
        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 1, 0]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = cam_pos

        scene.add(camera, pose=cam_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(width, height)
        flags = RenderFlags.SHADOWS_DIRECTIONAL
        if show_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME

        color, _ = renderer.render(scene, flags=flags)
        renderer.delete()

        frames.append(color)

    # Write video
    print(f"Writing video to {output_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved: {output_path}")

    return output_path


def create_animation_pyvista(
    obj_files: List[str],
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    camera_type: str = "orbit",
    preview: bool = False,
) -> str:
    """
    Create animation video using PyVista backend.
    Better quality but requires display or virtual framebuffer.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista not installed. Run: pip install pyvista")

    from tqdm import tqdm

    # Set up offscreen rendering if no preview
    if not preview:
        pv.OFF_SCREEN = True

    width, height = resolution

    # Load first mesh to get bounds
    first_mesh = pv.read(obj_files[0])
    center = first_mesh.center
    bounds = first_mesh.bounds
    scale = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    camera_distance = scale * 2.5

    # Create plotter
    plotter = pv.Plotter(off_screen=not preview, window_size=[width, height])
    plotter.set_background('white')

    print(f"Rendering {len(obj_files)} frames...")

    # Open movie file
    plotter.open_movie(output_path, framerate=fps)

    for frame_idx, obj_file in enumerate(tqdm(obj_files)):
        plotter.clear()

        # Load and add mesh
        mesh = pv.read(obj_file)
        plotter.add_mesh(mesh, color='lightblue', smooth_shading=True,
                        specular=0.5, specular_power=15)

        # Camera position (same as trimesh backend)
        if camera_type == "orbit":
            angle = (frame_idx / len(obj_files)) * 2 * np.pi
            cam_x = center[0] + camera_distance * np.sin(angle)
            cam_y = center[1] + camera_distance * 0.5
            cam_z = center[2] + camera_distance * np.cos(angle)
        elif camera_type == "front":
            cam_x = center[0] + camera_distance
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2]
        elif camera_type == "back":
            cam_x = center[0] - camera_distance
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2]
        elif camera_type == "side":
            cam_x = center[0]
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2] + camera_distance
        elif camera_type == "top":
            cam_x = center[0]
            cam_y = center[1] + camera_distance
            cam_z = center[2] + camera_distance * 0.1
        elif camera_type == "diagonal":
            cam_x = center[0] + camera_distance * 0.7
            cam_y = center[1] + camera_distance * 0.5
            cam_z = center[2] + camera_distance * 0.7
        else:
            cam_x = center[0]
            cam_y = center[1] + camera_distance * 0.3
            cam_z = center[2] + camera_distance

        plotter.camera_position = [(cam_x, cam_y, cam_z), center, (0, 1, 0)]

        # Add frame info
        plotter.add_text(f"Frame {frame_idx:04d}", position='upper_left', font_size=12)

        # Write frame
        plotter.write_frame()

    plotter.close()
    print(f"Video saved: {output_path}")

    return output_path


def create_side_by_side_video(
    obj_files: List[str],
    render_dir: str,
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 540),
) -> str:
    """
    Create side-by-side video: rendered mesh | original render.
    """
    import trimesh
    import pyrender
    from pyrender.constants import RenderFlags
    import cv2
    from tqdm import tqdm

    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    half_width = resolution[0] // 2
    height = resolution[1]

    # Find original renders
    render_files = sorted(glob.glob(os.path.join(render_dir, "cam_*", "*.png")))
    if not render_files:
        render_files = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    # Load first mesh for camera setup
    first_mesh = trimesh.load(obj_files[0])
    center = first_mesh.centroid
    scale = first_mesh.extents.max()
    camera_distance = scale * 2.5

    print(f"Creating side-by-side video ({len(obj_files)} frames)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    for frame_idx, obj_file in enumerate(tqdm(obj_files)):
        # Left: Rendered mesh
        mesh = trimesh.load(obj_file)
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        scene = pyrender.Scene(bg_color=[1, 1, 1])
        scene.add(mesh_pyrender)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        cam_pos = np.array([center[0], center[1] + camera_distance * 0.3, center[2] + camera_distance])
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = cam_pos
        scene.add(camera, pose=cam_pose)

        renderer = pyrender.OffscreenRenderer(half_width, height)
        left_frame, _ = renderer.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
        renderer.delete()

        # Right: Original render (if available)
        if frame_idx < len(render_files):
            right_frame = cv2.imread(render_files[frame_idx])
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
            right_frame = cv2.resize(right_frame, (half_width, height))
        else:
            right_frame = np.ones((height, half_width, 3), dtype=np.uint8) * 200

        # Combine
        combined = np.hstack([left_frame, right_frame])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        # Add labels
        cv2.putText(combined_bgr, "Fitted Mesh", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(combined_bgr, "Original", (half_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(combined_bgr, f"Frame {frame_idx:04d}", (resolution[0]//2 - 60, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        out.write(combined_bgr)

    out.release()
    print(f"Side-by-side video saved: {output_path}")

    return output_path


def export_params_to_json(
    params_dir: str,
    output_path: str,
) -> str:
    """Export parameters to JSON for external tools (Blender, etc.)."""
    import json
    import torch

    params_files = sorted(glob.glob(os.path.join(params_dir, "step_*.pkl")))

    all_params = []
    for pkl_file in params_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        frame_params = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                frame_params[key] = value.detach().cpu().numpy().tolist()
            else:
                frame_params[key] = value

        all_params.append({
            'file': os.path.basename(pkl_file),
            'params': frame_params
        })

    with open(output_path, 'w') as f:
        json.dump(all_params, f, indent=2)

    print(f"Parameters exported to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create mesh animation video")
    parser.add_argument("--result_dir", type=str, required=True,
                       help="Path to fitting result directory")
    parser.add_argument("--output_video", type=str, default=None,
                       help="Output video path (default: result_dir/animation.mp4)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Video frame rate (default: 30)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720],
                       help="Video resolution (default: 1280 720)")
    parser.add_argument("--camera", type=str, default="side",
                       choices=["side", "front", "back", "top", "diagonal", "orbit"],
                       help="Camera type (default: side - shows full body profile)")
    parser.add_argument("--backend", type=str, default="trimesh",
                       choices=["trimesh", "pyvista"],
                       help="Rendering backend (default: trimesh)")
    parser.add_argument("--preview", action="store_true",
                       help="Show preview window (pyvista only)")
    parser.add_argument("--wireframe", action="store_true",
                       help="Show wireframe overlay")
    parser.add_argument("--side_by_side", action="store_true",
                       help="Create side-by-side comparison with original renders")
    parser.add_argument("--export_params", action="store_true",
                       help="Export parameters to JSON")

    args = parser.parse_args()

    # Validate result directory
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        sys.exit(1)

    obj_dir = result_dir / "obj"
    params_dir = result_dir / "params"
    render_dir = result_dir / "render"

    if not obj_dir.exists():
        print(f"Error: OBJ directory not found: {obj_dir}")
        sys.exit(1)

    # Load mesh files
    obj_files = load_mesh_sequence(str(obj_dir))
    if not obj_files:
        print(f"Error: No OBJ files found in {obj_dir}")
        sys.exit(1)

    print(f"Found {len(obj_files)} mesh files")

    # Set output path
    if args.output_video:
        output_path = args.output_video
    else:
        output_path = str(result_dir / "animation.mp4")

    # Export params if requested
    if args.export_params and params_dir.exists():
        export_params_to_json(str(params_dir), str(result_dir / "params_export.json"))

    # Create animation
    if args.side_by_side and render_dir.exists():
        create_side_by_side_video(
            obj_files=obj_files,
            render_dir=str(render_dir),
            output_path=output_path.replace(".mp4", "_sidebyside.mp4"),
            fps=args.fps,
            resolution=(args.resolution[0], args.resolution[1] // 2),
        )

    if args.backend == "pyvista":
        create_animation_pyvista(
            obj_files=obj_files,
            output_path=output_path,
            fps=args.fps,
            resolution=tuple(args.resolution),
            camera_type=args.camera,
            preview=args.preview,
        )
    else:
        create_animation_trimesh(
            obj_files=obj_files,
            output_path=output_path,
            fps=args.fps,
            resolution=tuple(args.resolution),
            camera_type=args.camera,
            show_wireframe=args.wireframe,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
