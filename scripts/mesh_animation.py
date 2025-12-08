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
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf


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
        # Mouse mesh coordinate system:
        #   X-axis: body length (head to tail) - longest
        #   Y-axis: body width (left to right) - shortest
        #   Z-axis: body height (belly to back) - "up" direction
        # So camera up vector should be Z-axis [0, 0, 1]
        if camera_type == "orbit":
            angle = (frame_idx / len(obj_files)) * 2 * np.pi
            cam_x = center[0] + camera_distance * np.sin(angle)
            cam_y = center[1] + camera_distance * np.cos(angle)
            cam_z = center[2] + camera_distance * 0.3
        elif camera_type == "front":
            # View from +X direction (looking at mouse face)
            cam_x = center[0] + camera_distance
            cam_y = center[1]
            cam_z = center[2] + camera_distance * 0.2
        elif camera_type == "back":
            # View from -X direction (looking at mouse tail)
            cam_x = center[0] - camera_distance
            cam_y = center[1]
            cam_z = center[2] + camera_distance * 0.2
        elif camera_type == "side":
            # View from +Y direction (side profile, default recommended)
            cam_x = center[0]
            cam_y = center[1] + camera_distance
            cam_z = center[2] + camera_distance * 0.2
        elif camera_type == "top":
            # View from +Z direction (bird's eye view)
            cam_x = center[0]
            cam_y = center[1] + camera_distance * 0.1
            cam_z = center[2] + camera_distance
        elif camera_type == "diagonal":
            # 45-degree diagonal view (good for 3D perception)
            cam_x = center[0] + camera_distance * 0.5
            cam_y = center[1] + camera_distance * 0.7
            cam_z = center[2] + camera_distance * 0.5
        else:
            # Default: side view
            cam_x = center[0]
            cam_y = center[1] + camera_distance
            cam_z = center[2] + camera_distance * 0.2

        # Create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

        # Camera pose (look at center)
        # Up vector is Z-axis for this mesh coordinate system
        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)

        # Use Z-axis as up (mouse mesh convention)
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
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


def create_animation_grid(
    obj_files: List[str],
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    show_wireframe: bool = False,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> str:
    """
    Create grid animation with multiple camera views (2x2 layout).
    Views: front, side, top, diagonal
    """
    import trimesh
    import pyrender
    from pyrender.constants import RenderFlags
    import cv2
    from tqdm import tqdm

    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    # Grid layout: 2x2
    grid_views = ['front', 'side', 'top', 'diagonal']
    view_labels = ['Front', 'Side', 'Top', 'Diagonal']

    # Each cell size
    cell_width = resolution[0] // 2
    cell_height = resolution[1] // 2

    # Load first mesh to get bounds
    first_mesh = trimesh.load(obj_files[0])
    center = first_mesh.centroid
    scale = first_mesh.extents.max()
    camera_distance = scale * 2.5

    print(f"Rendering {len(obj_files)} frames (2x2 grid: {grid_views})...")

    frames = []

    for frame_idx, obj_file in enumerate(tqdm(obj_files)):
        mesh = trimesh.load(obj_file)

        grid_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        for view_idx, (view_type, view_label) in enumerate(zip(grid_views, view_labels)):
            # Calculate grid position
            row = view_idx // 2
            col = view_idx % 2
            x_offset = col * cell_width
            y_offset = row * cell_height

            # Create scene
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            scene = pyrender.Scene(bg_color=np.array(background_color) / 255.0)
            scene.add(mesh_pyrender)

            # Add lights
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(light, pose=np.eye(4))
            fill_light = pyrender.DirectionalLight(color=[0.7, 0.7, 0.8], intensity=1.5)
            fill_pose = np.eye(4)
            fill_pose[:3, :3] = trimesh.transformations.rotation_matrix(np.pi/3, [0, 1, 0])[:3, :3]
            scene.add(fill_light, pose=fill_pose)

            # Camera position based on view type
            if view_type == "front":
                cam_x = center[0] + camera_distance
                cam_y = center[1]
                cam_z = center[2] + camera_distance * 0.2
            elif view_type == "side":
                cam_x = center[0]
                cam_y = center[1] + camera_distance
                cam_z = center[2] + camera_distance * 0.2
            elif view_type == "top":
                cam_x = center[0]
                cam_y = center[1] + camera_distance * 0.1
                cam_z = center[2] + camera_distance
            elif view_type == "diagonal":
                cam_x = center[0] + camera_distance * 0.5
                cam_y = center[1] + camera_distance * 0.7
                cam_z = center[2] + camera_distance * 0.5
            else:
                cam_x = center[0]
                cam_y = center[1] + camera_distance
                cam_z = center[2] + camera_distance * 0.2

            # Camera setup
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
            cam_pos = np.array([cam_x, cam_y, cam_z])
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)

            world_up = np.array([0, 0, 1])
            right = np.cross(forward, world_up)
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
            renderer = pyrender.OffscreenRenderer(cell_width, cell_height)
            flags = RenderFlags.SHADOWS_DIRECTIONAL
            if show_wireframe:
                flags |= RenderFlags.ALL_WIREFRAME
            color, _ = renderer.render(scene, flags=flags)
            renderer.delete()

            # Add label to cell
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.putText(color_bgr, view_label, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
            color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            # Place in grid
            grid_frame[y_offset:y_offset+cell_height, x_offset:x_offset+cell_width] = color

        # Add frame number at bottom center
        grid_frame_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(grid_frame_bgr, f"Frame {frame_idx:04d}",
                   (resolution[0]//2 - 60, resolution[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        frames.append(grid_frame_bgr)

    # Write video
    print(f"Writing video to {output_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Grid video saved: {output_path}")

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
        # Mouse mesh: X=body length, Y=width, Z=height (up)
        if camera_type == "orbit":
            angle = (frame_idx / len(obj_files)) * 2 * np.pi
            cam_x = center[0] + camera_distance * np.sin(angle)
            cam_y = center[1] + camera_distance * np.cos(angle)
            cam_z = center[2] + camera_distance * 0.3
        elif camera_type == "front":
            cam_x = center[0] + camera_distance
            cam_y = center[1]
            cam_z = center[2] + camera_distance * 0.2
        elif camera_type == "back":
            cam_x = center[0] - camera_distance
            cam_y = center[1]
            cam_z = center[2] + camera_distance * 0.2
        elif camera_type == "side":
            cam_x = center[0]
            cam_y = center[1] + camera_distance
            cam_z = center[2] + camera_distance * 0.2
        elif camera_type == "top":
            cam_x = center[0]
            cam_y = center[1] + camera_distance * 0.1
            cam_z = center[2] + camera_distance
        elif camera_type == "diagonal":
            cam_x = center[0] + camera_distance * 0.5
            cam_y = center[1] + camera_distance * 0.7
            cam_z = center[2] + camera_distance * 0.5
        else:
            cam_x = center[0]
            cam_y = center[1] + camera_distance
            cam_z = center[2] + camera_distance * 0.2

        # Up vector is Z-axis for mouse mesh
        plotter.camera_position = [(cam_x, cam_y, cam_z), center, (0, 0, 1)]

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


# ============================================================================
# Keypoint and Overlay Video Functions
# ============================================================================

# Keypoint labels (22 keypoints) - Based on mouse_22_defs.py
# See docs/keypoint_definitions.md for DANNCE vs MAMMAL comparison
KEYPOINT_LABELS = {
    0: 'L_ear', 1: 'R_ear', 2: 'nose',
    3: 'neck', 4: 'body_mid',  # Note: "Medial Spine" in DANNCE, NOT actual body center
    5: 'tail_root', 6: 'tail_mid', 7: 'tail_end',
    8: 'L_paw', 9: 'L_paw_end', 10: 'L_elbow', 11: 'L_shoulder',
    12: 'R_paw', 13: 'R_paw_end', 14: 'R_elbow', 15: 'R_shoulder',
    16: 'L_foot', 17: 'L_knee', 18: 'L_hip',
    19: 'R_foot', 20: 'R_knee', 21: 'R_hip'
}

# Part-aware color scheme for keypoints (BGR format for OpenCV)
KEYPOINT_COLORS = {
    'head': (0, 255, 255),       # Yellow - indices 0, 1, 2
    'body': (255, 0, 255),       # Magenta - indices 3, 4
    'tail': (0, 165, 255),       # Orange - indices 5, 6, 7
    'L_front': (255, 0, 0),      # Blue - indices 8, 9, 10, 11
    'R_front': (0, 255, 0),      # Green - indices 12, 13, 14, 15
    'L_hind': (255, 255, 0),     # Cyan - indices 16, 17, 18
    'R_hind': (0, 0, 255),       # Red - indices 19, 20, 21
}

# Index to body part mapping
KEYPOINT_PART_MAP = {
    0: 'head', 1: 'head', 2: 'head',
    3: 'body', 4: 'body',
    5: 'tail', 6: 'tail', 7: 'tail',
    8: 'L_front', 9: 'L_front', 10: 'L_front', 11: 'L_front',
    12: 'R_front', 13: 'R_front', 14: 'R_front', 15: 'R_front',
    16: 'L_hind', 17: 'L_hind', 18: 'L_hind',
    19: 'R_hind', 20: 'R_hind', 21: 'R_hind',
}

# Bone connections for skeleton visualization
BONES = [
    [0, 2], [1, 2],
    [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [11, 3],
    [12, 13], [13, 14], [14, 15], [15, 3],
    [16, 17], [17, 18], [18, 5],
    [19, 20], [20, 21], [21, 5]
]


def get_keypoint_color(idx: int) -> Tuple[int, int, int]:
    """Get color for keypoint index based on body part."""
    part = KEYPOINT_PART_MAP.get(idx, 'body')
    return KEYPOINT_COLORS.get(part, (255, 255, 255))


def load_data_loader(result_dir: str):
    """Load data loader from config in result directory."""
    config_path = os.path.join(result_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Import data loader
    from data_seaker_video_new import DataSeakerDet
    data_loader = DataSeakerDet(cfg)

    return data_loader, cfg


def load_body_model():
    """Load the mouse body model for keypoint projection."""
    from articulation_th import ArticulationTorch
    return ArticulationTorch()


def project_keypoints_to_2d(
    keypoints_3d: np.ndarray,
    cam_dict: Dict,
) -> np.ndarray:
    """
    Project 3D keypoints to 2D using camera parameters.

    Args:
        keypoints_3d: (N, 3) array of 3D keypoints
        cam_dict: Camera parameters with R, T, K

    Returns:
        (N, 2) array of 2D keypoints
    """
    T = cam_dict['T'] / 1000
    if len(T.shape) > 1:
        T = T.squeeze()

    data2d = (keypoints_3d @ cam_dict['R'] + T) @ cam_dict['K']
    data2d = data2d[:, 0:2] / data2d[:, 2:]

    return data2d


def draw_keypoints_on_image(
    img: np.ndarray,
    keypoints_2d: np.ndarray,
    color: Tuple[int, int, int] = None,
    radius: int = 7,
    use_part_colors: bool = True,
    draw_bones: bool = True,
    confidence: np.ndarray = None,
    indices_to_draw: List[int] = None,
) -> np.ndarray:
    """
    Draw keypoints on image with optional bone connections.

    Args:
        img: Image to draw on (modified in-place)
        keypoints_2d: (N, 2) array of 2D keypoints
        color: Single color for all keypoints (overrides part colors)
        radius: Circle radius
        use_part_colors: Use part-aware coloring
        draw_bones: Draw bone connections
        confidence: Optional (N,) array of confidence values
        indices_to_draw: Specific indices to draw (None = all)

    Returns:
        Modified image
    """
    import cv2

    if indices_to_draw is None:
        indices_to_draw = list(range(keypoints_2d.shape[0]))

    # Draw bones first (so points appear on top)
    if draw_bones:
        for bone in BONES:
            idx0, idx1 = bone
            if idx0 not in indices_to_draw or idx1 not in indices_to_draw:
                continue
            if idx0 >= keypoints_2d.shape[0] or idx1 >= keypoints_2d.shape[0]:
                continue

            x0, y0 = keypoints_2d[idx0]
            x1, y1 = keypoints_2d[idx1]

            if math.isnan(x0) or math.isnan(y0) or math.isnan(x1) or math.isnan(y1):
                continue
            if (x0 == 0 and y0 == 0) or (x1 == 0 and y1 == 0):
                continue

            # Check confidence
            if confidence is not None:
                if confidence[idx0] < 0.25 or confidence[idx1] < 0.25:
                    continue

            p0 = (int(x0), int(y0))
            p1 = (int(x1), int(y1))

            if color is not None:
                bone_color = color
            else:
                bone_color = get_keypoint_color(idx0)

            cv2.line(img, p0, p1, bone_color, 2)

    # Draw keypoints
    for idx in indices_to_draw:
        if idx >= keypoints_2d.shape[0]:
            continue

        x, y = keypoints_2d[idx]

        if math.isnan(x) or math.isnan(y) or (x == 0 and y == 0):
            continue

        if confidence is not None and confidence[idx] < 0.25:
            continue

        p = (int(x), int(y))

        if color is not None:
            point_color = color
        elif use_part_colors:
            point_color = get_keypoint_color(idx)
        else:
            point_color = (0, 255, 0)

        cv2.circle(img, p, radius, point_color, -1)
        cv2.circle(img, p, radius, (255, 255, 255), 1)  # White outline

    return img


def render_mesh_overlay_on_image(
    img: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_dict: Dict,
    img_size: Tuple[int, int],
) -> np.ndarray:
    """
    Render mesh overlay on image using pyrender.

    Args:
        img: Background image
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        cam_dict: Camera parameters
        img_size: (H, W) image size

    Returns:
        Image with mesh overlay
    """
    import cv2
    import trimesh
    import pyrender
    from pyrender.constants import RenderFlags
    import copy

    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    renderer = pyrender.OffscreenRenderer(viewport_width=img_size[1], viewport_height=img_size[0])

    scene = pyrender.Scene()
    light_node = scene.add(pyrender.PointLight(color=np.ones(3), intensity=0.2))
    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=np.array([0.8, 0.6, 0.4]))))

    K = cam_dict['K'].T if cam_dict['K'].shape[0] == 3 else cam_dict['K']
    R = cam_dict['R'].T if cam_dict['R'].shape[0] == 3 else cam_dict['R']
    T = cam_dict['T'] / 1000

    # Fix T shape
    if T.shape == (1, 3):
        T = T.T
    elif T.shape == (3,):
        T = T.reshape(3, 1)
    elif T.shape == (1, 3, 1):
        T = T.squeeze().reshape(3, 1)
    elif T.shape == (3, 1, 1):
        T = T.squeeze()

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R.T
    camera_pose[:3, 3:4] = np.dot(-R.T, T)
    camera_pose[:, 1:3] = -camera_pose[:, 1:3]

    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    cam_node = scene.add(camera, name='cam', pose=camera_pose)
    light_node._matrix = camera_pose

    color, _ = renderer.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
    renderer.delete()

    color = copy.deepcopy(color)

    # Resize rendered image to match input image size
    if color.shape[:2] != img.shape[:2]:
        color = cv2.resize(color, (img.shape[1], img.shape[0]))

    # Composite: replace white background with original image
    background_mask = (color[:, :, 0] == 255) & (color[:, :, 1] == 255) & (color[:, :, 2] == 255)
    color[background_mask] = img[background_mask]

    return color


def create_keypoint_video(
    result_dir: str,
    output_path: str,
    fps: int = 30,
    keypoint_type: str = "both",  # "gt", "pred", "both"
    draw_bones: bool = True,
) -> str:
    """
    Create video with keypoint overlays on original images.

    Args:
        result_dir: Path to fitting result directory
        output_path: Output video path
        fps: Video frame rate
        keypoint_type: "gt" (ground truth only), "pred" (predicted only), "both"
        draw_bones: Draw bone connections

    Returns:
        Output video path
    """
    import cv2
    import torch
    from tqdm import tqdm
    from utils import pack_images

    # Load data loader and config
    data_loader, cfg = load_data_loader(result_dir)
    body_model = load_body_model()

    # Load params files
    params_dir = os.path.join(result_dir, "params")
    params_files = sorted(glob.glob(os.path.join(params_dir, "step_2_frame_*.pkl")))

    if not params_files:
        raise FileNotFoundError(f"No params files found in {params_dir}")

    # Extract frame numbers from filenames
    frame_numbers = []
    for pf in params_files:
        basename = os.path.basename(pf)
        # step_2_frame_000000.pkl -> 000000
        frame_num = int(basename.replace("step_2_frame_", "").replace(".pkl", ""))
        frame_numbers.append(frame_num)

    print(f"Creating keypoint video: {len(frame_numbers)} frames, type={keypoint_type}")

    # Get sparse indices if applicable
    sparse_indices = getattr(cfg.fitter, 'sparse_keypoint_indices', None)
    if sparse_indices:
        indices_to_draw = list(sparse_indices)
    else:
        indices_to_draw = list(range(22))

    out = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for frame_idx, (params_file, frame_num) in enumerate(tqdm(
        zip(params_files, frame_numbers), total=len(params_files), desc="Rendering keypoints"
    )):
        # Load params
        with open(params_file, 'rb') as f:
            params = pickle.load(f)

        # Fetch image and GT keypoints
        labels = data_loader.fetch(frame_num, with_img=True)
        imgs = labels["imgs"]
        gt_keypoints = labels["label2d"]  # (views, 22, 3)

        # Get predicted keypoints from body model
        for k, v in params.items():
            if not isinstance(v, torch.Tensor):
                params[k] = torch.tensor(v, dtype=torch.float32, device=device)

        V, J = body_model.forward(
            params["thetas"], params["bone_lengths"],
            params["rotation"], params["trans"] / 1000,
            params["scale"] / 1000, params["chest_deformer"]
        )
        pred_keypoints_3d = body_model.forward_keypoints22()[0].detach().cpu().numpy()

        view_images = []
        for view_idx, cam_dict in enumerate(data_loader.cams_dict_out):
            img = imgs[view_idx].copy()

            # Project predicted keypoints to 2D
            pred_2d = project_keypoints_to_2d(pred_keypoints_3d, cam_dict)

            # GT keypoints
            gt_2d = gt_keypoints[view_idx][:, :2]
            gt_conf = gt_keypoints[view_idx][:, 2]

            if keypoint_type in ["gt", "both"]:
                # Draw GT keypoints (red/cross markers)
                img = draw_keypoints_on_image(
                    img, gt_2d,
                    color=(0, 0, 255),  # Red for GT
                    radius=8,
                    draw_bones=draw_bones,
                    confidence=gt_conf,
                    indices_to_draw=indices_to_draw,
                )

            if keypoint_type in ["pred", "both"]:
                # Draw predicted keypoints (green)
                img = draw_keypoints_on_image(
                    img, pred_2d,
                    color=(0, 255, 0),  # Green for predicted
                    radius=6 if keypoint_type == "both" else 7,
                    draw_bones=draw_bones and keypoint_type != "both",  # Don't double draw bones
                    indices_to_draw=indices_to_draw,
                )

            # Add legend
            if keypoint_type == "both":
                cv2.rectangle(img, (5, 5), (180, 50), (40, 40, 40), -1)
                cv2.circle(img, (15, 20), 6, (0, 0, 255), -1)
                cv2.putText(img, "GT", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(img, (15, 40), 6, (0, 255, 0), -1)
                cv2.putText(img, "Predicted", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add view label
            cv2.putText(img, f"View {cfg.data.views_to_use[view_idx]}", (10, img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            view_images.append(img)

        # Pack all views into grid
        output_frame = pack_images(view_images)

        # Add frame info
        cv2.putText(output_frame, f"Frame {frame_num:06d}",
                   (output_frame.shape[1] // 2 - 60, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Initialize video writer on first frame
        if out is None:
            h, w = output_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out.write(output_frame)

    out.release()
    print(f"Keypoint video saved: {output_path}")
    return output_path


def create_full_overlay_video(
    result_dir: str,
    output_path: str,
    fps: int = 30,
    include_mesh: bool = True,
    include_gt_keypoints: bool = True,
    include_pred_keypoints: bool = True,
    draw_bones: bool = True,
) -> str:
    """
    Create video with mesh and keypoint overlays on original images.

    Args:
        result_dir: Path to fitting result directory
        output_path: Output video path
        fps: Video frame rate
        include_mesh: Include mesh overlay
        include_gt_keypoints: Include GT keypoints
        include_pred_keypoints: Include predicted keypoints
        draw_bones: Draw bone connections

    Returns:
        Output video path
    """
    import cv2
    import torch
    from tqdm import tqdm
    from utils import pack_images

    # Load data loader and config
    data_loader, cfg = load_data_loader(result_dir)
    body_model = load_body_model()

    # Load params files
    params_dir = os.path.join(result_dir, "params")
    params_files = sorted(glob.glob(os.path.join(params_dir, "step_2_frame_*.pkl")))

    if not params_files:
        raise FileNotFoundError(f"No params files found in {params_dir}")

    # Extract frame numbers
    frame_numbers = []
    for pf in params_files:
        basename = os.path.basename(pf)
        frame_num = int(basename.replace("step_2_frame_", "").replace(".pkl", ""))
        frame_numbers.append(frame_num)

    overlay_desc = []
    if include_mesh:
        overlay_desc.append("mesh")
    if include_gt_keypoints:
        overlay_desc.append("GT")
    if include_pred_keypoints:
        overlay_desc.append("pred")
    print(f"Creating overlay video: {len(frame_numbers)} frames, overlays=[{', '.join(overlay_desc)}]")

    # Get sparse indices if applicable
    sparse_indices = getattr(cfg.fitter, 'sparse_keypoint_indices', None)
    if sparse_indices:
        indices_to_draw = list(sparse_indices)
    else:
        indices_to_draw = list(range(22))

    out = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = (1024, 1152)  # Default MAMMAL image size (H, W)

    for frame_idx, (params_file, frame_num) in enumerate(tqdm(
        zip(params_files, frame_numbers), total=len(params_files), desc="Rendering overlay"
    )):
        # Load params
        with open(params_file, 'rb') as f:
            params = pickle.load(f)

        # Fetch image and GT keypoints
        labels = data_loader.fetch(frame_num, with_img=True)
        imgs = labels["imgs"]
        gt_keypoints = labels["label2d"]

        # Get mesh vertices and keypoints from body model
        for k, v in params.items():
            if not isinstance(v, torch.Tensor):
                params[k] = torch.tensor(v, dtype=torch.float32, device=device)

        V, J = body_model.forward(
            params["thetas"], params["bone_lengths"],
            params["rotation"], params["trans"] / 1000,
            params["scale"] / 1000, params["chest_deformer"]
        )
        vertices = V[0].detach().cpu().numpy()
        faces = body_model.faces_vert_np
        pred_keypoints_3d = body_model.forward_keypoints22()[0].detach().cpu().numpy()

        view_images = []
        for view_idx, cam_dict in enumerate(data_loader.cams_dict_out):
            img = imgs[view_idx].copy()

            # Render mesh overlay first
            if include_mesh:
                img = render_mesh_overlay_on_image(
                    img, vertices, faces, cam_dict, img_size
                )

            # Project predicted keypoints to 2D
            pred_2d = project_keypoints_to_2d(pred_keypoints_3d, cam_dict)

            # GT keypoints
            gt_2d = gt_keypoints[view_idx][:, :2]
            gt_conf = gt_keypoints[view_idx][:, 2]

            # Draw GT keypoints (red)
            if include_gt_keypoints:
                img = draw_keypoints_on_image(
                    img, gt_2d,
                    color=(0, 0, 255),
                    radius=8,
                    draw_bones=draw_bones,
                    confidence=gt_conf,
                    indices_to_draw=indices_to_draw,
                )

            # Draw predicted keypoints (green)
            if include_pred_keypoints:
                img = draw_keypoints_on_image(
                    img, pred_2d,
                    color=(0, 255, 0),
                    radius=6,
                    draw_bones=draw_bones and not include_gt_keypoints,
                    indices_to_draw=indices_to_draw,
                )

            # Add legend
            legend_y = 20
            cv2.rectangle(img, (5, 5), (180, 70 if include_mesh else 55), (40, 40, 40), -1)
            if include_gt_keypoints:
                cv2.circle(img, (15, legend_y), 6, (0, 0, 255), -1)
                cv2.putText(img, "GT Keypoints", (25, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                legend_y += 18
            if include_pred_keypoints:
                cv2.circle(img, (15, legend_y), 6, (0, 255, 0), -1)
                cv2.putText(img, "Pred Keypoints", (25, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                legend_y += 18
            if include_mesh:
                cv2.rectangle(img, (10, legend_y - 3), (20, legend_y + 8), (102, 153, 204), -1)
                cv2.putText(img, "Mesh", (25, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add view label
            cv2.putText(img, f"View {cfg.data.views_to_use[view_idx]}", (10, img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            view_images.append(img)

        # Pack all views into grid
        output_frame = pack_images(view_images)

        # Add frame info
        cv2.putText(output_frame, f"Frame {frame_num:06d}",
                   (output_frame.shape[1] // 2 - 60, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Initialize video writer on first frame
        if out is None:
            h, w = output_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out.write(output_frame)

    out.release()
    print(f"Overlay video saved: {output_path}")
    return output_path


def create_all_videos(
    result_dir: str,
    fps: int = 30,
    mesh_only: bool = True,
    keypoints_only: bool = True,
    full_overlay: bool = True,
    mesh_camera: str = "grid",
) -> Dict[str, str]:
    """
    Create all video types: mesh-only, keypoints-only, and full overlay.

    Args:
        result_dir: Path to fitting result directory
        fps: Video frame rate
        mesh_only: Create mesh-only video
        keypoints_only: Create keypoints-only video (GT + predicted)
        full_overlay: Create full overlay video (mesh + keypoints)
        mesh_camera: Camera type for mesh-only video

    Returns:
        Dict of video type to output path
    """
    result_dir = Path(result_dir)
    videos_dir = result_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    output_paths = {}

    # Load mesh sequence
    obj_dir = result_dir / "obj"
    obj_files = load_mesh_sequence(str(obj_dir))

    if mesh_only and obj_files:
        print("\n" + "="*50)
        print("Creating mesh-only video...")
        print("="*50)
        mesh_path = str(videos_dir / f"mesh_{mesh_camera}.mp4")
        if mesh_camera == "grid":
            create_animation_grid(obj_files, mesh_path, fps=fps)
        else:
            create_animation_trimesh(obj_files, mesh_path, fps=fps, camera_type=mesh_camera)
        output_paths["mesh_only"] = mesh_path

    if keypoints_only:
        print("\n" + "="*50)
        print("Creating keypoints video (GT + predicted)...")
        print("="*50)
        kp_path = str(videos_dir / "keypoints_comparison.mp4")
        create_keypoint_video(str(result_dir), kp_path, fps=fps, keypoint_type="both")
        output_paths["keypoints"] = kp_path

    if full_overlay:
        print("\n" + "="*50)
        print("Creating full overlay video (mesh + keypoints)...")
        print("="*50)
        overlay_path = str(videos_dir / "full_overlay.mp4")
        create_full_overlay_video(str(result_dir), overlay_path, fps=fps)
        output_paths["full_overlay"] = overlay_path

    return output_paths


# ============================================================================
# HTML Report and Summary Image Generation
# ============================================================================

def create_frame_summary_image(
    result_dir: str,
    frame_nums: List[int] = None,
    max_frames: int = 10,
) -> str:
    """
    Create a summary image showing key frames with all overlays.

    Args:
        result_dir: Path to fitting result directory
        frame_nums: Specific frame numbers to include (None = auto-select)
        max_frames: Maximum number of frames to include

    Returns:
        Path to summary image
    """
    import cv2
    from utils import pack_images

    result_dir = Path(result_dir)
    render_dir = result_dir / "render"

    # Find step_2 render images
    step2_files = sorted(glob.glob(str(render_dir / "step_2_frame_*.png")))

    if not step2_files:
        print("No step_2 render images found")
        return None

    # Select frames to include
    if frame_nums is None:
        # Auto-select evenly spaced frames
        n_files = len(step2_files)
        if n_files <= max_frames:
            selected_files = step2_files
        else:
            indices = np.linspace(0, n_files - 1, max_frames, dtype=int)
            selected_files = [step2_files[i] for i in indices]
    else:
        selected_files = []
        for num in frame_nums[:max_frames]:
            pattern = f"step_2_frame_{num:06d}.png"
            matches = [f for f in step2_files if pattern in f]
            if matches:
                selected_files.append(matches[0])

    if not selected_files:
        return None

    # Load and resize images
    images = []
    target_size = (400, 300)  # Thumbnail size

    for f in selected_files:
        img = cv2.imread(f)
        if img is not None:
            # Resize maintaining aspect ratio
            h, w = img.shape[:2]
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # Pad to target size
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            x_off = (target_size[0] - new_w) // 2
            y_off = (target_size[1] - new_h) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized

            # Add frame number
            basename = os.path.basename(f)
            frame_num = basename.replace("step_2_frame_", "").replace(".png", "")
            cv2.putText(canvas, f"Frame {frame_num}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            images.append(canvas)

    if not images:
        return None

    # Create grid
    summary_img = pack_images(images)

    # Add title
    title_h = 40
    final_img = np.zeros((summary_img.shape[0] + title_h, summary_img.shape[1], 3), dtype=np.uint8)
    final_img[:title_h] = (40, 40, 40)
    final_img[title_h:] = summary_img

    cv2.putText(final_img, "Fitting Results Summary - Step 2 (Final)", (10, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save
    output_path = str(result_dir / "summary_frames.png")
    cv2.imwrite(output_path, final_img)
    print(f"Summary image saved: {output_path}")

    return output_path


def create_html_report(
    result_dir: str,
    include_videos: bool = True,
    include_loss_plots: bool = True,
) -> str:
    """
    Create an HTML report with all visualizations and metrics.

    Args:
        result_dir: Path to fitting result directory
        include_videos: Embed video links in report
        include_loss_plots: Include loss history plots

    Returns:
        Path to HTML report
    """
    import json
    from datetime import datetime

    result_dir = Path(result_dir)

    # Load config
    config_path = result_dir / "config.yaml"
    config_str = ""
    if config_path.exists():
        cfg = OmegaConf.load(str(config_path))
        config_str = OmegaConf.to_yaml(cfg)

    # Load loss history
    loss_history_path = result_dir / "loss_history.json"
    loss_summary = {}
    if loss_history_path.exists():
        with open(loss_history_path) as f:
            loss_history = json.load(f)
        if loss_history:
            # Get final loss per frame
            frames = set(record['frame'] for record in loss_history)
            final_losses = []
            for frame in sorted(frames):
                frame_records = [r for r in loss_history if r['frame'] == frame]
                if frame_records:
                    final_losses.append(frame_records[-1]['total_loss'])
            loss_summary = {
                'total_frames': len(frames),
                'avg_final_loss': np.mean(final_losses) if final_losses else 0,
                'min_final_loss': min(final_losses) if final_losses else 0,
                'max_final_loss': max(final_losses) if final_losses else 0,
            }

    # Find available files
    videos_dir = result_dir / "videos"
    render_dir = result_dir / "render"

    video_files = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
    render_images = sorted(render_dir.glob("step_2_frame_*.png"))[:10] if render_dir.exists() else []
    loss_plot = result_dir / "loss_history.png"
    summary_image = result_dir / "summary_frames.png"

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAMMAL Mouse Fitting Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header .timestamp {{
            opacity: 0.8;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-card .label {{
            color: #666;
            margin-top: 5px;
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .video-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }}
        .video-card h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .video-card video {{
            width: 100%;
            border-radius: 5px;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .image-gallery img {{
            width: 100%;
            border-radius: 5px;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .image-gallery img:hover {{
            transform: scale(1.02);
        }}
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
        .summary-img {{
            width: 100%;
            max-width: 1200px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .loss-plot {{
            width: 100%;
            max-width: 800px;
            border-radius: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐭 MAMMAL Mouse Fitting Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="timestamp">Result Directory: {result_dir}</div>
    </div>

    <div class="section">
        <h2>📊 Fitting Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{loss_summary.get('total_frames', 'N/A')}</div>
                <div class="label">Total Frames</div>
            </div>
            <div class="metric-card">
                <div class="value">{loss_summary.get('avg_final_loss', 0):.2f}</div>
                <div class="label">Avg Final Loss</div>
            </div>
            <div class="metric-card">
                <div class="value">{loss_summary.get('min_final_loss', 0):.2f}</div>
                <div class="label">Min Final Loss</div>
            </div>
            <div class="metric-card">
                <div class="value">{loss_summary.get('max_final_loss', 0):.2f}</div>
                <div class="label">Max Final Loss</div>
            </div>
        </div>
    </div>
"""

    # Summary image section
    if summary_image.exists():
        html_content += f"""
    <div class="section">
        <h2>🖼️ Frame Summary</h2>
        <img src="{summary_image.name}" class="summary-img" alt="Frame Summary">
    </div>
"""

    # Videos section
    if include_videos and video_files:
        html_content += """
    <div class="section">
        <h2>🎬 Generated Videos</h2>
        <div class="video-grid">
"""
        video_titles = {
            'mesh_grid.mp4': '3D Mesh Animation (Grid View)',
            'keypoints_comparison.mp4': 'Keypoint Comparison (GT vs Predicted)',
            'full_overlay.mp4': 'Full Overlay (Mesh + Keypoints)',
            'keypoints_both.mp4': 'Keypoints (GT + Predicted)',
        }
        for vf in video_files:
            title = video_titles.get(vf.name, vf.name)
            html_content += f"""
            <div class="video-card">
                <h4>{title}</h4>
                <video controls>
                    <source src="videos/{vf.name}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
"""
        html_content += """
        </div>
    </div>
"""

    # Loss plot section
    if include_loss_plots and loss_plot.exists():
        html_content += f"""
    <div class="section">
        <h2>📈 Loss History</h2>
        <img src="{loss_plot.name}" class="loss-plot" alt="Loss History">
    </div>
"""

    # Render images section - create render_sample folder for portability
    if render_images:
        # Copy sample renders to render_sample folder for portable HTML
        render_sample_dir = result_dir / "render_sample"
        render_sample_dir.mkdir(exist_ok=True)
        import shutil
        for img_path in render_images[:10]:
            dest_path = render_sample_dir / img_path.name
            if not dest_path.exists():
                shutil.copy(img_path, dest_path)

        html_content += """
    <div class="section">
        <h2>🎨 Sample Renders (Step 2 - Final)</h2>
        <div class="image-gallery">
"""
        for img_path in render_images[:10]:
            html_content += f"""
            <img src="render_sample/{img_path.name}" alt="{img_path.name}">
"""
        html_content += """
        </div>
    </div>
"""

    # Config section
    if config_str:
        html_content += f"""
    <div class="section">
        <h2>⚙️ Configuration</h2>
        <pre>{config_str}</pre>
    </div>
"""

    html_content += """
</body>
</html>
"""

    # Save HTML
    output_path = result_dir / "report.html"
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML report saved: {output_path}")
    return str(output_path)


def generate_post_fitting_outputs(
    result_dir: str,
    fps: int = 30,
    create_videos: bool = True,
    create_report: bool = True,
    create_summary: bool = True,
) -> Dict[str, str]:
    """
    Generate all post-fitting visualizations and reports.
    This function should be called after fitting completes.

    Args:
        result_dir: Path to fitting result directory
        fps: Video frame rate
        create_videos: Create all video types
        create_report: Create HTML report
        create_summary: Create summary image

    Returns:
        Dict of output type to path
    """
    outputs = {}

    print("\n" + "=" * 60)
    print("Generating post-fitting outputs...")
    print("=" * 60)

    # Create summary image first (used in report)
    if create_summary:
        try:
            summary_path = create_frame_summary_image(result_dir)
            if summary_path:
                outputs["summary_image"] = summary_path
        except Exception as e:
            print(f"Warning: Failed to create summary image: {e}")

    # Create videos
    if create_videos:
        try:
            video_outputs = create_all_videos(
                result_dir=result_dir,
                fps=fps,
                mesh_only=True,
                keypoints_only=True,
                full_overlay=True,
            )
            outputs.update(video_outputs)
        except Exception as e:
            print(f"Warning: Failed to create videos: {e}")

    # Create HTML report
    if create_report:
        try:
            report_path = create_html_report(result_dir)
            outputs["html_report"] = report_path
        except Exception as e:
            print(f"Warning: Failed to create HTML report: {e}")

    print("\n" + "=" * 60)
    print("Post-fitting outputs complete!")
    print("=" * 60)
    for output_type, path in outputs.items():
        print(f"  {output_type}: {path}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Create mesh animation and overlay videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create all videos (default)
    python scripts/mesh_animation.py --result_dir results/fitting/experiment_name

    # Create only mesh video
    python scripts/mesh_animation.py --result_dir ... --mesh_only

    # Create only keypoint comparison video
    python scripts/mesh_animation.py --result_dir ... --keypoints_only

    # Create full overlay video (mesh + keypoints)
    python scripts/mesh_animation.py --result_dir ... --full_overlay

    # Legacy single video mode
    python scripts/mesh_animation.py --result_dir ... --camera orbit
        """
    )
    parser.add_argument("--result_dir", type=str, required=True,
                       help="Path to fitting result directory")
    parser.add_argument("--output_video", type=str, default=None,
                       help="Output video path (default: result_dir/videos/)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Video frame rate (default: 30)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720],
                       help="Video resolution for 3D mesh view (default: 1280 720)")

    # Video type options (new)
    video_group = parser.add_argument_group("Video Types (default: all)")
    video_group.add_argument("--all", action="store_true", default=True,
                            help="Create all video types (default)")
    video_group.add_argument("--mesh_only", action="store_true",
                            help="Create only 3D mesh animation video")
    video_group.add_argument("--keypoints_only", action="store_true",
                            help="Create only keypoint comparison video (GT vs predicted)")
    video_group.add_argument("--full_overlay", action="store_true",
                            help="Create only full overlay video (mesh + keypoints on original)")

    # Mesh video options
    mesh_group = parser.add_argument_group("Mesh Video Options")
    mesh_group.add_argument("--camera", type=str, default="grid",
                           choices=["grid", "side", "front", "back", "top", "diagonal", "orbit"],
                           help="Camera type for 3D mesh view (default: grid)")
    mesh_group.add_argument("--backend", type=str, default="trimesh",
                           choices=["trimesh", "pyvista"],
                           help="Rendering backend (default: trimesh)")
    mesh_group.add_argument("--preview", action="store_true",
                           help="Show preview window (pyvista only)")
    mesh_group.add_argument("--wireframe", action="store_true",
                           help="Show wireframe overlay")

    # Keypoint video options
    kp_group = parser.add_argument_group("Keypoint Video Options")
    kp_group.add_argument("--keypoint_type", type=str, default="both",
                         choices=["gt", "pred", "both"],
                         help="Keypoints to show: gt, pred, or both (default: both)")
    kp_group.add_argument("--no_bones", action="store_true",
                         help="Don't draw bone connections")

    # Legacy options
    legacy_group = parser.add_argument_group("Legacy Options")
    legacy_group.add_argument("--side_by_side", action="store_true",
                             help="Create side-by-side comparison with original renders")
    legacy_group.add_argument("--export_params", action="store_true",
                             help="Export parameters to JSON")

    args = parser.parse_args()

    # Determine which videos to create
    explicit_selection = args.mesh_only or args.keypoints_only or args.full_overlay
    if explicit_selection:
        args.all = False

    # Validate result directory
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        sys.exit(1)

    obj_dir = result_dir / "obj"
    params_dir = result_dir / "params"
    render_dir = result_dir / "render"

    # Create videos subdirectory
    videos_dir = result_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    # Export params if requested
    if args.export_params and params_dir.exists():
        export_params_to_json(str(params_dir), str(result_dir / "params_export.json"))

    # ====================================================================
    # Create videos based on selected options
    # ====================================================================
    output_paths = {}

    if args.all:
        # Create all video types
        print("\n" + "=" * 60)
        print("Creating all video types...")
        print("=" * 60)

        output_paths = create_all_videos(
            result_dir=str(result_dir),
            fps=args.fps,
            mesh_only=True,
            keypoints_only=True,
            full_overlay=True,
            mesh_camera=args.camera,
        )

    else:
        # Create selected video types
        obj_files = load_mesh_sequence(str(obj_dir)) if obj_dir.exists() else []

        if args.mesh_only:
            if not obj_files:
                print(f"Error: No OBJ files found in {obj_dir}")
                sys.exit(1)

            print(f"\nCreating mesh-only video ({len(obj_files)} frames)...")
            output_path = str(videos_dir / f"mesh_{args.camera}.mp4")

            if args.camera == "grid":
                create_animation_grid(
                    obj_files=obj_files,
                    output_path=output_path,
                    fps=args.fps,
                    resolution=tuple(args.resolution),
                    show_wireframe=args.wireframe,
                )
            elif args.backend == "pyvista":
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
            output_paths["mesh_only"] = output_path

        if args.keypoints_only:
            print("\nCreating keypoint comparison video...")
            kp_path = str(videos_dir / f"keypoints_{args.keypoint_type}.mp4")
            create_keypoint_video(
                result_dir=str(result_dir),
                output_path=kp_path,
                fps=args.fps,
                keypoint_type=args.keypoint_type,
                draw_bones=not args.no_bones,
            )
            output_paths["keypoints"] = kp_path

        if args.full_overlay:
            print("\nCreating full overlay video...")
            overlay_path = str(videos_dir / "full_overlay.mp4")
            create_full_overlay_video(
                result_dir=str(result_dir),
                output_path=overlay_path,
                fps=args.fps,
                include_mesh=True,
                include_gt_keypoints=True,
                include_pred_keypoints=True,
                draw_bones=not args.no_bones,
            )
            output_paths["full_overlay"] = overlay_path

    # Legacy side-by-side option
    if args.side_by_side and render_dir.exists():
        obj_files = load_mesh_sequence(str(obj_dir)) if obj_dir.exists() else []
        if obj_files:
            sbs_path = str(videos_dir / "side_by_side.mp4")
            create_side_by_side_video(
                obj_files=obj_files,
                render_dir=str(render_dir),
                output_path=sbs_path,
                fps=args.fps,
                resolution=(args.resolution[0], args.resolution[1] // 2),
            )
            output_paths["side_by_side"] = sbs_path

    # Print summary
    print("\n" + "=" * 60)
    print("Video generation complete!")
    print("=" * 60)
    for video_type, path in output_paths.items():
        print(f"  {video_type}: {path}")
    print()


if __name__ == "__main__":
    main()
