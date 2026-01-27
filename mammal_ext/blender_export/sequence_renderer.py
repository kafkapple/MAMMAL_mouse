"""
Sequence Renderer: Render textured mesh sequence as 6-view grid video.

Reads all step_2 OBJ frames from a fitting result, applies UV texture,
and renders a 6-camera grid video using the original calibration cameras.

Usage:
    python -m mammal_ext.blender_export.sequence_renderer \
        --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
        --texture results/sweep/run_wild-sweep-9/texture_final.png \
        --output_dir exports/renders/

    # Custom resolution and FPS
    python -m mammal_ext.blender_export.sequence_renderer \
        --result_dir ... --texture ... --output_dir ... \
        --image_size 512 --fps 15
"""

import os
import pickle
import argparse
import numpy as np
from typing import List, Tuple, Optional

# Set headless rendering before importing pyrender
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2

from .batch_export import find_obj_frames
from .obj_exporter import parse_obj_vertices


def load_cameras(
    data_dir: str = "data/examples/markerless_mouse_1_nerf",
) -> list:
    """Load 6-view camera calibration from new_cam.pkl."""
    cam_path = os.path.join(data_dir, "new_cam.pkl")
    with open(cam_path, 'rb') as f:
        cams = pickle.load(f)
    return cams


def opencv_to_opengl_pose(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV camera (R, T) to OpenGL/pyrender camera-to-world pose (4x4).

    OpenCV: Y-down, Z-forward
    OpenGL: Y-up, Z-backward
    """
    # World-to-camera in OpenCV
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()

    # Camera-to-world
    c2w = np.linalg.inv(w2c)

    # OpenCV to OpenGL: flip Y and Z axes
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w_gl = c2w @ flip

    return c2w_gl


def render_frame_6view(
    vertices: np.ndarray,
    renderer,
    cameras: list,
    image_size: int = 512,
) -> np.ndarray:
    """
    Render a single frame from 6 camera views and compose into 2x3 grid.

    Args:
        vertices: (N, 3) vertex positions (MAMMAL coordinates, mm)
        renderer: TexturedMeshRenderer instance
        cameras: List of camera dicts with 'R', 'T', 'K' keys
        image_size: Render resolution per view

    Returns:
        grid: (2*H, 3*W, 3) RGB image (uint8)
    """
    views = []
    for cam in cameras:
        R = np.array(cam['R'], dtype=np.float64)
        T = np.array(cam['T'], dtype=np.float64)
        K = np.array(cam['K'], dtype=np.float64)

        # Compute FOV from intrinsics
        fy = K[1, 1]
        fov_y = 2.0 * np.arctan(image_size / (2.0 * fy))

        # Camera pose for pyrender
        pose = opencv_to_opengl_pose(R, T)

        img = renderer.render_pyrender(
            vertices=vertices,
            camera_pose=pose,
            fov=np.degrees(fov_y),
        )
        views.append(img)

    # Compose 2x3 grid
    row1 = np.concatenate(views[:3], axis=1)
    row2 = np.concatenate(views[3:6], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    return grid


def render_sequence(
    result_dir: str,
    texture_path: str,
    output_dir: str,
    data_dir: str = "data/examples/markerless_mouse_1_nerf",
    model_dir: str = "mouse_model/mouse_txt",
    image_size: int = 512,
    fps: int = 15,
    max_frames: Optional[int] = None,
) -> str:
    """
    Render full mesh sequence as 6-view grid video.

    Args:
        result_dir: Fitting result directory
        texture_path: UV texture PNG
        output_dir: Output directory
        data_dir: Data directory with camera calibration
        model_dir: Body model UV data directory
        image_size: Render resolution per view
        fps: Output video FPS
        max_frames: Limit number of frames (None = all)

    Returns:
        Path to output video
    """
    from mammal_ext.visualization import TexturedMeshRenderer, VideoGenerator

    os.makedirs(output_dir, exist_ok=True)

    # Find frames
    obj_files = find_obj_frames(result_dir)
    if not obj_files:
        raise FileNotFoundError(f"No step_2_frame_*.obj in {result_dir}/obj/")

    if max_frames is not None:
        obj_files = obj_files[:max_frames]

    print(f"Rendering {len(obj_files)} frames as 6-view grid")
    print(f"  Result dir: {result_dir}")
    print(f"  Texture: {texture_path}")
    print(f"  Image size: {image_size}x{image_size} per view")
    print()

    # Load cameras
    cameras = load_cameras(data_dir)
    n_views = min(len(cameras), 6)
    cameras = cameras[:n_views]
    print(f"  Loaded {n_views} cameras from {data_dir}")

    # Initialize renderer
    renderer = TexturedMeshRenderer(
        image_size=(image_size, image_size),
        backend='pyrender',
    )
    renderer.load_mesh_data(model_dir)
    renderer.load_texture(texture_path)

    # Output video
    exp_name = os.path.basename(result_dir)
    video_path = os.path.join(output_dir, f"{exp_name}_6view_grid.mp4")
    video = VideoGenerator(output_path=video_path, fps=fps)

    # Also save first frame as sample image
    sample_path = os.path.join(output_dir, f"{exp_name}_6view_sample.png")

    for i, obj_path in enumerate(obj_files):
        vertices = parse_obj_vertices(obj_path)

        grid = render_frame_6view(
            vertices=vertices,
            renderer=renderer,
            cameras=cameras,
            image_size=image_size,
        )

        # Add frame label
        frame_name = os.path.basename(obj_path).replace('.obj', '')
        cv2.putText(
            grid, f"Frame {i}: {frame_name}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

        video.add_frame(grid)

        if i == 0:
            cv2.imwrite(sample_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  Sample image: {sample_path}")

        if (i + 1) % 20 == 0 or (i + 1) == len(obj_files):
            print(f"  [{i+1}/{len(obj_files)}] rendered")

    video.close()
    print(f"\nVideo saved: {video_path}")
    return video_path


def main():
    parser = argparse.ArgumentParser(
        description="Render mesh sequence as 6-view grid video",
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Fitting result directory')
    parser.add_argument('--texture', type=str, required=True,
                       help='UV texture PNG')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--data_dir', type=str,
                       default='data/examples/markerless_mouse_1_nerf',
                       help='Data directory with camera calibration')
    parser.add_argument('--model_dir', type=str, default='mouse_model/mouse_txt',
                       help='Model directory with UV definitions')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Render resolution per view')
    parser.add_argument('--fps', type=int, default=15,
                       help='Output video FPS')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Limit number of frames')

    args = parser.parse_args()

    render_sequence(
        result_dir=args.result_dir,
        texture_path=args.texture,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        image_size=args.image_size,
        fps=args.fps,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    main()
