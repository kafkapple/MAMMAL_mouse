"""
Video Generator Module

Generate MP4 videos from rendered frames.
Supports single view, grid layout, and side-by-side comparisons.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from pathlib import Path


class VideoGenerator:
    """
    Generate MP4 videos from rendered image frames.

    Features:
    - Single view video
    - Grid layout (multi-view)
    - Side-by-side comparison
    - Frame labels and annotations
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        codec: str = 'mp4v',
        resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            output_path: Path to output MP4 file
            fps: Frames per second
            codec: Video codec (default: mp4v)
            resolution: (width, height). Auto-detected from first frame if None
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.resolution = resolution

        self._writer = None
        self._frame_count = 0

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    def _init_writer(self, frame: np.ndarray) -> None:
        """Initialize video writer from first frame."""
        if self.resolution is None:
            h, w = frame.shape[:2]
            self.resolution = (w, h)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.resolution,
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add single frame to video.

        Args:
            frame: (H, W, 3) RGB image
        """
        # Initialize writer on first frame
        if self._writer is None:
            self._init_writer(frame)

        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Resize if needed
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.resolution:
            frame_bgr = cv2.resize(frame_bgr, self.resolution)

        self._writer.write(frame_bgr)
        self._frame_count += 1

    def add_grid_frame(
        self,
        frames: List[np.ndarray],
        labels: Optional[List[str]] = None,
        grid_cols: int = 2,
        padding: int = 2,
        label_color: Tuple[int, int, int] = (255, 255, 255),
        label_bg_color: Tuple[int, int, int] = (40, 40, 40),
    ) -> None:
        """
        Add multiple frames arranged in grid layout.

        Args:
            frames: List of (H, W, 3) RGB images
            labels: Optional labels for each frame
            grid_cols: Number of columns in grid
            padding: Pixels between grid cells
            label_color: Text color (RGB)
            label_bg_color: Label background color (RGB)
        """
        n_frames = len(frames)
        if n_frames == 0:
            return

        grid_rows = (n_frames + grid_cols - 1) // grid_cols

        # Get cell size from first frame
        cell_h, cell_w = frames[0].shape[:2]

        # Calculate total grid size
        total_w = grid_cols * cell_w + (grid_cols - 1) * padding
        total_h = grid_rows * cell_h + (grid_rows - 1) * padding

        # Create grid canvas
        grid = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        for i, frame in enumerate(frames):
            row = i // grid_cols
            col = i % grid_cols

            x = col * (cell_w + padding)
            y = row * (cell_h + padding)

            # Resize frame if needed
            if frame.shape[:2] != (cell_h, cell_w):
                frame = cv2.resize(frame, (cell_w, cell_h))

            # Place frame in grid
            grid[y:y + cell_h, x:x + cell_w] = frame

            # Add label if provided
            if labels and i < len(labels):
                self._add_label(
                    grid,
                    labels[i],
                    (x + 10, y + 25),
                    label_color,
                    label_bg_color,
                )

        self.add_frame(grid)

    def _add_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
    ) -> None:
        """Add text label with background."""
        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = position

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + baseline + 5),
            bg_color_bgr,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            text,
            position,
            font,
            font_scale,
            color_bgr,
            thickness,
        )

    def close(self) -> str:
        """
        Finalize and close video file.

        Returns:
            Path to saved video
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        print(f"Video saved: {self.output_path} ({self._frame_count} frames)")
        return self.output_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_grid_video(
    frame_sequences: List[List[np.ndarray]],
    output_path: str,
    labels: Optional[List[str]] = None,
    fps: int = 30,
    grid_cols: int = 2,
) -> str:
    """
    Create grid video from multiple view sequences.

    Args:
        frame_sequences: List of frame lists, one per view
        output_path: Output video path
        labels: Labels for each view
        fps: Frame rate
        grid_cols: Grid columns

    Returns:
        Path to saved video
    """
    n_views = len(frame_sequences)
    n_frames = min(len(seq) for seq in frame_sequences)

    if n_frames == 0:
        raise ValueError("No frames to process")

    with VideoGenerator(output_path, fps=fps) as gen:
        for i in range(n_frames):
            frames = [seq[i] for seq in frame_sequences]
            gen.add_grid_frame(frames, labels=labels, grid_cols=grid_cols)

    return output_path


def create_orbit_video(
    renderer,  # TexturedMeshRenderer
    vertices: np.ndarray,
    output_path: str,
    n_frames: int = 120,
    elevation: float = 30.0,
    distance_factor: float = 2.5,
    fps: int = 30,
) -> str:
    """
    Create 360-degree orbit video around mesh.

    Args:
        renderer: TexturedMeshRenderer instance
        vertices: (N, 3) vertex positions
        output_path: Output video path
        n_frames: Frames for full rotation
        elevation: Camera elevation in degrees
        distance_factor: Distance from mesh center
        fps: Frame rate

    Returns:
        Path to saved video
    """
    from .camera_paths import CameraPathGenerator, compute_mesh_bounds

    # Compute mesh bounds
    center, scale = compute_mesh_bounds(vertices)

    # Generate camera path
    cam_gen = CameraPathGenerator(center, scale)
    camera_poses = cam_gen.orbit_360(
        n_frames=n_frames,
        elevation=elevation,
        distance_factor=distance_factor,
    )

    # Render and save video
    with VideoGenerator(output_path, fps=fps) as gen:
        for pose in camera_poses:
            cam_matrix = pose.to_pyrender_pose()
            image = renderer.render_pyrender(vertices, cam_matrix)
            gen.add_frame(image)

    return output_path


def frames_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> str:
    """
    Simple utility to convert frame list to video.

    Args:
        frames: List of (H, W, 3) RGB images
        output_path: Output video path
        fps: Frame rate

    Returns:
        Path to saved video
    """
    with VideoGenerator(output_path, fps=fps) as gen:
        for frame in frames:
            gen.add_frame(frame)

    return output_path


def pack_images(
    images: List[np.ndarray],
    max_cols: int = 3,
    padding: int = 2,
    background: int = 0,
) -> np.ndarray:
    """
    Pack multiple images into a grid.

    Args:
        images: List of images (can be different sizes)
        max_cols: Maximum columns
        padding: Pixels between images
        background: Background gray value

    Returns:
        Combined grid image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    # Find max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Create canvas
    canvas_h = rows * max_h + (rows - 1) * padding
    canvas_w = cols * max_w + (cols - 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), background, dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = col * (max_w + padding)
        y = row * (max_h + padding)

        h, w = img.shape[:2]
        canvas[y:y + h, x:x + w] = img

    return canvas
