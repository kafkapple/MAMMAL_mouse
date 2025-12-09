"""
Rerun Exporter Module

Export UV-textured mesh visualization to Rerun RRD format.
Supports 3D mesh with vertex colors, 2D rendered images, and time series.

Based on patterns from pose-splatter/export_temporal_sequence_rerun.py
"""

import os
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# Lazy import rerun
RERUN_AVAILABLE = False
rr = None


def _check_rerun():
    """Check and import rerun if available."""
    global RERUN_AVAILABLE, rr
    if rr is not None:
        return RERUN_AVAILABLE
    try:
        import rerun as _rr
        rr = _rr
        RERUN_AVAILABLE = True
    except ImportError:
        RERUN_AVAILABLE = False
        print("Warning: rerun-sdk not installed. Install with: pip install rerun-sdk")
    return RERUN_AVAILABLE


class RerunExporter:
    """
    Export visualization to Rerun RRD format.

    Features:
    - 3D textured mesh with vertex colors
    - Multi-view rendered images
    - Time sequence support
    - Keypoint and skeleton visualization
    - Camera position markers

    Usage:
        exporter = RerunExporter("output.rrd")
        exporter.init()

        for frame_idx in range(n_frames):
            exporter.log_mesh(frame_idx, vertices, faces, colors)
            exporter.log_image(frame_idx, image, "front")

        exporter.save()
    """

    def __init__(
        self,
        output_path: str,
        app_name: str = "mammal_mouse_visualization",
    ):
        """
        Args:
            output_path: Path to save RRD file
            app_name: Rerun application name
        """
        if not _check_rerun():
            raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk")

        self.output_path = output_path
        self.app_name = app_name
        self._initialized = False

    def init(self) -> None:
        """Initialize Rerun recording."""
        rr.init(self.app_name, spawn=False)

        # Set coordinate system
        # Mouse mesh: X=body length, Y=width, Z=up
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self._initialized = True
        print(f"Rerun initialized: {self.app_name}")

    def save(self) -> str:
        """
        Save RRD file.

        Returns:
            Path to saved file
        """
        if not self._initialized:
            raise RuntimeError("Exporter not initialized. Call init() first.")

        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        rr.save(self.output_path)

        print(f"RRD saved: {self.output_path}")
        print(f"View with: rerun {self.output_path}")

        return self.output_path

    def log_mesh(
        self,
        frame_idx: int,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: np.ndarray,
        path: str = "mesh/textured",
    ) -> None:
        """
        Log textured mesh to Rerun.

        Args:
            frame_idx: Frame index for time series
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices
            vertex_colors: (N, 3) RGB colors in [0, 255] uint8
            path: Entity path in Rerun hierarchy
        """
        rr.set_time_sequence("frame", frame_idx)

        # Ensure correct dtypes
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)

        # Ensure vertex colors are uint8
        if vertex_colors.dtype != np.uint8:
            if vertex_colors.max() <= 1.0:
                vertex_colors = (vertex_colors * 255).astype(np.uint8)
            else:
                vertex_colors = vertex_colors.astype(np.uint8)

        rr.log(
            path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=vertex_colors,
            ),
        )

    def log_points(
        self,
        frame_idx: int,
        positions: np.ndarray,
        colors: Optional[np.ndarray] = None,
        radii: Optional[np.ndarray] = None,
        path: str = "points",
    ) -> None:
        """
        Log 3D points (e.g., keypoints, Gaussians).

        Args:
            frame_idx: Frame index
            positions: (N, 3) point positions
            colors: (N, 3) or (N, 4) RGB/RGBA colors
            radii: (N,) point radii
            path: Entity path
        """
        rr.set_time_sequence("frame", frame_idx)

        kwargs = {"positions": positions}

        if colors is not None:
            if colors.dtype != np.uint8:
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)
            kwargs["colors"] = colors

        if radii is not None:
            kwargs["radii"] = radii

        rr.log(path, rr.Points3D(**kwargs))

    def log_keypoints(
        self,
        frame_idx: int,
        keypoints_3d: np.ndarray,
        colors: Optional[np.ndarray] = None,
        radius: float = 0.005,
        path: str = "skeleton/keypoints",
    ) -> None:
        """
        Log 3D keypoints with default styling.

        Args:
            frame_idx: Frame index
            keypoints_3d: (K, 3) keypoint positions
            colors: (K, 3) colors. Default: green
            radius: Point radius
            path: Entity path
        """
        if colors is None:
            colors = np.full((len(keypoints_3d), 3), [0, 255, 0], dtype=np.uint8)

        radii = np.full(len(keypoints_3d), radius)

        self.log_points(frame_idx, keypoints_3d, colors, radii, path)

    def log_skeleton(
        self,
        frame_idx: int,
        keypoints_3d: np.ndarray,
        bones: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 200, 0),
        path: str = "skeleton/bones",
    ) -> None:
        """
        Log skeleton bone connections as line segments.

        Args:
            frame_idx: Frame index
            keypoints_3d: (K, 3) keypoint positions
            bones: List of (start_idx, end_idx) tuples
            color: RGB color
            path: Entity path
        """
        rr.set_time_sequence("frame", frame_idx)

        # Build line strips
        lines = []
        for i, j in bones:
            if i < len(keypoints_3d) and j < len(keypoints_3d):
                lines.append([keypoints_3d[i], keypoints_3d[j]])

        if lines:
            rr.log(
                path,
                rr.LineStrips3D(
                    lines,
                    colors=[color] * len(lines),
                    radii=[0.002] * len(lines),
                ),
            )

    def log_image(
        self,
        frame_idx: int,
        image: np.ndarray,
        view_name: str,
        path: str = "renders",
    ) -> None:
        """
        Log 2D rendered image.

        Args:
            frame_idx: Frame index
            image: (H, W, 3) RGB image in uint8 or float [0,1]
            view_name: Name of the view (e.g., "front", "orbit_000")
            path: Base entity path
        """
        rr.set_time_sequence("frame", frame_idx)

        # Convert to proper format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        rr.log(f"{path}/{view_name}", rr.Image(image))

    def log_camera(
        self,
        frame_idx: int,
        position: np.ndarray,
        name: str,
        color: Tuple[int, int, int] = (255, 100, 100),
        radius: float = 0.02,
        path: str = "cameras",
    ) -> None:
        """
        Log camera position marker.

        Args:
            frame_idx: Frame index
            position: (3,) camera position
            name: Camera name
            color: RGB color
            radius: Marker radius
            path: Base entity path
        """
        rr.set_time_sequence("frame", frame_idx)

        rr.log(
            f"{path}/{name}",
            rr.Points3D(
                positions=[position],
                colors=[color],
                radii=[radius],
            ),
        )

    def log_scalar(
        self,
        frame_idx: int,
        name: str,
        value: float,
        path: str = "metrics",
    ) -> None:
        """
        Log scalar metric for time series plot.

        Args:
            frame_idx: Frame index
            name: Metric name
            value: Scalar value
            path: Base entity path
        """
        rr.set_time_sequence("frame", frame_idx)

        # Handle different rerun versions
        try:
            rr.log(f"{path}/{name}", rr.Scalar(value))
        except AttributeError:
            # Older versions use TimeSeriesScalar
            rr.log(f"{path}/{name}", rr.TimeSeriesScalar(value))

    def log_text(
        self,
        frame_idx: int,
        text: str,
        path: str = "info",
    ) -> None:
        """
        Log text annotation (markdown supported).

        Args:
            frame_idx: Frame index
            text: Text content (markdown)
            path: Entity path
        """
        rr.set_time_sequence("frame", frame_idx)

        rr.log(path, rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))

    def log_static_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: np.ndarray,
        path: str = "mesh/reference",
    ) -> None:
        """
        Log static (non-animated) mesh.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices
            vertex_colors: (N, 3) RGB colors
            path: Entity path
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)

        if vertex_colors.dtype != np.uint8:
            if vertex_colors.max() <= 1.0:
                vertex_colors = (vertex_colors * 255).astype(np.uint8)
            else:
                vertex_colors = vertex_colors.astype(np.uint8)

        rr.log(
            path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=vertex_colors,
            ),
            static=True,
        )


# Keypoint and skeleton definitions for mouse model
MOUSE_KEYPOINT_LABELS = {
    0: 'L_ear', 1: 'R_ear', 2: 'nose',
    3: 'neck', 4: 'body_mid',
    5: 'tail_root', 6: 'tail_mid', 7: 'tail_end',
    8: 'L_paw', 9: 'L_paw_end', 10: 'L_elbow', 11: 'L_shoulder',
    12: 'R_paw', 13: 'R_paw_end', 14: 'R_elbow', 15: 'R_shoulder',
    16: 'L_foot', 17: 'L_knee', 18: 'L_hip',
    19: 'R_foot', 20: 'R_knee', 21: 'R_hip',
}

MOUSE_BONES = [
    (0, 2), (1, 2),  # Ears to nose
    (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),  # Spine
    (8, 9), (9, 10), (10, 11), (11, 3),  # Left front leg
    (12, 13), (13, 14), (14, 15), (15, 3),  # Right front leg
    (16, 17), (17, 18), (18, 5),  # Left hind leg
    (19, 20), (20, 21), (21, 5),  # Right hind leg
]

# Part-based colors for keypoints
MOUSE_KEYPOINT_COLORS = {
    'head': [255, 255, 0],      # Yellow
    'body': [255, 0, 255],      # Magenta
    'tail': [255, 165, 0],      # Orange
    'L_front': [0, 0, 255],     # Blue
    'R_front': [0, 255, 0],     # Green
    'L_hind': [0, 255, 255],    # Cyan
    'R_hind': [255, 0, 0],      # Red
}


def get_keypoint_colors() -> np.ndarray:
    """Get part-based colors for all 22 keypoints."""
    colors = np.zeros((22, 3), dtype=np.uint8)

    part_indices = {
        'head': [0, 1, 2],
        'body': [3, 4],
        'tail': [5, 6, 7],
        'L_front': [8, 9, 10, 11],
        'R_front': [12, 13, 14, 15],
        'L_hind': [16, 17, 18],
        'R_hind': [19, 20, 21],
    }

    for part, indices in part_indices.items():
        color = MOUSE_KEYPOINT_COLORS[part]
        for idx in indices:
            colors[idx] = color

    return colors
