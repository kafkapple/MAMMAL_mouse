"""
Camera Path Generator Module

Generate camera trajectories for mesh visualization.
Supports orbit, fixed views, multi-view, and novel view interpolation.

Mouse mesh coordinate system:
    X-axis: body length (head to tail)
    Y-axis: body width (left to right)
    Z-axis: body height (up direction)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CameraPose:
    """Camera pose representation."""

    position: np.ndarray  # (3,) camera position
    target: np.ndarray  # (3,) look-at target
    up: np.ndarray  # (3,) up vector
    name: str = 'camera'

    def to_extrinsic_matrix(self) -> np.ndarray:
        """
        Convert to 4x4 extrinsic matrix (world-to-camera transform).

        Returns:
            4x4 extrinsic matrix [R|t]
        """
        # Compute camera axes
        forward = self.target - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        right = np.cross(forward, self.up)
        if np.linalg.norm(right) < 1e-6:
            # Handle degenerate case (up parallel to forward)
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Rotation matrix (camera axes as rows)
        R = np.stack([right, up, -forward], axis=0)

        # Translation
        t = -R @ self.position

        # Build 4x4 matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        return extrinsic

    def to_pyrender_pose(self) -> np.ndarray:
        """
        Convert to pyrender camera pose matrix.

        Pyrender uses OpenGL convention where camera looks along -Z.

        Returns:
            4x4 camera pose matrix (camera-to-world transform)
        """
        forward = self.target - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        right = np.cross(forward, self.up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Camera pose (camera-to-world)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # OpenGL: camera looks along -Z
        pose[:3, 3] = self.position

        return pose


class CameraPathGenerator:
    """
    Generate camera trajectories for visualization.

    Supports multiple view modes:
    - orbit: 360-degree rotation around mesh
    - fixed: Standard viewpoints (front, back, side, top, diagonal)
    - multiview: Use cameras from fitting
    - novel: Custom azimuth/elevation angles
    """

    def __init__(
        self,
        mesh_center: np.ndarray,
        mesh_scale: float,
        up_vector: np.ndarray = None,
    ):
        """
        Args:
            mesh_center: (3,) center of the mesh
            mesh_scale: Maximum extent of the mesh
            up_vector: (3,) up direction. Default: Z-up [0, 0, 1]
        """
        self.center = np.array(mesh_center, dtype=np.float32)
        self.scale = float(mesh_scale)
        self.up = np.array(up_vector if up_vector is not None else [0, 0, 1], dtype=np.float32)

    def orbit_360(
        self,
        n_frames: int = 120,
        elevation: float = 30.0,
        distance_factor: float = 2.5,
        start_azimuth: float = 0.0,
    ) -> List[CameraPose]:
        """
        Generate 360-degree orbit camera path.

        Args:
            n_frames: Number of frames for full rotation
            elevation: Camera elevation angle in degrees
            distance_factor: Distance as multiple of mesh scale
            start_azimuth: Starting azimuth angle in degrees

        Returns:
            List of CameraPose for each frame
        """
        poses = []
        distance = self.scale * distance_factor
        elevation_rad = np.radians(elevation)

        for i in range(n_frames):
            azimuth = start_azimuth + (i / n_frames) * 360
            azimuth_rad = np.radians(azimuth)

            # Spherical to Cartesian
            # X-axis is body length, Y-axis is width, Z-axis is up
            # Orbit in XY plane with elevation along Z
            x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            z = distance * np.sin(elevation_rad)

            position = self.center + np.array([x, y, z])

            poses.append(CameraPose(
                position=position,
                target=self.center.copy(),
                up=self.up.copy(),
                name=f'orbit_{i:03d}',
            ))

        return poses

    def fixed_views(
        self,
        views: List[str] = None,
        distance_factor: float = 2.5,
    ) -> List[CameraPose]:
        """
        Generate standard fixed viewpoints.

        Mouse mesh coordinate system:
            X-axis: body length (head +X, tail -X)
            Y-axis: body width (left +Y, right -Y)
            Z-axis: body height (up +Z)

        Args:
            views: List of view names. Options:
                   'front' - looking at mouse face (+X direction)
                   'back' - looking at mouse tail (-X direction)
                   'side' - side profile (+Y direction)
                   'side_left' - same as 'side'
                   'side_right' - side profile (-Y direction)
                   'top' - bird's eye view (+Z direction)
                   'bottom' - from below (-Z direction)
                   'diagonal' - 45-degree view
            distance_factor: Distance as multiple of mesh scale

        Returns:
            List of CameraPose for each view
        """
        if views is None:
            views = ['front', 'side', 'top', 'diagonal']

        poses = []
        distance = self.scale * distance_factor

        view_configs = {
            'front': {
                'offset': np.array([distance, 0, distance * 0.2]),
                'name': 'front',
            },
            'back': {
                'offset': np.array([-distance, 0, distance * 0.2]),
                'name': 'back',
            },
            'side': {
                'offset': np.array([0, distance, distance * 0.2]),
                'name': 'side',
            },
            'side_left': {
                'offset': np.array([0, distance, distance * 0.2]),
                'name': 'side_left',
            },
            'side_right': {
                'offset': np.array([0, -distance, distance * 0.2]),
                'name': 'side_right',
            },
            'top': {
                'offset': np.array([0, distance * 0.1, distance]),
                'name': 'top',
            },
            'bottom': {
                'offset': np.array([0, distance * 0.1, -distance]),
                'name': 'bottom',
            },
            'diagonal': {
                'offset': np.array([distance * 0.5, distance * 0.7, distance * 0.5]),
                'name': 'diagonal',
            },
            'diagonal_back': {
                'offset': np.array([-distance * 0.5, distance * 0.7, distance * 0.5]),
                'name': 'diagonal_back',
            },
        }

        for view_name in views:
            if view_name not in view_configs:
                print(f"Warning: Unknown view '{view_name}', skipping")
                continue

            config = view_configs[view_name]
            position = self.center + config['offset']

            poses.append(CameraPose(
                position=position,
                target=self.center.copy(),
                up=self.up.copy(),
                name=config['name'],
            ))

        return poses

    def novel_views(
        self,
        azimuths: List[float],
        elevations: List[float],
        distance_factor: float = 2.5,
    ) -> List[CameraPose]:
        """
        Generate novel views from azimuth/elevation angles.

        Args:
            azimuths: List of azimuth angles in degrees
            elevations: List of elevation angles in degrees (same length as azimuths)
            distance_factor: Distance as multiple of mesh scale

        Returns:
            List of CameraPose for each view
        """
        if len(azimuths) != len(elevations):
            raise ValueError("azimuths and elevations must have same length")

        poses = []
        distance = self.scale * distance_factor

        for i, (az, el) in enumerate(zip(azimuths, elevations)):
            az_rad = np.radians(az)
            el_rad = np.radians(el)

            x = distance * np.cos(el_rad) * np.sin(az_rad)
            y = distance * np.cos(el_rad) * np.cos(az_rad)
            z = distance * np.sin(el_rad)

            position = self.center + np.array([x, y, z])

            poses.append(CameraPose(
                position=position,
                target=self.center.copy(),
                up=self.up.copy(),
                name=f'novel_az{az:.0f}_el{el:.0f}',
            ))

        return poses

    def interpolate_poses(
        self,
        start_pose: CameraPose,
        end_pose: CameraPose,
        n_frames: int,
    ) -> List[CameraPose]:
        """
        Interpolate between two camera poses.

        Args:
            start_pose: Starting camera pose
            end_pose: Ending camera pose
            n_frames: Number of interpolation frames

        Returns:
            List of interpolated CameraPose
        """
        poses = []

        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)

            # Linear interpolation
            position = (1 - t) * start_pose.position + t * end_pose.position
            target = (1 - t) * start_pose.target + t * end_pose.target

            # SLERP would be better for up vector, but linear is usually fine
            up = (1 - t) * start_pose.up + t * end_pose.up
            up = up / np.linalg.norm(up)

            poses.append(CameraPose(
                position=position,
                target=target,
                up=up,
                name=f'interp_{i:03d}',
            ))

        return poses

    @staticmethod
    def from_opencv_cameras(
        cam_dicts: List[Dict],
        names: List[str] = None,
    ) -> List[CameraPose]:
        """
        Create camera poses from OpenCV camera dictionaries.

        Args:
            cam_dicts: List of camera dicts with 'R' (3x3) and 'T' (3x1 or 3,) keys
            names: Optional names for each camera

        Returns:
            List of CameraPose
        """
        poses = []

        for i, cam_dict in enumerate(cam_dicts):
            R = cam_dict['R']
            T = cam_dict['T']

            # Handle T shape variations
            if T.ndim > 1:
                T = T.flatten()
            T = T / 1000  # Convert mm to m

            # Camera position: -R^T @ T
            position = -R.T @ T

            # Camera looks along +Z in OpenCV convention
            # Target is some distance in front of camera
            forward = R[2, :]  # Third row of R is camera's Z direction
            target = position + forward

            # Up direction is negative Y in OpenCV convention
            up = -R[1, :]

            name = names[i] if names else f'cam_{i}'

            poses.append(CameraPose(
                position=position,
                target=target,
                up=up,
                name=name,
            ))

        return poses


def compute_mesh_bounds(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute mesh center and scale from vertices.

    Args:
        vertices: (N, 3) vertex positions

    Returns:
        center: (3,) mesh center
        scale: Maximum extent
    """
    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)

    center = (min_v + max_v) / 2
    scale = (max_v - min_v).max()

    return center, scale
