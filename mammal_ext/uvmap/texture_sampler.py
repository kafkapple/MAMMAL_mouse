"""
Texture Sampler Module

Multi-view texture sampling from RGB images.
Projects mesh vertices to image space and samples colors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class TextureSampler(nn.Module):
    """
    Sample textures from multi-view RGB images.

    Projects 3D mesh vertices to 2D image coordinates,
    samples RGB values, and accumulates into UV texture map.
    """

    def __init__(
        self,
        uv_size: int = 512,
        device: str = 'cuda',
    ):
        """
        Args:
            uv_size: Resolution of UV texture map
            device: Torch device
        """
        super().__init__()
        self.uv_size = uv_size
        self.device = torch.device(device)

        # Camera parameters (set via set_cameras)
        self.cameras = []
        self.n_views = 0

    def set_cameras(
        self,
        cameras: List[Dict],
    ) -> None:
        """
        Set camera parameters for all views.

        Args:
            cameras: List of camera dicts with keys:
                - K: (3, 3) intrinsic matrix
                - R: (3, 3) rotation matrix
                - T: (3, 1) translation vector
        """
        self.cameras = []
        for cam in cameras:
            cam_dict = {
                'K': torch.from_numpy(cam['K'].T).float().to(self.device),
                'R': torch.from_numpy(cam['R'].T).float().to(self.device),
                # T is in mm, convert to m to match mesh coordinates (trans/1000, scale/1000)
                'T': torch.from_numpy(cam['T'] / 1000.0).float().to(self.device).squeeze(),
            }
            self.cameras.append(cam_dict)

        self.n_views = len(self.cameras)
        print(f"TextureSampler: {self.n_views} cameras set")

    def project_vertices(
        self,
        vertices: torch.Tensor,
        view_idx: int,
    ) -> torch.Tensor:
        """
        Project 3D vertices to 2D image coordinates.

        Args:
            vertices: (N, 3) vertex positions
            view_idx: Camera view index

        Returns:
            proj_2d: (N, 2) projected 2D coordinates
        """
        cam = self.cameras[view_idx]
        K, R, T = cam['K'], cam['R'], cam['T']

        # World to camera: X_cam = X_world @ R + T
        # Note: K, R are stored as original.T in set_cameras(), so use them directly
        # (not .T again, which would double-transpose back to original)
        vertices_cam = vertices @ R + T

        # Camera to image: x = X_cam @ K
        vertices_proj = vertices_cam @ K

        # Normalize by depth
        proj_2d = vertices_proj[:, :2] / (vertices_proj[:, 2:3] + 1e-8)

        return proj_2d

    def compute_visibility(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        view_idx: int,
    ) -> torch.Tensor:
        """
        Compute per-vertex visibility for a view.

        Uses normal-based visibility (backface culling).

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices
            view_idx: Camera view index

        Returns:
            visibility: (N,) visibility weights [0, 1]
        """
        cam = self.cameras[view_idx]
        R, T = cam['R'], cam['T']

        # Camera position in world coordinates
        # cam_pos = -R.T @ T
        cam_pos = -R @ T

        # Compute vertex normals
        normals = self._compute_vertex_normals(vertices, faces)

        # View direction: from vertex to camera
        view_dirs = F.normalize(cam_pos.unsqueeze(0) - vertices, dim=-1)

        # Visibility: cos(angle) between normal and view direction
        visibility = (normals * view_dirs).sum(dim=-1)
        visibility = torch.clamp(visibility, 0, 1)

        return visibility

    def sample_texture_single_view(
        self,
        image: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        view_idx: int,
        use_visibility: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample texture colors from a single view.

        Args:
            image: (3, H, W) RGB image
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices
            view_idx: Camera view index
            use_visibility: Whether to weight by visibility

        Returns:
            colors: (N, 3) sampled RGB colors per vertex
            weights: (N,) confidence weights
        """
        _, H, W = image.shape

        # Project vertices to image
        proj_2d = self.project_vertices(vertices, view_idx)

        # Normalize to [-1, 1] for grid_sample
        proj_norm = torch.zeros_like(proj_2d)
        proj_norm[:, 0] = (proj_2d[:, 0] / W) * 2 - 1  # x
        proj_norm[:, 1] = (proj_2d[:, 1] / H) * 2 - 1  # y

        # Sample colors using bilinear interpolation
        # grid_sample expects (N, H_out, W_out, 2)
        grid = proj_norm.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        image_batch = image.unsqueeze(0)  # (1, 3, H, W)

        sampled = F.grid_sample(
            image_batch,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )  # (1, 3, N, 1)

        colors = sampled.squeeze(0).squeeze(-1).T  # (N, 3)

        # Compute weights
        weights = torch.ones(vertices.shape[0], device=self.device)

        # Visibility weighting
        if use_visibility:
            visibility = self.compute_visibility(vertices, faces, view_idx)
            weights = weights * visibility

        # Out-of-bounds penalty
        in_bounds = (
            (proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < W) &
            (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < H)
        ).float()
        weights = weights * in_bounds

        return colors, weights

    def sample_texture_multi_view(
        self,
        images: List[torch.Tensor],
        vertices: torch.Tensor,
        faces: torch.Tensor,
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample and fuse textures from multiple views.

        Uses visibility-weighted averaging.

        Args:
            images: List of (3, H, W) RGB images
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices
            masks: Optional list of (H, W) foreground masks

        Returns:
            fused_colors: (N, 3) fused RGB colors
            total_weights: (N,) total confidence weights
        """
        N = vertices.shape[0]
        accumulated_colors = torch.zeros(N, 3, device=self.device)
        accumulated_weights = torch.zeros(N, device=self.device)

        for view_idx, image in enumerate(images):
            # Sample from this view
            colors, weights = self.sample_texture_single_view(
                image, vertices, faces, view_idx,
            )

            # Apply mask if provided
            if masks is not None and masks[view_idx] is not None:
                mask = masks[view_idx]
                proj_2d = self.project_vertices(vertices, view_idx)
                mask_weights = self._sample_mask(mask, proj_2d)
                weights = weights * mask_weights

            # Accumulate
            accumulated_colors += colors * weights.unsqueeze(1)
            accumulated_weights += weights

        # Normalize
        valid = accumulated_weights > 0
        fused_colors = torch.zeros_like(accumulated_colors)
        fused_colors[valid] = accumulated_colors[valid] / accumulated_weights[valid].unsqueeze(1)

        return fused_colors, accumulated_weights

    def _sample_mask(
        self,
        mask: torch.Tensor,
        proj_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample mask values at projected coordinates.

        Args:
            mask: (H, W) foreground mask
            proj_2d: (N, 2) projected coordinates

        Returns:
            mask_values: (N,) sampled mask values
        """
        H, W = mask.shape

        # Normalize coordinates
        proj_norm = torch.zeros_like(proj_2d)
        proj_norm[:, 0] = (proj_2d[:, 0] / W) * 2 - 1
        proj_norm[:, 1] = (proj_2d[:, 1] / H) * 2 - 1

        # Sample
        grid = proj_norm.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        mask_batch = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        sampled = F.grid_sample(
            mask_batch.float(),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

        return sampled.squeeze()

    def _compute_vertex_normals(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-vertex normals.
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = torch.cross(e1, e2, dim=-1)
        face_normals = F.normalize(face_normals, dim=-1)

        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(
                0,
                faces[:, i:i+1].expand(-1, 3),
                face_normals,
            )

        vertex_normals = F.normalize(vertex_normals, dim=-1)
        return vertex_normals


class TextureAccumulator:
    """
    Accumulate texture samples across frames.

    Maintains running average with confidence weighting.
    """

    def __init__(
        self,
        n_vertices: int,
        device: str = 'cuda',
    ):
        """
        Args:
            n_vertices: Number of mesh vertices
            device: Torch device
        """
        self.device = torch.device(device)
        self.n_vertices = n_vertices

        # Running accumulators
        self.color_sum = torch.zeros(n_vertices, 3, device=self.device)
        self.weight_sum = torch.zeros(n_vertices, device=self.device)
        self.n_frames = 0

    def add_frame(
        self,
        colors: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """
        Add texture samples from a frame.

        Args:
            colors: (N, 3) RGB colors
            weights: (N,) confidence weights
        """
        self.color_sum += colors * weights.unsqueeze(1)
        self.weight_sum += weights
        self.n_frames += 1

    def get_texture(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get accumulated texture.

        Returns:
            colors: (N, 3) averaged RGB colors
            confidence: (N,) confidence (number of valid samples)
        """
        valid = self.weight_sum > 0
        colors = torch.zeros(self.n_vertices, 3, device=self.device)
        colors[valid] = self.color_sum[valid] / self.weight_sum[valid].unsqueeze(1)

        confidence = self.weight_sum / (self.n_frames + 1e-8)

        return colors, confidence

    def reset(self) -> None:
        """Reset accumulators."""
        self.color_sum.zero_()
        self.weight_sum.zero_()
        self.n_frames = 0
