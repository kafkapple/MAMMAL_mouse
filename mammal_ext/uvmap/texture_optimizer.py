"""
Texture Optimizer Module

Photometric optimization for texture refinement.
Based on differentiable rendering with multi-view supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextureOptConfig:
    """Configuration for texture optimization."""
    # Texture parameters
    uv_size: int = 512
    texture_channels: int = 3

    # Optimization
    lr: float = 0.01
    n_iters: int = 100

    # Loss weights
    w_photo: float = 1.0          # Photometric loss
    w_tv: float = 1e-3            # Total variation regularization
    w_smooth: float = 1e-4        # Smoothness regularization

    # View selection
    min_visibility: float = 0.3   # Minimum visibility threshold


class TextureModel(nn.Module):
    """
    Learnable texture map.

    Supports:
    - Direct RGB texture
    - Residual texture (base + learnable delta)
    """

    def __init__(
        self,
        uv_size: int = 512,
        channels: int = 3,
        init_texture: Optional[torch.Tensor] = None,
        mode: str = 'direct',  # 'direct' or 'residual'
        device: str = 'cuda',
    ):
        """
        Args:
            uv_size: UV map resolution
            channels: Number of channels (3 for RGB)
            init_texture: Initial texture (C, H, W) or None
            mode: 'direct' for learnable texture, 'residual' for base + delta
            device: Torch device
        """
        super().__init__()
        self.uv_size = uv_size
        self.channels = channels
        self.mode = mode
        self.device = torch.device(device)

        if init_texture is not None:
            # Use provided initial texture
            if mode == 'residual':
                self.register_buffer('base_texture', init_texture.to(self.device))
                self.delta_texture = nn.Parameter(
                    torch.zeros_like(init_texture).to(self.device)
                )
            else:
                self.texture = nn.Parameter(init_texture.clone().to(self.device))
        else:
            # Initialize with gray
            init = torch.ones(channels, uv_size, uv_size, device=self.device) * 0.5
            if mode == 'residual':
                self.register_buffer('base_texture', init)
                self.delta_texture = nn.Parameter(torch.zeros_like(init))
            else:
                self.texture = nn.Parameter(init)

    def forward(self) -> torch.Tensor:
        """
        Get current texture map.

        Returns:
            texture: (C, H, W) texture map, clamped to [0, 1]
        """
        if self.mode == 'residual':
            texture = self.base_texture + self.delta_texture
        else:
            texture = self.texture

        return torch.clamp(texture, 0, 1)

    def sample_at_uv(
        self,
        uv_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample texture at UV coordinates.

        Args:
            uv_coords: (N, 2) UV coordinates in [0, 1]

        Returns:
            colors: (N, 3) RGB colors
        """
        texture = self.forward()

        # Convert to grid_sample format [-1, 1]
        grid = uv_coords * 2 - 1
        grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

        sampled = F.grid_sample(
            texture.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

        return sampled.squeeze(0).squeeze(-1).T  # (N, 3)


class TextureOptimizer:
    """
    Optimize texture map using photometric loss.

    Renders textured mesh to each view and minimizes
    RGB difference with ground truth images.
    """

    def __init__(
        self,
        config: TextureOptConfig,
        device: str = 'cuda',
    ):
        """
        Args:
            config: Optimization configuration
            device: Torch device
        """
        self.config = config
        self.device = torch.device(device)

        # Models (set via initialize)
        self.texture_model = None
        self.optimizer = None

        # Data (set via set_data)
        self.uv_coords = None
        self.faces_tex = None
        self.faces_vert = None

    def initialize(
        self,
        init_texture: Optional[torch.Tensor] = None,
        mode: str = 'direct',
    ) -> None:
        """
        Initialize texture model and optimizer.

        Args:
            init_texture: Initial texture or None
            mode: 'direct' or 'residual'
        """
        self.texture_model = TextureModel(
            uv_size=self.config.uv_size,
            channels=self.config.texture_channels,
            init_texture=init_texture,
            mode=mode,
            device=self.device,
        )

        self.optimizer = torch.optim.Adam(
            self.texture_model.parameters(),
            lr=self.config.lr,
        )

    def set_mesh_data(
        self,
        uv_coords: torch.Tensor,
        faces_tex: torch.Tensor,
        faces_vert: torch.Tensor,
    ) -> None:
        """
        Set mesh UV data.

        Args:
            uv_coords: (N_uv, 2) UV coordinates
            faces_tex: (F, 3) UV face indices
            faces_vert: (F, 3) vertex face indices
        """
        self.uv_coords = uv_coords.to(self.device)
        self.faces_tex = faces_tex.to(self.device)
        self.faces_vert = faces_vert.to(self.device)

        # Build UV to vertex mapping
        self._build_uv_vertex_map()

    def _build_uv_vertex_map(self) -> None:
        """Build mapping from UV indices to vertex indices."""
        n_uv = self.uv_coords.shape[0]
        self.uv_to_vert = torch.zeros(n_uv, dtype=torch.long, device=self.device)

        for f in range(self.faces_tex.shape[0]):
            for i in range(3):
                uv_idx = self.faces_tex[f, i]
                vert_idx = self.faces_vert[f, i]
                self.uv_to_vert[uv_idx] = vert_idx

    def compute_photometric_loss(
        self,
        vertices: torch.Tensor,
        image: torch.Tensor,
        camera: Dict,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute photometric loss for a single view.

        Args:
            vertices: (N, 3) vertex positions
            image: (3, H, W) target RGB image
            camera: Camera dict with K, R, T
            mask: (H, W) foreground mask (optional)

        Returns:
            loss: Scalar photometric loss
        """
        _, H, W = image.shape

        # Get texture colors at vertex positions (via UV mapping)
        vertex_colors = self._get_vertex_colors()

        # Project vertices
        proj_2d = self._project_vertices(vertices, camera)

        # Sample target colors at projected positions
        target_colors = self._sample_image(image, proj_2d)

        # Compute visibility
        visibility = self._compute_visibility(vertices, camera)

        # Apply mask if provided
        if mask is not None:
            mask_values = self._sample_image(mask.unsqueeze(0), proj_2d).squeeze()
            visibility = visibility * mask_values

        # In-bounds check
        in_bounds = (
            (proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < W) &
            (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < H)
        ).float()
        visibility = visibility * in_bounds

        # Photometric loss (L1)
        diff = (vertex_colors - target_colors).abs()
        weighted_diff = diff * visibility.unsqueeze(1)

        # Normalize by valid pixels
        n_valid = visibility.sum() + 1e-8
        loss = weighted_diff.sum() / n_valid

        return loss

    def compute_regularization_loss(self) -> Dict[str, torch.Tensor]:
        """
        Compute texture regularization losses.

        Returns:
            losses: Dict of regularization losses
        """
        texture = self.texture_model.forward()
        losses = {}

        # Total Variation loss (smoothness)
        if self.config.w_tv > 0:
            tv_h = (texture[:, 1:, :] - texture[:, :-1, :]).pow(2).mean()
            tv_w = (texture[:, :, 1:] - texture[:, :, :-1]).pow(2).mean()
            losses['tv'] = (tv_h + tv_w) * self.config.w_tv

        # L2 smoothness (penalize extreme values)
        if self.config.w_smooth > 0:
            losses['smooth'] = (texture - 0.5).pow(2).mean() * self.config.w_smooth

        return losses

    def optimize_step(
        self,
        vertices: torch.Tensor,
        images: List[torch.Tensor],
        cameras: List[Dict],
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Single optimization step.

        Args:
            vertices: (N, 3) vertex positions
            images: List of (3, H, W) target images
            cameras: List of camera dicts
            masks: Optional list of (H, W) masks

        Returns:
            losses: Dict of loss values
        """
        self.optimizer.zero_grad()

        # Photometric loss across views
        photo_loss = torch.tensor(0.0, device=self.device)
        n_views = len(images)

        for i, (image, camera) in enumerate(zip(images, cameras)):
            mask = masks[i] if masks is not None else None
            view_loss = self.compute_photometric_loss(vertices, image, camera, mask)
            photo_loss = photo_loss + view_loss

        photo_loss = photo_loss / n_views * self.config.w_photo

        # Regularization
        reg_losses = self.compute_regularization_loss()
        reg_loss = sum(reg_losses.values())

        # Total loss
        total_loss = photo_loss + reg_loss

        # Backward and step
        total_loss.backward()
        self.optimizer.step()

        # Return loss values
        losses = {
            'total': total_loss.item(),
            'photo': photo_loss.item(),
        }
        losses.update({k: v.item() for k, v in reg_losses.items()})

        return losses

    def optimize(
        self,
        vertices: torch.Tensor,
        images: List[torch.Tensor],
        cameras: List[Dict],
        masks: Optional[List[torch.Tensor]] = None,
        callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Full optimization loop.

        Args:
            vertices: (N, 3) vertex positions
            images: List of (3, H, W) target images
            cameras: List of camera dicts
            masks: Optional list of masks
            callback: Optional callback(iter, losses) called each iteration

        Returns:
            optimized_texture: (C, H, W) optimized texture map
        """
        for i in range(self.config.n_iters):
            losses = self.optimize_step(vertices, images, cameras, masks)

            if callback is not None:
                callback(i, losses)

            if i % 10 == 0:
                print(f"  Iter {i:3d}: loss={losses['total']:.4f} "
                      f"(photo={losses['photo']:.4f})")

        return self.texture_model.forward().detach()

    def _get_vertex_colors(self) -> torch.Tensor:
        """Get texture colors at vertex positions via UV mapping."""
        # Sample texture at UV coords
        uv_colors = self.texture_model.sample_at_uv(self.uv_coords)

        # Map to vertices
        n_verts = self.uv_to_vert.max() + 1
        vertex_colors = torch.zeros(n_verts, 3, device=self.device)
        vertex_counts = torch.zeros(n_verts, device=self.device)

        for uv_idx, vert_idx in enumerate(self.uv_to_vert):
            vertex_colors[vert_idx] += uv_colors[uv_idx]
            vertex_counts[vert_idx] += 1

        vertex_counts = vertex_counts.clamp(min=1)
        vertex_colors = vertex_colors / vertex_counts.unsqueeze(1)

        return vertex_colors

    def _project_vertices(
        self,
        vertices: torch.Tensor,
        camera: Dict,
    ) -> torch.Tensor:
        """Project vertices to 2D."""
        K = camera['K']
        R = camera['R']
        T = camera['T']

        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K.T).float().to(self.device)
            R = torch.from_numpy(R.T).float().to(self.device)
            T = torch.from_numpy(T).float().to(self.device).squeeze()

        vertices_cam = vertices @ R.T + T
        vertices_proj = vertices_cam @ K.T
        proj_2d = vertices_proj[:, :2] / (vertices_proj[:, 2:3] + 1e-8)

        return proj_2d

    def _sample_image(
        self,
        image: torch.Tensor,
        proj_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Sample image at projected coordinates."""
        C, H, W = image.shape

        proj_norm = torch.zeros_like(proj_2d)
        proj_norm[:, 0] = (proj_2d[:, 0] / W) * 2 - 1
        proj_norm[:, 1] = (proj_2d[:, 1] / H) * 2 - 1

        grid = proj_norm.unsqueeze(0).unsqueeze(2)
        sampled = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

        return sampled.squeeze(0).squeeze(-1).T

    def _compute_visibility(
        self,
        vertices: torch.Tensor,
        camera: Dict,
    ) -> torch.Tensor:
        """Compute vertex visibility for camera."""
        R = camera['R']
        T = camera['T']

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R.T).float().to(self.device)
            T = torch.from_numpy(T).float().to(self.device).squeeze()

        # Camera position
        cam_pos = -R @ T

        # Vertex normals
        normals = self._compute_vertex_normals(vertices)

        # View direction
        view_dirs = F.normalize(cam_pos.unsqueeze(0) - vertices, dim=-1)

        # Visibility
        visibility = (normals * view_dirs).sum(dim=-1)
        visibility = torch.clamp(visibility, 0, 1)

        # Threshold
        visibility = torch.where(
            visibility > self.config.min_visibility,
            visibility,
            torch.zeros_like(visibility),
        )

        return visibility

    def _compute_vertex_normals(
        self,
        vertices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vertex normals."""
        v0 = vertices[self.faces_vert[:, 0]]
        v1 = vertices[self.faces_vert[:, 1]]
        v2 = vertices[self.faces_vert[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = torch.cross(e1, e2, dim=-1)
        face_normals = F.normalize(face_normals, dim=-1)

        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(
                0,
                self.faces_vert[:, i:i+1].expand(-1, 3),
                face_normals,
            )

        return F.normalize(vertex_normals, dim=-1)

    def get_texture(self) -> torch.Tensor:
        """Get current texture map."""
        return self.texture_model.forward().detach()
