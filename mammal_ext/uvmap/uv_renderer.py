"""
UV Renderer Module

Differentiable UV space rendering for texture mapping.
Based on PyTorch3D rasterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from tqdm import tqdm
import logging

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    TexturesUV,
    TexturesVertex,
)

# Setup logger
logger = logging.getLogger(__name__)


class UVRenderer(nn.Module):
    """
    Differentiable UV space renderer.

    Renders mesh attributes (position, normal, color) into UV space.
    Supports both forward (3D->UV) and inverse (UV->3D) mapping.
    """

    def __init__(
        self,
        uv_size: int = 512,
        device: str = 'cuda',
    ):
        """
        Args:
            uv_size: Resolution of UV map (uv_size x uv_size)
            device: Torch device
        """
        super().__init__()
        self.uv_size = uv_size
        self.device = torch.device(device)

        # UV space rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=uv_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,  # Naive rasterization for UV space
        )

    def load_uv_data(
        self,
        vertices_path: str,
        textures_path: str,
        faces_vert_path: str,
        faces_tex_path: str,
    ) -> None:
        """
        Load UV coordinates and face indices from files.

        Args:
            vertices_path: Path to vertices.txt
            textures_path: Path to textures.txt (UV coordinates)
            faces_vert_path: Path to faces_vert.txt
            faces_tex_path: Path to faces_tex.txt
        """
        # Load data
        self.vertices_3d = np.loadtxt(vertices_path)
        self.uv_coords = np.loadtxt(textures_path)
        self.faces_vert = np.loadtxt(faces_vert_path, dtype=np.int64)
        self.faces_tex = np.loadtxt(faces_tex_path, dtype=np.int64)

        # Convert to tensors
        self.vertices_3d_th = torch.from_numpy(self.vertices_3d).float().to(self.device)
        self.uv_coords_th = torch.from_numpy(self.uv_coords).float().to(self.device)
        self.faces_vert_th = torch.from_numpy(self.faces_vert).long().to(self.device)
        self.faces_tex_th = torch.from_numpy(self.faces_tex).long().to(self.device)

        # Statistics
        self.n_vertices = self.vertices_3d.shape[0]
        self.n_uv_coords = self.uv_coords.shape[0]
        self.n_faces = self.faces_vert.shape[0]

        print(f"UV Data loaded:")
        print(f"  Vertices: {self.n_vertices}")
        print(f"  UV coords: {self.n_uv_coords}")
        print(f"  Faces: {self.n_faces}")

    def create_uv_mesh(self) -> Meshes:
        """
        Create mesh in UV space for rasterization.

        UV coordinates are treated as XY positions, Z=0.
        This allows rendering attributes into UV texture space.

        Returns:
            PyTorch3D Meshes object in UV space
        """
        # UV coords to 3D (x=u, y=v, z=0)
        uv_3d = torch.zeros(self.n_uv_coords, 3, device=self.device)
        uv_3d[:, 0] = self.uv_coords_th[:, 0]  # U -> X
        uv_3d[:, 1] = self.uv_coords_th[:, 1]  # V -> Y

        # Scale to [-1, 1] for rasterization
        uv_3d[:, :2] = uv_3d[:, :2] * 2 - 1

        # Create mesh with UV faces
        mesh = Meshes(
            verts=[uv_3d],
            faces=[self.faces_tex_th],
        )

        return mesh

    def render_position_map(
        self,
        vertices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render 3D positions into UV space.

        Args:
            vertices: (N, 3) vertex positions

        Returns:
            position_map: (3, H, W) position map in UV space
        """
        # Get positions per UV vertex via face mapping
        # Each UV vertex corresponds to a 3D vertex through face indices
        position_per_uv = self._map_vertex_attr_to_uv(vertices)

        # Render to UV space
        position_map = self._render_vertex_colors(position_per_uv)

        return position_map

    def render_normal_map(
        self,
        vertices: torch.Tensor,
        faces: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render vertex normals into UV space.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices (optional, uses stored if None)

        Returns:
            normal_map: (3, H, W) normal map in UV space
        """
        if faces is None:
            faces = self.faces_vert_th

        # Compute vertex normals
        normals = self._compute_vertex_normals(vertices, faces)

        # Map to UV and render
        normals_per_uv = self._map_vertex_attr_to_uv(normals)
        normal_map = self._render_vertex_colors(normals_per_uv)

        return normal_map

    def render_visibility_map(
        self,
        vertices: torch.Tensor,
        camera_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render visibility mask in UV space.

        Visibility is determined by dot product of normal and view direction.

        Args:
            vertices: (N, 3) vertex positions
            camera_position: (3,) camera position

        Returns:
            visibility_map: (1, H, W) visibility mask [0, 1]
        """
        # Compute normals
        normals = self._compute_vertex_normals(vertices, self.faces_vert_th)

        # View direction per vertex
        view_dirs = F.normalize(camera_position.unsqueeze(0) - vertices, dim=-1)

        # Visibility: dot(normal, view_dir) > 0
        visibility = (normals * view_dirs).sum(dim=-1, keepdim=True)
        visibility = torch.clamp(visibility, 0, 1)

        # Map to UV and render
        vis_per_uv = self._map_vertex_attr_to_uv(visibility)
        visibility_map = self._render_vertex_colors(vis_per_uv)

        return visibility_map

    def _map_vertex_attr_to_uv(
        self,
        vertex_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map vertex attributes to UV vertices via face correspondence.

        Args:
            vertex_attr: (N_vert, C) vertex attributes

        Returns:
            uv_attr: (N_uv, C) attributes per UV vertex
        """
        # Build mapping from UV index to vertex index
        # Using face correspondence: faces_tex[f, i] -> faces_vert[f, i]
        uv_to_vert = torch.zeros(self.n_uv_coords, dtype=torch.long, device=self.device)

        for f in range(self.n_faces):
            for i in range(3):
                uv_idx = self.faces_tex_th[f, i]
                vert_idx = self.faces_vert_th[f, i]
                uv_to_vert[uv_idx] = vert_idx

        # Gather attributes
        uv_attr = vertex_attr[uv_to_vert]

        return uv_attr

    def _render_vertex_colors(
        self,
        vertex_colors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render vertex colors to UV texture map.

        Args:
            vertex_colors: (N_uv, C) colors per UV vertex

        Returns:
            texture_map: (C, H, W) rendered texture
        """
        # Create UV mesh
        uv_mesh = self.create_uv_mesh()

        # Add vertex colors as texture
        textures = TexturesVertex(verts_features=[vertex_colors])
        uv_mesh.textures = textures

        # Simple orthographic projection for UV space
        # Rasterize and interpolate vertex colors
        texture_map = self._rasterize_uv_mesh(uv_mesh, vertex_colors)

        return texture_map

    def _rasterize_uv_mesh(
        self,
        mesh: Meshes,
        vertex_colors: torch.Tensor,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Rasterize UV mesh with vertex colors (vectorized implementation).

        Uses batch-parallel barycentric interpolation for speed.

        Args:
            mesh: UV space mesh
            vertex_colors: (N_uv, C) per-vertex colors
            show_progress: Whether to show progress bar

        Returns:
            texture: (C, H, W) rasterized texture
        """
        H = W = self.uv_size
        C = vertex_colors.shape[1]

        logger.info(f"Rasterizing {self.n_faces} faces to {H}x{W} texture...")

        # Initialize output
        texture = torch.zeros(C, H, W, device=self.device)
        weight_sum = torch.zeros(1, H, W, device=self.device)

        # Get UV coordinates
        uv = self.uv_coords_th.clone()

        # Process faces in batches for memory efficiency
        batch_size = 1000
        n_batches = (self.n_faces + batch_size - 1) // batch_size

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Rasterizing triangles", unit="batch")

        for batch_idx in iterator:
            start_f = batch_idx * batch_size
            end_f = min((batch_idx + 1) * batch_size, self.n_faces)

            self._rasterize_face_batch(
                texture, weight_sum,
                uv, vertex_colors,
                start_f, end_f,
                H, W,
            )

        # Normalize by weights
        mask = weight_sum > 0
        texture = torch.where(
            mask.expand(C, -1, -1),
            texture / (weight_sum + 1e-8),
            texture,
        )

        coverage = mask.float().mean().item() * 100
        logger.info(f"Rasterization complete. Coverage: {coverage:.1f}%")

        return texture

    def _rasterize_face_batch(
        self,
        texture: torch.Tensor,
        weight_sum: torch.Tensor,
        uv: torch.Tensor,
        vertex_colors: torch.Tensor,
        start_f: int,
        end_f: int,
        H: int,
        W: int,
    ) -> None:
        """
        Rasterize a batch of faces using vectorized operations.
        """
        for f in range(start_f, end_f):
            # Get UV triangle vertices
            uv_idx = self.faces_tex_th[f]
            tri_uv = uv[uv_idx]  # (3, 2)
            tri_colors = vertex_colors[uv_idx]  # (3, C)

            # Rasterize triangle (vectorized)
            self._rasterize_triangle_fast(
                texture, weight_sum,
                tri_uv, tri_colors,
                H, W,
            )

    def _rasterize_triangle_fast(
        self,
        texture: torch.Tensor,
        weight_sum: torch.Tensor,
        tri_uv: torch.Tensor,
        tri_colors: torch.Tensor,
        H: int,
        W: int,
    ) -> None:
        """
        Rasterize a single triangle using vectorized barycentric coordinates.
        """
        # Convert UV to pixel coordinates
        px = (tri_uv[:, 0] * W).long().clamp(0, W - 1)
        py = (tri_uv[:, 1] * H).long().clamp(0, H - 1)

        # Bounding box
        min_x = max(0, px.min().item())
        max_x = min(W - 1, px.max().item())
        min_y = max(0, py.min().item())
        max_y = min(H - 1, py.max().item())

        if max_x < min_x or max_y < min_y:
            return

        # Create grid of pixel coordinates in bounding box
        xs = torch.arange(min_x, max_x + 1, device=self.device)
        ys = torch.arange(min_y, max_y + 1, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        # Normalize to UV space
        grid_u = grid_x.float() / W
        grid_v = grid_y.float() / H
        points = torch.stack([grid_u, grid_v], dim=-1)  # (Ny, Nx, 2)

        # Compute barycentric coordinates for all points at once
        bary = self._barycentric_batch(points.reshape(-1, 2), tri_uv)  # (Ny*Nx, 3)
        bary = bary.reshape(len(ys), len(xs), 3)

        # Check which points are inside triangle
        inside = (bary >= -1e-5).all(dim=-1) & (bary <= 1 + 1e-5).all(dim=-1)

        if not inside.any():
            return

        # Interpolate colors for inside points
        # bary: (Ny, Nx, 3), tri_colors: (3, C)
        colors = torch.einsum('yxb,bc->yxc', bary, tri_colors)

        # Update texture at valid positions
        for i, y in enumerate(range(min_y, max_y + 1)):
            for j, x in enumerate(range(min_x, max_x + 1)):
                if inside[i, j]:
                    texture[:, y, x] += colors[i, j]
                    weight_sum[:, y, x] += 1

    def _barycentric_batch(
        self,
        points: torch.Tensor,
        tri: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute barycentric coordinates for batch of points.

        Args:
            points: (N, 2) points
            tri: (3, 2) triangle vertices

        Returns:
            bary: (N, 3) barycentric coordinates
        """
        v0 = tri[2] - tri[0]  # (2,)
        v1 = tri[1] - tri[0]  # (2,)
        v2 = points - tri[0]  # (N, 2)

        dot00 = (v0 * v0).sum()
        dot01 = (v0 * v1).sum()
        dot02 = (v2 * v0).sum(dim=-1)  # (N,)
        dot11 = (v1 * v1).sum()
        dot12 = (v2 * v1).sum(dim=-1)  # (N,)

        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom  # (N,)
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom  # (N,)

        return torch.stack([1 - u - v, v, u], dim=-1)  # (N, 3)

    def _compute_vertex_normals(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-vertex normals.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices

        Returns:
            normals: (N, 3) per-vertex normals
        """
        # Get face vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Face normals
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = torch.cross(e1, e2, dim=-1)
        face_normals = F.normalize(face_normals, dim=-1)

        # Accumulate to vertices
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(
                0,
                faces[:, i:i+1].expand(-1, 3),
                face_normals,
            )

        # Normalize
        vertex_normals = F.normalize(vertex_normals, dim=-1)

        return vertex_normals

    def get_uv_mask(self) -> torch.Tensor:
        """
        Get binary mask of valid UV regions.

        Returns:
            mask: (1, H, W) binary mask
        """
        # Render ones to get coverage
        ones = torch.ones(self.n_uv_coords, 1, device=self.device)
        mask = self._render_vertex_colors(ones)
        mask = (mask > 0).float()

        return mask


def create_uv_renderer(
    uv_size: int = 512,
    model_dir: str = 'mouse_model/mouse_txt',
    device: str = 'cuda',
) -> UVRenderer:
    """
    Factory function to create and initialize UV renderer.

    Args:
        uv_size: UV map resolution
        model_dir: Directory containing mesh files
        device: Torch device

    Returns:
        Initialized UVRenderer
    """
    import os

    renderer = UVRenderer(uv_size=uv_size, device=device)
    renderer.load_uv_data(
        vertices_path=os.path.join(model_dir, 'vertices.txt'),
        textures_path=os.path.join(model_dir, 'textures.txt'),
        faces_vert_path=os.path.join(model_dir, 'faces_vert.txt'),
        faces_tex_path=os.path.join(model_dir, 'faces_tex.txt'),
    )

    return renderer
