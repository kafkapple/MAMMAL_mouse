"""
Textured Mesh Renderer Module

Render UV-textured meshes using PyTorch3D and pyrender backends.
Supports both differentiable (PyTorch3D) and high-quality (pyrender) rendering.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path

# Set up headless rendering before importing pyrender
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2


class TexturedMeshRenderer:
    """
    Render textured mesh with UV mapping.

    Supports two backends:
    - pyrender: High-quality rendering with shadows (default for visualization)
    - pytorch3d: Differentiable rendering (for optimization)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        device: str = 'cuda',
        backend: str = 'pyrender',
    ):
        """
        Args:
            image_size: (width, height) of rendered images
            device: Torch device for PyTorch3D backend
            backend: 'pyrender' or 'pytorch3d'
        """
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.backend = backend

        # Mesh data (loaded later)
        self.vertices = None
        self.faces = None
        self.uv_coords = None
        self.faces_tex = None
        self.texture_image = None

        # Backend-specific setup
        self._renderer = None

    def load_mesh_data(
        self,
        model_dir: str,
    ) -> None:
        """
        Load mesh topology from model directory.

        Args:
            model_dir: Directory containing vertices.txt, faces_vert.txt, textures.txt, faces_tex.txt
        """
        model_path = Path(model_dir)

        self.vertices_template = np.loadtxt(model_path / 'vertices.txt')
        self.faces = np.loadtxt(model_path / 'faces_vert.txt', dtype=np.int64)
        self.uv_coords = np.loadtxt(model_path / 'textures.txt')
        self.faces_tex = np.loadtxt(model_path / 'faces_tex.txt', dtype=np.int64)

        # Convert to tensors
        self.faces_th = torch.from_numpy(self.faces).long().to(self.device)
        self.uv_coords_th = torch.from_numpy(self.uv_coords).float().to(self.device)
        self.faces_tex_th = torch.from_numpy(self.faces_tex).long().to(self.device)

        print(f"Mesh loaded: {self.vertices_template.shape[0]} vertices, {self.faces.shape[0]} faces")
        print(f"UV coords: {self.uv_coords.shape[0]}, UV faces: {self.faces_tex.shape[0]}")

    def load_texture(
        self,
        texture_path: str,
    ) -> None:
        """
        Load UV texture image.

        Args:
            texture_path: Path to texture image (PNG/JPG)
        """
        # Load image (BGR -> RGB)
        img = cv2.imread(texture_path)
        if img is None:
            raise FileNotFoundError(f"Texture not found: {texture_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        self.texture_image = img.astype(np.float32) / 255.0
        self.texture_th = torch.from_numpy(self.texture_image).to(self.device)

        print(f"Texture loaded: {self.texture_image.shape}")

    def sample_vertex_colors(
        self,
        vertices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sample texture colors at each vertex's UV coordinate.

        This is used for backends that don't support UV textures directly
        (e.g., Rerun's Mesh3D uses per-vertex colors).

        Args:
            vertices: (N, 3) vertex positions (not used, for API consistency)

        Returns:
            vertex_colors: (N, 3) RGB colors in [0, 255] uint8
        """
        if self.texture_image is None:
            raise RuntimeError("Texture not loaded. Call load_texture() first.")

        # Build UV-to-vertex mapping
        n_vertices = self.vertices_template.shape[0]
        n_uv = self.uv_coords.shape[0]

        # Map each 3D vertex to its UV coordinate
        # faces_tex[f, i] gives UV index for face f, corner i
        # faces[f, i] gives vertex index for face f, corner i
        vertex_uvs = np.zeros((n_vertices, 2), dtype=np.float32)
        vertex_counts = np.zeros(n_vertices, dtype=np.int32)

        for f in range(self.faces.shape[0]):
            for i in range(3):
                vert_idx = self.faces[f, i]
                uv_idx = self.faces_tex[f, i]
                vertex_uvs[vert_idx] += self.uv_coords[uv_idx]
                vertex_counts[vert_idx] += 1

        # Average UV for vertices with multiple mappings
        valid = vertex_counts > 0
        vertex_uvs[valid] /= vertex_counts[valid, None]

        # Sample texture at UV coordinates
        H, W = self.texture_image.shape[:2]

        # UV to pixel coordinates (UV origin is bottom-left in most conventions)
        u = vertex_uvs[:, 0]
        v = 1.0 - vertex_uvs[:, 1]  # Flip V

        px = np.clip((u * W).astype(np.int32), 0, W - 1)
        py = np.clip((v * H).astype(np.int32), 0, H - 1)

        # Sample colors
        colors = self.texture_image[py, px]  # (N, 3) in [0, 1]

        # Convert to uint8
        colors_uint8 = (colors * 255).astype(np.uint8)

        return colors_uint8

    def render_pyrender(
        self,
        vertices: np.ndarray,
        camera_pose: np.ndarray,
        fov: float = 45.0,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render textured mesh using pyrender backend.

        Args:
            vertices: (N, 3) vertex positions
            camera_pose: (4, 4) camera-to-world transform
            fov: Field of view in degrees
            background_color: RGB background in [0, 1]

        Returns:
            image: (H, W, 3) rendered RGB image in uint8
        """
        import trimesh
        import pyrender
        from pyrender.constants import RenderFlags

        width, height = self.image_size

        # Create trimesh with UV texture
        mesh_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=self.faces,
            process=False,
        )

        # Apply texture using vertex colors (simpler than UV for trimesh)
        vertex_colors = self.sample_vertex_colors()  # (N, 3) uint8

        # Convert to RGBA with full opacity for pyrender compatibility
        vertex_colors_rgba = np.concatenate([
            vertex_colors,
            np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
        ], axis=1)
        mesh_trimesh.visual.vertex_colors = vertex_colors_rgba

        # Create pyrender mesh (smooth=False to preserve vertex colors)
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

        # Create scene
        scene = pyrender.Scene(bg_color=np.array(background_color))
        scene.add(mesh_pyrender)

        # Add lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))

        fill_light = pyrender.DirectionalLight(color=[0.7, 0.7, 0.8], intensity=1.5)
        fill_pose = np.eye(4)
        fill_pose[:3, :3] = self._rotation_matrix_y(np.pi / 3)
        scene.add(fill_light, pose=fill_pose)

        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.radians(fov))
        scene.add(camera, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(width, height)
        flags = RenderFlags.SHADOWS_DIRECTIONAL
        color, depth = renderer.render(scene, flags=flags)
        renderer.delete()

        return color

    def render_pytorch3d(
        self,
        vertices: torch.Tensor,
        camera_R: torch.Tensor,
        camera_T: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        """
        Render textured mesh using PyTorch3D backend (differentiable).

        Args:
            vertices: (N, 3) or (B, N, 3) vertex positions
            camera_R: (3, 3) or (B, 3, 3) rotation matrix
            camera_T: (3,) or (B, 3) translation
            K: Optional (3, 3) or (B, 3, 3) intrinsic matrix
            background_color: RGB background in [0, 1]

        Returns:
            images: (B, H, W, 4) rendered RGBA images
        """
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            HardPhongShader,
            TexturesUV,
            PointLights,
            PerspectiveCameras,
        )

        # Ensure batch dimension
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        if camera_R.dim() == 2:
            camera_R = camera_R.unsqueeze(0)
        if camera_T.dim() == 1:
            camera_T = camera_T.unsqueeze(0)

        batch_size = vertices.shape[0]
        width, height = self.image_size

        # Create TexturesUV
        # texture_th: (H, W, 3) -> (1, H, W, 3)
        texture_maps = self.texture_th.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # UV coordinates: (N_uv, 2) -> (B, N_uv, 2)
        verts_uvs = self.uv_coords_th.unsqueeze(0).expand(batch_size, -1, -1)

        # UV face indices: (F, 3) -> (B, F, 3)
        faces_uvs = self.faces_tex_th.unsqueeze(0).expand(batch_size, -1, -1)

        textures = TexturesUV(
            maps=texture_maps,
            faces_uvs=faces_uvs,
            verts_uvs=verts_uvs,
        )

        # Create mesh
        faces_batch = self.faces_th.unsqueeze(0).expand(batch_size, -1, -1)
        meshes = Meshes(
            verts=list(vertices),
            faces=list(faces_batch),
            textures=textures,
        )

        # Setup cameras
        cameras = PerspectiveCameras(
            R=camera_R,
            T=camera_T,
            device=self.device,
        )

        # Setup renderer
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(
            location=[[0.0, 2.0, 2.0]],
            device=self.device,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
            ),
        )

        # Render
        images = renderer(meshes)

        return images

    def render(
        self,
        vertices: Union[np.ndarray, torch.Tensor],
        camera_pose: Optional[np.ndarray] = None,
        camera_R: Optional[torch.Tensor] = None,
        camera_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Unified render interface.

        Args:
            vertices: Vertex positions
            camera_pose: (4, 4) camera-to-world transform (pyrender)
            camera_R, camera_T: Camera parameters (pytorch3d)
            **kwargs: Additional backend-specific arguments

        Returns:
            Rendered image(s)
        """
        if self.backend == 'pyrender':
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            if camera_pose is None:
                raise ValueError("camera_pose required for pyrender backend")
            return self.render_pyrender(vertices, camera_pose, **kwargs)

        elif self.backend == 'pytorch3d':
            if isinstance(vertices, np.ndarray):
                vertices = torch.from_numpy(vertices).float().to(self.device)
            if camera_R is None or camera_T is None:
                raise ValueError("camera_R and camera_T required for pytorch3d backend")
            return self.render_pytorch3d(vertices, camera_R, camera_T, **kwargs)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def render_multiview(
        self,
        vertices: np.ndarray,
        camera_poses: List[np.ndarray],
        names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Render mesh from multiple viewpoints.

        Args:
            vertices: (N, 3) vertex positions
            camera_poses: List of (4, 4) camera-to-world transforms
            names: Optional names for each view
            **kwargs: Render arguments

        Returns:
            List of (name, image) tuples
        """
        results = []

        for i, pose in enumerate(camera_poses):
            name = names[i] if names else f'view_{i:03d}'
            image = self.render_pyrender(vertices, pose, **kwargs)
            results.append((name, image))

        return results

    @staticmethod
    def _rotation_matrix_y(angle: float) -> np.ndarray:
        """Create rotation matrix around Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ])


def create_textured_renderer(
    model_dir: str,
    texture_path: str,
    image_size: Tuple[int, int] = (1024, 1024),
    device: str = 'cuda',
    backend: str = 'pyrender',
) -> TexturedMeshRenderer:
    """
    Factory function to create and initialize textured renderer.

    Args:
        model_dir: Directory containing mesh files
        texture_path: Path to UV texture image
        image_size: Render resolution
        device: Torch device
        backend: 'pyrender' or 'pytorch3d'

    Returns:
        Initialized TexturedMeshRenderer
    """
    renderer = TexturedMeshRenderer(
        image_size=image_size,
        device=device,
        backend=backend,
    )

    renderer.load_mesh_data(model_dir)
    renderer.load_texture(texture_path)

    return renderer
