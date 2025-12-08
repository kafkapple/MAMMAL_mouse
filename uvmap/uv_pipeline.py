"""
UV Map Generation Pipeline

End-to-end pipeline for generating UV texture maps from
multi-view RGB images and fitted mesh sequences.
"""

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from .uv_renderer import UVRenderer, create_uv_renderer
from .texture_sampler import TextureSampler, TextureAccumulator
from .texture_optimizer import TextureOptimizer, TextureOptConfig, TextureModel


@dataclass
class UVPipelineConfig:
    """Configuration for UV map pipeline."""
    # Paths
    model_dir: str = 'mouse_model/mouse_txt'
    result_dir: str = ''  # Fitting result directory

    # UV map settings
    uv_size: int = 512

    # Frame selection
    start_frame: int = 0
    end_frame: int = -1  # -1 for all frames
    frame_interval: int = 1

    # Multi-view fusion
    use_visibility_weighting: bool = True
    visibility_threshold: float = 0.3

    # Optimization (optional refinement)
    do_optimization: bool = False
    opt_iters: int = 50
    opt_lr: float = 0.01
    opt_w_tv: float = 1e-3

    # Output
    output_dir: str = ''
    save_intermediate: bool = False


class UVMapPipeline:
    """
    Complete pipeline for UV texture map generation.

    Pipeline stages:
    1. Load mesh model and camera parameters
    2. Load fitted mesh parameters for each frame
    3. Sample textures from multi-view images
    4. Accumulate across frames
    5. (Optional) Photometric optimization
    6. Export final UV map
    """

    def __init__(
        self,
        config: UVPipelineConfig,
        device: str = 'cuda',
    ):
        """
        Args:
            config: Pipeline configuration
            device: Torch device
        """
        self.config = config
        self.device = torch.device(device)

        # Components (initialized in setup)
        self.uv_renderer = None
        self.texture_sampler = None
        self.texture_accumulator = None
        self.texture_optimizer = None

        # Data
        self.cameras = []
        self.n_views = 0
        self.frames = []

    def setup(self) -> None:
        """Initialize all pipeline components."""
        print("Setting up UV Map Pipeline...")

        # 1. UV Renderer
        self.uv_renderer = create_uv_renderer(
            uv_size=self.config.uv_size,
            model_dir=self.config.model_dir,
            device=str(self.device),
        )

        # 2. Texture Sampler
        self.texture_sampler = TextureSampler(
            uv_size=self.config.uv_size,
            device=str(self.device),
        )

        # 3. Load cameras from result directory
        self._load_cameras()

        # 4. Texture Accumulator
        self.texture_accumulator = TextureAccumulator(
            n_vertices=self.uv_renderer.n_vertices,
            device=str(self.device),
        )

        # 5. (Optional) Texture Optimizer
        if self.config.do_optimization:
            opt_config = TextureOptConfig(
                uv_size=self.config.uv_size,
                n_iters=self.config.opt_iters,
                lr=self.config.opt_lr,
                w_tv=self.config.opt_w_tv,
            )
            self.texture_optimizer = TextureOptimizer(
                config=opt_config,
                device=str(self.device),
            )
            self.texture_optimizer.set_mesh_data(
                uv_coords=self.uv_renderer.uv_coords_th,
                faces_tex=self.uv_renderer.faces_tex_th,
                faces_vert=self.uv_renderer.faces_vert_th,
            )

        # 6. Determine frames to process
        self._find_frames()

        # 7. Output directory
        if not self.config.output_dir:
            self.config.output_dir = os.path.join(self.config.result_dir, 'uvmap')
        os.makedirs(self.config.output_dir, exist_ok=True)

        print(f"Pipeline setup complete:")
        print(f"  Views: {self.n_views}")
        print(f"  Frames: {len(self.frames)}")
        print(f"  UV size: {self.config.uv_size}")

    def _load_cameras(self) -> None:
        """Load camera parameters from data directory."""
        # Try to load from config.yaml in result_dir
        config_path = os.path.join(self.config.result_dir, 'config.yaml')
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            data_dir = cfg.get('data', {}).get('data_dir', '')
            views_to_use = cfg.get('data', {}).get('views_to_use', [0,1,2,3,4,5])
        else:
            # Default paths
            data_dir = 'data/examples/markerless_mouse_1_nerf'
            views_to_use = [0,1,2,3,4,5]

        # Resolve data_dir path
        if not os.path.isabs(data_dir):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_root, data_dir)

        # Load camera pickle
        cam_path = os.path.join(data_dir, 'new_cam.pkl')
        if os.path.exists(cam_path):
            with open(cam_path, 'rb') as f:
                cams_dict = pickle.load(f)

            self.cameras = []
            for view_id in views_to_use:
                cam = cams_dict[view_id].copy()
                # Adjust formats
                if cam['T'].ndim == 1:
                    cam['T'] = cam['T'].reshape(-1, 1)
                self.cameras.append(cam)

            self.texture_sampler.set_cameras(self.cameras)
            self.n_views = len(self.cameras)
            self.views_to_use = views_to_use
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(f"Camera file not found: {cam_path}")

    def _find_frames(self) -> None:
        """Find available frames in result directory."""
        params_dir = os.path.join(self.config.result_dir, 'params')
        if not os.path.exists(params_dir):
            raise FileNotFoundError(f"Params directory not found: {params_dir}")

        # Find step_2 parameter files
        import glob
        param_files = sorted(glob.glob(os.path.join(params_dir, 'step_2_frame_*.pkl')))

        self.frames = []
        for pf in param_files:
            # Extract frame number from filename
            basename = os.path.basename(pf)
            frame_num = int(basename.split('_')[-1].replace('.pkl', ''))
            self.frames.append(frame_num)

        # Apply frame range
        if self.config.end_frame > 0:
            self.frames = [f for f in self.frames
                          if self.config.start_frame <= f < self.config.end_frame]

        # Apply interval
        if self.config.frame_interval > 1:
            self.frames = self.frames[::self.config.frame_interval]

    def load_frame_data(
        self,
        frame_idx: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Load mesh parameters and images for a frame.

        Args:
            frame_idx: Frame number

        Returns:
            vertices: (N, 3) mesh vertices
            images: List of (3, H, W) RGB images
            masks: List of (H, W) foreground masks
        """
        # Load mesh parameters
        param_path = os.path.join(
            self.config.result_dir, 'params',
            f'step_2_frame_{frame_idx:06d}.pkl'
        )
        with open(param_path, 'rb') as f:
            params = pickle.load(f)

        # Forward pass to get vertices
        vertices = self._forward_mesh(params)

        # Load images and masks
        images = []
        masks = []

        for view_id in self.views_to_use:
            # Load RGB image from video
            video_path = os.path.join(self.data_dir, 'videos_undist', f'{view_id}.mp4')
            img = self._load_video_frame(video_path, frame_idx)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) / 255.0
            images.append(img_tensor)

            # Load mask from video
            mask_path = os.path.join(self.data_dir, 'simpleclick_undist', f'{view_id}.mp4')
            mask = self._load_video_frame(mask_path, frame_idx, grayscale=True)
            mask_tensor = torch.from_numpy(mask).float().to(self.device) / 255.0
            masks.append(mask_tensor)

        return vertices, images, masks

    def _forward_mesh(
        self,
        params: Dict,
    ) -> torch.Tensor:
        """
        Compute mesh vertices from parameters.

        Args:
            params: Dict of mesh parameters

        Returns:
            vertices: (N, 3) vertex positions
        """
        # Import body model
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from articulation_th import ArticulationTorch

        # Initialize body model (lazy)
        if not hasattr(self, '_body_model'):
            self._body_model = ArticulationTorch()

        # Forward pass
        V, J = self._body_model.forward(
            params['thetas'],
            params['bone_lengths'],
            params['rotation'],
            params['trans'] / 1000,  # mm to m
            params['scale'] / 1000,
            params['chest_deformer'],
        )

        return V[0]  # Remove batch dimension

    def _load_video_frame(
        self,
        video_path: str,
        frame_idx: int,
        grayscale: bool = False,
    ) -> np.ndarray:
        """Load a single frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        if grayscale:
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]

        return frame

    def process_frame(
        self,
        frame_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single frame.

        Args:
            frame_idx: Frame number

        Returns:
            colors: (N, 3) vertex colors
            weights: (N,) confidence weights
        """
        # Load data
        vertices, images, masks = self.load_frame_data(frame_idx)

        # Sample textures from multi-view
        colors, weights = self.texture_sampler.sample_texture_multi_view(
            images=images,
            vertices=vertices,
            faces=self.uv_renderer.faces_vert_th,
            masks=masks if self.config.use_visibility_weighting else None,
        )

        return colors, weights

    def run(self) -> torch.Tensor:
        """
        Run full pipeline.

        Returns:
            texture_map: (3, H, W) final UV texture map
        """
        print("\n=== UV Map Generation Pipeline ===")

        # Setup if not done
        if self.uv_renderer is None:
            self.setup()

        # Stage 1: Accumulate textures across frames
        print("\n[Stage 1] Accumulating textures from frames...")
        self.texture_accumulator.reset()

        for frame_idx in tqdm(self.frames, desc="Processing frames"):
            try:
                colors, weights = self.process_frame(frame_idx)
                self.texture_accumulator.add_frame(colors, weights)

                # Save intermediate
                if self.config.save_intermediate and frame_idx % 10 == 0:
                    self._save_intermediate(frame_idx)

            except Exception as e:
                print(f"  Warning: Failed to process frame {frame_idx}: {e}")
                continue

        # Get accumulated vertex colors
        vertex_colors, confidence = self.texture_accumulator.get_texture()
        print(f"  Accumulated {self.texture_accumulator.n_frames} frames")
        print(f"  Coverage: {(confidence > 0).float().mean() * 100:.1f}%")

        # Stage 2: Render to UV space
        print("\n[Stage 2] Rendering to UV space...")
        texture_map = self.uv_renderer.render_position_map(vertex_colors)

        # Actually we need to render colors, not positions
        # Map vertex colors to UV space
        colors_uv = self.uv_renderer._map_vertex_attr_to_uv(vertex_colors)
        texture_map = self.uv_renderer._render_vertex_colors(colors_uv)

        # Stage 3: (Optional) Photometric optimization
        if self.config.do_optimization:
            print("\n[Stage 3] Photometric optimization...")
            texture_map = self._optimize_texture(texture_map)

        # Stage 4: Save results
        print("\n[Stage 4] Saving results...")
        self._save_results(texture_map, confidence)

        print("\n=== Pipeline Complete ===")
        return texture_map

    def _optimize_texture(
        self,
        init_texture: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine texture using photometric optimization.

        Args:
            init_texture: (3, H, W) initial texture

        Returns:
            optimized_texture: (3, H, W) refined texture
        """
        # Initialize optimizer with current texture
        self.texture_optimizer.initialize(
            init_texture=init_texture,
            mode='residual',
        )

        # Select subset of frames for optimization
        opt_frames = self.frames[::max(1, len(self.frames) // 10)]

        for frame_idx in tqdm(opt_frames, desc="Optimizing"):
            vertices, images, masks = self.load_frame_data(frame_idx)

            # Convert cameras to dict format
            cameras = []
            for cam in self.cameras:
                cameras.append({
                    'K': cam['K'],
                    'R': cam['R'],
                    'T': cam['T'],
                })

            # Optimization step
            _ = self.texture_optimizer.optimize_step(
                vertices=vertices,
                images=images,
                cameras=cameras,
                masks=masks,
            )

        return self.texture_optimizer.get_texture()

    def _save_intermediate(
        self,
        frame_idx: int,
    ) -> None:
        """Save intermediate results."""
        vertex_colors, _ = self.texture_accumulator.get_texture()
        colors_uv = self.uv_renderer._map_vertex_attr_to_uv(vertex_colors)
        texture = self.uv_renderer._render_vertex_colors(colors_uv)

        # Save as image
        img = (texture.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        path = os.path.join(self.config.output_dir, f'texture_frame_{frame_idx:06d}.png')
        cv2.imwrite(path, img)

    def _save_results(
        self,
        texture_map: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        """Save final results."""
        # Texture image (PNG)
        img = (texture_map.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        texture_path = os.path.join(self.config.output_dir, 'texture_final.png')
        cv2.imwrite(texture_path, img)
        print(f"  Texture saved: {texture_path}")

        # Confidence map
        conf_img = (confidence.cpu().numpy() * 255).astype(np.uint8)
        conf_path = os.path.join(self.config.output_dir, 'confidence.png')
        cv2.imwrite(conf_path, conf_img)

        # UV mask
        uv_mask = self.uv_renderer.get_uv_mask()
        mask_img = (uv_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        mask_path = os.path.join(self.config.output_dir, 'uv_mask.png')
        cv2.imwrite(mask_path, mask_img)

        # Texture tensor (for further processing)
        tensor_path = os.path.join(self.config.output_dir, 'texture.pt')
        torch.save({
            'texture': texture_map.cpu(),
            'confidence': confidence.cpu(),
            'uv_size': self.config.uv_size,
        }, tensor_path)
        print(f"  Tensor saved: {tensor_path}")


def run_uvmap_pipeline(
    result_dir: str,
    uv_size: int = 512,
    do_optimization: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Convenience function to run UV map pipeline.

    Args:
        result_dir: Path to fitting result directory
        uv_size: UV map resolution
        do_optimization: Whether to run photometric optimization
        **kwargs: Additional config options

    Returns:
        texture_map: (3, H, W) UV texture map
    """
    config = UVPipelineConfig(
        result_dir=result_dir,
        uv_size=uv_size,
        do_optimization=do_optimization,
        **kwargs,
    )

    pipeline = UVMapPipeline(config)
    pipeline.setup()

    return pipeline.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate UV texture map from fitting results')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to fitting result directory')
    parser.add_argument('--uv_size', type=int, default=512,
                       help='UV map resolution')
    parser.add_argument('--optimize', action='store_true',
                       help='Run photometric optimization')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1)
    parser.add_argument('--interval', type=int, default=1)

    args = parser.parse_args()

    texture = run_uvmap_pipeline(
        result_dir=args.result_dir,
        uv_size=args.uv_size,
        do_optimization=args.optimize,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_interval=args.interval,
    )

    print(f"Generated texture shape: {texture.shape}")
