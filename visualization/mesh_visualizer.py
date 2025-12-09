"""
Mesh Visualizer Module

Main orchestrator for UV-textured mesh visualization.
Integrates rendering, Rerun export, and video generation.

Usage:
    python -m visualization.mesh_visualizer \
        --result_dir results/fitting/experiment \
        --view_mode orbit \
        --save_rrd \
        --save_video

    # Or as library:
    from visualization import MeshVisualizer, VisualizationConfig

    config = VisualizationConfig(result_dir="results/fitting/exp")
    visualizer = MeshVisualizer(config)
    outputs = visualizer.run()
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import VisualizationConfig
from .camera_paths import CameraPathGenerator, CameraPose, compute_mesh_bounds
from .textured_renderer import TexturedMeshRenderer, create_textured_renderer
from .video_generator import VideoGenerator, pack_images


class MeshVisualizer:
    """
    Main orchestrator for UV-textured mesh visualization.

    Workflow:
    1. Load fitting results (params, mesh, UV texture)
    2. Setup renderer and camera paths
    3. For each frame:
       - Load/compute mesh vertices
       - Render from multiple viewpoints
       - Export to Rerun (if enabled)
    4. Generate output videos (if enabled)
    5. Save RRD file (if enabled)
    """

    def __init__(self, config: VisualizationConfig):
        """
        Args:
            config: Visualization configuration
        """
        self.config = config

        # Components (initialized in setup)
        self.renderer: Optional[TexturedMeshRenderer] = None
        self.body_model = None
        self.rerun_exporter = None

        # Data
        self.params_files: List[str] = []
        self.frame_indices: List[int] = []

        # Camera paths
        self.camera_poses: Dict[str, List[CameraPose]] = {}

    def setup(self) -> None:
        """Initialize all components."""
        print("\n=== Mesh Visualizer Setup ===")

        # Validate paths
        if not os.path.exists(self.config.result_dir):
            raise FileNotFoundError(f"Result directory not found: {self.config.result_dir}")

        # Find texture
        if self.config.texture_path is None:
            texture_path = os.path.join(self.config.result_dir, 'uvmap', 'texture_final.png')
            if not os.path.exists(texture_path):
                raise FileNotFoundError(f"Texture not found: {texture_path}")
            self.config.texture_path = texture_path

        # Create output directory
        output_dir = self.config.ensure_output_dir()
        print(f"Output directory: {output_dir}")

        # Load body model
        print("Loading body model...")
        from articulation_th import ArticulationTorch
        self.body_model = ArticulationTorch()

        # Setup renderer
        print("Setting up renderer...")
        self.renderer = create_textured_renderer(
            model_dir=self.config.model_dir,
            texture_path=self.config.texture_path,
            image_size=self.config.image_size,
            backend='pyrender',
        )

        # Find parameter files
        self._find_param_files()

        # Setup camera paths
        self._setup_cameras()

        # Setup Rerun exporter
        if self.config.save_rrd:
            print("Setting up Rerun exporter...")
            from .rerun_exporter import RerunExporter
            rrd_path = os.path.join(self.config.output_dir, 'visualization.rrd')
            self.rerun_exporter = RerunExporter(rrd_path, self.config.rerun_app_name)
            self.rerun_exporter.init()

        print("Setup complete!")

    def _find_param_files(self) -> None:
        """Find and filter parameter files."""
        params_dir = self.config.params_dir
        pattern = os.path.join(params_dir, "step_2_frame_*.pkl")
        self.params_files = sorted(glob.glob(pattern))

        if not self.params_files:
            # Try step_1 fallback
            pattern = os.path.join(params_dir, "step_1_frame_*.pkl")
            self.params_files = sorted(glob.glob(pattern))

        if not self.params_files:
            raise FileNotFoundError(f"No parameter files found in {params_dir}")

        # Extract frame indices
        self.frame_indices = []
        for pf in self.params_files:
            basename = os.path.basename(pf)
            # step_2_frame_000000.pkl -> 000000
            frame_str = basename.split('frame_')[1].replace('.pkl', '')
            self.frame_indices.append(int(frame_str))

        # Apply frame range filter
        start = self.config.start_frame
        end = self.config.end_frame if self.config.end_frame > 0 else len(self.params_files)
        interval = self.config.frame_interval

        indices = range(start, min(end, len(self.params_files)), interval)
        self.params_files = [self.params_files[i] for i in indices]
        self.frame_indices = [self.frame_indices[i] for i in indices]

        print(f"Found {len(self.params_files)} frames to process")

    def _setup_cameras(self) -> None:
        """Setup camera paths based on configuration."""
        # Load first frame to get mesh bounds
        vertices = self._load_vertices(self.params_files[0])
        center, scale = compute_mesh_bounds(vertices)

        cam_gen = CameraPathGenerator(center, scale)

        for mode in self.config.view_modes:
            if mode == 'orbit':
                self.camera_poses['orbit'] = cam_gen.orbit_360(
                    n_frames=self.config.orbit_frames,
                    elevation=self.config.orbit_elevation,
                    distance_factor=self.config.orbit_distance * 5,  # Scale adjustment
                )
            elif mode == 'fixed':
                self.camera_poses['fixed'] = cam_gen.fixed_views(
                    views=self.config.fixed_views,
                )
            elif mode == 'novel' and self.config.novel_azimuths:
                self.camera_poses['novel'] = cam_gen.novel_views(
                    azimuths=self.config.novel_azimuths,
                    elevations=self.config.novel_elevations,
                )

        print(f"Camera modes: {list(self.camera_poses.keys())}")

    def _load_vertices(self, params_path: str) -> np.ndarray:
        """Load and compute vertices from parameters."""
        with open(params_path, 'rb') as f:
            params = pickle.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to tensors
        for k, v in params.items():
            if not isinstance(v, torch.Tensor):
                params[k] = torch.tensor(v, dtype=torch.float32, device=device)

        # Forward pass through body model
        V, J = self.body_model.forward(
            params["thetas"],
            params["bone_lengths"],
            params["rotation"],
            params["trans"] / 1000,  # mm to m
            params["scale"] / 1000,
            params.get("chest_deformer", torch.zeros(1, 1, device=device)),
        )

        vertices = V[0].detach().cpu().numpy()
        return vertices

    def _load_keypoints(self, params_path: str) -> np.ndarray:
        """Load and compute 3D keypoints from parameters."""
        # First load vertices (sets up body model state)
        self._load_vertices(params_path)

        # Get keypoints
        keypoints = self.body_model.forward_keypoints22()[0].detach().cpu().numpy()
        return keypoints

    def run(self) -> Dict[str, str]:
        """
        Run visualization pipeline.

        Returns:
            Dictionary of output paths
        """
        if self.renderer is None:
            self.setup()

        print("\n=== Running Visualization Pipeline ===")
        outputs = {}

        # Collect frames for video generation
        video_frames: Dict[str, List[np.ndarray]] = {}

        # Process each frame
        for i, (params_path, frame_idx) in enumerate(tqdm(
            zip(self.params_files, self.frame_indices),
            total=len(self.params_files),
            desc="Processing frames",
        )):
            # Load mesh
            vertices = self._load_vertices(params_path)

            # Get vertex colors from texture
            vertex_colors = self.renderer.sample_vertex_colors()

            # Render from each camera mode
            for mode, poses in self.camera_poses.items():
                if mode == 'orbit':
                    # For orbit, use frame index to select camera pose
                    orbit_idx = i % len(poses)
                    pose = poses[orbit_idx]
                    cam_matrix = pose.to_pyrender_pose()
                    image = self.renderer.render_pyrender(vertices, cam_matrix)

                    if mode not in video_frames:
                        video_frames[mode] = []
                    video_frames[mode].append(image)

                    # Log to Rerun
                    if self.rerun_exporter:
                        self.rerun_exporter.log_image(frame_idx, image, f"orbit_{orbit_idx:03d}")

                elif mode in ['fixed', 'novel']:
                    # Render all fixed/novel views
                    for pose in poses:
                        cam_matrix = pose.to_pyrender_pose()
                        image = self.renderer.render_pyrender(vertices, cam_matrix)

                        key = f"{mode}_{pose.name}"
                        if key not in video_frames:
                            video_frames[key] = []
                        video_frames[key].append(image)

                        if self.rerun_exporter:
                            self.rerun_exporter.log_image(frame_idx, image, pose.name)

            # Log mesh to Rerun
            if self.rerun_exporter:
                self.rerun_exporter.log_mesh(
                    frame_idx,
                    vertices,
                    self.body_model.faces_vert_np,
                    vertex_colors,
                )

                # Log keypoints if enabled
                if self.config.show_keypoints:
                    keypoints = self._load_keypoints(params_path)
                    self.rerun_exporter.log_keypoints(frame_idx, keypoints)

                    if self.config.show_skeleton:
                        from .rerun_exporter import MOUSE_BONES
                        self.rerun_exporter.log_skeleton(frame_idx, keypoints, MOUSE_BONES)

        # Save videos
        if self.config.save_video:
            print("\n=== Generating Videos ===")
            for mode, frames in video_frames.items():
                video_path = os.path.join(self.config.output_dir, f"{mode}.mp4")
                with VideoGenerator(video_path, fps=self.config.video_fps) as gen:
                    for frame in frames:
                        gen.add_frame(frame)
                outputs[f"video_{mode}"] = video_path

        # Save Rerun
        if self.rerun_exporter:
            outputs["rrd"] = self.rerun_exporter.save()

        print("\n=== Visualization Complete ===")
        for key, path in outputs.items():
            print(f"  {key}: {path}")

        return outputs

    def render_single_frame(
        self,
        frame_idx: int = 0,
        view_mode: str = 'fixed',
    ) -> Dict[str, np.ndarray]:
        """
        Render single frame for quick preview.

        Args:
            frame_idx: Index in frame list (not actual frame number)
            view_mode: 'fixed', 'orbit', or specific view name

        Returns:
            Dictionary of view_name -> rendered image
        """
        if self.renderer is None:
            self.setup()

        params_path = self.params_files[frame_idx]
        vertices = self._load_vertices(params_path)

        results = {}

        if view_mode in self.camera_poses:
            poses = self.camera_poses[view_mode]
            for pose in poses:
                cam_matrix = pose.to_pyrender_pose()
                image = self.renderer.render_pyrender(vertices, cam_matrix)
                results[pose.name] = image
        else:
            # Single fixed view
            cam_gen = CameraPathGenerator(*compute_mesh_bounds(vertices))
            poses = cam_gen.fixed_views([view_mode])
            if poses:
                cam_matrix = poses[0].to_pyrender_pose()
                image = self.renderer.render_pyrender(vertices, cam_matrix)
                results[view_mode] = image

        return results


def visualize_fitting_results(
    result_dir: str,
    view_modes: List[str] = None,
    start_frame: int = 0,
    end_frame: int = -1,
    save_rrd: bool = True,
    save_video: bool = True,
    **kwargs,
) -> Dict[str, str]:
    """
    Main entry point for visualization.

    Args:
        result_dir: Fitting result directory
        view_modes: List of view modes ('orbit', 'fixed', 'novel')
        start_frame: Starting frame
        end_frame: Ending frame (-1 = all)
        save_rrd: Save Rerun RRD file
        save_video: Save MP4 videos
        **kwargs: Additional config options

    Returns:
        Dictionary of output paths
    """
    config = VisualizationConfig(
        result_dir=result_dir,
        view_modes=view_modes or ['orbit', 'fixed'],
        start_frame=start_frame,
        end_frame=end_frame,
        save_rrd=save_rrd,
        save_video=save_video,
        **kwargs,
    )

    visualizer = MeshVisualizer(config)
    return visualizer.run()


def render_preview(
    result_dir: str,
    frame_idx: int = 0,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Quick preview render of single frame.

    Args:
        result_dir: Fitting result directory
        frame_idx: Frame index to render
        output_path: Optional path to save image

    Returns:
        Rendered image grid
    """
    config = VisualizationConfig(
        result_dir=result_dir,
        view_modes=['fixed'],
        save_rrd=False,
        save_video=False,
    )

    visualizer = MeshVisualizer(config)
    visualizer.setup()

    results = visualizer.render_single_frame(frame_idx, 'fixed')

    # Pack into grid
    images = list(results.values())
    grid = pack_images(images, max_cols=2)

    if output_path:
        import cv2
        cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Preview saved: {output_path}")

    return grid


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UV-textured mesh visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--result_dir", type=str, required=True,
                       help="Path to fitting result directory")
    parser.add_argument("--view_modes", type=str, nargs='+',
                       default=['orbit', 'fixed'],
                       choices=['orbit', 'fixed', 'novel'],
                       help="View modes to render")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Starting frame index")
    parser.add_argument("--end_frame", type=int, default=-1,
                       help="Ending frame index (-1 = all)")
    parser.add_argument("--frame_interval", type=int, default=1,
                       help="Process every N-th frame")
    parser.add_argument("--save_rrd", action="store_true", default=True,
                       help="Save Rerun RRD file")
    parser.add_argument("--save_video", action="store_true", default=True,
                       help="Save MP4 videos")
    parser.add_argument("--no_rrd", action="store_true",
                       help="Disable Rerun output")
    parser.add_argument("--no_video", action="store_true",
                       help="Disable video output")
    parser.add_argument("--fps", type=int, default=30,
                       help="Video frame rate")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024],
                       help="Render image size (width height)")
    parser.add_argument("--show_keypoints", action="store_true",
                       help="Show 3D keypoints")
    parser.add_argument("--show_skeleton", action="store_true",
                       help="Show skeleton bones")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: result_dir/visualization)")

    args = parser.parse_args()

    config = VisualizationConfig(
        result_dir=args.result_dir,
        view_modes=args.view_modes,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_interval=args.frame_interval,
        save_rrd=args.save_rrd and not args.no_rrd,
        save_video=args.save_video and not args.no_video,
        video_fps=args.fps,
        image_size=tuple(args.image_size),
        show_keypoints=args.show_keypoints,
        show_skeleton=args.show_skeleton,
        output_dir=args.output_dir,
    )

    visualizer = MeshVisualizer(config)
    outputs = visualizer.run()

    return outputs


if __name__ == "__main__":
    main()
