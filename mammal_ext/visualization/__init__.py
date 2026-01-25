"""
Visualization Module for MAMMAL Mouse

UV-textured mesh rendering, Rerun RRD export, and video generation.

Usage:
    # Simple API
    from visualization import visualize_fitting_results, render_preview

    # Full visualization pipeline
    outputs = visualize_fitting_results(
        result_dir="results/fitting/experiment",
        view_modes=['orbit', 'fixed'],
        save_rrd=True,
        save_video=True,
    )

    # Quick preview
    preview = render_preview(
        result_dir="results/fitting/experiment",
        frame_idx=0,
        output_path="preview.png"
    )

    # Custom configuration
    from visualization import MeshVisualizer, VisualizationConfig

    config = VisualizationConfig(
        result_dir="results/fitting/experiment",
        view_modes=['orbit'],
        orbit_frames=60,
        image_size=(512, 512),
    )
    visualizer = MeshVisualizer(config)
    outputs = visualizer.run()

CLI Usage:
    python -m visualization.mesh_visualizer \\
        --result_dir results/fitting/experiment \\
        --view_modes orbit fixed \\
        --save_rrd --save_video
"""

from .config import VisualizationConfig, CameraConfig, RenderOutput
from .camera_paths import (
    CameraPathGenerator,
    CameraPose,
    compute_mesh_bounds,
)
from .textured_renderer import (
    TexturedMeshRenderer,
    create_textured_renderer,
)
from .video_generator import (
    VideoGenerator,
    create_grid_video,
    create_orbit_video,
    frames_to_video,
    pack_images,
)
from .mesh_visualizer import (
    MeshVisualizer,
    visualize_fitting_results,
    render_preview,
)

# Optional Rerun exporter (requires rerun-sdk)
try:
    from .rerun_exporter import (
        RerunExporter,
        MOUSE_KEYPOINT_LABELS,
        MOUSE_BONES,
        MOUSE_KEYPOINT_COLORS,
        get_keypoint_colors,
    )
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

__all__ = [
    # Config
    'VisualizationConfig',
    'CameraConfig',
    'RenderOutput',
    # Camera
    'CameraPathGenerator',
    'CameraPose',
    'compute_mesh_bounds',
    # Renderer
    'TexturedMeshRenderer',
    'create_textured_renderer',
    # Video
    'VideoGenerator',
    'create_grid_video',
    'create_orbit_video',
    'frames_to_video',
    'pack_images',
    # Main
    'MeshVisualizer',
    'visualize_fitting_results',
    'render_preview',
    # Rerun (optional)
    'RerunExporter',
    'MOUSE_KEYPOINT_LABELS',
    'MOUSE_BONES',
    'MOUSE_KEYPOINT_COLORS',
    'get_keypoint_colors',
    'RERUN_AVAILABLE',
]
