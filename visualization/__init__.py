"""
Backward compatibility wrapper for visualization module.

The visualization module has been moved to mammal_ext.visualization.
This wrapper maintains backward compatibility with existing code.

Migration:
    # Old (deprecated)
    from visualization import MeshVisualizer

    # New (recommended)
    from mammal_ext.visualization import MeshVisualizer
"""

# Re-export everything from mammal_ext.visualization
from mammal_ext.visualization import *
from mammal_ext.visualization import (
    VisualizationConfig,
    CameraConfig,
    RenderOutput,
    CameraPathGenerator,
    CameraPose,
    compute_mesh_bounds,
    TexturedMeshRenderer,
    create_textured_renderer,
    VideoGenerator,
    create_grid_video,
    create_orbit_video,
    frames_to_video,
    pack_images,
    MeshVisualizer,
    visualize_fitting_results,
    render_preview,
    RERUN_AVAILABLE,
)

# Conditional rerun exports
if RERUN_AVAILABLE:
    from mammal_ext.visualization import (
        RerunExporter,
        MOUSE_KEYPOINT_LABELS,
        MOUSE_BONES,
        MOUSE_KEYPOINT_COLORS,
        get_keypoint_colors,
    )

import warnings
warnings.warn(
    "visualization module has moved to mammal_ext.visualization. "
    "Please update imports: from mammal_ext.visualization import ...",
    DeprecationWarning,
    stacklevel=2
)
