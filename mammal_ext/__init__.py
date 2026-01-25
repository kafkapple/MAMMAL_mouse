"""MAMMAL Mouse Extension Package.

This package contains extensions and utilities built on top of the original
MAMMAL mouse fitting code. The goal is to keep original code modifications
minimal while providing configurable and modular enhancements.

Modules:
    config: Configuration utilities (GPU, loss weights, keypoints)
    fitting: Fitting pipeline extensions (debug grid, etc.)
    visualization: Mesh visualization, video generation, Rerun export
    preprocessing: Mask processing, keypoint estimation, SAM inference
    uvmap: UV texture mapping pipeline
"""

from mammal_ext.config import configure_gpu, get_loss_weights, get_keypoint_weights
from mammal_ext.fitting import DebugGridCollector, compress_existing_debug_folder

__version__ = "0.2.0"

__all__ = [
    # Config
    'configure_gpu',
    'get_loss_weights',
    'get_keypoint_weights',
    # Fitting
    'DebugGridCollector',
    'compress_existing_debug_folder',
]
