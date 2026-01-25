"""MAMMAL Mouse Extension Package.

This package contains extensions and utilities built on top of the original
MAMMAL mouse fitting code. The goal is to keep original code modifications
minimal while providing configurable and modular enhancements.

Modules:
    config: Configuration utilities (GPU, loss weights, keypoints)
    fitting: Fitting pipeline extensions
"""

from mammal_ext.config import configure_gpu, get_loss_weights, get_keypoint_weights

__version__ = "0.1.0"

__all__ = [
    'configure_gpu',
    'get_loss_weights',
    'get_keypoint_weights',
]
