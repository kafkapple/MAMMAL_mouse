"""Configuration utilities for MAMMAL mouse fitting."""

from mammal_ext.config.gpu import configure_gpu, get_default_gpu, GPU_DEFAULTS
from mammal_ext.config.loss_weights import get_loss_weights, DEFAULT_LOSS_WEIGHTS
from mammal_ext.config.keypoint_weights import get_keypoint_weights, MAMMAL_PAPER_KEYPOINT_WEIGHTS

__all__ = [
    # GPU
    'configure_gpu',
    'get_default_gpu',
    'GPU_DEFAULTS',
    # Loss weights
    'get_loss_weights',
    'DEFAULT_LOSS_WEIGHTS',
    # Keypoint weights
    'get_keypoint_weights',
    'MAMMAL_PAPER_KEYPOINT_WEIGHTS',
]
