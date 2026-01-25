"""Keypoint weight configuration utilities.

This module handles loading and parsing keypoint-specific weights from config.
Extracted from fitter_articulation.py MouseFitter.__init__().

Supports:
- Full 22-keypoint mode (default)
- Sparse keypoint mode (subset of keypoints)
- Per-keypoint weight customization
"""

from typing import Dict, List, Optional, NamedTuple
import numpy as np
from omegaconf import DictConfig


# Original MAMMAL paper keypoint weights (22 keypoints)
# Some keypoints get lower/higher weights based on detection reliability
MAMMAL_PAPER_KEYPOINT_WEIGHTS = {
    4: 0.4,    # Lower confidence
    5: 2.0,    # Higher confidence (important joint)
    6: 1.5,
    7: 1.5,
    11: 0.9,
    15: 0.9,
}

# Tail keypoint indices (for step2 weight boost)
TAIL_KEYPOINT_INDICES = [16, 17, 18, 19, 20, 21]


class KeypointWeightConfig(NamedTuple):
    """Container for keypoint weight settings."""
    weights: np.ndarray       # Shape: (keypoint_num,)
    sparse_indices: Optional[List[int]]
    tail_step2_weight: float


def get_keypoint_weights(
    cfg: Optional[DictConfig] = None,
    keypoint_num: int = 22
) -> KeypointWeightConfig:
    """Get keypoint weights from config or defaults.

    Args:
        cfg: Hydra config with optional keypoint_weights section.
        keypoint_num: Total number of keypoints (default 22).

    Returns:
        KeypointWeightConfig with weights array and metadata.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({
        ...     'fitter': {'keypoint_num': 22},
        ...     'keypoint_weights': {'default': 1.0, 'idx_5': 2.0}
        ... })
        >>> kw = get_keypoint_weights(cfg)
        >>> kw.weights[5]
        2.0
    """
    # Get keypoint_num from config if available
    if cfg is not None and hasattr(cfg, 'fitter'):
        keypoint_num = getattr(cfg.fitter, 'keypoint_num', keypoint_num)

    kw_cfg = getattr(cfg, 'keypoint_weights', None) if cfg else None
    fitter_cfg = getattr(cfg, 'fitter', None) if cfg else None

    # Get sparse keypoint indices if specified
    sparse_indices = None
    if fitter_cfg is not None:
        sparse_indices = getattr(fitter_cfg, 'sparse_keypoint_indices', None)
        if sparse_indices is not None:
            sparse_indices = list(sparse_indices)

    # Get default weight
    default_weight = getattr(kw_cfg, 'default', 1.0) if kw_cfg else 1.0

    # Initialize weights array
    weights = np.ones(keypoint_num) * default_weight

    # Handle sparse mode: if default=0, set sparse indices to 1
    if sparse_indices and default_weight == 0.0:
        for idx in sparse_indices:
            if 0 <= idx < keypoint_num:
                weights[idx] = 1.0

    # Apply individual index weights from config
    if kw_cfg:
        for idx in range(keypoint_num):
            attr_name = f'idx_{idx}'
            if hasattr(kw_cfg, attr_name):
                weights[idx] = getattr(kw_cfg, attr_name)
    else:
        # Fallback: use original MAMMAL paper weights
        for idx, weight in MAMMAL_PAPER_KEYPOINT_WEIGHTS.items():
            if idx < keypoint_num:
                weights[idx] = weight

    # Get tail weight for step2
    tail_step2_weight = getattr(kw_cfg, 'tail_step2', 10.0) if kw_cfg else 10.0

    return KeypointWeightConfig(weights, sparse_indices, tail_step2_weight)


def apply_step2_tail_weights(
    weights: np.ndarray,
    tail_weight: float = 10.0
) -> np.ndarray:
    """Apply step2 tail keypoint weight boost.

    In step2 optimization, tail keypoints get higher weight for
    better tail tracking.

    Args:
        weights: Current keypoint weights array.
        tail_weight: Weight to apply to tail keypoints.

    Returns:
        Modified weights array (also modifies in place).
    """
    for idx in TAIL_KEYPOINT_INDICES:
        if idx < len(weights):
            weights[idx] = tail_weight
    return weights
