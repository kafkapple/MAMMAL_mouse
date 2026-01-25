"""Loss weight configuration utilities.

This module handles loading and parsing loss weights from Hydra config.
Extracted from fitter_articulation.py MouseFitter.__init__().

Original MAMMAL paper loss weights are preserved as defaults.
"""

from typing import Dict, Any, Optional, NamedTuple
from omegaconf import DictConfig


# Default loss weights from MAMMAL paper (An et al., Nature Communications 2023)
DEFAULT_LOSS_WEIGHTS = {
    "theta": 3.0,           # Pose prior regularization
    "3d": 2.5,              # 3D keypoint loss (if GT available)
    "2d": 0.2,              # 2D keypoint reprojection loss
    "bone": 0.5,            # Bone length regularization
    "scale": 0.5,           # Scale regularization
    "mask": 10.0,           # Silhouette loss (base value)
    "chest_deformer": 0.1,  # Chest deformation regularization
    "stretch": 1.0,         # Stretch constraint
    "temp": 0.25,           # Temporal smoothness (params)
    "temp_d": 0.2,          # Temporal smoothness (deformer)
}

# Step-specific mask weights from MAMMAL paper
DEFAULT_MASK_WEIGHTS = {
    "step0": 0.0,      # Global pose (R,T,s) - no mask
    "step1": 0.0,      # Pose optimization - no mask
    "step2": 3000.0,   # Full optimization with silhouette refinement
}


class LossWeightConfig(NamedTuple):
    """Container for all loss weight settings."""
    weights: Dict[str, float]
    mask_step0: float
    mask_step1: float
    mask_step2: float


def get_loss_weights(cfg: Optional[DictConfig] = None) -> LossWeightConfig:
    """Get loss weights from config or defaults.

    Args:
        cfg: Hydra config with optional loss_weights section.

    Returns:
        LossWeightConfig with term weights and step-specific mask weights.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({'loss_weights': {'theta': 5.0}})
        >>> lw = get_loss_weights(cfg)
        >>> lw.weights['theta']
        5.0
        >>> lw.weights['bone']  # Default preserved
        0.5
    """
    weights = DEFAULT_LOSS_WEIGHTS.copy()
    mask_step0 = DEFAULT_MASK_WEIGHTS["step0"]
    mask_step1 = DEFAULT_MASK_WEIGHTS["step1"]
    mask_step2 = DEFAULT_MASK_WEIGHTS["step2"]

    if cfg is None:
        return LossWeightConfig(weights, mask_step0, mask_step1, mask_step2)

    lw_cfg = getattr(cfg, 'loss_weights', None)
    if lw_cfg is None:
        return LossWeightConfig(weights, mask_step0, mask_step1, mask_step2)

    # Load term weights
    for key in weights:
        if hasattr(lw_cfg, key):
            weights[key] = getattr(lw_cfg, key)

    # Load step-specific mask weights
    if hasattr(lw_cfg, 'mask_step0'):
        mask_step0 = getattr(lw_cfg, 'mask_step0')
    if hasattr(lw_cfg, 'mask_step1'):
        mask_step1 = getattr(lw_cfg, 'mask_step1')
    if hasattr(lw_cfg, 'mask_step2'):
        mask_step2 = getattr(lw_cfg, 'mask_step2')

    return LossWeightConfig(weights, mask_step0, mask_step1, mask_step2)


def apply_silhouette_mode_weights(
    weights: Dict[str, float],
    cfg: Optional[DictConfig] = None
) -> Dict[str, float]:
    """Apply silhouette-only mode weight adjustments.

    When keypoints are disabled, silhouette-only mode requires different
    weight balancing for stable optimization.

    Args:
        weights: Base term weights to modify.
        cfg: Config with optional 'silhouette' section.

    Returns:
        Modified weights dict (also modifies in place).
    """
    # Disable keypoint loss
    weights["2d"] = 0

    sil_cfg = getattr(cfg, 'silhouette', None) if cfg else None

    if sil_cfg:
        weights["scale"] = getattr(sil_cfg, 'scale_weight', 50.0)
        weights["theta"] = getattr(sil_cfg, 'theta_weight', 10.0)
        weights["bone"] = getattr(sil_cfg, 'bone_weight', 2.0)
    else:
        # Default silhouette-only mode weights
        weights["scale"] = 50.0
        weights["theta"] = 10.0
        weights["bone"] = 2.0

    return weights
