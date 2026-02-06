"""Photometric metrics for UV texture quality evaluation.

Helper functions for computing PSNR, SSIM, and mask-based metrics
used by the WandB sweep optimizer.
"""

import torch
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# ===== Photometric Metrics Helper Functions =====
# Based on 3DGS (SIGGRAPH 2023) and IQA literature

def compute_psnr_masked(
    rendered: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute PSNR between rendered and target images in masked region.

    Args:
        rendered: [H, W, 3] Rendered RGB (0-255, uint8)
        target: [H, W, 3] Original RGB (0-255, uint8)
        mask: [H, W] Boolean mask for valid region

    Returns:
        psnr_score: [0, 1] normalized PSNR score
        psnr_db: Raw PSNR value in dB
    """
    if mask.sum() < 100:  # Too few pixels
        return 0.0, 0.0

    # Convert to float
    rendered_f = rendered.astype(np.float32)
    target_f = target.astype(np.float32)

    # Apply mask
    rendered_masked = rendered_f[mask]
    target_masked = target_f[mask]

    # MSE calculation
    mse = np.mean((rendered_masked - target_masked) ** 2)

    if mse < 1e-10:
        psnr_db = 100.0
    else:
        psnr_db = 10 * np.log10(255.0 ** 2 / mse)

    # Normalize to [0, 1] (PSNR typically 15-40 dB for reasonable results)
    psnr_min, psnr_max = 15.0, 40.0
    psnr_score = np.clip((psnr_db - psnr_min) / (psnr_max - psnr_min), 0, 1)

    return psnr_score, psnr_db


def compute_ssim_masked(
    rendered: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute SSIM between rendered and target images.

    Uses bounding box of mask for efficient computation.

    Args:
        rendered: [H, W, 3] Rendered RGB (0-255, uint8)
        target: [H, W, 3] Original RGB (0-255, uint8)
        mask: [H, W] Boolean mask for valid region

    Returns:
        ssim_score: [0, 1] structural similarity
        ssim_raw: Raw SSIM value
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        logger.warning("skimage not available, returning 0 for SSIM")
        return 0.0, 0.0

    # Find bounding box of mask
    y_indices, x_indices = np.where(mask)
    if len(y_indices) < 10:
        return 0.0, 0.0

    y1, y2 = y_indices.min(), y_indices.max() + 1
    x1, x2 = x_indices.min(), x_indices.max() + 1

    # Ensure minimum size for SSIM (at least 7x7 for default window)
    if (y2 - y1) < 7 or (x2 - x1) < 7:
        return 0.0, 0.0

    rendered_crop = rendered[y1:y2, x1:x2]
    target_crop = target[y1:y2, x1:x2]

    # SSIM calculation (channel_axis for RGB)
    ssim_val = ssim(
        target_crop,
        rendered_crop,
        channel_axis=2,
        data_range=255,
        win_size=min(7, min(rendered_crop.shape[0], rendered_crop.shape[1]) // 2 * 2 - 1)
    )

    return ssim_val, ssim_val


def create_mesh_mask(
    rendered: np.ndarray,
    background_value: int = 255,
) -> np.ndarray:
    """
    Create a mask for mesh region (non-background pixels).

    Args:
        rendered: [H, W, 3] Rendered image with white background
        background_value: Background pixel value (255 for white)

    Returns:
        mask: [H, W] Boolean mask (True = mesh region)
    """
    # Check if pixel is not pure white (background)
    is_background = np.all(rendered == background_value, axis=2)
    return ~is_background
