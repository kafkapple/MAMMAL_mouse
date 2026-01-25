"""
SAM (Segment Anything Model) Inference Wrapper
"""
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import logging

logger = logging.getLogger(__name__)


class SAMInference:
    """
    Wrapper for SAM model inference with caching and optimization
    """

    def __init__(self, checkpoint_path="checkpoints/sam_vit_h_4b8939.pth",
                 model_type="vit_h", device=None):
        """
        Initialize SAM model

        Args:
            checkpoint_path: Path to SAM checkpoint
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            device: Device to use ('cuda' or 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing SAM on device: {self.device}")

        # Load SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)

        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        logger.info("SAM model loaded successfully")

    def generate_masks(self, image):
        """
        Generate masks for an image

        Args:
            image: Input image (H, W, 3) in RGB format

        Returns:
            List of mask dictionaries with keys:
                - 'segmentation': binary mask (H, W)
                - 'area': mask area in pixels
                - 'bbox': bounding box [x, y, w, h]
                - 'predicted_iou': predicted IoU
                - 'stability_score': stability score
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume input might be BGR (from OpenCV)
            # SAM expects RGB
            pass  # Let caller handle conversion

        masks = self.mask_generator.generate(image)
        return masks

    def generate_masks_batch(self, images, batch_size=4):
        """
        Generate masks for multiple images in batches

        Args:
            images: List of images
            batch_size: Number of images to process together

        Returns:
            List of mask lists
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = [self.generate_masks(img) for img in batch]
            results.extend(batch_results)

            # Clear cache periodically
            if self.device == "cuda" and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        return results

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'sam') and self.device == "cuda":
            torch.cuda.empty_cache()


def get_mask_stats(mask_dict):
    """
    Get statistics for a SAM mask

    Args:
        mask_dict: Mask dictionary from SAM

    Returns:
        Dictionary with mask statistics
    """
    return {
        'area': mask_dict['area'],
        'bbox': mask_dict['bbox'],
        'predicted_iou': mask_dict.get('predicted_iou', 0.0),
        'stability_score': mask_dict.get('stability_score', 0.0),
    }


def sort_masks_by_area(masks, reverse=True):
    """
    Sort masks by area

    Args:
        masks: List of SAM mask dictionaries
        reverse: If True, largest first

    Returns:
        Sorted list of masks
    """
    return sorted(masks, key=lambda x: x['area'], reverse=reverse)
