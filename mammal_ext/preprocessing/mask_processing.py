"""
Mask Processing - Extract mouse mask from SAM output
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_circularity(contour):
    """
    Calculate circularity of a contour (1.0 = perfect circle)

    Args:
        contour: OpenCV contour

    Returns:
        Circularity score (0-1)
    """
    area = cv2.contourArea(contour)
    if area == 0:
        return 0.0

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return min(circularity, 1.0)


def get_mask_properties(mask):
    """
    Get properties of a binary mask

    Args:
        mask: Binary mask (H, W)

    Returns:
        Dictionary with properties
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate properties
    area = cv2.contourArea(largest_contour)
    M = cv2.moments(largest_contour)

    if M['m00'] == 0:
        return None

    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Circularity
    circularity = calculate_circularity(largest_contour)

    return {
        'area': area,
        'centroid': (centroid_x, centroid_y),
        'bbox': (x, y, w, h),
        'circularity': circularity,
        'contour': largest_contour,
    }


def extract_mouse_mask(sam_masks, frame_shape, strategy='multi_stage'):
    """
    Extract mouse mask from SAM output

    Args:
        sam_masks: List of SAM mask dictionaries
        frame_shape: (H, W) of original frame
        strategy: 'multi_stage' or 'simple'

    Returns:
        Binary mask (H, W) with mouse, or None if not found
    """
    if len(sam_masks) == 0:
        logger.warning("No SAM masks provided")
        return None

    H, W = frame_shape[:2]
    total_area = H * W

    if strategy == 'simple':
        # Simple strategy: second largest mask
        sorted_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)
        if len(sorted_masks) >= 2:
            return sorted_masks[1]['segmentation'].astype(np.uint8) * 255
        else:
            return sorted_masks[0]['segmentation'].astype(np.uint8) * 255

    # Multi-stage strategy
    candidates = []

    for mask_dict in sam_masks:
        mask = mask_dict['segmentation']
        area = mask_dict['area']

        # Stage 1: Size filtering
        # Mouse should be 5-20% of frame
        coverage = area / total_area
        if coverage < 0.03 or coverage > 0.25:
            continue

        # Get mask properties
        props = get_mask_properties(mask)
        if props is None:
            continue

        # Stage 2: Shape filtering
        # Mouse is irregular (not circular like arena)
        if props['circularity'] > 0.85:
            continue

        # Stage 3: Position filtering
        # Mouse should be somewhat central (inside arena)
        cx, cy = props['centroid']
        frame_center_x, frame_center_y = W // 2, H // 2
        dist_from_center = np.sqrt((cx - frame_center_x)**2 + (cy - frame_center_y)**2)
        max_dist = min(W, H) * 0.45  # Should be within 45% of frame size from center

        if dist_from_center > max_dist:
            continue

        # Add to candidates with score
        score = area * (1.0 - props['circularity'])  # Prefer large, irregular masks
        candidates.append({
            'mask': mask,
            'score': score,
            'props': props,
        })

    # Select best candidate
    if len(candidates) == 0:
        logger.warning("No mouse candidates found, using fallback")
        # Fallback: second largest overall
        sorted_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)
        if len(sorted_masks) >= 2:
            return sorted_masks[1]['segmentation'].astype(np.uint8) * 255
        else:
            return sorted_masks[0]['segmentation'].astype(np.uint8) * 255

    # Return highest scoring candidate
    best_candidate = max(candidates, key=lambda x: x['score'])
    return best_candidate['mask'].astype(np.uint8) * 255


def clean_mask(mask, min_size=100):
    """
    Clean mask by removing small noise regions

    Args:
        mask: Binary mask (H, W)
        min_size: Minimum region size to keep

    Returns:
        Cleaned mask
    """
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    # Create output mask
    clean = np.zeros_like(mask, dtype=np.uint8)

    # Keep only large components
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            clean[labels == i] = 255

    return clean


def smooth_mask(mask, kernel_size=5):
    """
    Smooth mask edges

    Args:
        mask: Binary mask
        kernel_size: Morphological kernel size

    Returns:
        Smoothed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Close small holes
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Smooth edges
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened


class TemporalMaskFilter:
    """
    Temporal consistency filter for masks
    """

    def __init__(self, window_size=5, iou_threshold=0.5):
        """
        Args:
            window_size: Number of previous frames to consider
            iou_threshold: Minimum IoU with previous frames
        """
        self.window_size = window_size
        self.iou_threshold = iou_threshold
        self.history = []

    def filter(self, current_mask):
        """
        Filter current mask based on temporal consistency

        Args:
            current_mask: Current binary mask

        Returns:
            Filtered mask (or current if no history)
        """
        if current_mask is None:
            # Use previous frame if available
            if len(self.history) > 0:
                logger.warning("Current mask is None, using previous frame")
                return self.history[-1].copy()
            else:
                return None

        # Add to history
        self.history.append(current_mask.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # If not enough history, return current
        if len(self.history) < 2:
            return current_mask

        # Check consistency with previous frames
        ious = []
        for prev_mask in self.history[:-1]:
            iou = calculate_iou(current_mask, prev_mask)
            ious.append(iou)

        mean_iou = np.mean(ious)

        # If inconsistent, blend with previous frame
        if mean_iou < self.iou_threshold:
            logger.debug(f"Low temporal consistency (IoU={mean_iou:.3f}), blending with previous")
            # Blend current with previous
            prev_mask = self.history[-2]
            blended = np.maximum(current_mask, prev_mask)
            self.history[-1] = blended
            return blended

        return current_mask

    def reset(self):
        """Reset history"""
        self.history = []


def calculate_iou(mask1, mask2):
    """
    Calculate IoU between two binary masks

    Args:
        mask1, mask2: Binary masks

    Returns:
        IoU score
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()

    if union == 0:
        return 0.0

    return intersection / union
