"""
Preprocessing utilities for MAMMAL mouse data.

Modules:
    keypoint_estimation: Automatic keypoint detection and refinement
    mask_processing: SAM mask extraction and temporal filtering
    sam_inference: SAM model inference wrapper
    silhouette_renderer: Differentiable silhouette rendering and loss
    dannce_to_yolo: DANNCE → YOLO format conversion
    superanimal_detector: SuperAnimal keypoint detection
    yolo_keypoint_detector: YOLO-based keypoint detection
    visualize_yolo_labels: YOLO label visualization
"""

from .keypoint_estimation import (
    estimate_mammal_keypoints,
    refine_keypoints_with_skeleton,
    TemporalKeypointSmoother,
    validate_keypoints,
)
from .mask_processing import (
    extract_mouse_mask,
    clean_mask,
    smooth_mask,
    TemporalMaskFilter,
    calculate_iou,
)
from .sam_inference import SAMInference
from .silhouette_renderer import (
    SilhouetteRenderer,
    SilhouetteLoss,
    load_target_mask,
)

__all__ = [
    # Keypoint estimation
    "estimate_mammal_keypoints",
    "refine_keypoints_with_skeleton",
    "TemporalKeypointSmoother",
    "validate_keypoints",
    # Mask processing
    "extract_mouse_mask",
    "clean_mask",
    "smooth_mask",
    "TemporalMaskFilter",
    "calculate_iou",
    # SAM inference
    "SAMInference",
    # Silhouette
    "SilhouetteRenderer",
    "SilhouetteLoss",
    "load_target_mask",
]
