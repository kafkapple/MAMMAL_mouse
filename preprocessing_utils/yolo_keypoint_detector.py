"""
YOLOv8-based keypoint detection for MAMMAL mouse pose estimation

This module provides a ML-based keypoint detector using fine-tuned YOLOv8-Pose model.
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. YOLOv8KeypointDetector will not work.")

logger = logging.getLogger(__name__)


class YOLOv8KeypointDetector:
    """
    YOLOv8-based keypoint detector for MAMMAL 22 keypoints

    This detector uses a fine-tuned YOLOv8-Pose model trained on mouse images.
    It provides significantly better accuracy than geometric methods.

    Usage:
        detector = YOLOv8KeypointDetector('path/to/yolov8_mouse.pt')
        keypoints = detector.detect(rgb_image)  # Returns (22, 3) array
    """

    def __init__(self,
                 model_path: Union[str, Path],
                 device: str = 'cuda',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.7):
        """
        Initialize YOLOv8 keypoint detector

        Args:
            model_path: Path to trained YOLOv8-Pose model (.pt file)
            device: Device to run inference ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = device

        # Load model
        logger.info(f"Loading YOLOv8-Pose model from {model_path}")
        self.model = YOLO(str(model_path))

        # Set inference parameters
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        logger.info(f"YOLOv8KeypointDetector initialized (device: {device})")

    def detect(self,
               image: np.ndarray,
               return_bbox: bool = False,
               visualize: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect keypoints from RGB image

        Args:
            image: RGB image (H, W, 3) as numpy array
            return_bbox: If True, also return bounding box [x1, y1, x2, y2]
            visualize: If True, return visualization image

        Returns:
            keypoints: (22, 3) array of [x, y, confidence]
            bbox: (Optional) [x1, y1, x2, y2] bounding box
            vis_image: (Optional) visualization image
        """
        if image is None or image.size == 0:
            logger.error("Empty image provided")
            return self._get_empty_keypoints(return_bbox)

        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=256  # Match training size
        )

        # Extract keypoints from results
        keypoints, bbox = self._extract_keypoints(results, image.shape[:2])

        # Prepare return values
        if return_bbox:
            if visualize:
                vis_image = self._visualize(image, keypoints, bbox)
                return keypoints, bbox, vis_image
            else:
                return keypoints, bbox
        else:
            if visualize:
                vis_image = self._visualize(image, keypoints, bbox)
                return keypoints, vis_image
            else:
                return keypoints

    def _extract_keypoints(self, results, img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract keypoints and bbox from YOLO results

        Args:
            results: YOLO prediction results
            img_shape: (height, width) of original image

        Returns:
            keypoints: (22, 3) array of [x, y, confidence]
            bbox: [x1, y1, x2, y2] bounding box
        """
        if len(results) == 0 or results[0].keypoints is None:
            logger.warning("No detections found")
            return self._get_empty_keypoints(return_bbox=True)

        result = results[0]

        # Get keypoints (take first detection if multiple)
        if result.keypoints.xy.shape[0] > 0:
            keypoints_xy = result.keypoints.xy[0].cpu().numpy()  # (22, 2)
            keypoints_conf = result.keypoints.conf[0].cpu().numpy()  # (22,)

            # Combine to (22, 3)
            keypoints = np.zeros((22, 3), dtype=np.float32)
            keypoints[:, :2] = keypoints_xy
            keypoints[:, 2] = keypoints_conf

            # Get bbox
            if result.boxes is not None and len(result.boxes) > 0:
                bbox = result.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            else:
                # Infer bbox from keypoints
                bbox = self._infer_bbox_from_keypoints(keypoints, img_shape)
        else:
            logger.warning("No keypoints in detection")
            keypoints = np.zeros((22, 3), dtype=np.float32)
            bbox = np.array([0, 0, img_shape[1], img_shape[0]], dtype=np.float32)

        return keypoints, bbox

    def _infer_bbox_from_keypoints(self, keypoints: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Infer bounding box from keypoints

        Args:
            keypoints: (22, 3) array
            img_shape: (height, width)

        Returns:
            bbox: [x1, y1, x2, y2]
        """
        # Filter out low-confidence keypoints
        valid_kpts = keypoints[keypoints[:, 2] > 0.1, :2]

        if len(valid_kpts) > 0:
            x_min = valid_kpts[:, 0].min()
            y_min = valid_kpts[:, 1].min()
            x_max = valid_kpts[:, 0].max()
            y_max = valid_kpts[:, 1].max()

            # Add margin
            margin = 0.1
            w = x_max - x_min
            h = y_max - y_min
            x_min = max(0, x_min - margin * w)
            y_min = max(0, y_min - margin * h)
            x_max = min(img_shape[1], x_max + margin * w)
            y_max = min(img_shape[0], y_max + margin * h)

            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        else:
            # Default to full image
            bbox = np.array([0, 0, img_shape[1], img_shape[0]], dtype=np.float32)

        return bbox

    def _get_empty_keypoints(self, return_bbox: bool = False):
        """Return empty keypoints (for failure cases)"""
        keypoints = np.zeros((22, 3), dtype=np.float32)
        if return_bbox:
            bbox = np.zeros(4, dtype=np.float32)
            return keypoints, bbox
        return keypoints

    def _visualize(self, image: np.ndarray, keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Visualize keypoints and bbox on image

        Args:
            image: RGB image
            keypoints: (22, 3) keypoints
            bbox: [x1, y1, x2, y2]

        Returns:
            Visualization image
        """
        vis = image.copy()

        # Draw bbox
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Define keypoint groups and colors
        groups = {
            'head': (0, 6, (255, 0, 0)),        # Red for head
            'spine': (6, 14, (0, 255, 0)),      # Green for spine
            'limbs': (14, 18, (0, 0, 255)),     # Blue for limbs
            'tail': (18, 21, (255, 255, 0)),    # Yellow for tail
            'centroid': (21, 22, (255, 0, 255)) # Magenta for centroid
        }

        # Draw keypoints
        for group_name, (start, end, color) in groups.items():
            for i in range(start, end):
                x, y, conf = keypoints[i]
                if conf > 0.3:  # Only draw confident keypoints
                    # Circle size based on confidence
                    radius = int(3 + conf * 4)
                    cv2.circle(vis, (int(x), int(y)), radius, color, -1)

                    # Add keypoint index
                    cv2.putText(vis, str(i), (int(x) + 5, int(y) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw skeleton connections
        connections = [
            # Spine
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
            # Head to spine
            (5, 6),
            # Tail
            (13, 18), (18, 19), (19, 20),
        ]

        for i, j in connections:
            if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(vis, pt1, pt2, (200, 200, 200), 2)

        return vis

    def detect_batch(self, images: list) -> list:
        """
        Detect keypoints for a batch of images

        Args:
            images: List of RGB images

        Returns:
            List of keypoints arrays (22, 3) each
        """
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=256
        )

        keypoints_list = []
        for result, img in zip(results, images):
            keypoints, _ = self._extract_keypoints([result], img.shape[:2])
            keypoints_list.append(keypoints)

        return keypoints_list

    @staticmethod
    def get_keypoint_names():
        """Get names of all 22 MAMMAL keypoints"""
        return [
            "nose", "left_ear", "right_ear", "left_eye", "right_eye", "head_center",
            "spine_1", "spine_2", "spine_3", "spine_4", "spine_5", "spine_6", "spine_7", "spine_8",
            "left_front_paw", "right_front_paw", "left_rear_paw", "right_rear_paw",
            "tail_base", "tail_mid", "tail_tip", "centroid"
        ]


def demo():
    """
    Demo usage of YOLOv8KeypointDetector
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python yolo_keypoint_detector.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize detector
    print(f"Loading model: {model_path}")
    detector = YOLOv8KeypointDetector(model_path, device='cuda')

    # Detect keypoints
    print(f"Detecting keypoints in: {image_path}")
    keypoints, bbox, vis = detector.detect(image, return_bbox=True, visualize=True)

    # Print results
    print(f"\nDetected {np.sum(keypoints[:, 2] > 0.3)} keypoints with conf > 0.3")
    print(f"Bounding box: {bbox}")
    print(f"\nKeypoint confidences:")
    for i, name in enumerate(detector.get_keypoint_names()):
        print(f"  {i:2d} {name:20s}: {keypoints[i, 2]:.3f}")

    # Save visualization
    output_path = Path(image_path).stem + "_yolo_keypoints.png"
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, vis_bgr)
    print(f"\nVisualization saved: {output_path}")


if __name__ == '__main__':
    demo()
