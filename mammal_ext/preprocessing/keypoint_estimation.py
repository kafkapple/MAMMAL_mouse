"""
Keypoint Estimation - Generate MAMMAL 22 keypoints from mouse mask
"""
import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


def estimate_mammal_keypoints(mouse_mask, frame_idx=0):
    """
    Estimate 22 MAMMAL keypoints from mouse mask

    MAMMAL keypoint layout:
    0-5: Head (nose, ears, eyes, head center)
    6-13: Spine (8 points along backbone)
    14-17: Limbs (4 paws)
    18-20: Tail (3 points)
    21: Body centroid

    Args:
        mouse_mask: Binary mask of mouse (H, W)
        frame_idx: Frame index for logging

    Returns:
        keypoints: (22, 3) array of [x, y, confidence]
    """
    # Initialize keypoints
    keypoints = np.zeros((22, 3), dtype=np.float32)

    if mouse_mask is None or np.sum(mouse_mask) == 0:
        logger.warning(f"Frame {frame_idx}: Empty mask, returning zero keypoints")
        return keypoints

    # Get contour and properties
    contours, _ = cv2.findContours(
        mouse_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        logger.warning(f"Frame {frame_idx}: No contours found")
        return keypoints

    # Largest contour is the mouse
    contour = max(contours, key=cv2.contourArea)

    # Calculate body orientation using PCA
    points = np.argwhere(mouse_mask > 0)  # (N, 2) in (y, x) format
    if len(points) < 10:
        logger.warning(f"Frame {frame_idx}: Too few points for PCA")
        return keypoints

    # Swap to (x, y)
    points_xy = points[:, ::-1]

    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(points_xy)

    # Major axis (body direction)
    major_axis = pca.components_[0]  # Direction of maximum variance

    # Find extrema along major axis
    projected = points_xy @ major_axis
    head_idx = np.argmax(projected)
    tail_idx = np.argmin(projected)

    head_point = points_xy[head_idx]
    tail_point = points_xy[tail_idx]

    # Body centroid (keypoint 21)
    centroid = np.mean(points_xy, axis=0)
    keypoints[21] = [centroid[0], centroid[1], 0.95]

    # Determine which end is head (smaller end, or closer to top of frame)
    # Heuristic: head is usually at top or has smaller width
    head_region = points_xy[projected > np.percentile(projected, 75)]
    tail_region = points_xy[projected < np.percentile(projected, 25)]

    head_width = np.std(head_region[:, 1]) if len(head_region) > 0 else 0
    tail_width = np.std(tail_region[:, 1]) if len(tail_region) > 0 else 0

    # If tail region is wider, swap
    if tail_width < head_width:
        head_point, tail_point = tail_point, head_point
        major_axis = -major_axis

    # --- HEAD KEYPOINTS (0-5) ---
    # 0: Nose tip (head extreme)
    keypoints[0] = [head_point[0], head_point[1], 0.65]

    # 1-2: Ears (perpendicular to head direction)
    perpendicular = np.array([-major_axis[1], major_axis[0]])
    ear_offset = perpendicular * np.std(head_region[:, :], axis=0).mean() * 0.7 if len(head_region) > 0 else 5

    ear_base = head_point + major_axis * 10  # Slightly behind nose
    keypoints[1] = [ear_base[0] + ear_offset[0], ear_base[1] + ear_offset[1], 0.50]  # Left ear
    keypoints[2] = [ear_base[0] - ear_offset[0], ear_base[1] - ear_offset[1], 0.50]  # Right ear

    # 3-4: Eyes (between nose and ears)
    eye_pos = head_point + major_axis * 8
    eye_offset = perpendicular * 3
    keypoints[3] = [eye_pos[0] + eye_offset[0], eye_pos[1] + eye_offset[1], 0.40]  # Left eye
    keypoints[4] = [eye_pos[0] - eye_offset[0], eye_pos[1] - eye_offset[1], 0.40]  # Right eye

    # 5: Head center
    head_center = (head_point + ear_base) / 2
    keypoints[5] = [head_center[0], head_center[1], 0.75]

    # --- SPINE KEYPOINTS (6-13) ---
    # 8 points evenly distributed along backbone
    body_length = np.linalg.norm(head_point - tail_point)
    for i in range(8):
        t = (i + 1) / 9.0  # 1/9 to 8/9 along the body
        spine_point = head_point + (tail_point - head_point) * t
        keypoints[6 + i] = [spine_point[0], spine_point[1], 0.70]

    # --- LIMB KEYPOINTS (14-17) ---
    # Estimate paw positions from contour extrema in perpendicular direction
    # Front paws: near head, perpendicular to body
    front_region_mask = (projected > np.percentile(projected, 60))
    front_points = points_xy[front_region_mask]

    if len(front_points) > 0:
        front_center = np.mean(front_points, axis=0)

        # Find extrema perpendicular to body axis
        front_perp_proj = front_points @ perpendicular
        left_front_idx = np.argmax(front_perp_proj)
        right_front_idx = np.argmin(front_perp_proj)

        keypoints[14] = [front_points[left_front_idx][0], front_points[left_front_idx][1], 0.45]  # Left front
        keypoints[15] = [front_points[right_front_idx][0], front_points[right_front_idx][1], 0.45]  # Right front
    else:
        # Fallback
        front_pos = head_point + (tail_point - head_point) * 0.4
        paw_offset = perpendicular * 15
        keypoints[14] = [front_pos[0] + paw_offset[0], front_pos[1] + paw_offset[1], 0.35]
        keypoints[15] = [front_pos[0] - paw_offset[0], front_pos[1] - paw_offset[1], 0.35]

    # Rear paws: near tail, perpendicular to body
    rear_region_mask = (projected < np.percentile(projected, 40))
    rear_points = points_xy[rear_region_mask]

    if len(rear_points) > 0:
        rear_perp_proj = rear_points @ perpendicular
        left_rear_idx = np.argmax(rear_perp_proj)
        right_rear_idx = np.argmin(rear_perp_proj)

        keypoints[16] = [rear_points[left_rear_idx][0], rear_points[left_rear_idx][1], 0.45]  # Left rear
        keypoints[17] = [rear_points[right_rear_idx][0], rear_points[right_rear_idx][1], 0.45]  # Right rear
    else:
        # Fallback
        rear_pos = head_point + (tail_point - head_point) * 0.7
        paw_offset = perpendicular * 15
        keypoints[16] = [rear_pos[0] + paw_offset[0], rear_pos[1] + paw_offset[1], 0.35]
        keypoints[17] = [rear_pos[0] - paw_offset[0], rear_pos[1] - paw_offset[1], 0.35]

    # --- TAIL KEYPOINTS (18-20) ---
    # 18: Tail base (near tail extreme)
    tail_base = tail_point + major_axis * 10  # Slightly towards body
    keypoints[18] = [tail_base[0], tail_base[1], 0.70]

    # 19: Tail mid
    tail_mid = (tail_base + tail_point) / 2
    keypoints[19] = [tail_mid[0], tail_mid[1], 0.55]

    # 20: Tail tip (tail extreme)
    keypoints[20] = [tail_point[0], tail_point[1], 0.50]

    return keypoints


def refine_keypoints_with_skeleton(keypoints, mouse_mask):
    """
    Refine keypoints using skeletonization (optional enhancement)

    Args:
        keypoints: (22, 3) initial keypoints
        mouse_mask: Binary mask

    Returns:
        Refined keypoints
    """
    # This is an optional enhancement that can be added later
    # For now, return keypoints as-is
    return keypoints


class TemporalKeypointSmoother:
    """
    Temporal smoothing for keypoints
    """

    def __init__(self, window_size=5, alpha=0.7):
        """
        Args:
            window_size: Number of frames to average
            alpha: Smoothing factor (0=no smooth, 1=full smooth)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.history = []

    def smooth(self, keypoints):
        """
        Apply temporal smoothing

        Args:
            keypoints: (22, 3) current keypoints

        Returns:
            Smoothed keypoints
        """
        self.history.append(keypoints.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) == 1:
            return keypoints

        # Weighted average (more recent frames weighted higher)
        weights = np.linspace(1 - self.alpha, 1.0, len(self.history))
        weights /= weights.sum()

        smoothed = np.zeros_like(keypoints)
        for w, kpts in zip(weights, self.history):
            smoothed += w * kpts

        return smoothed

    def reset(self):
        """Reset history"""
        self.history = []


def validate_keypoints(keypoints, mask_shape):
    """
    Validate keypoints are within image bounds and reasonable

    Args:
        keypoints: (22, 3) keypoints
        mask_shape: (H, W) of image

    Returns:
        Valid keypoints (clipped to bounds)
    """
    H, W = mask_shape

    # Clip to image bounds
    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, W - 1)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, H - 1)

    # Ensure confidence is in [0, 1]
    keypoints[:, 2] = np.clip(keypoints[:, 2], 0.0, 1.0)

    return keypoints


def visualize_keypoints_on_frame(frame, keypoints, color=(0, 255, 0)):
    """
    Draw keypoints on frame for visualization

    Args:
        frame: Image (H, W, 3)
        keypoints: (22, 3) keypoints
        color: BGR color tuple

    Returns:
        Frame with keypoints drawn
    """
    vis = frame.copy()

    # Define keypoint groups and colors
    groups = {
        'head': (0, 6, (255, 0, 0)),        # Blue for head
        'spine': (6, 14, (0, 255, 0)),      # Green for spine
        'limbs': (14, 18, (0, 0, 255)),     # Red for limbs
        'tail': (18, 21, (255, 255, 0)),    # Cyan for tail
        'centroid': (21, 22, (255, 0, 255)) # Magenta for centroid
    }

    for group_name, (start, end, group_color) in groups.items():
        for i in range(start, end):
            x, y, conf = keypoints[i]
            if conf > 0.3:  # Only draw confident keypoints
                # Circle size based on confidence
                radius = int(3 + conf * 3)
                cv2.circle(vis, (int(x), int(y)), radius, group_color, -1)

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
            cv2.line(vis, pt1, pt2, (200, 200, 200), 1)

    return vis
