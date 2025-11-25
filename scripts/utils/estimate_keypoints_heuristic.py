"""
Simple Heuristic Keypoint Estimation from Silhouette
Not accurate, but can provide initial guess for optimization
"""
import cv2
import numpy as np
from pathlib import Path
import json


def estimate_keypoints_from_mask(mask):
    """
    Estimate basic keypoints from binary mask using heuristics

    Args:
        mask: Binary mask (H, W) numpy array

    Returns:
        keypoints: Dictionary of estimated keypoint locations
    """
    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)

    # Get moments for centroid
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Fit ellipse to get orientation
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center_x, center_y), (MA, ma), angle = ellipse
    else:
        return None

    # Get bounding box
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Find extremes along major axis
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Project all points onto major axis
    points = np.column_stack((x_indices, y_indices))
    axis_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    projections = np.dot(points - np.array([cx, cy]), axis_vector)

    # Find extreme points
    max_idx = np.argmax(projections)
    min_idx = np.argmin(projections)

    head_point = points[max_idx]
    tail_point = points[min_idx]

    # Determine which is head (usually smaller y, i.e., higher in image)
    if head_point[1] > tail_point[1]:
        head_point, tail_point = tail_point, head_point

    # Estimate keypoints along spine
    # Assume: head → neck → spine_mid → hip → tail_base
    spine_points = []
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pt = (1 - t) * head_point + t * tail_point
        spine_points.append(pt)

    keypoints = {
        'nose': {'x': float(spine_points[0][0]), 'y': float(spine_points[0][1]), 'confidence': 0.5},
        'neck': {'x': float(spine_points[1][0]), 'y': float(spine_points[1][1]), 'confidence': 0.4},
        'spine_mid': {'x': float(spine_points[2][0]), 'y': float(spine_points[2][1]), 'confidence': 0.6},
        'hip': {'x': float(spine_points[3][0]), 'y': float(spine_points[3][1]), 'confidence': 0.4},
        'tail_base': {'x': float(spine_points[4][0]), 'y': float(spine_points[4][1]), 'confidence': 0.5},
        'centroid': {'x': float(cx), 'y': float(cy), 'confidence': 0.9},
    }

    return keypoints


def process_masks(masks_dir, output_file):
    """
    Process all masks and estimate keypoints

    Args:
        masks_dir: Directory containing mask files
        output_file: JSON file to save keypoints
    """
    masks_dir = Path(masks_dir)
    mask_files = sorted(masks_dir.glob('*_mask.png'))

    all_keypoints = {}

    for mask_file in mask_files:
        # Get frame name
        frame_name = mask_file.stem.replace('_mask', '')

        # Load mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Estimate keypoints
        keypoints = estimate_keypoints_from_mask(mask)
        if keypoints is not None:
            all_keypoints[frame_name] = keypoints
            print(f"Processed {frame_name}: {len(keypoints)} keypoints")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_keypoints, f, indent=2)

    print(f"\nSaved keypoints to {output_file}")
    print(f"Total frames processed: {len(all_keypoints)}")

    return all_keypoints


def visualize_keypoints(image_path, mask_path, keypoints, output_path):
    """Visualize estimated keypoints on image"""
    import matplotlib.pyplot as plt

    # Load image and mask
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Show image with keypoints
    axes[0].imshow(image)
    for name, kp in keypoints.items():
        x, y = kp['x'], kp['y']
        conf = kp['confidence']
        color = 'red' if conf > 0.5 else 'orange'
        axes[0].plot(x, y, 'o', color=color, markersize=8)
        axes[0].text(x + 5, y, name, color='white', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    axes[0].set_title('Estimated Keypoints')
    axes[0].axis('off')

    # Show mask with keypoints
    axes[1].imshow(mask, cmap='gray')
    for name, kp in keypoints.items():
        x, y = kp['x'], kp['y']
        axes[1].plot(x, y, 'ro', markersize=6)
        axes[1].text(x + 3, y, name, color='yellow', fontsize=7)
    axes[1].set_title('Keypoints on Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate keypoints from masks")
    parser.add_argument('masks_dir', type=str, help='Directory with mask files')
    parser.add_argument('--output', type=str, default='estimated_keypoints.json',
                       help='Output JSON file')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization images')

    args = parser.parse_args()

    # Process all masks
    keypoints = process_masks(args.masks_dir, args.output)

    # Optionally create visualizations
    if args.visualize:
        masks_dir = Path(args.masks_dir)
        vis_dir = masks_dir.parent / 'keypoint_visualizations'
        vis_dir.mkdir(exist_ok=True)

        for frame_name, kps in keypoints.items():
            image_file = masks_dir / f"{frame_name}_cropped.png"
            mask_file = masks_dir / f"{frame_name}_mask.png"

            if image_file.exists() and mask_file.exists():
                output_file = vis_dir / f"{frame_name}_keypoints.png"
                visualize_keypoints(image_file, mask_file, kps, output_file)
