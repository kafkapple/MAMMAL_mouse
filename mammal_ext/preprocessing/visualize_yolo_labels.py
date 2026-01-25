#!/usr/bin/env python3
"""
Visualize YOLO pose labels on images
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

# MAMMAL 22 keypoint names
KEYPOINT_NAMES = [
    "nose", "left_ear", "right_ear", "left_eye", "right_eye", "head_center",
    "spine_1", "spine_2", "spine_3", "spine_4", "spine_5", "spine_6", "spine_7", "spine_8",
    "left_front_paw", "right_front_paw", "left_rear_paw", "right_rear_paw",
    "tail_base", "tail_mid", "tail_tip", "centroid"
]

# Skeleton connections (for visualization)
SKELETON = [
    # Head
    (0, 1), (0, 2),  # nose to ears
    (0, 5),          # nose to head_center
    (1, 5), (2, 5),  # ears to head_center

    # Spine
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),

    # Tail
    (13, 18), (18, 19), (19, 20),

    # Limbs (if visible)
    (6, 14), (6, 15),  # front paws
    (12, 16), (12, 17),  # rear paws
]

def parse_yolo_label(label_path, img_width, img_height):
    """
    Parse YOLO pose label file

    Format: <class_id> <x_center> <y_center> <width> <height> <kpt1_x> <kpt1_y> <kpt1_v> ... <kpt22_x> <kpt22_y> <kpt22_v>
    """
    with open(label_path, 'r') as f:
        line = f.readline().strip()

    parts = line.split()

    # Parse bbox
    class_id = int(parts[0])
    bbox = [float(x) for x in parts[1:5]]

    # Parse keypoints (22 keypoints × 3 values each)
    keypoints = []
    for i in range(5, len(parts), 3):
        x = float(parts[i]) * img_width
        y = float(parts[i+1]) * img_height
        v = int(parts[i+2])
        keypoints.append((x, y, v))

    return bbox, keypoints

def visualize_keypoints(image, keypoints, skeleton=True):
    """
    Draw keypoints and skeleton on image
    """
    img = image.copy()

    # Draw skeleton first (so it's behind keypoints)
    if skeleton:
        for i, j in SKELETON:
            if i < len(keypoints) and j < len(keypoints):
                x1, y1, v1 = keypoints[i]
                x2, y2, v2 = keypoints[j]
                if v1 > 0 and v2 > 0:  # Both visible
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    # Draw keypoints
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # Visible or occluded
            color = (0, 255, 0) if v == 2 else (0, 165, 255)  # Green=visible, Orange=occluded
            cv2.circle(img, (int(x), int(y)), 3, color, -1)

            # Add index label
            cv2.putText(img, str(i), (int(x)+5, int(y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO pose labels")
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                       help='Directory containing YOLO label files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to visualize')
    parser.add_argument('--skeleton', action='store_true', default=True,
                       help='Draw skeleton connections')

    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))

    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"Visualizing {len(image_files)} images...")

    for img_file in image_files:
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  ✗ Failed to load: {img_file.name}")
            continue

        h, w = img.shape[:2]

        # Find corresponding label
        label_file = labels_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            print(f"  ✗ No label for: {img_file.name}")
            continue

        # Parse label
        try:
            bbox, keypoints = parse_yolo_label(label_file, w, h)

            # Visualize
            vis_img = visualize_keypoints(img, keypoints, skeleton=args.skeleton)

            # Add info text
            n_visible = sum(1 for _, _, v in keypoints if v > 0)
            cv2.putText(vis_img, f"{n_visible}/22 keypoints", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save
            output_file = output_dir / img_file.name
            cv2.imwrite(str(output_file), vis_img)
            print(f"  ✓ {img_file.name} ({n_visible}/22 keypoints)")

        except Exception as e:
            print(f"  ✗ Error processing {img_file.name}: {e}")
            continue

    print(f"\n✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
