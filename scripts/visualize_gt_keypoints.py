#!/usr/bin/env python3
"""
Visualize all 22 GT keypoints on all 6 views with index and text labels.

Usage:
    python scripts/visualize_gt_keypoints.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--frame FRAME]

Output:
    - view_{i}_all_keypoints.png: Individual view with all 22 keypoints
    - all_views_22keypoints_frame{N}.png: 2x3 grid of all views
    - view1_detailed_labels.png: Detailed view with full labels
    - keypoint_legend.png: Color legend for keypoint indices
"""

import os
import sys
import argparse
import pickle
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# GT Keypoint labels (22 keypoints) - Based on mouse_22_defs.py
# NOTE: GT annotation order differs from model definition!
# GT: 0=L_ear, 1=R_ear, 2=nose
# Model: 0=nose, 1=L_ear, 2=R_ear
GT_KEYPOINT_LABELS = {
    0: 'L_ear',        # GT defines idx 0 as left_ear_tip
    1: 'R_ear',        # GT defines idx 1 as right_ear_tip
    2: 'nose',         # GT defines idx 2 as nose
    3: 'neck',
    4: 'body',
    5: 'tail_root',
    6: 'tail_mid',
    7: 'tail_end',
    8: 'L_paw',        # left front paw
    9: 'L_paw_end',    # left front paw tip
    10: 'L_elbow',     # left elbow
    11: 'L_shoulder',  # left shoulder
    12: 'R_paw',       # right front paw
    13: 'R_paw_end',   # right front paw tip
    14: 'R_elbow',     # right elbow
    15: 'R_shoulder',  # right shoulder
    16: 'L_foot',      # left hind foot
    17: 'L_knee',      # left knee
    18: 'L_hip',       # left hip
    19: 'R_foot',      # right hind foot
    20: 'R_knee',      # right knee
    21: 'R_hip',       # right hip
}

# Alias for backward compatibility
KEYPOINT_LABELS = GT_KEYPOINT_LABELS

# Color coding by body part (BGR format)
COLORS = {
    'head': (0, 255, 255),      # Yellow - indices 0, 1, 2
    'body': (255, 0, 255),      # Magenta - indices 3, 4
    'tail': (0, 165, 255),      # Orange - indices 5, 6, 7
    'L_front': (255, 0, 0),     # Blue - indices 8, 9, 10, 11
    'R_front': (0, 255, 0),     # Green - indices 12, 13, 14, 15
    'L_hind': (255, 255, 0),    # Cyan - indices 16, 17, 18
    'R_hind': (0, 0, 255),      # Red - indices 19, 20, 21
}

def get_color_for_index(idx):
    """Get color based on keypoint index."""
    if idx in [0, 1, 2]:
        return COLORS['head']
    elif idx in [3, 4]:
        return COLORS['body']
    elif idx in [5, 6, 7]:
        return COLORS['tail']
    elif idx in [8, 9, 10, 11]:
        return COLORS['L_front']
    elif idx in [12, 13, 14, 15]:
        return COLORS['R_front']
    elif idx in [16, 17, 18]:
        return COLORS['L_hind']
    elif idx in [19, 20, 21]:
        return COLORS['R_hind']
    return (255, 255, 255)  # White default


def draw_keypoints_with_labels(img, keypoints_2d, confidence=None,
                                show_labels=True, show_confidence=True,
                                min_confidence=0.25):
    """
    Draw all 22 keypoints with index:label annotations.

    Args:
        img: Image to draw on (BGR)
        keypoints_2d: 2D keypoints array [22, 2] or [22, 3] with confidence
        confidence: Optional confidence array [22] (if not in keypoints_2d)
        show_labels: Whether to show text labels
        show_confidence: Whether to show confidence values
        min_confidence: Minimum confidence to display keypoint

    Returns:
        Annotated image
    """
    img = img.copy()

    # Extract confidence if embedded in keypoints
    if keypoints_2d.shape[1] == 3:
        if confidence is None:
            confidence = keypoints_2d[:, 2]
        keypoints_2d = keypoints_2d[:, :2]

    for idx in range(22):
        x, y = keypoints_2d[idx]

        # Skip invalid points
        if np.isnan(x) or np.isnan(y) or (x == 0 and y == 0):
            continue

        # Skip low confidence points
        if confidence is not None and confidence[idx] < min_confidence:
            continue

        # Get color for this keypoint
        color = get_color_for_index(idx)

        # Draw circle
        p = (int(x), int(y))
        cv2.circle(img, p, 8, color, -1)
        cv2.circle(img, p, 8, (0, 0, 0), 1)  # Black border

        if show_labels:
            label = KEYPOINT_LABELS.get(idx, str(idx))

            # Build text
            if show_confidence and confidence is not None:
                text = f"{idx}:{label}({confidence[idx]:.2f})"
            else:
                text = f"{idx}:{label}"

            # Draw text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            text_x = int(x) + 10
            text_y = int(y) - 5

            # Background rectangle
            cv2.rectangle(img,
                (text_x - 2, text_y - text_h - 2),
                (text_x + text_w + 2, text_y + 2),
                (0, 0, 0), -1)

            # Text
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

    return img


def create_legend(width=400, height=600):
    """Create a legend image showing all keypoint colors and labels."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # Title
    cv2.putText(img, "22 Keypoint Legend", (20, 35), font, 0.8, (255, 255, 255), 2)

    y = 70
    line_height = 25

    for idx in range(22):
        color = get_color_for_index(idx)
        label = KEYPOINT_LABELS.get(idx, str(idx))

        # Draw colored circle
        cv2.circle(img, (30, y), 8, color, -1)

        # Draw text
        text = f"{idx}: {label}"
        cv2.putText(img, text, (50, y + 5), font, font_scale, (255, 255, 255), thickness)

        y += line_height

    return img


def load_gt_keypoints(pkl_path):
    """Load GT keypoints from result pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # GT keypoints are stored in 'keypoints_2d' with shape [N_frames, 22, 3]
    if 'keypoints_2d' in data:
        return data['keypoints_2d']
    elif 'target_2d' in data:
        return data['target_2d']
    else:
        raise KeyError(f"No keypoints found in {pkl_path}. Keys: {data.keys()}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GT keypoints on all views')
    parser.add_argument('--data_dir', type=str,
                        default='/home/joon/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf',
                        help='Dataset directory containing keypoints2d_undist/ and images/')
    parser.add_argument('--output_dir', type=str,
                        default='/home/joon/dev/MAMMAL_mouse/results/keypoint_visualization',
                        help='Output directory for visualization images')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to visualize')
    parser.add_argument('--no_labels', action='store_true',
                        help='Disable text labels')
    parser.add_argument('--no_confidence', action='store_true',
                        help='Disable confidence display')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = Path(args.data_dir)

    # Check for keypoints2d_undist directory (raw GT keypoints)
    kp_dir = data_dir / 'keypoints2d_undist'
    img_dir = data_dir / 'images'

    if kp_dir.exists():
        print(f"Loading GT keypoints from: {kp_dir}")
        view_files = sorted(kp_dir.glob('result_view_*.pkl'))
        use_raw_format = True
    else:
        # Try result_view_*.pkl in data_dir directly
        view_files = sorted(data_dir.glob('result_view_*.pkl'))
        use_raw_format = False

        if not view_files:
            # Search in outputs
            for subdir in Path('/home/joon/dev/MAMMAL_mouse/outputs').iterdir():
                if subdir.is_dir():
                    view_files = sorted(subdir.glob('result_view_*.pkl'))
                    if view_files:
                        print(f"Found files in {subdir}")
                        data_dir = subdir
                        break

    if not view_files:
        print("ERROR: No result files found.")
        print(f"Looked in: {data_dir}")
        print("Please specify --data_dir pointing to a dataset with keypoints2d_undist/")
        return

    print(f"Found {len(view_files)} view files")

    # Process each view
    view_images = []

    for view_idx, pkl_path in enumerate(view_files):
        print(f"Processing view {view_idx}: {pkl_path.name}")

        # Load keypoints
        with open(pkl_path, 'rb') as f:
            kp_data = pickle.load(f)

        # Handle different data formats
        if use_raw_format:
            # Raw format: numpy array [N_frames, 22, 3]
            if isinstance(kp_data, np.ndarray):
                keypoints = kp_data
            else:
                keypoints = kp_data.get('keypoints_2d', kp_data)

            if args.frame < len(keypoints):
                kp_2d = keypoints[args.frame]  # [22, 3]
            else:
                kp_2d = keypoints[0]

            # Load image from images or video directory
            img = None

            # Try images directory first
            if img_dir.exists():
                img_files = sorted((img_dir / f'view_{view_idx}').glob('*.png'))
                if not img_files:
                    img_files = sorted((img_dir / f'view_{view_idx}').glob('*.jpg'))
                if not img_files:
                    all_imgs = sorted(img_dir.glob(f'*view_{view_idx}*.png'))
                    if all_imgs:
                        img_files = all_imgs

                if img_files and args.frame < len(img_files):
                    img = cv2.imread(str(img_files[args.frame]))
                elif img_files:
                    img = cv2.imread(str(img_files[0]))

            # Try video directory if no images found
            if img is None:
                video_dir = data_dir / 'videos_undist'
                video_path = video_dir / f'{view_idx}.mp4'
                if video_path.exists():
                    cap = cv2.VideoCapture(str(video_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
                    ret, img = cap.read()
                    cap.release()
                    if not ret:
                        print(f"  Warning: Could not read frame {args.frame} from {video_path}")

            if img is None:
                print(f"  Warning: No images found for view {view_idx}")
                # Create blank image
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            # Result format: dict with 'imgs' and 'keypoints_2d'
            if 'imgs' in kp_data:
                imgs = kp_data['imgs']
                img = imgs[args.frame] if args.frame < len(imgs) else imgs[0]
            else:
                print(f"  Warning: No images in {pkl_path.name}")
                continue

            if 'keypoints_2d' in kp_data:
                keypoints = kp_data['keypoints_2d']
                kp_2d = keypoints[args.frame] if args.frame < len(keypoints) else keypoints[0]
            else:
                print(f"  Warning: No keypoints_2d in {pkl_path.name}")
                continue

        # Draw keypoints
        img_annotated = draw_keypoints_with_labels(
            img, kp_2d,
            show_labels=not args.no_labels,
            show_confidence=not args.no_confidence
        )

        # Add view label
        cv2.putText(img_annotated, f"View {view_idx} - GT Keypoints", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Save individual view
        output_path = os.path.join(args.output_dir, f"view_{view_idx}_gt_keypoints.png")
        cv2.imwrite(output_path, img_annotated)
        print(f"  Saved: {output_path}")

        view_images.append(img_annotated)

    # Create grid of all views (2x3)
    if len(view_images) >= 2:
        # Resize all images to same size
        target_h, target_w = 480, 640
        resized = []
        for img in view_images:
            resized.append(cv2.resize(img, (target_w, target_h)))

        # Pad to 6 images if needed
        while len(resized) < 6:
            resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

        # Create 2x3 grid
        row1 = np.hstack(resized[:3])
        row2 = np.hstack(resized[3:6])
        grid = np.vstack([row1, row2])

        grid_path = os.path.join(args.output_dir, f"all_views_22keypoints_frame{args.frame}.png")
        cv2.imwrite(grid_path, grid)
        print(f"Saved grid: {grid_path}")

    # Create and save legend
    legend = create_legend()
    legend_path = os.path.join(args.output_dir, "keypoint_legend.png")
    cv2.imwrite(legend_path, legend)
    print(f"Saved legend: {legend_path}")

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
