#!/usr/bin/env python3
"""
High-resolution GT keypoint visualization with improved label positioning.

Features:
- High resolution output (no downsampling)
- Smart label positioning to avoid overlap
- Confidence values displayed for all keypoints
- Semi-transparent backgrounds for readability
- Larger fonts for grid view

Usage:
    python scripts/visualize_gt_keypoints_hires.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--frame FRAME]
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
GT_KEYPOINT_LABELS = {
    0: 'L_ear',        # left_ear_tip
    1: 'R_ear',        # right_ear_tip
    2: 'nose',         # nose
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
    return (255, 255, 255)


def get_label_position(x, y, idx, img_w, img_h, occupied_regions, text_w, text_h):
    """
    Find optimal label position to avoid overlaps.

    Tries multiple positions around the keypoint and picks the first available.
    """
    # Define possible offsets (prioritized)
    offsets = [
        (15, -15),   # top-right (default)
        (15, 25),    # bottom-right
        (-text_w - 15, -15),  # top-left
        (-text_w - 15, 25),   # bottom-left
        (15, -35),   # further top-right
        (15, 45),    # further bottom-right
        (-text_w - 15, -35),  # further top-left
        (-text_w - 15, 45),   # further bottom-left
        (30, 0),     # right
        (-text_w - 30, 0),  # left
    ]

    for dx, dy in offsets:
        tx = int(x + dx)
        ty = int(y + dy)

        # Check bounds
        if tx < 5 or tx + text_w > img_w - 5:
            continue
        if ty - text_h < 5 or ty > img_h - 5:
            continue

        # Check overlap with existing labels
        new_region = (tx - 2, ty - text_h - 2, tx + text_w + 2, ty + 2)
        overlap = False
        for region in occupied_regions:
            if (new_region[0] < region[2] and new_region[2] > region[0] and
                new_region[1] < region[3] and new_region[3] > region[1]):
                overlap = True
                break

        if not overlap:
            return tx, ty, new_region

    # Fallback: use default position
    tx = int(x + 15)
    ty = int(y - 15)
    return tx, ty, (tx - 2, ty - text_h - 2, tx + text_w + 2, ty + 2)


def draw_keypoints_hires(img, keypoints_2d, font_scale=0.6, circle_radius=12,
                          show_confidence=True, alpha=0.7, show_labels=True):
    """
    Draw all 22 keypoints with high-quality labels and smart positioning.

    Args:
        img: Image to draw on (BGR)
        keypoints_2d: 2D keypoints array [22, 2] or [22, 3] with confidence
        font_scale: Font size multiplier
        circle_radius: Keypoint circle radius
        show_confidence: Whether to show confidence values
        alpha: Background transparency (0-1)
        show_labels: If False, only show colored circles with index numbers (no text labels)

    Returns:
        Annotated image
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Extract confidence if embedded
    if keypoints_2d.shape[1] == 3:
        confidence = keypoints_2d[:, 2]
        keypoints_2d = keypoints_2d[:, :2]
    else:
        confidence = np.ones(22)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Collect all label info first for smart positioning
    labels_info = []
    occupied_regions = []

    for idx in range(22):
        x, y = keypoints_2d[idx]

        # Skip invalid points (but still show low confidence)
        if np.isnan(x) or np.isnan(y) or (x == 0 and y == 0):
            continue

        label = GT_KEYPOINT_LABELS.get(idx, str(idx))
        conf = confidence[idx] if idx < len(confidence) else 0.0

        # Prepare text only if showing labels
        if show_labels:
            if show_confidence:
                text = f"{idx}:{label}({conf:.2f})"
            else:
                text = f"{idx}:{label}"
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            # Find optimal position
            tx, ty, region = get_label_position(x, y, idx, w, h, occupied_regions, text_w, text_h)
            occupied_regions.append(region)
        else:
            text = ""
            text_w, text_h = 0, 0
            tx, ty = int(x), int(y)

        labels_info.append({
            'idx': idx,
            'x': x, 'y': y,
            'tx': tx, 'ty': ty,
            'text': text,
            'text_w': text_w, 'text_h': text_h,
            'color': get_color_for_index(idx),
            'conf': conf
        })

    # Draw all elements
    # First pass: draw semi-transparent backgrounds (only if showing labels)
    if show_labels:
        overlay = img.copy()
        for info in labels_info:
            # Background rectangle
            cv2.rectangle(overlay,
                (info['tx'] - 3, info['ty'] - info['text_h'] - 3),
                (info['tx'] + info['text_w'] + 3, info['ty'] + 3),
                (20, 20, 20), -1)

        # Blend overlay
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Second pass: draw circles and text
    for info in labels_info:
        color = info['color']

        # Draw connecting line (subtle) - only if showing labels
        if show_labels:
            cv2.line(img, (int(info['x']), int(info['y'])),
                    (info['tx'], info['ty']), color, 1, cv2.LINE_AA)

        # Draw keypoint circle
        cv2.circle(img, (int(info['x']), int(info['y'])), circle_radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(info['x']), int(info['y'])), circle_radius, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw keypoint index inside circle
        idx_text = str(info['idx'])
        (idx_w, idx_h), _ = cv2.getTextSize(idx_text, font, font_scale * 0.6, 1)
        idx_x = int(info['x'] - idx_w / 2)
        idx_y = int(info['y'] + idx_h / 2)
        cv2.putText(img, idx_text, (idx_x, idx_y), font, font_scale * 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, idx_text, (idx_x, idx_y), font, font_scale * 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw label text (only if showing labels)
        if show_labels and info['text']:
            cv2.putText(img, info['text'], (info['tx'], info['ty']),
                       font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(img, info['text'], (info['tx'], info['ty']),
                       font, font_scale, color, thickness, cv2.LINE_AA)

    return img


def create_legend_hires(width=500, height=700):
    """Create a high-resolution legend image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Title
    cv2.putText(img, "22 GT Keypoint Legend", (20, 45), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "(Based on mouse_22_defs.py)", (20, 75), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    y = 110
    line_height = 28

    for idx in range(22):
        color = get_color_for_index(idx)
        label = GT_KEYPOINT_LABELS.get(idx, str(idx))

        # Draw colored circle
        cv2.circle(img, (35, y), 10, color, -1, cv2.LINE_AA)
        cv2.circle(img, (35, y), 10, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw text
        text = f"{idx:2d}: {label}"
        cv2.putText(img, text, (55, y + 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y += line_height

    # Add body part legend
    y += 20
    cv2.putText(img, "Body Parts:", (20, y), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y += 25

    parts = [
        ('Head (0-2)', COLORS['head']),
        ('Body (3-4)', COLORS['body']),
        ('Tail (5-7)', COLORS['tail']),
        ('L Front (8-11)', COLORS['L_front']),
        ('R Front (12-15)', COLORS['R_front']),
        ('L Hind (16-18)', COLORS['L_hind']),
        ('R Hind (19-21)', COLORS['R_hind']),
    ]

    for part_name, color in parts:
        cv2.circle(img, (35, y), 8, color, -1, cv2.LINE_AA)
        cv2.putText(img, part_name, (55, y + 5), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22

    return img


def main():
    parser = argparse.ArgumentParser(description='High-resolution GT keypoint visualization')
    parser.add_argument('--data_dir', type=str,
                        default='/home/joon/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf',
                        help='Dataset directory containing keypoints2d_undist/ and videos_undist/')
    parser.add_argument('--output_dir', type=str,
                        default='/home/joon/dev/MAMMAL_mouse/results/keypoint_visualization_hires',
                        help='Output directory for visualization images')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to visualize')
    parser.add_argument('--font_scale', type=float, default=0.55,
                        help='Font scale for labels')
    parser.add_argument('--circle_radius', type=int, default=10,
                        help='Keypoint circle radius')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = Path(args.data_dir)
    kp_dir = data_dir / 'keypoints2d_undist'

    if not kp_dir.exists():
        print(f"ERROR: keypoints2d_undist not found in {data_dir}")
        return

    view_files = sorted(kp_dir.glob('result_view_*.pkl'))
    print(f"Found {len(view_files)} view files in {kp_dir}")

    view_images = []

    for view_idx, pkl_path in enumerate(view_files):
        print(f"Processing view {view_idx}: {pkl_path.name}")

        # Load keypoints
        with open(pkl_path, 'rb') as f:
            kp_data = pickle.load(f)

        if isinstance(kp_data, np.ndarray):
            keypoints = kp_data
        else:
            keypoints = kp_data.get('keypoints_2d', kp_data)

        kp_2d = keypoints[args.frame] if args.frame < len(keypoints) else keypoints[0]

        # Load image from video
        video_dir = data_dir / 'videos_undist'
        video_path = video_dir / f'{view_idx}.mp4'

        img = None
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
            ret, img = cap.read()
            cap.release()

        if img is None:
            print(f"  Warning: Could not read frame from video, creating blank image")
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Draw keypoints with high quality (full labels version)
        img_annotated = draw_keypoints_hires(
            img, kp_2d,
            font_scale=args.font_scale,
            circle_radius=args.circle_radius,
            show_confidence=True,
            alpha=0.6,
            show_labels=True
        )

        # Add view title
        h, w = img_annotated.shape[:2]
        title = f"View {view_idx} - GT Keypoints (Frame {args.frame})"
        cv2.putText(img_annotated, title, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_annotated, title, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Save individual view at full resolution (with labels)
        output_path = os.path.join(args.output_dir, f"view_{view_idx}_gt_hires.png")
        cv2.imwrite(output_path, img_annotated, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"  Saved: {output_path} ({w}x{h})")

        # Draw keypoints without labels (clean version - numbers + colors only)
        img_clean = draw_keypoints_hires(
            img, kp_2d,
            font_scale=args.font_scale,
            circle_radius=args.circle_radius,
            show_confidence=False,
            alpha=0.6,
            show_labels=False
        )

        # Add minimal title for clean version
        cv2.putText(img_clean, f"View {view_idx}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_clean, f"View {view_idx}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Save clean version (numbers + colors only)
        clean_path = os.path.join(args.output_dir, f"view_{view_idx}_clean.png")
        cv2.imwrite(clean_path, img_clean, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"  Saved clean version: {clean_path}")

        view_images.append(img_annotated)

    # Create high-resolution grid (2x3)
    if len(view_images) >= 2:
        # Use full resolution for grid
        max_h = max(img.shape[0] for img in view_images)
        max_w = max(img.shape[1] for img in view_images)

        # Ensure consistent size
        resized = []
        for img in view_images:
            if img.shape[0] != max_h or img.shape[1] != max_w:
                img = cv2.resize(img, (max_w, max_h), interpolation=cv2.INTER_LANCZOS4)
            resized.append(img)

        # Pad to 6 if needed
        while len(resized) < 6:
            resized.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))

        # Create 2x3 grid
        row1 = np.hstack(resized[:3])
        row2 = np.hstack(resized[3:6])
        grid = np.vstack([row1, row2])

        grid_path = os.path.join(args.output_dir, f"all_views_grid_frame{args.frame}.png")
        cv2.imwrite(grid_path, grid, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"Saved grid: {grid_path} ({grid.shape[1]}x{grid.shape[0]})")

    # Create and save high-res legend
    legend = create_legend_hires()
    legend_path = os.path.join(args.output_dir, "keypoint_legend_hires.png")
    cv2.imwrite(legend_path, legend, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Saved legend: {legend_path}")

    print(f"\nAll high-resolution visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
