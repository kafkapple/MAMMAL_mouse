#!/usr/bin/env python
"""
Generate qualitative comparison images for ablation study.

Creates side-by-side comparison of rendered meshes across different
view/keypoint configurations.
"""

import argparse
import glob
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple


def find_render_image(result_dir: str, frame_idx: int) -> str:
    """Find render image for given frame."""
    patterns = [
        f'render/step_2_frame_{frame_idx:06d}.png',
        f'render/step_2/frame_{frame_idx:06d}.png',
    ]

    for pattern in patterns:
        path = os.path.join(result_dir, pattern)
        if os.path.exists(path):
            return path
    return None


def parse_experiment_label(exp_dir: str) -> str:
    """Generate a short label for the experiment."""
    name = os.path.basename(exp_dir)

    # Extract views
    import re
    view_match = re.search(r'_v(\d+)_', name)
    views = len(view_match.group(1)) if view_match else '?'

    # Extract keypoints
    if 'kp22' in name:
        kps = '22kp'
    elif 'sparse9' in name:
        kps = '9kp'
    elif 'sparse7' in name:
        kps = '7kp'
    elif 'sparse5' in name:
        kps = '5kp'
    elif 'sparse3' in name:
        kps = '3kp'
    else:
        kps = '?kp'

    return f"{views}V/{kps}"


def create_comparison_grid(experiments: List[Tuple[str, str]],
                          frame_indices: List[int],
                          output_path: str,
                          title: str = "Ablation Study: Qualitative Comparison"):
    """
    Create a comparison grid with experiments as columns and frames as rows.

    Args:
        experiments: List of (exp_dir, label) tuples
        frame_indices: Frame indices to show
        output_path: Output image path
        title: Title for the figure
    """
    # Load all images
    images = []  # [frame][exp]

    for frame_idx in frame_indices:
        row = []
        for exp_dir, label in experiments:
            img_path = find_render_image(exp_dir, frame_idx)
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    row.append((img, label))
                else:
                    row.append((None, label))
            else:
                row.append((None, label))
        images.append((frame_idx, row))

    # Filter out frames with no images
    images = [(f, r) for f, r in images if any(img is not None for img, _ in r)]

    if not images:
        print("No images found!")
        return None

    # Determine grid size
    num_rows = len(images)
    num_cols = len(experiments)

    # Target cell size
    cell_width = 400
    cell_height = 300
    header_height = 40
    frame_label_width = 80

    # Create canvas
    canvas_width = frame_label_width + cell_width * num_cols
    canvas_height = header_height + cell_height * num_rows + 60  # Extra for title

    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Draw title
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (30, 30, 30), 2)

    # Draw column headers (experiment labels)
    for j, (_, label) in enumerate(experiments):
        x = frame_label_width + j * cell_width + cell_width // 2 - 30
        y = 60 + header_height // 2 + 5
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 50, 50), 2)

    # Draw grid
    y_offset = 60 + header_height

    for i, (frame_idx, row) in enumerate(images):
        # Draw frame label
        cv2.putText(canvas, f"F{frame_idx:03d}", (10, y_offset + cell_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        for j, (img, _) in enumerate(row):
            x = frame_label_width + j * cell_width
            y = y_offset

            if img is not None:
                # Resize image to fit cell
                h, w = img.shape[:2]
                scale = min(cell_width / w, cell_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h))

                # Center in cell
                x_off = (cell_width - new_w) // 2
                y_off = (cell_height - new_h) // 2

                canvas[y + y_off:y + y_off + new_h, x + x_off:x + x_off + new_w] = img_resized
            else:
                # Draw placeholder
                cv2.rectangle(canvas, (x + 5, y + 5), (x + cell_width - 5, y + cell_height - 5),
                            (200, 200, 200), 2)
                cv2.putText(canvas, "No image", (x + cell_width // 2 - 30, y + cell_height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Draw cell border
            cv2.rectangle(canvas, (x, y), (x + cell_width, y + cell_height), (220, 220, 220), 1)

        y_offset += cell_height

    # Save
    cv2.imwrite(output_path, canvas)
    print(f"Comparison grid saved: {output_path}")

    return canvas


def create_row_comparison(experiments: List[Tuple[str, str]],
                         frame_idx: int,
                         output_path: str,
                         v2v_scores: dict = None):
    """
    Create a single-row comparison for one frame.
    """
    images_data = []

    for exp_dir, label in experiments:
        img_path = find_render_image(exp_dir, frame_idx)
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images_data.append((img, label, exp_dir))

    if not images_data:
        print(f"No images found for frame {frame_idx}")
        return None

    # Resize all images to same height
    target_height = 300
    resized = []

    for img, label, exp_dir in images_data:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        img_resized = cv2.resize(img, (new_w, target_height))
        resized.append((img_resized, label, exp_dir))

    # Add labels
    header_height = 50
    labeled = []

    for img, label, exp_dir in resized:
        h, w = img.shape[:2]
        new_img = np.ones((h + header_height, w, 3), dtype=np.uint8) * 255
        new_img[header_height:, :] = img

        # Draw label
        cv2.putText(new_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 30, 30), 2)

        # Add V2V score if available
        if v2v_scores:
            exp_name = os.path.basename(exp_dir)
            if exp_name in v2v_scores:
                v2v = v2v_scores[exp_name]
                cv2.putText(new_img, f"V2V: {v2v:.2f}mm", (10, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        labeled.append(new_img)

    # Stack horizontally
    result = np.hstack(labeled)

    # Add title
    title_height = 40
    final = np.ones((result.shape[0] + title_height, result.shape[1], 3), dtype=np.uint8) * 240
    final[title_height:, :] = result
    cv2.putText(final, f"Qualitative Comparison - Frame {frame_idx}",
               (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

    cv2.imwrite(output_path, final)
    print(f"Row comparison saved: {output_path}")

    return final


def main():
    parser = argparse.ArgumentParser(description='Generate qualitative comparison images')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Experiment directories (in order of display)')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 25, 50, 75],
                       help='Frame indices to compare')
    parser.add_argument('--output', default='docs/reports/qualitative_comparison.png',
                       help='Output image path')
    parser.add_argument('--mode', choices=['grid', 'row'], default='grid',
                       help='Comparison mode: grid (frames x experiments) or row (single frame)')
    parser.add_argument('--json', type=str, default=None,
                       help='Path to JSON results file for V2V scores')
    args = parser.parse_args()

    # Resolve experiments
    all_exp_dirs = []
    for pattern in args.experiments:
        matches = glob.glob(pattern)
        all_exp_dirs.extend(matches)
    all_exp_dirs = sorted(set(all_exp_dirs))

    if not all_exp_dirs:
        print("No experiment directories found!")
        return

    # Create experiment list with labels
    experiments = [(d, parse_experiment_label(d)) for d in all_exp_dirs]

    print(f"Found {len(experiments)} experiments:")
    for d, label in experiments:
        print(f"  {label}: {os.path.basename(d)}")

    # Load V2V scores if available
    v2v_scores = {}
    if args.json and os.path.exists(args.json):
        import json
        with open(args.json) as f:
            data = json.load(f)
        for item in data:
            v2v_scores[item['name']] = item['metrics']['v2v_mean']

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.mode == 'grid':
        create_comparison_grid(experiments, args.frames, args.output)
    else:
        for frame_idx in args.frames:
            output = args.output.replace('.png', f'_frame{frame_idx:03d}.png')
            create_row_comparison(experiments, frame_idx, output, v2v_scores)


if __name__ == '__main__':
    main()
