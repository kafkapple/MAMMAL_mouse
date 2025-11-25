#!/usr/bin/env python
"""Visualize fitting results with GT RGB images side by side.

Usage:
    python scripts/utils/visualize_fitting_comparison.py \
        --results results/cropped_fitting_final \
        --gt_dir data/100-KO-male-56-20200615_cropped \
        --output results/cropped_fitting_final/gallery.png
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_comparison_gallery(results_dir, gt_dir, output_path, max_frames=10, cols=2):
    """Create a gallery comparing GT RGB with fitting results."""

    # Find all frame directories
    frame_dirs = sorted(glob.glob(os.path.join(results_dir, 'frame_*')))
    if not frame_dirs:
        print(f"No frame directories found in {results_dir}")
        return

    # Limit frames
    frame_dirs = frame_dirs[:max_frames]
    n_frames = len(frame_dirs)

    print(f"Found {n_frames} frames")

    # Create figure
    rows = (n_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 12, rows * 4))
    axes = np.atleast_2d(axes)

    for i, frame_dir in enumerate(frame_dirs):
        frame_name = os.path.basename(frame_dir)
        frame_idx = frame_name.replace('frame_', '')

        row = i // cols
        col = (i % cols) * 2

        # Load GT RGB
        gt_patterns = [
            os.path.join(gt_dir, f'frame_{frame_idx}_cropped.png'),
            os.path.join(gt_dir, f'{frame_idx}_cropped.png'),
            os.path.join(gt_dir, f'frame_{frame_idx}.png'),
        ]

        gt_img = None
        for pattern in gt_patterns:
            if os.path.exists(pattern):
                gt_img = Image.open(pattern)
                break

        # Load comparison image
        comparison_path = os.path.join(frame_dir, 'comparison.png')
        comp_img = Image.open(comparison_path) if os.path.exists(comparison_path) else None

        # Plot GT
        ax_gt = axes[row, col]
        if gt_img is not None:
            ax_gt.imshow(gt_img)
            ax_gt.set_title(f'GT RGB - {frame_name}', fontsize=10)
        else:
            ax_gt.text(0.5, 0.5, 'GT not found', ha='center', va='center')
        ax_gt.axis('off')

        # Plot comparison
        ax_comp = axes[row, col + 1]
        if comp_img is not None:
            ax_comp.imshow(comp_img)
            ax_comp.set_title(f'Fitting Result - {frame_name}', fontsize=10)
        else:
            ax_comp.text(0.5, 0.5, 'Comparison not found', ha='center', va='center')
        ax_comp.axis('off')

    # Hide unused axes
    for i in range(n_frames, rows * cols):
        row = i // cols
        col = (i % cols) * 2
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved gallery to {output_path}")
    plt.close()


def create_single_comparison(frame_dir, gt_dir, output_path):
    """Create comparison for a single frame."""

    frame_name = os.path.basename(frame_dir)
    frame_idx = frame_name.replace('frame_', '')

    # Load images
    gt_patterns = [
        os.path.join(gt_dir, f'frame_{frame_idx}_cropped.png'),
        os.path.join(gt_dir, f'{frame_idx}_cropped.png'),
    ]

    gt_img = None
    for pattern in gt_patterns:
        if os.path.exists(pattern):
            gt_img = Image.open(pattern)
            break

    comparison_path = os.path.join(frame_dir, 'comparison.png')
    comp_img = Image.open(comparison_path) if os.path.exists(comparison_path) else None

    if gt_img is None or comp_img is None:
        print(f"Missing images for {frame_name}")
        return

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(gt_img)
    axes[0].set_title('GT RGB', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(comp_img)
    axes[1].set_title('Fitting Result (Mask | Rendered | Overlay)', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize fitting results with GT')
    parser.add_argument('--results', type=str, required=True,
                        help='Results directory (e.g., results/cropped_fitting_final)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='GT data directory (e.g., data/100-KO-male-56-20200615_cropped)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: results_dir/gallery_with_gt.png)')
    parser.add_argument('--max_frames', type=int, default=10,
                        help='Maximum frames to include')
    parser.add_argument('--frame', type=str, default=None,
                        help='Specific frame to visualize (e.g., frame_000000)')

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results, 'gallery_with_gt.png')

    if args.frame:
        # Single frame comparison
        frame_dir = os.path.join(args.results, args.frame)
        create_single_comparison(frame_dir, args.gt_dir, args.output)
    else:
        # Gallery of all frames
        create_comparison_gallery(args.results, args.gt_dir, args.output, args.max_frames)


if __name__ == '__main__':
    main()
