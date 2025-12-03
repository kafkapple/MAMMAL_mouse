#!/usr/bin/env python
"""
Compare results across multiple experiments.

Creates HTML report with:
- Vertical layout (one experiment per row)
- Images + quantitative metrics
- Configuration summary

Usage:
    python scripts/compare_experiments.py \
        results/fitting/exp1_* \
        results/fitting/exp2_* \
        --output comparison.html

    # PNG output (legacy)
    python scripts/compare_experiments.py \
        results/fitting/exp1_* \
        --output comparison.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(result_dir):
    """Load experiment config from result directory."""
    config_path = os.path.join(result_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def get_experiment_info(result_dir):
    """Extract experiment info from directory name and config."""
    info = {
        'name': os.path.basename(result_dir),
        'views': '?',
        'keypoints': '?',
        'frames': 0,
    }

    # Parse from directory name
    name = info['name']
    if '_v' in name:
        # Extract views from v012345 pattern
        import re
        match = re.search(r'_v(\d+)_', name)
        if match:
            info['views'] = len(match.group(1))

    if 'sparse' in name:
        info['keypoints'] = 'sparse 3'
    elif 'noKP' in name or 'no_keypoint' in name:
        info['keypoints'] = 'none'
    else:
        info['keypoints'] = '22 (full)'

    # Count frames from obj files
    obj_files = glob.glob(os.path.join(result_dir, 'obj', '*.obj'))
    info['frames'] = len(obj_files)

    # Load config for more details
    config = load_config(result_dir)
    if config:
        if 'data' in config and 'views_to_use' in config['data']:
            info['views'] = len(config['data']['views_to_use'])
            info['view_ids'] = config['data']['views_to_use']
        if 'fitter' in config:
            if 'sparse_keypoint_indices' in config['fitter']:
                info['keypoints'] = f"sparse {len(config['fitter']['sparse_keypoint_indices'])}"
            elif config['fitter'].get('use_keypoints', True) is False:
                info['keypoints'] = 'none'

    return info


def find_render_image(result_dir, frame_idx, step='step_2'):
    """Find render image for given frame and step."""
    patterns = [
        f'render/{step}_frame_{frame_idx:06d}.png',
        f'render/fitting_{frame_idx}.png',  # Old naming
        f'render/fitting_{frame_idx}_sil.png',  # Old naming step2
    ]

    for pattern in patterns:
        path = os.path.join(result_dir, pattern)
        if os.path.exists(path):
            return path

    return None


def create_comparison_gallery(result_dirs, frame_idx=0, output_path='comparison.png'):
    """Create comparison gallery from multiple experiments."""
    images = []
    labels = []

    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            continue

        info = get_experiment_info(result_dir)
        img_path = find_render_image(result_dir, frame_idx)

        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            images.append(img)

            # Create label
            label = f"Views: {info['views']} | KP: {info['keypoints']}"
            labels.append(label)
        else:
            print(f"Warning: No render image found for {result_dir}")

    if not images:
        print("No images found!")
        return None

    # Resize all images to same height
    target_height = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        scale = target_height / img.shape[0]
        new_width = int(img.shape[1] * scale)
        resized.append(cv2.resize(img, (new_width, target_height)))

    # Add labels
    labeled_images = []
    for img, label in zip(resized, labels):
        # Add header
        header_h = 40
        new_img = np.zeros((img.shape[0] + header_h, img.shape[1], 3), dtype=np.uint8)
        new_img[:header_h, :] = (40, 40, 40)
        new_img[header_h:, :] = img

        # Draw label
        cv2.putText(new_img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        labeled_images.append(new_img)

    # Stack horizontally
    gallery = np.hstack(labeled_images)

    # Add title
    title_h = 50
    final = np.zeros((gallery.shape[0] + title_h, gallery.shape[1], 3), dtype=np.uint8)
    final[:title_h, :] = (60, 60, 60)
    final[title_h:, :] = gallery

    title = f"Experiment Comparison - Frame {frame_idx}"
    cv2.putText(final, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    cv2.imwrite(output_path, final)
    print(f"Comparison saved to: {output_path}")

    return final


def create_summary_table(result_dirs, output_path='summary.txt'):
    """Create text summary of all experiments."""
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT COMPARISON SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Experiment':<50} {'Views':<8} {'Keypoints':<12} {'Frames':<8}")
    lines.append("-" * 80)

    for result_dir in sorted(result_dirs):
        if not os.path.isdir(result_dir):
            continue

        info = get_experiment_info(result_dir)
        name = info['name'][:48] if len(info['name']) > 48 else info['name']
        lines.append(f"{name:<50} {info['views']:<8} {info['keypoints']:<12} {info['frames']:<8}")

    lines.append("-" * 80)
    lines.append("")

    summary = "\n".join(lines)
    print(summary)

    with open(output_path, 'w') as f:
        f.write(summary)

    return summary


def create_html_report(result_dirs, frame_idx=0, output_path='comparison.html'):
    """Create HTML report with vertical layout."""
    import base64
    from datetime import datetime

    experiments = []
    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            continue

        info = get_experiment_info(result_dir)
        img_path = find_render_image(result_dir, frame_idx)

        if img_path and os.path.exists(img_path):
            # Read and encode image as base64
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            info['image_data'] = img_data
            info['image_path'] = img_path

            # Load loss history if available
            loss_path = os.path.join(result_dir, 'loss_history.json')
            if os.path.exists(loss_path):
                import json
                with open(loss_path, 'r') as f:
                    loss_data = json.load(f)
                if loss_data:
                    last_loss = loss_data[-1] if isinstance(loss_data, list) else loss_data
                    info['final_loss'] = last_loss.get('total', 'N/A')

            experiments.append(info)
        else:
            print(f"Warning: No render image found for {result_dir}")

    if not experiments:
        print("No experiments found!")
        return None

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Comparison Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .summary {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .experiment {{ background: white; margin-bottom: 20px; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .exp-header {{ background: #2c3e50; color: white; padding: 15px; }}
        .exp-header h3 {{ margin: 0; font-size: 14px; }}
        .exp-content {{ display: flex; flex-direction: column; }}
        .exp-image {{ width: 100%; max-height: 400px; object-fit: contain; background: #eee; }}
        .exp-info {{ padding: 15px; }}
        .exp-info table {{ width: 100%; border-collapse: collapse; }}
        .exp-info td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .exp-info td:first-child {{ font-weight: bold; width: 120px; color: #666; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Experiment Comparison Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Frame: {frame_idx}</p>

    <div class="summary">
        <h2>Summary</h2>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background:#2c3e50; color:white;">
                <th style="padding:10px; text-align:left;">Views</th>
                <th style="padding:10px; text-align:left;">Keypoints</th>
                <th style="padding:10px; text-align:left;">Frames</th>
                <th style="padding:10px; text-align:left;">Experiment</th>
            </tr>
"""

    for i, exp in enumerate(experiments):
        bg = '#f9f9f9' if i % 2 == 0 else 'white'
        html += f"""            <tr style="background:{bg};">
                <td style="padding:10px;">{exp['views']}</td>
                <td style="padding:10px;">{exp['keypoints']}</td>
                <td style="padding:10px;">{exp['frames']}</td>
                <td style="padding:10px; font-size:12px;">{exp['name'][:60]}</td>
            </tr>
"""

    html += """        </table>
    </div>
"""

    for exp in experiments:
        final_loss = exp.get('final_loss', 'N/A')
        html += f"""
    <div class="experiment">
        <div class="exp-header">
            <h3>Views: {exp['views']} | Keypoints: {exp['keypoints']}</h3>
        </div>
        <div class="exp-content">
            <img class="exp-image" src="data:image/png;base64,{exp['image_data']}" alt="Render">
            <div class="exp-info">
                <table>
                    <tr><td>Experiment</td><td style="font-size:11px;">{exp['name']}</td></tr>
                    <tr><td>Views</td><td class="metric">{exp['views']}</td></tr>
                    <tr><td>Keypoints</td><td>{exp['keypoints']}</td></tr>
                    <tr><td>Frames</td><td>{exp['frames']}</td></tr>
                    <tr><td>Final Loss</td><td>{final_loss}</td></tr>
                </table>
            </div>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")
    return html


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('result_dirs', nargs='+', help='Result directories (supports glob patterns)')
    parser.add_argument('--output', '-o', default='comparison.html', help='Output path (.html or .png)')
    parser.add_argument('--frame', '-f', type=int, default=0, help='Frame index to compare')
    parser.add_argument('--summary', '-s', action='store_true', help='Also create text summary')
    args = parser.parse_args()

    # Expand glob patterns
    all_dirs = []
    for pattern in args.result_dirs:
        matches = glob.glob(pattern)
        all_dirs.extend(matches)

    all_dirs = sorted(set(all_dirs))

    if not all_dirs:
        print("No result directories found!")
        sys.exit(1)

    print(f"Found {len(all_dirs)} experiments:")
    for d in all_dirs:
        print(f"  - {os.path.basename(d)}")
    print()

    # Create output based on extension
    if args.output.endswith('.html'):
        create_html_report(all_dirs, args.frame, args.output)
    else:
        create_comparison_gallery(all_dirs, args.frame, args.output)

    # Create summary if requested
    if args.summary:
        ext = '.html' if args.output.endswith('.html') else '.png'
        summary_path = args.output.replace(ext, '_summary.txt')
        create_summary_table(all_dirs, summary_path)


if __name__ == '__main__':
    main()
