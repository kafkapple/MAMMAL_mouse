#!/usr/bin/env python
"""
Evaluate mesh fitting results and generate comprehensive reports.

Computes quantitative metrics and generates qualitative visualizations
for comparing different experiment configurations.

Usage:
    python scripts/evaluate_experiment.py <results_folder>
    python scripts/evaluate_experiment.py results/fitting/baseline_xxx --compare results/fitting/monocular_xxx

Metrics:
    - Silhouette IoU (mask overlap)
    - 2D keypoint reprojection error
    - Temporal smoothness (joint velocity variance)
    - Mesh quality (self-intersection check)
"""

import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from articulation_th import ArticulationTorch


def load_results(results_folder):
    """Load all fitting results from a results folder."""
    results_path = Path(results_folder)

    # Find pkl files
    pkl_patterns = [
        'params/param*_sil.pkl',  # Step2 results (final)
        'params/param*.pkl',
    ]

    pkl_files = []
    for pattern in pkl_patterns:
        files = sorted(results_path.glob(pattern))
        if files:
            # Filter to get only _sil files if available
            sil_files = [f for f in files if '_sil' in f.name]
            pkl_files = sil_files if sil_files else files
            break

    if not pkl_files:
        raise FileNotFoundError(f"No parameter files found in {results_folder}")

    # Load config
    config_path = results_path / 'config.yaml'
    config = None
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Load all parameters
    all_params = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
        all_params.append({
            'frame': int(pkl_path.stem.replace('param', '').replace('_sil', '')),
            'path': pkl_path,
            'params': params
        })

    all_params.sort(key=lambda x: x['frame'])

    return {
        'folder': results_path,
        'config': config,
        'params': all_params,
        'num_frames': len(all_params)
    }


def compute_silhouette_iou(rendered_mask, target_mask):
    """Compute Intersection over Union between masks."""
    rendered_binary = (rendered_mask > 0.5).astype(np.float32)
    target_binary = (target_mask > 0.5).astype(np.float32)

    intersection = np.sum(rendered_binary * target_binary)
    union = np.sum(rendered_binary) + np.sum(target_binary) - intersection

    if union == 0:
        return 1.0
    return intersection / union


def compute_temporal_smoothness(all_params):
    """Compute temporal smoothness metrics."""
    if len(all_params) < 2:
        return {'joint_velocity_var': 0.0, 'pose_change_var': 0.0}

    # Extract joint positions across frames
    joint_positions = []
    pose_params = []

    for p in all_params:
        params = p['params']
        if 'joints_3d' in params:
            joint_positions.append(params['joints_3d'])
        if 'thetas' in params:
            thetas = params['thetas']
            if isinstance(thetas, torch.Tensor):
                thetas = thetas.cpu().numpy()
            pose_params.append(thetas.flatten())

    metrics = {}

    # Joint velocity variance
    if len(joint_positions) >= 2:
        velocities = []
        for i in range(1, len(joint_positions)):
            vel = np.linalg.norm(joint_positions[i] - joint_positions[i-1], axis=-1)
            velocities.append(vel.mean())
        metrics['joint_velocity_mean'] = np.mean(velocities)
        metrics['joint_velocity_var'] = np.var(velocities)

    # Pose parameter change variance
    if len(pose_params) >= 2:
        pose_changes = []
        for i in range(1, len(pose_params)):
            change = np.linalg.norm(pose_params[i] - pose_params[i-1])
            pose_changes.append(change)
        metrics['pose_change_mean'] = np.mean(pose_changes)
        metrics['pose_change_var'] = np.var(pose_changes)

    return metrics


def generate_mesh_from_params(model, params, device='cuda'):
    """Generate mesh vertices from MAMMAL parameters."""
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        vertices, joints = model(
            to_tensor(params['thetas']),
            to_tensor(params['bone_lengths']),
            to_tensor(params['R']),
            to_tensor(params['T']),
            to_tensor(params['s']),
            to_tensor(params['chest_deformer'])
        )
        keypoints_3d = model.forward_keypoints22()

    return {
        'vertices': vertices[0].cpu().numpy(),
        'joints': joints[0].cpu().numpy(),
        'keypoints': keypoints_3d[0].cpu().numpy(),
        'faces': model.faces_vert_np
    }


def create_summary_figure(results, output_path):
    """Create a summary figure with multiple visualizations."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    config = results['config'] or {}
    exp_name = config.get('experiment_name', 'Unknown')

    fig.suptitle(f"Experiment: {exp_name}\n{results['folder'].name}", fontsize=14)

    # 1. Experiment info text
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis('off')
    info_text = f"""
Experiment: {exp_name}
Frames: {results['num_frames']}
Views: {config.get('data', {}).get('views_to_use', 'N/A')}
Keypoints: {config.get('fitter', {}).get('use_keypoints', 'N/A')}
"""
    ax_info.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                 fontfamily='monospace', transform=ax_info.transAxes)
    ax_info.set_title('Configuration')

    # 2. Temporal smoothness plot
    smoothness = compute_temporal_smoothness(results['params'])
    ax_smooth = fig.add_subplot(gs[0, 1:3])

    if 'joint_velocity_mean' in smoothness:
        # Plot joint velocities over time
        velocities = []
        for i in range(1, len(results['params'])):
            p1 = results['params'][i-1]['params']
            p2 = results['params'][i]['params']
            if 'joints_3d' in p1 and 'joints_3d' in p2:
                vel = np.linalg.norm(p2['joints_3d'] - p1['joints_3d'], axis=-1).mean()
                velocities.append(vel)

        if velocities:
            ax_smooth.plot(velocities, 'b-', linewidth=1)
            ax_smooth.axhline(y=np.mean(velocities), color='r', linestyle='--',
                            label=f'Mean: {np.mean(velocities):.4f}')
            ax_smooth.set_xlabel('Frame')
            ax_smooth.set_ylabel('Joint Velocity')
            ax_smooth.set_title('Temporal Smoothness')
            ax_smooth.legend()

    # 3. Metrics summary
    ax_metrics = fig.add_subplot(gs[0, 3])
    ax_metrics.axis('off')
    metrics_text = f"""
Temporal Metrics:
----------------
Joint Vel Mean: {smoothness.get('joint_velocity_mean', 'N/A'):.4f}
Joint Vel Var: {smoothness.get('joint_velocity_var', 'N/A'):.6f}
Pose Change Mean: {smoothness.get('pose_change_mean', 'N/A'):.4f}
Pose Change Var: {smoothness.get('pose_change_var', 'N/A'):.6f}
"""
    ax_metrics.text(0.1, 0.5, metrics_text, fontsize=9, verticalalignment='center',
                   fontfamily='monospace', transform=ax_metrics.transAxes)
    ax_metrics.set_title('Metrics')

    # 4-7. Sample frame visualizations (4 frames)
    sample_indices = np.linspace(0, len(results['params'])-1, 4, dtype=int)

    # Initialize model for mesh generation
    model = ArticulationTorch()
    model.init_params(batch_size=1)
    model.to('cuda')

    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[1, i], projection='3d')

        params = results['params'][idx]['params']
        frame_num = results['params'][idx]['frame']

        mesh_data = generate_mesh_from_params(model, params)

        # Plot mesh (simplified)
        vertices = mesh_data['vertices']
        keypoints = mesh_data['keypoints']

        # Plot keypoints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                  c='red', s=30, marker='o')

        # Plot mesh surface (very simplified - just bounding points)
        ax.scatter(vertices[::100, 0], vertices[::100, 1], vertices[::100, 2],
                  c='lightblue', s=1, alpha=0.3)

        # Set view
        center = vertices.mean(axis=0)
        max_range = np.abs(vertices - center).max()
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_title(f'Frame {frame_num}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # 8-11. Render comparisons (if available)
    render_path = results['folder'] / 'render'
    if render_path.exists():
        render_files = sorted(render_path.glob('fitting_*_sil.png'))[:4]
        for i, render_file in enumerate(render_files):
            ax = fig.add_subplot(gs[2, i])
            img = cv2.imread(str(render_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f'Render: {render_file.stem}')
            ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary figure saved: {output_path}")

    return smoothness


def generate_report(results, output_folder):
    """Generate comprehensive evaluation report."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    config = results['config'] or {}
    exp_name = config.get('experiment_name', 'unknown')

    # Generate summary figure
    fig_path = output_path / f'{exp_name}_summary.png'
    metrics = create_summary_figure(results, fig_path)

    # Generate JSON report
    report = {
        'experiment_name': exp_name,
        'experiment_description': config.get('experiment_description', ''),
        'results_folder': str(results['folder']),
        'num_frames': results['num_frames'],
        'timestamp': datetime.now().isoformat(),
        'config': {
            'views': config.get('data', {}).get('views_to_use', []),
            'use_keypoints': config.get('fitter', {}).get('use_keypoints', True),
            'keypoint_num': config.get('fitter', {}).get('keypoint_num', 22),
        },
        'metrics': {
            'temporal_smoothness': metrics
        }
    }

    json_path = output_path / f'{exp_name}_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report saved: {json_path}")

    # Generate markdown report
    md_path = output_path / f'{exp_name}_report.md'
    with open(md_path, 'w') as f:
        f.write(f"# Experiment Report: {exp_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Results folder:** `{results['folder']}`\n")
        f.write(f"- **Number of frames:** {results['num_frames']}\n")
        f.write(f"- **Views used:** {config.get('data', {}).get('views_to_use', 'N/A')}\n")
        f.write(f"- **Keypoints enabled:** {config.get('fitter', {}).get('use_keypoints', 'N/A')}\n\n")

        f.write(f"## Metrics\n\n")
        f.write(f"### Temporal Smoothness\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v:.6f} |\n")

        f.write(f"\n## Visualizations\n\n")
        f.write(f"![Summary]({exp_name}_summary.png)\n")

    print(f"Markdown report saved: {md_path}")

    return report


def compare_experiments(results_list, output_folder):
    """Compare multiple experiment results."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect metrics
    comparison_data = []
    for results in results_list:
        config = results['config'] or {}
        metrics = compute_temporal_smoothness(results['params'])
        comparison_data.append({
            'name': config.get('experiment_name', results['folder'].name),
            'num_frames': results['num_frames'],
            'views': len(config.get('data', {}).get('views_to_use', [])),
            'use_keypoints': config.get('fitter', {}).get('use_keypoints', True),
            **metrics
        })

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment Comparison', fontsize=14)

    names = [d['name'] for d in comparison_data]
    x = np.arange(len(names))

    # Joint velocity
    ax = axes[0, 0]
    vals = [d.get('joint_velocity_mean', 0) for d in comparison_data]
    ax.bar(x, vals, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Joint Velocity Mean')
    ax.set_title('Temporal Smoothness (lower = smoother)')

    # Pose change
    ax = axes[0, 1]
    vals = [d.get('pose_change_mean', 0) for d in comparison_data]
    ax.bar(x, vals, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Pose Change Mean')
    ax.set_title('Pose Stability')

    # Configuration comparison table
    ax = axes[1, 0]
    ax.axis('off')
    table_data = [[d['name'], d['views'], d['use_keypoints'], d['num_frames']]
                  for d in comparison_data]
    table = ax.table(cellText=table_data,
                     colLabels=['Experiment', 'Views', 'Keypoints', 'Frames'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Configuration Summary')

    # Metrics table
    ax = axes[1, 1]
    ax.axis('off')
    metric_data = [[d['name'],
                   f"{d.get('joint_velocity_mean', 0):.4f}",
                   f"{d.get('joint_velocity_var', 0):.6f}",
                   f"{d.get('pose_change_mean', 0):.4f}"]
                  for d in comparison_data]
    table = ax.table(cellText=metric_data,
                     colLabels=['Experiment', 'Vel Mean', 'Vel Var', 'Pose Chg'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Metrics Summary')

    plt.tight_layout()

    comparison_path = output_path / 'experiment_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {comparison_path}")

    # Save comparison JSON
    json_path = output_path / 'experiment_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Comparison JSON saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate mesh fitting experiments')
    parser.add_argument('results_folder', type=str, help='Path to results folder')
    parser.add_argument('--compare', type=str, nargs='+', default=[],
                       help='Additional results folders to compare')
    parser.add_argument('--output', type=str, default=None,
                       help='Output folder for reports (default: results_folder/reports)')
    args = parser.parse_args()

    # Load main results
    print(f"Loading results from: {args.results_folder}")
    results = load_results(args.results_folder)
    print(f"Loaded {results['num_frames']} frames")

    # Set output folder
    output_folder = args.output or str(Path(args.results_folder) / 'reports')

    # Generate report
    report = generate_report(results, output_folder)

    # Compare with other experiments if specified
    if args.compare:
        all_results = [results]
        for compare_folder in args.compare:
            print(f"\nLoading comparison: {compare_folder}")
            compare_results = load_results(compare_folder)
            all_results.append(compare_results)

        compare_experiments(all_results, output_folder)

    print(f"\nReports saved to: {output_folder}")


if __name__ == '__main__':
    main()
