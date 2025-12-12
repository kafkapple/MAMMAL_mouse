#!/usr/bin/env python
"""
Ablation Study: Mesh Comparison Script

Compares mesh fitting results against a baseline (6-view, 22-keypoint) to quantify
the quality degradation when reducing views or keypoints.

Metrics:
- V2V (Vertex-to-Vertex): Mean/Max distance between corresponding vertices
- Chamfer Distance: Average of bidirectional nearest neighbor distances
- Surface Distance: Point-to-surface distance

Usage:
    python scripts/compare_mesh_ablation.py \
        --baseline results/fitting/markerless_mouse_1_nerf_v012345_kp22_* \
        --experiments results/fitting/markerless_mouse_1_nerf_v* \
        --output docs/reports/ablation_quantitative_results.md
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import yaml
import re


@dataclass
class MeshMetrics:
    """Container for mesh comparison metrics."""
    v2v_mean: float = 0.0       # Vertex-to-vertex mean distance
    v2v_max: float = 0.0        # Vertex-to-vertex max distance
    v2v_std: float = 0.0        # Vertex-to-vertex std
    v2v_median: float = 0.0     # Vertex-to-vertex median
    chamfer: float = 0.0        # Chamfer distance
    hausdorff: float = 0.0      # Hausdorff distance (max of min distances)


@dataclass
class ExperimentResult:
    """Container for experiment comparison results."""
    name: str
    dir_path: str
    num_views: int
    num_keypoints: int
    view_ids: List[int] = field(default_factory=list)
    keypoint_type: str = ""
    num_frames: int = 0
    metrics_per_frame: List[MeshMetrics] = field(default_factory=list)
    mean_metrics: Optional[MeshMetrics] = None

    def compute_summary(self):
        """Compute mean metrics across all frames."""
        if not self.metrics_per_frame:
            return

        self.mean_metrics = MeshMetrics(
            v2v_mean=np.mean([m.v2v_mean for m in self.metrics_per_frame]),
            v2v_max=np.mean([m.v2v_max for m in self.metrics_per_frame]),
            v2v_std=np.mean([m.v2v_std for m in self.metrics_per_frame]),
            v2v_median=np.mean([m.v2v_median for m in self.metrics_per_frame]),
            chamfer=np.mean([m.chamfer for m in self.metrics_per_frame]),
            hausdorff=np.mean([m.hausdorff for m in self.metrics_per_frame]),
        )


def load_mesh(obj_path: str) -> Optional[trimesh.Trimesh]:
    """Load mesh from OBJ file."""
    try:
        mesh = trimesh.load(obj_path, process=False)
        return mesh
    except Exception as e:
        print(f"Error loading {obj_path}: {e}")
        return None


def compute_v2v_distance(vertices1: np.ndarray, vertices2: np.ndarray) -> MeshMetrics:
    """
    Compute vertex-to-vertex distance metrics.
    Assumes same topology (same number of vertices in corresponding positions).
    """
    # Direct vertex correspondence distance
    distances = np.linalg.norm(vertices1 - vertices2, axis=1)

    metrics = MeshMetrics(
        v2v_mean=float(np.mean(distances)),
        v2v_max=float(np.max(distances)),
        v2v_std=float(np.std(distances)),
        v2v_median=float(np.median(distances)),
    )

    return metrics


def compute_chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> Tuple[float, float]:
    """
    Compute Chamfer distance and Hausdorff distance.

    Chamfer: Average of bidirectional nearest neighbor distances
    Hausdorff: Maximum of minimum distances
    """
    # Build KD-trees
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # Find nearest neighbors
    dist1, _ = tree2.query(points1)  # For each point in 1, find nearest in 2
    dist2, _ = tree1.query(points2)  # For each point in 2, find nearest in 1

    # Chamfer distance (average of averages)
    chamfer = (np.mean(dist1) + np.mean(dist2)) / 2

    # Hausdorff distance (max of maxes)
    hausdorff = max(np.max(dist1), np.max(dist2))

    return float(chamfer), float(hausdorff)


def compare_meshes(baseline_mesh: trimesh.Trimesh,
                   experiment_mesh: trimesh.Trimesh) -> MeshMetrics:
    """Compare two meshes and return all metrics."""
    v1 = np.array(baseline_mesh.vertices)
    v2 = np.array(experiment_mesh.vertices)

    # V2V metrics (same topology assumed)
    metrics = compute_v2v_distance(v1, v2)

    # Chamfer and Hausdorff
    chamfer, hausdorff = compute_chamfer_distance(v1, v2)
    metrics.chamfer = chamfer
    metrics.hausdorff = hausdorff

    return metrics


def parse_experiment_info(exp_dir: str) -> Dict:
    """Parse experiment information from directory name and config."""
    info = {
        'name': os.path.basename(exp_dir),
        'num_views': 6,
        'num_keypoints': 22,
        'view_ids': [0, 1, 2, 3, 4, 5],
        'keypoint_type': 'full_22',
    }

    name = info['name']

    # Parse views from directory name (e.g., v012345, v0123, v024)
    view_match = re.search(r'_v(\d+)_', name)
    if view_match:
        view_str = view_match.group(1)
        info['view_ids'] = [int(c) for c in view_str]
        info['num_views'] = len(info['view_ids'])

    # Parse keypoints from directory name
    if 'kp22' in name:
        info['num_keypoints'] = 22
        info['keypoint_type'] = 'full_22'
    elif 'sparse9' in name:
        info['num_keypoints'] = 9
        info['keypoint_type'] = 'sparse_9_dlc'
    elif 'sparse7' in name:
        info['num_keypoints'] = 7
        info['keypoint_type'] = 'sparse_7_mars'
    elif 'sparse5' in name:
        info['num_keypoints'] = 5
        info['keypoint_type'] = 'sparse_5_minimal'
    elif 'sparse3' in name:
        info['num_keypoints'] = 3
        info['keypoint_type'] = 'sparse_3_core'

    # Load config for more accurate info
    config_path = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config:
                if 'data' in config and 'views_to_use' in config['data']:
                    info['view_ids'] = config['data']['views_to_use']
                    info['num_views'] = len(info['view_ids'])
                if 'fitter' in config:
                    if 'sparse_keypoint_indices' in config['fitter']:
                        indices = config['fitter']['sparse_keypoint_indices']
                        info['num_keypoints'] = len(indices) if indices else 22
                    if 'num_keypoints' in config['fitter']:
                        info['num_keypoints'] = config['fitter']['num_keypoints']
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    return info


def run_comparison(baseline_dir: str, experiment_dir: str,
                   frame_indices: Optional[List[int]] = None,
                   verbose: bool = True) -> ExperimentResult:
    """
    Compare experiment meshes against baseline.

    Args:
        baseline_dir: Path to baseline experiment directory
        experiment_dir: Path to experiment directory to compare
        frame_indices: Specific frames to compare (None = all)
        verbose: Print progress
    """
    info = parse_experiment_info(experiment_dir)

    result = ExperimentResult(
        name=info['name'],
        dir_path=experiment_dir,
        num_views=info['num_views'],
        num_keypoints=info['num_keypoints'],
        view_ids=info['view_ids'],
        keypoint_type=info['keypoint_type'],
    )

    # Find obj files
    baseline_obj_dir = os.path.join(baseline_dir, 'obj')
    exp_obj_dir = os.path.join(experiment_dir, 'obj')

    if not os.path.exists(baseline_obj_dir):
        print(f"Error: Baseline obj directory not found: {baseline_obj_dir}")
        return result

    if not os.path.exists(exp_obj_dir):
        print(f"Error: Experiment obj directory not found: {exp_obj_dir}")
        return result

    # Get list of mesh files
    baseline_files = sorted(glob.glob(os.path.join(baseline_obj_dir, 'step_2_frame_*.obj')))
    exp_files = sorted(glob.glob(os.path.join(exp_obj_dir, 'step_2_frame_*.obj')))

    if not baseline_files:
        print(f"Error: No baseline mesh files found in {baseline_obj_dir}")
        return result

    if not exp_files:
        print(f"Error: No experiment mesh files found in {exp_obj_dir}")
        return result

    # Create filename mapping
    baseline_map = {os.path.basename(f): f for f in baseline_files}
    exp_map = {os.path.basename(f): f for f in exp_files}

    # Find common files
    common_files = sorted(set(baseline_map.keys()) & set(exp_map.keys()))

    if frame_indices is not None:
        common_files = [f for f in common_files
                       if int(re.search(r'frame_(\d+)', f).group(1)) in frame_indices]

    result.num_frames = len(common_files)

    if verbose:
        print(f"Comparing {result.name}: {result.num_frames} frames")

    # Compare each frame
    for i, filename in enumerate(common_files):
        baseline_mesh = load_mesh(baseline_map[filename])
        exp_mesh = load_mesh(exp_map[filename])

        if baseline_mesh is None or exp_mesh is None:
            continue

        metrics = compare_meshes(baseline_mesh, exp_mesh)
        result.metrics_per_frame.append(metrics)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{result.num_frames} frames")

    result.compute_summary()

    if verbose and result.mean_metrics:
        print(f"  Mean V2V: {result.mean_metrics.v2v_mean:.4f} mm")
        print(f"  Mean Chamfer: {result.mean_metrics.chamfer:.4f} mm")

    return result


def generate_report(results: List[ExperimentResult],
                    baseline_name: str,
                    output_path: str):
    """Generate markdown report with quantitative comparison."""

    # Sort by mean V2V distance
    sorted_results = sorted(results, key=lambda r: r.mean_metrics.v2v_mean if r.mean_metrics else float('inf'))

    report_lines = []
    report_lines.append("---")
    report_lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
    report_lines.append("tags: [ablation-study, quantitative-analysis, mesh-comparison]")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("# Ablation Study: Quantitative Mesh Comparison Results")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Baseline:** {baseline_name}")
    report_lines.append("")

    # Summary table
    report_lines.append("## Summary (Sorted by V2V Distance)")
    report_lines.append("")
    report_lines.append("| Rank | Views | Keypoints | V2V Mean (mm) | V2V Max (mm) | Chamfer (mm) | Hausdorff (mm) |")
    report_lines.append("|------|-------|-----------|---------------|--------------|--------------|----------------|")

    for i, r in enumerate(sorted_results, 1):
        if r.mean_metrics is None:
            continue
        m = r.mean_metrics
        report_lines.append(
            f"| {i} | {r.num_views} ({','.join(map(str, r.view_ids))}) | {r.num_keypoints} ({r.keypoint_type}) | "
            f"{m.v2v_mean:.4f} | {m.v2v_max:.4f} | {m.chamfer:.4f} | {m.hausdorff:.4f} |"
        )

    report_lines.append("")

    # Trade-off analysis
    report_lines.append("## Trade-off Analysis")
    report_lines.append("")
    report_lines.append("### Best Configurations by Category")
    report_lines.append("")

    # Find best for each category
    view_ablation = [r for r in sorted_results if r.num_keypoints == 3]
    kp_ablation = [r for r in sorted_results if r.num_views == 6]

    if view_ablation:
        best_view = min(view_ablation, key=lambda r: r.mean_metrics.v2v_mean if r.mean_metrics else float('inf'))
        report_lines.append(f"**Best View Configuration (with 3 keypoints):** {best_view.num_views} views "
                          f"(V2V: {best_view.mean_metrics.v2v_mean:.4f} mm)")

    if kp_ablation:
        best_kp = min(kp_ablation, key=lambda r: r.mean_metrics.v2v_mean if r.mean_metrics else float('inf'))
        report_lines.append(f"**Best Keypoint Configuration (with 6 views):** {best_kp.num_keypoints} keypoints "
                          f"(V2V: {best_kp.mean_metrics.v2v_mean:.4f} mm)")

    report_lines.append("")

    # Efficiency score (quality per resource)
    report_lines.append("### Efficiency Score (Lower V2V / Lower Resources = Better)")
    report_lines.append("")
    report_lines.append("| Configuration | Resources (V×KP) | V2V Mean | Efficiency Score |")
    report_lines.append("|---------------|------------------|----------|------------------|")

    for r in sorted_results:
        if r.mean_metrics is None:
            continue
        resources = r.num_views * r.num_keypoints
        # Inverse efficiency: lower is better (want low error with low resources)
        # Score = V2V_error * log(resources + 1) - lower is better
        efficiency = r.mean_metrics.v2v_mean * np.log(resources + 1)
        report_lines.append(
            f"| {r.num_views}V {r.num_keypoints}KP | {resources} | {r.mean_metrics.v2v_mean:.4f} | {efficiency:.4f} |"
        )

    report_lines.append("")

    # Detailed results per experiment
    report_lines.append("## Detailed Results per Configuration")
    report_lines.append("")

    for r in sorted_results:
        if r.mean_metrics is None:
            continue
        m = r.mean_metrics
        report_lines.append(f"### {r.num_views} Views, {r.num_keypoints} Keypoints ({r.keypoint_type})")
        report_lines.append("")
        report_lines.append(f"- **Directory:** `{r.name}`")
        report_lines.append(f"- **View IDs:** {r.view_ids}")
        report_lines.append(f"- **Frames Compared:** {r.num_frames}")
        report_lines.append("")
        report_lines.append("| Metric | Value |")
        report_lines.append("|--------|-------|")
        report_lines.append(f"| V2V Mean | {m.v2v_mean:.4f} mm |")
        report_lines.append(f"| V2V Max | {m.v2v_max:.4f} mm |")
        report_lines.append(f"| V2V Std | {m.v2v_std:.4f} mm |")
        report_lines.append(f"| V2V Median | {m.v2v_median:.4f} mm |")
        report_lines.append(f"| Chamfer Distance | {m.chamfer:.4f} mm |")
        report_lines.append(f"| Hausdorff Distance | {m.hausdorff:.4f} mm |")
        report_lines.append("")

    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Report saved to: {output_path}")

    return sorted_results


def generate_visualizations(results: List[ExperimentResult],
                           output_dir: str):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Sort results
    sorted_results = sorted(results, key=lambda r: r.mean_metrics.v2v_mean if r.mean_metrics else float('inf'))

    # Filter valid results
    valid_results = [r for r in sorted_results if r.mean_metrics is not None]

    if not valid_results:
        return

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Ablation Study: Mesh Comparison Results', fontsize=14, fontweight='bold')

    # 1. Bar chart - V2V Mean by configuration
    ax = axes[0, 0]
    labels = [f"{r.num_views}V/{r.num_keypoints}KP" for r in valid_results]
    v2v_means = [r.mean_metrics.v2v_mean for r in valid_results]
    colors = ['green' if v < 0.5 else 'orange' if v < 1.0 else 'red' for v in v2v_means]
    bars = ax.bar(range(len(labels)), v2v_means, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('V2V Mean Distance (mm)')
    ax.set_title('Vertex-to-Vertex Mean Distance')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.5mm')
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, v2v_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. View ablation subplot
    ax = axes[0, 1]
    view_ablation = [r for r in valid_results if r.num_keypoints == 3]
    if view_ablation:
        views = [r.num_views for r in view_ablation]
        v2v = [r.mean_metrics.v2v_mean for r in view_ablation]
        ax.plot(views, v2v, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Views')
        ax.set_ylabel('V2V Mean Distance (mm)')
        ax.set_title('View Ablation (3 Keypoints)')
        ax.grid(True, alpha=0.3)
        for v, d in zip(views, v2v):
            ax.annotate(f'{d:.3f}', (v, d), textcoords="offset points",
                       xytext=(0,10), ha='center')

    # 3. Keypoint ablation subplot
    ax = axes[1, 0]
    kp_ablation = [r for r in valid_results if r.num_views == 6]
    if kp_ablation:
        kps = [r.num_keypoints for r in kp_ablation]
        v2v = [r.mean_metrics.v2v_mean for r in kp_ablation]
        ax.plot(kps, v2v, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Keypoints')
        ax.set_ylabel('V2V Mean Distance (mm)')
        ax.set_title('Keypoint Ablation (6 Views)')
        ax.grid(True, alpha=0.3)
        for k, d in zip(kps, v2v):
            ax.annotate(f'{d:.3f}', (k, d), textcoords="offset points",
                       xytext=(0,10), ha='center')

    # 4. 2D view: Views vs Keypoints heatmap
    ax = axes[1, 1]
    # Create data for heatmap
    all_views = sorted(set(r.num_views for r in valid_results))
    all_kps = sorted(set(r.num_keypoints for r in valid_results))

    heatmap_data = np.full((len(all_kps), len(all_views)), np.nan)
    for r in valid_results:
        v_idx = all_views.index(r.num_views)
        k_idx = all_kps.index(r.num_keypoints)
        heatmap_data[k_idx, v_idx] = r.mean_metrics.v2v_mean

    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(all_views)))
    ax.set_xticklabels(all_views)
    ax.set_yticks(range(len(all_kps)))
    ax.set_yticklabels(all_kps)
    ax.set_xlabel('Number of Views')
    ax.set_ylabel('Number of Keypoints')
    ax.set_title('V2V Distance Heatmap (mm)')

    # Add text annotations
    for i in range(len(all_kps)):
        for j in range(len(all_views)):
            if not np.isnan(heatmap_data[i, j]):
                ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                       ha='center', va='center', fontsize=9,
                       color='white' if heatmap_data[i, j] > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='V2V Mean (mm)')

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {fig_path}")

    return fig_path


def save_json_results(results: List[ExperimentResult], output_path: str):
    """Save results as JSON for further analysis."""
    data = []
    for r in results:
        if r.mean_metrics is None:
            continue
        data.append({
            'name': r.name,
            'num_views': r.num_views,
            'num_keypoints': r.num_keypoints,
            'view_ids': r.view_ids,
            'keypoint_type': r.keypoint_type,
            'num_frames': r.num_frames,
            'metrics': {
                'v2v_mean': r.mean_metrics.v2v_mean,
                'v2v_max': r.mean_metrics.v2v_max,
                'v2v_std': r.mean_metrics.v2v_std,
                'v2v_median': r.mean_metrics.v2v_median,
                'chamfer': r.mean_metrics.chamfer,
                'hausdorff': r.mean_metrics.hausdorff,
            }
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare mesh ablation study results')
    parser.add_argument('--baseline', required=True,
                       help='Path to baseline experiment directory (6-view, 22-keypoint)')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Paths to experiment directories (supports glob patterns)')
    parser.add_argument('--output', default='docs/reports/ablation_quantitative_results.md',
                       help='Output markdown report path')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                       help='Specific frame indices to compare (default: all)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of frames to sample (default: all)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    args = parser.parse_args()

    # Resolve baseline path
    baseline_matches = glob.glob(args.baseline)
    if not baseline_matches:
        print(f"Error: No baseline found matching {args.baseline}")
        sys.exit(1)
    baseline_dir = baseline_matches[0]
    print(f"Baseline: {baseline_dir}")

    # Resolve experiment paths
    all_exp_dirs = []
    for pattern in args.experiments:
        matches = glob.glob(pattern)
        all_exp_dirs.extend(matches)
    all_exp_dirs = sorted(set(all_exp_dirs))

    # Exclude baseline from experiments
    all_exp_dirs = [d for d in all_exp_dirs if d != baseline_dir and os.path.isdir(d)]

    if not all_exp_dirs:
        print("Error: No experiment directories found")
        sys.exit(1)

    print(f"Found {len(all_exp_dirs)} experiments to compare:")
    for d in all_exp_dirs:
        print(f"  - {os.path.basename(d)}")
    print()

    # Determine frames to compare
    frame_indices = args.frames
    if args.sample and frame_indices is None:
        # Sample frames uniformly
        obj_files = sorted(glob.glob(os.path.join(baseline_dir, 'obj', 'step_2_frame_*.obj')))
        total_frames = len(obj_files)
        if total_frames > args.sample:
            indices = np.linspace(0, total_frames - 1, args.sample, dtype=int)
            frame_indices = list(indices)
            print(f"Sampling {len(frame_indices)} frames from {total_frames} total")

    # Run comparisons
    results = []
    for exp_dir in all_exp_dirs:
        result = run_comparison(
            baseline_dir, exp_dir,
            frame_indices=frame_indices,
            verbose=not args.quiet
        )
        results.append(result)

    # Generate outputs
    output_path = args.output

    # Generate report
    sorted_results = generate_report(results, os.path.basename(baseline_dir), output_path)

    # Generate visualizations
    vis_dir = os.path.dirname(output_path)
    fig_path = generate_visualizations(results, vis_dir)

    # Save JSON
    json_path = output_path.replace('.md', '.json')
    save_json_results(results, json_path)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Compared {len(results)} experiments against baseline")
    print(f"Report: {output_path}")
    print(f"Visualization: {fig_path}")
    print(f"JSON: {json_path}")

    # Print top 3
    print("\nTop 3 configurations (closest to baseline):")
    for i, r in enumerate(sorted_results[:3], 1):
        if r.mean_metrics:
            print(f"  {i}. {r.num_views} views, {r.num_keypoints} keypoints: "
                  f"V2V={r.mean_metrics.v2v_mean:.4f}mm")


if __name__ == '__main__':
    main()
