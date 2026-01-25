"""
UV Map Experiment Runner

Systematic evaluation of UV mapping parameters.
Generates comparison grids for qualitative and quantitative analysis.
"""

import os
import json
import torch
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from itertools import product
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    name: str = ""

    # Core parameters to tune
    visibility_threshold: float = 0.3
    uv_size: int = 512
    w_tv: float = 1e-3
    w_smooth: float = 1e-4
    fusion_method: str = 'visibility_weighted'  # 'average', 'visibility_weighted', 'max_visibility'
    use_mask: bool = True

    # Optimization
    do_optimization: bool = False
    opt_iters: int = 50
    opt_lr: float = 0.01

    # Frame selection
    frame_interval: int = 1
    max_frames: int = 50  # Limit for faster experiments

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig

    # Quantitative metrics
    coverage: float = 0.0           # % of UV pixels with valid data
    mean_confidence: float = 0.0    # Average confidence across vertices
    color_consistency: float = 0.0  # Cross-view color variance
    photometric_error: float = 0.0  # Render vs GT difference
    seam_discontinuity: float = 0.0 # Color jump at UV seams

    # Timing
    runtime_seconds: float = 0.0

    # Paths
    texture_path: str = ""
    debug_path: str = ""


class UVMapEvaluator:
    """
    Evaluate UV texture map quality.

    Metrics:
    - Coverage: Valid UV region percentage
    - Confidence: Sampling confidence distribution
    - Color Consistency: Multi-view color agreement
    - Photometric Error: Rendered vs GT image difference
    - Seam Discontinuity: Artifact at UV boundaries
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)

    def compute_coverage(
        self,
        texture: torch.Tensor,
        uv_mask: torch.Tensor,
    ) -> float:
        """
        Compute texture coverage percentage.

        Args:
            texture: (3, H, W) texture map
            uv_mask: (1, H, W) valid UV regions

        Returns:
            coverage: Percentage [0, 100]
        """
        # Non-zero pixels in texture
        has_color = (texture.abs().sum(dim=0) > 0.01).float()

        # Valid UV region
        valid_uv = (uv_mask.squeeze() > 0.5).float()

        # Coverage = colored pixels / valid UV pixels
        n_valid = valid_uv.sum()
        if n_valid == 0:
            return 0.0

        n_covered = (has_color * valid_uv).sum()
        coverage = (n_covered / n_valid * 100).item()

        return coverage

    def compute_confidence_stats(
        self,
        confidence: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute confidence distribution statistics.

        Args:
            confidence: (N,) per-vertex confidence

        Returns:
            stats: Dict with mean, std, min, max, percentiles
        """
        conf = confidence[confidence > 0]  # Only valid vertices

        if len(conf) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p25': 0, 'p75': 0}

        return {
            'mean': conf.mean().item(),
            'std': conf.std().item(),
            'min': conf.min().item(),
            'max': conf.max().item(),
            'p25': torch.quantile(conf, 0.25).item(),
            'p75': torch.quantile(conf, 0.75).item(),
        }

    def compute_color_consistency(
        self,
        per_view_colors: List[torch.Tensor],
        per_view_weights: List[torch.Tensor],
    ) -> float:
        """
        Compute cross-view color consistency.

        Lower variance = more consistent = better.

        Args:
            per_view_colors: List of (N, 3) colors per view
            per_view_weights: List of (N,) weights per view

        Returns:
            consistency: Mean color std across views (lower is better)
        """
        n_views = len(per_view_colors)
        if n_views < 2:
            return 0.0

        # Stack colors: (V, N, 3)
        colors = torch.stack(per_view_colors, dim=0)
        weights = torch.stack(per_view_weights, dim=0)

        # Find vertices visible in multiple views
        visible_count = (weights > 0.1).sum(dim=0)
        multi_visible = visible_count >= 2

        if multi_visible.sum() == 0:
            return 0.0

        # Compute variance for multi-visible vertices
        colors_subset = colors[:, multi_visible, :]  # (V, M, 3)
        weights_subset = weights[:, multi_visible]   # (V, M)

        # Weighted mean
        weighted_sum = (colors_subset * weights_subset.unsqueeze(-1)).sum(dim=0)
        weight_sum = weights_subset.sum(dim=0, keepdim=True).clamp(min=1e-8)
        mean_color = weighted_sum / weight_sum.T  # (M, 3)

        # Variance from mean
        diff = colors_subset - mean_color.unsqueeze(0)  # (V, M, 3)
        variance = (diff.pow(2) * weights_subset.unsqueeze(-1)).sum(dim=0) / weight_sum.T

        # Mean std
        consistency = variance.sqrt().mean().item()

        return consistency

    def compute_photometric_error(
        self,
        rendered_images: List[torch.Tensor],
        gt_images: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute photometric error between rendered and GT images.

        Args:
            rendered_images: List of (3, H, W) rendered images
            gt_images: List of (3, H, W) ground truth images
            masks: Optional list of (H, W) foreground masks

        Returns:
            errors: Dict with MAE, MSE, PSNR
        """
        total_mae = 0.0
        total_mse = 0.0
        n_pixels = 0

        for i, (rendered, gt) in enumerate(zip(rendered_images, gt_images)):
            if masks is not None and masks[i] is not None:
                mask = masks[i].unsqueeze(0)
            else:
                mask = torch.ones(1, rendered.shape[1], rendered.shape[2],
                                 device=rendered.device)

            diff = (rendered - gt) * mask
            n = mask.sum()

            total_mae += diff.abs().sum().item()
            total_mse += diff.pow(2).sum().item()
            n_pixels += n.item()

        if n_pixels == 0:
            return {'mae': 0, 'mse': 0, 'psnr': 0}

        mae = total_mae / (n_pixels * 3)
        mse = total_mse / (n_pixels * 3)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        return {'mae': mae, 'mse': mse, 'psnr': psnr}

    def compute_seam_discontinuity(
        self,
        texture: torch.Tensor,
        seam_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute color discontinuity at UV seams.

        Seams are boundaries where UV wraps.
        High discontinuity = visible seam artifacts.

        Args:
            texture: (3, H, W) texture map
            seam_mask: (H, W) mask of seam pixels (optional)

        Returns:
            discontinuity: Mean color jump at seams
        """
        # Compute gradient magnitude
        grad_h = (texture[:, 1:, :] - texture[:, :-1, :]).abs()
        grad_w = (texture[:, :, 1:] - texture[:, :, :-1]).abs()

        # If no seam mask, use high-gradient regions as proxy
        if seam_mask is None:
            # Top 5% gradient pixels
            grad_mag = grad_h[:, :, :-1].pow(2) + grad_w[:, :-1, :].pow(2)
            grad_mag = grad_mag.sum(dim=0).sqrt()
            threshold = torch.quantile(grad_mag.flatten(), 0.95)
            seam_proxy = (grad_mag > threshold).float()

            masked_grad = grad_mag[seam_proxy > 0.5]
            if masked_grad.numel() == 0:
                discontinuity = 0.0
            else:
                discontinuity = masked_grad.mean().item()
        else:
            # Use provided seam mask
            seam = seam_mask[:-1, :-1]
            grad_at_seam = grad_h[:, :, :-1] + grad_w[:, :-1, :]
            grad_at_seam = grad_at_seam.sum(dim=0)

            if seam.sum() > 0:
                discontinuity = grad_at_seam[seam > 0.5].mean().item()
            else:
                discontinuity = 0.0

        return discontinuity


class ExperimentRunner:
    """
    Run systematic UV mapping experiments.

    Features:
    - Grid search over parameter combinations
    - Parallel evaluation
    - Result aggregation and comparison
    - Visual debug outputs
    """

    def __init__(
        self,
        result_dir: str,
        output_dir: str,
        device: str = 'cuda',
    ):
        """
        Args:
            result_dir: Fitting result directory
            output_dir: Experiment output directory
            device: Torch device
        """
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.device = device

        self.evaluator = UVMapEvaluator(device)
        self.results: List[ExperimentResult] = []

        os.makedirs(output_dir, exist_ok=True)

    def generate_experiment_grid(
        self,
        param_grid: Dict[str, List[Any]],
    ) -> List[ExperimentConfig]:
        """
        Generate experiment configurations from parameter grid.

        Args:
            param_grid: Dict mapping param names to value lists

        Returns:
            configs: List of ExperimentConfig
        """
        # Get all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        configs = []
        for i, combo in enumerate(combinations):
            config = ExperimentConfig()
            for key, val in zip(keys, combo):
                setattr(config, key, val)

            # Generate filesystem-safe name
            config.name = self._generate_safe_name(i, keys, combo)

            configs.append(config)

        return configs

    def _generate_safe_name(
        self,
        exp_idx: int,
        keys: List[str],
        values: List[Any],
    ) -> str:
        """
        Generate filesystem-safe experiment name.

        Replaces problematic characters:
        - '=' -> '-'
        - '.' -> 'p' (for decimals)
        - spaces -> '_'

        Args:
            exp_idx: Experiment index
            keys: Parameter names
            values: Parameter values

        Returns:
            safe_name: Filesystem-safe experiment name
        """
        parts = []
        for k, v in zip(keys, values):
            # Short key (3-4 chars)
            short_key = k[:4].replace('_', '')

            # Format value
            if isinstance(v, float):
                # Convert float: 0.3 -> 0p3, 1e-3 -> 1em3
                v_str = str(v)
                if 'e-' in v_str:
                    v_str = v_str.replace('e-', 'em')
                elif 'e' in v_str:
                    v_str = v_str.replace('e', 'ep')
                v_str = v_str.replace('.', 'p')
            elif isinstance(v, bool):
                v_str = 'T' if v else 'F'
            else:
                v_str = str(v)

            # Remove/replace special chars
            v_str = v_str.replace(' ', '').replace('=', '').replace('/', '')

            parts.append(f"{short_key}{v_str}")

        return f"exp{exp_idx:03d}_{'_'.join(parts)}"

    def run_single_experiment(
        self,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        """
        Run a single experiment with given config.

        Args:
            config: Experiment configuration

        Returns:
            result: Experiment results
        """
        from .uv_pipeline import UVMapPipeline, UVPipelineConfig
        from .texture_optimizer import TextureOptConfig
        import time

        start_time = time.time()

        # Create pipeline config
        pipeline_config = UVPipelineConfig(
            result_dir=self.result_dir,
            uv_size=config.uv_size,
            use_visibility_weighting=(config.fusion_method != 'average'),
            visibility_threshold=config.visibility_threshold,
            do_optimization=config.do_optimization,
            opt_iters=config.opt_iters,
            opt_lr=config.opt_lr,
            opt_w_tv=config.w_tv,
            frame_interval=config.frame_interval,
            output_dir=os.path.join(self.output_dir, config.name),
            save_intermediate=False,
        )

        # Run pipeline
        pipeline = UVMapPipeline(pipeline_config, device=self.device)
        pipeline.setup()

        # Limit frames for faster experiments
        if config.max_frames > 0:
            pipeline.frames = pipeline.frames[:config.max_frames]

        texture = pipeline.run()

        runtime = time.time() - start_time

        # Evaluate
        vertex_colors, confidence = pipeline.texture_accumulator.get_texture()
        uv_mask = pipeline.uv_renderer.get_uv_mask()

        coverage = self.evaluator.compute_coverage(texture, uv_mask)
        conf_stats = self.evaluator.compute_confidence_stats(confidence)
        seam_disc = self.evaluator.compute_seam_discontinuity(texture)

        # Create result
        result = ExperimentResult(
            config=config,
            coverage=coverage,
            mean_confidence=conf_stats['mean'],
            seam_discontinuity=seam_disc,
            runtime_seconds=runtime,
            texture_path=os.path.join(pipeline_config.output_dir, 'texture_final.png'),
            debug_path=pipeline_config.output_dir,
        )

        # Save debug visualizations
        self._save_debug_viz(pipeline, config, result)

        return result

    def run_all_experiments(
        self,
        configs: List[ExperimentConfig],
        parallel: bool = False,
    ) -> List[ExperimentResult]:
        """
        Run all experiments in the grid.

        Args:
            configs: List of experiment configs
            parallel: Whether to run in parallel (TODO)

        Returns:
            results: List of experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running {len(configs)} UV Map Experiments")
        print(f"{'='*60}")

        self.results = []

        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config.name}")
            try:
                result = self.run_single_experiment(config)
                self.results.append(result)
                print(f"  Coverage: {result.coverage:.1f}%")
                print(f"  Confidence: {result.mean_confidence:.3f}")
                print(f"  Seam: {result.seam_discontinuity:.4f}")
                print(f"  Runtime: {result.runtime_seconds:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        # Save summary
        self._save_summary()

        return self.results

    def _save_debug_viz(
        self,
        pipeline,
        config: ExperimentConfig,
        result: ExperimentResult,
    ) -> None:
        """Save debug visualizations for experiment."""
        debug_dir = result.debug_path
        os.makedirs(debug_dir, exist_ok=True)

        # 1. Confidence heatmap
        vertex_colors, confidence = pipeline.texture_accumulator.get_texture()
        conf_map = self._render_confidence_map(pipeline, confidence)
        cv2.imwrite(os.path.join(debug_dir, 'confidence_heatmap.png'), conf_map)

        # 2. Per-view contribution map (if available)
        # TODO: Track per-view contributions

        # 3. Config info
        with open(os.path.join(debug_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    def _render_confidence_map(
        self,
        pipeline,
        confidence: torch.Tensor,
    ) -> np.ndarray:
        """Render confidence as heatmap in UV space."""
        # Map confidence to UV
        conf_uv = pipeline.uv_renderer._map_vertex_attr_to_uv(confidence.unsqueeze(1))
        conf_map = pipeline.uv_renderer._render_vertex_colors(conf_uv)

        # Normalize and colormap
        conf_np = conf_map.squeeze().cpu().numpy()
        conf_np = (conf_np - conf_np.min()) / (conf_np.max() - conf_np.min() + 1e-8)
        conf_colored = cv2.applyColorMap((conf_np * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return conf_colored

    def _save_summary(self) -> None:
        """Save experiment summary and comparison."""
        if not self.results:
            return

        summary_path = os.path.join(self.output_dir, 'summary.json')

        # Convert to serializable format
        summary_data = []
        for r in self.results:
            entry = {
                'name': r.config.name,
                'config': r.config.to_dict(),
                'metrics': {
                    'coverage': r.coverage,
                    'mean_confidence': r.mean_confidence,
                    'seam_discontinuity': r.seam_discontinuity,
                    'runtime_seconds': r.runtime_seconds,
                },
                'texture_path': r.texture_path,
            }
            summary_data.append(entry)

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nSummary saved: {summary_path}")

        # Generate comparison grid
        self._generate_comparison_grid()

    def _generate_comparison_grid(self) -> None:
        """Generate visual comparison grid of all experiments."""
        if len(self.results) == 0:
            return

        # Load all textures
        textures = []
        labels = []

        for r in self.results:
            if os.path.exists(r.texture_path):
                img = cv2.imread(r.texture_path)
                if img is not None:
                    textures.append(img)
                    # Short label
                    label = f"{r.config.name[:20]}\nCov:{r.coverage:.0f}% Conf:{r.mean_confidence:.2f}"
                    labels.append(label)

        if not textures:
            return

        # Create grid
        n = len(textures)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        # Resize to common size
        thumb_size = 256
        thumbs = [cv2.resize(t, (thumb_size, thumb_size)) for t in textures]

        # Build grid image
        grid_h = rows * (thumb_size + 40)  # Extra space for labels
        grid_w = cols * thumb_size
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

        for i, (thumb, label) in enumerate(zip(thumbs, labels)):
            row = i // cols
            col = i % cols
            y = row * (thumb_size + 40)
            x = col * thumb_size

            grid[y:y+thumb_size, x:x+thumb_size] = thumb

            # Add label
            for j, line in enumerate(label.split('\n')):
                cv2.putText(grid, line, (x + 5, y + thumb_size + 15 + j*15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Save
        grid_path = os.path.join(self.output_dir, 'comparison_grid.png')
        cv2.imwrite(grid_path, grid)
        print(f"Comparison grid saved: {grid_path}")


def run_ablation_study(
    result_dir: str,
    output_dir: str,
    quick: bool = True,
) -> List[ExperimentResult]:
    """
    Run ablation study on key UV mapping parameters.

    Args:
        result_dir: Fitting result directory
        output_dir: Output directory for experiments
        quick: Use reduced parameter grid for faster testing

    Returns:
        results: List of experiment results
    """
    # Define parameter grid
    if quick:
        # Quick test: fewer combinations
        param_grid = {
            'visibility_threshold': [0.2, 0.5],
            'uv_size': [256, 512],
            'fusion_method': ['average', 'visibility_weighted'],
            'do_optimization': [False],
            'max_frames': [20],
        }
    else:
        # Full ablation
        param_grid = {
            'visibility_threshold': [0.1, 0.3, 0.5, 0.7],
            'uv_size': [256, 512, 1024],
            'w_tv': [0, 1e-4, 1e-3, 1e-2],
            'fusion_method': ['average', 'visibility_weighted', 'max_visibility'],
            'use_mask': [True, False],
            'do_optimization': [False, True],
            'max_frames': [50],
        }

    # Create runner
    runner = ExperimentRunner(
        result_dir=result_dir,
        output_dir=output_dir,
    )

    # Generate configs
    configs = runner.generate_experiment_grid(param_grid)
    print(f"Generated {len(configs)} experiment configurations")

    # Run experiments
    results = runner.run_all_experiments(configs)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UV Map Ablation Study')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Fitting result directory')
    parser.add_argument('--output_dir', type=str, default='results/uvmap',
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer combinations')

    args = parser.parse_args()

    results = run_ablation_study(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        quick=args.quick,
    )

    print(f"\nCompleted {len(results)} experiments")
