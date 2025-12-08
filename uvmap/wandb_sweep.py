"""
WandB Sweep-based Hyperparameter Optimization for UV Mapping

Provides simple and robust hyperparameter search using Weights & Biases Sweeps.
Default method for UV mapping parameter optimization.

Features:
- Bayesian optimization via WandB Sweeps
- Real-time metric visualization dashboard
- Parallel agent support
- Automatic best config extraction
"""

import os
import json
import time
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Check wandb availability (lazy import)
WANDB_AVAILABLE = False
wandb = None

def _check_wandb():
    """Check and import wandb if available."""
    global WANDB_AVAILABLE, wandb
    if wandb is not None:
        return WANDB_AVAILABLE
    try:
        import wandb as _wandb
        wandb = _wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
    return WANDB_AVAILABLE


@dataclass
class WandBSweepConfig:
    """Configuration for WandB Sweep optimization."""
    # WandB settings
    project: str = "uvmap-optimization"
    entity: Optional[str] = None  # WandB username/team
    sweep_name: str = "uvmap-sweep"

    # Sweep settings
    method: str = "bayes"  # 'bayes', 'random', 'grid'
    metric_name: str = "score"  # Metric to optimize
    metric_goal: str = "maximize"  # 'maximize' or 'minimize'

    # Run settings
    count: int = 30  # Number of runs

    # Objective weights (for composite score)
    w_coverage: float = 0.4
    w_psnr: float = 0.3
    w_seam: float = 0.3  # Lower seam discontinuity is better

    # Output
    output_dir: str = "wandb_sweep_results"


# Default parameter search space
DEFAULT_SWEEP_PARAMS = {
    'visibility_threshold': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.7,
    },
    'uv_size': {
        'values': [256, 512, 1024],
    },
    'fusion_method': {
        'values': ['average', 'visibility_weighted', 'max_visibility'],
    },
    'w_tv': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-2,
    },
    'do_optimization': {
        'values': [False, True],
    },
    'opt_iters': {
        'values': [30, 50, 100],
    },
}


class WandBSweepOptimizer:
    """
    WandB Sweep-based hyperparameter optimizer for UV mapping.

    Usage:
        optimizer = WandBSweepOptimizer(config)
        best_params = optimizer.run_sweep(fitting_result_dir)

    Or start agent for existing sweep:
        optimizer.run_agent(sweep_id, fitting_result_dir)
    """

    def __init__(
        self,
        config: WandBSweepConfig,
        param_space: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            config: Sweep configuration
            param_space: Parameter search space (uses default if None)
        """
        if not _check_wandb():
            raise ImportError("wandb is required. Install with: pip install wandb")

        self.config = config
        self.param_space = param_space or DEFAULT_SWEEP_PARAMS.copy()

        os.makedirs(config.output_dir, exist_ok=True)

    def create_sweep(self) -> str:
        """
        Create a new WandB sweep.

        Returns:
            sweep_id: WandB sweep ID
        """
        sweep_config = {
            'name': self.config.sweep_name,
            'method': self.config.method,
            'metric': {
                'name': self.config.metric_name,
                'goal': self.config.metric_goal,
            },
            'parameters': self.param_space,
        }

        sweep_id = wandb.sweep(
            sweep_config,
            project=self.config.project,
            entity=self.config.entity,
        )

        logger.info(f"Created sweep: {sweep_id}")
        logger.info(f"Dashboard: https://wandb.ai/{self.config.entity or 'your-entity'}/{self.config.project}/sweeps/{sweep_id}")

        return sweep_id

    def run_sweep(
        self,
        fitting_result_dir: str,
        sweep_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run complete sweep optimization.

        Args:
            fitting_result_dir: Path to fitting results
            sweep_id: Existing sweep ID (creates new if None)

        Returns:
            best_params: Best found parameters
        """
        # Create sweep if needed
        if sweep_id is None:
            sweep_id = self.create_sweep()

        # Save sweep info
        sweep_info = {
            'sweep_id': sweep_id,
            'fitting_result_dir': fitting_result_dir,
            'config': asdict(self.config),
        }
        with open(os.path.join(self.config.output_dir, 'sweep_info.json'), 'w') as f:
            json.dump(sweep_info, f, indent=2)

        # Define training function
        def train_fn():
            self._run_single_trial(fitting_result_dir)

        # Run agent
        wandb.agent(
            sweep_id,
            function=train_fn,
            count=self.config.count,
            project=self.config.project,
            entity=self.config.entity,
        )

        # Get best run
        best_params = self._get_best_params(sweep_id)

        return best_params

    def run_agent(
        self,
        sweep_id: str,
        fitting_result_dir: str,
        count: Optional[int] = None,
    ) -> None:
        """
        Run sweep agent for existing sweep.

        Useful for distributed optimization across multiple machines.

        Args:
            sweep_id: WandB sweep ID
            fitting_result_dir: Path to fitting results
            count: Number of runs for this agent (None = unlimited)
        """
        def train_fn():
            self._run_single_trial(fitting_result_dir)

        wandb.agent(
            sweep_id,
            function=train_fn,
            count=count or self.config.count,
            project=self.config.project,
            entity=self.config.entity,
        )

    def _run_single_trial(
        self,
        fitting_result_dir: str,
    ) -> None:
        """
        Run a single trial with current wandb config.
        """
        # Initialize wandb run
        run = wandb.init()
        config = wandb.config

        try:
            # Log trial start
            logger.info(f"Trial started: {run.name}")
            logger.info(f"Parameters: {dict(config)}")

            start_time = time.time()

            # Run UV mapping with current params
            metrics = self._evaluate_config(fitting_result_dir, dict(config))

            runtime = time.time() - start_time

            # Compute composite score
            score = self._compute_score(metrics)

            # Log all metrics
            wandb.log({
                'coverage': metrics['coverage'],
                'mean_confidence': metrics['mean_confidence'],
                'seam_discontinuity': metrics['seam_discontinuity'],
                'runtime_seconds': runtime,
                'score': score,
            })

            # Log texture image if available
            if 'texture_path' in metrics and os.path.exists(metrics['texture_path']):
                wandb.log({
                    'texture': wandb.Image(metrics['texture_path']),
                })

            logger.info(f"Trial complete: score={score:.4f}, coverage={metrics['coverage']:.1f}%")

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            wandb.log({'error': str(e), 'score': 0.0})

        finally:
            wandb.finish()

    def _evaluate_config(
        self,
        fitting_result_dir: str,
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Evaluate UV mapping with given parameters.

        Args:
            fitting_result_dir: Path to fitting results
            params: Parameter dictionary

        Returns:
            metrics: Evaluation metrics
        """
        from .uv_pipeline import UVMapPipeline, UVPipelineConfig
        from .experiment_runner import UVMapEvaluator

        # Create pipeline config from params
        pipeline_config = UVPipelineConfig(
            result_dir=fitting_result_dir,
            uv_size=params.get('uv_size', 512),
            use_visibility_weighting=(params.get('fusion_method', 'visibility_weighted') != 'average'),
            visibility_threshold=params.get('visibility_threshold', 0.3),
            do_optimization=params.get('do_optimization', False),
            opt_iters=params.get('opt_iters', 50),
            opt_w_tv=params.get('w_tv', 1e-3),
            frame_interval=1,
            output_dir=os.path.join(self.config.output_dir, f"run_{wandb.run.name}"),
            save_intermediate=False,
        )

        # Run pipeline
        pipeline = UVMapPipeline(pipeline_config)
        pipeline.setup()

        # Limit frames for faster evaluation
        max_frames = params.get('max_frames', 30)
        if max_frames > 0:
            pipeline.frames = pipeline.frames[:max_frames]

        texture = pipeline.run()

        # Evaluate
        evaluator = UVMapEvaluator()
        vertex_colors, confidence = pipeline.texture_accumulator.get_texture()
        uv_mask = pipeline.uv_renderer.get_uv_mask()

        coverage = evaluator.compute_coverage(texture, uv_mask)
        conf_stats = evaluator.compute_confidence_stats(confidence)
        seam_disc = evaluator.compute_seam_discontinuity(texture)

        metrics = {
            'coverage': coverage,
            'mean_confidence': conf_stats['mean'],
            'seam_discontinuity': seam_disc,
            'texture_path': os.path.join(pipeline_config.output_dir, 'texture_final.png'),
        }

        return metrics

    def _compute_score(
        self,
        metrics: Dict[str, float],
    ) -> float:
        """
        Compute composite optimization score.

        Higher is better.
        """
        coverage_score = metrics['coverage'] / 100.0  # Normalize to [0, 1]
        confidence_score = metrics['mean_confidence']  # Already [0, 1]

        # Invert seam (lower is better)
        seam_score = max(0, 1.0 - metrics['seam_discontinuity'] * 10)

        score = (
            self.config.w_coverage * coverage_score +
            self.config.w_psnr * confidence_score +
            self.config.w_seam * seam_score
        )

        return score

    def _get_best_params(
        self,
        sweep_id: str,
    ) -> Dict[str, Any]:
        """
        Get best parameters from completed sweep.

        Args:
            sweep_id: WandB sweep ID

        Returns:
            best_params: Best found parameters
        """
        api = wandb.Api()
        sweep = api.sweep(f"{self.config.entity}/{self.config.project}/{sweep_id}")

        # Get best run
        best_run = sweep.best_run()
        best_params = dict(best_run.config)

        # Save best config
        best_info = {
            'sweep_id': sweep_id,
            'best_run_name': best_run.name,
            'best_params': best_params,
            'best_score': best_run.summary.get('score', 0),
            'best_coverage': best_run.summary.get('coverage', 0),
        }

        with open(os.path.join(self.config.output_dir, 'best_config.json'), 'w') as f:
            json.dump(best_info, f, indent=2)

        logger.info(f"Best run: {best_run.name}")
        logger.info(f"Best score: {best_info['best_score']:.4f}")
        logger.info(f"Best params: {best_params}")

        return best_params


def run_wandb_sweep(
    fitting_result_dir: str,
    output_dir: str = "wandb_sweep_results",
    n_trials: int = 30,
    project: str = "uvmap-optimization",
) -> Dict[str, Any]:
    """
    Convenience function to run WandB sweep optimization.

    Args:
        fitting_result_dir: Path to fitting results
        output_dir: Output directory
        n_trials: Number of trials
        project: WandB project name

    Returns:
        best_params: Best found parameters
    """
    config = WandBSweepConfig(
        project=project,
        output_dir=output_dir,
        count=n_trials,
    )

    optimizer = WandBSweepOptimizer(config)
    best_params = optimizer.run_sweep(fitting_result_dir)

    return best_params


# CLI support
if __name__ == '__main__':
    import argparse
    import sys

    # Check wandb first
    if not _check_wandb():
        print("ERROR: wandb not installed.")
        print("Install with: pip install wandb")
        print("\nAlternative: Use Optuna optimizer instead:")
        print("  python -m uvmap.optuna_optimizer --result_dir <path>")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='WandB Sweep UV Map Optimization')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Fitting result directory')
    parser.add_argument('--output_dir', type=str, default='wandb_sweep_results',
                       help='Output directory')
    parser.add_argument('--project', type=str, default='uvmap-optimization',
                       help='WandB project name')
    parser.add_argument('--count', type=int, default=30,
                       help='Number of trials')
    parser.add_argument('--sweep_id', type=str, default=None,
                       help='Existing sweep ID (to join as agent)')

    args = parser.parse_args()

    config = WandBSweepConfig(
        project=args.project,
        output_dir=args.output_dir,
        count=args.count,
    )

    optimizer = WandBSweepOptimizer(config)

    if args.sweep_id:
        # Join existing sweep as agent
        optimizer.run_agent(args.sweep_id, args.result_dir)
    else:
        # Create and run new sweep
        best_params = optimizer.run_sweep(args.result_dir)
        print(f"\nBest parameters: {best_params}")
