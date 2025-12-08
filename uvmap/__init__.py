"""
MAMMAL Mouse UV Map Module

UV texture mapping pipeline for mouse mesh model.
Inspired by VHAP (Versatile Head Alignment with Adaptive Appearance Priors).

Modules:
- uv_renderer: Differentiable UV space rendering
- texture_sampler: Multi-view texture sampling
- texture_optimizer: Photometric optimization
- uv_pipeline: End-to-end UV map generation
- experiment_runner: Systematic parameter ablation
- evaluation_report: HTML report generation
- wandb_sweep: WandB Sweep hyperparameter optimization (DEFAULT)
- optuna_optimizer: Optuna Bayesian hyperparameter optimization (alternative)

Default Optimizer: WandB Sweep
    - Real-time dashboard visualization
    - Easy parallel agent support
    - Bayesian/Random/Grid search methods
"""

from .uv_renderer import UVRenderer, create_uv_renderer
from .texture_sampler import TextureSampler, TextureAccumulator
from .texture_optimizer import TextureOptimizer, TextureOptConfig, TextureModel
from .uv_pipeline import UVMapPipeline, UVPipelineConfig, run_uvmap_pipeline
from .experiment_runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    UVMapEvaluator,
    run_ablation_study,
)
from .evaluation_report import HTMLReportGenerator, generate_evaluation_report

# WandB Sweep (DEFAULT optimizer)
try:
    from .wandb_sweep import (
        WandBSweepOptimizer,
        WandBSweepConfig,
        run_wandb_sweep,
        DEFAULT_SWEEP_PARAMS,
    )
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optuna (alternative optimizer)
from .optuna_optimizer import (
    QUALITATIVE_CRITERIA,
    OptimizationConfig as OptunaConfig,
)

# Lazy import for Optuna classes (only when optuna is available)
def _get_optuna_optimizer():
    from .optuna_optimizer import OptunaUVOptimizer
    return OptunaUVOptimizer

def _get_multi_objective_optimizer():
    from .optuna_optimizer import MultiObjectiveUVMapObjective as MultiObjectiveUVOptimizer
    return MultiObjectiveUVOptimizer

__all__ = [
    # Core
    'UVRenderer',
    'create_uv_renderer',
    'TextureSampler',
    'TextureAccumulator',
    'TextureOptimizer',
    'TextureOptConfig',
    'TextureModel',
    # Pipeline
    'UVMapPipeline',
    'UVPipelineConfig',
    'run_uvmap_pipeline',
    # Experiments
    'ExperimentConfig',
    'ExperimentResult',
    'ExperimentRunner',
    'UVMapEvaluator',
    'run_ablation_study',
    # Reports
    'HTMLReportGenerator',
    'generate_evaluation_report',
    # WandB Sweep Optimization (DEFAULT)
    'WandBSweepOptimizer',
    'WandBSweepConfig',
    'run_wandb_sweep',
    'DEFAULT_SWEEP_PARAMS',
    'WANDB_AVAILABLE',
    # Optuna Optimization (alternative)
    'OptunaConfig',
    'QUALITATIVE_CRITERIA',
    # Helper function
    'optimize_uvmap',
]


def optimize_uvmap(
    fitting_result_dir: str,
    output_dir: str = "uvmap_optimization",
    method: str = "wandb",  # 'wandb' (default) or 'optuna'
    n_trials: int = 30,
    **kwargs,
):
    """
    Unified interface for UV map hyperparameter optimization.

    Args:
        fitting_result_dir: Path to fitting results
        output_dir: Output directory
        method: Optimization method ('wandb' or 'optuna')
        n_trials: Number of optimization trials
        **kwargs: Additional method-specific arguments

    Returns:
        best_params: Best found parameters
    """
    if method == "wandb":
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Use method='optuna' or install wandb.")
        return run_wandb_sweep(
            fitting_result_dir=fitting_result_dir,
            output_dir=output_dir,
            n_trials=n_trials,
            **kwargs,
        )
    elif method == "optuna":
        from .optuna_optimizer import OptunaUVOptimizer
        config = OptunaConfig(
            n_trials=n_trials,
            output_dir=output_dir,
            **kwargs,
        )
        optimizer = OptunaUVOptimizer(fitting_result_dir, config)
        return optimizer.optimize()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'wandb' or 'optuna'.")
