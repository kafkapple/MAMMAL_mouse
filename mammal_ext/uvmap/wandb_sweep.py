"""WandB Sweep-based Hyperparameter Optimization for UV Mapping.

This module has been split into:
- sweep_config.py: WandBSweepConfig, get_sweep_params_for_stage
- sweep_metrics.py: compute_psnr_masked, compute_ssim_masked, create_mesh_mask
- sweep_runner.py: WandBSweepOptimizer, run_wandb_sweep

This file re-exports all public symbols for backward compatibility.
"""

# Re-export all public symbols
from mammal_ext.uvmap.sweep_config import WandBSweepConfig, get_sweep_params_for_stage
from mammal_ext.uvmap.sweep_metrics import compute_psnr_masked, compute_ssim_masked, create_mesh_mask
from mammal_ext.uvmap.sweep_runner import WandBSweepOptimizer, run_wandb_sweep

__all__ = [
    "WandBSweepConfig",
    "get_sweep_params_for_stage",
    "compute_psnr_masked",
    "compute_ssim_masked",
    "create_mesh_mask",
    "WandBSweepOptimizer",
    "run_wandb_sweep",
]
