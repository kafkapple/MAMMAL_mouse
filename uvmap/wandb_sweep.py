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
from typing import Dict, List, Any, Optional, Callable, Tuple
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


# ===== Photometric Metrics Helper Functions =====
# Based on 3DGS (SIGGRAPH 2023) and IQA literature

def compute_psnr_masked(
    rendered: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute PSNR between rendered and target images in masked region.

    Args:
        rendered: [H, W, 3] Rendered RGB (0-255, uint8)
        target: [H, W, 3] Original RGB (0-255, uint8)
        mask: [H, W] Boolean mask for valid region

    Returns:
        psnr_score: [0, 1] normalized PSNR score
        psnr_db: Raw PSNR value in dB
    """
    if mask.sum() < 100:  # Too few pixels
        return 0.0, 0.0

    # Convert to float
    rendered_f = rendered.astype(np.float32)
    target_f = target.astype(np.float32)

    # Apply mask
    rendered_masked = rendered_f[mask]
    target_masked = target_f[mask]

    # MSE calculation
    mse = np.mean((rendered_masked - target_masked) ** 2)

    if mse < 1e-10:
        psnr_db = 100.0
    else:
        psnr_db = 10 * np.log10(255.0 ** 2 / mse)

    # Normalize to [0, 1] (PSNR typically 15-40 dB for reasonable results)
    psnr_min, psnr_max = 15.0, 40.0
    psnr_score = np.clip((psnr_db - psnr_min) / (psnr_max - psnr_min), 0, 1)

    return psnr_score, psnr_db


def compute_ssim_masked(
    rendered: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute SSIM between rendered and target images.

    Uses bounding box of mask for efficient computation.

    Args:
        rendered: [H, W, 3] Rendered RGB (0-255, uint8)
        target: [H, W, 3] Original RGB (0-255, uint8)
        mask: [H, W] Boolean mask for valid region

    Returns:
        ssim_score: [0, 1] structural similarity
        ssim_raw: Raw SSIM value
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        logger.warning("skimage not available, returning 0 for SSIM")
        return 0.0, 0.0

    # Find bounding box of mask
    y_indices, x_indices = np.where(mask)
    if len(y_indices) < 10:
        return 0.0, 0.0

    y1, y2 = y_indices.min(), y_indices.max() + 1
    x1, x2 = x_indices.min(), x_indices.max() + 1

    # Ensure minimum size for SSIM (at least 7x7 for default window)
    if (y2 - y1) < 7 or (x2 - x1) < 7:
        return 0.0, 0.0

    rendered_crop = rendered[y1:y2, x1:x2]
    target_crop = target[y1:y2, x1:x2]

    # SSIM calculation (channel_axis for RGB)
    ssim_val = ssim(
        target_crop,
        rendered_crop,
        channel_axis=2,
        data_range=255,
        win_size=min(7, min(rendered_crop.shape[0], rendered_crop.shape[1]) // 2 * 2 - 1)
    )

    return ssim_val, ssim_val


def create_mesh_mask(
    rendered: np.ndarray,
    background_value: int = 255,
) -> np.ndarray:
    """
    Create a mask for mesh region (non-background pixels).

    Args:
        rendered: [H, W, 3] Rendered image with white background
        background_value: Background pixel value (255 for white)

    Returns:
        mask: [H, W] Boolean mask (True = mesh region)
    """
    # Check if pixel is not pure white (background)
    is_background = np.all(rendered == background_value, axis=2)
    return ~is_background


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

    # Frame sampling for faster evaluation
    max_frames: int = 20  # Limit frames (0 = all frames)
    frame_sampling: str = "uniform"  # 'uniform', 'random', 'keyframes'

    # Objective weights (for composite score v3)
    # Based on 3DGS (SIGGRAPH 2023): L = (1-λ)×L1 + λ×D-SSIM, λ=0.2
    w_photo: float = 0.50   # Photometric (PSNR-based) - primary quality metric
    w_ssim: float = 0.15    # Structural similarity - perceptual quality
    w_coverage: float = 0.20  # UV space coverage
    w_seam: float = 0.15    # Seam discontinuity (lower is better)

    # ===== Search Space Optimization (v2) =====
    # uv_size 고정 옵션: Resolution Bias 제거
    # - True: uv_size=512로 고정 (권장, 탐색 효율화)
    # - False: [256, 512, 1024] 중 탐색
    fix_uv_size: bool = True
    fixed_uv_size: int = 512  # fix_uv_size=True일 때 사용할 해상도

    # 2-Stage Optimization 전략
    # - stage_a: do_optimization=False, 구조 파라미터만 최적화 (빠른 탐색)
    # - stage_b: do_optimization=True, 미세 조정 파라미터 최적화
    # - full: 모든 파라미터 동시 최적화 (기존 방식)
    optimization_stage: str = "full"  # 'stage_a', 'stage_b', 'full'

    # ===== Visualization Logging =====
    # Enable 3D mesh rendering visualization in wandb
    # Disabled by default - 6-view projection grid (log_projection_grid) is more informative
    log_rendered_mesh: bool = False
    render_views: List[str] = field(default_factory=lambda: ['front', 'side', 'diagonal'])
    render_image_size: Tuple[int, int] = (512, 512)
    render_distance_factor: float = 2.5  # Camera distance as multiple of mesh scale (smaller = closer)
    log_orbit_video: bool = False  # Slower, but more informative
    orbit_frames: int = 30  # Frames for orbit video (if enabled)

    # ===== 6-View Projection Grid Settings =====
    log_projection_grid: bool = True  # Enable 6-view comparison grid logging
    projection_face_sampling: int = 1  # Render ALL faces for accurate visualization (1=full quality)
    projection_max_width: int = 2560  # Max grid width (pixels)

    # Output
    output_dir: str = "results/sweep"


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


def get_sweep_params_for_stage(
    stage: str,
    fix_uv_size: bool = True,
    fixed_uv_size: int = 512,
    stage_a_best_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    2-Stage Optimization을 위한 파라미터 공간 생성.

    Stage 전략:
    - stage_a (Structure): Sampling, Fusion, Visibility 관련 파라미터만 최적화 (빠른 탐색)
    - stage_b (Refinement): Stage A Best Config 기반, 미세 조정 파라미터만 최적화
    - full: 모든 파라미터 동시 최적화 (기존 방식)

    왜 2-Stage가 필요한가?
    - 서로 다른 성격의 파라미터를 분리하여 탐색 효율성 ↑
    - Stage A에서 구조적 최적점을 찾고, Stage B에서 품질 미세 조정
    - 총 탐색 비용: Stage A(20회) + Stage B(20회) < Full(50회) 동등 수준

    Args:
        stage: 'stage_a', 'stage_b', 'full'
        fix_uv_size: True면 uv_size 고정 (Resolution Bias 제거)
        fixed_uv_size: 고정할 uv_size 값
        stage_a_best_config: Stage B 실행 시 Stage A의 Best Config

    Returns:
        params: WandB Sweep parameter space
    """
    if stage == "stage_a":
        # ===== Stage A: Structure Optimization =====
        # - do_optimization=False (빠른 평가)
        # - Sampling, Fusion, Visibility 파라미터만 탐색
        params = {
            'visibility_threshold': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.7,
            },
            'fusion_method': {
                'values': ['average', 'visibility_weighted', 'max_visibility'],
            },
            # do_optimization 고정 (False)
            'do_optimization': {
                'value': False,
            },
        }

    elif stage == "stage_b":
        # ===== Stage B: Refinement Optimization =====
        # - Stage A의 Best Config 고정
        # - Photometric optimization 관련 파라미터만 탐색
        if stage_a_best_config is None:
            logger.warning("stage_b requires stage_a_best_config, using defaults")
            stage_a_best_config = {
                'visibility_threshold': 0.3,
                'fusion_method': 'visibility_weighted',
            }

        params = {
            # Stage A에서 찾은 값 고정
            'visibility_threshold': {
                'value': stage_a_best_config.get('visibility_threshold', 0.3),
            },
            'fusion_method': {
                'value': stage_a_best_config.get('fusion_method', 'visibility_weighted'),
            },
            # Refinement 파라미터 탐색
            'do_optimization': {
                'value': True,  # 항상 True
            },
            'opt_iters': {
                'values': [30, 50, 100, 150],
            },
            'w_tv': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2,
            },
            'opt_lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2,
            },
        }

    else:  # "full"
        # ===== Full: 기존 방식 (모든 파라미터 동시 탐색) =====
        params = DEFAULT_SWEEP_PARAMS.copy()

    # ===== uv_size 처리 =====
    if fix_uv_size:
        # Resolution Bias 제거: uv_size 고정
        params['uv_size'] = {'value': fixed_uv_size}
        logger.info(f"uv_size fixed to {fixed_uv_size} (Resolution Bias 제거)")
    elif 'uv_size' not in params:
        # Stage A/B에서 uv_size가 없으면 추가
        params['uv_size'] = {'value': fixed_uv_size}

    return params


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
        stage_a_best_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            config: Sweep configuration
            param_space: Parameter search space (uses default if None)
            stage_a_best_config: Stage A best config (for stage_b)
        """
        if not _check_wandb():
            raise ImportError("wandb is required. Install with: pip install wandb")

        self.config = config
        self.stage_a_best_config = stage_a_best_config

        # ===== 파라미터 공간 자동 선택 (v2) =====
        # optimization_stage에 따라 적절한 파라미터 공간 사용
        if param_space is not None:
            self.param_space = param_space
        else:
            self.param_space = get_sweep_params_for_stage(
                stage=config.optimization_stage,
                fix_uv_size=config.fix_uv_size,
                fixed_uv_size=config.fixed_uv_size,
                stage_a_best_config=stage_a_best_config,
            )
            logger.info(f"Using {config.optimization_stage} stage params: {list(self.param_space.keys())}")

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
        # Set wandb log directory to results/wandb/
        wandb_dir = os.path.join(os.path.dirname(self.config.output_dir), 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ['WANDB_DIR'] = os.path.abspath(wandb_dir)

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
        # Set wandb log directory to results/wandb/
        wandb_dir = os.path.join(os.path.dirname(self.config.output_dir), 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ['WANDB_DIR'] = os.path.abspath(wandb_dir)

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

            # ===== Compute photometric metrics (PSNR/SSIM) via 6-view projection =====
            # Must be done BEFORE _compute_score() for score v3
            projection_result = self._render_6view_projection_grid(metrics, frame_idx=0)

            # Merge photometric metrics into metrics dict for score calculation
            metrics['mean_psnr_score'] = projection_result.get('mean_psnr_score', 0.0)
            metrics['mean_ssim_score'] = projection_result.get('mean_ssim_score', 0.0)
            metrics['mean_psnr_db'] = projection_result.get('mean_psnr_db', 0.0)

            runtime = time.time() - start_time

            # Compute composite score (v3 with photometric metrics)
            score = self._compute_score(metrics)

            # Log all metrics
            log_dict = {
                'coverage': metrics['coverage'],
                'mean_confidence': metrics['mean_confidence'],
                'seam_discontinuity': metrics['seam_discontinuity'],
                'mean_psnr_score': metrics['mean_psnr_score'],
                'mean_ssim_score': metrics['mean_ssim_score'],
                'mean_psnr_db': metrics['mean_psnr_db'],
                'runtime_seconds': runtime,
                'score': score,
            }

            # Log visualization images
            if 'texture_path' in metrics and os.path.exists(metrics['texture_path']):
                log_dict['uv_texture'] = wandb.Image(
                    metrics['texture_path'],
                    caption=f"UV Texture (cov={metrics['coverage']:.1f}%)"
                )

            if 'confidence_path' in metrics and os.path.exists(metrics['confidence_path']):
                log_dict['confidence_map'] = wandb.Image(
                    metrics['confidence_path'],
                    caption=f"Confidence (mean={metrics['mean_confidence']:.3f})"
                )

            if 'uv_mask_path' in metrics and os.path.exists(metrics['uv_mask_path']):
                log_dict['uv_mask'] = wandb.Image(
                    metrics['uv_mask_path'],
                    caption="UV Mask"
                )

            # ===== Log 6-View Projection Grid (already computed above) =====
            grid_path = projection_result.get('grid_path')
            if grid_path and os.path.exists(grid_path):
                log_dict['projection_6view'] = wandb.Image(
                    grid_path,
                    caption=f"6-View Projection (PSNR={metrics['mean_psnr_db']:.1f}dB, SSIM={metrics['mean_ssim_score']:.3f})"
                )

            # ===== Log 3D Rendered Mesh Visualization =====
            if self.config.log_rendered_mesh and 'texture_path' in metrics:
                try:
                    render_outputs = self._render_mesh_visualization(
                        fitting_result_dir,
                        metrics['texture_path'],
                    )

                    # Log rendered images
                    for view_name, img_path in render_outputs.get('images', {}).items():
                        if os.path.exists(img_path):
                            log_dict[f'render_{view_name}'] = wandb.Image(
                                img_path,
                                caption=f"Rendered {view_name} view"
                            )

                    # Log orbit video (if enabled)
                    if 'orbit_video' in render_outputs and os.path.exists(render_outputs['orbit_video']):
                        log_dict['render_orbit'] = wandb.Video(
                            render_outputs['orbit_video'],
                            caption="360° orbit view"
                        )

                except Exception as e:
                    logger.warning(f"Failed to render mesh visualization: {e}")

            wandb.log(log_dict)

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

        # Apply frame sampling based on config
        max_frames = self.config.max_frames
        if max_frames > 0 and len(pipeline.frames) > max_frames:
            pipeline.frames = self._sample_frames(
                pipeline.frames,
                max_frames,
                self.config.frame_sampling,
            )

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
            'confidence_path': os.path.join(pipeline_config.output_dir, 'confidence.png'),
            'uv_mask_path': os.path.join(pipeline_config.output_dir, 'uv_mask.png'),
            # Additional info for 6-view projection visualization
            'output_dir': pipeline_config.output_dir,
            'data_dir': pipeline.data_dir,
            'views_to_use': pipeline.views_to_use,
            'vertex_colors': vertex_colors,
            'fitting_result_dir': fitting_result_dir,
        }

        return metrics

    def _sample_frames(
        self,
        frames: List[int],
        max_frames: int,
        method: str = "uniform",
    ) -> List[int]:
        """
        Sample frames for faster evaluation.

        Args:
            frames: List of all frame indices
            max_frames: Maximum number of frames to use
            method: Sampling method
                - 'uniform': Evenly spaced (default, best coverage)
                - 'random': Random sampling (good for diversity)
                - 'keyframes': First, middle, last + uniform (motion aware)

        Returns:
            sampled_frames: Subset of frame indices
        """
        n_total = len(frames)
        if n_total <= max_frames:
            return frames

        if method == "uniform":
            # Evenly spaced sampling
            indices = np.linspace(0, n_total - 1, max_frames, dtype=int)
            return [frames[i] for i in indices]

        elif method == "random":
            # Random sampling (deterministic with seed for reproducibility)
            np.random.seed(42)
            indices = np.sort(np.random.choice(n_total, max_frames, replace=False))
            return [frames[i] for i in indices]

        elif method == "keyframes":
            # First, middle, last + uniform fill
            keyframe_indices = [0, n_total // 2, n_total - 1]
            remaining = max_frames - len(keyframe_indices)
            if remaining > 0:
                # Fill uniformly between keyframes
                fill_indices = np.linspace(0, n_total - 1, remaining + 2, dtype=int)[1:-1]
                all_indices = sorted(set(keyframe_indices) | set(fill_indices.tolist()))
            else:
                all_indices = keyframe_indices[:max_frames]
            return [frames[i] for i in all_indices]

        else:
            logger.warning(f"Unknown sampling method '{method}', using uniform")
            return self._sample_frames(frames, max_frames, "uniform")

    def _compute_score(
        self,
        metrics: Dict[str, float],
    ) -> float:
        """
        Compute composite optimization score (v3 - Photometric-Aware).

        Higher is better.

        개선사항 (v3):
        1. Photometric Score 추가: PSNR 기반 Rendered vs Original 비교
           - 3DGS (SIGGRAPH 2023) 기반: L = (1-λ)×L1 + λ×D-SSIM
           - PSNR normalized to [0, 1] (15-40 dB range)

        2. SSIM Score 추가: Structural similarity (human perception)
           - skimage.metrics.structural_similarity 사용
           - Bounding box crop으로 효율적 계산

        3. Seam Score: Exponential Decay (v2에서 유지)
           - exp(-k * seam) → 연속적인 기울기로 Optimizer 신호 유지

        4. Coverage Gating: 최소 커버리지 미달 시 페널티
           - Coverage < 80%이면 전체 점수 * 0.1
        """
        coverage_score = metrics['coverage'] / 100.0  # Normalize to [0, 1]

        # ===== Photometric Scores (v3 NEW) =====
        # PSNR score: from 6-view projection comparison
        photo_score = metrics.get('mean_psnr_score', 0.0)
        ssim_score_val = metrics.get('mean_ssim_score', 0.0)

        # Fallback: if photometric metrics not computed, use confidence as proxy
        if photo_score == 0.0 and ssim_score_val == 0.0:
            logger.warning("Photometric metrics not available, using mean_confidence as fallback")
            photo_score = metrics.get('mean_confidence', 0.0)
            ssim_score_val = metrics.get('mean_confidence', 0.0)

        # ===== Seam Score: Exponential Decay (v2) =====
        # Score = exp(-k * seam), k=15 (민감도 상수)
        #   - seam=0.0 → score=1.0 (완벽)
        #   - seam=0.05 → score≈0.47 (양호)
        #   - seam=0.1 → score≈0.22 (주의 필요)
        #   - seam=0.2 → score≈0.05 (나쁨, but 기울기 존재)
        seam_val = metrics['seam_discontinuity']
        if np.isnan(seam_val):
            logger.warning(f"seam_discontinuity is NaN, using 0.0")
            seam_val = 0.0

        seam_sensitivity = 15.0  # k: 민감도 상수 (값이 클수록 seam에 민감)
        seam_score = np.exp(-seam_sensitivity * seam_val)

        # ===== 가중 합산 (v3) =====
        # Based on 3DGS (Kerbl et al., SIGGRAPH 2023):
        #   - w_photo=0.50: Primary photometric quality
        #   - w_ssim=0.15: Structural/perceptual similarity
        #   - w_coverage=0.20: UV space utilization
        #   - w_seam=0.15: Texture continuity at seams
        score = (
            self.config.w_photo * photo_score +
            self.config.w_ssim * ssim_score_val +
            self.config.w_coverage * coverage_score +
            self.config.w_seam * seam_score
        )

        # ===== Coverage Gating (v2) =====
        # 커버리지가 최소 기준(80%) 미달 시 전체 점수에 페널티
        # 목적: UV 공간 활용도가 낮은 결과는 다른 지표가 좋아도 탈락
        coverage_threshold = 80.0  # 최소 커버리지 기준 (%)
        coverage_penalty = 0.1     # 페널티 계수

        if metrics['coverage'] < coverage_threshold:
            logger.info(f"Coverage Gating: {metrics['coverage']:.1f}% < {coverage_threshold}% → score *= {coverage_penalty}")
            score *= coverage_penalty

        # Final NaN check
        if np.isnan(score):
            logger.warning(f"Score is NaN! metrics={metrics}")
            return 0.0

        return score

    def _render_mesh_visualization(
        self,
        fitting_result_dir: str,
        texture_path: str,
    ) -> Dict[str, Any]:
        """
        Render UV-textured mesh from multiple viewpoints.

        Creates images and optional video for wandb logging.

        Args:
            fitting_result_dir: Path to fitting results
            texture_path: Path to UV texture image

        Returns:
            outputs: Dictionary with 'images' and optionally 'orbit_video'
        """
        from visualization.textured_renderer import create_textured_renderer
        from visualization.camera_paths import CameraPathGenerator, compute_mesh_bounds
        from visualization.video_generator import VideoGenerator
        import cv2
        import pickle
        import glob

        outputs = {'images': {}}
        run_dir = os.path.dirname(texture_path)

        # Find first parameter file to get mesh
        params_dir = os.path.join(fitting_result_dir, 'params')
        params_files = sorted(glob.glob(os.path.join(params_dir, 'step_2_frame_*.pkl')))
        if not params_files:
            params_files = sorted(glob.glob(os.path.join(params_dir, 'step_1_frame_*.pkl')))
        if not params_files:
            logger.warning("No parameter files found for rendering")
            return outputs

        # Load body model and get vertices
        from articulation_th import ArticulationTorch
        body_model = ArticulationTorch()

        with open(params_files[0], 'rb') as f:
            params = pickle.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for k, v in params.items():
            if not isinstance(v, torch.Tensor):
                params[k] = torch.tensor(v, dtype=torch.float32, device=device)

        V, J = body_model.forward(
            params["thetas"], params["bone_lengths"],
            params["rotation"], params["trans"] / 1000,
            params["scale"] / 1000,
            params.get("chest_deformer", torch.zeros(1, 1, device=device)),
        )
        vertices = V[0].detach().cpu().numpy()

        # Setup renderer
        try:
            renderer = create_textured_renderer(
                model_dir='mouse_model/mouse_txt',
                texture_path=texture_path,
                image_size=self.config.render_image_size,
                backend='pyrender',
            )
        except Exception as e:
            logger.warning(f"Failed to create renderer: {e}")
            return outputs

        # Setup cameras
        center, scale = compute_mesh_bounds(vertices)
        cam_gen = CameraPathGenerator(center, scale)

        # Render fixed views
        fixed_poses = cam_gen.fixed_views(views=self.config.render_views, distance_factor=self.config.render_distance_factor)

        for pose in fixed_poses:
            try:
                cam_matrix = pose.to_pyrender_pose()
                image = renderer.render_pyrender(vertices, cam_matrix)

                # Save image
                img_path = os.path.join(run_dir, f'render_{pose.name}.png')
                cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                outputs['images'][pose.name] = img_path

            except Exception as e:
                logger.warning(f"Failed to render {pose.name}: {e}")

        # Render orbit video (if enabled)
        if self.config.log_orbit_video:
            try:
                orbit_poses = cam_gen.orbit_360(
                    n_frames=self.config.orbit_frames,
                    elevation=30.0,
                    distance_factor=self.config.render_distance_factor,
                )

                video_path = os.path.join(run_dir, 'render_orbit.mp4')
                with VideoGenerator(video_path, fps=15) as gen:
                    for pose in orbit_poses:
                        cam_matrix = pose.to_pyrender_pose()
                        image = renderer.render_pyrender(vertices, cam_matrix)
                        gen.add_frame(image)

                outputs['orbit_video'] = video_path

            except Exception as e:
                logger.warning(f"Failed to render orbit video: {e}")

        return outputs

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

    def _render_6view_projection_grid(
        self,
        metrics: Dict[str, Any],
        frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Render textured mesh projected to 6 camera views as a comparison grid.

        Also computes photometric metrics (PSNR, SSIM) for score v3.

        Layout: Each view shows [Original | Rendered] side by side for comparison.
        Grid: 2 rows x 3 columns of view pairs.

        Args:
            metrics: Metrics dict containing vertex_colors, data_dir, etc.
            frame_idx: Frame index to render (default: 0)

        Returns:
            Dict containing:
                - grid_path: Path to saved grid image (or None if failed)
                - mean_psnr_score: Average PSNR score across views [0, 1]
                - mean_ssim_score: Average SSIM score across views [0, 1]
                - mean_psnr_db: Average raw PSNR value (dB)
                - per_view_psnr: List of per-view PSNR scores
                - per_view_ssim: List of per-view SSIM scores
        """
        import cv2
        import pickle

        try:
            output_dir = metrics['output_dir']
            data_dir = metrics['data_dir']
            views_to_use = metrics['views_to_use']
            vertex_colors = metrics['vertex_colors']
            fitting_result_dir = metrics['fitting_result_dir']

            # Load camera parameters
            cam_path = os.path.join(data_dir, 'new_cam.pkl')
            with open(cam_path, 'rb') as f:
                cams = pickle.load(f)

            # Load mesh for first frame
            params_dir = os.path.join(fitting_result_dir, 'params')
            import glob
            param_files = sorted(glob.glob(os.path.join(params_dir, 'step_2_frame_*.pkl')))
            if not param_files:
                logger.warning("No parameter files found")
                return None

            with open(param_files[frame_idx], 'rb') as f:
                params = pickle.load(f)

            # Forward mesh
            from articulation_th import ArticulationTorch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for k, v in params.items():
                if not isinstance(v, torch.Tensor):
                    params[k] = torch.tensor(v, dtype=torch.float32, device=device)

            body_model = ArticulationTorch()
            V, J = body_model.forward(
                params['thetas'], params['bone_lengths'],
                params['rotation'], params['trans'] / 1000,
                params['scale'] / 1000,
                params.get('chest_deformer', torch.zeros(1, 1, device=device)),
            )
            vertices = V[0].detach()

            # Load UV renderer for faces
            from .uv_renderer import create_uv_renderer
            uv_renderer = create_uv_renderer(uv_size=256, model_dir='mouse_model/mouse_txt')
            faces = uv_renderer.faces_vert_th

            # Setup texture sampler
            from .texture_sampler import TextureSampler
            sampler = TextureSampler(uv_size=256, device=str(device))
            cam_list = [cams[i] for i in views_to_use]
            sampler.set_cameras(cam_list)

            # Render each view - create [Original | Rendered] pairs
            view_pairs = []
            colors_np = vertex_colors.cpu().numpy()
            faces_np = faces.cpu().numpy()

            # Photometric metrics collectors
            psnr_scores = []
            psnr_dbs = []
            ssim_scores = []

            for view_idx, view_id in enumerate(views_to_use):
                # Load original image (BGR)
                video_path = os.path.join(data_dir, 'videos_undist', f'{view_id}.mp4')
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, image_bgr = cap.read()
                cap.release()

                if not ret:
                    logger.warning(f"Failed to read frame from view {view_id}")
                    continue

                H, W = image_bgr.shape[:2]

                # Original image with label
                original = image_bgr.copy()
                cv2.putText(original, f'View {view_id} - Original', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Rendered image on white background (better visibility for dark mouse fur)
                rendered = np.ones_like(image_bgr) * 255

                # Project vertices
                proj_2d = sampler.project_vertices(vertices, view_idx=view_idx).cpu().numpy()

                # Draw textured triangles on black background (sample every Nth for speed)
                face_sampling = self.config.projection_face_sampling
                for face in faces_np[::face_sampling]:
                    pts = proj_2d[face].astype(np.int32)
                    face_colors = colors_np[face]

                    # Skip out-of-bounds faces
                    if (pts[:, 0] < 0).any() or (pts[:, 0] >= W).any() or \
                       (pts[:, 1] < 0).any() or (pts[:, 1] >= H).any():
                        continue

                    # Average color (RGB -> BGR for OpenCV)
                    avg_color_rgb = face_colors.mean(axis=0) * 255
                    avg_color_bgr = tuple(int(c) for c in avg_color_rgb[::-1])
                    cv2.fillPoly(rendered, [pts], avg_color_bgr)

                # ===== Compute photometric metrics BEFORE adding text labels =====
                # Create mask for mesh region (non-white pixels)
                mesh_mask = create_mesh_mask(rendered, background_value=255)

                # Compute PSNR in masked region
                psnr_score, psnr_db = compute_psnr_masked(rendered, image_bgr, mesh_mask)
                psnr_scores.append(psnr_score)
                psnr_dbs.append(psnr_db)

                # Compute SSIM in masked region
                ssim_score, _ = compute_ssim_masked(rendered, image_bgr, mesh_mask)
                ssim_scores.append(ssim_score)

                # Add label to rendered (black text on white background)
                cv2.putText(rendered, f'View {view_id} - Rendered', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Create side-by-side pair [Original | Rendered]
                pair = np.hstack([original, rendered])
                view_pairs.append(pair)

            if len(view_pairs) < 6:
                logger.warning(f"Only {len(view_pairs)} views rendered")
                # Pad with black images if needed
                while len(view_pairs) < 6:
                    view_pairs.append(np.zeros_like(view_pairs[0]))

            # Create 2x3 grid of pairs
            row1 = np.hstack(view_pairs[:3])
            row2 = np.hstack(view_pairs[3:6])
            grid = np.vstack([row1, row2])

            # Resize if too large
            max_width = self.config.projection_max_width
            if grid.shape[1] > max_width:
                scale = max_width / grid.shape[1]
                grid = cv2.resize(grid, None, fx=scale, fy=scale)

            # Save
            grid_path = os.path.join(output_dir, 'projection_6view_grid.png')
            cv2.imwrite(grid_path, grid)

            # Compute mean metrics
            mean_psnr_score = np.mean(psnr_scores) if psnr_scores else 0.0
            mean_ssim_score = np.mean(ssim_scores) if ssim_scores else 0.0
            mean_psnr_db = np.mean(psnr_dbs) if psnr_dbs else 0.0

            logger.info(f"6-view projection: PSNR={mean_psnr_db:.1f}dB (score={mean_psnr_score:.3f}), SSIM={mean_ssim_score:.3f}")
            logger.info(f"6-view projection grid saved: {grid_path}")

            return {
                'grid_path': grid_path,
                'mean_psnr_score': mean_psnr_score,
                'mean_ssim_score': mean_ssim_score,
                'mean_psnr_db': mean_psnr_db,
                'per_view_psnr': psnr_scores,
                'per_view_ssim': ssim_scores,
            }

        except Exception as e:
            logger.warning(f"Failed to render 6-view projection grid: {e}")
            import traceback
            traceback.print_exc()
            return {
                'grid_path': None,
                'mean_psnr_score': 0.0,
                'mean_ssim_score': 0.0,
                'mean_psnr_db': 0.0,
                'per_view_psnr': [],
                'per_view_ssim': [],
            }


def run_wandb_sweep(
    fitting_result_dir: str,
    output_dir: str = "results/sweep",
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

    parser = argparse.ArgumentParser(
        description='WandB Sweep UV Map Optimization (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 실행 (권장: uv_size 고정, full stage)
  python -m uvmap.wandb_sweep --result_dir results/fitting/xxx --count 30

  # 2-Stage Optimization
  # Stage A: 구조 파라미터 최적화 (빠른 탐색)
  python -m uvmap.wandb_sweep --result_dir results/fitting/xxx \\
      --stage stage_a --count 20

  # Stage B: Stage A 결과 기반 미세 조정
  python -m uvmap.wandb_sweep --result_dir results/fitting/xxx \\
      --stage stage_b --stage_a_config results/sweep/best_config.json --count 20

  # uv_size 탐색 포함 (기존 방식)
  python -m uvmap.wandb_sweep --result_dir results/fitting/xxx \\
      --no_fix_uv_size --count 50
        """)

    parser.add_argument('--result_dir', type=str, required=True,
                       help='Fitting result directory')
    parser.add_argument('--output_dir', type=str, default='results/sweep',
                       help='Output directory')
    parser.add_argument('--project', type=str, default='uvmap-optimization',
                       help='WandB project name')
    parser.add_argument('--count', type=int, default=30,
                       help='Number of trials')
    parser.add_argument('--sweep_id', type=str, default=None,
                       help='Existing sweep ID (to join as agent)')
    parser.add_argument('--max_frames', type=int, default=20,
                       help='Max frames for evaluation (0=all, default=20 for fast search)')
    parser.add_argument('--frame_sampling', type=str, default='uniform',
                       choices=['uniform', 'random', 'keyframes'],
                       help='Frame sampling method')
    parser.add_argument('--create_only', action='store_true',
                       help='Only create sweep (no agent). Use with other servers via --sweep_id')

    # ===== v2 새로운 옵션들 =====
    parser.add_argument('--stage', type=str, default='full',
                       choices=['stage_a', 'stage_b', 'full'],
                       help='Optimization stage (stage_a: structure, stage_b: refinement, full: all)')
    parser.add_argument('--no_fix_uv_size', action='store_true',
                       help='Do NOT fix uv_size (search over [256,512,1024]). Default: fixed to 512')
    parser.add_argument('--uv_size', type=int, default=512,
                       help='Fixed uv_size value (default: 512)')
    parser.add_argument('--stage_a_config', type=str, default=None,
                       help='Path to stage_a best_config.json (for stage_b)')

    args = parser.parse_args()

    # Stage B 설정 로드
    stage_a_best_config = None
    if args.stage == 'stage_b' and args.stage_a_config:
        with open(args.stage_a_config) as f:
            stage_a_info = json.load(f)
            stage_a_best_config = stage_a_info.get('best_params', {})
            print(f"Loaded Stage A config: {stage_a_best_config}")

    config = WandBSweepConfig(
        project=args.project,
        output_dir=args.output_dir,
        count=args.count,
        max_frames=args.max_frames,
        frame_sampling=args.frame_sampling,
        # v2 옵션
        fix_uv_size=not args.no_fix_uv_size,
        fixed_uv_size=args.uv_size,
        optimization_stage=args.stage,
    )

    optimizer = WandBSweepOptimizer(config, stage_a_best_config=stage_a_best_config)

    # 실행 모드 출력
    print(f"\n{'='*60}")
    print(f"UV Map HPO v2 - {args.stage.upper()} Stage")
    print(f"{'='*60}")
    print(f"uv_size: {'fixed=' + str(args.uv_size) if not args.no_fix_uv_size else 'search [256,512,1024]'}")
    print(f"Parameters: {list(optimizer.param_space.keys())}")
    print(f"{'='*60}\n")

    if args.create_only:
        # Only create sweep, don't run agent
        sweep_id = optimizer.create_sweep()
        print(f"\n{'='*60}")
        print(f"Sweep created: {sweep_id}")
        print(f"Dashboard: https://wandb.ai/{config.entity or 'YOUR_ENTITY'}/{config.project}/sweeps/{sweep_id}")
        print(f"\nTo run agents on other servers:")
        print(f"  python -m uvmap.wandb_sweep \\")
        print(f"      --result_dir {args.result_dir} \\")
        print(f"      --sweep_id {sweep_id} \\")
        print(f"      --count 10")
        print(f"{'='*60}")
    elif args.sweep_id:
        # Join existing sweep as agent
        optimizer.run_agent(args.sweep_id, args.result_dir, count=args.count)
    else:
        # Create and run new sweep
        best_params = optimizer.run_sweep(args.result_dir)
        print(f"\nBest parameters: {best_params}")

        # 다음 단계 가이드 출력
        if args.stage == 'stage_a':
            print(f"\n{'='*60}")
            print("Stage A 완료! 다음 단계:")
            print(f"  python -m uvmap.wandb_sweep \\")
            print(f"      --result_dir {args.result_dir} \\")
            print(f"      --stage stage_b \\")
            print(f"      --stage_a_config {args.output_dir}/best_config.json \\")
            print(f"      --count 20")
            print(f"{'='*60}")
