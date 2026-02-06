"""WandB Sweep configuration and parameter definitions.

Defines sweep search spaces, default configurations, and
stage-specific parameter ranges for UV texture optimization.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

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


