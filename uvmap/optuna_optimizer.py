"""
Optuna-based UV Map Parameter Optimization

Systematic hyperparameter search using Bayesian optimization.
Supports single-objective and multi-objective optimization.

References:
- Optuna TPE: https://optuna.readthedocs.io/
- RLGS (Gaussian Splatting HPO): https://arxiv.org/html/2508.04078v1
- Image Quality Metrics: https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import optuna
    from optuna.trial import Trial
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_pareto_front,
        plot_slice,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")


@dataclass
class OptimizationConfig:
    """Configuration for Optuna optimization."""
    # Study settings
    study_name: str = "uvmap_optimization"
    n_trials: int = 50
    timeout: Optional[float] = None  # seconds

    # Optimization mode
    mode: str = "single"  # "single" or "multi"
    direction: str = "maximize"  # for single objective

    # Parameter search space
    param_space: Dict = None

    # Evaluation settings
    max_frames: int = 20  # Limit frames for faster evaluation

    # Output
    output_dir: str = "optuna_results"

    def __post_init__(self):
        if self.param_space is None:
            # Default search space based on literature review
            self.param_space = {
                "visibility_threshold": {
                    "type": "float",
                    "low": 0.1,
                    "high": 0.7,
                    "log": False,
                },
                "uv_size": {
                    "type": "categorical",
                    "choices": [256, 512, 1024],
                },
                "w_tv": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 1e-1,
                    "log": True,  # Log scale for regularization weights
                },
                "fusion_method": {
                    "type": "categorical",
                    "choices": ["average", "visibility_weighted", "max_visibility"],
                },
                "opt_lr": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-1,
                    "log": True,
                },
                "use_mask": {
                    "type": "categorical",
                    "choices": [True, False],
                },
            }


class UVMapObjective:
    """
    Objective function for Optuna optimization.

    Evaluation Metrics (based on literature):
    - Coverage: UV space utilization (maximize)
    - Photometric: PSNR/SSIM between rendered and GT (maximize)
    - Seam Quality: Color discontinuity at UV boundaries (minimize)
    - Color Consistency: Cross-view variance (minimize)

    References:
    - Seamless Texture Optimization (CGF 2024)
    - Image Quality Assessment (PMC7817470)
    """

    def __init__(
        self,
        result_dir: str,
        config: OptimizationConfig,
        device: str = "cuda",
    ):
        self.result_dir = result_dir
        self.config = config
        self.device = device

        # Lazy imports to avoid circular dependency
        self._pipeline = None
        self._evaluator = None

    def _get_pipeline(self):
        """Lazy load pipeline."""
        if self._pipeline is None:
            from .uv_pipeline import UVMapPipeline, UVPipelineConfig
            self._pipeline_config = UVPipelineConfig(
                result_dir=self.result_dir,
                frame_interval=max(1, 100 // self.config.max_frames),
            )
        return self._pipeline_config

    def _get_evaluator(self):
        """Lazy load evaluator."""
        if self._evaluator is None:
            from .experiment_runner import UVMapEvaluator
            self._evaluator = UVMapEvaluator(self.device)
        return self._evaluator

    def sample_params(self, trial: Trial) -> Dict:
        """
        Sample hyperparameters from search space.

        Uses Optuna's suggest_* methods for efficient sampling.
        """
        params = {}

        for name, spec in self.config.param_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    spec["choices"],
                )

        return params

    def evaluate(self, params: Dict) -> Dict[str, float]:
        """
        Evaluate UV map quality with given parameters.

        Returns dict of metrics:
        - coverage: [0, 100] higher is better
        - psnr: [0, inf] higher is better
        - seam_score: [0, 1] lower is better
        - consistency: [0, 1] lower is better
        """
        from .uv_pipeline import UVMapPipeline, UVPipelineConfig

        # Create pipeline with trial params
        pipeline_config = UVPipelineConfig(
            result_dir=self.result_dir,
            uv_size=params.get("uv_size", 512),
            visibility_threshold=params.get("visibility_threshold", 0.3),
            do_optimization=params.get("do_optimization", False),
            opt_lr=params.get("opt_lr", 0.01),
            opt_w_tv=params.get("w_tv", 1e-3),
            frame_interval=max(1, 100 // self.config.max_frames),
            save_intermediate=False,
        )

        # Temporarily redirect output
        import tempfile
        pipeline_config.output_dir = tempfile.mkdtemp()

        try:
            # Run pipeline
            pipeline = UVMapPipeline(pipeline_config, device=self.device)
            pipeline.setup()
            pipeline.frames = pipeline.frames[:self.config.max_frames]

            # Set fusion method
            if hasattr(pipeline.texture_sampler, 'fusion_method'):
                pipeline.texture_sampler.fusion_method = params.get(
                    "fusion_method", "visibility_weighted"
                )

            texture = pipeline.run()

            # Evaluate
            evaluator = self._get_evaluator()
            vertex_colors, confidence = pipeline.texture_accumulator.get_texture()
            uv_mask = pipeline.uv_renderer.get_uv_mask()

            coverage = evaluator.compute_coverage(texture, uv_mask)
            seam_score = evaluator.compute_seam_discontinuity(texture)
            conf_stats = evaluator.compute_confidence_stats(confidence)

            # Estimate PSNR (simplified - would need actual rendering for full eval)
            # Use confidence as proxy for now
            psnr_proxy = conf_stats['mean'] * 30  # Scale to PSNR-like range

            metrics = {
                "coverage": coverage,
                "psnr": psnr_proxy,
                "seam_score": seam_score,
                "consistency": 1.0 - conf_stats['mean'],  # Inverse of confidence
            }

        except Exception as e:
            print(f"Evaluation failed: {e}")
            metrics = {
                "coverage": 0.0,
                "psnr": 0.0,
                "seam_score": 1.0,
                "consistency": 1.0,
            }

        finally:
            # Cleanup temp dir
            import shutil
            if os.path.exists(pipeline_config.output_dir):
                shutil.rmtree(pipeline_config.output_dir, ignore_errors=True)

        return metrics

    def compute_single_objective(self, metrics: Dict[str, float]) -> float:
        """
        Compute single scalar objective from metrics.

        Weighted combination based on literature recommendations:
        - Coverage most important (40%)
        - Photometric quality (30%)
        - Seam quality (30%)
        """
        # Normalize metrics to [0, 1]
        coverage_norm = metrics["coverage"] / 100.0
        psnr_norm = min(metrics["psnr"] / 40.0, 1.0)  # Cap at 40 dB
        seam_norm = 1.0 - min(metrics["seam_score"], 1.0)  # Invert (lower is better)

        # Weighted sum
        score = (
            0.4 * coverage_norm +
            0.3 * psnr_norm +
            0.3 * seam_norm
        )

        return score

    def __call__(self, trial: Trial) -> float:
        """
        Optuna objective function (single-objective).
        """
        params = self.sample_params(trial)
        metrics = self.evaluate(params)

        # Store metrics as trial attributes for analysis
        for name, value in metrics.items():
            trial.set_user_attr(name, value)

        return self.compute_single_objective(metrics)


class MultiObjectiveUVMapObjective(UVMapObjective):
    """
    Multi-objective optimization for UV mapping.

    Returns Pareto front of solutions trading off:
    1. Coverage (maximize)
    2. PSNR (maximize)
    3. Seam quality (minimize -> maximize negative)
    """

    def __call__(self, trial: Trial) -> Tuple[float, float, float]:
        """
        Optuna objective function (multi-objective).

        Returns tuple of objectives to optimize.
        """
        params = self.sample_params(trial)
        metrics = self.evaluate(params)

        # Store for analysis
        for name, value in metrics.items():
            trial.set_user_attr(name, value)

        # Return objectives (all to maximize)
        return (
            metrics["coverage"],
            metrics["psnr"],
            -metrics["seam_score"],  # Negate to maximize
        )


class OptunaUVOptimizer:
    """
    Main class for Optuna-based UV map optimization.

    Usage:
        optimizer = OptunaUVOptimizer(result_dir, config)
        best_params = optimizer.optimize()
        optimizer.generate_report()
    """

    def __init__(
        self,
        result_dir: str,
        config: Optional[OptimizationConfig] = None,
        device: str = "cuda",
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required. Install with: pip install optuna")

        self.result_dir = result_dir
        self.config = config or OptimizationConfig()
        self.device = device

        self.study = None
        self.best_params = None

        os.makedirs(self.config.output_dir, exist_ok=True)

    def optimize(self) -> Dict:
        """
        Run Optuna optimization.

        Returns best parameters found.
        """
        print(f"\n{'='*60}")
        print(f"Optuna UV Map Optimization")
        print(f"Mode: {self.config.mode}")
        print(f"Trials: {self.config.n_trials}")
        print(f"{'='*60}\n")

        if self.config.mode == "single":
            return self._optimize_single()
        else:
            return self._optimize_multi()

    def _optimize_single(self) -> Dict:
        """Single-objective optimization."""
        objective = UVMapObjective(
            self.result_dir,
            self.config,
            self.device,
        )

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        print(f"\nBest score: {self.study.best_value:.4f}")
        print(f"Best params: {self.best_params}")

        return self.best_params

    def _optimize_multi(self) -> List[Dict]:
        """Multi-objective optimization with Pareto front."""
        objective = MultiObjectiveUVMapObjective(
            self.result_dir,
            self.config,
            self.device,
        )

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            directions=["maximize", "maximize", "maximize"],  # coverage, psnr, -seam
            sampler=optuna.samplers.NSGAIISampler(seed=42),
        )

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        # Get Pareto front
        pareto_trials = self.study.best_trials
        print(f"\nPareto front: {len(pareto_trials)} solutions")

        pareto_params = []
        for i, trial in enumerate(pareto_trials):
            print(f"  [{i}] Coverage={trial.values[0]:.1f}%, "
                  f"PSNR={trial.values[1]:.2f}, "
                  f"Seam={-trial.values[2]:.4f}")
            pareto_params.append(trial.params)

        self.best_params = pareto_params
        return pareto_params

    def generate_report(self) -> str:
        """
        Generate optimization report with visualizations.

        Returns path to HTML report.
        """
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        report_dir = self.config.output_dir

        # Save study results
        results_path = os.path.join(report_dir, "study_results.json")
        results = {
            "study_name": self.config.study_name,
            "n_trials": len(self.study.trials),
            "mode": self.config.mode,
            "best_params": self.best_params if isinstance(self.best_params, dict)
                          else [p for p in self.best_params],
            "trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "values": t.values if hasattr(t, 'values') else [t.value],
                    "user_attrs": t.user_attrs,
                }
                for t in self.study.trials
            ],
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate visualizations
        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(os.path.join(report_dir, "optimization_history.html"))

            # Parameter importance
            if self.config.mode == "single":
                fig = plot_param_importances(self.study)
                fig.write_html(os.path.join(report_dir, "param_importance.html"))

                # Slice plot
                fig = plot_slice(self.study)
                fig.write_html(os.path.join(report_dir, "slice_plot.html"))
            else:
                # Pareto front
                fig = plot_pareto_front(
                    self.study,
                    target_names=["Coverage", "PSNR", "Seam Quality"],
                )
                fig.write_html(os.path.join(report_dir, "pareto_front.html"))

        except Exception as e:
            print(f"Warning: Visualization failed: {e}")

        # Generate summary HTML
        html_path = os.path.join(report_dir, "optimization_report.html")
        self._generate_html_report(html_path, results)

        print(f"\nReport saved to: {report_dir}")
        return html_path

    def _generate_html_report(self, path: str, results: Dict) -> None:
        """Generate HTML summary report."""
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Optuna UV Map Optimization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{
            display: inline-block;
            padding: 15px;
            margin: 10px;
            background: #f0f0f0;
            border-radius: 8px;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        .best {{ background: #e8f5e9; }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>🎯 Optuna UV Map Optimization Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Summary</h2>
    <div class="metric">
        <div>Total Trials</div>
        <div class="metric-value">{results['n_trials']}</div>
    </div>
    <div class="metric">
        <div>Mode</div>
        <div class="metric-value">{results['mode']}</div>
    </div>

    <h2>Best Parameters</h2>
    <pre>{json.dumps(results['best_params'], indent=2)}</pre>

    <h2>Trial History</h2>
    <table>
        <tr>
            <th>#</th>
            <th>Coverage</th>
            <th>PSNR</th>
            <th>Seam</th>
            <th>Parameters</th>
        </tr>
'''

        for t in results['trials'][-20:]:  # Last 20 trials
            attrs = t.get('user_attrs', {})
            html += f'''
        <tr>
            <td>{t['number']}</td>
            <td>{attrs.get('coverage', 'N/A'):.1f}%</td>
            <td>{attrs.get('psnr', 'N/A'):.2f}</td>
            <td>{attrs.get('seam_score', 'N/A'):.4f}</td>
            <td><small>{json.dumps(t['params'])}</small></td>
        </tr>
'''

        html += '''
    </table>

    <h2>Visualizations</h2>
    <ul>
        <li><a href="optimization_history.html">Optimization History</a></li>
        <li><a href="param_importance.html">Parameter Importance</a></li>
        <li><a href="slice_plot.html">Parameter Slice Plot</a></li>
    </ul>
</body>
</html>
'''

        with open(path, 'w') as f:
            f.write(html)


def run_optuna_optimization(
    result_dir: str,
    n_trials: int = 30,
    mode: str = "single",
    output_dir: str = "optuna_uvmap",
) -> Dict:
    """
    Convenience function to run Optuna optimization.

    Args:
        result_dir: Path to fitting results
        n_trials: Number of optimization trials
        mode: "single" or "multi" objective
        output_dir: Output directory

    Returns:
        Best parameters found
    """
    config = OptimizationConfig(
        n_trials=n_trials,
        mode=mode,
        output_dir=output_dir,
    )

    optimizer = OptunaUVOptimizer(result_dir, config)
    best_params = optimizer.optimize()
    optimizer.generate_report()

    return best_params


# Qualitative Evaluation Guidelines (for human assessment)
QUALITATIVE_CRITERIA = """
## 정성적 평가 기준 (Human Assessment)

### 1. Seam Visibility (UV 경계 가시성)
- **Good**: 경계가 육안으로 보이지 않음
- **Acceptable**: 확대 시에만 경계 보임
- **Poor**: 일반 스케일에서 명확한 색상 점프

평가 방법: texture_final.png를 100%, 200%, 400%로 확대하며 검사

### 2. Ghosting Artifacts (고스팅)
- **Good**: 선명한 디테일, 이중 윤곽 없음
- **Acceptable**: 미세한 블러, 눈/코 등 디테일 영역에서만 관찰
- **Poor**: 명확한 이중 이미지 또는 심한 블러

평가 방법: 눈, 귀, 발가락 등 고주파 디테일 영역 검사

### 3. Color Consistency (색상 일관성)
- **Good**: 동일 신체 부위가 모든 뷰에서 동일 색상
- **Acceptable**: 약간의 색상 변화, 자연스러운 조명 차이로 설명 가능
- **Poor**: 동일 부위가 뷰마다 다른 색상 (patchy appearance)

평가 방법: 렌더링된 mesh를 다양한 각도에서 회전하며 확인

### 4. Coverage Completeness (커버리지)
- **Good**: UV 공간 90%+ 유효 텍스처
- **Acceptable**: 70-90%, 보이지 않는 영역만 누락
- **Poor**: 70% 미만, 가시 영역에 holes

평가 방법: confidence.png와 texture_final.png 비교

### 5. Detail Preservation (디테일 보존)
- **Good**: 털 패턴, 피부 텍스처 등 미세 디테일 유지
- **Acceptable**: 주요 특징 유지, 미세 디테일 약간 손실
- **Poor**: 전체적으로 평탄화됨 (over-smoothed)

평가 방법: 원본 이미지와 렌더링 결과 Side-by-side 비교
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna UV Map Optimization")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--output_dir", type=str, default="optuna_uvmap")

    args = parser.parse_args()

    print(QUALITATIVE_CRITERIA)

    best = run_optuna_optimization(
        result_dir=args.result_dir,
        n_trials=args.n_trials,
        mode=args.mode,
        output_dir=args.output_dir,
    )

    print(f"\nBest parameters: {best}")
