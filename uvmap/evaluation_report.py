"""
UV Map Evaluation Report Generator

Generate comprehensive HTML report with:
- Quantitative metrics comparison table
- Visual comparison grid
- Per-experiment detail pages
- Interactive parameter sensitivity plots
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import base64


@dataclass
class ReportConfig:
    """Report generation configuration."""
    title: str = "UV Map Ablation Study"
    output_path: str = "report.html"
    include_images: bool = True
    max_images_per_row: int = 4


class HTMLReportGenerator:
    """
    Generate HTML report for UV map experiments.

    Features:
    - Summary statistics table
    - Sortable metrics comparison
    - Side-by-side texture comparison
    - Parameter sensitivity analysis
    """

    def __init__(self, config: ReportConfig):
        self.config = config

    def generate_report(
        self,
        summary_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate HTML report from experiment summary.

        Args:
            summary_path: Path to summary.json
            output_path: Output HTML path (optional)

        Returns:
            html: Generated HTML string
        """
        # Load summary
        with open(summary_path, 'r') as f:
            experiments = json.load(f)

        if output_path is None:
            output_path = self.config.output_path

        # Generate HTML sections
        html_parts = [
            self._generate_header(),
            self._generate_summary_section(experiments),
            self._generate_metrics_table(experiments),
            self._generate_visual_comparison(experiments),
            self._generate_sensitivity_analysis(experiments),
            self._generate_best_config_section(experiments),
            self._generate_footer(),
        ]

        html = '\n'.join(html_parts)

        # Save
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Report saved: {output_path}")
        return html

    def _generate_header(self) -> str:
        """Generate HTML header with styles."""
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
        }}
        .summary-cards {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin: 20px 0;
        }}
        .card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            min-width: 150px;
            text-align: center;
        }}
        .card-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .card-label {{
            font-size: 12px;
            opacity: 0.8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            cursor: pointer;
        }}
        th:hover {{
            background: #e9ecef;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .good {{ color: #28a745; font-weight: bold; }}
        .bad {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .image-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: white;
        }}
        .image-card img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: #333;
        }}
        .image-card .caption {{
            padding: 10px;
            font-size: 12px;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .best {{
            background: #d4edda;
            border: 2px solid #28a745;
        }}
        .chart {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .bar {{
            height: 20px;
            background: linear-gradient(90deg, #4CAF50, #81C784);
            border-radius: 4px;
            margin: 5px 0;
            transition: width 0.3s;
        }}
        .config-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 13px;
            margin: 10px 0;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>🎨 {self.config.title}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
'''

    def _generate_summary_section(self, experiments: List[Dict]) -> str:
        """Generate summary statistics cards."""
        n_exp = len(experiments)

        # Calculate stats
        coverages = [e['metrics']['coverage'] for e in experiments]
        confidences = [e['metrics']['mean_confidence'] for e in experiments]
        runtimes = [e['metrics']['runtime_seconds'] for e in experiments]

        best_coverage = max(coverages) if coverages else 0
        best_confidence = max(confidences) if confidences else 0
        avg_runtime = np.mean(runtimes) if runtimes else 0

        return f'''
    <h2>📊 Summary</h2>
    <div class="summary-cards">
        <div class="card">
            <div class="card-value">{n_exp}</div>
            <div class="card-label">Experiments</div>
        </div>
        <div class="card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <div class="card-value">{best_coverage:.1f}%</div>
            <div class="card-label">Best Coverage</div>
        </div>
        <div class="card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="card-value">{best_confidence:.3f}</div>
            <div class="card-label">Best Confidence</div>
        </div>
        <div class="card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="card-value">{avg_runtime:.1f}s</div>
            <div class="card-label">Avg Runtime</div>
        </div>
    </div>
'''

    def _generate_metrics_table(self, experiments: List[Dict]) -> str:
        """Generate sortable metrics comparison table."""
        rows = []

        # Sort by coverage (descending)
        sorted_exp = sorted(experiments, key=lambda x: x['metrics']['coverage'], reverse=True)

        for i, exp in enumerate(sorted_exp):
            cfg = exp['config']
            m = exp['metrics']

            # Highlight best
            row_class = 'best' if i == 0 else ''

            # Format coverage with color
            cov = m['coverage']
            cov_class = 'good' if cov > 80 else ('bad' if cov < 50 else 'neutral')

            rows.append(f'''
            <tr class="{row_class}">
                <td>{i+1}</td>
                <td>{exp['name'][:30]}...</td>
                <td>{cfg.get('visibility_threshold', 'N/A')}</td>
                <td>{cfg.get('uv_size', 'N/A')}</td>
                <td>{cfg.get('fusion_method', 'N/A')}</td>
                <td>{cfg.get('w_tv', 'N/A')}</td>
                <td class="{cov_class}">{cov:.1f}%</td>
                <td>{m['mean_confidence']:.3f}</td>
                <td>{m.get('seam_discontinuity', 0):.4f}</td>
                <td>{m['runtime_seconds']:.1f}s</td>
            </tr>
            ''')

        return f'''
    <h2>📋 Metrics Comparison</h2>
    <p>Click column headers to sort. <span class="highlight">Best result highlighted.</span></p>
    <table id="metrics-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Experiment</th>
                <th>Vis. Thresh</th>
                <th>UV Size</th>
                <th>Fusion</th>
                <th>TV Weight</th>
                <th>Coverage ↓</th>
                <th>Confidence</th>
                <th>Seam Disc.</th>
                <th>Runtime</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
'''

    def _generate_visual_comparison(self, experiments: List[Dict]) -> str:
        """Generate visual comparison grid."""
        if not self.config.include_images:
            return ''

        cards = []
        for exp in experiments[:12]:  # Limit to 12 images
            img_path = exp.get('texture_path', '')
            if os.path.exists(img_path):
                # Embed image as base64
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                img_src = f'data:image/png;base64,{img_data}'
            else:
                img_src = ''
                continue

            m = exp['metrics']
            cfg = exp['config']

            cards.append(f'''
            <div class="image-card">
                <img src="{img_src}" alt="{exp['name']}">
                <div class="caption">
                    <strong>{exp['name'][:25]}...</strong><br>
                    Coverage: {m['coverage']:.1f}% | Conf: {m['mean_confidence']:.3f}<br>
                    UV: {cfg.get('uv_size')} | Vis: {cfg.get('visibility_threshold')}
                </div>
            </div>
            ''')

        return f'''
    <h2>🖼️ Visual Comparison</h2>
    <div class="image-grid">
        {''.join(cards)}
    </div>
'''

    def _generate_sensitivity_analysis(self, experiments: List[Dict]) -> str:
        """Generate parameter sensitivity analysis."""
        # Group by parameters
        param_effects = {}

        for param in ['visibility_threshold', 'uv_size', 'fusion_method', 'w_tv']:
            param_effects[param] = {}

            for exp in experiments:
                cfg = exp['config']
                val = str(cfg.get(param, 'N/A'))
                coverage = exp['metrics']['coverage']

                if val not in param_effects[param]:
                    param_effects[param][val] = []
                param_effects[param][val].append(coverage)

        # Generate bar charts
        charts = []
        for param, values in param_effects.items():
            if len(values) <= 1:
                continue

            bars = []
            max_cov = max(max(v) for v in values.values()) if values else 100

            for val, coverages in sorted(values.items()):
                avg_cov = np.mean(coverages)
                width = (avg_cov / max_cov) * 100

                bars.append(f'''
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <span style="width: 100px; font-size: 13px;">{val}</span>
                    <div class="bar" style="width: {width}%;"></div>
                    <span style="margin-left: 10px; font-size: 13px;">{avg_cov:.1f}%</span>
                </div>
                ''')

            charts.append(f'''
            <div class="chart">
                <h4 style="margin-top: 0;">Effect of {param}</h4>
                {''.join(bars)}
            </div>
            ''')

        return f'''
    <h2>📈 Parameter Sensitivity</h2>
    <p>Average coverage by parameter value:</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        {''.join(charts)}
    </div>
'''

    def _generate_best_config_section(self, experiments: List[Dict]) -> str:
        """Generate recommended configuration section."""
        if not experiments:
            return ''

        # Find best by coverage
        best = max(experiments, key=lambda x: x['metrics']['coverage'])
        cfg = best['config']

        config_str = json.dumps(cfg, indent=2)

        return f'''
    <h2>🏆 Recommended Configuration</h2>
    <p>Best performing configuration based on coverage:</p>
    <div class="config-box">
        <pre>{config_str}</pre>
    </div>
    <p>
        <strong>Results:</strong>
        Coverage: <span class="good">{best['metrics']['coverage']:.1f}%</span> |
        Confidence: {best['metrics']['mean_confidence']:.3f} |
        Runtime: {best['metrics']['runtime_seconds']:.1f}s
    </p>
'''

    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return '''
</div>
<script>
// Simple table sorting
document.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
        const table = th.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const idx = Array.from(th.parentNode.children).indexOf(th);
        const asc = th.dataset.sort !== 'asc';

        rows.sort((a, b) => {
            const aVal = a.children[idx].textContent;
            const bVal = b.children[idx].textContent;
            const aNum = parseFloat(aVal) || aVal;
            const bNum = parseFloat(bVal) || bVal;
            return asc ? (aNum > bNum ? 1 : -1) : (aNum < bNum ? 1 : -1);
        });

        th.dataset.sort = asc ? 'asc' : 'desc';
        rows.forEach(row => tbody.appendChild(row));
    });
});
</script>
</body>
</html>
'''


def generate_evaluation_report(
    experiment_dir: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate evaluation report from experiment directory.

    Args:
        experiment_dir: Directory containing experiment results
        output_path: Output HTML path

    Returns:
        html: Generated HTML
    """
    summary_path = os.path.join(experiment_dir, 'summary.json')

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    if output_path is None:
        output_path = os.path.join(experiment_dir, 'report.html')

    config = ReportConfig(
        title="UV Map Ablation Study Results",
        output_path=output_path,
    )

    generator = HTMLReportGenerator(config)
    return generator.generate_report(summary_path, output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate UV Map Evaluation Report')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment results directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML path')

    args = parser.parse_args()

    generate_evaluation_report(args.experiment_dir, args.output)
