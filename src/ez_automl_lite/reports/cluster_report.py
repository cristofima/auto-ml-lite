"""
Clustering Results Report Generator.
"""

from typing import Dict, Any
from datetime import datetime, timezone

def generate_cluster_report(
    output_path: str,
    job_id: str,
    metrics: Dict[str, Any],
    dataset_info: Dict[str, Any]
) -> None:
    """Generate clustering results report."""
    print("Generating clustering report...")
    try:
        report = ClusterReportGenerator(
            job_id=job_id,
            metrics=metrics,
            dataset_info=dataset_info
        )
        html = report.generate()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Clustering report saved to: {output_path}")
    except Exception as e:
        print(f"Error generating clustering report: {str(e)}")

class ClusterReportGenerator:
    """Generate clustering results report with premium CSS-only visuals."""
    
    def __init__(self, job_id: str, metrics: Dict[str, Any], dataset_info: Dict[str, Any]):
        self.job_id = job_id
        self.metrics = metrics
        self.dataset_info = dataset_info
        self.search_results = metrics.get('search_results', [])

    def _get_css(self) -> str:
        return """
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                margin: 0; padding: 20px; background: #f5f7fa; color: #333; line-height: 1.5;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; margin-bottom: 30px; }
            h2 { color: #333; margin-top: 30px; border-left: 4px solid #1a73e8; padding-left: 10px; }
            
            .card {
                background: white; border-radius: 8px; padding: 25px;
                margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }
            
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
            
            .stat-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; border-radius: 12px; text-align: center;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            .stat-box.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3); }
            .stat-box.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3); }
            .stat-number { font-size: 2.5em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
            
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 14px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; font-weight: 600; color: #555; }
            
            .bar-outer { background: #eee; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 5px; }
            .bar-inner { background: #1a73e8; height: 100%; border-radius: 5px; }
            
            .badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 700; background: #e3f2fd; color: #1565c0; }
            
            /* Visualizations */
            .chart-container { position: relative; height: 350px; width: 100%; border: 1px solid #eee; background: white; border-radius: 8px; margin-top: 20px; }
            .chart-svg { width: 100%; height: 100%; }
            .scatter-pt { opacity: 0.7; transition: r 0.2s; }
            .scatter-pt:hover { r: 6; opacity: 1; stroke: #333; stroke-width: 1; }
            
            .cluster-0 { fill: #4facfe; }
            .cluster-1 { fill: #f093fb; }
            .cluster-2 { fill: #11998e; }
            .cluster-3 { fill: #f7971e; }
            .cluster-4 { fill: #667eea; }
            .cluster-5 { fill: #e55353; }
            .cluster-6 { fill: #321fdb; }
            .cluster-7 { fill: #f9b115; }
            
            .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; justify-content: center; }
            .legend-item { display: flex; align-items: center; font-size: 0.85em; color: #666; }
            .legend-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
            
            .chart-line { fill: none; stroke: #1a73e8; stroke-width: 2; }
            .chart-axis { stroke: #ddd; stroke-width: 1; }
            
            @media (max-width: 768px) {
                .grid-2 { grid-template-columns: 1fr; }
            }
        </style>
        """

    def _generate_results_table(self) -> str:
        html = '<table><tr><th>Clusters (K)</th><th>Silhouette Score</th><th>Calinski-Harabasz</th><th>Inertia</th><th>Quality</th></tr>'
        
        # Max scores for progress bars
        max_sil = max([r['silhouette'] for r in self.search_results]) if self.search_results else 1
        
        for res in self.search_results:
            is_optimal = res['k'] == self.metrics.get('optimal_k')
            row_style = 'style="background: #f1f8e9; font-weight: bold;"' if is_optimal else ""
            optimal_badge = '<span class="badge" style="background:#28a745; color:white;">OPTIMAL</span>' if is_optimal else ""
            
            # Simplified quality assessment
            quality = "Good" if res['silhouette'] > 0.5 else "Moderate" if res['silhouette'] > 0.25 else "Poor"
            inertia_val = f"{res.get('inertia', 0):.2f}" if res.get('inertia', 0) > 0 else "-"
            
            html += f"""
            <tr {row_style}>
                <td>{res['k']} {optimal_badge}</td>
                <td>
                    {res['silhouette']:.4f}
                    <div class="bar-outer"><div class="bar-inner" style="width: {max(0, res['silhouette'])/max_sil*100}%"></div></div>
                </td>
                <td>{res['calinski']:.2f}</td>
                <td>{inertia_val}</td>
                <td>{quality}</td>
            </tr>
            """
        html += '</table>'
        return html

    def _generate_pca_plot(self) -> str:
        """Generate PCA scatter plot SVG"""
        pdo = self.dataset_info.get('pca_data', [])
        if not pdo: return "<p>No PCA visualization data available.</p>"
        
        # Normalize coordinates to 0..100
        xs = [p['x'] for p in pdo]
        ys = [p['y'] for p in pdo]
        
        if not xs or not ys: return ""
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        
        # Padding
        padding = 10
        width = 100 - 2*padding
        height = 100 - 2*padding
        
        points_svg = ""
        clusters = set()
        
        for p in pdo:
            cx = padding + ((p['x'] - min_x) / range_x) * width
            cy = 100 - (padding + ((p['y'] - min_y) / range_y) * height) # Flip Y
            c_id = p['cluster']
            clusters.add(c_id)
            points_svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" class="scatter-pt cluster-{c_id % 8}" />'
            
        legend_html = '<div class="legend">'
        for c_id in sorted(clusters):
            legend_html += f'<div class="legend-item"><div class="legend-dot cluster-{c_id % 8}"></div>Cluster {c_id}</div>'
        legend_html += '</div>'
        
        return f"""
        <div class="chart-container">
            <svg class="chart-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                <rect x="0" y="0" width="100" height="100" fill="#fafafa" rx="4" />
                <!-- Axes -->
                <line x1="{padding}" y1="{100-padding}" x2="{100-padding}" y2="{100-padding}" class="chart-axis" />
                <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{100-padding}" class="chart-axis" />
                
                {points_svg}
            </svg>
            <div style="text-align:center; font-size:0.8em; color:#888; margin-top:5px;">PCA Projection (2D)</div>
        </div>
        {legend_html}
        """

    def generate(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        algo = self.metrics.get('algorithm', 'Unknown')
        
        return f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Clustering Report</title>{self._get_css()}</head>
        <body><div class="container">
            <h1>❄️ Automated Clustering Report</h1>
            
            <div class="card">
                <div class="grid">
                    <div class="stat-box green">
                        <div class="stat-number">{self.metrics.get('optimal_k')}</div>
                        <div class="stat-label">Optimal Clusters</div>
                    </div>
                    <div class="stat-box blue">
                        <div class="stat-number">{self.metrics.get('best_silhouette', 0):.4f}</div>
                        <div class="stat-label">Best Silhouette</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{algo}</div>
                        <div class="stat-label">Algorithm</div>
                    </div>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="card">
                    <h2>🗺️ Cluster Visualization (PCA)</h2>
                    <p style="color: #666; font-size: 0.9em;">
                        2D projection of the dataset showing cluster separation.
                    </p>
                    {self._generate_pca_plot()}
                </div>
                
                <div class="card">
                    <h2>📈 Selection Analysis</h2>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 20px;">
                        Evaluation of internal consistency (Silhouette) and separation (Calinski-Harabasz).
                    </p>
                    {self._generate_results_table()}
                </div>
            </div>
            
            <div class="card" style="text-align: center; color: #999; font-size: 0.9em;">
                <p>Generated by <strong>ez-automl-lite</strong> &bull; {timestamp}</p>
            </div>
        </div></body></html>
        """
