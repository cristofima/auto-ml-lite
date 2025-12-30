"""
Clustering Results Report Generator.
"""

from typing import Dict, Any, List
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
        </style>
        """

    def _generate_results_table(self) -> str:
        html = '<table><tr><th>Clusters (K)</th><th>Silhouette Score</th><th>Calinski-Harabasz</th><th>Quality</th></tr>'
        
        # Max scores for progress bars
        max_sil = max([r['silhouette'] for r in self.search_results]) if self.search_results else 1
        max_cal = max([r['calinski'] for r in self.search_results]) if self.search_results else 1
        
        for res in self.search_results:
            is_optimal = res['k'] == self.metrics.get('optimal_k')
            row_style = 'style="background: #f1f8e9; font-weight: bold;"' if is_optimal else ""
            optimal_badge = '<span class="badge" style="background:#28a745; color:white;">OPTIMAL</span>' if is_optimal else ""
            
            # Simplified quality assessment
            quality = "Good" if res['silhouette'] > 0.4 else "Moderate" if res['silhouette'] > 0.2 else "Poor"
            
            html += f"""
            <tr {row_style}>
                <td>{res['k']} {optimal_badge}</td>
                <td>
                    {res['silhouette']:.4f}
                    <div class="bar-outer"><div class="bar-inner" style="width: {res['silhouette']/max_sil*100}%"></div></div>
                </td>
                <td>{res['calinski']:.2f}</td>
                <td>{quality}</td>
            </tr>
            """
        html += '</table>'
        return html

    def generate(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
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
                        <div class="stat-number">{self.metrics.get('execution_time', 0):.1f}s</div>
                        <div class="stat-label">Execution Time</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 Selection Analysis</h2>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 20px;">
                    We evaluated multiple values of K to find the configuration that maximizes group internal consistency and separation.
                </p>
                {self._generate_results_table()}
            </div>
            
            <div class="card" style="text-align: center; color: #999; font-size: 0.9em;">
                <p>Generated by <strong>auto-ml-lite</strong> &bull; {timestamp}</p>
            </div>
        </div></body></html>
        """
