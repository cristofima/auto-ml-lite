"""
Anomaly Detection Results Report Generator.
"""

from typing import Dict, Any
import numpy as np
from datetime import datetime, timezone

def generate_anomaly_report(
    output_path: str,
    job_id: str,
    metrics: Dict[str, Any],
    dataset_info: Dict[str, Any]
) -> None:
    """Generate anomaly detection results report."""
    print("Generating anomaly report...")
    try:
        report = AnomalyReportGenerator(
            job_id=job_id,
            metrics=metrics,
            dataset_info=dataset_info
        )
        html = report.generate()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Anomaly report saved to: {output_path}")
    except Exception as e:
        import traceback
        print(f"Error generating anomaly report: {str(e)}")
        traceback.print_exc()

class AnomalyReportGenerator:
    """Generate anomaly detection results report with premium CSS-only visuals."""
    
    def __init__(self, job_id: str, metrics: Dict[str, Any], dataset_info: Dict[str, Any]):
        self.job_id = job_id
        self.metrics = metrics
        self.dataset_info = dataset_info

    def _get_css(self) -> str:
        return """
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                margin: 0; padding: 20px; background: #f5f7fa; color: #333; line-height: 1.5;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #d32f2f; border-bottom: 3px solid #d32f2f; padding-bottom: 10px; margin-bottom: 30px; }
            h2 { color: #333; margin-top: 30px; border-left: 4px solid #d32f2f; padding-left: 10px; }
            h3 { color: #555; margin-top: 20px; }
            
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
            .stat-box.red { background: linear-gradient(135deg, #d32f2f 0%, #ff5252 100%); box-shadow: 0 4px 15px rgba(211, 47, 47, 0.3); }
            .stat-box.blue { background: linear-gradient(135deg, #1976d2 0%, #448aff 100%); box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3); }
            .stat-box.info { background: linear-gradient(135deg, #455a64 0%, #90a4ae 100%); box-shadow: 0 4px 15px rgba(69, 90, 100, 0.3); }
            
            .stat-number { font-size: 2.2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
            
            table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; font-weight: 600; color: #555; }
            tr:hover { background: #fffde7; }
            
            .score-badge { 
                padding: 2px 8px; border-radius: 4px; font-weight: bold; font-family: monospace;
            }
            
            .summary-item { margin-bottom: 10px; display: flex; justify-content: space-between; }
            .summary-label { font-weight: 500; color: #666; }
            .summary-value { font-family: monospace; font-weight: bold; }
        </style>
        """

    def _generate_samples_table(self) -> str:
        samples = self.metrics.get('top_anomalies_samples', [])
        scores = self.metrics.get('top_anomalies_scores', [])
        
        if not samples:
            return "<p>No sample data available.</p>"
            
        # Get column names
        cols = list(samples[0].keys())
        
        html = '<table><thead><tr><th>Score</th>'
        for col in cols[:6]: # Limit columns for readability
            html += f'<th>{col}</th>'
        html += '</tr></thead><tbody>'
        
        for i, sample in enumerate(samples):
            score = scores[i]
            # Use color intensity for score
            # Higher negative score = more anomalous. typically ranges from -0.5 to 0
            # Normalize for color: score -0.5 (very red) to 0 (less red)
            intensity = min(1.0, abs(score) * 2)
            bg = f'rgba(211, 47, 47, {0.1 + intensity * 0.4})'
            
            html += f'<tr style="background: {bg};">'
            html += f'<td><span class="score-badge">{score:.4f}</span></td>'
            for col in cols[:6]:
                val = sample[col]
                val_str = f"{val:.4f}" if isinstance(val, (float, np.float64)) else str(val)
                html += f'<td>{val_str}</td>'
            html += '</tr>'
            
        html += '</tbody></table>'
        if len(cols) > 6:
            html += f'<p style="font-size: 0.8em; color: #888;">* Showing first 6 of {len(cols)} columns.</p>'
        return html

    def generate(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Anomaly Report</title>{self._get_css()}</head>
        <body><div class="container">
            <h1>🚨 Automated Anomaly Report</h1>
            
            <div class="card">
                <div class="grid">
                    <div class="stat-box red">
                        <div class="stat-number">{self.metrics.get('anomaly_count')}</div>
                        <div class="stat-label">Anomalies Detected</div>
                    </div>
                    <div class="stat-box blue">
                        <div class="stat-number">{self.metrics.get('anomaly_percentage', 0):.2f}%</div>
                        <div class="stat-label">Contamination Rate</div>
                    </div>
                    <div class="stat-box info">
                        <div class="stat-number">{self.dataset_info.get('rows')}</div>
                        <div class="stat-label">Total Rows</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="grid-2">
                    <div>
                        <h2>⚡ Training Statistics</h2>
                        <div class="summary-item"><span class="summary-label">Algorithm</span><span class="summary-value">Isolation Forest</span></div>
                        <div class="summary-item"><span class="summary-label">Execution Time</span><span class="summary-value">{self.metrics.get('execution_time', 0):.2f}s</span></div>
                        <div class="summary-item"><span class="summary-label">Features Analyzed</span><span class="summary-value">{self.dataset_info.get('features')}</span></div>
                    </div>
                    <div>
                        <h2>📊 Score Profile</h2>
                        <div class="summary-item"><span class="summary-label">Mean Score</span><span class="summary-value">{self.metrics.get('mean_anomaly_score', 0):.4f}</span></div>
                        <div class="summary-item"><span class="summary-label">Min Score (Top Anomaly)</span><span class="summary-value" style="color: #d32f2f;">{self.metrics.get('min_anomaly_score', 0):.4f}</span></div>
                        <div class="summary-item"><span class="summary-label">Max Score (Normal)</span><span class="summary-value">{self.metrics.get('max_anomaly_score', 0):.4f}</span></div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>🔍 Top Anomalies Sample</h2>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 20px;">
                    Detailed profile of the rows with the highest anomaly scores. Lower scores represent higher deviations.
                </p>
                {self._generate_samples_table()}
            </div>
            
            <div class="card" style="text-align: center; color: #999; font-size: 0.9em;">
                <p>Generated by <strong>auto-ml-lite</strong> &bull; {timestamp}</p>
                <p>Job ID: <code>{self.job_id}</code></p>
            </div>
        </div></body></html>
        """
