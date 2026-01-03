"""
Anomaly Detection Report Generator.
"""

from datetime import UTC, datetime
from typing import Any


def generate_anomaly_report(
    output_path: str, job_id: str, metrics: dict[str, Any], dataset_info: dict[str, Any]
) -> None:
    """Generate anomaly detection results report."""
    print("Generating anomaly report...")
    try:
        report = AnomalyReportGenerator(
            job_id=job_id, metrics=metrics, dataset_info=dataset_info
        )
        html = report.generate()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Anomaly report saved to: {output_path}")
    except Exception as e:
        print(f"Error generating anomaly report: {e!s}")


class AnomalyReportGenerator:
    """Generate anomaly detection results report with premium CSS-only visuals."""

    def __init__(
        self, job_id: str, metrics: dict[str, Any], dataset_info: dict[str, Any]
    ):
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
            h1 { color: #d93025; border-bottom: 3px solid #d93025; padding-bottom: 10px; margin-bottom: 30px; }
            h2 { color: #333; margin-top: 30px; border-left: 4px solid #d93025; padding-left: 10px; }

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
            .stat-box.red { background: linear-gradient(135deg, #ff5f6d 0%, #ffc371 100%); box-shadow: 0 4px 15px rgba(255, 95, 109, 0.3); }
            .stat-box.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3); }
            .stat-box.gray { background: linear-gradient(135deg, #606c88 0%, #3f4c6b 100%); box-shadow: 0 4px 15px rgba(96, 108, 136, 0.3); }

            .stat-number { font-size: 2.5em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }

            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; font-weight: 600; color: #555; }
            tr:hover { background: #fcfcfc; }

            /* Visualizations */
            .chart-container { position: relative; height: 350px; width: 100%; border: 1px solid #eee; background: white; border-radius: 8px; margin-top: 20px; }
            .chart-svg { width: 100%; height: 100%; }
            .scatter-pt { opacity: 0.7; transition: r 0.2s; }
            .scatter-pt:hover { r: 6; opacity: 1; stroke: #333; stroke-width: 1; }

            .pt-normal { fill: #1a73e8; opacity: 0.3; }
            .pt-anomaly { fill: #d93025; opacity: 0.8; r: 4; }

            .hist-chart { display: flex; align-items: flex-end; height: 150px; gap: 2px; margin: 20px 0; border-bottom: 2px solid #eee; padding-bottom: 5px; }
            .hist-bar { background: #1a73e8; min-width: 5px; flex: 1; opacity: 0.7; }
            .hist-bar:hover { opacity: 1; }

            .legend { display: flex; justify-content: center; gap: 20px; margin-top: 10px; }
            .legend-item { display: flex; align-items: center; font-size: 0.9em; color: #666; }
            .legend-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }

            @media (max-width: 768px) {
                .grid-2 { grid-template-columns: 1fr; }
            }
        </style>
        """

    def _generate_top_anomalies(self) -> str:
        samples = self.metrics.get("top_anomalies_samples", [])
        scores = self.metrics.get("top_anomalies_scores", [])

        if not samples:
            return "<p>No anomaly details available.</p>"

        # Determine columns to show (first 5 for brevity)
        if len(samples) > 0:
            cols = list(samples[0].keys())[:5]
        else:
            return ""

        html = (
            "<table><tr><th>Anomaly Score</th>"
            + "".join(f"<th>{c}</th>" for c in cols)
            + "</tr>"
        )

        for i, row in enumerate(samples):
            score = scores[i] if i < len(scores) else 0
            # Highlight high anomalies? In IsoForest, lower/negative is more anomalous.
            # We assume metrics passed raw decision_function. IsolationForest: < 0 is anomaly.

            html += (
                f'<tr><td style="color: #d93025; font-weight: bold;">{score:.4f}</td>'
            )
            for c in cols:
                val = row.get(c, "")
                if isinstance(val, float):
                    val = f"{val:.4f}"
                html += f"<td>{val}</td>"
            html += "</tr>"

        html += "</table>"
        html += '<p style="font-size:0.8em; color:#888;">Showing top 10 most anomalous samples (lowest scores).</p>'
        return html

    def _generate_pca_plot(self) -> str:
        """Generate PCA scatter plot SVG"""
        pdo = self.dataset_info.get("pca_data", [])
        if not pdo:
            return "<p>No visualization data available.</p>"

        # Normalize coordinates
        xs = [p["x"] for p in pdo]
        ys = [p["y"] for p in pdo]
        if not xs:
            return ""

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1

        padding = 10
        width = 100 - 2 * padding
        height = 100 - 2 * padding

        points_svg = ""
        # Draw normal first, then anomalies on top
        normal_pts = [p for p in pdo if p["label"] == 1]
        anomaly_pts = [p for p in pdo if p["label"] == -1]

        for p in normal_pts + anomaly_pts:
            cx = padding + ((p["x"] - min_x) / range_x) * width
            cy = 100 - (padding + ((p["y"] - min_y) / range_y) * height)
            cls = "pt-normal" if p["label"] == 1 else "pt-anomaly"
            points_svg += (
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" class="scatter-pt {cls}" />'
            )

        return f"""
        <div class="chart-container">
            <svg class="chart-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                 <rect x="0" y="0" width="100" height="100" fill="#fafafa" rx="4" />
                {points_svg}
            </svg>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background: #1a73e8; opacity:0.5;"></div>Normal</div>
                <div class="legend-item"><div class="legend-dot" style="background: #d93025;"></div>Anomaly</div>
            </div>
        </div>
        """

    def generate(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        algo = self.metrics.get("algorithm", "Unknown")

        return f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Anomaly Report</title>{self._get_css()}</head>
        <body><div class="container">
            <h1>🚨 Automated Anomaly Report</h1>

            <div class="card">
                <div class="grid">
                    <div class="stat-box red">
                        <div class="stat-number">{self.metrics.get('anomaly_count')}</div>
                        <div class="stat-label">Anomalies Found</div>
                    </div>
                    <div class="stat-box green">
                        <div class="stat-number">{self.metrics.get('anomaly_percentage', 0):.2f}%</div>
                        <div class="stat-label">Contamination Rate</div>
                    </div>
                    <div class="stat-box gray">
                        <div class="stat-number" style="font-size: 1.8em; margin-top: 10px;">{algo}</div>
                        <div class="stat-label">Algorithm</div>
                    </div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h2>🗺️ Anomaly Visualization</h2>
                    <p style="color: #666; font-size: 0.9em;">
                        2D projection highlighting anomalies (Red) vs normal data (Blue).
                    </p>
                    {self._generate_pca_plot()}
                </div>

                <div class="card">
                    <h2>📋 Top Anomalies</h2>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 20px;">
                        Samples with the highest anomaly scores.
                    </p>
                    {self._generate_top_anomalies()}
                </div>
            </div>

            <div class="card" style="text-align: center; color: #999; font-size: 0.9em;">
                <p>Generated by <strong>ez-automl-lite</strong> &bull; {timestamp}</p>
            </div>
        </div></body></html>
        """
