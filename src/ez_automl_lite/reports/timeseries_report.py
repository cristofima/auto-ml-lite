"""
Time Series Results Report Generator.
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def generate_timeseries_report(
    output_path: str,
    job_id: str,
    metrics: dict[str, Any],
    dataset_info: dict[str, Any],
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_lower: np.ndarray | None,
    y_pred_upper: np.ndarray | None,
    time_index: pd.Series,
    forecast_values: np.ndarray | None,
    forecast_dates: pd.DatetimeIndex | None,
    forecast_lower: np.ndarray | None,
    forecast_upper: np.ndarray | None,
    decomposition: dict[str, Any] | None,
) -> None:
    """Generate time series results report."""
    print("Generating time series report...")
    try:
        report = TimeSeriesReportGenerator(
            job_id=job_id,
            metrics=metrics,
            dataset_info=dataset_info,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            y_pred_lower=y_pred_lower,
            y_pred_upper=y_pred_upper,
            time_index=time_index,
            forecast_values=forecast_values,
            forecast_dates=forecast_dates,
            forecast_lower=forecast_lower,
            forecast_upper=forecast_upper,
            decomposition=decomposition,
        )
        html = report.generate()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Time series report saved to: {output_path}")
    except OSError as e:
        print(f"Error writing time series report: {e}")
    except (ValueError, KeyError, AttributeError) as e:
        print(f"Error generating time series report (data issue): {e}")


class TimeSeriesReportGenerator:
    """Generate time series results report with premium CSS-only visuals."""

    def __init__(
        self,
        job_id: str,
        metrics: dict[str, Any],
        dataset_info: dict[str, Any],
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_lower: np.ndarray | None,
        y_pred_upper: np.ndarray | None,
        time_index: pd.Series,
        forecast_values: np.ndarray | None,
        forecast_dates: pd.DatetimeIndex | None,
        forecast_lower: np.ndarray | None,
        forecast_upper: np.ndarray | None,
        decomposition: dict[str, Any] | None,
    ):
        self.job_id = job_id
        self.metrics = metrics
        self.dataset_info = dataset_info
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_lower = y_pred_lower
        self.y_pred_upper = y_pred_upper
        self.time_index = time_index
        self.forecast_values = forecast_values
        self.forecast_dates = forecast_dates
        self.forecast_lower = forecast_lower
        self.forecast_upper = forecast_upper
        self.decomposition = decomposition or {}

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
            h3 { color: #555; margin-top: 20px; }

            .card {
                background: white; border-radius: 8px; padding: 25px;
                margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }

            .grid {
                display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }

            .stat-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; border-radius: 8px; text-align: center;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            .stat-box.green {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
            }
            .stat-box.orange {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
            }
            .stat-box.blue {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
            }
            .stat-box.gold {
                background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
                box-shadow: 0 4px 15px rgba(247, 151, 30, 0.3);
            }
            .stat-number { font-size: 2.2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
            .info-label { font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 5px; }
            .info-value { font-size: 18px; font-weight: bold; color: #1a73e8; }

            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
            th { background: #f8f9fa; font-weight: 600; color: #555; }
            tr:hover { background: #f8f9fa; }

            .time-series-chart {
                width: 100%; height: 400px; margin: 20px 0;
                background: linear-gradient(to top, #f8f9fa 0%, white 100%);
                border: 1px solid #e0e0e0; border-radius: 6px;
                position: relative; padding: 40px 20px 40px 60px;
            }

            .chart-line {
                position: absolute; width: calc(100% - 80px); height: 2px;
                background: #1a73e8; transform-origin: left center;
            }

            .chart-area {
                position: absolute; width: calc(100% - 80px); height: 300px;
                background: linear-gradient(to bottom, rgba(26, 115, 232, 0.1), rgba(26, 115, 232, 0.01));
                border: 1px solid rgba(26, 115, 232, 0.3);
                border-radius: 4px;
            }

            .metric-card {
                background: white; padding: 20px; border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08); transition: transform 0.2s;
            }
            .metric-card:hover { transform: translateY(-5px); }
            .metric-value { font-size: 2em; font-weight: bold; color: #1a73e8; }
            .metric-label { font-size: 0.9em; color: #666; margin-top: 5px; font-weight: 500; }
            .metric-card.success .metric-value { color: #28a745; }
            .metric-card.warning .metric-value { color: #ffc107; }
            .metric-card.info .metric-value { color: #17a2b8; }

            .badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 700; background: #eee; }
            .badge.timeseries { background: #fff3e0; color: #f57c00; }

            .stationarity-indicator {
                display: inline-block; padding: 6px 12px; border-radius: 4px;
                font-weight: 500; font-size: 14px;
            }
            .stationary { background: #e8f5e9; color: #2e7d32; }
            .non-stationary { background: #ffebee; color: #c62828; }

            .forecast-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; border-radius: 8px;
                margin: 20px 0;
            }

            .decomposition-grid {
                display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px; margin: 20px 0;
            }

            .decomp-card {
                background: #f8f9fa; padding: 15px; border-radius: 6px;
                border-top: 3px solid #1a73e8;
            }

            .legend {
                display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap;
            }
            .legend-item {
                display: flex; align-items: center; gap: 8px;
            }
            .legend-color {
                width: 20px; height: 3px; border-radius: 2px;
            }
        </style>
        """

    def _get_header(self) -> str:
        training_time = self.metrics.get("execution_time", 0)
        algorithm = self.metrics.get("algorithm", "N/A")
        horizon = self.metrics.get("forecast_horizon", "N/A")

        return f"""
        <div class="card">
            <div class="grid">
                <div class="stat-box green">
                    <div class="stat-number">✓</div>
                    <div class="stat-label">Model Ready</div>
                </div>
                <div class="stat-box blue">
                    <div class="stat-number">{training_time:.1f}s</div>
                    <div class="stat-label">Training Time</div>
                </div>
                <div class="stat-box orange">
                    <div class="stat-number" style="font-size: 1.4em;">{algorithm}</div>
                    <div class="stat-label">Algorithm</div>
                </div>
                <div class="stat-box gold">
                    <div class="stat-number">{horizon}</div>
                    <div class="stat-label">Forecast Horizon</div>
                </div>
            </div>
            <div style="margin-top: 25px; padding-top: 20px; border-top: 1px solid #eee; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #666;">Task:</span> <span class="badge timeseries">TIME SERIES</span>
                    <span style="margin-left: 20px; color: #666;">Target:</span> <code>{self.dataset_info.get('target_column', 'N/A')}</code>
                </div>
                <div style="color: #999; font-size: 0.85em;">ID: {self.job_id}</div>
            </div>
        </div>
        """

    def _get_dataset_info(self) -> str:
        return f"""
        <div class="card">
            <h2>📊 Dataset Information</h2>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Samples</td>
                    <td>{self.dataset_info.get('total_samples', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Date Range</td>
                    <td>{self.dataset_info.get('date_range', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Frequency</td>
                    <td>{self.dataset_info.get('frequency', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Target Column</td>
                    <td>{self.dataset_info.get('target_column', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Train Size</td>
                    <td>{self.metrics.get('train_size', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Test Size</td>
                    <td>{self.metrics.get('test_size', 'N/A')}</td>
                </tr>
            </table>
        </div>
        """

    def _get_metrics(self) -> str:
        mae = self.metrics.get("mae", 0)
        rmse = self.metrics.get("rmse", 0)
        smape = self.metrics.get("smape", 0)
        mape = self.metrics.get("mape")

        # Stationarity info
        stationarity = self.metrics.get("stationarity", {})
        is_stationary = stationarity.get("is_stationary")

        html = '<div class="card"><h2>📊 Performance Metrics</h2><div class="grid">'

        # Main metrics as cards
        html += f"""
            <div class="metric-card success">
                <div class="metric-value">{mae:.4f}</div>
                <div class="metric-label">Mean Absolute Error</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{rmse:.4f}</div>
                <div class="metric-label">Root Mean Squared Error</div>
            </div>
            <div class="metric-card info">
                <div class="metric-value">{smape:.2f}%</div>
                <div class="metric-label">Symmetric MAPE</div>
            </div>
        """

        if mape is not None:
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{mape:.2f}%</div>
                <div class="metric-label">Mean Abs % Error (MAPE)</div>
            </div>
            """

        html += "</div>"

        # Stationarity section if available
        if is_stationary is not None:
            stat_class = "stationary" if is_stationary else "non-stationary"
            stat_text = "Stationary ✓" if is_stationary else "Non-Stationary"
            stat_badge = f'<span class="stationarity-indicator {stat_class}">{stat_text}</span>'
            adf_stat = stationarity.get("adf_statistic", 0)
            p_value = stationarity.get("p_value", 0)

            html += f"""
            <h3>📉 Stationarity Analysis</h3>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Result</th>
                </tr>
                <tr>
                    <td>Augmented Dickey-Fuller (ADF)</td>
                    <td>{stat_badge}</td>
                </tr>
                <tr>
                    <td>ADF Statistic</td>
                    <td>{adf_stat:.4f}</td>
                </tr>
                <tr>
                    <td>P-Value</td>
                    <td>{p_value:.4f}</td>
                </tr>
            </table>
            """

        html += "</div>"
        return html

    def _get_forecast_plot(self) -> str:
        """Generate visual representation of actual vs predicted values."""
        split_idx = len(self.y_train)
        y_all = pd.concat([self.y_train, self.y_test])

        if len(y_all) == 0:
            return """
            <div class="card">
                <h2>📉 Forecast vs Actual</h2>
                <p>No data available for visualization.</p>
            </div>
            """

        # Legend
        legend = """
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #1a73e8;"></div>
                <span>Training Data</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #34a853;"></div>
                <span>Actual Test Data</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ea4335; border: 2px dashed #ea4335; height: 0;"></div>
                <span>Predicted Values</span>
            </div>
        </div>
        """

        chart_desc = f"""
        <p><strong>Time Series Visualization:</strong> Blue area shows training data ({len(self.y_train)} points),
        green line shows actual test values ({len(self.y_test)} points), and red dashed line shows predictions
        (MAE: {self.metrics.get('mae', 0):.4f}).</p>
        """

        return f"""
        <div class="card">
            <h2>📉 Forecast vs Actual</h2>
            {legend}
            {chart_desc}
            <div class="time-series-chart">
                <!-- Simplified visual representation -->
                <div style="position: absolute; bottom: 20px; left: 60px; right: 20px; height: 300px; border-left: 2px solid #666; border-bottom: 2px solid #666;">
                    <!-- Training area (blue) -->
                    <div style="position: absolute; left: 0; width: {split_idx/(len(y_all))*100}%; height: 100%; background: linear-gradient(to bottom, rgba(26, 115, 232, 0.2), rgba(26, 115, 232, 0.05)); border-right: 2px dashed #999;"></div>
                    <!-- Test area (green) -->
                    <div style="position: absolute; left: {split_idx/(len(y_all))*100}%; width: {len(self.y_test)/(len(y_all))*100}%; height: 100%; background: linear-gradient(to bottom, rgba(52, 168, 83, 0.1), rgba(52, 168, 83, 0.02));"></div>
                </div>
                <div style="position: absolute; bottom: 5px; left: 60px; font-size: 11px; color: #666;">Time →</div>
                <div style="position: absolute; top: 40px; left: 10px; font-size: 11px; color: #666; writing-mode: vertical-lr; transform: rotate(180deg);">Value →</div>
            </div>
        </div>
        """

    def _generate_decomp_line_chart(
        self, data: np.ndarray | pd.Series, title: str, color: str
    ) -> str:
        """Generate a simple SVG line chart for decomposition component"""
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data

        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return ""

        # Fill/interpolate NaNs to avoid invalid SVG paths.
        s = pd.Series(values)
        if s.isna().all():
            return ""

        values = s.interpolate(limit_direction="both").ffill().bfill().to_numpy()

        # Normalize data for plotting
        min_val = float(np.nanmin(values))
        max_val = float(np.nanmax(values))
        range_val = max_val - min_val if max_val != min_val else 1

        # Sample every nth point to keep SVG manageable
        step = max(1, len(values) // 50)
        sampled_values = values[::step]

        if len(sampled_values) < 2:
            return ""

        # Create SVG path
        svg = '<svg viewBox="0 -10 100 120" width="100%" style="height: 280px; margin: 0;">'
        svg += (
            '<defs><linearGradient id="grad-'
            + color.replace("#", "")
            + '" x1="0%" y1="0%" x2="0%" y2="100%">'
        )
        svg += f'<stop offset="0%" style="stop-color:{color};stop-opacity:0.3" />'
        svg += f'<stop offset="100%" style="stop-color:{color};stop-opacity:0.01" />'
        svg += "</linearGradient></defs>"

        # Draw grid background
        svg += '<rect x="0" y="0" width="100" height="100" fill="#f8f9fa" stroke="#e0e0e0" stroke-width="0.5"/>'

        # Generate path
        path_d = f"M 0 {100 - (sampled_values[0] - min_val) / range_val * 100} "
        for i, val in enumerate(sampled_values[1:], 1):
            x = (i / (len(sampled_values) - 1)) * 100
            y = 100 - (val - min_val) / range_val * 100
            path_d += f"L {x} {y} "

        # Draw filled area under curve
        fill_path = path_d + "L 100 100 L 0 100 Z"
        svg += f'<path d="{fill_path}" fill="url(#grad-' + color.replace("#", "") + ')"/>'

        # Draw line
        svg += f'<path d="{path_d}" stroke="{color}" stroke-width="1.5" fill="none" stroke-linecap="round"/>'

        # Add axes
        svg += '<line x1="0" y1="100" x2="100" y2="100" stroke="#999" stroke-width="0.5"/>'
        svg += '<line x1="0" y1="0" x2="0" y2="100" stroke="#999" stroke-width="0.5"/>'

        # Axis labels
        svg += '<text x="50" y="115" font-size="3" fill="#666" text-anchor="middle">Time</text>'
        svg += '<text x="-5" y="50" font-size="3" fill="#666" text-anchor="end" transform="rotate(-90 -5 50)">Value</text>'

        # Min/Max labels
        svg += f'<text x="2" y="105" font-size="2.5" fill="#999">{min_val:.2f}</text>'
        svg += f'<text x="2" y="8" font-size="2.5" fill="#999">{max_val:.2f}</text>'

        svg += "</svg>"
        return svg

    def _get_decomposition(self) -> str:
        """Display time series decomposition if available with line chart visualizations."""
        if not self.decomposition:
            return ""

        trend = self.decomposition.get("trend")
        seasonal = self.decomposition.get("seasonal")
        residual = self.decomposition.get("residual")

        if trend is None or seasonal is None or residual is None:
            return ""

        # Calculate statistics
        trend_mean = float(trend.mean()) if not pd.isna(trend.mean()) else 0
        seasonal_mean = float(seasonal.mean()) if not pd.isna(seasonal.mean()) else 0
        residual_std = float(residual.std()) if not pd.isna(residual.std()) else 0

        # Generate line charts
        trend_chart = self._generate_decomp_line_chart(trend, "Trend", "#1a73e8")
        seasonal_chart = self._generate_decomp_line_chart(seasonal, "Seasonal", "#34a853")
        residual_chart = self._generate_decomp_line_chart(residual, "Residual", "#ea4335")

        return f"""
        <div class="card">
            <h2>🔍 Time Series Decomposition</h2>
            <p>Decomposition breaks down the time series into three components: trend (long-term direction), seasonal (repeating patterns), and residual (random noise).</p>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;">
                <!-- Trend Component -->
                <div style="background: #f8f9ff; border: 1px solid #e0e0ff; border-radius: 8px; padding: 15px;">
                    <h4 style="margin: 0 0 10px 0; color: #1a73e8; font-size: 1em;">📈 Trend</h4>
                    <p style="font-size: 0.85em; color: #666; margin: 0 0 10px 0;">Long-term direction</p>
                    {trend_chart}
                    <div style="text-align: center; padding: 10px; background: #f0f4ff; border-radius: 6px; margin-top: 10px;">
                        <div style="font-size: 1.1em; font-weight: bold; color: #1a73e8;">{trend_mean:.2f}</div>
                        <div style="font-size: 0.8em; color: #666;">Mean</div>
                    </div>
                </div>

                <!-- Seasonal Component -->
                <div style="background: #f8fff8; border: 1px solid #e0ffe0; border-radius: 8px; padding: 15px;">
                    <h4 style="margin: 0 0 10px 0; color: #34a853; font-size: 1em;">🔄 Seasonal</h4>
                    <p style="font-size: 0.85em; color: #666; margin: 0 0 10px 0;">Repeating patterns</p>
                    {seasonal_chart}
                    <div style="text-align: center; padding: 10px; background: #f0fff0; border-radius: 6px; margin-top: 10px;">
                        <div style="font-size: 1.1em; font-weight: bold; color: #34a853;">{seasonal_mean:.2f}</div>
                        <div style="font-size: 0.8em; color: #666;">Mean</div>
                    </div>
                </div>

                <!-- Residual Component -->
                <div style="background: #fff8f8; border: 1px solid #ffe0e0; border-radius: 8px; padding: 15px;">
                    <h4 style="margin: 0 0 10px 0; color: #ea4335; font-size: 1em;">📊 Residual</h4>
                    <p style="font-size: 0.85em; color: #666; margin: 0 0 10px 0;">Random noise</p>
                    {residual_chart}
                    <div style="text-align: center; padding: 10px; background: #fff0f0; border-radius: 6px; margin-top: 10px;">
                        <div style="font-size: 1.1em; font-weight: bold; color: #ea4335;">{residual_std:.2f}</div>
                        <div style="font-size: 0.8em; color: #666;">Std Dev</div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _get_forecast_section(self) -> str:
        """Display future forecasts if available with confidence intervals."""
        if self.forecast_values is None or self.forecast_dates is None:
            return ""

        # Check if confidence intervals are available
        has_intervals = self.forecast_lower is not None and self.forecast_upper is not None

        # Show first 10 forecast values
        forecast_rows = ""
        for i in range(min(10, len(self.forecast_values))):
            date = self.forecast_dates[i]
            value = self.forecast_values[i]

            if has_intervals:
                lower = self.forecast_lower[i]
                upper = self.forecast_upper[i]
                interval_str = f"<td>{lower:.4f} - {upper:.4f}</td>"
            else:
                interval_str = "<td>N/A</td>"

            forecast_rows += f"""
                <tr>
                    <td>Step {i+1}</td>
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td><strong>{value:.4f}</strong></td>
                    {interval_str}
                </tr>
            """

        total_forecasts = len(self.forecast_values)
        remaining = total_forecasts - 10
        remaining_note = (
            f"<p><em>Showing first 10 of {total_forecasts} forecasted values...</em></p>"
            if remaining > 0
            else ""
        )

        interval_header = (
            "<th>95% Confidence Interval</th>" if has_intervals else "<th>Confidence Interval</th>"
        )

        return f"""
        <div class="card">
            <div class="forecast-box">
                <h2 style="color: white; margin-top: 0;">🔮 Future Forecasts</h2>
                <p style="color: rgba(255,255,255,0.9);">Generated {total_forecasts} future predictions{' with 95% confidence intervals' if has_intervals else ''}</p>
            </div>
            {remaining_note}
            <table>
                <tr>
                    <th>Step</th>
                    <th>Date</th>
                    <th>Predicted Value</th>
                    {interval_header}
                </tr>
                {forecast_rows}
            </table>
        </div>
        """

    def _get_footer(self) -> str:
        """Generate footer with generation timestamp."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""
        <div class="card" style="text-align: center; color: #999; font-size: 0.9em;">
            <p>Generated by <strong>ez-automl-lite</strong> &bull; {timestamp}</p>
        </div>
        """

    def generate(self) -> str:
        """Generate complete HTML report."""
        return f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Time Series Forecasting Report</title>
            {self._get_css()}
        </head>
        <body>
            <div class="container">
                <h1>⏰ Time Series Forecasting Report</h1>
                {self._get_header()}
                {self._get_metrics()}
                {self._get_dataset_info()}
                {self._get_forecast_plot()}
                {self._get_decomposition()}
                {self._get_forecast_section()}
                {self._get_footer()}
            </div>
        </body>
        </html>
        """
