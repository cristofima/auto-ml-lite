"""
Training Results Report Generator.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.metrics import roc_curve, auc


def generate_training_report(
    output_path: str,
    job_id: str,
    problem_type: str,
    metrics: Dict[str, Any],
    feature_importance: Dict[str, float],
    training_config: Dict[str, Any],
    preprocessing_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    y_test: Optional[pd.Series] = None,
    y_pred: Optional[np.ndarray] = None,
    y_pred_proba: Optional[np.ndarray] = None
) -> None:
    """Generate training results report."""
    print("Generating training report...")
    try:
        report = TrainingReportGenerator(
            job_id=job_id,
            problem_type=problem_type,
            metrics=metrics,
            feature_importance=feature_importance,
            training_config=training_config,
            preprocessing_info=preprocessing_info,
            dataset_info=dataset_info,
            y_test=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba
        )
        html = report.generate()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Training report saved to: {output_path}")
    except Exception as e:
        import traceback
        print(f"Error generating training report: {str(e)}")
        traceback.print_exc()


class TrainingReportGenerator:
    """Generate training results report with CSS-only visualizations matching the premium design."""
    
    def __init__(
        self,
        job_id: str,
        problem_type: str,
        metrics: Dict[str, Any],
        feature_importance: Dict[str, float],
        training_config: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
        dataset_info: Dict[str, Any],
        y_test: Optional[pd.Series] = None,
        y_pred: Optional[np.ndarray] = None,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> None:
        self.job_id = job_id
        self.problem_type = problem_type
        self.metrics = metrics
        self.feature_importance = feature_importance
        self.training_config = training_config
        self.preprocessing_info = preprocessing_info
        self.dataset_info = dataset_info
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
    
    def _get_css(self) -> str:
        """Return premium CSS styles"""
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
            
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
            
            .stat-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; border-radius: 12px; text-align: center;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            .stat-box.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3); }
            .stat-box.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3); }
            .stat-box.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3); }
            .stat-box.gold { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); box-shadow: 0 4px 15px rgba(247, 151, 30, 0.3); }
            .stat-number { font-size: 2.2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
            
            .metric-card {
                background: #f8f9fa; border-radius: 12px; padding: 20px;
                text-align: center; border: 1px solid #e9ecef;
                transition: transform 0.2s;
            }
            .metric-card:hover { transform: translateY(-5px); }
            .metric-value { font-size: 2em; font-weight: bold; color: #1a73e8; }
            .metric-label { font-size: 0.9em; color: #666; margin-top: 5px; font-weight: 500; }
            .metric-card.success .metric-value { color: #28a745; }
            .metric-card.warning .metric-value { color: #ffc107; }
            
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 14px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; font-weight: 600; color: #555; text-transform: uppercase; font-size: 0.85em; }
            tr:hover { background: #fcfcfc; }
            
            .bar-container { 
                background: #e9ecef; border-radius: 6px; height: 12px; 
                overflow: hidden; margin: 5px 0;
            }
            .bar { 
                height: 100%; border-radius: 6px; 
                transition: width 0.6s ease-out;
            }
            .bar.primary { background: linear-gradient(90deg, #667eea, #764ba2); }
            .bar.success { background: linear-gradient(90deg, #11998e, #38ef7d); }
            .bar.warning { background: linear-gradient(90deg, #f7971e, #ffd200); }
            .bar.info { background: linear-gradient(90deg, #4facfe, #00f2fe); }
            
            .feature-bar { display: flex; align-items: center; margin: 12px 0; }
            .feature-name { width: 220px; font-size: 0.95em; color: #333; font-weight: 500; }
            .feature-bar-container { flex: 1; margin: 0 15px; }
            .feature-value { width: 70px; text-align: right; font-size: 0.9em; color: #666; font-family: monospace; }
            
            .badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 700; background: #eee; }
            .badge.classification { background: #e3f2fd; color: #1565c0; }
            .badge.regression { background: #f3e5f5; color: #7b1fa2; }
            
            /* Visualizations */
            .hist-chart { display: flex; align-items: flex-end; height: 120px; gap: 4px; margin: 20px 0; border-bottom: 2px solid #eee; padding-bottom: 5px; }
            .hist-bar { background: #667eea; border-radius: 4px 4px 0 0; flex: 1; min-width: 10px; position: relative; }
            .hist-bar:hover { background: #764ba2; }
            .hist-bar:hover::after {
                content: attr(data-value);
                position: absolute; top: -25px; left: 50%; transform: translateX(-50%);
                background: #333; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;
            }
            
            .chart-container { position: relative; height: 300px; width: 100%; border: 1px solid #eee; background: white; border-radius: 8px; margin-top: 20px; }
            .chart-svg { width: 100%; height: 100%; }
            .chart-line { fill: none; stroke: #1a73e8; stroke-width: 3; vector-effect: non-scaling-stroke; }
            .chart-area { fill: rgba(26, 115, 232, 0.1); }
            .chart-axis { stroke: #ddd; stroke-width: 1; }
            .chart-label { font-size: 12px; fill: #666; }
            
            .matrix-container { display: flex; justify_content: center; margin-top: 20px; }
            .matrix { display: grid; gap: 8px; max-width: 600px; width: 100%; position: relative; }
            .matrix-cell { aspect-ratio: 1.2; display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; position: relative; overflow: hidden; }
            .matrix-value { font-size: 1.4em; z-index: 1; }
            .matrix-label { font-size: 0.65em; position: absolute; top: 5px; opacity: 0.8; z-index: 1; }
            .matrix-header-row { display: grid; gap: 8px; max-width: 600px; width: 100%; margin-bottom: 5px; text-align: center; font-size: 0.8em; font-weight: bold; }
            .matrix-row-container { display: flex; align-items: center; gap: 10px; }
            .matrix-row-label { width: 80px; text-align: right; font-size: 0.8em; font-weight: bold; }
            
            @media (max-width: 768px) {
                .grid-2 { grid-template-columns: 1fr; }
                .feature-name { width: 140px; }
            }
        </style>
        """

    def _generate_header(self) -> str:
        training_time = self.metrics.get('training_time', 0)
        best_model = self.metrics.get('best_estimator', 'N/A')
        feature_count = self.preprocessing_info.get('feature_count', 'N/A')
        time_budget = self.metrics.get('time_budget', 'N/A')
        
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
                    <div class="stat-number" style="font-size: 1.4em;">{best_model}</div>
                    <div class="stat-label">Best Estimator</div>
                </div>
                <div class="stat-box gold">
                    <div class="stat-number">{time_budget}s</div>
                    <div class="stat-label">Time Budget</div>
                </div>
            </div>
            <div style="margin-top: 25px; padding-top: 20px; border-top: 1px solid #eee; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #666;">Task:</span> <span class="badge {self.problem_type}">{self.problem_type.upper()}</span>
                    <span style="margin-left: 20px; color: #666;">Target:</span> <code>{self.dataset_info.get('target_column', 'N/A')}</code>
                </div>
                <div style="color: #999; font-size: 0.85em;">ID: {self.job_id}</div>
            </div>
        </div>
        """

    def _generate_metrics(self) -> str:
        html = '<div class="card"><h2>📊 Performance Metrics</h2><div class="grid">'
        if self.problem_type == 'classification':
            met_list = [
                ('accuracy', 'Accuracy', 'success'), 
                ('f1_score', 'F1 Score', 'success'), 
                ('roc_auc', 'ROC AUC', 'primary'),
                ('log_loss', 'Log Loss', 'warning'),
                ('precision', 'Precision', 'info'), 
                ('recall', 'Recall', 'info')
            ]
        else:
            met_list = [
                ('r2_score', 'R² Score (Variance)', 'success'), 
                ('rmse', 'Root Mean Squared Error', 'warning'), 
                ('mae', 'Mean Absolute Error', 'warning'),
                ('mape', 'Mean Abs % Error (MAPE)', 'info')
            ]
            
        for key, label, cls in met_list:
            if key not in self.metrics: continue
            
            val = self.metrics.get(key, 0)
            
            # Format logic
            if key == 'mape':
                val_str = f"{val:.2%}"
            elif key in ['r2_score', 'accuracy', 'roc_auc']:
                val_str = f"{val:.2%}" # Show as percentage-like for consistency or standard? Usually AUC is 0.xx. Let's do 0.95 (95%)
                val_str = f"{val:.4f}"
            else:
                val_str = f"{val:.4f}"
                
            html += f"""
            <div class="metric-card {cls}">
                <div class="metric-value">{val_str}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
        html += '</div></div>'
        return html

    def _generate_regression_diagnostics(self) -> str:
        """Enhanced regression diagnostics: Residuals distribution and sample table"""
        if self.y_test is None or self.y_pred is None: return ""
        
        y_true = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        residuals = self.y_pred - y_true
        
        # Calculate histogram for residuals
        counts, bins = np.histogram(residuals, bins=15)
        max_count = max(counts) if len(counts) > 0 else 1
        
        html = '<div class="card"><h2>🔬 Regression Diagnostics</h2>'
        html += '<div class="grid-2">'
        
        # Residuals Histogram
        html += '<div><h3>Residuals Distribution</h3><p style="font-size:0.85em; color:#666;">Shows if errors are normally distributed (ideal for regression).</p>'
        html += '<div class="hist-chart">'
        for count in counts:
            h = (count / max_count) * 100
            html += f'<div class="hist-bar" style="height: {max(h, 2)}%;" data-value="{count}"></div>'
        html += '</div></div>'
        
        # Error metrics summary
        avg_err = np.mean(np.abs(residuals))
        max_err = np.max(np.abs(residuals))
        html += f"""
        <div>
            <h3>Error Analysis</h3>
            <table style="margin-top: 0;">
                <tr><td>Average Abs. Error</td><td><strong>{avg_err:.4f}</strong></td></tr>
                <tr><td>Maximum Abs. Error</td><td>{max_err:.4f}</td></tr>
                <tr><td>Underpredictions</td><td>{sum(residuals < 0)}</td></tr>
                <tr><td>Overpredictions</td><td>{sum(residuals > 0)}</td></tr>
            </table>
        </div>
        """
        html += '</div>'
        
        # Sample table
        html += '<h3>Detailed Sample Predictions</h3>'
        html += '<table><tr><th>Actual Value</th><th>Predicted Value</th><th>Residual (Error)</th><th>Relative Error</th></tr>'
        indices = np.linspace(0, len(y_true)-1, 10, dtype=int)
        for idx in indices:
            true_val = y_true[idx]
            pred_val = self.y_pred[idx]
            res = pred_val - true_val
            pct = abs(res / true_val * 100) if true_val != 0 else 0
            res_color = "#28a745" if abs(res) < avg_err else "#dc3545"
            html += f"""
            <tr>
                <td>{true_val:.4f}</td>
                <td>{pred_val:.4f}</td>
                <td style="color: {res_color}; font-weight: bold;">{res:+.4f}</td>
                <td>{pct:.2f}%</td>
            </tr>
            """
        html += '</table></div>'
        return html

    def _generate_roc_curve(self) -> str:
        """Generate SVG ROC Curve"""
        if self.y_test is None or self.y_pred_proba is None: return ""
        if len(np.unique(self.y_test)) > 2: return "<p>ROC Curve is only available for binary classification.</p>"
        
        try:
            # Assuming binary
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            # Create SVG path
            # ViewBox 0 0 100 100. Flip Y axis logic (1-y)
            path_d = "M 0 100 "
            for f, t in zip(fpr, tpr, strict=True):
                path_d += f"L {f*100:.1f} {(1-t)*100:.1f} "
            path_d += "L 100 0" # End
            
            svg = f"""
            <div class="chart-container">
                <svg class="chart-svg" viewBox="-5 -5 110 115" preserveAspectRatio="none">
                    <!-- Grid -->
                    <line x1="0" y1="0" x2="0" y2="100" class="chart-axis" />
                    <line x1="0" y1="100" x2="100" y2="100" class="chart-axis" />
                    <line x1="0" y1="0" x2="100" y2="0" class="chart-axis" stroke-dasharray="2,2" />
                    <line x1="100" y1="0" x2="100" y2="100" class="chart-axis" stroke-dasharray="2,2" />
                    <line x1="0" y1="100" x2="100" y2="0" class="chart-axis" stroke-dasharray="2,2" stroke="#ccc" />
                    
                    <!-- Labels -->
                    <text x="50" y="112" text-anchor="middle" class="chart-label">False Positive Rate</text>
                    <text x="-5" y="55" text-anchor="middle" transform="rotate(-90, -5, 55)" class="chart-label">True Positive Rate</text>
                    
                    <!-- Curve -->
                    <path d="{path_d}" class="chart-line" />
                    <text x="60" y="80" style="font-size: 14px; font-weight: bold; fill: #1a73e8;">AUC = {roc_auc:.4f}</text>
                </svg>
            </div>
            """
            return svg
        except Exception as e:
            return f"<p>Could not generate ROC curve: {str(e)}</p>"

    def _generate_classification_diagnostics(self) -> str:
        """Enhanced classification diagnostics: Confusion Matrix, class breakdown, and ROC"""
        if self.y_test is None or self.y_pred is None: return ""
        
        from sklearn.metrics import confusion_matrix, classification_report
        unique_labels = sorted(np.unique(self.y_test))
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        html = '<div class="card"><h2>🔬 Classification Diagnostics</h2>'
        
        html += '<div class="grid-2">'
        
        # Left Column: Matrix
        html += '<div>'
        if len(unique_labels) <= 6:
            html += '<h3>Confusion Matrix</h3><p style="font-size:0.85em; color:#666; margin-bottom:15px;">Predicted vs Actual classes.</p>'
            
            # Matrix Headers
            html += '<div style="margin-left: 90px;">' # Offset for row labels
            html += f'<div class="matrix-header-row" style="grid-template-columns: repeat({len(unique_labels)}, 1fr);">'
            for label in unique_labels: html += f'<div>PRED: {label}</div>'
            html += '</div></div>'
            
            max_val = cm.max() if cm.max() > 0 else 1
            for i, row in enumerate(cm):
                html += '<div class="matrix-row-container">'
                html += f'<div class="matrix-row-label">ACTUAL: {unique_labels[i]}</div>'
                html += f'<div class="matrix" style="grid-template-columns: repeat({len(unique_labels)}, 1fr);">'
                for j, val in enumerate(row):
                    opacity = 0.1 + (val / max_val) * 0.9
                    bg_color = f'rgba(26, 115, 232, {opacity})' if i == j else f'rgba(220, 53, 69, {opacity})'
                    text_color = '#fff' if opacity > 0.5 else '#333'
                    html += f'<div class="matrix-cell" style="background: {bg_color}; color: {text_color};">{val}</div>'
                html += '</div></div>'
        html += '</div>'
        
        # Right Column: ROC Curve (if binary)
        if len(unique_labels) == 2:
            html += '<div><h3>ROC Curve</h3><p style="font-size:0.85em; color:#666;">Receiver Operating Characteristic (Binary Only).</p>'
            html += self._generate_roc_curve()
            html += '</div>'
        else:
             html += '<div><h3>Multiclass Details</h3><p>ROC Curve available for binary classification.</p></div>'
             
        html += '</div>' # End Grid-2
        
        # Performance by Class table
        html += '<h3 style="margin-top:30px;">Class-wise Performance</h3>'
        try:
            report_dict = classification_report(self.y_test, self.y_pred, output_dict=True)
            html += '<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>'
            for label, met in report_dict.items():
                if label in ['accuracy', 'macro avg', 'weighted avg']: continue
                html += f"""
                <tr>
                    <td><strong>{label}</strong></td>
                    <td>{met['precision']:.2%}</td>
                    <td>{met['recall']:.2%}</td>
                    <td>{met['f1-score']:.2%}</td>
                    <td>{int(met['support']):,}</td>
                </tr>
                """
            html += '</table>'
        except:
            html += '<p>Detailed class metrics not available.</p>'
            
        html += '</div>'
        return html

    def _generate_importance(self) -> str:
        if not self.feature_importance: return ""
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        max_imp = max(v for _, v in sorted_features) if sorted_features else 1
        
        html = '<div class="card"><h2>📈 Feature Importance</h2><p style="color:#666; font-size:0.9em; margin-bottom:20px;">Top features contributing to the model\'s performance.</p>'
        colors = ['primary', 'success', 'warning', 'info']
        for i, (name, val) in enumerate(sorted_features):
            pct = (val / max_imp) * 100
            color = colors[i % len(colors)]
            html += f"""
            <div class="feature-bar">
                <div class="feature-name" title="{name}">{name}</div>
                <div class="feature-bar-container">
                    <div class="bar-container"><div class="bar {color}" style="width: {pct}%"></div></div>
                </div>
                <div class="feature-value">{val:.2f}</div>
            </div>
            """
        html += '</div>'
        return html

    def generate(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        diag_section = ""
        if self.problem_type == 'regression':
            diag_section = self._generate_regression_diagnostics()
        elif self.problem_type == 'classification':
            diag_section = self._generate_classification_diagnostics()

        return f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Training Report</title>{self._get_css()}</head>
        <body><div class="container">
            <h1>🤖 AutoML Training Report</h1>
            {self._generate_header()}
            {self._generate_metrics()}
            {diag_section}
            {self._generate_importance()}
            <div class="card" style="text-align: center; color: #999; font-size: 0.9em;">
                <p>Generated by <strong>ez-automl-lite</strong> &bull; {timestamp}</p>
            </div>
        </div></body></html>
        """

