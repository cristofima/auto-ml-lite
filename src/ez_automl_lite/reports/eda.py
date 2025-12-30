"""
EDA (Exploratory Data Analysis) Report Generator.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime, timezone

# Import shared utilities
from ez_automl_lite.utils.detection import detect_problem_type, is_id_column


def generate_eda_report(df: pd.DataFrame, target_column: str, output_path: str):
    """
    Generate comprehensive EDA report with pure HTML/CSS.
    """
    print("Generating comprehensive EDA report...")
    
    try:
        report = EDAReportGenerator(df, target_column)
        html = report.generate()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"EDA report saved to: {output_path}")
        
    except Exception as e:
        print(f"Error generating EDA report: {str(e)}")
        # Fallback to minimal report
        generate_minimal_report(df, target_column, output_path)


class EDAReportGenerator:
    """Generate comprehensive EDA report with CSS-only visualizations matching the premium theme."""
    
    def __init__(self, df: pd.DataFrame, target_column: str) -> None:
        self.df = df
        self.target_column = target_column
        self.target = df[target_column]
        self.features = df.drop(columns=[target_column])
        self.problem_type = detect_problem_type(self.target)
        self.warnings: List[str] = []
        self.excluded_columns: List[Tuple[str, str]] = []
        
        # Analyze columns
        self._analyze_columns()
    
    def _analyze_columns(self) -> None:
        """Analyze and categorize columns"""
        for col in self.features.columns:
            series = self.features[col]
            
            # Check for ID columns
            if is_id_column(col, series):
                self.excluded_columns.append((col, "ID/Identifier column"))
                continue
            
            # Check for constant columns
            if series.nunique() <= 1:
                self.excluded_columns.append((col, "Constant value (no variance)"))
                continue
            
            # Check for high cardinality categorical
            if series.dtype == 'object' and series.nunique() / len(series) > 0.5:
                self.excluded_columns.append((col, f"High cardinality ({series.nunique()} unique values)"))
        
        # Add warnings
        missing_cols = [col for col in self.df.columns if self.df[col].isnull().any()]
        if missing_cols:
            self.warnings.append(f"Missing values detected in {len(missing_cols)} column(s)")
        
        if self.problem_type == 'classification':
            class_counts = self.target.value_counts()
            if len(class_counts) > 0 and class_counts.min() > 0:
                imbalance_ratio = class_counts.max() / class_counts.min()
                if imbalance_ratio > 3:
                    self.warnings.append(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        else:
            skew = self.target.skew()
            if abs(skew) > 1:
                self.warnings.append(f"High skewness detected ({skew:.2f}) in target variable")

    def _get_css(self) -> str:
        """Return premium CSS styles"""
        return """
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                margin: 0; padding: 20px; background: #f5f7fa; color: #333;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }
            h2 { color: #333; margin-top: 30px; }
            h3 { color: #555; margin-top: 20px; }
            
            .card {
                background: white; border-radius: 8px; padding: 20px;
                margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; }
            .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
            
            .stat-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; border-radius: 8px; text-align: center;
            }
            .stat-box.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
            .stat-box.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
            .stat-box.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
            .stat-number { font-size: 2.5em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; }
            
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; font-weight: 600; color: #555; }
            tr:hover { background: #f8f9fa; }
            
            .bar-container { 
                background: #e9ecef; border-radius: 4px; height: 20px; 
                overflow: hidden; margin: 3px 0;
            }
            .bar { 
                height: 100%; border-radius: 4px; 
                display: flex; align-items: center; padding-left: 8px;
                font-size: 11px; color: white; font-weight: 500;
                transition: width 0.3s ease;
            }
            .bar.primary { background: linear-gradient(90deg, #667eea, #764ba2); }
            .bar.success { background: linear-gradient(90deg, #11998e, #38ef7d); }
            .bar.warning { background: linear-gradient(90deg, #f093fb, #f5576c); }
            .bar.info { background: linear-gradient(90deg, #4facfe, #00f2fe); }
            
            .warning-box {
                background: #fff3cd; border-left: 4px solid #ffc107;
                padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0;
            }
            
            .badge {
                display: inline-block; padding: 3px 8px; border-radius: 12px;
                font-size: 0.8em; font-weight: 500;
            }
            .badge.classification { background: #e3f2fd; color: #1565c0; }
            .badge.regression { background: #f3e5f5; color: #7b1fa2; }
            .badge.numeric { background: #e8f5e9; color: #2e7d32; }
            .badge.categorical { background: #fff3e0; color: #e65100; }
            .badge.excluded { background: #ffebee; color: #c62828; }
            
            .mini-chart { display: flex; align-items: flex-end; height: 40px; gap: 2px; }
            .mini-bar { background: #667eea; border-radius: 2px 2px 0 0; min-width: 8px; }
            
            @media (max-width: 768px) {
                .grid-2 { grid-template-columns: 1fr; }
            }
        </style>
        """
    
    def _generate_overview(self) -> str:
        """Generate dataset overview section"""
        n_rows, n_cols = self.df.shape
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_numeric = len(numeric_cols)
        n_categorical = n_cols - n_numeric
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        
        return f"""
        <div class="card">
            <h2>📊 Dataset Overview</h2>
            <div class="grid">
                <div class="stat-box">
                    <div class="stat-number">{n_rows:,}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                <div class="stat-box green">
                    <div class="stat-number">{n_cols}</div>
                    <div class="stat-label">Total Columns</div>
                </div>
                <div class="stat-box orange">
                    <div class="stat-number">{n_numeric}</div>
                    <div class="stat-label">Numeric Features</div>
                </div>
                <div class="stat-box blue">
                    <div class="stat-number">{n_categorical}</div>
                    <div class="stat-label">Categorical Features</div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <p><strong>Target Column:</strong> <code>{self.target_column}</code></p>
                <p><strong>Problem Type:</strong> <span class="badge {self.problem_type}">{self.problem_type.upper()}</span></p>
                <p><strong>Memory Usage:</strong> {memory_mb:.2f} MB</p>
            </div>
        </div>
        """
    
    def _generate_target_analysis(self) -> str:
        """Generate target variable analysis"""
        html = '<div class="card"><h2>🎯 Target Variable Analysis</h2>'
        
        if self.problem_type == 'classification':
            class_counts = self.target.value_counts()
            total = len(self.target)
            
            html += '<h3>Class Distribution</h3>'
            html += '<table><tr><th>Class</th><th>Count</th><th>Percentage</th><th>Distribution</th></tr>'
            
            colors = ['primary', 'success', 'warning', 'info']
            for i, (cls, count) in enumerate(class_counts.items()):
                pct = count / total * 100
                color = colors[i % len(colors)]
                html += f"""
                <tr>
                    <td><strong>{cls}</strong></td>
                    <td>{count:,}</td>
                    <td>{pct:.1f}%</td>
                    <td>
                        <div class="bar-container">
                            <div class="bar {color}" style="width: {pct}%">{pct:.1f}%</div>
                        </div>
                    </td>
                </tr>
                """
            html += '</table>'
        else:
            stats = self.target.describe()
            html += f"""
            <div class="grid-2">
                <div>
                    <h3>Statistics</h3>
                    <table>
                        <tr><td>Mean</td><td><strong>{stats['mean']:.4f}</strong></td></tr>
                        <tr><td>Median</td><td>{stats['50%']:.4f}</td></tr>
                        <tr><td>Std Dev</td><td>{stats['std']:.4f}</td></tr>
                        <tr><td>Min</td><td>{stats['min']:.4f}</td></tr>
                        <tr><td>Max</td><td>{stats['max']:.4f}</td></tr>
                    </table>
                </div>
                <div>
                    <h3>Distribution</h3>
                    {self._generate_histogram(self.target)}
                </div>
            </div>
            """
        html += '</div>'
        return html
    
    def _generate_histogram(self, series: pd.Series, bins: int = 20) -> str:
        """Generate a basic CSS histogram"""
        try:
            counts, _ = np.histogram(series.dropna(), bins=bins)
            max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 1
            
            html = '<div class="mini-chart">'
            for count in counts:
                height = int((count / max_count) * 40)
                html += f'<div class="mini-bar" style="height: {max(height, 2)}px;"></div>'
            html += '</div>'
            return html
        except:
            return "Chart not available"

    def _generate_correlations(self) -> str:
        """Generate correlation with target for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) <= 1:
            return ""
        
        try:
            corrs = numeric_df.corr()[self.target_column].sort_values(ascending=False)
            corrs = corrs.drop(self.target_column)
            
            html = '<div class="card"><h2>📈 Correlation with Target</h2>'
            html += '<table><tr><th>Feature</th><th>Correlation</th><th>Strength</th></tr>'
            
            for col, val in corrs.items():
                strength = "Weak"
                if abs(val) > 0.7: strength = "Strong"
                elif abs(val) > 0.4: strength = "Moderate"
                
                # Check for NaN correlations
                if pd.isna(val):
                    val_str = "N/A"
                else:
                    val_str = f"{val:.3f}"
                
                html += f"""
                <tr>
                    <td><code>{col}</code></td>
                    <td>{val_str}</td>
                    <td>{strength}</td>
                </tr>
                """
            html += '</table></div>'
            return html
        except:
            return ""

    def _generate_warnings(self) -> str:
        """Generate preprocessing warnings and excluded columns"""
        if not self.warnings and not self.excluded_columns:
            return ''
            
        html = '<div class="card"><h2>⚠️ Preprocessing Notes</h2>'
        
        if self.excluded_columns:
            html += '<h3>Columns to be Excluded</h3>'
            html += '<table><tr><th>Column</th><th>Reason</th></tr>'
            for col, reason in self.excluded_columns:
                html += f'<tr><td><code>{col}</code></td><td>{reason}</td></tr>'
            html += '</table>'
            
        if self.warnings:
            for warning in self.warnings:
                html += f'<div class="warning-box">{warning}</div>'
                
        html += '</div>'
        return html

    def _generate_column_details(self) -> str:
        """Generate table with details for all columns"""
        html = '<div class="card"><h2>📋 Column Details</h2>'
        html += '<table><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Samples</th></tr>'
        
        for col in self.df.columns:
            series = self.df[col]
            missing_pct = series.isnull().sum() / len(series) * 100
            unique = series.nunique()
            
            is_numeric = pd.api.types.is_numeric_dtype(series)
            type_badge = '<span class="badge numeric">numeric</span>' if is_numeric else '<span class="badge categorical">categorical</span>'
            
            samples = series.dropna().head(3).tolist()
            samples_str = ", ".join([str(s)[:20] for s in samples])
            
            html += f"""
            <tr>
                <td><code>{col}</code></td>
                <td>{type_badge}</td>
                <td>{missing_pct:.1f}%</td>
                <td>{unique:,}</td>
                <td style="font-size: 0.85em; color: #666;">{samples_str}</td>
            </tr>
            """
            
        html += '</table></div>'
        return html

    def generate(self) -> str:
        """Assemble the full report"""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>EDA Report - {self.target_column}</title>
            {self._get_css()}
        </head>
        <body>
            <div class="container">
                <h1>📊 Exploratory Data Analysis</h1>
                {self._generate_overview()}
                {self._generate_target_analysis()}
                {self._generate_warnings()}
                {self._generate_correlations()}
                {self._generate_column_details()}
                
                <div class="card" style="text-align: center; color: #666;">
                    <p>Generated by ez-automl-lite</p>
                    <p style="font-size: 0.85em;">{timestamp}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html


def generate_minimal_report(df: pd.DataFrame, target_column: str, output_path: str):
    """Fallback minimal report"""
    html = f"""
    <html>
    <head><title>Minimal EDA Report</title></head>
    <body style="font-family: sans-serif; padding: 20px;">
        <h1>Minimal EDA Report</h1>
        <p><strong>Target:</strong> {target_column}</p>
        <p><strong>Dataset Shape:</strong> {df.shape[0]} rows, {df.shape[1]} columns</p>
        <h2>Column Types</h2>
        <ul>
            {"".join([f"<li>{col}: {dtype}</li>" for col, dtype in df.dtypes.items()])}
        </ul>
    </body>
    </html>
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
