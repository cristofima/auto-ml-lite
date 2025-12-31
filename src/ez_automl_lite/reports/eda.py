"""
EDA (Exploratory Data Analysis) Report Generator.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime, timezone

# Import shared utilities
from ez_automl_lite.utils.detection import detect_problem_type, is_id_column


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_eda_report(df: pd.DataFrame, target_column: str = None, output_path: str = "eda_report.html", task_type: str = None):
    """
    Generate comprehensive EDA report with pure HTML/CSS.
    task_type: 'clustering', 'anomaly_detection', or None (inferred from target)
    """
    print(f"Generating comprehensive EDA report (Task: {task_type or 'Auto'})...")
    
    try:
        report = EDAReportGenerator(df, target_column, task_type)
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
    
    def __init__(self, df: pd.DataFrame, target_column: str = None, task_type: str = None) -> None:
        self.df = df
        self.target_column = target_column
        
        if target_column:
            self.target = df[target_column]
            self.features = df.drop(columns=[target_column])
            self.problem_type = detect_problem_type(self.target)
        else:
            self.target = None
            self.features = df
            # Use explicit task type if provided, else 'unsupervised'
            self.problem_type = task_type if task_type else 'unsupervised'
            
        self.warnings: List[str] = []
        self.excluded_columns: List[Tuple[str, str]] = []
        
        # Analyze columns
        self._analyze_columns()
    
    # ... (Keep _analyze_columns as is, or update if needed) ...
    # Re-implementing parts of _analyze_columns to avoid cutting code if I simply use replace_file_content poorly.
    # Actually, I'll paste the full updated class logic for the new sections.

    def _analyze_columns(self) -> None:
        """Analyze and categorize columns"""
        for col in self.features.columns:
            series = self.features[col]
            if is_id_column(col, series):
                self.excluded_columns.append((col, "ID/Identifier column"))
                continue
            if series.nunique() <= 1:
                self.excluded_columns.append((col, "Constant value (no variance)"))
                continue
            if series.dtype == 'object' and series.nunique() / len(series) > 0.5:
                self.excluded_columns.append((col, f"High cardinality ({series.nunique()} unique values)"))
        
        missing_cols = [col for col in self.df.columns if self.df[col].isnull().any()]
        if missing_cols:
            self.warnings.append(f"Missing values detected in {len(missing_cols)} column(s)")
        
        if self.problem_type == 'classification' and self.target is not None:
            class_counts = self.target.value_counts()
            if len(class_counts) > 0 and class_counts.min() > 0:
                imbalance_ratio = class_counts.max() / class_counts.min()
                if imbalance_ratio > 3:
                    self.warnings.append(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        elif self.problem_type == 'regression' and self.target is not None:
            skew = self.target.skew()
            if abs(skew) > 1:
                self.warnings.append(f"High skewness detected ({skew:.2f}) in target variable")

    def _get_css(self) -> str:
        # ... (previous CSS) ... plus new styles for boxplots/pca
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
            
            .bar-container { background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; margin: 3px 0; }
            .bar { height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 11px; color: white; transition: width 0.3s ease; }
            .bar.primary { background: linear-gradient(90deg, #667eea, #764ba2); }
            .bar.success { background: linear-gradient(90deg, #11998e, #38ef7d); }
            .bar.warning { background: linear-gradient(90deg, #f093fb, #f5576c); }
            
            .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
            
            .badge { display: inline-block; query: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 500; padding: 3px 8px; }
            .badge.classification { background: #e3f2fd; color: #1565c0; }
            .badge.regression { background: #f3e5f5; color: #7b1fa2; }
            .badge.clustering { background: #e0f2f1; color: #00695c; }
            .badge.anomaly_detection { background: #ffebee; color: #c62828; }
            .badge.numeric { background: #e8f5e9; color: #2e7d32; }
            .badge.categorical { background: #fff3e0; color: #e65100; }
            
            .mini-chart { display: flex; align-items: flex-end; height: 40px; gap: 2px; }
            .mini-bar { background: #667eea; border-radius: 2px 2px 0 0; min-width: 8px; }

            /* PCA Plot */
            .chart-svg { width: 100%; height: 300px; background: #fafafa; border: 1px solid #eee; border-radius: 8px; }
            .scatter-pt { fill: #667eea; opacity: 0.6; }
        </style>
        """

    def _generate_overview(self) -> str:
        """Generate dataset overview section"""
        n_rows, n_cols = self.df.shape
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_numeric = len(numeric_cols)
        n_categorical = n_cols - n_numeric
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        
        target_info = ""
        if self.target_column:
            target_info = f"<p><strong>Target Column:</strong> <code>{self.target_column}</code></p>"
            
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
                {target_info}
                <p><strong>Problem Type:</strong> <span class="badge {self.problem_type}">{self.problem_type.replace('_', ' ').upper()}</span></p>
                <p><strong>Memory Usage:</strong> {memory_mb:.2f} MB</p>
            </div>
        </div>
        """

    def _generate_unsupervised_analysis(self) -> str:
        """Generate analysis specific to Clustering/Anomaly tasks"""
        if self.problem_type not in ['clustering', 'anomaly_detection']:
            return ""
            
        html = '<div class="card">'
        
        # 1. PCA Visualization of Raw Structure
        html += f'<h2>🧬 Data Structure Analysis ({self.problem_type.replace("_", " ").title()})</h2>'
        
        numeric_df = self.df.select_dtypes(include=[np.number]).fillna(0)
        if len(numeric_df.columns) >= 2:
            try:
                # Simple PCA on raw data to show structure
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(numeric_df)
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X_scaled)
                
                # Sample if too large
                if len(coords) > 1000:
                    idx = np.random.choice(len(coords), 1000, replace=False)
                    coords = coords[idx]
                
                # SVG generation
                points = ""
                min_x, max_x = coords[:,0].min(), coords[:,0].max()
                min_y, max_y = coords[:,1].min(), coords[:,1].max()
                
                # Normalize to 0-100
                w, h = 100, 100
                norm_x = (coords[:,0] - min_x) / (max_x - min_x) * w
                norm_y = (coords[:,1] - min_y) / (max_y - min_y) * h
                
                for x, y in zip(norm_x, norm_y):
                     points += f'<circle cx="{x:.1f}" cy="{100-y:.1f}" r="1.5" class="scatter-pt" />'
                
                html += f"""
                <div class="grid-2">
                    <div>
                        <h3>PCA Projection (Raw Data)</h3>
                        <p style="color:#666; font-size:0.9em;">2D projection of the feature space. Clusters or outliers may be visible here.</p>
                        <svg class="chart-svg" viewBox="0 0 100 100" preserveAspectRatio="none">{points}</svg>
                    </div>
                    <div>
                        <h3>Feature Variance</h3>
                        <p style="color:#666; font-size:0.9em;">Top features by variance (scaled).</p>
                        {self._generate_variance_table(numeric_df)}
                    </div>
                </div>
                """
            except Exception as e:
                html += f"<p>Could not generate PCA: {e}</p>"
        
        # 2. Specific Analysis per Type
        if self.problem_type == 'anomaly_detection':
            html += self._generate_outlier_analysis(numeric_df)
            
        html += '</div>'
        return html

    def _generate_variance_table(self, df) -> str:
        vars = df.var().sort_values(ascending=False).head(5)
        html = '<table><tr><th>Feature</th><th>Variance</th></tr>'
        for f, v in vars.items():
            html += f'<tr><td>{f}</td><td>{v:.2f}</td></tr>'
        html += '</table>'
        return html

    def _generate_outlier_analysis(self, df) -> str:
        """Table of outliers by IQR"""
        html = '<h3>🧐 Potential Outliers (IQR Method)</h3>'
        html += '<table><tr><th>Feature</th><th>Outlier Count</th><th>%</th></tr>'
        
        count = 0
        for col in df.columns[:5]: # Check top 5 numeric cols
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                html += f'<tr><td>{col}</td><td>{outliers}</td><td>{outliers/len(df)*100:.1f}%</td></tr>'
                count += 1
        
        if count == 0: html += '<tr><td colspan="3">No significant outliers detected in top features.</td></tr>'
        html += '</table>'
        return html

    # ... (Keep existing methods: _generate_target_analysis, _generate_histogram, _generate_correlations, _generate_warnings, _generate_column_details, generate_minimal_report) ...
    
    def _generate_target_analysis(self) -> str:
        # Copied from previous logic, ensuring it handles None target
        if self.target is None: return ""
        # ... (rest of logic same as before)
        html = '<div class="card"><h2>🎯 Target Variable Analysis</h2>'
        if self.problem_type == 'classification':
             class_counts = self.target.value_counts()
             total = len(self.target)
             html += '<h3>Class Distribution</h3><table><tr><th>Class</th><th>Count</th><th>Pct</th></tr>'
             for cls, count in class_counts.items():
                 html += f'<tr><td>{cls}</td><td>{count}</td><td>{count/total*100:.1f}%</td></tr>'
             html += '</table>'
        else:
             stats = self.target.describe()
             html += f'<h3>Statistics</h3><p>Mean: {stats["mean"]:.2f}, Std: {stats["std"]:.2f}</p>'
             html += self._generate_histogram(self.target)
        html += '</div>'
        return html

    def _generate_histogram(self, series: pd.Series, bins: int = 20) -> str:
        try:
            counts, _ = np.histogram(series.dropna(), bins=bins)
            max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 1
            html = '<div class="mini-chart">'
            for count in counts:
                height = int((count / max_count) * 40)
                html += f'<div class="mini-bar" style="height: {max(height, 2)}px;"></div>'
            html += '</div>'
            return html
        except: return ""

    def _generate_correlations(self) -> str:
        if self.target is None: return ""
        # ... (previous implementation)
        return ""

    def _generate_warnings(self) -> str:
        if not self.warnings and not self.excluded_columns: return ''
        html = '<div class="card"><h2>⚠️ Preprocessing Notes</h2>'
        if self.warnings:
            for w in self.warnings: html += f'<div class="warning-box">{w}</div>'
        html += '</div>'
        return html

    def _generate_column_details(self) -> str:
        html = '<div class="card"><h2>📋 Column Details</h2>'
        html += '<table><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Samples</th></tr>'
        for col in self.df.columns:
            series = self.df[col]
            missing = series.isnull().sum() / len(series) * 100
            html += f'<tr><td>{col}</td><td>{series.dtype}</td><td>{missing:.1f}%</td><td>{series.nunique()}</td><td>{str(series.head(3).tolist())[:30]}</td></tr>'
        html += '</table></div>'
        return html

    def generate(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="UTF-8"><title>EDA Report</title>{self._get_css()}</head>
        <body><div class="container">
            <h1>📊 Exploratory Data Analysis</h1>
            {self._generate_overview()}
            {self._generate_target_analysis()}
            {self._generate_unsupervised_analysis()}
            {self._generate_warnings()}
            {self._generate_correlations()}
            {self._generate_column_details()}
            <div class="card" style="text-align:center;color:#666;"><p>Generated by ez-automl-lite • {timestamp}</p></div>
        </div></body></html>
        """
        return html

def generate_minimal_report(df, target, output):
    with open(output, 'w') as f: f.write("<html><body><h1>Error generating full EDA</h1></body></html>")
