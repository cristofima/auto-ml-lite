# Technical Documentation

This directory contains technical details for the **ez-automl-lite** engine.

## Core Capabilities

### 1. Supervised Learning (AutoML)
The `AutoML` class handles both **Regression** and **Classification**.
- **Engine**: Powered by FLAML for efficient hyperparameter tuning.
- **Diagnostics**: 
  - *Regression*: Residuals distribution histogram and Error/Relative-Error sample tables.
  - *Classification*: CSS heatmap-based Confusion Matrix and Class-wise metrics (Precision, Recall, F1).

### 2. Unsupervised Clustering (AutoCluster)
- **Algorithm**: MiniBatchKMeans for performance.
- **Automation**: Iterative K-search (default 2 to 10 clusters).
- **Selection Criteria**: Multi-metric evaluation using Silhouette and Calinski-Harabasz scores to find the "elbow" or optimal grouping.

### 3. Anomaly Detection (AutoAnomaly)
- **Algorithm**: Isolation Forest.
- **Reporting**: Rows with the highest anomaly scores are profiled and highlighted in the report with heat-intensity styling.

## Shared Infrastructure

### Premium Diagnostics
All reports are generated in `src/ez_automl_lite/reports/`. They use **zero external dependencies** (no Javascript libraries, no Google Fonts, no external CSS frameworks). This ensures:
1. **Security**: No scripts are executed in the browser.
2. **Speed**: Reports open instantly.
3. **Portability**: Reports render perfectly in offline environments or air-gapped systems.

### Job ID Tracking
Every instance of `AutoML`, `AutoCluster`, or `AutoAnomaly` generates a unique `job_id` (UUID4) unless a custom one is provided. This ID is embedded in all generated files for audit trailing.

### Auto-Preprocessor
Shared across modules to handle:
- Automated numeric scaling.
- Missing value imputation (mean/median/constant).
- Categorical encoding (LabelEncoding for targets, mapping for features).
- Outlier filtering and constant feature removal.
