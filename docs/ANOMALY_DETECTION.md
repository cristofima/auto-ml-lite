# 🚨 Anomaly Detection

Anomaly Detection is an *unsupervised* task (or semi-supervised) aimed at identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

## ✅ Supported Algorithms
`ez-automl-lite` supports robust outlier detection algorithms:

| Algorithm | ID | Description |
| :--- | :--- | :--- |
| **Isolation Forest** | `isolation_forest` | (Default) Isolates observations by randomly selecting a feature and split value. Anomalies are isolated faster (fewer splits). Efficient for high dimensions. |
| **Local Outlier Factor** | `lof` | Measures the local deviation of density of a given sample with respect to its neighbors. |
| **One-Class SVM** | `svm` | Captures the shape of the data distribution (frontier) and classifies points outside it as outliers. |

## 📊 Metrics & Outputs
- **Anomaly Label**: `-1` for anomalies, `1` for normal data.
- **Anomaly Score**: A continuous score indicating the degree of "abnormality". Lower scores (more negative) usually indicate higher abnormality.
- **Contamination**: The estimated proportion of outliers in the dataset (configurable, default `auto` or `0.05`).

## 📈 Reports

### 1. EDA Report (`.eda()`, Task: `anomaly_detection`)
Specialized EDA for identifying potential outliers before modeling:
- **Extreme Value Analysis (IQR)**: Automatically calculates potential outliers using the Interquartile Range (IQR) method for each feature.
- **Density Distributions**: Histograms showing the spread of data, highlighting tails where anomalies might reside.
- **Feature Variance**: Ranking features by variance.

### 2. Anomaly Report (`.report()`)
Results of the detection model:
- **Total Anomalies**: Count and percentage of contamination found.
- **PCA Visualization**: 2D Scatter plot highlighting the detected anomalies (usually in red) vs normal points (blue).
- **Score Distribution**: A density plot of the anomaly scores, helping you set a custom threshold if needed.
