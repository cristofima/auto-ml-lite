# 🚀 ez-automl-lite

A lightweight, serverless-optimized AutoML library for Python. Build, evaluate, and export high-performance machine learning models with just 3 lines of code.

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **3-Line API**: Designed for simplicity and speed across 4 different ML tasks.
- **Serverless-First**: Optimized for AWS Lambda/Azure Functions and low-memory environments.
- **Premium Reports**: Professional HTML/CSS reports for all tasks (No external JS or Internet required).
- **Comprehensive Analytics**: Supports Regression, Classification, Clustering, and Anomaly Detection.
- **ONNX Export**: One-click export for cross-platform deployment.

---

## 📦 Installation

```bash
pip install "ez-automl-lite[all]"
```

---

## 🚀 The 4 Core Modules

### 1. Regression
Automated training with residual analysis and error diagnostics.
```python
from ez_automl_lite import AutoML
aml = AutoML(target="target").fit(df)
aml.report("regression_report.html")
```

### 2. Classification
Visual Confusion Matrices and detailed class-wise performance metrics.
```python
from ez_automl_lite import AutoML
aml = AutoML(target="label").fit(df)
aml.report("classification_report.html")
```

### 3. Clustering (Unsupervised)
Automated optimal K-search using Silhouette and Calinski-Harabasz scores.
```python
from ez_automl_lite import AutoCluster
ac = AutoCluster(max_clusters=8).fit(df)
ac.report("clustering_report.html")
```

### 4. Anomaly Detection
Profile-based detection using Isolation Forest with detailed sample analysis.
```python
from ez_automl_lite import AutoAnomaly
aa = AutoAnomaly(contamination=0.05).fit(df)
aa.report("anomaly_report.html")
```

---

## 📂 Examples & Scripts

Check the `examples/` directory for full implementation scripts:
- `examples/regression_example.py`
- `examples/classification_example.py`
- `examples/clustering_example.py`
- `examples/anomaly_example.py`

---

## 🛠️ Performance & Export

- **ONNX Export**: Cross-platform models in one line: `aml.export_onnx("model.onnx")`.
- **EDA**: Generate pre-training analysis: `aml.eda(df, "eda.html")`.
- **UUIDs**: Every training session generates a unique ID for easy tracking.

---

## 🗺️ Roadmap

- [x] Core Package Refactor
- [x] Premium CSS-only Reports
- [x] AutoCluster & AutoAnomaly implementation
- [x] Cross-platform ONNX support
- [ ] PyPI Automated Release Workflow

---

## 🤝 Contributing & License

Created by [Cristopher Coronado](https://github.com/cristofima). 
Distributed under the MIT License.
