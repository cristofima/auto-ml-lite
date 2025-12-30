# 🚀 auto-ml-lite

A lightweight, serverless-optimized AutoML library for Python. Build, evaluate, and export high-performance machine learning models with just 3 lines of code.

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **3-Line API**: Designed for simplicity and speed.
- **Serverless-First**: Optimized for AWS Lambda/Azure Functions and low-memory environments.
- **Premium Reports**: Professional HTML/CSS reports for EDA and Training diagnostics (no external JS dependencies).
- **Automated Preprocessing**: Smart detection of problem types, missing value imputation, and categorical encoding.
- **ONNX Export**: One-click export for cross-platform deployment.
- **Modern Stack**: Powered by FLAML, LightGBM, XGBoost, and Scikit-Learn.

---

## 📦 Installation

Install the core package:
```bash
pip install auto-ml-lite
```

Install with all features (Reporting & ONNX Export):
```bash
pip install "auto-ml-lite[all]"
```

---

## 🚀 Quick Start

### Regression Example
```python
from auto_ml_lite import AutoML
import pandas as pd
from sklearn.datasets import load_diabetes

# 1. Load your data
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 2. Fit the model (The "3-Line" magic)
aml = AutoML(target="target", time_budget=60)
aml.fit(df)

# 3. Generate a premium report
aml.report("training_results.html")
```

### Classification Example
```python
from auto_ml_lite import AutoML
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
aml = AutoML(target="target", time_budget=30).fit(pd.DataFrame(data.data, columns=data.feature_names).assign(target=data.target))
aml.report("classification_diagnostics.html")
```

---

## 🛠️ Key Capabilities

### 🔍 Automated EDA
Generate comprehensive Exploratory Data Analysis reports before training:
```python
aml.eda(df, output_path="eda_report.html")
```

### 📉 Diagnostic Visualizations
- **Regression**: Residual distribution histograms and actual vs. predicted tables.
- **Classification**: Visual Confusion Matrices (CSS heatmaps) and class-wise performance breakdown.

### 🌐 Cross-Platform Export
Export your model to ONNX format for high-speed inference in any language:
```python
aml.export_onnx("model.onnx")
```

---

## 🗺️ Roadmap

- [x] Core Package Refactor
- [x] Premium Reports (CSS-only)
- [x] ONNX Export Native Support
- [ ] **AutoCluster**: Automated unsupervised clustering.
- [ ] **AutoAnomaly**: Advanced outlier and anomaly detection.
- [ ] PyPI Automated Release Workflow.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created by [Cristopher Coronado](https://github.com/cristofima)*
