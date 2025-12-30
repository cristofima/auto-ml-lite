# ez-automl-lite

A lightweight AutoML library for Python, optimized for simplicity and cross-platform deployment.

## Features

- **3-Line API**: Train, evaluate, and predict with minimal code.
- **Auto-Preprocessing**: Automatic handling of missing values, categorical encoding, and feature selection.
- **FLAML Backend**: Powered by Microsoft's FLAML for fast and efficient hyperparameter tuning.
- **ONNX Export**: Built-in support for exporting models to ONNX for production inference.
- **Visual Reports**: Generate comprehensive EDA and Training reports in HTML.

## Installation

```bash
pip install ez-automl-lite
```

For ONNX support:
```bash
pip install ez-automl-lite[onnx]
```

## Quick Start

```python
from ez_automl_lite import AutoML
import pandas as pd

# 1. Load data
df = pd.read_csv("your_data.csv")

# 2. Initialize and Train
aml = AutoML(target="target_column", time_budget=60)
aml.fit(df)

# 3. Predict and Report
predictions = aml.predict(new_df)
aml.report("training_report.html")
aml.export_onnx("model.onnx")
```

## Running on AWS Fargate

This package is designed to run efficiently on AWS Fargate Spot instances. The core logic handles training and report generation, while orchesration can be handled by AWS Batch or Step Functions.
