# 📚 ez-automl-lite Documentation

Welcome to the **ez-automl-lite** documentation. This framework makes Machine Learning accessible and efficient using the power of FLAML.

## 🚀 Quick Start
See the [Main README](../README.md) for installation and basic usage.

## 🤖 Supported Tasks

`ez-automl-lite` supports four main types of analysis. Click on each section for detailed information on algorithms, metrics, and reports.

| Task | Description | Example Use Cases |
| :--- | :--- | :--- |
| [**Classification**](./CLASSIFICATION.md) | Predict categories (Yes/No, Spam/Ham). | Customer Churn, Fraud Detection, Image Class. |
| [**Regression**](./REGRESSION.md) | Predict continuous numbers. | House Prices, Sales Forecasting, Temperature. |
| [**Clustering**](./CLUSTERING.md) | Group similar items (Unsupervised). | Customer Segmentation, Market Basket Analysis. |
| [**Anomaly Detection**](./ANOMALY_DETECTION.md) | Find rare events/outliers (Unsupervised). | Intrusion Detection, Defect Detection, Fraud. |

## 📊 Core Features

### 1. Smart Algorithm Selection
The system automatically selects the best model (e.g., XGBoost, LightGBM, Random Forest) and hyperparameters within your time budget.

### 2. Automated Reports
Generate "premium-feel" HTML reports with a single line of code:
- **`model.eda(df)`**: Exploratory Data Analysis (before training).
- **`model.report()`**: Detailed Training Performance (after training).

### 3. Deployment Ready
- **ONNX Export**: All models can be exported to ONNX format for high-speed inference.
- **Docker Support**: Ready-to-use Dockerfile for easy deployment.

## 📂 Project Structure

- `src/ez_automl_lite/`: Source code.
- `examples/`: Ready-to-run Python scripts for each task.
