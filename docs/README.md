# Technical Documentation

This directory contains technical documentation for **auto-ml-lite**.

## Package Structure

The package is organized into the following modules:

- `src/auto_ml_lite/core/`: Logic for preprocessing, training, and model export.
- `src/auto_ml_lite/reports/`: Generators for premium HTML/CSS reports (EDA and Training).
- `src/auto_ml_lite/utils/`: Utility functions for automated detection and data handling.

## Key Concepts

### Automated Preprocessing
The `AutoPreprocessor` class handles:
- **Problem Detection**: Automatically identifies if the task is Regression or Classification.
- **Column Filtering**: Drops ID columns, constant features, and high-cardinality categorical data that won't help the model.
- **Encoding**: Implements robust categorical encoding and handles missing values.

### Premium Diagnostics
Reports are generated using pure HTML and CSS, making them extremely lightweight and portable without requiring external JavaScript libraries or internet access to render properly.

### ONNX Export
MODELS are converted to the Open Neural Network Exchange format, ensuring they can be run in production environments with minimal dependencies and high performance.
