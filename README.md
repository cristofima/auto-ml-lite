# AutoML Lite

A lightweight, serverless-optimized AutoML library for Python.

## Project Structure

- `src/auto_ml_lite/`: Core package source code.
- `docs/`: Project documentation and analysis.
- `test_package.py`: Simple test script to verify library functionality.

## Roadmap

- [x] Refactor core logic into a reusable package.
- [x] Streamline API to 3 lines of code.
- [x] Remove legacy backend/api orchestration.
- [ ] Add Clustering support.
- [ ] Add Anomaly Detection support.
- [ ] Implement automated PyPI publishing workflow.

## Development

To install dependencies for development:

```bash
pip install -e .[all]
```

## Usage

```python
from auto_ml_lite import AutoML
import pandas as pd

aml = AutoML(target="price", time_budget=60)
aml.fit(df)
aml.report("report.html")
```
