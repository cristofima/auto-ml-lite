"""ONNX model exporter for trained sklearn/FLAML models."""

from typing import Any

import numpy as np
import pandas as pd


# ONNX imports
try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# LightGBM ONNX converter
try:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import (
        FloatTensorType as LGBMFloatTensorType,
    )

    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False


def _is_lightgbm_model(model: Any) -> bool:
    """Check if the model is a LightGBM model."""
    model_type: str = type(model).__name__
    return model_type in ("LGBMClassifier", "LGBMRegressor", "LGBMRanker")


def _convert_lightgbm_to_onnx(model: Any, n_features: int, output_path: str) -> bool:
    """Convert LightGBM model to ONNX using onnxmltools."""
    if not ONNXMLTOOLS_AVAILABLE:
        print("Warning: onnxmltools not available. LightGBM ONNX export skipped.")
        return False

    try:
        initial_types = [("input", LGBMFloatTensorType([None, n_features]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=15)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return True
    except Exception as e:
        print(f"Warning: LightGBM ONNX conversion failed: {e}")
        return False


def _extract_sklearn_model(model: Any) -> Any:
    """Extract the underlying sklearn/lightgbm model from FLAML wrappers."""
    if hasattr(model, "model"):
        inner_model = model.model
    else:
        inner_model = model

    if hasattr(inner_model, "estimator"):
        return inner_model.estimator

    if hasattr(inner_model, "_model"):
        return inner_model._model

    return inner_model


def export_model_to_onnx(
    model: Any,
    X_sample: pd.DataFrame,
    output_path: str,
) -> bool:
    """Export a trained sklearn model to ONNX format."""
    if not ONNX_AVAILABLE:
        print("Warning: skl2onnx not available. ONNX export skipped.")
        return False

    try:
        sklearn_model = _extract_sklearn_model(model)

        if isinstance(X_sample, pd.DataFrame):
            X_array = X_sample.values.astype(np.float32)
        else:
            X_array = np.array(X_sample).astype(np.float32)

        if len(X_array) > 1:
            X_array = X_array[:1]

        n_features = X_array.shape[1]

        if _is_lightgbm_model(sklearn_model):
            return _convert_lightgbm_to_onnx(sklearn_model, n_features, output_path)

        initial_types = [("input", FloatTensorType([None, n_features]))]
        options = None
        if hasattr(sklearn_model, "predict_proba"):
            options = {type(sklearn_model): {"zipmap": False}}

        onnx_model = to_onnx(
            sklearn_model,
            X_array,
            initial_types=initial_types,
            target_opset=15,
            options=options,
        )

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        return True

    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        return False
