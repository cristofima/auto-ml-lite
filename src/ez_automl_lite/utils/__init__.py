"""Shared utilities."""

from ez_automl_lite.utils.detection import (
    detect_problem_type,
    is_constant_column,
    is_high_cardinality_categorical,
    is_id_column,
)

__all__ = [
    "detect_problem_type",
    "is_constant_column",
    "is_high_cardinality_categorical",
    "is_id_column",
]
