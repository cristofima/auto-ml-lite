"""
Shared utilities for detection and analysis.
"""

import pandas as pd
import re


# Common patterns for ID/identifier columns (case insensitive)
ID_PATTERNS = [
    r'^id$',
    r'_id$',
    r'^id_',
    r'_id_',
    r'^uuid$',
    r'^guid$',
    r'order.*id',
    r'customer.*id',
    r'user.*id',
    r'transaction.*id',
    r'product.*id',
    r'session.*id',
    r'^index$',
    r'^row.*num',
    r'^serial',
    r'^record.*id',
]


def detect_problem_type(y: pd.Series) -> str:
    """
    Detect if problem is classification or regression.
    """
    # Guard against empty target
    if len(y) == 0:
        return 'classification'  # Default fallback
    
    # Non-numeric target is always classification
    if not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    
    n_unique = y.nunique()
    unique_ratio = n_unique / len(y) if len(y) > 0 else 0
    
    # Check if values are integer-like (no decimals)
    try:
        is_integer_like = (y.dropna() == y.dropna().astype(int)).all()
    except (ValueError, TypeError):
        is_integer_like = False
    
    # Classification criteria:
    # 1. Integer-like values with few unique values (typical class labels: 0,1,2,...)
    if is_integer_like and n_unique <= 10:
        return 'classification'
    
    # 2. Low cardinality with low ratio - likely categorical encoded as numbers
    if n_unique < 20 and unique_ratio < 0.05:
        return 'classification'
    
    return 'regression'


def is_id_column(col_name: str, series: pd.Series) -> bool:
    """
    Detect if a column is likely an ID/identifier column.
    """
    col_lower = col_name.lower().strip()
    
    # Check name patterns
    for pattern in ID_PATTERNS:
        if re.search(pattern, col_lower):
            return True
    
    # Check data characteristics for numeric columns
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        n_total = len(series)
        
        # If all values are unique and sequential, likely an ID
        if n_unique == n_total and n_total > 0:
            if series.dtype in ['int64', 'int32', 'int']:
                sorted_vals = series.sort_values()
                is_sequential = (sorted_vals.diff().dropna() == 1).all()
                if is_sequential:
                    return True
    
    # Check for string columns that look like IDs (high cardinality)
    if series.dtype == 'object':
        n_unique = series.nunique()
        n_total = len(series)
        
        # If almost all values are unique, likely an ID
        if n_total > 0 and n_unique / n_total > 0.95:
            sample = series.dropna().head(100)
            # Check if values look like codes/IDs (alphanumeric patterns)
            if sample.apply(lambda x: bool(re.match(r'^[A-Za-z0-9\-_]+$', str(x)))).mean() > 0.9:
                return True
    
    return False


def is_constant_column(series: pd.Series) -> bool:
    """Detect if a column has only one unique value."""
    return series.nunique() <= 1


def is_high_cardinality_categorical(series: pd.Series, threshold: float = 0.5) -> bool:
    """Detect categorical columns with too many unique values."""
    if series.dtype != 'object':
        return False
    
    n_unique = series.nunique()
    n_total = len(series)
    
    if n_total == 0:
        return False
        
    return n_unique / n_total > threshold
