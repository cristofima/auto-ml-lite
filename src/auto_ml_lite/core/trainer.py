"""AutoML model training using FLAML."""

import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from typing import Dict, Tuple
import time


def train_automl_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
    time_budget: int = 300
) -> Tuple[AutoML, Dict, Dict, pd.Series, np.ndarray]:
    """
    Train AutoML model using FLAML
    """
    print(f"Training {problem_type} model with FLAML...")
    print(f"Time budget: {time_budget} seconds")
    
    # Initialize AutoML
    automl = AutoML()
    
    # Configure settings based on problem type
    if problem_type == 'classification':
        task = 'classification'
        # Detect if multiclass classification
        n_classes = y_train.nunique()
        print(f"Number of classes: {n_classes}")
        is_multiclass = n_classes > 2
        metric = 'accuracy' if is_multiclass else 'f1'
    else:
        task = 'regression'
        metric = 'r2'
    
    print(f"Using metric: {metric}")
    
    # Start training
    start_time = time.time()
    
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task,
        metric=metric,
        time_budget=time_budget,
        estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'histgb'],
        verbose=1,
        log_file_name='flaml_training.log',
        early_stop=True,
        retrain_full=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best model: {automl.best_estimator}")
    
    # Make predictions
    y_pred = automl.predict(X_test)
    
    # Calculate metrics
    if problem_type == 'classification':
        metrics = calculate_classification_metrics(y_test, y_pred)
    else:
        metrics = calculate_regression_metrics(y_test, y_pred)
    
    metrics['training_time'] = training_time
    metrics['best_estimator'] = automl.best_estimator
    
    # Get feature importance
    feature_importance = get_feature_importance(automl, X_train.columns)
    
    return automl, metrics, feature_importance, y_test, y_pred


def calculate_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    }
    return metrics


def calculate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        'r2_score': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mean_absolute_error(y_true, y_pred))
    }
    return metrics


def get_feature_importance(model: AutoML, feature_names: pd.Index) -> Dict[str, float]:
    """Extract feature importance from the model"""
    try:
        best_model = model.model
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = dict(zip(feature_names, [float(x) for x in importances.tolist()]))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if importances is not None:
                feature_importance = dict(zip(feature_names, [float(x) for x in importances.tolist()]))
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        equal_importance = 1.0 / len(feature_names) if len(feature_names) > 0 else 0
        return {name: float(equal_importance) for name in feature_names}
        
    except Exception as e:
        print(f"Error extracting feature importance: {str(e)}")
        equal_importance = 1.0 / len(feature_names) if len(feature_names) > 0 else 0
        return {name: float(equal_importance) for name in feature_names}
