"""AutoML model training using FLAML."""

import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
    log_loss, roc_auc_score, mean_absolute_percentage_error
)
from typing import Dict, Tuple, Optional
import time


def train_automl_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
    time_budget: Optional[int] = None
) -> Tuple[AutoML, Dict, Dict, pd.Series, np.ndarray, Optional[np.ndarray]]:
    """
    Train AutoML model using FLAML with dynamic settings.
    Returns: automl, metrics, feature_importance, y_test, y_pred, y_pred_proba
    """
    print(f"Training {problem_type} model with FLAML...")
    
    # --- Dynamic Time Budget ---
    if time_budget is None:
        rows = len(X_train)
        cols = X_train.shape[1]
        cells = rows * cols
        
        if cells < 100000:       # < 100k cells
            time_budget = 60
        elif cells < 1000000:    # < 1M cells
            time_budget = 180
        else:                    # > 1M cells
            time_budget = 300
            
        print(f"Dynamic time budget calculated: {time_budget}s (Cells: {cells})")
    else:
        print(f"Time budget: {time_budget} seconds")
    
    # Initialize AutoML
    automl = AutoML()
    
    # --- Smart Algorithm Selection ---
    estimator_list = []
    
    # Configure settings based on problem type
    if problem_type == 'classification':
        task = 'classification'
        # Detect if multiclass classification
        n_classes = y_train.nunique()
        print(f"Number of classes: {n_classes}")
        is_multiclass = n_classes > 2
        metric = 'accuracy' if is_multiclass else 'f1'
        
        # Classification algorithms
        estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree']
        
        # Add Linear models if dataset is not huge (slow on large data)
        if len(X_train) < 100000:
            estimator_list.extend(['lrl1', 'lrl2']) # Logistic Regression variants
            
    else:
        task = 'regression'
        metric = 'r2'
        
        # Regression algorithms
        estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree', 'histgb']
    
    print(f"Using metric: {metric}")
    print(f"Algorithms: {estimator_list}")
    
    # Start training
    start_time = time.time()
    
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task,
        metric=metric,
        time_budget=time_budget,
        estimator_list=estimator_list,
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
    y_pred_proba = None
    
    # Calculate metrics
    if problem_type == 'classification':
        # Get probabilities for advanced metrics
        try:
            y_pred_proba = automl.predict_proba(X_test)
        except Exception:
            y_pred_proba = None
            
        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
    else:
        metrics = calculate_regression_metrics(y_test, y_pred)
    
    metrics['training_time'] = training_time
    metrics['best_estimator'] = automl.best_estimator
    metrics['time_budget'] = time_budget
    metrics['estimator_list'] = estimator_list
    
    # Get feature importance
    feature_importance = get_feature_importance(automl, X_train.columns)
    
    return automl, metrics, feature_importance, y_test, y_pred, y_pred_proba


def calculate_classification_metrics(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate classification metrics including LogLoss and ROC-AUC"""
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    }
    
    # Advanced metrics if probabilities are available
    if y_pred_proba is not None:
        try:
            # check if multiclass
            n_classes = len(np.unique(y_true))
            if n_classes > 2:
                metrics['log_loss'] = float(log_loss(y_true, y_pred_proba, labels=np.unique(y_true)))
                # For multiclass ROC-AUC, need to handle 'ovr'
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
            else:
                # Binary case: use probability of positive class
                if y_pred_proba.shape[1] == 2:
                    pos_probs = y_pred_proba[:, 1]
                    metrics['log_loss'] = float(log_loss(y_true, y_pred_proba))
                    metrics['roc_auc'] = float(roc_auc_score(y_true, pos_probs))
        except Exception as e:
            print(f"Warning: Could not calculate advanced metrics: {e}")
            
    return metrics


def calculate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics including MAPE"""
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        'r2_score': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mean_absolute_error(y_true, y_pred))
    }
    
    # MAPE can fail if y_true contains zeros
    try:
        if not (y_true == 0).any():
            metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
        else:
            print("Warning: MAPE not calculated due to zero values in y_true")
    except Exception as e:
        print(f"Warning: Could not calculate MAPE: {e}")
    
    return metrics


def get_feature_importance(model: AutoML, feature_names: pd.Index) -> Dict[str, float]:
    """Extract feature importance from the model"""
    try:
        best_model = model.model
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = dict(zip(feature_names, [float(x) for x in importances.tolist()], strict=True))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if importances is not None:
                feature_importance = dict(zip(feature_names, [float(x) for x in importances.tolist()]))
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # For linear models like LogisticRegression, importances are coefficients
        if hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_)
            if importances.ndim > 1:
                importances = np.mean(importances, axis=0) # Average over classes for multiclass
            feature_importance = dict(zip(feature_names, [float(x) for x in importances.tolist()]))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        equal_importance = 1.0 / len(feature_names) if len(feature_names) > 0 else 0
        return {name: float(equal_importance) for name in feature_names}
        
    except Exception as e:
        print(f"Error extracting feature importance: {str(e)}")
        equal_importance = 1.0 / len(feature_names) if len(feature_names) > 0 else 0
        return {name: float(equal_importance) for name in feature_names}
