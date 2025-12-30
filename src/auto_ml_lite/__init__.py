"""
auto-ml-lite: A lightweight AutoML library for Python.
"""

import os
import pandas as pd
import joblib
from typing import Optional, Dict, Any

from auto_ml_lite.core.preprocessor import AutoPreprocessor
from auto_ml_lite.core.trainer import train_automl_model
from auto_ml_lite.core.exporter import export_model_to_onnx
from auto_ml_lite.reports.eda import generate_eda_report
from auto_ml_lite.reports.training import generate_training_report

__version__ = "0.1.0"

class AutoML:
    """
    Main entry point for auto-ml-lite.
    """
    
    def __init__(self, target: str, time_budget: int = 300):
        self.target = target
        self.time_budget = time_budget
        self.preprocessor = AutoPreprocessor(target)
        self.model = None
        self.metrics = None
        self.feature_importance = None
        self.problem_type = None
        self.dataset_info = {}
        self.job_id = "local_job"

    def fit(self, df: pd.DataFrame):
        """
        Train the AutoML model.
        """
        self.dataset_info = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'target_column': self.target
        }
        
        X_train, X_test, y_train, y_test, problem_type = self.preprocessor.preprocess(df)
        self.problem_type = problem_type
        self.dataset_info.update({
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
        })
        
        model, metrics, feature_importance, y_test_out, y_pred_out = train_automl_model(
            X_train, X_test, y_train, y_test,
            problem_type=problem_type,
            time_budget=self.time_budget
        )
        
        self.model = model
        self.metrics = metrics
        self.feature_importance = feature_importance
        self.y_test = y_test_out
        self.y_pred = y_pred_out
        self.X_sample = X_train.iloc[:1]
        
        return self

    def predict(self, df: pd.DataFrame):
        """
        Make predictions on new data.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Note: Preprocessing for prediction needs to be consistent
        # In this lite version, we assume the user pass same structure OR we need a transform method
        # For simplicity in this first version, we'll implement a basic transform in preprocessor
        
        # Simple transform logic (placeholder for actual implementation in preprocessor)
        X = df.copy()
        if self.target in X.columns:
            X = X.drop(columns=[self.target])
        
        # Categorical encoding
        for col, le in self.preprocessor.label_encoders.items():
            if col != '__target__' and col in X.columns:
                X[col] = X[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        # Ensure column order matches training
        X = X[self.preprocessor.feature_columns]
        
        # Fill missing
        X = X.fillna(0) # Simple fallback
        
        return self.model.predict(X)

    def report(self, output_path: str = "training_report.html"):
        """
        Generate training report.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        generate_training_report(
            output_path=output_path,
            job_id=self.job_id,
            problem_type=self.problem_type,
            metrics=self.metrics,
            feature_importance=self.feature_importance,
            training_config={'time_budget': self.time_budget},
            preprocessing_info={
                'feature_columns': self.preprocessor.feature_columns,
                'dropped_columns': self.preprocessor.dropped_columns,
                'feature_count': len(self.preprocessor.feature_columns)
            },
            dataset_info=self.dataset_info,
            y_test=self.y_test,
            y_pred=self.y_pred
        )

    def eda(self, df: pd.DataFrame, output_path: str = "eda_report.html"):
        """
        Generate EDA report.
        """
        generate_eda_report(df, self.target, output_path)

    def save(self, path: str):
        """
        Save the entire AutoML object.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a saved AutoML object.
        """
        return joblib.load(path)

    def export_onnx(self, path: str):
        """
        Export the trained model to ONNX.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return export_model_to_onnx(self.model, self.X_sample, path)
