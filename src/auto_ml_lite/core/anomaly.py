"""
Automated Anomaly Detection Module.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional
import time
import uuid
from auto_ml_lite.reports.anomaly_report import generate_anomaly_report

class AutoAnomaly:
    """
    Automated Anomaly Detection using Isolation Forest.
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42, job_id: Optional[str] = None):
        self.contamination = contamination
        self.random_state = random_state
        self.job_id = job_id or str(uuid.uuid4())
        self.model = None
        self.metrics = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.dataset_info = {}

    def fit(self, df: pd.DataFrame) -> 'AutoAnomaly':
        """
        Train the anomaly detection model.
        """
        print(f"Starting anomaly detection training (Contamination: {self.contamination})...")
        
        # Preprocessing: Numeric features only
        X = df.select_dtypes(include=[np.number]).copy()
        X = X.fillna(X.median())
        self.feature_columns = list(X.columns)
        
        X_scaled = self.scaler.fit_transform(X)
        
        start_time = time.time()
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Fit and predict labels (-1 for anomalies, 1 for normal)
        labels = self.model.fit_predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        execution_time = time.time() - start_time
        
        anomaly_count = int(np.sum(labels == -1))
        normal_count = int(np.sum(labels == 1))
        
        self.metrics = {
            'anomaly_count': anomaly_count,
            'normal_count': normal_count,
            'anomaly_percentage': (anomaly_count / len(df)) * 100,
            'execution_time': execution_time,
            'mean_anomaly_score': float(np.mean(scores)),
            'min_anomaly_score': float(np.min(scores)),
            'max_anomaly_score': float(np.max(scores))
        }
        
        # Store some stats for the report
        self.dataset_info = {
            'rows': len(df),
            'features': len(self.feature_columns)
        }
        
        # Store indices of top anomalies for report profiling
        anomaly_indices = np.where(labels == -1)[0]
        # Sort by score (lower is more anomalous)
        top_anomaly_indices = anomaly_indices[np.argsort(scores[anomaly_indices])][:10]
        self.metrics['top_anomalies_samples'] = df.iloc[top_anomaly_indices].to_dict('records')
        self.metrics['top_anomalies_scores'] = scores[top_anomaly_indices].tolist()

        print(f"Analysis complete. Found {anomaly_count} anomalies ({self.metrics['anomaly_percentage']:.2f}%).")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict if samples are anomalies.
        Returns: 1 for normal, -1 for anomalies.
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
            
        X = df[self.feature_columns].select_dtypes(include=[np.number]).copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores. Lower scores mean more anomalous.
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
            
        X = df[self.feature_columns].select_dtypes(include=[np.number]).copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def report(self, output_path: str = "anomaly_report.html"):
        """Generate anomaly detection report."""
        if self.model is None:
            raise ValueError("Model not fitted.")
            
        generate_anomaly_report(
            output_path=output_path,
            job_id=self.job_id,
            metrics=self.metrics,
            dataset_info=self.dataset_info
        )
