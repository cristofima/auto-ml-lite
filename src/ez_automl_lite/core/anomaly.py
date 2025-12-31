"""
Automated Anomaly Detection Module.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional
import time
import uuid
from ez_automl_lite.reports.anomaly_report import generate_anomaly_report
from ez_automl_lite.reports.eda import generate_eda_report

class AutoAnomaly:
    """
    Automated Anomaly Detection using Isolation Forest, LOF, or OneClassSVM.
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
        self.algorithm_name = "Isolation Forest"

    def fit(self, df: pd.DataFrame, algorithm: str = 'auto') -> 'AutoAnomaly':
        """
        Train the anomaly detection model.
        algorithm: 'auto', 'isolation_forest', 'lof', 'svm'
        """
        print(f"Starting anomaly detection training (Contamination: {self.contamination}, Algo: {algorithm})...")
        
        # Preprocessing: Numeric features only
        X_raw = df.select_dtypes(include=[np.number]).copy()
        X_raw = X_raw.fillna(X_raw.median())
        self.feature_columns = list(X_raw.columns)
        
        X_scaled = self.scaler.fit_transform(X_raw)
        
        if algorithm == 'auto':
            # Default to IsolationForest as it is robust and fast
            algorithm = 'isolation_forest'
            
        self.algorithm_name = algorithm.replace('_', ' ').title()
        
        start_time = time.time()
        
        if algorithm == 'lof':
            # novelty=True allows predict() on new data
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20
            )
        elif algorithm == 'svm':
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        else: # isolation_forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100
            )
            
        # Fit logic
        self.model.fit(X_scaled)
        
        # Predict labels (-1 for anomalies, 1 for normal)
        labels = self.model.predict(X_scaled)
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
            'max_anomaly_score': float(np.max(scores)),
            'algorithm': self.algorithm_name
        }
        
        # Store some stats for the report
        self.dataset_info = {
            'rows': len(df),
            'features': len(self.feature_columns)
        }
        
        # Store indices of top anomalies for report profiling
        anomaly_indices = np.where(labels == -1)[0]
        # Sort by score (lower is more anomalous)
        if len(anomaly_indices) > 0:
            top_anomaly_indices = anomaly_indices[np.argsort(scores[anomaly_indices])][:10]
            self.metrics['top_anomalies_samples'] = df.iloc[top_anomaly_indices].to_dict('records')
            self.metrics['top_anomalies_scores'] = scores[top_anomaly_indices].tolist()
        else:
            self.metrics['top_anomalies_samples'] = []
            self.metrics['top_anomalies_scores'] = []

        # --- PCA Transformation for Visualization ---
        print("Calculating PCA for visualization...")
        self._calculate_pca(X_scaled, labels)

        print(f"Analysis complete. Found {anomaly_count} anomalies ({self.metrics['anomaly_percentage']:.2f}%).")
        return self

    def _calculate_pca(self, X_scaled, labels):
        """Reduce dimensions to 2D for reporting"""
        try:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)
            
            # Sample for json serialization if too large
            limit = 1000
            if len(coords) > limit:
                indices = np.random.choice(len(coords), limit, replace=False)
                coords = coords[indices]
                labels = labels[indices]
                
            self.dataset_info['pca_data'] = [
                {'x': float(c[0]), 'y': float(c[1]), 'label': int(l)} 
                for c, l in zip(coords, labels)
            ]
            
        except Exception as e:
            print(f"PCA calculation failed: {e}")
            self.dataset_info['pca_data'] = []

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

    def eda(self, df: pd.DataFrame, output_path: str = "anomaly_eda.html"):
        """Generate EDA report for the anomaly detection dataset."""
        generate_eda_report(df, target_column=None, output_path=output_path, task_type='anomaly_detection')

