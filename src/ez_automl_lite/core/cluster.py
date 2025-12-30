"""
Automated Clustering Module.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from typing import Optional
import time
import uuid
from ez_automl_lite.reports.cluster_report import generate_cluster_report

class AutoCluster:
    """
    Automated Clustering logic to find the optimal number of groups.
    """
    
    def __init__(self, max_clusters: int = 10, random_state: int = 42, job_id: Optional[str] = None):
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.job_id = job_id or str(uuid.uuid4())
        self.best_model = None
        self.optimal_k = None
        self.metrics = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.dataset_info = {}

    def fit(self, df: pd.DataFrame) -> 'AutoCluster':
        """
        Search for the optimal number of clusters using Silhouette and Calinski-Harabasz scores.
        """
        print(f"Starting automated clustering (Max K: {self.max_clusters})...")
        
        # Preprocessing: Fill numeric missing and scale
        X = df.select_dtypes(include=[np.number]).copy()
        X = X.fillna(X.mean())
        self.feature_columns = list(X.columns)
        
        X_scaled = self.scaler.fit_transform(X)
        
        best_silhouette = -1
        results = []
        
        # Search range from 2 to max_clusters
        search_range = range(2, min(self.max_clusters + 1, len(df)))
        
        start_time = time.time()
        
        for k in search_range:
            # Use MiniBatchKMeans for speed in larger datasets
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, n_init=3)
            labels = kmeans.fit_predict(X_scaled)
            
            # Silhouette Score (higher is better)
            # Sample for silhouete if dataset is too large to keep it fast
            if len(X_scaled) > 5000:
                indices = np.random.choice(len(X_scaled), 5000, replace=False)
                s_score = silhouette_score(X_scaled[indices], labels[indices])
            else:
                s_score = silhouette_score(X_scaled, labels)
                
            ch_score = calinski_harabasz_score(X_scaled, labels)
            
            results.append({
                'k': k,
                'silhouette': s_score,
                'calinski': ch_score
            })
            
            print(f"  K={k}: Silhouette={s_score:.4f}, Calinski={ch_score:.2f}")
            
            if s_score > best_silhouette:
                best_silhouette = s_score
                self.optimal_k = k
                self.best_model = kmeans
        
        total_time = time.time() - start_time
        
        self.metrics = {
            'optimal_k': self.optimal_k,
            'best_silhouette': best_silhouette,
            'search_results': results,
            'execution_time': total_time
        }
        
        print(f"Optimal clusters found: {self.optimal_k} (Silhouette: {best_silhouette:.4f})")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Assign clusters to new data."""
        if self.best_model is None:
            raise ValueError("Model not fitted.")
            
        X = df[self.feature_columns].select_dtypes(include=[np.number]).copy()
        X = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)

    def report(self, output_path: str = "cluster_report.html"):
        """Generate clustering report."""
        if self.best_model is None:
            raise ValueError("Model not fitted.")
            
        generate_cluster_report(
            output_path=output_path,
            job_id=self.job_id,
            metrics=self.metrics,
            dataset_info=self.dataset_info
        )
