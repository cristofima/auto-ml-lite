"""
Automated Clustering Module.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from typing import Optional
import time
import uuid
from ez_automl_lite.reports.cluster_report import generate_cluster_report
from ez_automl_lite.reports.eda import generate_eda_report

class AutoCluster:
    """
    Automated Clustering logic to find the optimal number of groups.
    Supports K-Means, Agglomerative Clustering, and DBSCAN.
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
        self.algorithm_name = "K-Means" # default
        self.knn_predictor = None

    def fit(self, df: pd.DataFrame, algorithm: str = 'auto') -> 'AutoCluster':
        """
        Search for the optimal number of clusters or fit specific algorithm.
        algorithm: 'auto', 'kmeans', 'agglomerative', 'dbscan'
        """
        print(f"Starting automated clustering (Max K: {self.max_clusters}, Algo: {algorithm})...")
        
        # Preprocessing: Fill numeric missing and scale
        X_raw = df.select_dtypes(include=[np.number]).copy()
        X_raw = X_raw.fillna(X_raw.mean())
        self.feature_columns = list(X_raw.columns)
        
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # --- Algorithm Selection ---
        if algorithm == 'auto':
            # Simple heuristic: Use KMeans for larger data, Agglomerative for small (< 2000 rows)
            algorithm = 'kmeans' if len(X_scaled) > 2000 else 'agglomerative'
        
        self.algorithm_name = algorithm.capitalize()
        
        start_time = time.time()
        
        best_silhouette = -1
        results = []
        best_model = None
        optimal_k = -1

        # --- DBSCAN Logic (Density based, no K search) ---
        if algorithm == 'dbscan':
            print("Running DBSCAN...")
            # Try a default eps or a small grid search could be implemented, keeping simple for Lite.
            # Defaulting to eps=0.5, min_samples=5
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters_ > 1:
                s_score = self._calculate_silhouette(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)
            else:
                s_score = -1
                ch_score = 0
            
            best_silhouette = s_score
            optimal_k = n_clusters_
            best_model = dbscan
            
            results.append({
                'k': n_clusters_,
                'silhouette': s_score,
                'calinski': ch_score,
                'inertia': 0 # DBSCAN doesn't utilize inertia
            })
            
        else:
            # --- K-Means / Agglomerative Logic (Iterate K) ---
            print(f"Running {algorithm} search K=2..{self.max_clusters}...")
            search_range = range(2, min(self.max_clusters + 1, len(df)))
            
            for k in search_range:
                if algorithm == 'agglomerative':
                    model = AgglomerativeClustering(n_clusters=k)
                    labels = model.fit_predict(X_scaled)
                    inertia = 0 # Agglomerative doesn't have inertia attribute directly matched to KMeans loss
                else: # kmeans
                    # Use MiniBatchKMeans for speed
                    model = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, n_init=3)
                    labels = model.fit_predict(X_scaled)
                    inertia = model.inertia_
                
                s_score = self._calculate_silhouette(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)
                
                results.append({
                    'k': k,
                    'silhouette': s_score,
                    'calinski': ch_score,
                    'inertia': inertia
                })
                
                print(f"  K={k}: Silhouette={s_score:.4f}, Calinski={ch_score:.2f}")
                
                if s_score > best_silhouette:
                    best_silhouette = s_score
                    optimal_k = k
                    best_model = model

        total_time = time.time() - start_time
        
        self.best_model = best_model
        self.optimal_k = optimal_k
        self.metrics = {
            'optimal_k': self.optimal_k,
            'best_silhouette': best_silhouette,
            'search_results': results,
            'execution_time': total_time,
            'algorithm': self.algorithm_name
        }
        
        # --- Train KNN for Fallback Prediction (DBSCAN/Agglomerative) ---
        if algorithm in ['dbscan', 'agglomerative'] and best_model is not None:
            if hasattr(best_model, 'labels_'):
                 try:
                     print("Training fallback KNN predictor for clustering inference...")
                     self.knn_predictor = KNeighborsClassifier(n_neighbors=1)
                     self.knn_predictor.fit(X_scaled, best_model.labels_)
                 except Exception as e:
                     print(f"Warning: Could not train fallback predictor: {e}")
        
        # --- PCA Transformation for Visualization ---
        print("Calculating PCA for visualization...")
        self._calculate_pca(X_scaled)
        
        print(f"Optimal clusters ({self.algorithm_name}): {self.optimal_k} (Silhouette: {best_silhouette:.4f})")
        return self

    def _calculate_silhouette(self, X, labels):
        if len(set(labels)) < 2: return -1
        if len(X) > 5000:
            indices = np.random.choice(len(X), 5000, replace=False)
            return silhouette_score(X[indices], labels[indices])
        return silhouette_score(X, labels)

    def _calculate_pca(self, X_scaled):
        """Reduce dimensions to 2D for reporting"""
        try:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)
            
            # Predict labels for the whole dataset with best model
            if hasattr(self.best_model, 'predict'):
                labels = self.best_model.predict(X_scaled)
            elif hasattr(self.best_model, 'labels_'): # DBSCAN / Agglomerative
                labels = self.best_model.labels_
            else:
                labels = np.zeros(len(X_scaled))
                
            # Sample for json serialization if too large
            limit = 1000
            if len(coords) > limit:
                indices = np.random.choice(len(coords), limit, replace=False)
                coords = coords[indices]
                labels = labels[indices]
                
            self.dataset_info['pca_data'] = [
                {'x': float(c[0]), 'y': float(c[1]), 'cluster': int(l)} 
                for c, l in zip(coords, labels)
            ]
            self.dataset_info['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
            
        except Exception as e:
            print(f"PCA calculation failed: {e}")
            self.dataset_info['pca_data'] = []

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Assign clusters to new data."""
        if self.best_model is None:
            raise ValueError("Model not fitted.")
            
        X = df[self.feature_columns].select_dtypes(include=[np.number]).copy()
        X = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.best_model, 'predict'):
            return self.best_model.predict(X_scaled)
        elif hasattr(self.best_model, 'fit_predict'):
            # Some models like DBSCAN don't have predict() for new data in sklearn easily (transductive)
            # We would need KNN to assign to nearest core sample, but for "Lite" we might just warn
            print("Warning: Model does not support direct prediction (e.g. DBSCAN/Agglomerative). Returning fit labels if appropriate or error.")
            # For simplicity, if we are predicting on training set (which is common immediately), return labels_
            # But this method signature implies new data.
            # We will implement a simple Nearest Centroid for Agglomerative/DBSCAN fallback
            return self._predict_nearest(X_scaled)
            
        return np.zeros(len(df))

    def _predict_nearest(self, X_scaled):
        # Fallback prediction using centroids of training clusters
        if hasattr(self.best_model, 'labels_'):
            labels = self.best_model.labels_
        else:
            raise ValueError("Model does not have labels_ attribute.")
        
        # We need the original training data to train the KNN
        # But we don't store X_scaled in self (to save memory usually).
        # PROPOSAL: We should store X_scaled (or a subset) if we want to enable prediction for these models.
        # Or we rely on the user passing the same data structure and we can't easily do it without training data.
        # Wait, in 'fit' we have X_scaled. We should train the KNN *there* and store it.
        
        if not hasattr(self, 'knn_predictor') or self.knn_predictor is None:
             raise ValueError("Prediction fallback model not available. Ensure fit() was called.")
             
        return self.knn_predictor.predict(X_scaled)

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

    def eda(self, df: pd.DataFrame, output_path: str = "cluster_eda.html"):
        """Generate EDA report for the clustering dataset."""
        generate_eda_report(df, target_column=None, output_path=output_path, task_type='clustering')

