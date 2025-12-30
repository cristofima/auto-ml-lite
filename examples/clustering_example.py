import pandas as pd
from auto_ml_lite import AutoCluster
from sklearn.datasets import load_iris
import os

def test_clustering():
    print("\n--- Testing Automated Clustering on Iris Dataset ---")
    
    # 1. Load data
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. Initialize and Train
    ac = AutoCluster(max_clusters=6)
    print(f"Job ID: {ac.job_id}")
    
    ac.fit(df)
    
    print("\n--- Clustering Results ---")
    print(f"Optimal K: {ac.optimal_k}")
    print(f"Best Silhouette Score: {ac.metrics.get('best_silhouette'):.4f}")
    
    # 3. Generate Report
    report_name = "iris_cluster_report.html"
    ac.report(report_name)
    print(f"Clustering report generated: {report_name}")
    
    # 4. Predict
    labels = ac.predict(df.head())
    print(f"Sample Cluster Assignments: {labels}")

if __name__ == "__main__":
    test_clustering()
