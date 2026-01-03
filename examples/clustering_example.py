import pandas as pd
from sklearn.datasets import load_iris

from ez_automl_lite import AutoCluster


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

    # 3. Generate Reports
    eda_report_name = "iris_cluster_eda_report.html"
    ac.eda(df, eda_report_name)
    print(f"EDA Report generated: {eda_report_name}")

    training_report_name = "iris_cluster_training_report.html"
    ac.report(training_report_name)
    print(f"Clustering Training Report generated: {training_report_name}")

    # 4. Predict
    labels = ac.predict(df.head())
    print(f"Sample Cluster Assignments: {labels}")


if __name__ == "__main__":
    test_clustering()
