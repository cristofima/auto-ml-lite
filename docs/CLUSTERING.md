# 🧩 Clustering

Clustering is an *unsupervised* learning task used to group similar data points together. It does not require labels.

## ✅ Supported Algorithms
`ez-automl-lite` automatically selects or tests the following algorithms:

| Algorithm | ID | Description |
| :--- | :--- | :--- |
| **K-Means** | `kmeans` | Partitioning method that separates data into K groups. Best for spherical clusters. |
| **Agglomerative** | `agglomerative` | Hierarchical clustering. Builds a tree of clusters. Good for smaller datasets or non-flat geometry. |
| **DBSCAN** | `dbscan` | Density-based clustering. Finds clusters of arbitrary shape and detects noise (outliers). |

> **Auto Logic**: The system attempts to find the optimal number of clusters (`k`) for K-Means and Agglomerative by testing a range (e.g., 2 to 10) and maximizing the *Silhouette Score*.

## 📊 Metrics (Evaluation)
Since there are no "true" labels, we use internal validation metrics:

- **Silhouette Score**: Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Range: -1 to 1 (Higher is better).
- **Calinski-Harabasz Index**: Ratio of the sum of between-clusters dispersion and of within-cluster dispersion. (Higher is better).
- **Inertia**: Sum of squared distances of samples to their closest cluster center (Elbow Method).

## 📈 Reports

### 1. EDA Report (`.eda()`, Task: `clustering`)
Specialized EDA for clustering tasks:
- **PCA Projection (Raw)**: A 2D scatter plot of the raw data (projected via PCA) to visually estimate if natural clusters exist.
- **Feature Variance**: Table showing features with the highest variance, which often drive clustering results.
- **Boxplots**: Distribution of top features to check for spread and scale.

### 2. Clustering Report (`.report()`)
Results of the clustering process:
- **Optimal K**: The automatically determined best number of clusters.
- **Cluster Visualization (PCA)**: A 2D plot showing the final assigned clusters.
- **Elbow Curve**: (For K-Means) Visualization of Inertia vs K to justify the choice of clusters.
- **Cluster Centers/Profile**: (Coming soon) Summary of what defines each cluster.
