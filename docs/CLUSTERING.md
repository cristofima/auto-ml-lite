# 🧩 Clustering

Clustering is an *unsupervised* learning task used to group similar data points together. It does not require labels.

## ✅ Supported Algorithms
`ez-automl-lite` automatically selects or tests the following algorithms:

| Algorithm | ID | Description |
| :--- | :--- | :--- |
| **K-Means** | `kmeans` | Partitioning method that separates data into K groups. Best for spherical clusters. |
| **Agglomerative** | `agglomerative` | Hierarchical clustering. Builds a tree of clusters. Good for smaller datasets or non-flat geometry. |
| **DBSCAN** | `dbscan` | Density-based clustering. Finds clusters of arbitrary shape and detects noise (outliers). **NEW**: Automatic `eps` selection via k-distance elbow detection. |

> **Auto Logic**: The system attempts to find the optimal number of clusters (`k`) for K-Means and Agglomerative by testing a range (e.g., 2 to 10) and maximizing the *Silhouette Score*. For DBSCAN, the `eps` parameter is automatically selected using k-distance graph elbow detection (requires `kneed` library).

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

## 🔧 DBSCAN Automatic Parameter Selection

DBSCAN requires two main parameters:
- **`eps`**: Maximum distance between two samples for them to be considered in the same neighborhood
- **`min_samples`**: Minimum number of points required to form a dense region (default: 5)

### Automatic `eps` Selection
The library now automatically selects an optimal `eps` value using the **k-distance elbow detection method**:

1. Computes k-NN distances for all points (where k = `min_samples`)
2. Sorts these distances in ascending order
3. Detects the "elbow" point (maximum curvature) using the Kneedle algorithm
4. Uses the distance at the elbow as the optimal `eps`

### Usage Examples

**Automatic eps selection (recommended):**
```python
from ez_automl_lite import AutoCluster

# eps will be automatically selected
cluster = AutoCluster(max_clusters=10)
cluster.fit(df, algorithm='dbscan')  # Auto-selects eps
```

**Manual eps override:**
```python
# Provide custom eps if you know the optimal value
cluster = AutoCluster(max_clusters=10)
cluster.fit(df, algorithm='dbscan', eps=0.3, min_samples=5)
```

**Requirements:**
To use automatic eps selection, install the optional clustering dependency:
```bash
pip install "ez-automl-lite[cluster]"
# or install kneed directly
pip install kneed
```

If `kneed` is not installed, the system will fall back to a default `eps=0.5` with a warning message.
