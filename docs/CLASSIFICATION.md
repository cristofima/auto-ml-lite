# 🎯 Classification

Classification is a supervised learning task where the goal is to predict the categorical class labels of new instances based on past observations.

## ✅ Supported Algorithms
`ez-automl-lite` uses FLAML to automatically select the best algorithm and hyperparameters. The following algorithms are supported:

| Algorithm | ID | Description |
| :--- | :--- | :--- |
| **LightGBM** | `lgbm` | Faster and lighter version of Gradient Boosting. Excellent for large datasets. |
| **XGBoost** | `xgboost` | Powerful and widely used Gradient Boosting implementation. |
| **Random Forest** | `rf` | Ensemble of decision trees. Robust and less prone to overfitting. |
| **Extra Trees** | `extra_tree` | Extremely Randomized Trees, similar to RF but often faster. |
| **Logistic Regression** | `lrl1`, `lrl2` | Linear model using L1 or L2 regularization. Good baseline. |

> **Note**: Linear models (`lrl1`, `lrl2`) are automatically disabled for large datasets (>100k rows) to ensure speed.

## 📊 Metrics (Training)
The following metrics are calculated during training to evaluate model performance:

- **Accuracy**: Proportion of correct predictions.
- **F1 Score**: Harmonic mean of Precision and Recall. Best for imbalanced classes.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability to find all positive instances.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve. Measures ability to distinguish classes.
- **Log Loss**: Measures the performance of a classification model where the prediction input is a probability value between 0 and 1.

## 📈 Reports

### 1. EDA Report (`.eda()`)
Generated using `automl.eda(df)`, this report helps you understand your data before training.
- **Target Analysis**: Class distribution (counts and percentages). Checks for class imbalance.
- **Feature Overview**: Basic stats, missing values, and cardinality check.
- **Correlations**: Correlation of numerical features with the target class (if encoded).

### 2. Training Report (`.report()`)
Generated after `automl.fit()`, this report details the model's performance.
- **Model Details**: Best algorithm selected, hyperparameters, and training time.
- **Confusion Matrix**: Visual representation of True Positives, False Positives, etc.
- **ROC Curve**: Evaluating the trade-off between True Positive Rate and False Positive Rate.
- **Feature Importance**: Bar chart showing which features contributed most to the prediction.
- **Class Performance**: Detailed table of Precision, Recall, and F1-Score for each class.
