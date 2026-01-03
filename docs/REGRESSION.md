# 📈 Regression

Regression is a supervised learning task where the goal is to predict a continuous numerical value.

## ✅ Supported Algorithms
`ez-automl-lite` leverages FLAML to find the best regressor. Supported algorithms include:

| Algorithm | ID | Description |
| :--- | :--- | :--- |
| **LightGBM** | `lgbm` | High-performance gradient boosting decision tree. |
| **XGBoost** | `xgboost` | Scalable and accurate implementation of gradient boosting. |
| **Random Forest** | `rf` | Ensemble method using multiple decision trees. |
| **Extra Trees** | `extra_tree` | Similar to Random Forest but with more randomness in splits. |
| **Histogram GB** | `histgb` | Scikit-learn's histogram-based gradient boosting (inspired by LightGBM). |

## 📊 Metrics (Training)
We calculate standard regression metrics to evaluate errors:

- **R² Score**: Coefficient of determination. Represents the proportion of variance explained by the model. (Closer to 1 is better).
- **RMSE**: Root Mean Squared Error. Standard error metric penalizing large errors.
- **MAE**: Mean Absolute Error. Average magnitude of errors.
- **MAPE**: Mean Absolute Percentage Error. Error expressed as a percentage (easier to interpret).

## 📈 Reports

### 1. EDA Report (`.eda()`)
Analyze the target variable and features before training.
- **Target Distribution**: Histogram and statistics (Mean, Median, Skewness, Kurtosis) of the target variable.
- **Correlation**: Top numerical features correlated with the target.
- **Outliers**: Quick check for extreme values in the target.

### 2. Training Report (`.report()`)
Evaluate the trained regression model.
- **Model Summary**: Best estimator, R² score, and error metrics.
- **Predicted vs Actual**: Scatter plot to visualize how well predictions match actual values. Ideally, points lie on the diagonal line.
- **Residual Plot**: Analysis of errors (residuals). Helps detect bias or heteroscedasticity.
- **Feature Importance**: Ranking of features based on their impact on the target variable.
