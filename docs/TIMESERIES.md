# ⏰ Time Series Forecasting

Time Series Forecasting predicts future values based on historical temporal patterns, trends, and seasonality.

## ✅ Supported Algorithms
`ez-automl-lite` automatically selects and configures the best time series algorithm:

| Algorithm | ID | Description |
| :--- | :--- | :--- |
| **ARIMA** | `arima` | (Default) AutoRegressive Integrated Moving Average. Captures temporal dependencies and trends through automatic (p, d, q) parameter search. Best for univariate series with clear patterns. |
| **Prophet** | `prophet` | Facebook's forecasting tool designed for business time series. Handles missing data, outliers, and automatically detects multiple seasonality patterns (yearly, weekly, daily). |

> **Auto Selection**: If `algorithm='auto'` (default), ARIMA is selected if `statsmodels` is available, otherwise Prophet.

## 📊 Metrics (Training)
The following metrics are calculated to evaluate forecasting performance:

- **MAE**: Mean Absolute Error. Average magnitude of prediction errors.
- **RMSE**: Root Mean Squared Error. Penalizes larger errors more heavily.
- **sMAPE**: Symmetric Mean Absolute Percentage Error. Scale-independent percentage metric (0-100%).
- **MAPE**: Mean Absolute Percentage Error. Traditional percentage error (only if target has no zeros).
- **Stationarity**: ADF (Augmented Dickey-Fuller) test results indicating if series is stationary (p-value < 0.05 means stationary).

> **Lower values = Better performance** for all metrics.

## 📈 Reports

### 1. EDA Report (`.eda()`)
Generated using `automl.eda(df, 'output.html')`, this report helps you understand temporal patterns before training.
- **Time Series Plot**: Historical values over time showing trends and patterns.
- **Temporal Statistics**: Min, max, mean, variance across the timeline.
- **Seasonality Indicators**: Visual cues for periodic patterns.
- **Missing Values**: Temporal gaps in the data.

### 2. Training Report (`.report()`)
Generated after `automl.fit()`, this report details the model's forecasting performance.
- **Performance Metrics**: MAE, RMSE, sMAPE, MAPE displayed as cards.
- **Stationarity Analysis**: ADF test results with statistic and p-value.
- **Forecast vs Actual**: Visualization comparing predictions against actual test values.
- **Time Series Decomposition**: Breakdown into trend, seasonal, and residual components.
- **Dataset Information**: Date range, frequency, sample counts.
- **Future Forecasts**: Table of predicted values with dates and 95% confidence intervals (when available).
- **Confidence Intervals**: Uncertainty bounds for predictions (always available for Prophet, attempted for ARIMA).

---

## 🚀 Basic Usage

```python
from ez_automl_lite import AutoTimeSeries
import pandas as pd

# Load your time series data
df = pd.DataFrame({
    'date': pd.date_range('2022-01-01', periods=365, freq='D'),
    'sales': [100, 102, 98, ...]  # Your time series values
})

# Initialize
automl = AutoTimeSeries(
    time_column='date',
    target_column='sales',
    frequency='D',
    forecast_horizon=30,
    scaling='absmax',  # 'absmax' or 'minmax' (Prophet only)
    holidays_mode=None  # Optional: 'additive' or 'multiplicative' (Prophet only)
)

# Train
automl.fit(df, algorithm='auto', test_size=0.2)

# Forecast
forecast = automl.predict(30)

# Generate reports
automl.eda(df, 'timeseries_eda.html')
automl.report('timeseries_training.html')
```

## 📋 Parameters

### `AutoTimeSeries()`
- **time_column** (str): Name of the datetime column
- **target_column** (str): Name of the target column to forecast
- **frequency** (str, default='D'): Time series frequency
  - `'D'`: Daily | `'W'`: Weekly | `'M'`: Monthly | `'Q'`: Quarterly | `'H'`: Hourly
- **forecast_horizon** (int, default=30): Number of future steps to forecast
- **random_state** (int, default=42): Random seed for reproducibility
- **scaling** (str, default='absmax'): Scaling method for Prophet
  - `'absmax'`: Scale by maximum absolute value (default)
  - `'minmax'`: Scale between 0 and 1
- **holidays_mode** (str, optional): Separate mode for holidays in Prophet
  - `'additive'`: Additive effect of holidays
  - `'multiplicative'`: Multiplicative effect of holidays
  - If not specified, uses the same mode as `seasonality_mode`
- **job_id** (str, optional): Job identifier for tracking
> **Note**: `scaling` and `holidays_mode` parameters only affect Prophet models. They are ignored when using ARIMA.
### `.fit()`
- **df** (pd.DataFrame): DataFrame with time series data
- **algorithm** (str, default='auto'): Algorithm to use
  - `'auto'`: Automatically selects ARIMA or Prophet
  - `'arima'`: Uses ARIMA/SARIMA
  - `'prophet'`: Uses Facebook Prophet
- **test_size** (float, default=0.2): Fraction of data for testing (temporal split)

### `.predict()`
- **horizon** (int, optional): Number of steps to forecast (defaults to `forecast_horizon`)
- **Returns**: Array of predicted values

## 🔍 Algorithm Details

### ARIMA Configuration
The implementation automatically searches for optimal parameters:
- Tests common configurations: `(1,1,1)`, `(2,1,2)`, `(1,0,1)`, `(0,1,1)`, `(2,0,2)`
- Selects model with lowest **AIC** (Akaike Information Criterion)
- Handles differencing automatically based on stationarity
- Attempts to provide confidence intervals when generating forecasts

### Prophet Configuration
Prophet automatically detects and models:
- **Trend**: Linear or logistic growth
- **Seasonality**: Yearly, weekly, and daily patterns
- **Holidays**: Can be added via custom configuration (with separate mode support)
- **Changepoints**: Automatic detection of trend changes
- **Scaling**: Supports both absmax and minmax scaling methods
- **Confidence Intervals**: Provides 95% prediction intervals for uncertainty quantification
- **Error Handling**: Automatic fallback to default settings if custom configuration fails

## 📊 Time Series Components

The decomposition breaks the series into:

1. **Trend**: Long-term progression (increasing/decreasing pattern)
2. **Seasonal**: Repeating patterns at fixed intervals (daily, weekly, yearly)
3. **Residual**: Random noise after removing trend and seasonality

Understanding these components helps identify which algorithm works best:
- **Strong trend, weak seasonality** → ARIMA
- **Complex seasonality, missing data** → Prophet

## 💡 Example: Sales Forecasting

```python
import numpy as np
import pandas as pd
from ez_automl_lite import AutoTimeSeries

# Create synthetic sales data with trend and seasonality
dates = pd.date_range('2022-01-01', periods=365, freq='D')
trend = np.linspace(100, 200, 365)
seasonality = 20 * np.sin(np.linspace(0, 4*np.pi, 365))
noise = np.random.normal(0, 5, 365)
sales = trend + seasonality + noise

df = pd.DataFrame({'date': dates, 'sales': sales})

# Train model
automl = AutoTimeSeries(
    time_column='date',
    target_column='sales',
    frequency='D',
    forecast_horizon=30
)

automl.fit(df, algorithm='arima')

# Check metrics
print(f"MAE: {automl.metrics['mae']:.2f}")
print(f"RMSE: {automl.metrics['rmse']:.2f}")
print(f"Is Stationary: {automl.metrics['stationarity']['is_stationary']}")

# Forecast next 30 days
forecast = automl.predict(30)
print(f"Next 30 days forecast: {forecast}")

# Generate report
automl.report('sales_forecast.html')
```

## 📦 Installation

Install time series dependencies:

```bash
pip install "ez-automl-lite[timeseries]"
```

This installs:
- `statsmodels>=0.14.6` (for ARIMA)
- `prophet>=1.2.1` (for Prophet)

## ✅ Best Practices

### Data Preparation
1. **Datetime Format**: Ensure time column is properly formatted (`pd.to_datetime`)
2. **No Missing Dates**: Fill gaps or handle missing values before training
3. **Consistent Frequency**: Ensure regular intervals (daily, weekly, etc.)
4. **Check for Outliers**: Extreme values can affect forecasting

### Frequency Selection
- Match frequency to your data collection interval
- Daily data: `'D'` | Weekly: `'W'` | Monthly: `'M'`
- Consistent intervals improve accuracy

### Test Size
- Use 20-30% for test set with sufficient data
- For shorter series (< 100 points), use smaller test fraction
- Ensure test set covers at least one seasonal cycle

### Seasonality Requirements
- **Daily patterns**: Need at least 2 weeks of data
- **Weekly patterns**: Need at least 8 weeks
- **Yearly patterns**: Need at least 2 years
- Prophet handles complex seasonality better than ARIMA

### Forecast Horizon
- Shorter horizons (7-30 steps) are more accurate
- Uncertainty increases exponentially with longer forecasts
- For long-term forecasts, consider external factors and retraining

## ⚠️ Limitations

- Only supports **univariate forecasting** (single target column)
- Does not handle multivariate time series (multiple related targets)
- Requires **sequential, time-ordered** data with consistent frequency
- Minimum data requirements:
  - At least 50 data points for reliable ARIMA estimation
  - At least 2 seasonal cycles for meaningful decomposition analysis
  - Prophet requires at least 2 non-null observations
- ARIMA parameter search tests only a limited set of common configurations
- External regressors and custom holidays are not currently supported
- Confidence intervals may not be available for all ARIMA models

## 🔧 Troubleshooting

### Issue: "No time series library available"
**Solution**: Install dependencies:
```bash
pip install "ez-automl-lite[timeseries]"
```

### Issue: Poor forecasting accuracy
**Checklist**:
- [ ] Check for missing values or gaps
- [ ] Verify frequency matches data
- [ ] Test both ARIMA and Prophet
- [ ] Increase training data size
- [ ] Check for outliers affecting the model

### Issue: "Series is non-stationary"
**Impact**: ARIMA will automatically difference the series (d parameter)
**Action**: No action needed - this is expected and handled automatically

### Issue: Prophet warnings about seasonality
**Solution**: Ensure you have enough data for the seasonal period:
- Yearly: Need 2+ years
- Weekly: Need 2+ months
- Daily: Need 2+ weeks
