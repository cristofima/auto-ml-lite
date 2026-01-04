"""
Time Series Forecasting Example using ez-automl-lite.
"""

import numpy as np
import pandas as pd

from ez_automl_lite import AutoTimeSeries


def main():  # noqa: PLR0915
    print("=" * 60)
    print("Time Series Forecasting Example")
    print("=" * 60)

    # Generate synthetic time series data with trend, seasonality, and noise
    # Note: sklearn doesn't have built-in time series datasets, so we create realistic synthetic data
    # This simulates daily sales data with seasonal patterns
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=365, freq="D")

    # Components
    trend = np.linspace(100, 200, 365)  # Upward trend
    seasonality = 20 * np.sin(np.linspace(0, 4 * np.pi, 365))  # Yearly seasonality
    weekly_pattern = 5 * np.sin(np.linspace(0, 52 * np.pi, 365))  # Weekly pattern
    noise = np.random.normal(0, 5, 365)  # Random noise

    values = trend + seasonality + weekly_pattern + noise

    df = pd.DataFrame({"date": dates, "sales": values})

    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sales range: {df['sales'].min():.2f} to {df['sales'].max():.2f}")

    # Initialize AutoTimeSeries
    automl = AutoTimeSeries(
        time_column="date",
        target_column="sales",
        frequency="D",  # Daily frequency
        forecast_horizon=30,  # Forecast next 30 days
        random_state=42,
        scaling="absmax",  # 'absmax' (default) or 'minmax'
        holidays_mode=None,  # Optional: 'additive' or 'multiplicative'
    )

    # Generate EDA report (optional - requires sweetviz)
    print("\n" + "=" * 60)
    print("Generating EDA Report...")
    print("=" * 60)
    try:
        automl.eda(df, "timeseries_eda_report.html")
        print("✓ EDA report generated")
    except Exception as e:
        print(f"Note: EDA report generation skipped (optional dependency): {e}")

    # Train the model
    print("\n" + "=" * 60)
    print("Training Time Series Model...")
    print("=" * 60)
    automl.fit(df, algorithm="auto", test_size=0.2)

    # Display metrics
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Algorithm: {automl.metrics['algorithm']}")
    print(f"MAE: {automl.metrics['mae']:.4f}")
    print(f"RMSE: {automl.metrics['rmse']:.4f}")
    print(f"sMAPE: {automl.metrics['smape']:.2f}%")
    if automl.metrics.get("mape") is not None:
        print(f"MAPE: {automl.metrics['mape']:.2f}%")

    # Check stationarity
    stationarity = automl.metrics.get("stationarity", {})
    if stationarity.get("is_stationary") is not None:
        is_stationary = stationarity["is_stationary"]
        stat_text = "Stationary" if is_stationary else "Non-Stationary"
        print(f"\nStationarity: {stat_text}")
        print(f"ADF Statistic: {stationarity['adf_statistic']:.4f}")
        print(f"P-Value: {stationarity['p_value']:.4f}")

    print(f"\nTraining Time: {automl.metrics['execution_time']:.2f} seconds")

    # Generate forecasts
    print("\n" + "=" * 60)
    print("Generating Future Forecasts...")
    print("=" * 60)
    forecast = automl.predict(horizon=30)
    print("Forecasted next 30 days:")
    print(f"Mean: {forecast.mean():.2f}")
    print(f"Min: {forecast.min():.2f}")
    print(f"Max: {forecast.max():.2f}")
    print(f"\nFirst 5 forecast values: {forecast[:5]}")

    # Display confidence intervals if available
    if automl.forecast_lower is not None and automl.forecast_upper is not None:
        print("\n" + "-" * 60)
        print("95% Confidence Intervals:")
        print("-" * 60)
        for i in range(5):
            print(
                f"Day {i+1}: {forecast[i]:.2f} (CI: {automl.forecast_lower[i]:.2f} - {automl.forecast_upper[i]:.2f})"
            )

    # Generate training report
    print("\n" + "=" * 60)
    print("Generating Training Report...")
    print("=" * 60)
    automl.report("timeseries_training_report.html")

    print("\n" + "=" * 60)
    print("Time Series Example Completed Successfully!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  - timeseries_eda_report.html")
    print("  - timeseries_training_report.html")


if __name__ == "__main__":
    main()
